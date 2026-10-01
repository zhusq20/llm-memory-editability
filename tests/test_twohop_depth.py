import copy

import numpy as np
import torch

from llm_memory_editability.twohop_depth import (
    BOS,
    CONFIG,
    ENTITY,
    EOS,
    audit_world,
    build_model,
    diagnostics,
    evaluate,
    make_stream,
    optimizer_for,
    patched_logits,
    read,
    states_and_logits,
    train_step,
    training_tensors,
    world_arrays,
)


def small():
    cfg = read(CONFIG)
    cfg.update(entities=32, steps=16, batch_per_kind=8, diagnostic_cases=8)
    return cfg, world_arrays(142890, cfg)


def test_frozen_worlds_have_no_composition_leakage_and_full_atomic_coverage():
    cfg = read(CONFIG)
    for seed in cfg["worlds"]:
        w = world_arrays(seed, cfg)
        audit = audit_world(w, cfg)
        assert audit["train_compositions"] == audit["test_compositions"] == 1024
        train = set(map(tuple, w["composite_x"][w["train_mask"]]))
        test = set(map(tuple, w["composite_x"][~w["train_mask"]]))
        assert train.isdisjoint(test)


def test_repeated_generation_and_training_streams_are_reproducible():
    cfg, w = small()
    other = world_arrays(142890, cfg)
    for key in w:
        np.testing.assert_array_equal(w[key], other[key])
    cfg["steps"] = 128
    stream = make_stream(w, 100, cfg)
    for key, arr in stream.items():
        assert len(arr) == cfg["steps"] * cfg["batch_per_kind"]
        counts = np.bincount(arr)
        assert counts.min() == counts.max()
        np.testing.assert_array_equal(arr, make_stream(w, 100, cfg)[key])


def test_answers_and_eos_are_only_supervised_after_the_query():
    cfg, w = small()
    t = training_tensors(w, "cpu")
    for kind, length in (("atomic", 3), ("composite", 4)):
        x, pos, y = t[kind]
        assert torch.all(pos[:, 0] == length - 1)
        assert torch.all(pos[:, 1] == length)
        assert torch.all(x[:, length] == y[:, 0])
        assert torch.all(y[:, 1] == EOS)
        assert torch.all(x[:, length + 1] == EOS)


def test_parameter_controls_match_the_six_layer_model():
    cfg = read(CONFIG)
    counts = {
        a["name"]: sum(p.numel() for p in build_model(a, cfg).parameters())
        for a in cfg["architectures"]
    }
    for name in ("d1w312", "d2w220"):
        assert abs(counts[name] / counts["d6w128"] - 1) < 0.01


def test_causal_prefix_and_intervention_identity_and_terminal_control():
    torch.set_num_threads(1)
    cfg, w = small()
    model = build_model({"layers": 3, "width": 16, "heads": 2}, cfg).eval()
    q = torch.tensor(w["composite_x"][:8])
    states, logits = states_and_logits(model, q, "cpu")
    torch.testing.assert_close(model(q), logits, atol=0, rtol=0)
    short, _ = states_and_logits(model, q[:, :3], "cpu")
    for layer in range(4):
        torch.testing.assert_close(states[layer][:, :3], short[layer], atol=1e-6, rtol=1e-5)
        z = patched_logits(model, q, layer, [2], states[layer][:, [2]], "cpu")
        torch.testing.assert_close(z, logits[:, -1], atol=0, rtol=0)
    changed = states[-1][:, [2]] + 100
    z = patched_logits(model, q, 3, [2], changed, "cpu")
    torch.testing.assert_close(z, logits[:, -1], atol=0, rtol=0)


def test_embedding_prefix_swap_equals_running_the_counterfactual_query():
    cfg, w = small()
    model = build_model({"layers": 2, "width": 16, "heads": 2}, cfg).eval()
    cases = w["cases"]
    q = torch.tensor(w["composite_x"][cases[:, 0]])
    alternate = q.clone()
    alternate[:, 1] = torch.tensor(ENTITY + cases[:, 6])
    donor, _ = states_and_logits(model, alternate[:, :3], "cpu")
    actual = patched_logits(model, q, 0, [1, 2], donor[0][:, [1, 2]], "cpu")
    torch.testing.assert_close(actual, model(alternate)[:, -1], atol=0, rtol=0)
    diag = diagnostics(model, w, cfg, "cpu")
    np.testing.assert_array_equal(diag["bridge_0_pred"], diag["clean_pred"])


def test_cpu_resume_is_bitwise_identical():
    cfg, w = small()
    torch.manual_seed(42)
    a = build_model({"layers": 2, "width": 16, "heads": 2}, cfg)
    oa = optimizer_for(a, cfg, "cpu")
    tensors = training_tensors(w, "cpu")
    streams = make_stream(w, 10, cfg)
    for step in range(8):
        ix = {k: torch.tensor(v[step * 8 : (step + 1) * 8]) for k, v in streams.items()}
        train_step(a, oa, tensors, ix, step, cfg, "cpu")
    b = build_model({"layers": 2, "width": 16, "heads": 2}, cfg)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    ob = optimizer_for(b, cfg, "cpu")
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    for step in range(8, 16):
        ix = {k: torch.tensor(v[step * 8 : (step + 1) * 8]) for k, v in streams.items()}
        train_step(a, oa, tensors, ix, step, cfg, "cpu")
        train_step(b, ob, tensors, ix, step, cfg, "cpu")
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_two_call_baseline_uses_its_generated_entity():
    cfg, w = small()

    class WrongBridge(torch.nn.Module):
        def forward(self, x):
            z = torch.zeros(len(x), x.shape[1], ENTITY + cfg["entities"] + cfg["relations"])
            # Always emits entity 0 for an answer, EOS after an emitted entity.
            prediction = torch.where(x[:, -1] < ENTITY + cfg["entities"], EOS, ENTITY)
            z[torch.arange(len(x)), -1, prediction] = 10
            return z

    _, arrays = evaluate(WrongBridge(), w, cfg, "cpu")
    assert np.all(arrays["two_call_bridge"] == ENTITY)
    assert np.all(arrays["two_call_pred"] == ENTITY)
    assert np.any(w["bridge"][~w["train_mask"]] != ENTITY)
    assert np.all(w["atomic_x"][:, 0] == BOS)
