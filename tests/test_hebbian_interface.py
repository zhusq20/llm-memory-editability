"""Scientific contracts for frozen-memory SSFR replacement, on tiny CPU models."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import hebbian_interface as interface

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / interface.DEFAULT_SPEC["source_path"] / "src/hebbian").is_dir(),
    reason="The pinned official hebbian-mlps source checkout is required",
)


@pytest.fixture(scope="module")
def spec():
    return {
        **interface.DEFAULT_SPEC,
        "num_facts": 8,
        "d_model": 8,
        "hidden_dim": 16,
        "junk_len": 2,
        "junk_vocab_size": 3,
        "batch_size": 8,
        "mlp_epochs": 3,
        "reader_steps": 4,
        "eval_every": 2,
        "eval_repeats": 2,
    }


def components(spec, seed=53):
    config = interface.make_config(spec, seed, "cpu")
    from hebbian.transformer.fact_store import build_factset, build_token_embeddings
    from hebbian.transformer.model import GPT
    from hebbian.transformer.utils import (
        copy_embeddings_to_gpt,
        create_gpt_config,
        insert_mlp_into_gpt,
    )

    factset = build_factset(config, seed=seed)
    embeddings = build_token_embeddings(
        factset, spec["junk_vocab_size"], embedding_init="spherical", seed=seed
    )
    torch.manual_seed(seed)
    model = GPT(create_gpt_config(config.train_config, config.dataset_config)).eval()
    copy_embeddings_to_gpt(model, embeddings)
    memory = interface._new_memory(spec, "cpu")
    insert_mlp_into_gpt(model, memory, embeddings)
    return model, factset, embeddings


@pytest.fixture(scope="module")
def tiny_run(spec, tmp_path_factory):
    output = tmp_path_factory.mktemp("hebbian-interface")
    summary = interface.run_world(spec, 31, output, "cpu")
    return summary, output


def test_architecture_and_only_qk_trainable(spec):
    model, _, _ = components(spec)
    block = model.transformer.h[0]
    assert len(model.transformer.h) == 1
    assert model.transformer.wpe is None
    assert not model.config.use_rope
    assert not block.attn_residual and not block.mlp_residual
    for projection in (block.attn.c_v, block.attn.c_proj):
        torch.testing.assert_close(projection.weight, torch.eye(spec["d_model"]))
        assert not projection.weight.requires_grad
    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert trainable == {
        "transformer.h.0.attn.c_q.weight",
        "transformer.h.0.attn.c_k.weight",
    }
    x = torch.randn(5, spec["d_model"])
    for normalization in (block.ln_2, model.transformer.ln_f):
        torch.testing.assert_close(normalization(x).norm(dim=-1), torch.ones(5))


def test_query_inputs_have_no_answer_and_do_not_depend_on_mapping(spec):
    interface.make_config(spec, 53, "cpu")
    from hebbian.data.synthetics.factsets import BijectiveMapping

    mapping_a = BijectiveMapping.from_permutation(list(range(spec["num_facts"])))
    mapping_b = BijectiveMapping.from_permutation(list(reversed(mapping_a.outputs)))
    torch.manual_seed(713)
    rng = torch.random.get_rng_state().clone()
    inputs, keys = interface.fixed_inputs(spec, mapping_a, 1234)
    assert torch.equal(rng, torch.random.get_rng_state())
    other, other_keys = interface.fixed_inputs(spec, mapping_b, 1234)
    assert torch.equal(inputs, other) and torch.equal(keys, other_keys)
    assert inputs.shape == (spec["num_facts"] * spec["eval_repeats"], 2 * spec["junk_len"] + 2)
    assert torch.equal(inputs[:, spec["junk_len"]], keys)
    query_id = spec["num_facts"] + spec["junk_vocab_size"]
    assert inputs[:, -1].eq(query_id).all()
    assert torch.equal(keys.bincount(), torch.full((spec["num_facts"],), spec["eval_repeats"]))
    context = torch.cat((inputs[:, : spec["junk_len"]], inputs[:, spec["junk_len"] + 1 : -1]), 1)
    assert context.ge(spec["num_facts"]).all() and context.lt(query_id).all()

    batch, targets = next(iter(interface.make_batches(spec, mapping_b, num_batches=1)))
    original_keys = batch[:, spec["junk_len"]]
    expected = torch.tensor(mapping_b.outputs)[original_keys]
    assert torch.equal(targets[:, -2], expected)
    assert torch.equal(batch[:, -1], expected)
    assert targets[:, :-2].eq(-100).all() and targets[:, -1].eq(-100).all()
    assert batch[:, :-1][:, -1].eq(query_id).all()


def test_future_answer_cannot_change_query_logits(spec):
    model, factset, _ = components(spec)
    inputs, _ = interface.fixed_inputs(spec, factset.mapping, 180)
    full = torch.cat((inputs, torch.zeros(len(inputs), 1, dtype=torch.long)), 1)
    altered = full.clone()
    altered[:, -1] = spec["num_facts"] - 1
    targets = torch.zeros_like(full)
    with torch.no_grad():
        original_logits, _ = model(full, targets)
        altered_logits, _ = model(altered, targets)
        removed_logits, _ = model(inputs)
    torch.testing.assert_close(original_logits[:, :-1], altered_logits[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(original_logits[:, -2], removed_logits[:, 0], rtol=1e-5, atol=1e-6)


def test_standalone_distinguishes_fact_candidates_from_full_head(spec, tmp_path):
    interface.make_config(spec, 1, "cpu")
    from hebbian.data.synthetics.factsets import BijectiveMapping, Factset

    values = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    factset = Factset(values, values, BijectiveMapping.from_permutation([0, 1]), 2, 2)
    embeddings = torch.nn.Embedding.from_pretrained(torch.cat((values, torch.tensor([[0.0, 1.0]]))))
    memory = torch.nn.Linear(2, 2)
    with torch.no_grad():
        memory.weight.copy_(torch.tensor([[0.1, 0.0], [0.0, 0.0]]))
        memory.bias.copy_(torch.tensor([0.0, 2.0]))
    report = interface.standalone(
        memory, factset, embeddings, torch.ones(2, dtype=torch.bool), tmp_path / "scores.npz"
    )
    assert report["fact_accuracy"] == 1.0
    assert report["full_accuracy"] == 0.0
    saved = np.load(tmp_path / "scores.npz")
    assert (saved["pred"] == 2).all()


def test_clean_key_readout_matches_actual_normalization_and_head(spec, tmp_path):
    model, factset, embeddings = components(spec)
    inputs, keys = interface.fixed_inputs(spec, factset.mapping, 612)
    targets = torch.tensor(factset.mapping.outputs)[keys]
    changed = torch.ones(spec["num_facts"], dtype=torch.bool)
    interface.standalone(
        model.transformer.h[0].mlp, factset, embeddings, changed, tmp_path / "standalone.npz"
    )
    stored_keys = factset.input_embeddings[keys]

    def exact_keys(module, args):
        replaced = args[0].clone()
        replaced[:, -1] = stored_keys
        return (replaced,)

    hook = model.transformer.h[0].mlp.register_forward_pre_hook(exact_keys)
    try:
        with torch.no_grad():
            logits, _ = model(inputs)
    finally:
        hook.remove()
    expected = torch.from_numpy(np.load(tmp_path / "standalone.npz")["scores"])[keys]
    torch.testing.assert_close(logits[:, 0], expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(logits[:, 0].argmax(-1).eq(targets), expected.argmax(-1).eq(targets))


def test_evaluation_preserves_rng_weights_mode_and_hooks(spec):
    model, factset, embeddings = components(spec)
    inputs, keys = interface.fixed_inputs(spec, factset.mapping, 161)
    model.train()
    state_hash = interface.tensor_hash(model.state_dict())
    torch.manual_seed(818)
    rng = torch.random.get_rng_state().clone()
    interface.evaluate_reader(
        model,
        inputs,
        keys,
        torch.tensor(factset.mapping.outputs)[keys],
        torch.ones(len(keys), dtype=torch.bool),
        embeddings,
        spec["batch_size"],
    )
    assert model.training
    assert not model.transformer.h[0].mlp._forward_pre_hooks
    assert state_hash == interface.tensor_hash(model.state_dict())
    assert torch.equal(rng, torch.random.get_rng_state())


def test_complete_cross_reload_scores_and_restore(spec, tiny_run):
    summary, output = tiny_run
    from hebbian.transformer.utils import insert_mlp_into_gpt

    world = np.load(output / "world.npz")
    inputs = torch.from_numpy(world["inputs"])
    keys = torch.from_numpy(world["key"])
    map_a, map_b = torch.from_numpy(world["mapping_A"]), torch.from_numpy(world["mapping_B"])
    changed = (map_a != map_b)[keys]
    assert not summary["audits"]["B_used_for_reader_optimization_or_selection"]
    assert all(
        value
        for key, value in summary["audits"].items()
        if key != "B_used_for_reader_optimization_or_selection"
    )
    for objective in ("ce", "mse"):
        model, _ = interface.load_checkpoint(output / f"reader-{objective}.pt")
        initial_logits, _ = model(inputs)
        state_before = interface.tensor_hash(model.state_dict())
        original_memory = model.transformer.h[0].mlp
        non_mlp_before = interface.tensor_hash(interface._state(model, non_mlp=True))
        records = summary["readers"][objective]
        assert set(records["endpoints"]) == {"A", "wrong_A_on_B", "B_ce", "B_mse"}
        assert records["frozen_hash_before"] == records["frozen_hash_after"]
        a_saved = np.load(output / f"reader-{objective}-A.npz")
        wrong = np.load(output / f"reader-{objective}-wrong_A_on_B.npz")
        np.testing.assert_array_equal(a_saved["scores"], wrong["scores"])
        np.testing.assert_array_equal(wrong["target"], map_b[keys])
        for replacement in ("ce", "mse"):
            memory, _ = interface.load_checkpoint(output / f"memory-{replacement}-B.pt")
            assert not any(parameter.requires_grad for parameter in memory.parameters())
            insert_mlp_into_gpt(model.eval(), memory, model.transformer.wte)
            with torch.no_grad():
                logits, _ = model(inputs)
            saved = np.load(output / f"reader-{objective}-B_{replacement}.npz")
            torch.testing.assert_close(
                logits[:, 0], torch.from_numpy(saved["scores"]), rtol=1e-5, atol=1e-6
            )
            assert records["endpoints"][f"B_{replacement}"]["n_changed"] == int(changed.sum())
            assert non_mlp_before == interface.tensor_hash(interface._state(model, non_mlp=True))
        insert_mlp_into_gpt(model.eval(), original_memory, model.transformer.wte)
        restored_logits, _ = model(inputs)
        assert state_before == interface.tensor_hash(model.state_dict())
        torch.testing.assert_close(initial_logits, restored_logits, rtol=0, atol=0)


def test_B_mapping_does_not_affect_A_reader_training(spec, tiny_run, tmp_path, monkeypatch):
    original, _ = tiny_run
    from hebbian.data.synthetics.factsets import BijectiveMapping
    from hebbian.transformer import train

    original_factory = train._make_eval_factset

    def alternate_B(factset, seed):
        replacement = copy.copy(original_factory(factset, seed))
        replacement.mapping = BijectiveMapping.from_permutation(
            [(value + 1) % spec["num_facts"] for value in replacement.mapping.outputs]
        )
        return replacement

    monkeypatch.setattr(train, "_make_eval_factset", alternate_B)
    alternate = interface.run_world(spec, 31, tmp_path, "cpu")
    assert original["world"]["mapping_B"] != alternate["world"]["mapping_B"]
    for objective in ("ce", "mse"):
        for name in ("initial_non_mlp_hash", "final_non_mlp_hash", "training_stream_hash"):
            assert original["readers"][objective][name] == alternate["readers"][objective][name]
        assert (
            original["memories"][objective]["A"]["final_hash"]
            == alternate["memories"][objective]["A"]["final_hash"]
        )
        assert (
            original["memories"][objective]["B"]["final_hash"]
            != alternate["memories"][objective]["B"]["final_hash"]
        )
