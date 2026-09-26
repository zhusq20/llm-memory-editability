"""Controls that would change the interpretation of protocol experiments."""

import hashlib
import json

import numpy as np
import pytest

from llm_memory_editability.bios_data import (
    ANS,
    EOS,
    N_BASE,
    N_QUERIES,
    REVISION,
    SOURCE_FOLDER,
    audit_world,
    curriculum,
    make_world,
    paired_edit,
)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("source")
    folder = root / SOURCE_FOLDER / "fields"
    folder.mkdir(parents=True)
    files = []
    counts = {
        "first_name": 400,
        "middle_name": 400,
        "last_name": 1000,
        "city": 200,
        "company": 263,
        "university": 300,
        "field": 100,
    }
    for name, count in counts.items():
        path = folder / (name + ".txt")
        path.write_text("\n".join(f"{name}_{i}" for i in range(count)))
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(json.dumps({"revision": REVISION, "files": files}))
    return make_world(7, root)


def test_truth_and_symbolic_queries_do_not_leak_answers(world):
    assert audit_world(world)["base_facts"] == 12352
    assert world.prompts.shape == (14400, 5)
    rows = np.arange(N_QUERIES)
    assert np.all(world.prompts[rows, world.lengths - 1] == ANS)
    assert np.all(world.answers >= 4)
    # Exact default lookup shortcut exposes how aggregate accuracy hides exceptions.
    actual = world.answers[2112:4160]
    predicted = world.city_tokens[world.defaults[world.employers]]
    assert np.mean(predicted == actual) == 0.9375
    assert not np.any(predicted[world.exceptions] == actual[world.exceptions])


def test_curricula_have_exact_exposure_and_identical_consolidation(world):
    sa, ass = (curriculum(world, order, 7280) for order in ("SA", "AS"))
    np.testing.assert_array_equal(sa[:2640], ass[2640:5280])
    np.testing.assert_array_equal(sa[2640:5280], ass[:2640])
    np.testing.assert_array_equal(sa[5280:], ass[5280:])
    before = np.bincount(sa[:5280].ravel(), minlength=N_QUERIES)
    np.testing.assert_array_equal(before[:2112], np.full(2112, 160))
    np.testing.assert_array_equal(before[2112:N_BASE], np.full(10240, 33))
    assert before[N_BASE:].sum() == 0
    assert np.mean(sa[5280:] >= N_BASE) == 0.2
    np.testing.assert_array_equal(
        np.bincount(sa.ravel(), minlength=N_QUERIES), np.bincount(ass.ravel(), minlength=N_QUERIES)
    )


@pytest.mark.parametrize("k", [1, 4, 16])
def test_updates_recompute_truth_and_partition_unseen_retention(world, k):
    pair = paired_edit(world, support=1, k=k)
    assert len(pair["E"]) == 93 * k and len(pair["D"]) == 96 * k
    for target in (pair["coherent"], pair["exception"]):
        changed = np.flatnonzero(target != world.answers)
        np.testing.assert_array_equal(changed, np.sort(np.concatenate([pair["E"], pair["D"]])))
        np.testing.assert_array_equal(target[N_BASE:], target[2048 + world.employers])
        np.testing.assert_array_equal(
            target[2112:4160][world.exceptions], world.answers[2112:4160][world.exceptions]
        )
        assert np.all(target[pair["strata"] >= 0] == world.answers[pair["strata"] >= 0])
    np.testing.assert_array_equal(
        np.sort(pair["coherent"][pair["E"]]), np.sort(pair["exception"][pair["E"]])
    )
    assert not set(pair["E"]) & set(pair["replay"])
    assert not set(pair["D"]) & set(pair["replay"])
    assert not set(pair["heldout"]) & set(pair["replay"])
    assert set(pair["strata"][pair["heldout"]]) == {0, 1, 2, 3}
    assert np.all(pair["replay"] < N_BASE)
    assert np.sum(pair["coherent"] != pair["exception"]) == 45 * k


def test_decoder_causality_answer_alignment_and_parameter_scope(world):
    torch = pytest.importorskip("torch")
    from llm_memory_editability.bios_model import CausalLM, ModelConfig, select_parameters
    from llm_memory_editability.bios_train import tensor_data

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = CausalLM(ModelConfig(world.vocab_size, width=16, layers=3, heads=2))
    data = tensor_data(world, torch.device("cpu"))
    ids = torch.tensor([0, 2112, N_BASE])
    tokens = data["tokens"][ids]
    positions = data["positions"][ids]
    assert torch.equal(tokens[torch.arange(3), positions[:, 1]], data["labels"][ids, 0])
    assert torch.all(data["labels"][ids, 1] == EOS)
    logits = model(tokens)
    perturbed = tokens.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % world.vocab_size
    torch.testing.assert_close(model(perturbed)[:, :-1], logits[:, :-1])
    torch.testing.assert_close(
        model(tokens, positions), logits[torch.arange(3)[:, None], positions]
    )
    selected = select_parameters(model, "down", start=0)
    assert len(selected) == 1 and selected[0] is model.blocks[1].mlp.down.weight
    assert [n for n, p in model.named_parameters() if p.requires_grad] == [
        "blocks.1.mlp.down.weight"
    ]
    assert len(select_parameters(model, "all")) == len(list(model.parameters()))


def test_generation_uses_own_prediction_and_requires_eos(world):
    torch = pytest.importorskip("torch")
    from llm_memory_editability.bios_train import evaluate, tensor_data

    class WrongThenEos(torch.nn.Module):
        def forward(self, tokens, positions):
            logits = torch.full((len(tokens), 1, world.vocab_size), -100.0)
            if tokens.shape[1] == 5:
                logits[:, :, 4] = 100
            else:
                # Correct answers must never be supplied to the generation continuation.
                assert torch.all(tokens[torch.arange(len(tokens)), positions[:, 0]] == 4)
                logits[:, :, EOS] = 100
            return logits

    metrics, arrays = evaluate(WrongThenEos(), tensor_data(world, torch.device("cpu")), world)
    np.testing.assert_array_equal(arrays["correct"], world.answers == 4)
    assert metrics["nontermination_rate"] == 0


def test_branch_zero_function_and_identical_parameter_budgets(world):
    torch = pytest.importorskip("torch")
    from llm_memory_editability.bios_branch import attach_branch, remove_branch
    from llm_memory_editability.bios_model import CausalLM, ModelConfig
    from llm_memory_editability.bios_train import tensor_data

    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(world.vocab_size, width=16, layers=3, heads=2))
    data = tensor_data(world, torch.device("cpu"))
    ids = torch.tensor([0, N_BASE])  # Different prompt lengths must both be handled.
    tokens, positions = data["tokens"][ids], data["positions"][ids]
    before = model(tokens, positions).detach()
    q = torch.linalg.qr(torch.randn(16, 4)).Q
    branch, handles = attach_branch(model, (q, torch.zeros(4), torch.ones(4)), layer=1, width=4)
    assert sum(p.numel() for p in branch.parameters()) == 4 * (4 + 16)
    after = model(tokens, positions)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(branch.positions, torch.tensor([3, 4]))
    after.square().sum().backward()
    assert branch.b.grad.abs().sum() > 0
    remove_branch(model, handles)
    torch.testing.assert_close(before, model(tokens, positions), rtol=0, atol=0)


def test_retention_counts_old_correct_destruction_and_probe_controls(world):
    pytest.importorskip("torch")
    from llm_memory_editability.bios_edit import score_edit
    from llm_memory_editability.bios_measure import probe_plan

    pair = paired_edit(world)
    old = np.ones(N_QUERIES, dtype=bool)
    correct = np.ones(N_QUERIES, dtype=bool)
    held = pair["heldout"][pair["strata"][pair["heldout"]] == 0]
    correct[held[0]] = False
    old[held[1]] = False  # Repair must not cancel the other fact's destruction.
    result = score_edit(
        world, pair, pair["coherent"], {"correct": correct, "value_nll": np.zeros(N_QUERIES)}, old
    )
    assert result["U_heldout_destruction"]["0"]["broken"] == 1
    assert result["U_heldout_destruction"]["0"]["known"] == len(held) - 1
    assert result["joint_pass"] is False
    for plan in probe_plan(world):
        assert np.all(world.relation[plan["ids"][plan["group"] == 2]] == 4)
        assert world.defaults[plan["same_answer"]] == world.defaults[plan["recipient"]]
        assert world.defaults[plan["donor"]] != world.defaults[plan["recipient"]]


def test_zero_causal_shift_preserves_generated_answers_and_removes_hooks(world, monkeypatch):
    torch = pytest.importorskip("torch")
    from llm_memory_editability import bios_measure
    from llm_memory_editability.bios_model import CausalLM, ModelConfig
    from llm_memory_editability.bios_train import tensor_data

    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(world.vocab_size, width=8, layers=1, heads=1))
    model.eval()
    data = tensor_data(world, torch.device("cpu"))
    plans = bios_measure.probe_plan(world)[:1]
    monkeypatch.setattr(bios_measure, "probe_plan", lambda _world: plans)
    means = [torch.zeros(64, 32)]
    before = model(data["prompts"][:2], data["lengths"][:2, None] - 1).detach()
    probes = bios_measure.causal_probes(model, world, data, means, behavior=True)
    assert len(probes) == 5
    for probe in probes:
        assert probe["shift_norm"] == 0
        assert probe["selective_shared_score"] == 0
        for group in probe["generation"].values():
            assert group["baseline_old_accuracy"] == group["changed_old_accuracy"]
            assert group["old_broken"] == 0
            assert group["donor_generated_delta"] == 0
        assert sum(g["n"] for g in probe["generation"].values()) == 96
    assert not model.blocks[0].mlp.activation._forward_hooks
    torch.testing.assert_close(before, model(data["prompts"][:2], data["lengths"][:2, None] - 1))


def test_edit_geometry_uses_old_features_and_target_dependent_gradients(world):
    torch = pytest.importorskip("torch")
    from llm_memory_editability.bios_model import CausalLM, ModelConfig
    from llm_memory_editability.bios_train import tensor_data
    from scripts.measure_bios_edit_geometry import measure

    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(world.vocab_size, width=8, layers=2, heads=1))
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    data = tensor_data(world, torch.device("cpu"))
    pair = paired_edit(world)
    pair["replay"] = pair["replay"][:16]
    rows, coherent = measure(model, world, data, pair, "coherent", (0, 1))
    _, exception = measure(model, world, data, pair, "exception", (0, 1))
    for layer in (0, 1):
        # Teacher-forced answers may affect gradients, never preceding prompt features.
        np.testing.assert_array_equal(
            coherent[f"E_z_layer_{layer}"], exception[f"E_z_layer_{layer}"]
        )
        grad = exception[f"E_h_loss_gradient_layer_{layer}"]
        assert np.isfinite(grad).all() and np.linalg.norm(grad) > 0
        assert not np.array_equal(grad, coherent[f"E_h_loss_gradient_layer_{layer}"])
        assert not model.blocks[layer].mlp._forward_hooks
    assert all(row["E_n"] == 93 and row["R_n"] == 16 for row in rows)
    assert not set(exception["R"]) & set(pair["D"])
    assert not set(exception["R"]) & set(pair["heldout"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
