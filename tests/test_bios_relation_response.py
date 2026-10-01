"""Scientific contracts for four-cell edits and source-isolated derivatives."""

import torch

from llm_memory_editability.bios_direction_cross import cross_tasks
from llm_memory_editability.bios_relation_response import (
    CELLS,
    check_reused_task,
    factorial_effects,
    factorial_tasks,
    isolated_directions,
)


def test_factorial_truth_prefixes_and_reuse():
    tasks = factorial_tasks(0, 0, "cpu")
    old = cross_tasks(0, 0, "cpu")
    assert len(tasks) == 24
    for p in range(6):
        four = tasks[4 * p : 4 * p + 4]
        for cell, task in zip(CELLS, four, strict=True):
            root, actual = task.sets["E"]
            assert task.labels[root, 0] == task.metadata[cell[0]]
            assert task.labels[actual, 0] == task.metadata[cell[1]]
            assert (task.labels[task.sets["D"], 0] == task.metadata[cell[0]]).all()
            rows = torch.arange(len(task.tokens))
            assert torch.equal(task.tokens[rows, task.positions[:, 0] + 1], task.labels[:, 0])
            for name in ("R", "V", "U"):
                ids = task.sets[name]
                assert torch.equal(task.labels[ids], task.old_labels[ids])
                assert torch.equal(task.tokens[ids], four[0].tokens[ids])
            assert task.sets == four[0].sets
        check_reused_task(four[0], old[2 * p].manifest())
        check_reused_task(four[1], old[2 * p + 1].manifest())


def test_d_is_not_used_to_construct_direct_controls():
    torch.manual_seed(27)
    gradients = torch.randn(3, 2, 3, 4, dtype=torch.float64)
    controls, info = isolated_directions(gradients)
    assert info["rank"] == 4 and info["control_residual"] < 1e-10
    actual = torch.einsum("nk,pqk->npq", controls.flatten(1), gradients[:2].flatten(2))
    expected = torch.tensor([[[0.5, -0.5], [0, 0]], [[0, 0], [0.5, -0.5]]]).double()
    torch.testing.assert_close(actual, expected)
    altered = gradients.clone()
    altered[2] *= -100
    new, _ = isolated_directions(altered)
    torch.testing.assert_close(new, controls)


def test_singular_controls_are_reported_not_assumed_identifiable():
    gradients = torch.ones(3, 2, 3, 4).double()
    _, info = isolated_directions(gradients)
    assert info["rank"] == 1
    assert info["control_residual"] > 0.1


def test_factorial_contrasts_distinguish_root_and_person():
    root = factorial_effects(dict(aa=3, ab=3, ba=-3, bb=-3))
    person = factorial_effects(dict(aa=3, ab=-3, ba=3, bb=-3))
    assert root == dict(intercept=0, root=3, actual=0, interaction=0)
    assert person == dict(intercept=0, root=0, actual=3, interaction=0)


def test_cached_derivative_matches_full_model_without_answer_prefix():
    from llm_memory_editability.bios_model import CausalLM, ModelConfig
    from llm_memory_editability.bios_relation_response import derivatives, make_probe

    torch.manual_seed(89)
    task = factorial_tasks(0, 0, "cpu")[0]
    model = CausalLM(
        ModelConfig(int(task.tokens.max()) + 1, width=8, layers=5, heads=2, context=8)
    ).double()
    for p in model.parameters():
        p.requires_grad_(False)
    w = model.blocks[4].mlp.down.weight
    probe = make_probe(model, task)
    logits, _, grad = derivatives(probe, w, task.metadata)
    ids = task.sets["E"] + task.sets["D_focal"]
    w.requires_grad_(True)
    native = model(task.tokens[ids], task.positions[ids, :1])[:, 0]
    torch.testing.assert_close(logits, native)
    a, c = task.metadata["a"], task.metadata["c"]
    actual = torch.autograd.grad(native[2, a] - native[2, c], w)[0]
    torch.testing.assert_close(actual, grad[2, 0])


def test_first_update_matches_frozen_editor_and_excludes_d(tmp_path, monkeypatch):
    from llm_memory_editability import bios_direction as engine
    from llm_memory_editability.bios_direction_cross import bounded_representation_geometry
    from llm_memory_editability.bios_model import CausalLM, ModelConfig
    from llm_memory_editability.bios_relation_response import first_adam_update

    torch.manual_seed(93)
    task = factorial_tasks(0, 0, "cpu")[2]
    model = CausalLM(
        ModelConfig(int(task.tokens.max()) + 1, width=8, layers=5, heads=2, context=8)
    ).double()
    for p in model.parameters():
        p.requires_grad_(False)
    config = dict(lr=0.0001, level=10.0, damping=0.01, rank=8, refresh=16, trust_fraction=0.05)
    delta = first_adam_update(model, task, config)
    original = task.labels.clone()
    task.labels[task.sets["D"], 0] = task.metadata["c"]
    torch.testing.assert_close(delta, first_adam_update(model, task, config))
    task.labels.copy_(original)
    monkeypatch.setattr(engine, "representation_geometry", bounded_representation_geometry)
    engine.edit_batch(model, 4, [task], "func-soft-adam", [config], 1, tmp_path)
    state = torch.load(tmp_path / "state.pt", weights_only=False)
    torch.testing.assert_close(state["weights"][0] - state["parent"], delta, rtol=2e-6, atol=1e-11)
