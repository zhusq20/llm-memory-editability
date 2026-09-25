"""Research controls and trainable nonlinear models, without convergence assumptions."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from llm_memory_editability.neural import evaluate  # noqa: E402
from llm_memory_editability.neural_data import make_world  # noqa: E402
from llm_memory_editability.neural_models import build_model, edit_parameters  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _model(model_type):
    return build_model(
        model_type,
        n_entities=24,
        n_attributes=8,
        n_answers=4,
        width=8,
        hidden=16,
        layers=1,
        heads=2,
    )


@pytest.mark.parametrize("seed", [0, 7, 23])
def test_paired_updates_match_support_coverage_and_per_entity_answer_distribution(seed):
    world = make_world(seed=seed)
    rule_mask = world.rule != world.original
    exception_mask = world.exception != world.original

    np.testing.assert_array_equal(rule_mask, world.edit_mask)
    np.testing.assert_array_equal(exception_mask, world.edit_mask)
    assert world.edit_mask.any() and (~world.edit_mask).any()
    assert np.any(world.rule != world.exception)
    np.testing.assert_array_equal(world.rule[~world.edit_mask], world.original[~world.edit_mask])
    np.testing.assert_array_equal(
        world.exception[~world.edit_mask], world.original[~world.edit_mask]
    )
    for axis in (0, 1):
        np.testing.assert_array_equal(
            np.unique(world.x[rule_mask, axis]), np.unique(world.x[exception_mask, axis])
        )
    for entity in np.unique(world.x[:, 0]):
        selected = world.edit_mask & (world.x[:, 0] == entity)
        np.testing.assert_array_equal(
            np.bincount(world.rule[selected], minlength=4),
            np.bincount(world.exception[selected], minlength=4),
        )


def test_rule_preserves_group_attribute_structure_and_exception_breaks_it():
    world = make_world(seed=7)
    broken_cells = 0
    for group in np.unique(world.group_ids):
        for attribute in np.unique(world.x[:, 1]):
            selected = (world.group_ids[world.x[:, 0]] == group) & (world.x[:, 1] == attribute)
            assert len(np.unique(world.original[selected])) == 1
            assert len(np.unique(world.rule[selected])) == 1
            broken_cells += len(np.unique(world.exception[selected])) > 1
    assert broken_cells > 0


def test_world_generation_is_seeded():
    first = make_world(seed=7)
    same = make_world(seed=7)
    different = make_world(seed=8)
    for field in ("x", "original", "rule", "exception", "edit_mask", "group_ids"):
        np.testing.assert_array_equal(getattr(first, field), getattr(same, field))
    assert any(
        not np.array_equal(getattr(first, field), getattr(different, field))
        for field in ("original", "rule", "exception", "group_ids")
    )


def test_world_requires_retained_facts():
    with pytest.raises(ValueError, match="retained facts"):
        make_world(n_groups=1)


def test_metrics_use_target_specific_margins_and_correct_fact_subsets():
    logits = torch.tensor([[3, 1, 0], [0, 2, 1], [3, 4, 0], [1, 0, 3]], dtype=torch.float64)
    targets = torch.tensor([0, 2, 1, 0])
    edit_mask = torch.tensor([True, True, False, False])

    metrics = evaluate(torch.nn.Identity(), logits, targets, edit_mask)

    assert metrics["all_accuracy"] == 0.5
    assert metrics["edit_accuracy"] == 0.5
    assert metrics["retain_accuracy"] == 0.5
    assert metrics["edit_margin"] == 0.5  # mean of +2 and -1 against strongest competitor
    assert metrics["retain_margin"] == -0.5  # mean of +1 and -2
    expected_nll = (np.logaddexp.reduce([3, 1, 0]) - 3 + np.logaddexp.reduce([0, 2, 1]) - 1) / 2
    assert metrics["edit_target_nll"] == pytest.approx(expected_nll)


@pytest.mark.parametrize("model_type", ["mlp", "transformer"])
def test_nonlinear_models_produce_answer_logits_and_backpropagate(model_type):
    torch.manual_seed(0)
    model = _model(model_type)
    world = make_world(seed=0)
    logits = model(torch.as_tensor(world.x[:8]))

    assert logits.shape == (8, 4)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.as_tensor(world.original[:8])).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0
    if model_type == "transformer":
        assert any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())
    else:
        assert any(isinstance(module, (torch.nn.GELU, torch.nn.ReLU)) for module in model.modules())


@pytest.mark.parametrize("model_type", ["mlp", "transformer"])
def test_ffn_edit_updates_only_selected_parameters_and_clones_are_independent(model_type):
    torch.manual_seed(0)
    baseline = _model(model_type)
    edited = copy.deepcopy(baseline)
    other_case = copy.deepcopy(baseline)
    initial = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
    selected = edit_parameters(edited, scope="ffn")
    selected_ids = {id(parameter) for parameter in selected}
    expected_names = {
        name
        for name, _ in edited.named_parameters()
        if (
            name.startswith("hidden.")
            if model_type == "mlp"
            else ".linear1." in name or ".linear2." in name
        )
    }
    actual_names = {
        name for name, parameter in edited.named_parameters() if parameter.requires_grad
    }

    assert expected_names and actual_names == expected_names
    assert selected_ids == {
        id(parameter) for parameter in edited.parameters() if parameter.requires_grad
    }
    world = make_world(seed=0)
    optimizer = torch.optim.SGD(selected, lr=0.1)
    loss = torch.nn.functional.cross_entropy(
        edited(torch.as_tensor(world.x)), torch.as_tensor(world.rule)
    )
    loss.backward()
    optimizer.step()

    changed_names = {
        name
        for name, parameter in edited.named_parameters()
        if not torch.equal(parameter, initial[name])
    }
    assert changed_names and changed_names <= expected_names
    for untouched in (baseline, other_case):
        for name, value in untouched.state_dict().items():
            assert torch.equal(value, initial[name])
    all_selected = edit_parameters(edited, scope="all")
    assert {id(parameter) for parameter in all_selected} == {
        id(parameter) for parameter in edited.parameters()
    }
    assert all(parameter.requires_grad for parameter in edited.parameters())


def test_neural_cli_reports_paired_edit_and_scratch_budgets_as_strict_json(tmp_path):
    output_path = tmp_path / "neural.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_memory_editability.neural",
            "--model",
            "mlp",
            "--seeds",
            "0",
            "--pretrain-steps",
            "2",
            "--edit-steps",
            "2",
            "--scratch-steps",
            "2",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = json.loads(completed.stdout)
    json.dumps(report, allow_nan=False)
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert report["config"]["model"] == "mlp"
    assert report["config"]["seeds"] == [0]
    for budget in ("pretrain_steps", "edit_steps", "scratch_steps"):
        assert report["config"][budget] == 2
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert run["baseline_training"]["steps"] == 2
    assert set(run["cases"]) == {"rule", "exception"}
    for case in run["cases"].values():
        assert case["edit_training"]["steps"] == 2
        assert case["scratch_training"]["steps"] == 2
        for stage in ("before", "after", "scratch"):
            metrics = case[stage]
            for name in ("all_accuracy", "edit_accuracy", "retain_accuracy"):
                assert 0 <= metrics[name] <= 1
            assert metrics["edit_target_nll"] >= 0
            assert np.isfinite(metrics["edit_margin"])
            assert np.isfinite(metrics["retain_margin"])
    assert (
        run["cases"]["rule"]["before"]["retain_accuracy"]
        == run["cases"]["exception"]["before"]["retain_accuracy"]
    )
