"""CPU-only boundaries for endpoint audit; do not load models or start GPU work."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "endpoint_audit", ROOT / "scripts/audit_architecture_bridge_endpoints.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def prediction():
    return {
        "text": "answer",
        "tokens": [4, 9],
        "em": True,
        "f1": 1.0,
        "alias_em": True,
        "ended_eos": True,
        "scores": {
            "correct": {"tokens": [4, 9], "sum_logp": -2.0, "mean_logp": -1.0},
            "competitor": {"tokens": [5, 9], "sum_logp": -4.0, "mean_logp": -2.0},
            "margin": 1.0,
        },
    }


def test_token_segments_are_concatenated_without_retokenizing_or_losing_eos():
    prefix = torch.tensor([[10, 11]], dtype=torch.long)
    suffix = torch.tensor([[20, 21]], dtype=torch.long)
    target = torch.tensor([[30, 31, 9]], dtype=torch.long)
    inputs, kept_logits = audit.teacher_forcing_inputs(prefix, suffix, target)
    assert inputs.tolist() == [[10, 11, 20, 21, 30, 31]]
    assert kept_logits == 3
    assert target.tolist() == [[30, 31, 9]]
    inputs, kept_logits = audit.teacher_forcing_inputs(prefix, suffix, torch.tensor([[9]]))
    assert inputs.tolist() == [[10, 11, 20, 21]] and kept_logits == 1


def test_empty_teacher_forcing_segment_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        audit.teacher_forcing_inputs(
            torch.tensor([[1]]), torch.zeros((1, 0), dtype=torch.long), torch.tensor([[9]])
        )


def test_weight_restoration_is_exact_even_on_exception():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]), requires_grad=False)
    original = audit.tensor_hash(parameter)
    with pytest.raises(RuntimeError, match="replay failed"):
        with audit.temporary_weight(parameter, torch.tensor([[7.0, 8.0]])):
            assert parameter.tolist() == [[7, 8]]
            raise RuntimeError("replay failed")
    assert parameter.tolist() == [[1, 2]]
    assert audit.tensor_hash(parameter) == original
    assert parameter.requires_grad is False and parameter.grad is None


@pytest.mark.parametrize(
    "replacement",
    [torch.ones(3), torch.ones((1, 2), dtype=torch.float64), torch.tensor([[float("nan"), 1.0]])],
)
def test_endpoint_shape_dtype_and_finiteness_cannot_be_silently_changed(replacement):
    parameter = torch.nn.Parameter(torch.ones((1, 2)), requires_grad=False)
    with pytest.raises(ValueError):
        with audit.temporary_weight(parameter, replacement):
            raise AssertionError("Invalid endpoint was accepted")
    assert torch.equal(parameter, torch.ones((1, 2)))


def test_replay_does_not_enable_training():
    parameter = torch.nn.Parameter(torch.ones(2), requires_grad=True)
    with pytest.raises(ValueError, match="must not enable training"):
        with audit.temporary_weight(parameter, torch.zeros(2)):
            raise AssertionError("Trainable parameter accepted")


def test_token_mismatch_fails_even_with_identical_numeric_scores():
    recorder = audit.Audit()
    expected, actual = prediction(), prediction()
    actual["tokens"] = [5, 9]
    audit.compare_prediction(recorder, "test", actual, expected)
    assert any(
        check["name"] == "test:generated_tokens" and not check["passed"]
        for check in recorder.checks
    )
    assert recorder.max_numeric_difference == 0


def test_absolute_tolerance_is_fixed_and_nonfinite_scores_fail():
    recorder = audit.Audit()
    assert recorder.close("within", 0.0, 0.00009)
    assert not recorder.close("outside", 100.0, 100.00011)
    assert not recorder.close("nan", float("nan"), 0.0)
    assert audit.TOLERANCE == 1e-4
    json.dumps(recorder.checks, allow_nan=False)


def test_other_parameter_in_place_change_is_detected():
    model = torch.nn.Linear(2, 2).requires_grad_(False)
    before = audit.parameter_signature(model, model.weight)
    with torch.no_grad():
        model.bias.add_(1)
    recorder = audit.Audit()
    audit.compare_parameter_signatures(
        recorder, "untouched", audit.parameter_signature(model, model.weight), before
    )
    assert recorder.checks[-1]["passed"] is False
    assert recorder.checks[-1]["actual"] == ["bias"]


def test_frozen_loader_uses_snapshot_even_if_working_tree_is_broken(tmp_path):
    artifact = tmp_path / audit.ARTIFACT
    snapshot = artifact / "execution-source"
    archived = snapshot / audit.SOURCE
    archived.parent.mkdir(parents=True)
    archived.write_text('MARKER = "frozen"\n')
    working = tmp_path / audit.SOURCE
    working.parent.mkdir(parents=True)
    working.write_text('raise RuntimeError("working tree must not execute")\n')
    module = audit.load_frozen_engine(tmp_path, snapshot)
    assert module.MARKER == "frozen"
    assert module.ROOT == tmp_path
    assert module.CONFIG == snapshot / audit.CONFIG


def test_missing_matrix_is_enumerated_without_loading_any_model(tmp_path):
    config = {"models": {"a": {}, "b": {}}, "local_learning": {"episodes": 4}}
    cases = [{"dataset": "mquake", "id": "1121"}]
    missing = audit.completion_missing(tmp_path, config, cases)
    assert len(missing) == 18
    assert sum(name.endswith("endpoint.pt") for name in missing) == 8


def test_case_comparison_requires_the_entire_twenty_two_row_matrix():
    config = read_config = json.loads((ROOT / "configs/architecture-bridge-v1.json").read_text())
    row = {
        "id": "1121",
        "dataset": "mquake",
        "split": "development",
        "group": "group",
        "exclusion_reason": None,
    }
    row.update(
        {name: prediction() for name in ("unassisted", "first_hop", "second_hop", "oracle_input")}
    )
    row["interventions"] = [
        {"layer": layer, "condition": condition, **prediction()}
        for layer in read_config["models"]["qwen3"]["layers"]
        for condition in read_config["conditions"]
    ]
    actual = copy.deepcopy(row)
    actual["interventions"].pop()
    recorder = audit.Audit()
    audit.compare_case(recorder, actual, row, "qwen3", config)
    assert any(
        check["name"].endswith("observed_condition_count") and not check["passed"]
        for check in recorder.checks
    )
