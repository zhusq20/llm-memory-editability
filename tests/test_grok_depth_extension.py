"""Contract checks for paired timing, censoring and unchanged experimental inputs."""

from pathlib import Path

import pytest

from llm_memory_editability.grok_depth_extension import read_json, threshold_time, validate_pair

ROOT = Path(__file__).resolve().parents[1]


def rows(values):
    return [
        {
            "step": i * 2000,
            "test_composite": {"accuracy": value},
            "examples": i * 2000 * 256,
            "counts": {"atomic": i * 70000, "composite": i * 442000},
            "estimated_training_flops": i * 1e12,
            "effective_input_tokens": i * 1978000,
            "supervised_tokens": i * 1024000,
        }
        for i, value in enumerate(values)
    ]


def score(values, target=0.95):
    return threshold_time(rows(values), target, {"atomic": 1024, "train_composite": 5838}, 128000)


def test_threshold_requires_three_consecutive_nodes_not_best_checkpoint():
    result = score([0.96, 0.96, 0.94, 0.95, 0.95, 0.96, 0.90])
    assert result["step"] == 6000
    assert result["confirmed_at_step"] == 10000
    assert result["previous_evaluation_step"] == 4000
    assert result["examples"] == 6000 * 256
    assert result["atomic_exposures"] == 210000 / 1024


def test_two_final_success_nodes_are_censored_and_not_interpolated():
    result = score([0.91, 0.94, 0.97, 0.98])
    assert not result["reached"] and result["step"] is None
    assert result["last_observed_step"] == 6000


def test_threshold_is_inclusive_and_exact_without_percent_rounding():
    assert score([0.95, 0.95, 0.95])["step"] == 0
    assert not score([0.95 - 1e-12, 0.95, 0.95])["reached"]


def test_duplicate_nodes_are_rejected():
    data = rows([1, 1, 1])
    data[1]["step"] = 0
    with pytest.raises(AssertionError):
        threshold_time(data, 0.95, {"atomic": 1024, "train_composite": 5838}, 128000)


def test_full_paired_matrix_changes_only_depth_and_phase():
    cfg = read_json(ROOT / "configs/grok-depth-extension-v1.json")
    old = read_json(ROOT / "configs/grok-depth-v1.json")
    assert cfg["base"] == old["base"]
    seen = set()
    for run_id, item in cfg["runs"].items():
        spec = {**cfg["base"], **item}
        baseline = {**old["base"], **old["runs"][cfg["baseline_for_run"][run_id]]}
        validate_pair(spec, baseline)
        seen.add((spec["world_seed"], spec["initialization"], spec["layers"]))
    assert seen == {
        (w, s, d) for w in (143011, 143012, 143013) for s in (14301, 14302) for d in (3, 4)
    }


def test_pair_validation_rejects_learning_rate_or_node_changes():
    original = {"layers": 2, "phase": "confirmation", "lr": 1e-4, "nodes": [0, 2, 4]}
    for change in ({"lr": 1e-3}, {"nodes": [0, 1, 4]}):
        with pytest.raises(AssertionError):
            validate_pair({**original, "layers": 3, "phase": "extension", **change}, original)
