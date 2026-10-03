"""Scientific checks: scoring prerequisites, world weighting and paired contrasts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "report_multihop_scaling", SCRIPTS / "report_multihop_scaling.py"
)
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)
SPEC_SCAN = importlib.util.spec_from_file_location(
    "analyze_multihop_scaling", SCRIPTS / "analyze_multihop_scaling.py"
)
SCAN = importlib.util.module_from_spec(SPEC_SCAN)
SPEC_SCAN.loader.exec_module(SCAN)


def test_worlds_receive_equal_weight_with_unequal_initializer_count():
    rows = [
        {"world": 1, "initialization": 1, "task": "atomic", "accuracy": 1.0},
        {"world": 1, "initialization": 2, "task": "atomic", "accuracy": 1.0},
        {"world": 2, "initialization": 1, "task": "atomic", "accuracy": 0.0},
    ]
    worlds, aggregate = REPORT.world_balanced(rows, ("task",), ("accuracy",))
    assert len(worlds) == 2
    assert aggregate[0]["accuracy"] == 0.5
    assert aggregate[0]["runs"] == 3
    with pytest.raises(ValueError, match="Duplicate"):
        REPORT.world_balanced(rows + rows[:1], ("task",), ("accuracy",))


def test_contrasts_are_paired_within_initialization_and_not_selected_by_score():
    rows = []
    for initialization, offset in ((3, 0), (4, 0.1)):
        for width in (128, 256):
            for phi in (1.0, 4.0):
                for architecture in ("standard", "loop"):
                    for depth in (2, 4):
                        value = offset + 0.02 * (depth == 4) + 0.03 * (phi == 4)
                        value += 0.05 * (depth == 4 and phi == 4)
                        value += 0.07 * (width == 256 and phi == 4)
                        rows.append(
                            {
                                "world": 1,
                                "initialization": initialization,
                                "step": 64000,
                                "task": "familiar_3",
                                "width": width,
                                "phi": phi,
                                "architecture": architecture,
                                "executed_depth": depth,
                                "accuracy": value,
                            }
                        )
    contrasts = REPORT.paired_contrasts(rows)
    depth = [row for row in contrasts if row["contrast"] == "support_x_depth"]
    width = [row for row in contrasts if row["contrast"] == "width_x_support"]
    assert len(depth) == 8 and len(width) == 8
    assert all(row["difference_pp"] == pytest.approx(5) for row in depth)
    assert all(row["difference_pp"] == pytest.approx(7) for row in width)
    incomplete = REPORT.paired_contrasts(rows[:-1])
    assert len(incomplete) < len(contrasts)


def test_raw_eos_and_empty_pool():
    rows = np.array([[3, 67, 9], [4, 67, 10], [5, 67, 11]])
    accuracy, correct = REPORT.generated_score(rows, np.array([[9, 1], [10, 2], [12, 1]]))
    assert accuracy == pytest.approx(1 / 3)
    np.testing.assert_array_equal(correct, [True, False, False])
    accuracy, correct = REPORT.generated_score(np.empty((0, 3)), np.empty((0, 2)))
    assert accuracy is None and correct.shape == (0,)
    with pytest.raises(ValueError, match="answer and EOS"):
        REPORT.generated_score(rows, np.zeros((3, 1)))


def test_independent_graph_audit_checks_constituents_and_path_not_just_terminal():
    import torch

    from llm_memory_editability.multihop_scaling import build_world, construct, evaluate

    torch.set_num_threads(1)
    spec = {
        "world_seed": 901031,
        "phi": 1.0,
        "width": 8,
        "heads": 2,
        "layers": 1,
        "repeats": 2,
        "initialization": 901032,
    }
    world = build_world(spec)
    model = construct(spec, "cpu")
    metrics, predictions = evaluate(model, world, "cpu")
    assert REPORT.audit_predictions(world, predictions, metrics)["passed"]
    world_digest = REPORT.common_digest(world, world["metadata"]["id_mask"])
    assert world_digest == world["metadata"]["common_evaluation_sha256"]
    broken = {key: value.copy() for key, value in predictions.items()}
    broken["familiar_2_coverage"][0] = not broken["familiar_2_coverage"][0]
    with pytest.raises(AssertionError):
        REPORT.audit_predictions(world, broken, metrics)
    broken = {key: value.copy() for key, value in predictions.items()}
    broken["familiar_2_autonomous_path_correct"][0] = not broken[
        "familiar_2_autonomous_path_correct"
    ][0]
    with pytest.raises(AssertionError):
        REPORT.audit_predictions(world, broken, metrics)


def test_role_experience_separates_position_practice_and_atomic_exposure():
    atoms = np.array([[3, 7, 4], [4, 7, 5], [5, 7, 5]])
    world = {"atomic": atoms}
    exposures = {"atomic": np.array([10, 10, 10])}
    for hop in (2, 3, 4):
        trained = np.array([[3, *([7] * hop), 5]])
        world[f"train_{hop}"] = trained
        world[f"familiar_{hop}"] = np.array([[4, *([7] * hop), 5]])
        world[f"strict_{hop}"] = np.empty((0, hop + 2), dtype=np.int64)
        exposures[f"train_{hop}"] = np.array([5])
    records = REPORT.training_experience(world, exposures)
    familiar = next(row for row in records if row["task"] == "familiar_2")
    assert familiar["all_facts_in_any_composition_training"] == 1
    assert familiar["all_facts_in_same_position_training"] == 0
    assert familiar["atomic_training_draws"] == 30
    assert familiar["position_1_role_coverage"] == 0
    assert familiar["position_1_atomic_exposure_mean"] == 10


def test_same_weight_differences_use_training_r_and_include_all_r():
    rows = []
    for repeats, accuracy in ((1, 0.4), (2, 0.5), (4, 0.3), (6, 0.2)):
        rows.append(
            {
                "run": "test",
                "world": 1,
                "initialization": 1,
                "width": 128,
                "phi": 4,
                "architecture": "loop",
                "executed_depth": 2,
                "repeats": 2,
                "step": 8000,
                "task": "familiar_2",
                "test_repeats": repeats,
                "accuracy": accuracy,
            }
        )
    result = SCAN.scan_differences(rows)
    assert [row["test_repeats"] for row in result] == [1, 2, 4, 6]
    assert [row["difference_pp"] for row in result] == pytest.approx([-10, 0, -20, -30])
    assert SCAN.scan_differences(rows[:1]) == []
