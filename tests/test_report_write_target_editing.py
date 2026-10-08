"""The report preserves world weighting, paired provenance, and incomplete status."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/report_write_target_editing.py"
SPEC = importlib.util.spec_from_file_location("report_write_target_editing", SCRIPT)
reporter = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(reporter)
finally:
    sys.path.pop(0)


def row(world, initialization, case, objective="final_ce", arm="edit", accuracy=0.0):
    tasks = {
        name: {"accuracy": accuracy, "old_answer_accuracy": 1 - accuracy}
        for name in ("E_new", "D_first_strict", "U_atomic", "U_strict")
    }
    return {
        "phase": "confirmation",
        "world": world,
        "initialization": initialization,
        "case_id": case,
        "objective": objective,
        "arm": arm,
        "old_fact": [21, 13, 31],
        "new_fact": [21, 13, 32],
        "parent_model_sha256": f"parent-{world}-{initialization}",
        "replay_indices": [0, 1],
        "steps": 512,
        "tasks": tasks,
        "early": {
            "new_entity_accuracy": accuracy,
            "old_entity_accuracy": 1 - accuracy,
            "new_direction_cosine": accuracy,
        },
    }


def test_hierarchical_average_does_not_count_cases_as_worlds():
    rows = [row(1, 1, str(case), accuracy=1.0) for case in range(10)]
    rows += [row(1, 2, "a", accuracy=0.0), row(2, 1, "b", accuracy=0.0)]
    # World1: mean(mean(10 ones), mean(1 zero))=.5; world2:0; final=.25.
    assert reporter.hierarchical_mean(rows, lambda r: r["tasks"]["E_new"]["accuracy"]) == 0.25


def test_pairing_uses_same_parent_case_and_retains_missing_counterparts():
    rows = [
        row(1, 1, "a", accuracy=0.25),
        row(1, 1, "a", arm="sham", accuracy=0.0),
        row(1, 1, "a", objective="early_ce", accuracy=0.75),
        row(1, 1, "b", objective="early_ce", accuracy=1.0),
    ]
    result = reporter.paired_differences(rows, "objective_minus_final_ce")
    assert result["missing_counterpart_n"] == 1
    assert result["summaries"][0]["paired_case_observations_n"] == 1
    assert result["summaries"][0]["equal_world_delta"]["D_new"] == 0.5
    rows[2]["replay_indices"] = [2, 3]
    with pytest.raises(AssertionError, match="replay_indices"):
        reporter.paired_differences(rows, "objective_minus_final_ce")


def test_duplicate_observations_across_input_roots_are_rejected():
    value = row(1, 1, "a")
    with pytest.raises(ValueError, match="Duplicate repeated measure"):
        reporter.paired_differences([value, copy.deepcopy(value)], "edit_minus_sham")


def test_partial_reports_never_treat_pending_runs_as_zero_scores(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    (root / "frozen-config.json").write_text(
        json.dumps(
            {
                "specs": [
                    {
                        "name": "pending-arm",
                        "phase": "confirmation",
                        "per_cell": 4,
                        "objectives": ["final_ce"],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Incomplete batch"):
        reporter.report([root])
    result = reporter.report([root], partial=True)
    assert not result["complete"] and result["completed_runs_n"] == 0
    assert result["expected_runs_n"] == 1
    assert result["case_records"] == [] and result["by_objective_arm"] == []
    assert "部分结果" in (root / "report/report.md").read_text()


def test_denominators_and_subset_correct_counts_remain_explicit():
    rows = [row(1, 1, "a"), row(2, 1, "b")]
    rows[0]["tasks"]["D_first_strict"] = {
        "n": 4,
        "correct": 1,
        "accuracy": 0.25,
        "parent_correct_n": 3,
        "old_answer_correct": 2,
        "old_answer_accuracy": 0.5,
        "old_retained_on_parent_correct": 2,
        "correct_on_parent_correct": 1,
    }
    rows[1]["tasks"]["D_first_strict"] = {
        "n": 2,
        "correct": 2,
        "accuracy": 1.0,
        "parent_correct_n": 1,
        "old_answer_correct": 0,
        "old_answer_accuracy": 0.0,
        "old_retained_on_parent_correct": 0,
        "correct_on_parent_correct": 1,
    }
    result = reporter.aggregate_task(rows, "D_first_strict")
    assert result["correct_n"] == 3 and result["query_evaluations_n"] == 6
    assert result["pooled_accuracy"] == 0.5 and result["equal_world_accuracy"] == 0.625
    assert result["parent_correct_n"] == 4 and result["correct_on_parent_correct_n"] == 2
    assert result["accuracy_on_parent_correct"] == 0.5
