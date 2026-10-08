"""Reporting must keep incomplete endpoints and recipe-selection isolation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/report_sequential_transfer.py"
SPEC = importlib.util.spec_from_file_location("sequential_report", MODULE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def spec(history):
    return {
        "name": history,
        "history": history,
        "stage_a_steps": 2,
        "stage_b_steps": 2,
        "data_file": "development.json",
        "data_sha256": "data",
        "initialization": 17,
    }


def record(step, aa=0.8, bb=0.1):
    return {
        "step": step,
        "metrics": {
            pool: {"n": 8, "answer_accuracy": bb if pool == "BB" else aa, "nll": 0.5}
            for pool in report.POOLS
        },
    }


def test_missing_endpoints_are_null_not_latest_checkpoint(tmp_path):
    selected = spec("sequential_composition")
    out = tmp_path / "runs" / selected["name"]
    out.mkdir(parents=True)
    (out / "learning.json").write_text(json.dumps([record(0), record(2), record(3)]))
    summary = report.summarize({"specs": [selected]}, tmp_path)
    assert summary["runs"][0]["latest_step"] == 3
    assert summary["runs"][0]["stage_a"]["AA"]["answer_accuracy"] == 0.8
    assert summary["runs"][0]["endpoint"]["BB"] is None
    assert summary["runs"][0]["AA_change"] is None


def test_readiness_never_depends_on_B_composition():
    stage_a, endpoint = record(2), record(4)
    original = report.readiness_evidence(spec("joint"), stage_a, endpoint, True)
    changed = copy.deepcopy(endpoint)
    for pool in ("BB", "BA", "AB"):
        changed["metrics"][pool] = {"n": 999, "answer_accuracy": 1.0}
    assert report.readiness_evidence(spec("joint"), stage_a, changed, True) == original
    assert not any(pool in original["endpoint"] for pool in ("BB", "BA", "AB"))


def test_paired_hash_and_sham_changes_are_reported(tmp_path):
    histories = (
        "sequential_composition",
        "sequential_atomic",
        "joint",
        "sequential_composition_sham",
    )
    specs = [spec(history) for history in histories]
    for selected in specs:
        out = tmp_path / "runs" / selected["name"]
        out.mkdir(parents=True)
        final_aa = 0.7 if selected["history"] == "sequential_composition" else 0.75
        report.write(out / "learning.json", [record(2), record(4, aa=final_aa)])
        report.write(out / "run.json", {"initial_model_sha256": "identical"})
        report.write(
            out / "sampling-plan.json",
            {
                "multiset_sha256": "same",
                "examples": 24,
                "supervised_tokens": 48,
            },
        )
    summary = report.summarize({"specs": specs}, tmp_path)
    pair = summary["paired_comparisons"][0]
    assert pair["initial_hashes_identical"] is True
    assert pair["joint_sequential_multiset_identical"] is True
    assert pair["AA_change_B_minus_sham"] == pytest.approx(-0.05)
    assert "sequential_composition" in report.render(summary)


def test_hierarchical_mean_does_not_pool_initializations_across_splits():
    rows = [
        {"split": "one", "value": 0.0},
        {"split": "one", "value": 1.0},
        {"split": "two", "value": 1.0},
    ]
    result = report.hierarchical_values(rows, lambda row: row["value"])
    assert result["mean"] == 0.75
    assert result["by_split"]["one"]["mean"] == 0.5
    rows.append({"split": "two", "value": None})
    partial = report.hierarchical_values(rows, lambda row: row["value"])
    assert partial["mean"] is None
    assert partial["available_mean"] == 0.75
    assert partial["complete"] is False


def test_prediction_counts_are_checked_and_retention_differs_from_net_change():
    before = {"AA": [{"id": str(i), "alias_em": int(i < 2)} for i in range(4)]}
    after = {"AA": [{"id": str(i), "alias_em": int(1 <= i < 3)} for i in range(4)]}
    transition = report.preservation(before, after)
    assert transition["stage_a_correct"] == transition["endpoint_correct"] == 2
    assert transition["retained_correct"] == 1
    assert transition["lost_correct"] == transition["gained_correct"] == 1
    assert transition["retention_of_stage_a_correct"] == 0.5
    value = {"metrics": {"AA": {"n": 4, "answer_accuracy": 0.5}}}
    assert report.metric(value, "AA", after)["correct"] == 2
    value["metrics"]["AA"]["answer_accuracy"] = 0.75
    with pytest.raises(ValueError, match="accuracy disagrees"):
        report.metric(value, "AA", after)


def test_pooled_counts_preserve_prediction_denominators():
    rows = [
        {
            "split": "one",
            "endpoint": {
                "BB": {
                    "n": 2,
                    "correct": 1,
                    "answer_accuracy": 0.5,
                    "counts_verified_from_predictions": True,
                }
            },
        },
        {
            "split": "two",
            "endpoint": {
                "BB": {
                    "n": 10,
                    "correct": 10,
                    "answer_accuracy": 1.0,
                    "counts_verified_from_predictions": True,
                }
            },
        },
    ]
    result = report.aggregate_cell(rows, "endpoint", "BB")
    assert result["mean"] == 0.75
    assert result["pooled_correct"] == 11
    assert result["pooled_prediction_n"] == 12
    assert result["mean"] != result["pooled_correct"] / result["pooled_prediction_n"]
