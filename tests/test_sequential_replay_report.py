"""Protect paired evidence, actual retention, and fixed-endpoint reporting."""

import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/report_sequential_replay.py"
SPEC = importlib.util.spec_from_file_location("sequential_replay_report", MODULE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def observations(step, *, aa=(1, 1, 0, 0), bb=(0, 0, 0, 0)):
    raw = {}
    for pool in report.POOLS:
        values = bb if pool == "BB" else aa
        raw[pool] = [{"id": f"{pool}-{i}", "alias_em": value} for i, value in enumerate(values)]
    for pool in report.COMPOSITIONS:
        raw[pool + "_autonomous"] = [{"id": item["id"], "alias_em": 1} for item in raw[pool]]
    record = {
        "step": step,
        "branch_step": max(0, step - 2),
        "metrics": {
            pool: {
                "n": len(items),
                "answer_accuracy": sum(item["alias_em"] for item in items) / len(items),
            }
            for pool, items in raw.items()
        },
    }
    return record, raw


def batch(tmp_path):
    parent = tmp_path / "parent"
    parent_spec = {"initialization": 17, "split": "split-0", "data_sha256": "data"}
    report.write(parent / "run.json", {"spec": parent_spec})
    baseline, raw = observations(2)
    report.write(parent / "learning.json", [baseline])
    report.write(parent / "predictions-0000002.json", raw)
    specs = []
    for arm in report.ARMS:
        spec = {
            "name": arm,
            "replay_arm": arm,
            "parent_run_dir": str(parent),
            "original_stage_a_steps": 2,
            "stage_b_steps": 2,
            "parent_checkpoint_sha256": "checkpoint",
        }
        specs.append(spec)
        out = tmp_path / "runs" / arm
        report.write(
            out / "run.json", {"parent_spec": parent_spec, "parent_model_sha256": "weights"}
        )
        report.write(out / "sampling-plan.json", {"new_stream_sha256": "same-B"})
        aa = (0, 1, 1, 0) if arm == "atom_replay" else (1, 1, 1, 0)
        bb = (0, 0, 1, 0) if arm == "atom_replay" else (0, 1, 1, 0)
        endpoint, predictions = observations(4, aa=aa, bb=bb)
        report.write(out / "learning.json", [baseline, endpoint])
        report.write(out / "predictions-0000002.json", raw)
        report.write(out / "predictions-0000004.json", predictions)
        report.write(out / "audit.json", {"passed": True})
    return {"specs": specs, "phase": "paired_extension"}


def test_fixed_endpoints_do_not_use_latest_or_parent_endpoint(tmp_path):
    config = batch(tmp_path)
    out = tmp_path / "runs/full_replay"
    intermediate, raw = observations(3, aa=(1, 1, 1, 1), bb=(1, 1, 1, 1))
    report.write(out / "learning.json", [intermediate])
    report.write(out / "predictions-0000003.json", raw)
    summary = report.summarize(config, tmp_path)
    full = summary["runs"][1]
    assert full["stage_a_source"] == "parent_recorded"
    assert full["stage_a"]["AA"]["correct"] == 2
    assert full["latest_step"] == 3
    assert full["endpoint"]["BB"] is None
    assert full["changes"]["BB"] is None
    assert summary["arm_aggregates"]["full_replay"]["endpoint"]["BB"]["mean"] is None


def test_actual_retention_and_paired_raw_counts(tmp_path):
    summary = report.summarize(batch(tmp_path), tmp_path)
    atomic = summary["runs"][0]
    assert atomic["changes"]["AA"] == 0
    assert atomic["transitions"]["AA"]["retained_correct"] == 1
    assert atomic["transitions"]["AA"]["lost_correct"] == 1
    assert atomic["transitions"]["AA"]["gained_correct"] == 1
    assert atomic["transitions"]["AA"]["retention_of_before_correct"] == 0.5
    full = summary["arm_aggregates"]["full_replay"]
    assert full["endpoint"]["BB"]["pooled_correct"] == 2
    assert full["endpoint"]["BB"]["pooled_prediction_n"] == 4
    pair = summary["paired_comparisons"][0]
    assert pair["pairing_verified"] is True
    assert pair["endpoint_full_minus_atomic"]["BB"] == 0.25
    assert pair["change_full_minus_atomic"]["BB"] == 0.25
    assert pair["endpoint_autonomous_full_minus_atomic"]["BB"] == 0
    assert "2/4" in report.render(summary)


def test_mismatched_new_fact_stream_invalidates_paired_contrast(tmp_path):
    config = batch(tmp_path)
    report.write(
        tmp_path / "runs/full_replay/sampling-plan.json", {"new_stream_sha256": "different-B"}
    )
    summary = report.summarize(config, tmp_path)
    pair = summary["paired_comparisons"][0]
    assert pair["pairing_verified"] is False
    assert pair["endpoint_full_minus_atomic"]["BB"] is None
    assert summary["runs"][1]["endpoint"]["BB"]["correct"] == 2


def test_hierarchical_means_keep_raw_count_unit_and_incomplete_matrix():
    rows = [
        {
            "split": split,
            "endpoint": {
                "BB": {
                    "answer_accuracy": correct / n,
                    "correct": correct,
                    "n": n,
                    "counts_verified_from_predictions": True,
                }
            },
        }
        for split, correct, n in (("one", 0, 2), ("one", 2, 2), ("two", 10, 10))
    ]
    result = report.aggregate_cell(rows, "endpoint", "BB")
    assert result["mean"] == 0.75
    assert result["pooled_correct"] == 12
    assert result["pooled_prediction_n"] == 14
    rows.append({"split": "two", "endpoint": {"BB": None}})
    result = report.aggregate_cell(rows, "endpoint", "BB")
    assert result["mean"] is None
    assert result["available_mean"] == 0.75
    assert result["counts_complete"] is False


def test_development_operating_evidence_excludes_B_composition(tmp_path):
    config = batch(tmp_path)
    original = report.summarize(config, tmp_path)["development_readiness_evidence"]
    out = tmp_path / "runs/full_replay"
    learning = report.read(out / "learning.json")
    raw = report.read(out / "predictions-0000004.json")
    for pool in ("BA", "AB", "BB"):
        learning[-1]["metrics"][pool]["answer_accuracy"] = 1.0
        for item in raw[pool]:
            item["alias_em"] = 1
    report.write(out / "learning.json", learning)
    report.write(out / "predictions-0000004.json", raw)
    assert report.summarize(config, tmp_path)["development_readiness_evidence"] == original


def test_prediction_disagreement_and_duplicate_ids_are_errors():
    record, raw = observations(4)
    record["metrics"]["AA"]["answer_accuracy"] = 1.0
    with pytest.raises(ValueError, match="accuracy disagrees"):
        report.metric(record, "AA", raw)
    raw["AA"].append(raw["AA"][0])
    with pytest.raises(ValueError, match="unique IDs"):
        report.preservation(raw, raw, "AA")
