"""Pairing, missingness, and EOS checks that affect report conclusions."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bridge_report", ROOT / "scripts/report_architecture_bridge.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def prediction(value, eos=9):
    correct = {"tokens": [2, eos], "sum_logp": -2.0, "mean_logp": -1.0}
    other = {"tokens": [3, eos], "sum_logp": -4.0, "mean_logp": -2.0}
    return {
        "em": value,
        "f1": value,
        "alias_em": value,
        "tokens": [2, eos],
        "ended_eos": True,
        "scores": {"correct": correct, "competitor": other, "margin": 1.0},
    }


def test_paired_comparison_never_matches_by_row_order_or_treats_missing_as_zero():
    cases = []
    for identifier, baseline, changed in [("a", 0, 1), ("b", 1, 0), ("c", 0, None)]:
        rows = [{"layer": 6, "condition": "baseline", **prediction(baseline)}]
        if changed is not None:
            rows.insert(0, {"layer": 6, "condition": "mlp_correct", **prediction(changed)})
        cases.append({"id": identifier, "group": identifier, "interventions": rows})
    result = report.contrast(cases, 6, {"mlp_correct": 1, "baseline": -1}, "em", 200, 42)
    assert result["n_cases"] == 2
    assert result["n_groups"] == 2
    assert result["estimate"] == 0
    assert (
        report.contrast(cases, 14, {"mlp_correct": 1, "baseline": -1}, "em", 200, 42)["estimate"]
        is None
    )


def test_repeated_group_is_one_bootstrap_unit():
    result = report.cluster_interval([1, 1, 1], ["same"] * 3, 100, 42)
    assert result["n_cases"] == 3 and result["n_groups"] == 1
    assert result["ci95"] == [None, None]
    assert result["estimate"] == 1
    with pytest.raises(ValueError, match="One group"):
        report.cluster_interval([1], [])


def test_factorial_interaction_uses_all_four_paired_cells():
    rows = [
        {"layer": 6, "condition": condition, **prediction(value)}
        for condition, value in [
            ("baseline", 0),
            ("mlp_correct", 1),
            ("state_correct", 1),
            ("joint_correct", 1),
        ]
    ]
    result = report.contrast(
        [{"group": "one", "interventions": rows}],
        6,
        {"joint_correct": 1, "mlp_correct": -1, "state_correct": -1, "baseline": 1},
        "em",
        100,
        42,
    )
    assert result["estimate"] == -1


def test_eos_and_log_probability_contract_errors_are_detected():
    result = prediction(1)
    assert not report.prediction_errors(result, 9, 32)
    result["scores"]["correct"]["tokens"] = [2]
    errors = report.prediction_errors(result, 9, 32)
    assert "correct_missing_eos_target" in errors
    assert "correct_sum_mean_inconsistent" in errors
    result["ended_eos"] = False
    assert "ended_eos_mismatch" in report.prediction_errors(result, 9, 32)


def test_empty_run_is_partial_and_has_no_zero_filled_metrics(tmp_path):
    path = tmp_path / "configs/architecture-bridge-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text((ROOT / "configs/architecture-bridge-v1.json").read_text())
    summary, audit, _ = report.build_report(tmp_path)
    assert summary["status"] == "partial"
    assert summary["complete"] is False and audit["complete"] is False
    assert summary["counts"]["expected_case_results"] == 112
    assert summary["counts"]["expected_learning_episodes"] == 8
    assert all(row["em"] is None and row["n"] == 0 for row in summary["interventions"])
    assert all(row["estimate"] is None for row in summary["paired_contrasts"])
    assert {row["split"] for row in summary["interventions"]} == {"development", "evaluation"}
    json.dumps(summary, allow_nan=False)


def test_learning_hash_changes_are_rejected(tmp_path):
    for relative in [
        "configs/architecture-bridge-v1.json",
        "data/architecture-bridge-v1/cases.json",
        "data/architecture-bridge-v1/learning.json",
    ]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / relative).read_text())
    artifact = tmp_path / "docs/development-artifacts/architecture-bridge-v1"
    artifact.mkdir(parents=True)
    lock = {
        name + "_sha256": report.digest(tmp_path / relative)
        for name, relative in [
            ("config", "configs/architecture-bridge-v1.json"),
            ("cases", "data/architecture-bridge-v1/cases.json"),
            ("learning", "data/architecture-bridge-v1/learning.json"),
        ]
    }
    (artifact / "data-lock.json").write_text(json.dumps(lock))
    learning = tmp_path / "data/architecture-bridge-v1/learning.json"
    learning.write_text(learning.read_text() + "\n")
    summary, audit, _ = report.build_report(tmp_path)
    assert summary["status"] == "invalid"
    assert any(check["name"] == "frozen_hash:learning" for check in audit["integrity_failures"])
