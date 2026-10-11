"""Cloud endpoint checks cannot equate an unfinished upload with audited completion."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "z1_cloud_endpoint", Path(__file__).resolve().parents[1] / "scripts/check_path_overlap_cloud.py"
)
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


def learning_metrics(record, previous=None):
    return {
        "training/step": record["step"],
        "eval/accuracy": record["metrics"]["accuracy"],
        "training/optimizer_updates": record["optimizer_updates"],
    }


def history():
    return [
        {
            "step": step,
            "optimizer_updates": 0,
            "source_training_step": (step - 1) % 14,
            "stage": "select" if step <= 14 else "science" if step <= 28 else "audit",
            "condition": "inherited" if step <= 28 else "audit",
            "wall_seconds": float(step),
            "metrics": {"accuracy": 0.5},
        }
        for step in range(1, 43)
    ]


def remote_run(expected, spec):
    rows = [{"_step": step, **deepcopy(metrics)} for step, metrics in expected.items()]
    return SimpleNamespace(
        state="finished",
        config={"spec": spec},
        summary={
            "scientific_final_step": 42,
            "independently_reloaded": True,
            "has_failure_record": False,
            "training_state": "complete",
        },
        scan_history=lambda **kw: iter(rows),
        rows=rows,
    )


def test_cloud_history_requires_all_42_analysis_records_and_exact_counts():
    expected = verification.expected_uploaded_metrics(history(), learning_metrics, list(range(14)))
    remote = remote_run(expected, {"optimizer_updates": 0})
    result = verification.compare_cloud_run(remote, expected, remote.config["spec"])
    assert result["passed"] is True
    assert result["analysis_records"] == 42
    assert result["metric_values_checked"] >= 42 * 6
    remote.rows[-1]["training/optimizer_updates"] = 1
    with pytest.raises(ValueError, match="metric differs"):
        verification.compare_cloud_run(remote, expected, remote.config["spec"])


@pytest.mark.parametrize(
    "failure",
    ["unfinished", "failed", "unaudited", "missing", "duplicate", "accuracy", "spec", "index"],
)
def test_unfinished_failed_partial_or_different_upload_cannot_pass(failure):
    expected = verification.expected_uploaded_metrics(history(), learning_metrics)
    remote = remote_run(expected, {"scientific": "frozen"})
    original_spec = deepcopy(remote.config["spec"])
    if failure == "unfinished":
        remote.state = "running"
    elif failure == "failed":
        remote.summary["has_failure_record"] = True
    elif failure == "unaudited":
        remote.summary["independently_reloaded"] = False
    elif failure == "missing":
        remote.rows.pop()
    elif failure == "duplicate":
        remote.rows.append(remote.rows[-1])
    elif failure == "accuracy":
        remote.rows[0]["eval/accuracy"] += 0.001
    elif failure == "spec":
        remote.config["spec"]["scientific"] = "changed"
    else:
        remote.rows[-1]["_step"] = 41
    with pytest.raises(ValueError):
        verification.compare_cloud_run(remote, expected, original_spec)


@pytest.mark.parametrize("failure", ["unfinished", "failed"])
def test_local_unfinished_or_failed_controller_is_rejected_before_cloud_api(tmp_path, failure):
    state = {
        "state": "audit_running" if failure == "unfinished" else "finished_with_failures",
        "failed": [] if failure == "unfinished" else ["failed audit"],
    }
    with pytest.raises(ValueError, match="Controller failed or did not complete"):
        verification.require_local_endpoint(tmp_path, state)


def test_local_analysis_source_nodes_must_match_frozen_order():
    records = history()
    records[-1]["source_training_step"] = 99
    with pytest.raises(ValueError, match="frozen 14 analysis nodes"):
        verification.expected_uploaded_metrics(records, learning_metrics, list(range(14)))


def test_endpoint_uses_actual_tracking_ids_and_reports_336_records(tmp_path, monkeypatch):
    base, runs, nodes = tmp_path, [f"run-{index}" for index in range(8)], list(range(14))
    verification.write(
        base / "controller-state.json",
        {
            "state": "complete",
            "active": [],
            "failed": [],
            "completed": runs,
        },
    )
    contract_path = base / "frozen/source/configs/v3.json"
    verification.write(contract_path, {"source": {"runs": runs, "nodes": nodes}})
    verification.write(
        base / "frozen/freeze.json",
        {
            "contract": {"path": "configs/v3.json", "sha256": verification.sha(contract_path)},
        },
    )
    verification.write(base / "summary.json", {"complete": True})
    verification.write(
        base / "audit-recovery-receipt.json",
        {
            "scientific_freeze_sha256": verification.sha(base / "frozen/freeze.json"),
            "comparison_contract": {"cosine_only": True},
        },
    )
    verification.write(
        base / "completion-manifest.json",
        {
            "passed": True,
            "runs": runs,
            "independently_replayed_nodes": 112,
            "new_optimizer_updates": 0,
            "new_scientific_interventions": 0,
            "audit_recovery_receipt_sha256": verification.sha(base / "audit-recovery-receipt.json"),
            "comparison_contract": {"cosine_only": True},
            "summary_sha256": verification.sha(base / "summary.json"),
        },
    )
    verification.write(
        base / "tracking-completion.json",
        {
            "passed": True,
            "registered_runs": 8,
            "runs": {run: {"last_logged_step": 42, "final_step": 42} for run in runs},
        },
    )
    queried, remotes = [], {}
    for index, run in enumerate(runs):
        folder = base / "runs" / run
        verification.write(
            folder / "audit.json",
            {
                "passed": True,
                "fresh_process": True,
                "fresh_container": True,
                "nodes": [{"step": step, "passed": True} for step in nodes],
            },
        )
        verification.write(
            folder / "complete.json",
            {
                "audit_sha256": verification.sha(folder / "audit.json"),
                "analysis_index": 42,
            },
        )
        verification.write(folder / "learning.json", history())
        metadata_spec = {"scientific": "frozen", "optimizer_updates": 0}
        verification.write(folder / "run.json", {"spec": metadata_spec})
        actual_id = f"actual-{index}"
        verification.write(
            base / "tracking-wandb" / run / "state.json",
            {
                "wandb_run_id": actual_id,
                "run_id": f"different-legacy-id-{index}",
                "entity": "team",
                "project": "project",
                "mode": "online",
                "last_logged_step": 42,
            },
        )
        expected = verification.expected_uploaded_metrics(history(), learning_metrics)
        remotes[f"team/project/{actual_id}"] = remote_run(expected, metadata_spec)
    monkeypatch.setattr(
        verification,
        "load_tracking",
        lambda base: SimpleNamespace(
            learning_metrics=learning_metrics,
            public_config=lambda value: value,
        ),
    )

    def query(path):
        queried.append(path)
        return remotes[path]

    result = verification.verify_endpoint(base, SimpleNamespace(run=query))
    assert result["passed"] is True
    assert result["analysis_records"] == 336
    assert len(queried) == 8
    assert all("actual-" in path for path in queried)
    verification.write(base / "runs" / runs[0] / "failure.json", {"failure": True})
    with pytest.raises(ValueError, match="Local run lacks"):
        verification.verify_endpoint(base, SimpleNamespace(run=query))
