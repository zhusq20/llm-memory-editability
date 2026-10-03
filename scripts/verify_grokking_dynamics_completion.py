"""Check registered completeness, immutable source snapshots and audit evidence."""

from __future__ import annotations

import json
import platform
import shutil

import numpy as np
import torch
from run_grokking_dynamics import ARTIFACTS, RESULTS, config_path, run_name

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import file_hash


def read(path):
    return json.loads(path.read_text())


def main():
    source_checks = []
    phases = []
    for phase in ["calibration", "development"]:
        config = read(config_path(phase))
        frozen = ARTIFACTS / phase / "source"
        for path, sha in config["source"].items():
            assert file_hash(frozen / path) == sha
            source_checks.append(
                {
                    "phase": phase,
                    "path": path,
                    "snapshot_sha256": sha,
                    "current_matches": file_hash(path) == sha,
                }
            )
        execution = read(ARTIFACTS / phase / "execution.json")
        assert len(execution["statuses"]) == len(config["specs"])
        assert all(status.get("returncode", 0) == 0 for status in execution["statuses"])
        for spec in config["specs"]:
            out = RESULTS / phase / run_name(spec)
            complete = read(out / "complete.json")
            assert complete["spec"] == spec
            assert complete["data_sha256"] == config["data_sha256"]
            assert file_hash(out / "latest.pt") == complete["checkpoint_sha256"]
            assert read(out / "audit.json")["passed"]
            assert [row["step"] for row in read(out / "learning.json")] == spec["nodes"]
        summary = read(ARTIFACTS / phase / "report/summary.json")
        assert summary["runs"] == len(config["specs"])
        phases.append(
            {
                "phase": phase,
                "runs": len(config["specs"]),
                "updates": summary["total_updates"],
                "training_seconds": summary["total_training_seconds"],
                "parallel_execution_seconds": execution["seconds"],
            }
        )
    independent = read(ARTIFACTS / "development/independent-audit.json")
    assert independent["passed"] and independent["runs"] == 12
    assert independent["retained_checkpoint_replays"] == 168
    assert independent["rescored_learning_nodes"] == 840
    assert independent["max_nll_difference"] == 0
    mechanism = []
    for phase in ["calibration", "development"]:
        folder = ARTIFACTS / "mechanism" / phase
        config = read(folder / "config.json")
        for path, sha in config["source"].items():
            assert file_hash(folder / "source" / path) == sha
        audit_tasks, branches = 0, 0
        for entry in config["matrix"]:
            name = f"{run_name(entry['spec'])}-t{entry['node']:06d}"
            out = RESULTS / "mechanism" / phase / name
            assert read(out / "complete.json")["passed"]
            audit = read(out / "independent-audit.json")
            assert audit["passed"]
            expected = len(config["cases"]) * len(config["learning_rates"]) * 2
            assert audit["branches"] == expected
            branches += audit["branches"]
            audit_tasks += audit["tasks"]
            for record in out.glob("case*.json"):
                assert read(record)["exact_weight_reload_passed"]
            if phase == "development":
                assert read(out / "trace/trace.json")["traced_forward_and_self_patch_passed"]
        execution = read(folder / "execution.json")
        assert all(row["returncode"] == 0 for row in execution["statuses"])
        summary = read(folder / "summary.json")
        assert summary["branches"] == branches
        mechanism.append(
            {
                "phase": phase,
                "states": len(config["matrix"]),
                "branches": branches,
                "updates": summary["updates"],
                "independently_reloaded_tasks": audit_tasks,
                "trace_cells": summary["trace_cells"],
                "branch_seconds": summary["branch_seconds"],
                "parallel_execution_seconds": execution["seconds"],
            }
        )
    assert read(ARTIFACTS / "calibration/decision.json")["passed"]
    assert read(ARTIFACTS / "mechanism/calibration/decision.json")["lr"] == 0.0001
    assert mechanism[-1]["branches"] == 576 and mechanism[-1]["trace_cells"] == 576
    assert mechanism[-1]["independently_reloaded_tasks"] == 5760
    analysis_sources = [
        "scripts/audit_grokking_dynamics.py",
        "scripts/summarize_grokking_dynamics.py",
        "scripts/verify_grokking_dynamics_completion.py",
        "scripts/audit_grokking_balanced_edits.py",
    ]
    for path in analysis_sources:
        target = ARTIFACTS / "analysis-source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    analysis_hashes = {path: file_hash(path) for path in analysis_sources}
    required = [
        "assessment.json",
        "training-aggregate.csv",
        "edit-aggregate.csv",
        "edit-matched-panels.csv",
        "trace-aggregate.csv",
        "donor-role-audit.json",
        "development/report/learning.png",
        "development/report/learning.pdf",
        "development/report/mlp-norm.png",
        "answer-nll.png",
        "edit-propagation.png",
    ]
    artifacts = {path: file_hash(ARTIFACTS / path) for path in required}
    balanced_folder = ARTIFACTS / "balanced-editor"
    balanced_summary = read(balanced_folder / "calibration/summary.json")
    assert balanced_summary["branches"] == 16
    assert read(balanced_folder / "audit.json")["passed"]
    balanced_decision = read(balanced_folder / "calibration/decision.json")
    assert not balanced_decision["passed"]
    assert not (balanced_folder / "development/config.json").exists()
    assert "48 passed" in (ARTIFACTS / "quality-tests.log").read_text()
    record = {
        "status": "complete",
        "created_utc": utc(),
        "phases": phases,
        "mechanism": mechanism,
        "independent_training_audit": {k: v for k, v in independent.items() if k != "audits"},
        "source_snapshot_checks": source_checks,
        "analysis_sources": analysis_hashes,
        "artifact_sha256": artifacts,
        "balanced_editor_calibration": {
            "branches": 16,
            "updates": 3200,
            "operation_gate_passed": False,
            "independent_audit": read(balanced_folder / "audit.json"),
            "development_started": False,
        },
        "total_optimization_updates": sum(p["updates"] for p in phases + mechanism) + 3200,
        "independent_development_worlds": 1,
        "paired_initializations": 2,
        "registered_failures": 0,
        "all_registered_conditions_retained": True,
        "scientific_tests": {"passed": 48, "log": "quality-tests.log"},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "tf32": True,
        },
        "notes": [
            "The calibration runner's long string was wrapped after calibration, before the "
            "development freeze; historical source snapshots and operations are preserved.",
            "Mechanism scheduling waits for parent checkpoints before acquiring a GPU; this was "
            "added after editor calibration and before the development mechanism freeze.",
            "Analysis and donor-role audit are posthoc descriptive; "
            "budgets and selection are unchanged.",
            "Counterfactual strict-recipient donors change successor composition roles; "
            "no original strict-chain repair or systematic transfer is claimed.",
            "Matched edit panels retain changing and fixed parent-correct subsets separately. "
            "Early second-hop edit retention fails; "
            "propagation alone cannot establish a mechanism.",
            "No formal confirmation or outcome-dependent training extension was executed.",
        ],
    }
    write_json(ARTIFACTS / "completion-manifest.json", record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "total_updates": record["total_optimization_updates"],
                "training_runs": 18,
                "main_checkpoints": 168,
                "main_edit_branches": 576,
            }
        )
    )


if __name__ == "__main__":
    main()
