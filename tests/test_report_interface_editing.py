"""Denominators, world weighting and outcome-blind development selection."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/report_interface_editing.py"
SPEC = importlib.util.spec_from_file_location("report_interface_editing", PATH)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def token_predictions(rows, success):
    generated = np.column_stack([rows[:, -1], np.full(len(rows), 5), np.ones(len(rows))]).astype(
        np.int64
    )
    generated[np.logical_not(success), 0] = 999
    return generated


def raw_fixture(e_success=True):
    tasks = {
        "E_new": np.array([[21, 13, 36]]),
        "E_old": np.array([[21, 13, 35]]),
        "U_atomic": np.array([[23, 13, 33], [24, 13, 34], [25, 13, 35]]),
        "R_atomic": np.array([[26, 13, 36]]),
        "necessary_atomic": np.array([[36, 17, 81], [36, 18, 82]]),
        "D_first_strict": np.array(
            [[21, 13, 36, 17, 81], [21, 13, 36, 18, 82], [21, 13, 36, 19, 83]]
        ),
    }
    old_d = tasks["D_first_strict"].copy()
    old_d[:, 2] = 35
    old_d[:, -1] += 10
    parent, current = {}, {}
    for name, rows in tasks.items():
        parent_correct = np.ones(len(rows), dtype=bool)
        current_correct = np.ones(len(rows), dtype=bool)
        if name == "U_atomic":
            parent_correct = np.array([True, True, False])
            current_correct = np.array([True, False, True])
        elif name == "necessary_atomic":
            parent_correct = np.array([True, False])
        elif name == "E_new":
            parent_correct[:] = False
            current_correct[:] = e_success
        elif name == "E_old":
            current_correct[:] = not e_success
        elif name.startswith("D_"):
            parent_correct = np.array([True, True, False])
            current_correct = np.array([True, False, True])
        parent_generated = token_predictions(
            old_d if name.startswith("D_") else rows, parent_correct
        )
        current_generated = token_predictions(rows, current_correct)
        parent[name + "__generated"] = parent_generated
        parent[name + "__correct"] = reporter.full_correct(parent_generated, rows)
        current[name + "__generated"] = current_generated
        current[name + "__correct"] = current_correct
        if name.startswith("D_"):
            current[name + "__old_rows"] = old_d
            current[name + "__new_rows"] = rows
            current[name + "__parent_correct"] = parent_correct
    return tasks, current, parent


def test_parent_correct_denominators_and_changed_D_truth():
    tasks, current, parent = raw_fixture()
    scored = reporter.score_case_node(tasks, current, parent)
    assert scored["E_new_success"] and not scored["E_old_success"]
    u = scored["tasks"]["U_atomic"]
    assert u["accuracy"] == 2 / 3
    assert u["retention_on_parent_correct"] == 1 / 2
    assert u["parent_correct_coverage"] == 2 / 3
    d = scored["tasks"]["D_first_strict"]
    assert d["accuracy"] == 2 / 3
    assert d["parent_correct_n"] == 2
    assert d["accuracy_on_parent_correct_E_success"] == 1 / 2
    assert d["parent_correct_E_success_coverage"] == 2 / 3
    failed_tasks, failed, original = raw_fixture(e_success=False)
    rejected = reporter.score_case_node(failed_tasks, failed, original)["tasks"]["D_first_strict"]
    assert rejected["parent_correct_E_success_n"] == 0
    assert rejected["accuracy_on_parent_correct_E_success"] is None
    assert rejected["accuracy"] == 2 / 3


def decision_records():
    records, infos = [], []
    for parent in ("aligned-parent", "ce-parent"):
        for lr in (1e-5, 3e-5, 1e-4):
            name = f"{parent}-{lr}"
            infos.append({"run": name, "valid_for_decision": True})
            for index in range(10):
                # Fixed endpoint: smallest lr fails E. Midpoint would pass,
                # but must never be substituted for the fixed endpoint.
                for node in (32, 512):
                    success = node == 32 or lr > 1e-5 or index < 8
                    records.append(
                        {
                            "parent": parent,
                            "case_id": f"case-{index}",
                            "lr": lr,
                            "branch": "edit",
                            "node": node,
                            "endpoint": 512,
                            "E_new_success": success,
                            "tasks": {
                                "U_atomic": {
                                    "retention_on_parent_correct": 0.99,
                                    "parent_correct_n": 100,
                                    "n": 102,
                                },
                                "necessary_atomic": {
                                    "retention_on_parent_correct": 1.0,
                                    "parent_correct_n": 2,
                                    "n": 3,
                                },
                                "D_first_strict": {"accuracy": 0.0},
                            },
                        }
                    )
    return records, infos


def test_selection_is_fixed_endpoint_target_equal_and_ignores_D_and_sham():
    records, infos = decision_records()
    names = [i["run"] for i in infos]
    decision = reporter.development_decision(records, infos, names, "development")
    assert decision["selected_lr"] == 3e-5
    assert decision["candidates"][0]["means_target_equal"]["E_new"] == 0.8
    altered = copy.deepcopy(records)
    for row in altered:
        row["tasks"]["D_first_strict"]["accuracy"] = float(row["lr"] == 1e-4)
    sham = copy.deepcopy(altered)
    for row in sham:
        row["branch"] = "sham"
        row["E_new_success"] = False
    other = reporter.development_decision(altered + sham, infos, names, "development")
    assert other == decision


def test_missing_audit_or_parent_denominator_cannot_select_lr():
    records, infos = decision_records()
    names = [i["run"] for i in infos]
    infos[0]["valid_for_decision"] = False
    assert (
        reporter.development_decision(records, infos, names, "development")["status"] == "pending"
    )
    infos[0]["valid_for_decision"] = True
    for row in records:
        if row["lr"] == 3e-5 and row["case_id"] == "case-0":
            row["tasks"]["necessary_atomic"].update(
                parent_correct_n=0, retention_on_parent_correct=None
            )
    assert (
        reporter.development_decision(records, infos, names, "development")["selected_lr"] == 1e-4
    )
    assert (
        reporter.development_decision(records, infos, names, "confirmation")["status"]
        == "evaluation_only"
    )


def test_sham_uses_paired_edit_success_for_matched_D_subset():
    records = []
    for branch, e_success in (("edit", True), ("sham", False)):
        tasks, current, parent = raw_fixture(e_success)
        records.append(
            {
                "parent": "p",
                "lr": 1e-4,
                "case_id": "c",
                "node": 512,
                "branch": branch,
                **reporter.score_case_node(tasks, current, parent),
            }
        )
    reporter.add_paired_edit_conditions(records)
    sham = records[1]["tasks"]["D_first_strict"]
    assert sham["parent_correct_E_success_n"] == 0
    assert sham["parent_correct_paired_edit_E_success_n"] == 2
    assert sham["accuracy_on_parent_correct_paired_edit_E_success"] == 0.5


def test_world_aggregation_never_treats_many_targets_as_many_worlds():
    records = []
    for world, initializations, target_count, accuracy in (
        (1, [10, 11], 9, 1.0),
        (2, [10], 1, 0.0),
    ):
        for initialization in initializations:
            for index in range(target_count):
                records.append(
                    {
                        "parent": f"p{world}-{initialization}",
                        "parent_arm": "aligned",
                        "model_condition": "aligned",
                        "world": world,
                        "initialization": initialization,
                        "lr": 1e-4,
                        "role": "first",
                        "case_pool": "strict",
                        "branch": "edit",
                        "node": 512,
                        "case_id": str(index),
                        "tasks": {
                            "D_first_strict": {
                                "accuracy": accuracy,
                                "n": 1,
                                "correct": int(accuracy),
                            }
                        },
                    }
                )
    parents, worlds, total = reporter.aggregate(records)
    assert len(parents) == 3 and len(worlds) == 2
    assert total[0]["independent_worlds"] == 2
    assert total[0]["rates_world_equal"]["accuracy"] == 0.5


def test_report_reads_tokens_writes_summary_and_freezes_decision(tmp_path):
    root = tmp_path / "batch"
    run = root / "runs" / "aligned-lr"
    case = run / "first_strict-00"
    case.mkdir(parents=True)
    manifest = {
        "spec": {
            "name": "aligned-lr",
            "phase": "development",
            "parent_dir": "/old/aligned",
            "lr": 1e-4,
            "nodes": [0, 512],
        },
        "parent_spec": {"arm": "aligned", "world": 1, "initialization": 2},
        "case_sha256": "test",
        "branches": 2,
        "cases": [{"case_id": "first_strict-00", "role": "first", "stratum": "strict"}],
    }
    reporter.write(root / "frozen-config.json", {"specs": [manifest["spec"]]})
    reporter.write(run / "run.json", manifest)
    reporter.write(run / "complete.json", {})
    reporter.write(run / "audit.json", {"passed": True})
    tasks, current, parent = raw_fixture()
    np.savez_compressed(case / "tasks.npz", **tasks)
    np.savez_compressed(case / "parent-predictions.npz", **parent)
    for branch in ("edit", "sham"):
        directory = case / f"lr0.0001-{branch}"
        directory.mkdir()
        reporter.write(directory / "learning.json", [{"step": 0}, {"step": 512}])
        for node in (0, 512):
            np.savez_compressed(directory / f"predictions-{node:06d}.npz", **current)
    summary, decision = reporter.report(root, "development")
    assert summary["status"] == "complete" and summary["case_node_records"] == 4
    assert decision["status"] == "no_eligible_learning_rate"  # U retention is 1/2.
    assert (root / "report/summary.json").exists()
    assert reporter.report(root, "development")[1] == decision
    saved = json.loads((root / "report/decision.json").read_text())
    saved["selected_lr"] = 1e-4
    reporter.write(root / "report/decision.json", saved)
    with pytest.raises(ValueError, match="frozen"):
        reporter.report(root, "development")
