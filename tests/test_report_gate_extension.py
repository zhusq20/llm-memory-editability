"""Reporting must preserve missing runs and reject invalid paired evidence."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
report = importlib.import_module("report_gate_extension")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def run(folder, arm, value, world="world", panel="ood_3", score="answer_accuracy", repeats=4):
    nodes = [
        {"step": 0, "metrics": {panel: {score: 0.1}}, "examples": 0},
        {"step": 10, "metrics": {panel: {score: value}}, "examples": 4},
    ]
    complete = {
        "state": "complete",
        "arm": arm,
        "step": 10,
        "budget": 10,
        "endpoint": nodes[-1],
        "world_sha256": world,
        "model_sha256": arm,
        "config_sha256": "config",
    }
    grid = {
        key: complete[key]
        for key in ("arm", "step", "world_sha256", "model_sha256", "config_sha256")
    }
    grid.update(
        training_repeats=repeats,
        metrics={str(repeats): nodes[-1]["metrics"], "1": {panel: {score: 1.0}}},
    )
    write(folder / "learning.json", nodes)
    write(folder / "complete.json", complete)
    write(folder / "grid.json", grid)
    np.savez_compressed(folder / "exposure.npz", counts=np.array([2, 2]))


def plan(cases=("confirmation-1",), family="confirmation"):
    return {
        "batch": "test",
        "jobs": [
            {"case": case, "family": family, "arm": arm, "steps": 10}
            for case in cases
            for arm in report.ARMS
        ],
    }


def populate(output, cases=("confirmation-1",), family="confirmation"):
    for case in cases:
        for arm, value in zip(report.ARMS, (0.5, 0.6, 0.8), strict=True):
            run(
                output / "cases" / case / "runs" / arm,
                arm,
                value,
                world=case,
                panel="test_oo" if family == "wang" else "ood_3",
                score="accuracy" if family == "wang" else "answer_accuracy",
                repeats=2 if family == "wang" else 4,
            )


def test_missing_runs_are_pending_not_zero_or_failure(tmp_path):
    result = report.collect(plan(), tmp_path)
    assert result["state"] == "pending" and result["passed"]
    assert (
        result["cases"]["confirmation-1"]["comparisons"]["token_minus_fixed"]["difference_pp"]
        is None
    )
    assert result["families"]["confirmation"]["token_minus_fixed"]["mean_difference_pp"] is None


def test_three_world_pairs_mean_sd_and_fixed_horizon_ignore_better_r1(tmp_path):
    cases = tuple(f"confirmation-{i}" for i in (1, 2, 3))
    populate(tmp_path, cases)
    result = report.collect(plan(cases), tmp_path)
    assert result["state"] == "complete" and result["passed"]
    comparison = result["families"]["confirmation"]["token_minus_fixed"]
    assert comparison["planned_worlds"] == comparison["completed_worlds"] == 3
    assert comparison["mean_difference_pp"] == pytest.approx(20)
    assert comparison["sample_sd_pp"] == 0
    arm = result["cases"][cases[0]]["arms"]["token-update"]
    assert arm["primary_score"] == 0.8
    assert arm["threshold_first_observed_intervals"]["75"] == {
        "lower_step": 0,
        "upper_step": 10,
        "left_censored": False,
    }
    assert arm["threshold_first_observed_intervals"]["90"] is None


@pytest.mark.parametrize("damage", ["world", "exposure", "grid_model", "grid_score", "curve"])
def test_corrupt_or_unpaired_evidence_explicitly_fails(tmp_path, damage):
    populate(tmp_path)
    folder = tmp_path / "cases" / "confirmation-1" / "runs" / "token-update"
    if damage == "exposure":
        np.savez_compressed(folder / "exposure.npz", counts=np.array([1, 3]))
    elif damage == "world":
        for filename in ("complete.json", "grid.json"):
            value = report.read(folder / filename)
            value["world_sha256"] = "wrong-world"
            write(folder / filename, value)
    elif damage.startswith("grid"):
        value = report.read(folder / "grid.json")
        if damage == "grid_model":
            value["model_sha256"] = "wrong-model"
        else:
            value["metrics"]["4"]["ood_3"]["answer_accuracy"] = 0.9
        write(folder / "grid.json", value)
    else:
        value = report.read(folder / "learning.json")
        value[-1]["metrics"]["ood_3"]["answer_accuracy"] = 0.99
        write(folder / "learning.json", value)
    result = report.collect(plan(), tmp_path)
    assert result["state"] == "failed" and result["passed"] is False
    assert (
        result["cases"]["confirmation-1"]["comparisons"]["token_minus_fixed"]["state"] == "failed"
    )


def test_remote_p_continuation_pending_and_threshold_left_censoring(tmp_path):
    configured = plan(("continuation",), "continuation")
    configured["jobs"] = [job for job in configured["jobs"] if job["arm"] != "paper"]
    configured["reused_baseline"] = {"root": str(tmp_path / "remote"), "steps": 10}
    for arm, value in (("fixed-update", 0.6), ("token-update", 0.8)):
        run(tmp_path / "cases" / "continuation" / "runs" / arm, arm, value)
    result = report.collect(configured, tmp_path)
    case = result["cases"]["continuation"]
    assert case["comparisons"]["token_minus_fixed"]["state"] == "complete"
    assert case["comparisons"]["token_minus_paper"]["state"] == "pending"
    assert result["passed"] and result["state"] == "pending"
    assert report.thresholds([{"step": 300000, "primary_score": 0.8}])["75"] == {
        "lower_step": None,
        "upper_step": 300000,
        "left_censored": True,
    }


def test_wang_uses_answer_plus_eos_and_r2(tmp_path):
    populate(tmp_path, ("wang",), "wang")
    result = report.collect(plan(("wang",), "wang"), tmp_path)
    assert result["state"] == "complete"
    assert result["cases"]["wang"]["primary"] == {
        "panel": "test_oo",
        "metric": "accuracy",
        "inference_repeats": 2,
    }


def test_repeated_graphs_cannot_count_as_independent_worlds(tmp_path):
    cases = ("confirmation-1", "confirmation-2")
    for case in cases:
        for arm, value in zip(report.ARMS, (0.5, 0.6, 0.8), strict=True):
            run(tmp_path / "cases" / case / "runs" / arm, arm, value, world="same-world")
    result = report.collect(plan(cases), tmp_path)
    assert result["passed"] is False and result["state"] == "failed"
    family = result["families"]["confirmation"]["token_minus_fixed"]
    assert family["mean_difference_pp"] is None
    assert "independent worlds" in family["failure"]


@pytest.mark.parametrize("source", ["failure_file", "exit", "controller", "score_range"])
def test_failed_job_or_out_of_range_score_cannot_appear_pending(tmp_path, source):
    case = tmp_path / "cases" / "confirmation-1"
    if source == "failure_file":
        write(case / "failures" / "token-update-train.json", {"error": "boom"})
    elif source == "exit":
        write(case / "execution" / "token-update" / "grid" / "exit.json", {"code": 1})
    elif source == "controller":
        write(
            tmp_path / "controller-state.json",
            {
                "jobs": {
                    "confirmation-1--token-update": {"state": "needs_attention", "error": "boom"}
                }
            },
        )
    else:
        write(
            case / "runs" / "token-update" / "learning.json",
            [{"step": 0, "metrics": {"ood_3": {"answer_accuracy": 1.5}}}],
        )
    result = report.collect(plan(), tmp_path)
    assert result["passed"] is False and result["state"] == "failed"
    assert result["cases"]["confirmation-1"]["arms"]["token-update"]["state"] == "failed"
