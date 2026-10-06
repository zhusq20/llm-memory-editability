"""Run the originally planned scans after calibration, retaining every outcome."""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
import traceback
from pathlib import Path

import execute_capacity_scaling as capacity

read, write, digest, now = capacity.read, capacity.write, capacity.digest, capacity.now


def reuse_calibration(config):
    """Validate sealed, audited calibration results without writing to them."""
    continuation = config["continuation"]
    assert continuation["prerequisite_policy"] == "report_only"
    assert digest(continuation["parent_config"]) == continuation["parent_config_sha256"]
    parent = read(continuation["parent_config"])
    # Only execution/provenance fields may change in this continuation.
    for key in (
        "base_spec",
        "load_heads",
        "fixed_target_support_heads",
        "budget",
        "runtime",
        "tracking",
        "calibration_learning_rates",
        "selection_rule",
        "prerequisite_thresholds",
        "long_calibration_epochs",
        "capacity_thresholds",
        "sensitivity_thresholds",
        "batch",
        "results_root",
        "gpus",
        "predecessors",
    ):
        assert config[key] == parent[key], key
    for filename, expected in continuation["retained_files"].items():
        assert digest(filename) == expected, f"Retained calibration changed: {filename}"
    root = Path(config["results_root"])
    decision_path = root / "calibration-decision.json"
    assert str(decision_path) in continuation["retained_files"]
    decision = read(decision_path)
    paths = {}
    for stage in ("calibration", "calibration-long"):
        stage_root = root / stage
        state = read(stage_root / "controller-state.json")
        frozen = read(stage_root / "frozen-config.json")
        assert state["state"] == "complete"
        assert set(state["completed"]) == {s["name"] for s in frozen["runs"]}
        assert read(stage_root / "tracking-completion.json")["passed"] is True
        for spec in frozen["runs"]:
            out = stage_root / "runs" / spec["name"]
            assert read(out / "audit.json")["passed"] is True
            assert read(out / "run.json")["spec"] == spec
            for filename in ("run.json", "audit.json", "latest.pt", "world.npz", "exposures.npz"):
                assert str(out / filename) in continuation["retained_files"]
            paths[spec["name"]] = out
    expected = {f"calibration-lr{lr:g}" for lr in config["calibration_learning_rates"]}
    assert set(paths) == expected | {"calibration-long"}
    selected = Path(decision["selected"])
    assert selected == paths["calibration-long"]
    recipe = decision["recipe"]
    selected_spec = read(selected / "run.json")["spec"]
    assert recipe == {k: selected_spec[k] for k in ("learning_rate", "epochs")}
    assert recipe["epochs"] == config["long_calibration_epochs"]
    assert decision["passed"] == capacity.prerequisite_pass(
        read(selected / "audit.json")["metrics"], config["prerequisite_thresholds"]
    )
    return paths, selected, recipe, decision


def remaining_stages(config, recipe):
    scan = [
        capacity.make_spec(config, f"load-h{h}", heads_n=h, **recipe)
        for h in config["load_heads"][1:]
    ]
    background = [
        capacity.make_spec(
            config,
            f"background-h{h}",
            heads_n=h,
            support_heads_n=config["fixed_target_support_heads"],
            **recipe,
        )
        for h in config["load_heads"][1:]
    ]
    return [("load-scan", scan), ("fixed-target", background)]


def control_locked(config):
    root = Path(config["results_root"])
    paths, selected, recipe, decision = reuse_calibration(config)
    stages = remaining_stages(config, recipe)
    assert (
        len(paths) + sum(len(specs) for _, specs in stages)
        <= config["budget"]["maximum_training_runs"]
    )
    record = {
        "prerequisite_policy": "report_only",
        "original_calibration_decision": decision,
        "recipe": recipe,
        "reused": {name: str(path) for name, path in paths.items()},
        "remaining_stages": {name: specs for name, specs in stages},
        "authorization": config["continuation"]["authorization"],
    }
    record_path = root / "continuation-decision.json"
    if record_path.exists():
        assert read(record_path) == record
    else:
        write(record_path, record)
    for stage, specs in stages:
        write(
            root / "controller-state.json",
            {
                "state": stage,
                "updated_utc": now(),
                "completed": list(paths),
                "prerequisite_policy": "report_only",
                "calibration_passed": decision["passed"],
                "remaining_planned": [
                    s["name"] for _, group in stages for s in group if s["name"] not in paths
                ],
            },
        )
        paths.update(capacity.run_stage(config, stage, specs))
    # Recheck that execution left the original calibrations untouched.
    reuse_calibration(config)
    summary = capacity.report(config, list(paths.values()), selected, recipe)
    summary["calibration_prerequisite"] = decision
    summary["prerequisite_policy"] = "report_only"
    summary.setdefault("limitations", []).append(
        "Low-load held-out composition was not established. Report the full load response, "
        "including improvements, failures and nonmonotonicity; do not attribute a gap to "
        "capacity or interference from the failed calibration alone."
    )
    write(root / "summary.json", summary)
    write(
        root / "controller-state.json",
        {
            "state": "complete",
            "updated_utc": now(),
            "completed": list(paths),
            "training_runs": len(paths),
            "reused_training_runs": 3,
            "new_training_runs": 6,
            "calibration_passed": decision["passed"],
            "prerequisite_policy": "report_only",
            "next": "Review all load and fixed-target outcomes before any new matrix",
        },
    )


def controller(path):
    config = read(path)
    assert digest(path) == read(path.parent / "execution-lock.json")["config_sha256"]
    for filename, expected in config["source_files"].items():
        assert digest(filename) == expected, filename
    root = Path(config["results_root"])
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            control_locked(config)
        except BaseException:
            write(
                root / "controller-state.json",
                {
                    "state": "controller_error",
                    "updated_utc": now(),
                    "error": traceback.format_exc(),
                    "prerequisite_policy": "report_only",
                },
            )
            raise


def launch(path):
    config = read(path)
    # Fail before detaching if the archived inputs cannot be reused.
    reuse_calibration(config)
    root = Path(config["results_root"])
    script = Path(config["source_root"]) / "scripts/continue_capacity_scaling.py"
    command = [sys.executable, "-u", str(script), "controller", "--config", str(path)]
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            command,
            env=capacity.scheduler.host_env(config["source_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(root / "controller-process.json", {"pid": process.pid, "command": command, "utc": now()})
    print({"pid": process.pid, "results": str(root)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "controller"])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    {"launch": launch, "controller": controller}[args.action](args.config.resolve())
