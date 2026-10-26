#!/usr/bin/env python3
"""One arm: train, independent reload process, then read-only cloud verification."""

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def write(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def reload_matches(expected, actual):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(reload_matches(value, actual[key]) for key, value in expected.items())
        )
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-6, abs_tol=1e-7
        )
    return expected == actual


def cloud(config, base, arm):
    import wandb
    from run_loop_native_handoff import learning_metrics

    out = base / "runs" / arm
    complete = json.loads((out / "complete.json").read_text())
    grid = json.loads((out / "grid.json").read_text())
    tracking = json.loads((out / "tracking.json").read_text())
    identity = f"{config['tracking']['entity']}/{config['tracking']['project']}/{tracking['id']}"
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb-key").read_text().strip()
    remote = wandb.Api(timeout=90).run(identity)
    expected = learning_metrics(complete["endpoint"])
    mismatch = []
    for key, value in expected.items():
        actual = remote.summary.get(key)
        if not isinstance(actual, (int, float)) or not math.isclose(
            actual, value, rel_tol=1e-6, abs_tol=1e-7
        ):
            mismatch.append(key)
    checks = {
        "training_step": complete["step"] == 256000,
        "reload_step": grid["step"] == 256000,
        "reload_checkpoint": grid["checkpoint_sha256"] == complete["final_checkpoint_sha256"],
        "reload_endpoint_metrics": all(
            reload_matches(metrics, grid["weights"][kind]["answer_and_eos"][rounds])
            for kind, by_round in complete["endpoint"]["metrics"].items()
            for rounds, metrics in by_round.items()
        ),
        "cloud_finished": remote.state == "finished",
        "cloud_config_exact": remote.config.get("experiment") == config,
        "cloud_endpoint_metrics": not mismatch,
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics_checked": len(expected),
        "mismatched_keys": mismatch,
        "url": tracking["url"],
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    write(out / "cloud-verification.json", report)
    if not report["passed"]:
        raise RuntimeError(
            "Independent endpoint/cloud verification failed; original evidence retained"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    allowed = {"supervision", "truncate", "supervision-truncate", "floor4"}
    if args.arm not in allowed or config["batch"] != "loop-recipe-ablation-20261025":
        raise ValueError("Only the four new ablation arms are allowed")
    directory = args.base / "pipeline" / args.arm
    directory.mkdir(parents=True, exist_ok=False)
    state = {
        "arm": args.arm,
        "state": "running",
        "stages": {},
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        for stage in ["train", "grid"]:
            state["stage"] = stage
            write(directory / "state.json", state)
            command = [
                sys.executable,
                "-u",
                "scripts/run_loop_recipe_ablation.py",
                "--config",
                str(args.config),
                "--base",
                str(args.base),
                "--arm",
                args.arm,
                "--stage",
                stage,
            ]
            with (directory / f"{stage}.log").open("w") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["stages"][stage] = {
                    "pid": proc.pid,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                write(directory / "state.json", state)
                code = proc.wait()
            state["stages"][stage].update(
                exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat()
            )
            write(directory / "state.json", state)
            if code:
                raise RuntimeError(f"{stage} exited {code}; no automatic retry")
        state["stage"] = "cloud"
        write(directory / "state.json", state)
        cloud(config, args.base, args.arm)
        state["state"] = "complete"
    except BaseException:
        state.update(state="needs_attention", error=traceback.format_exc())
        write(directory / "state.json", state)
        raise
    state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write(directory / "state.json", state)


if __name__ == "__main__":
    main()
