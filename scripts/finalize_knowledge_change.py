"""Finish analysis and independently verify W&B when the batch becomes terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    write_json(args.root / "finalizer-process.json", dict(pid=os.getpid(), utc=utc()))
    terminal = {"complete", "finished_with_failures", "development_prerequisite_not_met"}
    while True:
        state = read(args.root / "controller-state.json")
        if state["state"] in terminal:
            break
        time.sleep(15)
    subprocess.run([sys.executable, str(args.analyzer), "--root", str(args.root)], check=True)
    source_valid = all(
        hashlib.sha256((Path(config["source_root"]) / path).read_bytes()).hexdigest() == digest
        for path, digest in config["source"].items()
    )
    assert source_valid, "Frozen sources changed"
    local = []
    for name in state["completed"]:
        directory = args.root / "runs" / name
        assert read(directory / "audit.json")["passed"]
        local.append(dict(name=name, audited=True))
    tracking_ready = False
    for _ in range(120):
        path = args.root / "tracking-completion.json"
        if path.exists() and read(path)["passed"]:
            tracking_ready = True
            break
        time.sleep(15)
    if not tracking_ready:
        write_json(
            args.root / "finalizer-failure.json", dict(reason="W&B tracking not drained", utc=utc())
        )
        raise RuntimeError("W&B tracking not drained")
    import wandb

    api = wandb.Api(timeout=30)
    remote = []
    for name in state["completed"]:
        directory = args.root / "runs" / name
        tracking = read(args.root / "tracking-wandb" / name / "state.json")
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['run_id']}")
        last = read(directory / "learning.json")[-1]["step"]
        assert run.name == name and run.group == config["batch"]
        assert run.state == "finished", (name, run.state)
        assert run.summary.get("_step") == last, (name, run.summary.get("_step"), last)
        assert run.summary.get("independently_reloaded") is True
        assert tracking["last_logged_step"] == last
        remote.append(dict(name=name, run_id=run.id, last_step=last, url=run.url))
    write_json(
        args.root / "cloud-final-audit.json",
        dict(passed=True, runs=remote, registered_runs=len(remote), utc=utc()),
    )
    report = read(args.root / "report/summary.json")
    write_json(
        args.root / "completion-manifest.json",
        dict(
            executed_jobs_audited=True,
            full_confirmation_complete=state["state"] == "complete" and not report["missing"],
            state=state["state"],
            completed=len(state["completed"]),
            failed=state["failed"],
            missing_planned_jobs=report["missing"],
            reader_trainings=len(report["endpoints"]),
            edit_sham_branches=len(report["editing_cases"]),
            sources_unchanged=source_valid,
            cloud_runs_verified=len(remote),
            allocated_gpu_hours=state.get("allocated_gpu_hours"),
            utc=utc(),
        ),
    )
    print(json.dumps(dict(local=len(local), cloud=len(remote), state=state["state"])), flush=True)


if __name__ == "__main__":
    main()
