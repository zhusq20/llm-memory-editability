"""Read-only endpoint collector; never launches or restarts a model process."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ARMS = (
    "pd_lora",
    "sft_lora",
    "pd_full",
    "sft_full",
    "pd_down",
    "sft_down",
    "pd_down_lora",
    "sft_down_lora",
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def local_state(batch):
    result = {}
    for arm in ARMS:
        run = batch / "runs" / arm
        complete = read(run / "complete.json", {})
        audit = read(run / "audit/audit.json", {})
        evaluate = read(run / "paper-eval/evaluate.json", {})
        progress = complete or read(run / "progress.json", {})
        result[arm] = {
            "exposure": progress.get("exposure", 0),
            "train_complete": complete.get("exposure") == 1489200,
            "audit_passed": audit.get("passed") is True,
            "evaluation_complete": evaluate.get("complete") is True,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            arms = local_state(args.batch)
            ready = all(
                r["train_complete"] and r["audit_passed"] and r["evaluation_complete"]
                for r in arms.values()
            )
            controller = read(args.batch / "controller-state.json", {})
            exits = {}
            if ready:
                for arm in ARMS:
                    job = controller.get("jobs", {}).get(arm, {})
                    if job.get("state") != "complete" or not job.get("container"):
                        ready = False
                        break
                    result = subprocess.run(
                        [
                            "docker",
                            "--context",
                            "lm-memory",
                            "inspect",
                            "--format",
                            "{{json .State}}",
                            job["container"],
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode:
                        ready = False
                        break
                    state = json.loads(result.stdout)
                    exits[arm] = {"status": state["Status"], "exit_code": state["ExitCode"]}
                    if state["Status"] != "exited" or state["ExitCode"] != 0:
                        ready = False
                        break
            failed = [
                a
                for a in ARMS
                if controller.get("jobs", {}).get(a, {}).get("state")
                in {"failed", "needs_attention"}
            ]
            write(
                args.output / "collector-state.json",
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "state": "needs_attention"
                    if failed
                    else "collecting"
                    if ready
                    else "waiting_for_endpoints",
                    "arms": arms,
                    "failed_arms": failed,
                    "controller_heartbeat": controller.get("scheduler_heartbeat_utc"),
                    "training_processes_modified": False,
                    "container_exits": exits,
                },
            )
            if failed:
                return 2
            if ready:
                for attempt in range(1, 4):
                    commands = [
                        [
                            sys.executable,
                            str(args.scripts / "verify_knowledge_factorial_cloud_20261025.py"),
                            "--batch",
                            str(args.batch),
                            "--credential",
                            str(args.credential),
                            "--output",
                            str(args.output / "cloud-snapshot.json"),
                        ],
                        [
                            sys.executable,
                            str(args.scripts / "analyze_knowledge_factorial_20261025.py"),
                            "--batch",
                            str(args.batch),
                            "--output",
                            str(args.output),
                            "--cloud-snapshot",
                            str(args.output / "cloud-snapshot.json"),
                        ],
                    ]
                    with (args.output / f"collector-attempt-{attempt}.log").open("a") as log:
                        success = all(
                            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                            == 0
                            for command in commands
                        )
                    if success and read(args.output / "summary.json", {}).get("complete"):
                        write(
                            args.output / "collector-state.json",
                            {
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "state": "complete",
                                "arms": arms,
                                "local_and_cloud_verified": True,
                                "container_exits": exits,
                                "training_processes_modified": False,
                            },
                        )
                        return 0
                    time.sleep(args.poll_seconds)
                write(
                    args.output / "collector-state.json",
                    {
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "state": "needs_attention",
                        "reason": "Local endpoints complete; cloud or unified analysis "
                        "still unverified after three read-only attempts",
                        "arms": arms,
                        "training_processes_modified": False,
                    },
                )
                return 2
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
