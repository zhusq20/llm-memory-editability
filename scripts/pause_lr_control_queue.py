#!/usr/bin/env python3
"""One-time pause of four unstarted arms; keep two existing runs monitored."""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from continue_lr_control_handoff import ACTIVE, WAITING


def main():
    batch = Path("/ossfs/workspace/llm-memory-editability/loop-discoloop-lr-control-v1")
    handoff = batch / "handoff"
    handoff.mkdir(exist_ok=True)
    receipt = handoff / "paused-queue.json"
    if receipt.exists():
        raise FileExistsError("Pause already recorded; do not repeat takeover")
    process = json.loads((batch / "controller-process.json").read_text())
    pid = process["pid"]
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    if not any(value.endswith(b"/resume_lr_control_gpu7.py") for value in argv):
        raise ValueError("Controller identity differs")
    os.kill(pid, signal.SIGSTOP)
    committed = False
    try:
        for _ in range(100):
            if "State:\tT" in Path(f"/proc/{pid}/status").read_text():
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Controller did not stop")
        state = json.loads((batch / "controller-state.json").read_text())
        running = subprocess.run(
            ["docker", "--context", "lm-memory", "ps", "--format", "{{.Names}}"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.splitlines()
        for job in WAITING:
            if state["jobs"][job]["state"] != "waiting_for_idle_resources":
                raise ValueError(f"{job} is no longer waiting; inspect before transfer")
            if any(name.startswith(f"loop-discoloop-lr-control-v1-{job}-") for name in running):
                raise ValueError(f"{job} has a running container")
            if (batch / "execution" / job / "train").exists():
                raise ValueError(f"{job} already has a scientific claim")
        utc = datetime.now(timezone.utc).isoformat()
        (handoff / "controller-state-before-pause.json").write_text(json.dumps(state, indent=2))
        receipt.write_text(
            json.dumps(
                {
                    "utc": utc,
                    "paused_jobs": WAITING,
                    "continuing_jobs": ACTIVE,
                    "old_controller_pid": pid,
                    "authorization": "User requested pausing the four queued arms for another GPU",
                    "containers_stopped": [],
                    "scientific_source_changed": False,
                    "original_controller_action": (
                        "terminate scheduler only; running Docker jobs persist"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        committed = True
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        for _ in range(100):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Old controller remains; do not start replacement monitors")
        for job in WAITING:
            state["jobs"][job].update(state="paused_for_handoff", paused_utc=utc)
        state.update(
            state="running_with_four_paused",
            utc=utc,
            handoff_receipt=str(receipt),
            monitoring="handoff/*-status.json",
        )
        (batch / "controller-state.json").write_text(json.dumps(state, indent=2) + "\n")
        (batch / "science-schedule.json").write_text(json.dumps(state["jobs"], indent=2) + "\n")
        runner = Path(__file__).with_name("continue_lr_control_handoff.py").resolve()
        for job in ACTIVE:
            old = state["jobs"][job]
            command = [
                sys.executable,
                str(runner),
                "--batch",
                str(batch),
                "--job",
                job,
                "--gpu",
                str(old["gpu"]),
                "--cpus",
                ",".join(map(str, old["cpu_set"])),
                "--wandb-token-file",
                str(Path.home() / ".cache/llm-memory-ttcd/wandb-token"),
                "--execute",
            ]
            with (handoff / f"{job}-monitor.log").open("x") as log:
                child = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            print(json.dumps({"job": job, "monitor_pid": child.pid}), flush=True)
        print(str(receipt), flush=True)
    finally:
        if not committed:
            os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
