#!/usr/bin/env python3
"""Run the frozen small fact-usage matrix, retaining logs and resumable state."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--per-gpu", type=int, default=1)
    args = parser.parse_args()
    if args.per_gpu < 1:
        raise ValueError("per-gpu must be positive")
    config = json.loads((ROOT / args.config).read_text())
    artifact = ROOT / "docs/development-artifacts/grok-usage-v1"
    artifact.mkdir(parents=True, exist_ok=True)
    handle = (artifact / (config["experiment"] + ".lock")).open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, guard, records = queue.Queue(), threading.Lock(), {}
    for run_id in config["runs"]:
        pending.put(run_id)
    status_path = artifact / (config["experiment"] + "-queue.json")

    def record(run_id, **fields):
        with guard:
            records.setdefault(run_id, {}).update(fields)
            write_json(
                status_path,
                {
                    "config": args.config,
                    "pid": os.getpid(),
                    "updated_utc": utc(),
                    "runs": records,
                    "expected_runs": list(config["runs"]),
                },
            )

    def worker(gpu):
        while True:
            try:
                run_id = pending.get_nowait()
            except queue.Empty:
                return
            spec = {**config["base"], **config["runs"][run_id]}
            out = ROOT / config["output_root"] / spec["phase"] / run_id
            if (out / "complete.json").exists():
                saved = json.loads((out / "complete.json").read_text())
                if saved["spec"] != spec:
                    raise ValueError("Completed specification changed")
                record(run_id, state="already_complete", gpu=gpu)
                continue
            attempt = 1
            while True:
                logfile = artifact / f"{run_id}-attempt-{attempt}.log"
                try:
                    log = logfile.open("x")
                    break
                except FileExistsError:
                    attempt += 1
            resume = (out / "latest.pt").exists()
            command = [sys.executable, "scripts/run_grok_usage.py", run_id, "--config", args.config]
            if resume:
                command.append("--resume")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1"}
            record(
                run_id,
                state="running",
                gpu=gpu,
                started_utc=utc(),
                log=str(logfile.relative_to(ROOT)),
                resume=resume,
            )
            with log:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                record(run_id, child_pid=process.pid)
                code = process.wait()
            record(
                run_id,
                state="complete" if code == 0 else "failed",
                exit_code=code,
                finished_utc=utc(),
            )
            print(json.dumps({"run": run_id, "exit_code": code, "utc": utc()}), flush=True)

    slots = args.gpus * args.per_gpu
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        for _result in pool.map(worker, slots):
            pass
    if any(row["state"] == "failed" for row in records.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
