#!/usr/bin/env python3
"""Execute frozen loop matrices on selected GPUs, retaining every attempt log."""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def load_jobs(config_paths, selected=None, root=None):
    """Read registered jobs in config order without constructing any data worlds."""
    root = ROOT if root is None else Path(root)
    jobs, by_id = [], {}
    for config_path in config_paths:
        cfg = json.loads((root / config_path).read_text())
        for run_id, overrides in cfg["runs"].items():
            if run_id in by_id:
                raise ValueError(f"Duplicate run across configurations: {run_id}")
            job = {
                "config": str(config_path),
                "run": run_id,
                "experiment": cfg["experiment"],
                "output_root": cfg["output_root"],
                "source_lock": cfg["source_lock"],
                "spec": {**cfg["base"], **overrides},
            }
            jobs.append(job)
            by_id[run_id] = job
    if selected is not None:
        if len(set(selected)) != len(selected) or any(key not in by_id for key in selected):
            raise ValueError("Duplicate or unknown selected run")
        jobs = [by_id[key] for key in selected]
    return jobs


def open_attempt_log(directory, run_id, resume=False):
    """Create a new log atomically; repeated restarts cannot replace old evidence."""
    stem = run_id + ("-resume" if resume else "")
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"-attempt-{attempt}"
        path = Path(directory) / f"{stem}{suffix}.log"
        try:
            return path, path.open("x", encoding="utf-8")
        except FileExistsError:
            attempt += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--additional-configs", nargs="+", default=[])
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--per-gpu", type=int, default=1)
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args()
    if args.per_gpu < 1 or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("GPU IDs must be nonnegative and per-gpu must be positive")
    config_paths = [args.config, *args.additional_configs]
    jobs = load_jobs(config_paths, args.runs)
    if not jobs:
        raise ValueError("No registered jobs selected")
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    slots = args.gpus * args.per_gpu
    artifact = ROOT / "docs/development-artifacts/grok-loop-v1"
    artifact.mkdir(parents=True, exist_ok=True)
    queue_name = jobs[0]["experiment"] + ("-multi" if len(config_paths) > 1 else "")

    def worker(slot, gpu):
        records = []
        status = artifact / f"{queue_name}-queue-{slot}.json"

        def record_status():
            write_json(status, {"config": args.config, "configs": config_paths, "runs": records})

        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return True
            key, spec = job["run"], job["spec"]
            directory = ROOT / job["output_root"] / spec["phase"] / key
            complete = directory / "complete.json"
            record = {
                "run": key,
                "config": job["config"],
                "source_lock": job["source_lock"],
                "gpu": gpu,
            }
            records.append(record)
            if complete.exists():
                if json.loads(complete.read_text())["spec"] != spec:
                    record.update(state="failed", error="Completed specification changed")
                    record_status()
                    return False
                record["state"] = "already_complete"
                record_status()
                continue
            resume = (directory / "latest.pt").exists()
            command = [sys.executable, "scripts/run_grok_loop.py", key, "--config", job["config"]]
            if resume:
                command.append("--resume")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1"}
            logfile, log = open_attempt_log(artifact, key, resume)
            record.update(
                state="running",
                started_utc=utc(),
                resume=resume,
                log=str(logfile.relative_to(ROOT)),
            )
            record_status()
            with log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            record.update(
                state="complete" if result.returncode == 0 else "failed",
                finished_utc=utc(),
                exit_code=result.returncode,
            )
            record_status()
            print(json.dumps(record), flush=True)
            if result.returncode:
                return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(slots)]
        results = [future.result() for future in futures]
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
