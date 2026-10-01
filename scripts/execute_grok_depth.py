#!/usr/bin/env python3
"""Run an explicitly registered matrix on named GPUs, preserving every attempt."""

import argparse
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=["primary_runs", "conditional_wide_runs"])
    parser.add_argument("--gpus", nargs="+", default=["2", "5"])
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/grok-depth-v1.json").read_text())
    if args.matrix == "conditional_wide_runs":
        decision = ROOT / "docs/development-artifacts/grok-depth-v1/wide-control-decision.json"
        assert json.loads(decision.read_text())["execute"]
    jobs = queue.Queue()
    for run_id in cfg[args.matrix]:
        jobs.put(run_id)
    logdir = ROOT / "results/grok-depth-v1/launches"
    logdir.mkdir(parents=True, exist_ok=True)

    def worker(gpu):
        records = []
        while True:
            try:
                name = jobs.get_nowait()
            except queue.Empty:
                return records
            out = ROOT / "results/grok-depth-v1/confirmation" / name
            if (out / "complete.json").exists():
                records.append({"run_id": name, "state": "already_complete"})
                continue
            cmd = [sys.executable, "scripts/run_grok_depth.py", name]
            if (out / "latest.pt").exists():
                cmd.append("--resume")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            log = logdir / f"{name}-{stamp}.log"
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(ROOT / "src")}
            with log.open("w") as handle:
                p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record = {"run_id": name, "gpu": gpu, "exit_code": p.returncode, "log": str(log)}
            records.append(record)
            print(json.dumps(record), flush=True)
            if p.returncode:
                return records

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        results = list(pool.map(worker, args.gpus))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    (logdir / f"matrix-{args.matrix}-{stamp}.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r.get("exit_code", 0) != 0 for records in results for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
