#!/usr/bin/env python3
"""Coordinate all authorized v2 arms, without outcome-dependent stopping."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/hebbian-future-v2"


def status(state, **kwargs):
    value = {"time": time.time(), "pid": os.getpid(), "state": state, **kwargs}
    temp = ART / "execution-status.tmp"
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(ART / "execution-status.json")
    print(json.dumps(value), flush=True)


def main():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        OPENBLAS_NUM_THREADS="4",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        MPLCONFIGDIR=str(ART / "matplotlib-cache"),
    )
    jobs = [("real", ["real", "--device", "cuda:2"])]
    jobs += [
        (f"chains-{i}", ["chains", "--device", f"cuda:{i + 3}", "--shard", str(i), "--shards", "7"])
        for i in range(7)
    ]
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    running = []
    for name, args in jobs:
        with (ART / "logs" / f"{name}.log").open("a") as log:
            proc = subprocess.Popen(
                [sys.executable, "scripts/run_hebbian_followup.py", *args],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        running.append((name, proc))
    analysis = None
    while True:
        codes = {name: proc.poll() for name, proc in running}
        if codes["real"] == 0 and analysis is None:
            with (ART / "logs/real-analysis.log").open("a") as log:
                analysis = subprocess.Popen(
                    [sys.executable, "scripts/report_hebbian_followup.py", "real"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        status("running", workers=[{"name": n, "pid": p.pid, "exit": codes[n]} for n, p in running])
        if all(v is not None for v in codes.values()):
            break
        time.sleep(15)
    if any(v != 0 for v in codes.values()) or analysis is None or analysis.wait() != 0:
        status("technical_failure", codes=codes)
        return 1
    for stage in ("chains", "plot", "audit"):
        subprocess.run(
            [sys.executable, "scripts/report_hebbian_followup.py", stage],
            cwd=ROOT,
            env=env,
            check=True,
        )
    status("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
