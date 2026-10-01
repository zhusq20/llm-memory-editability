#!/usr/bin/env python3
"""Run independent exploratory arms to completion; failures do not gate other arms."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/hebbian-future-v1"


def status(state, **extra):
    row = {"time": time.time(), "pid": os.getpid(), "state": state, **extra}
    temporary = ART / "execution-status.tmp"
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(ART / "execution-status.json")
    print(json.dumps(row), flush=True)


def main():
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT / "src"),
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MPLCONFIGDIR=str(ART / "matplotlib-cache"),
        TOKENIZERS_PARALLELISM="false",
    )
    sys.path.insert(0, str(ROOT / "src"))
    from llm_memory_editability.hebbian_future import digest, write

    paths = [
        Path(__file__),
        ROOT / "src/llm_memory_editability/hebbian_model.py",
        ROOT / "src/llm_memory_editability/hebbian_data.py",
        ROOT / "data/hebbian-learning-v1/source/qwen3-0.6b-base/model.safetensors",
        ROOT / "data/hebbian-learning-v1/source/qwen3-0.6b-base/config.json",
    ]
    lock = ART / "execution-source-lock.json"
    if not lock.exists():
        write(
            lock,
            {
                "before_launch": time.time(),
                "files": {str(p.relative_to(ROOT)): digest(p) for p in paths},
            },
        )
    jobs = []
    commands = [("real", ["real", "--device", "cuda:2"])]
    commands += [
        (f"chains-{i}", ["chains", "--device", f"cuda:{i + 3}", "--shard", str(i), "--shards", "7"])
        for i in range(7)
    ]
    (ART / "logs").mkdir(exist_ok=True)
    for name, args in commands:
        with (ART / "logs" / (name + ".log")).open("a") as log:
            p = subprocess.Popen(
                [sys.executable, "scripts/run_hebbian_future.py", *args],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        jobs.append((name, p))
    real_analysis = None
    while True:
        codes = {name: proc.poll() for name, proc in jobs}
        if codes["real"] == 0 and real_analysis is None:
            with (ART / "logs/real-analysis.log").open("a") as log:
                real_analysis = subprocess.Popen(
                    [sys.executable, "scripts/report_hebbian_future.py", "real"],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        status(
            "running",
            workers=[{"name": n, "pid": p.pid, "exit": codes[n]} for n, p in jobs],
            real_analysis_exit=None if real_analysis is None else real_analysis.poll(),
        )
        if all(code is not None for code in codes.values()):
            break
        time.sleep(15)
    if any(code != 0 for code in codes.values()):
        status("technical_failure", codes=codes)
        return 1
    if real_analysis is not None and real_analysis.wait() != 0:
        status("real_analysis_failure")
        return 1
    for stage in ("chains", "plot", "audit"):
        subprocess.run(
            [sys.executable, "scripts/report_hebbian_future.py", stage],
            cwd=ROOT,
            env=env,
            check=True,
        )
    status("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
