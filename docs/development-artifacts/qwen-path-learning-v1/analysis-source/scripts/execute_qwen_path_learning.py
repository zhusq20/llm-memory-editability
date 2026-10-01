#!/usr/bin/env python3
"""Durable development coordinator; does not expand the preregistered matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/qwen-path-learning-v1"
WORKER = ROOT / "scripts/run_qwen_path_learning.py"


def status(state, **kwargs):
    obj = {"time": time.time(), "state": state, "coordinator_pid": os.getpid(), **kwargs}
    path = ART / "execution-status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)
    print(json.dumps(obj), flush=True)


def parallel(stage, commands):
    jobs = []
    for name, arguments in commands:
        log = (ART / "logs" / f"{name}.log").open("a")
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="4")
        proc = subprocess.Popen(
            [sys.executable, str(WORKER), stage, *arguments],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.close()
        jobs.append((name, proc))
    while True:
        codes = {name: proc.poll() for name, proc in jobs}
        status(
            stage, workers=[{"name": name, "pid": p.pid, "exit": codes[name]} for name, p in jobs]
        )
        if all(code is not None for code in codes.values()):
            if any(code != 0 for code in codes.values()):
                raise RuntimeError(f"{stage} worker failure: {codes}")
            return
        time.sleep(15)


def serial(stage):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, str(WORKER), stage], cwd=ROOT, env=env, check=True)


def main():
    try:
        parallel(
            "train",
            [
                (f"train-{i}", ["--device", f"cuda:{gpu}", "--shard", str(i), "--shards", "8"])
                for i, gpu in enumerate(range(2, 10))
            ],
        )
        serial("select-lr")
        parallel(
            "diagnose",
            [
                (f"diagnose-{i}", ["--device", f"cuda:{i + 2}", "--episode", str(i)])
                for i in range(4)
            ],
        )
        serial("decide")
        decision = json.loads((ART / "candidate-decision.json").read_text())
        status(
            "development_complete",
            decision=decision["status"],
            candidate=decision["candidate"],
            confirmation_started=False,
        )
    except Exception as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
