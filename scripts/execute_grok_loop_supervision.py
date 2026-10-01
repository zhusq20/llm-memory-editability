#!/usr/bin/env python3
"""Execute the registered continuations on explicitly selected GPUs."""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--config", required=True)
p.add_argument("--gpus", nargs="+", required=True, type=int)
p.add_argument("--action", choices=("train", "audit"), default="train")
args = p.parse_args()
config = json.loads(Path(args.config).read_text())
artifact = Path("docs/development-artifacts/grok-loop-supervision-v1")
pending = queue.Queue()
for run_id in config["runs"]:
    pending.put(run_id)


def worker(gpu):
    while True:
        try:
            run_id = pending.get_nowait()
        except queue.Empty:
            return
        out = Path(config["output_root"]) / run_id
        spec = {**config["base"], **config["runs"][run_id]}
        if args.action == "audit":
            if not (out / "complete.json").exists():
                raise ValueError(f"Missing endpoint: {run_id}")
            path = artifact / f"{run_id}-audit.log"
            with path.open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/run_grok_loop_supervision.py",
                        "audit",
                        "--config",
                        args.config,
                        "--run",
                        run_id,
                    ],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(json.dumps({"run": run_id, "state": "audited"}), flush=True)
            continue
        if (out / "complete.json").exists():
            if json.loads((out / "complete.json").read_text())["spec"] != spec:
                raise ValueError("Completed contract changed")
            continue
        command = [
            sys.executable,
            "scripts/run_grok_loop_supervision.py",
            "train",
            "--config",
            args.config,
            "--run",
            run_id,
        ]
        if (out / "latest.pt").exists():
            command.append("--resume")
        elif out.exists():
            raise ValueError(f"Incomplete startup: inspect {out} before restarting")
        attempt = 1
        while (artifact / f"{run_id}-attempt-{attempt}.log").exists():
            attempt += 1
        path = artifact / f"{run_id}-attempt-{attempt}.log"
        print(json.dumps({"run": run_id, "gpu": gpu, "log": str(path)}), flush=True)
        with path.open("x") as log:
            result = subprocess.run(
                command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Failed {run_id}; inspect {path}")
        print(json.dumps({"run": run_id, "state": "complete"}), flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
    list(pool.map(worker, args.gpus))
