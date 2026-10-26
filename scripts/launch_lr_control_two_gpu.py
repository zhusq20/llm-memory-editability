#!/usr/bin/env python3
"""Dispatch the four transferred arms on GPU1/2; each arm runs exactly once."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

LANES = {
    1: ("64,65,66,67", ("loop-lr5e-4", "discoloop-lr2e-4")),
    2: ("68,69,70,71", ("loop-lr1e-3", "discoloop-lr5e-4")),
}


def save(path, value):
    tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    batch, origin = args.batch.resolve(), args.origin.resolve()
    state = batch / "handoff"
    amendment = state / "runtime-amendment.json"
    entry = state / "continue_lr_control_handoff.py"
    receipt = json.loads(amendment.read_text())
    if socket.gethostname() != receipt["host"] or receipt["batch_directory"] != str(batch):
        raise ValueError("Wrong host or batch directory")
    if not args.execute:
        print(json.dumps({"lanes": LANES, "scientific_updates": 1200000}, indent=2))
        return
    # NFS root-squash: Docker must traverse newly created bind-mount paths.
    # Credentials stay in the existing private file outside the experiment tree.
    os.umask(0o022)
    with (state / "two-gpu.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (state / "two-gpu-process.json").exists():
            raise FileExistsError("Existing dispatcher receipt; inspect before recovery")
        # Retain leases in the shared original directory for the whole transfer.
        # An old handoff invocation cannot submit these jobs on the source host.
        origin_locks = []
        for _, jobs in LANES.values():
            for job in jobs:
                if (origin / "execution" / job / "train").exists():
                    raise ValueError(f"Source already claimed scientific training: {job}")
                lock = (origin / "handoff" / f"{job}.lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                origin_locks.append(lock)
        save(
            state / "two-gpu-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "lanes": LANES,
                "utc": datetime.now(timezone.utc).isoformat(),
                "automatic_retry": False,
            },
        )

        def lane(gpu, cpus, jobs):
            outcomes = {}
            for job in jobs:
                save(
                    state / f"lane-{gpu}.json",
                    {"state": "running", "job": job, "outcomes": outcomes},
                )
                command = [
                    sys.executable,
                    str(entry),
                    "--batch",
                    str(batch),
                    "--job",
                    job,
                    "--gpu",
                    str(gpu),
                    "--cpus",
                    cpus,
                    "--wandb-token-file",
                    str(args.wandb_token_file.resolve()),
                    "--runtime-amendment",
                    str(amendment),
                    "--execute",
                ]
                with (state / f"{job}-monitor.log").open("x") as log:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                outcomes[job] = code
                save(state / f"{job}-monitor-exit.json", {"code": code})
                # A failed arm is preserved; the independent next arm may proceed.
            save(state / f"lane-{gpu}.json", {"state": "finished", "outcomes": outcomes})
            return outcomes

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, gpu, *values) for gpu, values in LANES.items()]
            outcomes = {}
            for future in futures:
                outcomes.update(future.result())
        passed = all(code == 0 for code in outcomes.values())
        save(
            state / "two-gpu-result.json",
            {"state": "complete" if passed else "needs_attention", "outcomes": outcomes},
        )
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
