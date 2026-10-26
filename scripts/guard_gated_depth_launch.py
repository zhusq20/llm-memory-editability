#!/usr/bin/env python3
"""Scheduler-only amendment: recheck the GPU immediately before create/start.

The original frozen scientific source, image, data and stage claims stay intact.
An existing claimed container is recovered by the original controller.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def wait_for_gpu(module, config, base, job, stage, row):
    runtime = config["runtime"]
    path = Path(base) / "admission-rechecks" / f"{job}-{stage}.json"
    while True:
        rows, blocked = module.resource_snapshot()
        for index, reason in module.previous_queue_blockers(runtime).items():
            blocked.setdefault(index, []).append(reason)
        index = row["gpu"]
        if index not in rows or rows[index]["uuid"] != row["uuid"]:
            raise ValueError("Admitted GPU identity changed before launch")
        available = index in module.idle_candidates(runtime, rows, blocked)
        module.write(
            path,
            {
                "utc": module.utc(),
                "gpu": index,
                "uuid": row["uuid"],
                "available": available,
                "snapshot": rows[index],
                "blocked": blocked.get(index, []),
                "job": job,
                "stage": stage,
            },
        )
        if available:
            return
        time.sleep(runtime["poll_seconds"])


def install(module):
    original_command, original_docker = module.container_command, module.docker
    pending = {}

    def command(config, base, job, stage, lane, row=None, arm=None, seed=None):
        result = original_command(config, base, job, stage, lane, row, arm, seed)
        if row is not None and not (Path(base) / "execution" / job / stage / "claim.json").exists():
            wait_for_gpu(module, config, base, job, stage, row)
            pending[result[0]] = config, base, job, stage, row
        return result

    def docker(*args, **kwargs):
        if len(args) >= 2 and args[0] == "start" and args[1] in pending:
            wait_for_gpu(module, *pending[args[1]])
            pending.pop(args[1])
        return original_docker(*args, **kwargs)

    module.container_command, module.docker = command, docker


def main():
    base = Path(sys.argv[sys.argv.index("--base") + 1]).resolve()
    amendment = json.loads((base / "schedule-amendment.json").read_text())
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if actual != amendment["scheduler_guard_sha256"]:
        raise ValueError("Scheduler amendment hash differs")
    path = base / "source/scripts/launch_gated_depth.py"
    spec = importlib.util.spec_from_file_location("frozen_gated_depth_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install(module)
    module.main()


if __name__ == "__main__":
    main()
