#!/usr/bin/env python3
"""Execute the explicitly amended depth-1 comparison at 1.5M on GPU4."""

import argparse
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path

from migrate_gate_extension_gpu4 import GPU, UUID, filter_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.base / "source/scripts"), str(args.base / "source/src")]
    import launch_gate_extension as original

    common = original.common
    config_path = "configs/gate-depth-long-gpu4-v1.json"
    config = common.read(args.base / "source" / config_path)
    assert config["training"]["steps"] == 1500000
    assert config["family"] == "depth" and 300000 in config["evaluation_nodes"]
    common.verify_frozen(args.base, config)
    common.credential_archive(args.wandb_token_file)
    jobs = [
        {
            "id": f"depth-1--{arm}",
            "case": "depth-1",
            "arm": arm,
            "config": config_path,
            "runner": "scripts/run_gate_extension.py",
        }
        for arm in ("paper", "fixed-update", "token-update")
    ]

    def snapshot():
        rows, blocked = original.snapshot()
        return filter_snapshot(common, rows, blocked)

    common.snapshot = snapshot
    args.output.mkdir(parents=True, exist_ok=True)
    with (
        (args.output / "controller.lock").open("a") as lock,
        Path("/tmp/lm-memory-gate-extension-handoff-4.lock").open("a") as gpu_lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "controller-process.json").exists():
            raise FileExistsError("Preserve previous attempt")
        common.write(
            args.output / "controller-process.json",
            {
                "pid": os.getpid(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu": GPU,
                "gpu_uuid": UUID,
                "freeze_sha256": common.sha(args.base / "freeze.json"),
            },
        )
        state = {
            "state": "scheduled",
            "total_scientific_updates": 4500000,
            "additional_updates_vs_parent": 3600000,
            "jobs": {j["id"]: {"state": "pending", **j} for j in jobs},
        }

        class Owner:
            base, output, token_file = args.base, args.output, args.wandb_token_file
            cpus = [84, 85]

            def update(self, job=None, **values):
                (state if job is None else state["jobs"][job]).update(values)
                state["updated_utc"] = common.utc()
                common.write(self.output / "controller-state.json", state)

            def wait_idle(self, gpu, uuid):
                while True:
                    rows, blocked = snapshot()
                    if original.idle(gpu, uuid, rows, blocked):
                        return
                    self.update(
                        state="waiting_for_idle_gpu",
                        resources=rows.get(gpu),
                        reservations=blocked.get(gpu, []),
                    )
                    time.sleep(20)

        owner = Owner()
        try:
            owner.update(state="scientific_training")
            for job in jobs:
                original.JobController(owner, job, GPU, UUID, 0).worker(job["arm"], GPU, UUID)
            owner.update(state="training_and_grid_complete_audit_not_requested")
            common.write(args.output / "controller-exit.json", {"code": 0, "utc": common.utc()})
        except BaseException as error:
            owner.update(
                state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            common.write(args.output / "controller-exit.json", {"code": 1, "utc": common.utc()})
            raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
