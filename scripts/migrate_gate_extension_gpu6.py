#!/usr/bin/env python3
"""Claim selected unstarted confirmation jobs via their shared atomic stage guard.

The original scheduler creates execution/<arm>/train with exist_ok=False before
Docker create. Claiming that directory prevents duplicate submission even when
that scheduler lives in another process namespace. Scientific source/configs
stay byte-identical. Relative output links preserve the original report paths.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

GPU = 6
UUID = "GPU-d1972508-51b0-ae3d-188b-f1613ec33e07"
IMAGE = "sha256:fa200da9dd78d3423ffc30572bda8b88a5b1c7e72a25509bf63e6ae4a5a3ef6f"
CPUS = [12, 13]
LEASE_PREFIX = "lm-memory-gate-extension"
SNAPSHOT_FILTER = None


def select_jobs(plan, identifiers):
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Select distinct existing job IDs")
    lookup = {job["id"]: job for job in plan["jobs"]}
    if any(identifier not in lookup for identifier in identifiers):
        raise ValueError("Every selected job must exist in the frozen plan")
    jobs = [lookup[identifier] for identifier in identifiers]
    if any(
        job["family"] != "confirmation" or job["start_step"] != 0 or job["steps"] != 300000
        for job in jobs
    ):
        raise ValueError("Only existing 300k confirmation jobs may be transferred")
    return jobs


def claim_job(old_output, output, job, write):
    old_output, output = Path(old_output), Path(output)
    current = json.loads((old_output / "controller-state.json").read_text())
    if current["jobs"][job["id"]]["state"] != "pending":
        raise ValueError(f"Job is no longer pending: {job['id']}")
    old_case = old_output / "cases" / job["case"]
    new_case = output / "cases" / job["case"]
    old_run = old_case / "runs" / job["arm"]
    if old_run.exists() or old_run.is_symlink():
        raise FileExistsError(f"Preserve existing run: {old_run}")
    claim = old_case / "execution" / job["arm"] / "train"
    claim.mkdir(parents=True, exist_ok=False)  # Same atomic guard as original scheduler.
    write(
        claim / "handoff.json",
        {
            "job": job,
            "state": f"transferred_to_local_gpu{GPU}",
            "host": socket.gethostname(),
            "gpu": GPU,
            "gpu_uuid": UUID,
            "output": str(new_case),
            "old_scheduler_behavior": "existing-stage guard refuses duplicate before Docker create",
            "scientific_updates_added": 0,
        },
    )
    targets = {
        old_run: new_case / "runs" / job["arm"],
        old_case / "execution" / job["arm"] / "grid": new_case / "execution" / job["arm"] / "grid",
    }
    for name in ("command.json", "launch.json", "exit.json", "heartbeat.json", "stdout.log"):
        targets[claim / name] = new_case / "execution" / job["arm"] / "train" / name
    for stage in ("train", "grid"):
        name = f"{job['arm']}-{stage}.json"
        targets[old_case / "failures" / name] = new_case / "failures" / name
    for link, target in targets.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
    return str(claim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(base / "source/scripts"), str(base / "source/src")]
    import launch_gate_extension as original

    common = original.common
    plan = common.read(base / "source/configs/gate-generalization-extension-v1.json")
    jobs = select_jobs(plan, args.jobs)
    common.verify_frozen(base, {"runtime": {"image": IMAGE}})
    common.credential_archive(args.wandb_token_file)

    def snapshot():
        rows, blocked = original.snapshot()
        if rows[GPU]["uuid"] != UUID:
            raise ValueError("The requested local GPU identity changed")
        if SNAPSHOT_FILTER is not None:
            rows, blocked = SNAPSHOT_FILTER(common, rows, blocked)
        return rows, blocked

    common.snapshot = snapshot
    output.mkdir(parents=True, exist_ok=True)
    with (
        (output / "controller.lock").open("a") as lock,
        Path(f"/tmp/{LEASE_PREFIX}-{GPU}.lock").open("a") as gpu_lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Do not restart an existing transfer")
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu": GPU,
                "gpu_uuid": UUID,
                "freeze_sha256": common.sha(base / "freeze.json"),
            },
        )
        state = {"state": "claiming_pending_jobs", "jobs": {}, "new_scientific_updates": 0}

        def update(job=None, **values):
            target = state if job is None else state["jobs"].setdefault(job, {})
            target.update(values)
            state["updated_utc"] = common.utc()
            common.write(output / "controller-state.json", state)

        class Receiver(common.Controller):
            def __init__(self, job):
                config = common.read(base / "source" / job["config"])
                # Transport override only: the runner reads the ORIGINAL config bytes.
                config["runtime"] = {**config["runtime"], "image": IMAGE, "cpu_first": CPUS[0]}
                super().__init__(
                    config, base, output, job["config"], {GPU: UUID}, args.wandb_token_file
                )
                self.job = job
                self.output = output / "cases" / job["case"]
                self.output.mkdir(parents=True, exist_ok=True)

            def mirror_output(self, force=False):
                return None

            def update(self, name=None, **values):
                update(self.job["id"], **values)

            def command(self, name, stage, gpu, uuid):
                label, command, _ = super().command(name, stage, gpu, uuid)
                command[command.index("--cpuset-cpus") + 1] = ",".join(map(str, CPUS))
                old = f"type=bind,source={base / 'data'},target={common.DATA_TARGET},readonly"
                command[command.index(old)] = (
                    f"type=bind,source={base / 'data'},target=/inputs,readonly"
                )
                i = command.index("scripts/run_paper_update_gate.py")
                command[i:] = [
                    self.job["runner"],
                    "--stage",
                    stage,
                    "--config",
                    self.config_relative,
                    "--arm",
                    name,
                    "--output",
                    "/output",
                ]
                return label, command, CPUS

        try:
            for job in jobs:
                claim = claim_job(args.old_output, output, job, common.write)
                update(job["id"], state=f"queued_on_gpu{GPU}", claim=claim, **job)
            amendment = {
                "utc": common.utc(),
                "jobs": jobs,
                "gpu": GPU,
                "gpu_uuid": UUID,
                "original_output": str(args.old_output),
                "new_output": str(output),
                "actual_image": IMAGE,
                "original_image": plan["image"],
                "science_source_and_config_changed": False,
                "new_scientific_updates": 0,
                "old_scheduler_note": "Its stage guard may report FileExistsError for these "
                "transferred jobs. Read this receiver status and shared output links; "
                "this is duplicate prevention, not a failed scientific run.",
            }
            common.write(output / "schedule-amendment.json", amendment)
            common.write(args.old_output / f"schedule-amendment-{output.name}.json", amendment)
            update(state="running_transferred_queue")
            for job in jobs:
                receiver = Receiver(job)
                for stage in ("train", "grid"):
                    while True:
                        rows, blocked = snapshot()
                        if original.idle(GPU, UUID, rows, blocked):
                            break
                        update(
                            job["id"],
                            state="waiting_for_idle_gpu",
                            resources=rows[GPU],
                            reservations=blocked.get(GPU, []),
                        )
                        time.sleep(20)
                    receiver.execute(job["arm"], stage, GPU, UUID)
                update(job["id"], state="training_and_grid_complete_audit_not_requested")
            update(state="complete")
            common.write(output / "controller-exit.json", {"code": 0, "utc": common.utc()})
        except BaseException as error:
            update(state="needs_attention", error=repr(error), traceback=traceback.format_exc())
            common.write(output / "controller-exit.json", {"code": 1, "utc": common.utc()})
            raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
