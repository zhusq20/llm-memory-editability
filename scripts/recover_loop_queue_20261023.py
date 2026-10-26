"""Manual target-host recovery of the exact authorized Loop queue."""

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import lm_resource_snapshot as resources


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = args.base
    recovery = base / "recovery-20261023"
    migration = read(base / "gpu-migration-20261022-13/amendment.json")
    for name, expected in migration["original_sha256"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen controller/config changed: {name}")
    sys.path.insert(0, str(base / "control"))
    import resume_loop_transfer as previous

    original = previous.original

    class Controller(previous.Controller):
        def __init__(self):
            original.Controller.__init__(self, base, args.wandb_token_file)
            self.amendment_path = recovery / "amendment.json"
            self.amendment = {"maximum_local_user_gpus": 4}
            self.plan["gpus"] = [1, 6, 3, 2]
            self.plan["cpu_sets"] = ["70,71", "72,73", "74,75", "78,79"]
            self.plan["gpu_uuids"].update(migration["target_gpu_uuids"])
            self.plan["ignored_created_ids"].append(migration["released_reservation_id"])
            self.state = copy.deepcopy(read(base / "controller-state.json"))
            self.checked = set()

        def existing_container(self, job, stage):
            if stage != "train" or job not in migration["migrations"]:
                return super().existing_container(job, stage)
            expected = migration["migrations"][job]
            receipt = read(base / "execution" / job / stage / "container.json")
            if receipt["id"] != expected["new_container_id"]:
                raise ValueError("Migrated container receipt differs")
            c = previous.inspect_container(receipt["id"])
            if (
                c["Name"] != "/" + expected["new_container_name"]
                or c["Image"] != self.plan["image"]
            ):
                raise ValueError("Migrated training identity changed")
            argv = c["Config"]["Cmd"]
            if argv[argv.index("--job") + 1] != job or "train" not in argv:
                raise ValueError("Migrated command changed")
            return c

    controller = Controller()
    for path, expected in controller.plan["frozen_files"].items():
        if original.digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for job, task in controller.tasks.items():
        if (
            task["kind"] == "new"
            and original.digest(Path(task["source"]) / task["config"]) != task["config_sha256"]
        ):
            raise ValueError(f"Frozen scientific config changed: {job}")
    pending = [
        j
        for phase in controller.plan["phases"]
        for j in phase
        if controller.tasks[j]["kind"] == "new" and not (base / "execution" / j / "train").exists()
    ]
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "read-only",
                    "target": "sn4622129202",
                    "pending_training": pending,
                    "gpu_pool": controller.plan["gpus"],
                    "max_user_gpus": 4,
                },
                indent=2,
            )
        )
        return
    if socket.gethostname() != "sn4622129202":
        raise ValueError("Run only on sn4622129202; other hosts require a separate migration plan")
    resources.verify_code_bundle(Path(__file__).resolve().parent)
    if args.wandb_token_file is None:
        raise ValueError("--wandb-token-file is required for execution")
    original.credential_archive(args.wandb_token_file)
    handles = []
    try:
        for path in [
            base / "controller.lock",
            *map(Path, controller.plan["source_controller_locks"]),
        ]:
            h = path.open("a+")
            handles.append(h)
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (recovery / "controller-process.json").exists():
            raise FileExistsError("Recovery already submitted; do not duplicate it")
        # Check every existing stage against LM Docker before reconciling stale
        # job-level states. Any failed or ambiguous launch remains a hard stop.
        for job, task in controller.tasks.items():
            for stage in task["stages"]:
                directory = base / "execution" / job / stage
                action = previous.stage_action(directory)
                if action == "unstarted":
                    continue
                c = controller.existing_container(job, stage)
                if c["State"]["Status"] not in ("running", "exited") or c["State"]["OOMKilled"]:
                    raise RuntimeError(f"Unclear/failed container: {job}/{stage}")
                if c["State"]["Status"] == "exited" and c["State"]["ExitCode"] != 0:
                    raise RuntimeError(f"Failed stage preserved: {job}/{stage}")
                if action == "complete" and c["State"]["Status"] != "exited":
                    raise RuntimeError(f"Exit receipt mismatch: {job}/{stage}")
        original.write(recovery / "before-state.json", controller.state)
        controller.state.pop("error", None)
        for job in controller.tasks:
            # next_stage checks receipts, and completed jobs execute no model.
            controller.state["jobs"][job]["state"] = "recovering"

        def guarded_snapshot():
            while True:
                try:
                    return resources.snapshot()
                except resources.ResourceUnavailable as exc:
                    controller.update(
                        resource_wait=str(exc), scheduler_heartbeat_utc=original.utc()
                    )
                    time.sleep(30)

        original.snapshot = guarded_snapshot
        process = dict(
            pid=os.getpid(),
            host=socket.gethostname(),
            utc=original.utc(),
            controller_sha256=original.digest(__file__),
            scheduler="recovery-20261023",
        )
        original.write(recovery / "controller-process.json", process)
        original.write(base / "controller-process.json", process)
        original.write(
            recovery / "amendment.json",
            dict(
                authorization=(
                    "2026-10-23 user requests resuming terminated queues; "
                    "user manually launches Loop"
                ),
                previous_migration=str(base / "gpu-migration-20261022-13/amendment.json"),
                scientific_changes=False,
                gpu_pool=[1, 6, 3, 2],
                maximum_local_user_gpus=4,
                retry_failed_training=False,
                source_followups="Two source-host completed models remain deferred; no retraining",
            ),
        )
        controller.update(
            state="running", available_gpu_pool=[1, 6, 3, 2], scheduler="recovery-20261023"
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for index, phase in enumerate(controller.plan["phases"]):
                    controller.update(phase=index)
                    controller.run_phase(phase, pool)
            controller.update(state="complete")
        except Exception as exc:
            controller.update(state="failed", error=str(exc))
            raise
    finally:
        for h in handles:
            h.close()


if __name__ == "__main__":
    os.umask(0o022)
    main()
