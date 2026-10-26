#!/usr/bin/env python3
"""Scheduling-only takeover: first LR arm on user-designated GPU7, others queue.

Uses the original frozen container builder, trainer, inputs, config and audits.
No science stage may exist before this one-time takeover. GPU7 can host the
first arm alongside current jobs with at least 24GiB free, retaining full-size validation;
other arms retain idle-GPU admission, ignoring only the two stale default GPU7
reservations explicitly recorded below. No other container or CUDA PID is stopped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import importlib.util
import os
import sys
import threading
import time
import traceback
from pathlib import Path

FIRST_JOB = "loop-lr2e-4"
GPU7_MIN_FREE_MIB = 24576
GPU7_UUID = "GPU-462bb54b-05c2-4014-6be9-70d6afec79d0"
STALE_DEFAULT = {
    "/large-taylor-k14-sigma003-to1700-eval-gpu7-20261017",
    "/large-taylor-k14-three-sigma-eval-gpu7-recovery-20261016",
}


def load_frozen(path):
    path = Path(path)
    sys.path.insert(0, str(path.parents[1] / "src"))
    spec = importlib.util.spec_from_file_location("lr_control_frozen_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admit(module, config, name, snapshot, gpu=None):
    if gpu != 7:
        return module.admission(config, name, snapshot, gpu)
    actual = snapshot["gpus"].get(7)
    if actual is None or actual["uuid"] != GPU7_UUID:
        raise ValueError("User-designated GPU7 UUID changed")
    adjusted = copy.deepcopy(snapshot)
    designated = name == FIRST_JOB
    for container in adjusted["containers"]:
        ignore = designated or (
            container.get("inspection_context") == "default" and container["Name"] in STALE_DEFAULT
        )
        if ignore:
            for request in container["HostConfig"].get("DeviceRequests") or []:
                request["DeviceIDs"] = [
                    value
                    for value in request.get("DeviceIDs") or []
                    if value not in ("7", GPU7_UUID)
                ]
    admission_config = config
    if designated:
        # The user expressly selected this shared card. Require 24GiB free and
        # retain full-size short-training/reload gates before formal training.
        admission_config = copy.deepcopy(config)
        admission_config["runtime"]["min_free_gpu_mib"] = GPU7_MIN_FREE_MIB
        # Only this explicitly designated first arm can share GPU7.
        adjusted["gpus"][7]["used_mib"] = 0
        adjusted["gpus"][7]["utilization"] = 0
        adjusted["processes"] = [p for p in snapshot["processes"] if p["uuid"] != GPU7_UUID]
    row = module.admission(admission_config, name, adjusted, 7)
    if row is not None:
        row.update(actual)
        row.update(
            {
                "schedule_amendment": "gpu7-user-designation-v1",
                "user_designated_gpu": 7,
                "exclusive_idle_gpu": not designated,
                "sharing_allowed_for_this_arm": designated,
                "minimum_free_gpu_mib": (
                    GPU7_MIN_FREE_MIB if designated else config["runtime"]["min_free_gpu_mib"]
                ),
                "observed_existing_gpu_pids": [
                    p for p in snapshot["processes"] if p["uuid"] == GPU7_UUID
                ],
                "existing_tasks_stopped": False,
            }
        )
    return row


class Lease:
    def __init__(self, files):
        self.files = files

    def close(self):
        for held in reversed(self.files):
            held.close()


def acquire_gpu_lock(root, gpu, name):
    """Coordinate our GPU7 stages without breaking another job's exclusive lock."""
    paths = []
    if gpu == 7:
        paths.append(root / "lr-control-user-gpu7.lock")
    if gpu != 7 or name != FIRST_JOB:
        paths.append(root / f"gpu-{gpu}.lock")
    files = []
    try:
        for path in paths:
            held = path.open("a")
            files.append(held)
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Lease(files)
    except BlockingIOError:
        Lease(files).close()
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    os.umask(0o022)
    batch = args.batch
    module = load_frozen(batch / "source/scripts/launch_lr_control.py")
    config = module.read(batch / "source/configs/loop-discoloop-lr-control-v1.json")
    module.validate(config)
    amendment = module.read(args.amendment)
    if amendment["first_job"] != FIRST_JOB or amendment["gpu_uuid"] != GPU7_UUID:
        raise ValueError("Unexpected scheduling amendment")
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Existing private W&B credential required")
    with (batch / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        module.verify(batch)
        state = module.read(batch / "controller-state.json")
        if state["contracts"]["state"] != "passed":
            raise ValueError("Original frozen CPU contracts must already have passed")
        if any(
            value.get("completed_stages") or value["state"] != "waiting_for_idle_resources"
            for value in state["jobs"].values()
        ):
            raise ValueError("Takeover is restricted to the original zero-stage waiting queue")
        active = module.docker("ps", "--filter", f"name={module.BATCH}", "-q").stdout.split()
        if active:
            raise ValueError("Do not take over any running stage")
        state.update(
            state="running_or_waiting", scheduling_amendment=str(args.amendment), utc=module.utc()
        )
        module.write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": module.utc(),
                "automatic_retry": False,
                "source_manifest_sha256": module.sha(batch / "source-manifest.json"),
                "config_sha256": module.sha(
                    batch / "source/configs/loop-discoloop-lr-control-v1.json"
                ),
                "scheduling_amendment": str(args.amendment),
                "scheduler_source_sha256": module.sha(Path(__file__)),
            },
        )
        state_lock, resources_lock = threading.Lock(), threading.Lock()

        def update(name, **values):
            with state_lock:
                state["jobs"][name].update(values)
                state["utc"] = module.utc()
                module.write(batch / "controller-state.json", state)
                module.write(batch / "science-schedule.json", state["jobs"])

        lock_root = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
        lock_root.mkdir(exist_ok=True)

        def wait_slot(name, stage, retained=None):
            deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
            preferences = [7] if name == FIRST_JOB else config["runtime"]["gpu_preference"]
            while time.monotonic() < deadline:
                with resources_lock:
                    snapshot = module.resource_snapshot()
                    if stage == "cloud-audit":
                        row = admit(module, config, name, snapshot)
                        if row is not None:
                            return None, row
                    else:
                        for gpu in preferences:
                            held = retained
                            if held is None:
                                held = acquire_gpu_lock(lock_root, gpu, name)
                                if held is None:
                                    continue
                            row = admit(module, config, name, snapshot, gpu)
                            if row is not None:
                                return held, row
                            if held is not retained:
                                held.close()
                    update(
                        name,
                        state="waiting_for_idle_resources",
                        stage=stage,
                        last_resource_check_utc=snapshot["utc"],
                        observed_gpu_memory={
                            str(k): v["used_mib"] for k, v in snapshot["gpus"].items()
                        },
                    )
                time.sleep(5 if name == FIRST_JOB else config["runtime"]["poll_seconds"])
            raise TimeoutError("Resource waiting timeout; existing tasks unchanged")

        def worker(name):
            completed, retained = [], None
            try:
                for stage in module.STAGES:
                    held, row = wait_slot(name, stage, retained)
                    if name == FIRST_JOB and stage != "cloud-audit":
                        retained = held
                    try:
                        module.execute(
                            config,
                            batch,
                            name,
                            stage,
                            token,
                            row,
                            lambda **values: update(name, **values),
                        )
                    finally:
                        if held is not None and held is not retained:
                            held.close()
                    completed.append(stage)
                    update(name, completed_stages=completed, utc=module.utc())
                    if stage == "audit" and retained is not None:
                        retained.close()
                        retained = None
                update(
                    name,
                    state="complete",
                    independent_audit_passed=True,
                    cloud_verified=True,
                    scientific_updates=300000,
                    utc=module.utc(),
                )
                return True
            except BaseException as error:
                update(
                    name,
                    state="needs_attention",
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    completed_stages=completed,
                    utc=module.utc(),
                )
                return False
            finally:
                if retained is not None:
                    retained.close()

        module.write(batch / "controller-state.json", state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            names = [FIRST_JOB, *(name for name in config["jobs"] if name != FIRST_JOB)]
            outcomes = list(pool.map(worker, names))
        state.update(state="complete" if all(outcomes) else "needs_attention", utc=module.utc())
        if all(outcomes):
            module.comparison(config, batch)
            module.write(batch / "completion-manifest.json", state)
        module.write(batch / "controller-state.json", state)
        if not all(outcomes):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
