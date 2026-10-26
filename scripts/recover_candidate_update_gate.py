#!/usr/bin/env python3
"""Queue the candidate gate after the existing GPU2 run, without adding a GPU lane."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pwd
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_candidate_update_gate as launch  # noqa: E402
import launch_ungated_loop_continuation as dependency_check  # noqa: E402

common, reservations = launch.common, launch.reservations


def owned_gpus():
    """Count current user's host CUDA processes and owner-labelled Docker GPUs."""
    owner = pwd.getpwuid(os.getuid()).pw_name
    rows, _ = reservations.snapshot()
    used = set()
    for context in ("default", "lm-memory"):
        ids = common.docker("ps", "-q", context=context).stdout.split()
        if not ids:
            continue
        for item in json.loads(common.docker("inspect", *ids, context=context).stdout):
            if (item["Config"].get("Labels") or {}).get("io.gpu-quota.owner") != owner:
                continue
            for request in item["HostConfig"].get("DeviceRequests") or []:
                if request.get("Count") == -1:
                    used.update(row["uuid"] for row in rows.values())
                for device in request.get("DeviceIDs") or []:
                    used.add(rows[int(device)]["uuid"] if device.isdigit() else device)
    report = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    for line in report.splitlines():
        pid, uuid = [part.strip() for part in line.split(",")]
        try:
            if Path(f"/proc/{pid}").stat().st_uid == os.getuid():
                used.add(uuid)
        except FileNotFoundError:
            continue
    return used


def capacity_available(current, selected, maximum):
    return len(set(current) | {selected}) <= maximum


class Controller(launch.Controller):
    def worker(self, name, gpu, uuid):
        recovery = self.config["scheduling_recovery"]
        dependency = recovery["dependency"]
        deadline = time.monotonic() + 7 * 24 * 3600
        self.update(state="waiting_for_predecessor_and_capacity")
        while not dependency_check.predecessor_ready(dependency):
            if time.monotonic() > deadline:
                raise TimeoutError("Predecessor wait exceeded seven days")
            self.update(name, state="waiting_for_predecessor", gpu=gpu, dependency=dependency)
            time.sleep(30)
        super().worker(name, gpu, uuid)

    def execute(self, name, stage, gpu, uuid):
        deadline = time.monotonic() + 7 * 24 * 3600
        maximum = self.config["scheduling_recovery"]["maximum_concurrent_owned_gpus"]
        while True:
            current = owned_gpus()
            if capacity_available(current, uuid, maximum):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Conservative GPU capacity wait exceeded seven days")
            self.update(name, state="waiting_for_gpu_capacity", owned_gpu_count=len(current))
            time.sleep(30)
        return super().execute(name, stage, gpu, uuid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    config = common.read(base / "source" / launch.RELATIVE)
    rt = config["runtime"]
    if rt["gpu"] != 2 or rt["docker_context"] != "lm-memory":
        raise ValueError("Recovery must wait on the existing GPU2 lane")
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    common.snapshot = reservations.snapshot
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Do not resubmit this recovery")
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / launch.RELATIVE),
                "scheduling_recovery": config["scheduling_recovery"],
            },
        )
        controller = Controller(
            config,
            base,
            output,
            launch.RELATIVE,
            {rt["gpu"]: rt["gpu_uuid"]},
            args.wandb_token_file,
        )
        controller.output = output
        passed = controller.run()
        common.write(
            output / "controller-exit.json", {"code": 0 if passed else 1, "utc": common.utc()}
        )
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
