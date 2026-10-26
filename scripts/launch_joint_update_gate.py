#!/usr/bin/env python3
"""Run one frozen local joint-gate experiment on the user-selected GPU5."""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_gate_extension as reservations  # noqa: E402
import launch_paper_update_gate as common  # noqa: E402


def admitted_snapshot(gpu, uuid, authorized):
    rows, blocked = reservations.snapshot()
    row = rows.get(gpu)
    if row and row["uuid"] == uuid and authorized:
        # User explicitly confirmed this host's utilization display anomaly.
        # All real process, memory, UUID and Docker reservation checks remain.
        rows = {**rows, gpu: {**row, "reported_utilization": row["utilization"], "utilization": 0}}
    return rows, blocked


class Controller(common.Controller):
    def mirror_output(self, force=False):
        return None  # output is a direct persistent-storage path

    def command(self, name, stage, gpu, uuid):
        label, command, cpus = super().command(name, stage, gpu, uuid)
        command[command.index("scripts/run_paper_update_gate.py")] = (
            "scripts/run_joint_update_gate.py"
        )
        return label, command, cpus

    def worker(self, name, gpu, uuid):
        for stage in ("train", "grid"):
            while True:
                rows, blocked = common.snapshot()
                if reservations.idle(gpu, uuid, rows, blocked):
                    break
                self.update(
                    name,
                    state="waiting_for_idle_gpu",
                    stage=stage,
                    gpu=gpu,
                    resources=rows.get(gpu),
                    reservations=blocked.get(gpu, []),
                )
                time.sleep(20)
            self.execute(name, stage, gpu, uuid)
        self.update(name, state="training_and_grid_complete_audit_not_requested", grid=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    relative = "configs/joint-update-gate-multihop-v1.json"
    config = common.read(base / "source" / relative)
    rt = config["runtime"]
    gpu, uuid = rt["gpu"], rt["gpu_uuid"]
    if gpu != 5 or rt["docker_context"] != "lm-memory":
        raise ValueError("Only local LM GPU5 is authorized for this run")
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    receipt = common.read(output / "preflight/joint-token.json")
    if receipt.get("passed") is not True or receipt["config_sha256"] != common.sha(
        base / "source" / relative
    ):
        raise ValueError("The exact frozen configuration must pass the CUDA check")
    common.snapshot = lambda: admitted_snapshot(gpu, uuid, rt["display_anomaly_user_authorized"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Preserve earlier submission; do not restart")
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu_uuids": {gpu: uuid},
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / relative),
                "utilization_display_exception": (
                    "explicit user authorization; real occupancy checks retained"
                ),
            },
        )
        controller = Controller(config, base, output, relative, {gpu: uuid}, args.wandb_token_file)
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
