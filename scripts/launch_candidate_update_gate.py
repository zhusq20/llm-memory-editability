#!/usr/bin/env python3
"""Run the candidate-conditioned joint gate on an independently idle GPU."""

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
import launch_joint_update_gate as prior  # noqa: E402

common, reservations = prior.common, prior.reservations
RELATIVE = "configs/candidate-update-gate-multihop-v1.json"


class Controller(common.Controller):
    def mirror_output(self, force=False):
        return None

    def command(self, name, stage, gpu, uuid):
        label, command, cpus = super().command(name, stage, gpu, uuid)
        command[command.index("scripts/run_paper_update_gate.py")] = (
            "scripts/run_candidate_update_gate.py"
        )
        return label, command, cpus

    def worker(self, name, gpu, uuid):
        for stage in ("preflight", "train", "grid"):
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
            if stage == "train":
                receipt = common.read(self.output / "preflight" / f"{name}.json")
                if receipt.get("passed") is not True or receipt["config_sha256"] != common.sha(
                    self.base / "source" / self.config_relative
                ):
                    raise ValueError("The exact candidate-gate configuration must pass CUDA checks")
            self.update(
                state={
                    "preflight": "engineering_validation",
                    "train": "scientific_training",
                    "grid": "endpoint_evaluation",
                }[stage]
            )
            self.execute(name, stage, gpu, uuid)
        self.update(name, state="training_and_grid_complete_audit_not_requested", grid=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    config = common.read(base / "source" / RELATIVE)
    rt = config["runtime"]
    gpu, uuid = rt["gpu"], rt["gpu_uuid"]
    if gpu != 0 or rt["docker_context"] != "lm-memory":
        raise ValueError("Only the frozen independently idle GPU0 is selected")
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    common.snapshot = reservations.snapshot
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Preserve prior submission; do not restart")
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu_uuids": {gpu: uuid},
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / RELATIVE),
                "comparison": config["comparison"],
            },
        )
        controller = Controller(config, base, output, RELATIVE, {gpu: uuid}, args.wandb_token_file)
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
