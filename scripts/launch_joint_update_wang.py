#!/usr/bin/env python3
"""Queue the original-Wang joint gate after the existing GPU5 train and grid."""

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
RELATIVE = "configs/joint-update-gate-wang-v1.json"


def predecessor_ready(folder, dependency):
    """Training alone, GPU idleness, or a stale status cannot release this run."""
    folder = Path(folder)
    if common.sha(folder / "freeze.json") != dependency["freeze_sha256"]:
        raise ValueError("Predecessor identity changed")
    exit_path = folder / "controller-exit.json"
    if exit_path.exists() and common.read(exit_path)["code"] != 0:
        raise RuntimeError("Predecessor failed; preserve it and hold the dependent run")
    if (folder / "controller-state.json").exists():
        state = common.read(folder / "controller-state.json")
        if state.get("state") == "needs_attention":
            raise RuntimeError("Predecessor needs attention; dependent run was not started")
    for stage in dependency["required_stages"]:
        path = folder / "execution" / dependency["arm"] / stage / "exit.json"
        if not path.exists():
            return False
        if common.read(path)["code"] != 0:
            raise RuntimeError(f"Predecessor {stage} failed")
    if not exit_path.exists():
        return False
    run = folder / "runs" / dependency["arm"]
    complete, grid = common.read(run / "complete.json"), common.read(run / "grid.json")
    if (
        complete["state"] != "complete"
        or complete["step"] != dependency["budget"]
        or complete["budget"] != dependency["budget"]
    ):
        raise ValueError("Predecessor did not finish its scientific budget")
    for key in ("step", "arm", "world_sha256", "model_sha256", "config_sha256"):
        if complete[key] != grid[key]:
            raise ValueError(f"Predecessor train/grid identity mismatch: {key}")
    return True


class Controller(common.Controller):
    def mirror_output(self, force=False):
        return None

    def command(self, name, stage, gpu, uuid):
        label, command, cpus = super().command(name, stage, gpu, uuid)
        command[command.index("scripts/run_paper_update_gate.py")] = (
            "scripts/run_joint_update_wang.py"
        )
        mount = f"type=bind,source={self.base / 'data'},target={common.DATA_TARGET},readonly"
        command[command.index(mount)] = (
            f"type=bind,source={self.base / 'data'},target=/inputs,readonly"
        )
        return label, command, cpus

    def worker(self, name, gpu, uuid):
        dependency = self.config["dependency"]
        predecessor = Path(self.config["runtime"]["persistent_root"]) / dependency["batch"]
        self.update(state="waiting_for_predecessor")
        self.update(name, state="waiting_for_predecessor", predecessor=str(predecessor), gpu=gpu)
        while not predecessor_ready(predecessor, dependency):
            status_path = predecessor / "runs" / dependency["arm"] / "status.json"
            self.update(
                name,
                state="waiting_for_predecessor",
                predecessor_progress=common.read(status_path) if status_path.exists() else {},
            )
            time.sleep(30)
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
                    raise ValueError("The exact Wang configuration must pass CUDA checks")
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
    if gpu != 5 or rt["docker_context"] != "lm-memory":
        raise ValueError("User selected GPU5 after the current run")
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    common.snapshot = lambda: prior.admitted_snapshot(
        gpu, uuid, rt["display_anomaly_user_authorized"]
    )
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
                "dependency": config["dependency"],
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
