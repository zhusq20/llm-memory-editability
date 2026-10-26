#!/usr/bin/env python3
"""Queue two ungated baselines behind the current GPU1/2 continuation and grids."""

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
import launch_paper_update_gate as common  # noqa: E402

RELATIVE = "configs/ungated-loop-continuation-v1.json"


def predecessor_ready(dependency):
    root, name = Path(dependency["root"]), dependency["arm"]
    for stage in ("train", "grid"):
        folder = root / "execution" / name / stage
        if not (folder / "exit.json").exists():
            return False
        if common.read(folder / "exit.json")["code"] != 0:
            raise RuntimeError(f"Predecessor {name}/{stage} failed; hold this baseline")
    complete = common.read(root / "runs" / name / "complete.json")
    grid = common.read(root / "runs" / name / "grid.json")
    if (
        complete["state"] != "complete"
        or complete["step"] != complete["budget"]
        or complete["step"] != 1500000
    ):
        raise ValueError("Predecessor budget incomplete")
    if complete["config_sha256"] != dependency["config_sha256"]:
        raise ValueError("Predecessor configuration changed")
    for key in ("arm", "step", "model_sha256", "config_sha256", "world_sha256"):
        if complete[key] != grid[key]:
            raise ValueError("Predecessor train/grid mismatch")
    return True


def idle(gpu, uuid, rows, blocked):
    row = rows.get(gpu)
    return bool(
        row
        and row["uuid"] == uuid
        and not blocked.get(gpu)
        and row["used"] <= 512
        and row["utilization"] <= 5
        and row["free"] >= 18000
    )


class Controller(common.Controller):
    def mirror_output(self, force=False):
        return None  # All outputs are on persistent storage.

    def command(self, name, stage, gpu, uuid):
        label, args, cpus = super().command(name, stage, gpu, uuid)
        mount = f"type=bind,source={self.base / 'data'},target={common.DATA_TARGET},readonly"
        args[args.index(mount)] = f"type=bind,source={self.base / 'data'},target=/inputs,readonly"
        index = args.index("scripts/run_paper_update_gate.py")
        args[index:] = [
            "scripts/run_ungated_loop_continuation.py",
            "--stage",
            stage,
            "--config",
            self.config_relative,
            "--output",
            "/output",
            "--arm",
            name,
        ]
        return label, args, cpus

    def worker(self, name, gpu, uuid):
        deadline = time.monotonic() + self.config["runtime"]["waiting_timeout_hours"] * 3600
        dependency = self.config["dependencies"][name]
        try:
            while not predecessor_ready(dependency):
                if time.monotonic() > deadline:
                    raise TimeoutError("Predecessor wait exceeded seven days")
                self.update(name, state="waiting_for_predecessor", gpu=gpu, predecessor=dependency)
                time.sleep(30)
            for stage in ("preflight", "train", "grid"):
                while True:
                    rows, blocked = common.snapshot()
                    if idle(gpu, uuid, rows, blocked):
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("No exclusive GPU admitted within seven days")
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
                    if not receipt["passed"] or receipt["config_sha256"] != common.sha(
                        self.base / "source" / RELATIVE
                    ):
                        raise ValueError("CUDA resume check did not pass for this config")
                self.execute(name, stage, gpu, uuid)
            self.update(name, state="training_and_grid_complete_independent_audit_not_requested")
        except BaseException as error:
            self.update(name, state="needs_attention", error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    config = common.read(args.base / "source" / RELATIVE)
    common.verify_frozen(args.base, config)
    rows, _ = common.snapshot()
    gpus = {int(gpu): uuid for gpu, uuid in config["runtime"]["gpu_uuids"].items()}
    if any(rows[gpu]["uuid"] != uuid for gpu, uuid in gpus.items()):
        raise ValueError("Scheduled GPU identities changed")
    with (args.output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / "controller-process.json").exists():
            raise FileExistsError("Do not submit or restart this batch twice")
        common.write(
            args.output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "freeze_sha256": common.sha(args.base / "freeze.json"),
            },
        )
        controller = Controller(
            config, args.base, args.output, RELATIVE, gpus, args.wandb_token_file
        )
        controller.output = args.output
        passed = controller.run()
        common.write(
            args.output / "controller-exit.json", {"code": 0 if passed else 1, "utc": common.utc()}
        )
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
