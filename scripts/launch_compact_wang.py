#!/usr/bin/env python3
"""Run the authorized compact Wang arm exclusively on GPU2, then evaluate R1-4."""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_joint_update_wang as prior  # noqa: E402
from recover_candidate_update_gate import capacity_available, owned_gpus  # noqa: E402

common, reservations = prior.common, prior.reservations
RELATIVE = "configs/compact-wang-joint-gate-v1.json"


class Controller(prior.Controller):
    def command(self, name, stage, gpu, uuid):
        label, command, cpus = super().command(name, stage, gpu, uuid)
        command[command.index("scripts/run_joint_update_wang.py")] = "scripts/run_compact_wang.py"
        command[4:4] = ["--label", f"lm-memory.batch={self.config['batch']}"]
        return label, command, cpus

    def execute(self, name, stage, gpu, uuid):
        try:
            return super().execute(name, stage, gpu, uuid)
        except subprocess.CalledProcessError as error:
            # Container-create failures otherwise lose the useful daemon error.
            common.write(
                self.output / "failures" / f"{name}-{stage}-subprocess.json",
                {
                    "returncode": error.returncode,
                    "stderr": error.stderr.decode(errors="replace")
                    if isinstance(error.stderr, bytes)
                    else error.stderr,
                    "utc": common.utc(),
                },
            )
            raise

    def worker(self, name, gpu, uuid):
        for stage in ("preflight", "train", "grid"):
            deadline = time.monotonic() + 7 * 24 * 3600
            while True:
                rows, blocked = reservations.snapshot()
                current = owned_gpus()
                if reservations.idle(gpu, uuid, rows, blocked) and capacity_available(
                    current, uuid, 4
                ):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU2 resource wait exceeded seven days")
                self.update(
                    name,
                    state="waiting_for_gpu2",
                    stage=stage,
                    resources=rows.get(gpu),
                    reservations=blocked.get(gpu, []),
                    owned_gpu_count=len(current),
                )
                time.sleep(20)
            if stage == "train":
                receipt = common.read(self.output / "preflight" / f"{name}.json")
                if not receipt["passed"] or receipt["config_sha256"] != common.sha(
                    self.base / "source" / self.config_relative
                ):
                    raise ValueError("Frozen compact Wang CUDA engineering check required")
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
    if rt["gpu"] != 2 or rt["docker_context"] != "lm-memory":
        raise ValueError("User authorized only LM GPU2")
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    common.snapshot = reservations.snapshot
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Preserve previous submission; no duplicate or automatic retry")
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu": 2,
                "gpu_uuid": rt["gpu_uuid"],
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / RELATIVE),
            },
        )
        controller = Controller(
            config, base, output, RELATIVE, {2: rt["gpu_uuid"]}, args.wandb_token_file
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
