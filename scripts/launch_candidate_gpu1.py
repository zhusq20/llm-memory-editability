#!/usr/bin/env python3
"""User-requested third scheduling attempt for the unchanged candidate gate."""

from __future__ import annotations

import argparse
import fcntl
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import recover_candidate_update_gate as recovery  # noqa: E402

common, reservations = recovery.common, recovery.reservations
RELATIVE = recovery.launch.RELATIVE
REFERENCE = "configs/candidate-update-gate-previous-v2.json"


def validate_migration(config, previous):
    ignored = {"runtime", "scheduling_recovery"}
    if {k: v for k, v in config.items() if k not in ignored} != {
        k: v for k, v in previous.items() if k not in ignored
    }:
        raise ValueError("Only scheduling may change; preserve every scientific field")
    changed_runtime = {"gpu", "gpu_uuid", "cpu_first"}
    if {k: v for k, v in config["runtime"].items() if k not in changed_runtime} != {
        k: v for k, v in previous["runtime"].items() if k not in changed_runtime
    }:
        raise ValueError("Runtime image, precision, resources and monitoring must be preserved")
    rt = config["runtime"]
    if (rt["gpu"], rt["gpu_uuid"], rt["cpu_first"]) != (
        1,
        "GPU-364f0e00-7f5f-e6a7-9fea-03eb31c0bb3b",
        4,
    ):
        raise ValueError("Use exactly the authorized GPU1 and released CPU4/5")
    migration = config["scheduling_recovery"]
    if (
        migration["attempt"] != "gpu1-recovery-v3"
        or migration["maximum_concurrent_owned_gpus"] != 4
    ):
        raise ValueError("Do not widen GPU capacity or reuse a previous attempt")
    if migration["dependency"]["arm"] != "disco-l1-r4":
        raise ValueError("GPU1 predecessor must complete train and grid")


class Controller(recovery.Controller):
    def command(self, name, stage, gpu, uuid):
        label, command, cpus = super().command(name, stage, gpu, uuid)
        command[4:4] = ["--label", f"lm-memory.batch={self.config['batch']}"]
        return label, command, cpus

    def execute(self, name, stage, gpu, uuid):
        try:
            return super().execute(name, stage, gpu, uuid)
        except subprocess.CalledProcessError as error:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    config = common.read(base / "source" / RELATIVE)
    validate_migration(config, common.read(base / "source" / REFERENCE))
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    common.snapshot = reservations.snapshot
    rt = config["runtime"]
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Do not resubmit this migration attempt")
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu": 1,
                "gpu_uuid": rt["gpu_uuid"],
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / RELATIVE),
                "scheduling_recovery": config["scheduling_recovery"],
            },
        )
        controller = Controller(
            config, base, output, RELATIVE, {1: rt["gpu_uuid"]}, args.wandb_token_file
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
