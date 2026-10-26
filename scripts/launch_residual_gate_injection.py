#!/usr/bin/env python3
"""Run the two authorized injection ablations with their ordinary CE recipe."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_gate_extension as reservations  # noqa: E402
import launch_paper_update_gate as common  # noqa: E402

CONFIG = "configs/residual-gate-injection-ablation-v1.json"


def verify_frozen(base, config):
    """Check the same immutable bytes concurrently on the network filesystem."""
    manifest = common.read(base / "freeze.json")
    if manifest["image"] != config["runtime"]["image"]:
        raise ValueError("Frozen runtime image differs from the selected configuration")

    def check(item):
        relative, expected = item
        if common.sha(base / relative) != expected:
            raise ValueError(f"Frozen artifact changed: {relative}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, manifest["files"].items()))
    if (
        json.loads(common.docker("image", "inspect", manifest["image"]).stdout)[0]["Id"]
        != (manifest["image"])
    ):
        raise ValueError("Docker image identity differs")


def validate_authorized(config):
    if config["batch"] != "residual-gate-injection-ablation-v1":
        raise ValueError("Unexpected batch")
    if config["training"]["steps"] != 1500000:
        raise ValueError("The authorized endpoint is 1500000 steps per arm")
    if list(config["arms"]) != ["only-u", "fixed-q-plus-u"]:
        raise ValueError("Only the two authorized ablations may run")
    for name, injection in (("only-u", "none"), ("fixed-q-plus-u", "fixed1")):
        arm = config["arms"][name]
        if (
            arm["injection"] != injection
            or arm["update_mode"] != "token"
            or arm["initial_update"] != 0.95
            or arm["unique_layers"] != 1
            or arm["repeats"] != 4
        ):
            raise ValueError(f"The authorized architecture changed: {name}")
    if config["runtime"]["docker_context"] != "lm-memory":
        raise ValueError("Only the LM Docker context is permitted")
    if config["runtime"]["tracking"] != "online":
        raise ValueError("Online tracking is required")


def verify_preflights(base, output, config):
    expected = common.sha(base / "source" / CONFIG)
    for name in config["arms"]:
        receipt = common.read(output / "preflight" / f"{name}.json")
        if receipt.get("passed") is not True or receipt.get("config_sha256") != expected:
            raise ValueError(f"The frozen configuration needs a passing CUDA check: {name}")


class Controller(common.Controller):
    def mirror_output(self, force=False):
        return None  # Output is already on persistent storage.

    def command(self, name, stage, gpu, uuid):
        label, command, cpus = super().command(name, stage, gpu, uuid)
        command[command.index("scripts/run_paper_update_gate.py")] = (
            "scripts/run_residual_gate_injection.py"
        )
        return label, command, cpus

    def worker(self, name, gpu, uuid):
        try:
            for stage in ("train", "grid"):
                started = time.monotonic()
                while True:
                    rows, blocked = common.snapshot()
                    if reservations.idle(gpu, uuid, rows, blocked):
                        break
                    if time.monotonic() - started > 24 * 3600:
                        raise TimeoutError(f"GPU {gpu} did not become independently idle")
                    self.update(
                        name,
                        state="waiting_for_idle_gpu",
                        stage=stage,
                        gpu=gpu,
                        resources=rows.get(gpu),
                        reservations=blocked.get(gpu, []),
                    )
                    time.sleep(15)
                self.execute(name, stage, gpu, uuid)
            self.update(name, state="training_and_grid_complete_audit_not_requested", grid=True)
        except BaseException as error:
            self.update(
                name, state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    config = common.read(base / "source" / CONFIG)
    validate_authorized(config)
    verify_frozen(base, config)
    verify_preflights(base, output, config)
    common.credential_archive(args.wandb_token_file)
    common.snapshot = reservations.snapshot
    common.verify_frozen = verify_frozen
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Preserve this submission; no automatic restart")
        rows, blocked = common.snapshot()
        gpus = common.admit(args.gpus, rows, blocked)
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu_uuids": gpus,
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / CONFIG),
                "scientific_arms": 2,
                "scientific_updates": 3000000,
                "stages": ["train", "grid"],
                "independent_audits_requested": False,
            },
        )
        controller = Controller(config, base, output, CONFIG, gpus, args.wandb_token_file)
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
