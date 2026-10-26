#!/usr/bin/env python3
"""Run four authorized checkpoint continuations in two exclusive GPU lanes.

Each lane runs one training container followed by its endpoint grid, then the
second arm. Failed stages are retained and never retried; the other lane remains
independent. The inherited executor checks both Docker contexts for reservations
before every stage and only creates containers in the LM Docker context.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import launch_paper_update_gate as original  # noqa: E402

BATCH = "paper-update-gate-continuation-v1"
ARMS = ("paper", "fixed-update", "token-update", "token-update-half-lr")
LANES = (("token-update", "paper"), ("token-update-half-lr", "fixed-update"))
RUNNER = "scripts/run_paper_update_continuation.py"


def validate_config(config):
    """Check this continuation's scope without importing the training stack."""
    if config["batch"] != BATCH or tuple(config["arms"]) != ARMS:
        raise ValueError("Exactly the four ordered continuation arms are authorized")
    if config["training"]["steps"] != 1500000:
        raise ValueError("All arms must end at 1500000 total updates")
    if config["continuation"]["start_step"] != 300000:
        raise ValueError("All arms must continue their 300000-update checkpoint")
    if config["budget"]["new_scientific_updates"] != 4800000:
        raise ValueError("Exactly 4800000 additional scientific updates are authorized")
    if config["training"]["lr"] != 1e-4:
        raise ValueError("The backbone learning rate must remain 1e-4")
    runtime = config["runtime"]
    if runtime["docker_context"] != "lm-memory" or runtime["tracking"] != "online":
        raise ValueError("LM Docker and online tracking are required")
    if runtime.get("maximum_parallel", 2) != 2:
        raise ValueError("Exactly two independently idle GPU lanes are authorized")
    if runtime["cpu_threads"] != 2 or runtime["memory_gib"] != 16:
        raise ValueError("The frozen two-CPU, 16-GiB container allocation must remain")
    if runtime["gpu_memory_fraction"] != 0.16:
        raise ValueError("The existing GPU memory allocation must remain unchanged")
    nodes = config["evaluation_nodes"]
    if not nodes or nodes != sorted(set(nodes)) or nodes[-1] != 1500000:
        raise ValueError("Evaluation nodes must be unique, increasing, and end at 1500000")
    if any(node < 300000 or node > 1500000 for node in nodes):
        raise ValueError("Continuation evaluation nodes must be within the resumed budget")


class Controller(original.Controller):
    def __init__(self, config, base, mirror, config_relative, gpus, token_file):
        validate_config(config)
        if len(gpus) != 2 or len(set(gpus.values())) != 2:
            raise ValueError("Two distinct admitted GPUs are required")
        super().__init__(config, base, mirror, config_relative, gpus, token_file, False)
        self.lane_gpus = tuple(gpus)
        self.state.update(
            state="preparing_checkpoint_continuation",
            continuation_start_step=300000,
            target_step=1500000,
            new_scientific_updates=4800000,
            lanes=[
                {"gpu": gpu, "gpu_uuid": gpus[gpu], "arms": list(names)}
                for gpu, names in zip(self.lane_gpus, LANES, strict=True)
            ],
        )
        for index, names in enumerate(LANES):
            for name in names:
                self.state["arms"][name].update(lane=index, gpu=self.lane_gpus[index])

    def command(self, name, stage, gpu, uuid):
        if stage not in ("train", "grid"):
            raise ValueError("Only training and endpoint-grid stages are authorized")
        lane = next(index for index, names in enumerate(LANES) if name in names)
        if gpu != self.lane_gpus[lane] or uuid != self.gpus[gpu]:
            raise ValueError("An arm cannot move to another lane or GPU")
        label, args, _ = super().command(name, stage, gpu, uuid)
        cpus = self.cpus[2 * lane : 2 * lane + 2]
        args[args.index("--cpuset-cpus") + 1] = ",".join(map(str, cpus))
        old_mount = f"type=bind,source={self.base / 'data'},target={original.DATA_TARGET},readonly"
        args[args.index(old_mount)] = (
            f"type=bind,source={self.base / 'data'},target=/inputs,readonly"
        )
        runner_index = args.index("scripts/run_paper_update_gate.py")
        args[runner_index:] = [
            RUNNER,
            "--stage",
            stage,
            "--config",
            self.config_relative,
            "--arm",
            name,
            "--output",
            "/output",
        ]
        return label, args, cpus

    def lane(self, names, gpu):
        errors = []
        for name in names:
            try:
                self.worker(name, gpu, self.gpus[gpu])
            except BaseException as error:
                # execute() will reject any still-running container reservation
                # before the next arm. No own-prefix exemption hides GPU use.
                errors.append({"arm": name, "gpu": gpu, "error": repr(error)})
            finally:
                self.mirror_output(force=True)
        return errors

    def run(self):
        self.update(state="scientific_continuation")
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {
                pool.submit(self.lane, names, gpu): gpu
                for names, gpu in zip(LANES, self.lane_gpus, strict=True)
            }
            for future in concurrent.futures.as_completed(jobs):
                try:
                    errors.extend(future.result())
                except BaseException as error:
                    errors.append(
                        {
                            "gpu": jobs[future],
                            "error": repr(error),
                            "traceback": traceback.format_exc(),
                        }
                    )
        self.update(
            state=(
                "needs_attention"
                if errors
                else "training_and_grid_complete_independent_audit_not_requested"
            ),
            errors=errors,
            grid_evaluated=not errors,
        )
        self.mirror_output(force=True)
        return not errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--config", default=f"configs/{BATCH}.json")
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    config_path = (base / "source" / args.config).resolve()
    relative = config_path.relative_to(base / "source")
    config = original.read(config_path)
    validate_config(config)
    lanes = [{"gpu": gpu, "arms": list(names)} for gpu, names in zip(args.gpus, LANES, strict=True)]
    if not args.execute:
        print(
            json.dumps(
                {
                    "batch": config["batch"],
                    "arms": list(config["arms"]),
                    "lanes": lanes,
                    "stages": ["train", "grid"],
                    "start_step": 300000,
                    "target_step": 1500000,
                    "new_scientific_updates": 4800000,
                }
            )
        )
        return
    original.verify_frozen(base, config)
    original.credential_archive(args.wandb_token_file)
    output = base / "output"
    output.mkdir(exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Preserve this controller attempt; no automatic restart")
        rows, blocked = original.snapshot()
        gpus = original.admit(args.gpus, rows, blocked)
        controller = Controller(config, base, args.mirror, relative, gpus, args.wandb_token_file)
        args.mirror.mkdir(parents=True, exist_ok=True)
        source_archive = original.archive_source(base, args.mirror)
        original.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": original.utc(),
                "freeze_sha256": original.sha(base / "freeze.json"),
                "config_sha256": original.sha(config_path),
                "gpu_uuids": gpus,
                "lanes": lanes,
                "direct_training_user_requested": True,
                "independent_audits_requested": False,
                "source_archive": source_archive,
                "base": str(base),
                "mirror": str(args.mirror.resolve()),
                "argv": sys.argv,
            },
        )
        original.write(args.mirror / "freeze.json", original.read(base / "freeze.json"))
        try:
            passed = controller.run()
        except BaseException as error:
            controller.update(
                state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            original.write(
                output / "controller-exit.json",
                {"code": 1, "utc": original.utc(), "error": repr(error)},
            )
            controller.mirror_output(force=True)
            raise
        original.write(
            output / "controller-exit.json", {"code": 0 if passed else 1, "utc": original.utc()}
        )
        controller.mirror_output(force=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
