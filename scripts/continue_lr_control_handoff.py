#!/usr/bin/env python3
"""Continue explicitly selected LR arms; never submit the six-arm launcher."""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True

WAITING = ("loop-lr5e-4", "loop-lr1e-3", "discoloop-lr2e-4", "discoloop-lr5e-4")
ACTIVE = ("loop-lr2e-4", "discoloop-lr1e-3")


def verify_runtime(module, batch, image, *, inputs=True):
    """Bind unchanged frozen files to an explicitly approved replacement image."""
    manifest = module.read(batch / "source-manifest.json")
    for relative, expected in manifest["files"].items():
        if not inputs and not relative.startswith("source/"):
            continue
        if module.sha(batch / relative) != expected:
            raise ValueError(f"Frozen bytes changed: {relative}")
    if module.sha(batch / "baseline-reuse.json") != manifest["baseline_reuse_sha256"]:
        raise ValueError("Baseline reuse receipt changed")
    if module.sha(batch / "decay-contract.json") != manifest["decay_contract_sha256"]:
        raise ValueError("Decay contract changed")
    actual = json.loads(module.docker("image", "inspect", image).stdout)[0]["Id"]
    if actual != image:
        raise ValueError("Replacement runtime must be pinned by its full image ID")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--job", choices=WAITING + ACTIVE, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpus", required=True, help="Four distinct CPU numbers, e.g. 4,5,6,7")
    parser.add_argument("--docker-context", default="lm-memory")
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--runtime-amendment", type=Path)
    parser.add_argument("--min-total-gpu-mib", type=int, default=90000)
    parser.add_argument("--min-free-gpu-mib", type=int, default=80000)
    parser.add_argument("--execute", action="store_true", help="Default only prints the plan")
    args = parser.parse_args()
    batch = args.batch.resolve()
    sys.path.insert(0, str(batch / "source/src"))
    # The scheduler needs only the pure-Python lr_control contract. Avoid the
    # package's unrelated eager NumPy imports on a minimal host Python install.
    package = types.ModuleType("llm_memory_editability")
    package.__path__ = [str(batch / "source/src/llm_memory_editability")]
    sys.modules["llm_memory_editability"] = package
    spec = importlib.util.spec_from_file_location(
        "frozen_lr_launcher", batch / "source/scripts/launch_lr_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.read(batch / "source/configs/loop-discoloop-lr-control-v1.json")
    module.validate(config)
    cpus = [int(value) for value in args.cpus.split(",")]
    if len(cpus) != 4 or len(set(cpus)) != 4:
        raise ValueError("Exactly four distinct CPUs required")
    # Scheduling changes only; the config mounted inside /frozen stays byte-identical.
    config = copy.deepcopy(config)
    if args.runtime_amendment:
        amendment = module.read(args.runtime_amendment)
        if (
            amendment["original_image"] != config["runtime"]["image"]
            or amendment["source_manifest_sha256"] != module.sha(batch / "source-manifest.json")
            or args.job not in amendment["jobs"]
            or not amendment["user_authorized_rebuilt_image"]
        ):
            raise ValueError("Runtime amendment does not bind this frozen batch/job")
        config["runtime"]["image"] = amendment["actual_image"]

        def verify(batch, *, inputs=True):
            verify_runtime(module, batch, amendment["actual_image"], inputs=inputs)

        module.verify = verify
    config["runtime"]["cpu_sets"][args.job] = cpus
    config["runtime"]["min_total_gpu_mib"] = args.min_total_gpu_mib
    config["runtime"]["min_free_gpu_mib"] = args.min_free_gpu_mib
    original_docker = module.docker

    def docker(*values, context=None, **kwargs):
        selected = args.docker_context if context in (None, "lm-memory") else context
        return original_docker(*values, context=selected, **kwargs)

    module.docker = docker
    module.DOCKER = ["docker", "--context", args.docker_context]
    original_command = module.command

    def command(*values):
        name, values = original_command(*values)
        values[values.index("--user") + 1] = f"{os.getuid()}:{os.getgid()}"
        if args.runtime_amendment:
            position = values.index(config["runtime"]["image"])
            values[position:position] = [
                "--env",
                f"WANDB_DOCKER={config['runtime']['image']}",
                "--env",
                f"WANDB_TAGS={amendment['host']},runtime-amended",
            ]
        return name, values

    module.command = command
    sharing_admission = None
    if args.job == ACTIVE[0] and args.gpu == 7:
        # Retain the user's existing GPU7 sharing authorization for its audits.
        # This exception never applies to any of the four transferred arms.
        scheduler_path = (
            batch / "resource-amendment-gpu7-v2/source/scripts/resume_lr_control_gpu7.py"
        )
        scheduler_spec = importlib.util.spec_from_file_location(
            "existing_gpu7_policy", scheduler_path
        )
        scheduler = importlib.util.module_from_spec(scheduler_spec)
        scheduler_spec.loader.exec_module(scheduler)
        sharing_admission = scheduler.admit
    stages = []
    for stage in module.STAGES:
        folder = batch / "execution" / args.job / stage
        if (folder / "exit.json").exists():
            if module.read(folder / "exit.json")["code"] != 0:
                raise RuntimeError(f"Failed stage preserved; no automatic retry: {folder}")
        else:
            stages.append(stage)
    print(
        json.dumps(
            {
                "job": args.job,
                "gpu": args.gpu,
                "cpus": cpus,
                "remaining_stages": stages,
                "execute": args.execute,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return
    token = args.wandb_token_file.resolve()
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Private W&B token file required (chmod 600)")
    if args.job in WAITING:
        if not (batch / "handoff/paused-queue.json").is_file():
            raise ValueError("Original queue must have an explicit pause/handoff receipt")
        if (batch / "execution" / args.job / "train").exists():
            raise ValueError("Transferred arm already has a train claim; refuse duplicate")
    module.verify(batch)
    lock = batch / "handoff" / f"{args.job}.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = batch / "handoff" / f"{args.job}-status.json"

        def update(**values):
            state = module.read(status_path) if status_path.exists() else {}
            state.update(values, utc=module.utc(), job=args.job)
            module.write(status_path, state)

        module.write(
            batch / "handoff" / f"{args.job}-process.json",
            {
                "pid": os.getpid(),
                "utc": module.utc(),
                "gpu": args.gpu,
                "cpus": cpus,
                "script_sha256": module.sha(Path(__file__)),
            },
        )
        for stage in stages:
            folder = batch / "execution" / args.job / stage
            if folder.exists():
                # Attach to the exact existing container; never relaunch a claimed stage.
                claim = module.read(folder / "claim.json")
                name = claim["container"]
                while True:
                    item = json.loads(module.docker("inspect", name).stdout)[0]
                    command_receipt = module.read(folder / "command.json")
                    if item["Config"]["Image"] != config["runtime"]["image"]:
                        raise ValueError("Existing container image differs")
                    if str(args.gpu) != str(command_receipt["admission"]["gpu"]):
                        raise ValueError("Attached container GPU differs")
                    update(state="running", stage=stage, container=name)
                    if not item["State"]["Running"]:
                        module.write(
                            folder / "exit.json",
                            {
                                "code": item["State"]["ExitCode"],
                                "state": item["State"],
                                "utc": module.utc(),
                            },
                        )
                        if item["State"]["ExitCode"] != 0:
                            raise RuntimeError(f"Existing stage failed: {name}")
                        break
                    time.sleep(30)
                continue
            # Hold the host GPU lease through each newly launched GPU stage.
            lease_path = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
            lease_path.mkdir(exist_ok=True)
            lease_name = (
                "lr-control-user-gpu7.lock"
                if sharing_admission is not None
                else f"gpu-{args.gpu}.lock"
            )
            with (lease_path / lease_name).open("a") as gpu_lock:
                deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
                while True:
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        row = None
                    else:
                        gpu = None if stage == "cloud-audit" else args.gpu
                        snapshot = module.resource_snapshot()
                        row = (
                            sharing_admission(module, config, args.job, snapshot, gpu)
                            if sharing_admission is not None
                            else module.admission(config, args.job, snapshot, gpu)
                        )
                        if row is None:
                            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    if row is not None:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Waiting for idle resources; no task was stopped")
                    update(state="waiting_for_idle_resources", stage=stage)
                    time.sleep(30)
                module.execute(config, batch, args.job, stage, token, row, update)
        update(
            state="complete",
            independent_audit_passed=True,
            cloud_verified=True,
            scientific_updates=300000,
        )


if __name__ == "__main__":
    main()
