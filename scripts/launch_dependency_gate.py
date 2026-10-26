#!/usr/bin/env python3
"""Two persistent GPU lanes for the frozen sequence dependency evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_gate_extension as reservations  # noqa: E402
import launch_paper_update_gate as common  # noqa: E402
from launch_residual_gate_injection import verify_frozen  # noqa: E402
from recover_candidate_update_gate import owned_gpus  # noqa: E402

CONFIG = "configs/sequence-dependency-gate-v1.json"
ARMS = ("loop", "only-u", "disco", "fixed-q-plus-u")
TASKS = ("listops", "arithmetic")


def validate(config):
    if config["batch"] != "sequence-dependency-gate-v1":
        raise ValueError("Unexpected batch")
    if tuple(config["arms"]) != ARMS or tuple(config["tasks"]) != TASKS:
        raise ValueError("Preserve the two-task four-arm matrix")
    expected = (("none", "full"), ("none", "token"), ("fixed1", "full"), ("fixed1", "token"))
    for name, (injection, mode) in zip(ARMS, expected, strict=True):
        arm = config["arms"][name]
        if (arm["injection"], arm["update_mode"], arm["initial_update"]) != (injection, mode, 0.95):
            raise ValueError(f"Wrong gate definition: {name}")
    if any(config["tasks"][task]["training"]["steps"] != 100000 for task in TASKS):
        raise ValueError("This development matrix has fixed 100k endpoints")
    if config["budget"]["new_scientific_updates"] != 800000:
        raise ValueError("The total budget must remain 800k")
    if config["runtime"]["docker_context"] != "lm-memory":
        raise ValueError("Only the LM Docker context is allowed")
    if config["runtime"]["tracking"] != "online":
        raise ValueError("Online learning curves are required")


def lane_jobs(lane):
    if lane not in (0, 1):
        raise ValueError("Exactly two lanes")
    return [(task, ARMS[offset + lane]) for offset in (0, 2) for task in TASKS]


class JobController(common.Controller):
    def __init__(self, owner, task, arm, lane):
        config = {**owner.config, "batch": f"{owner.config['batch']}-{task}"}
        super().__init__(config, owner.base, owner.output, CONFIG, owner.gpus, owner.token_file)
        self.owner, self.task, self.arm, self.lane = owner, task, arm, lane
        self.output = owner.output / task
        self.output.mkdir(parents=True, exist_ok=True)
        self.cpus = self.cpus[2 * lane : 2 * lane + 2]

    def mirror_output(self, force=False):
        return None  # Output is already on the persistent volume.

    def update(self, name=None, **values):
        self.owner.update(f"{self.task}/{self.arm}", **values)

    def command(self, name, stage, gpu, uuid):
        label, args, _ = super().command(name, stage, gpu, uuid)
        args[args.index("--cpuset-cpus") + 1] = ",".join(map(str, self.cpus))
        old = f"type=bind,source={self.base / 'data'},target={common.DATA_TARGET},readonly"
        args[args.index(old)] = f"type=bind,source={self.base / 'data'},target=/inputs,readonly"
        runner_index = args.index("scripts/run_paper_update_gate.py")
        args[runner_index] = "scripts/run_dependency_gate.py"
        args.insert(runner_index + 1, "--stage")
        args += ["--task", self.task]
        return label, args, self.cpus


class Controller:
    def __init__(self, config, base, output, gpus, token_file, phase):
        self.config, self.base, self.output = config, base, output
        self.gpus, self.token_file, self.phase = gpus, token_file, phase
        self.mutex = threading.RLock()
        self.state = {
            "batch": config["batch"],
            "phase": phase,
            "state": "preparing",
            "jobs": {f"{task}/{arm}": {"state": "pending"} for task in TASKS for arm in ARMS},
        }

    def update(self, job=None, **values):
        with self.mutex:
            (self.state if job is None else self.state["jobs"][job]).update(values)
            self.state["updated_utc"] = common.utc()
            common.write(self.output / f"{self.phase}-state.json", self.state)

    def worker(self, lane, gpu, uuid):
        for task, arm in lane_jobs(lane):
            job = f"{task}/{arm}"
            current = JobController(self, task, arm, lane)
            try:
                stages = ("preflight",) if self.phase == "preflight" else ("train", "grid")
                for stage in stages:
                    started = time.monotonic()
                    while True:
                        rows, blocked = reservations.snapshot()
                        capacity = len(owned_gpus() | set(self.gpus.values())) <= 4
                        if reservations.idle(gpu, uuid, rows, blocked) and capacity:
                            break
                        if time.monotonic() - started > 168 * 3600:
                            raise TimeoutError("GPU availability/capacity wait exceeded seven days")
                        self.update(job, state="waiting_for_idle_gpu_and_capacity", stage=stage)
                        time.sleep(20)
                    current.execute(arm, stage, gpu, uuid)
                    if stage == "preflight":
                        verify_receipt(self.base, self.output, task, arm)
                self.update(job, state="complete", stages=list(stages))
            except BaseException as error:
                self.update(
                    job,
                    state="needs_attention",
                    error=repr(error),
                    traceback=traceback.format_exc(),
                )
                raise

    def run(self):
        self.update(state="running")
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(self.worker, lane, gpu, uuid)
                for lane, (gpu, uuid) in enumerate(self.gpus.items())
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    errors.append(repr(error))
        self.update(state="needs_attention" if errors else "complete", errors=errors)
        return not errors


def verify_receipt(base, output, task, arm):
    receipt = common.read(output / task / "preflight" / f"{arm}.json")
    if receipt.get("passed") is not True or receipt.get("config_sha256") != common.sha(
        base / "source" / CONFIG
    ):
        raise ValueError(f"Missing frozen full-size CUDA check: {task}/{arm}")


def verify_pairing(output):
    result = {}
    for task in TASKS:
        receipts = [common.read(output / task / "preflight" / f"{arm}.json") for arm in ARMS]
        for field in ("initial_backbone_sha256", "world_sha256", "panels_sha256"):
            if len({row[field] for row in receipts}) != 1:
                raise ValueError(f"Four arms do not share {field}: {task}")
        result[task] = {
            "initial_backbone_sha256": receipts[0]["initial_backbone_sha256"],
            "world_sha256": receipts[0]["world_sha256"],
            "paired_arms": list(ARMS),
        }
    common.write(output / "paired-preflight.json", {"passed": True, "tasks": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "science"), required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    config = common.read(base / "source" / CONFIG)
    validate(config)
    verify_frozen(base, config)
    if args.phase == "science":
        common.credential_archive(args.wandb_token_file)
        for task in TASKS:
            for arm in ARMS:
                verify_receipt(base, output, task, arm)
        verify_pairing(output)
    common.snapshot, common.verify_frozen = reservations.snapshot, verify_frozen
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = output / f"{args.phase}-process.json"
        if receipt.exists():
            raise FileExistsError("Preserve this phase; no automatic resubmission")
        rows, blocked = reservations.snapshot()
        gpus = common.admit(args.gpus, rows, blocked)
        if len(owned_gpus() | set(gpus.values())) > 4:
            raise ValueError("Starting these lanes would exceed the four-GPU ownership limit")
        common.write(
            receipt,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "argv": sys.argv,
                "utc": common.utc(),
                "gpu_uuids": gpus,
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / CONFIG),
                "scientific_updates": 800000 if args.phase == "science" else 0,
                "engineering_retained_updates": 0,
                "lanes": {gpu: lane_jobs(lane) for lane, gpu in enumerate(gpus)},
            },
        )
        controller = Controller(config, base, output, gpus, args.wandb_token_file, args.phase)
        try:
            passed = controller.run()
            if passed and args.phase == "preflight":
                verify_pairing(output)
            if passed and args.phase == "science":
                from report_dependency_gate import build_report

                build_report(base / "source" / CONFIG, output)
        except BaseException as error:
            controller.update(
                state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            common.write(
                output / f"{args.phase}-exit.json",
                {"code": 1, "utc": common.utc(), "error": repr(error)},
            )
            raise
        common.write(
            output / f"{args.phase}-exit.json", {"code": 0 if passed else 1, "utc": common.utc()}
        )
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
