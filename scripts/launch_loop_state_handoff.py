#!/usr/bin/env python3
"""Freeze and queue one raw-state handoff experiment without taking occupied GPUs."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from launch_loop_transfer import blockers, credential_archive, snapshot  # noqa: E402

BATCH = "loop-state-handoff-development-v1"
CONFIG = f"configs/{BATCH}.json"
DOCKER = ["docker", "--context", "lm-memory"]
STAGES = ("environment", "smoke", "smoke-grid", "train", "grid", "cloud")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(config):
    if config["batch"] != BATCH or not config["status"].startswith("authorized"):
        raise ValueError("Only the authorized new batch may be submitted")
    if config["arms"] != {"state-mlp": {"handoff": "state-mlp"}}:
        raise ValueError("This authorization contains exactly one new state-mlp arm")
    if config["training"]["steps"] != 1500000 or config["smoke_steps"] != 32:
        raise ValueError("The scientific and engineering budgets are fixed")
    if config["runtime"]["docker_context"] != "lm-memory":
        raise ValueError("Use the LM Docker context")
    if config["tracking"]["entity"] != "zsqzz" or config["tracking"]["mode"] != "online":
        raise ValueError("Online tracking to the configured Academic Team is required")


def freeze(base):
    config = json.loads((ROOT / CONFIG).read_text())
    validate(config)
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]
    files += [
        ROOT / "pyproject.toml",
        ROOT / CONFIG,
        ROOT / "configs/loop-native-handoff-development-v1.json",
        ROOT / "configs/loop-position-invariance-development-v1.json",
        ROOT / f"docs/development-artifacts/{BATCH}/protocol.md",
        ROOT / "tests/test_loop_state_handoff.py",
        ROOT / "tests/test_loop_state_runner.py",
        ROOT / "tests/test_loop_state_launcher.py",
        ROOT / "tests/test_loop_native_handoff.py",
        ROOT / "tests/test_loop_position_invariance.py",
    ]
    frozen = {}
    for path in sorted(set(files)):
        target = source / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        target.chmod(0o644)
        frozen[str(target.relative_to(base))] = digest(target)
    inputs = {}
    for name in ("world.npz", "panels.npz", "complete.json"):
        path = Path(config["input_root"]) / name
        inputs[str(path)] = digest(path)
    for name, field in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        if inputs[str(Path(config["input_root"]) / name)] != config["input_origin"][field]:
            raise ValueError(f"Original paired data changed: {name}")
    comparators = {}
    for name, item in config["reused_comparators"].items():
        run = Path(item["run"])
        complete_path = run / "complete.json"
        complete = json.loads(complete_path.read_text())
        checkpoint = run / f"checkpoint-{complete['step']:07d}.pt"
        if complete["step"] != 1500000 or digest(checkpoint) != complete["final_checkpoint_sha256"]:
            raise ValueError(f"Comparator endpoint is incomplete or changed: {name}")
        comparators[name] = {
            "complete": str(complete_path),
            "complete_sha256": digest(complete_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "new_training_updates": 0,
        }
    (base / "artifacts").mkdir()
    (base / "engineering").mkdir()
    for path in [base, *base.rglob("*")]:
        if path.is_dir():
            path.chmod(0o755)
    write(
        base / "freeze.json",
        {
            "batch": BATCH,
            "utc": utc(),
            "image": config["runtime"]["image"],
            "files": frozen,
            "inputs": inputs,
            "reused_comparators": comparators,
            "training_updates": 1500000,
        },
    )
    return {"base": str(base), "files": len(frozen), "scientific_updates": 1500000}


def verify_freeze(base):
    manifest = json.loads((base / "freeze.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(base / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in manifest["inputs"].items():
        if digest(name) != expected:
            raise ValueError(f"Frozen data changed: {name}")
    return manifest


def priority_reasons(runtime, gpu):
    if gpu not in runtime["priority_gpu_ids"]:
        return []
    path = Path(runtime["priority_controller_state"])
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return ["Existing priority queue state is unavailable"]
    if state.get("state") != "complete":
        return ["GPU remains reserved for the existing loop transfer queue"]
    if any(job.get("state") != "complete" for job in state.get("jobs", {}).values()):
        return ["Existing loop transfer jobs are not all complete"]
    return []


def available(snap, runtime, gpu):
    if gpu not in snap["rows"]:
        return ["GPU is unavailable"]
    # No reservation overrides: created containers and external CUDA PIDs also block.
    return priority_reasons(runtime, gpu) + blockers(snap, gpu, [])


def stage_command(config, base, stage, gpu, uuid):
    runtime = config["runtime"]
    label = f"lsh-1022-state-mlp-{stage}"
    boot = "import os,sys;os.umask(0o022);"
    if stage in ("train", "cloud"):
        boot += (
            "from pathlib import Path;"
            "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
        )
    boot += "os.execvp(sys.argv[1],sys.argv[1:])"
    argv = [
        *DOCKER,
        "create",
        "--name",
        label,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpuset-cpus",
        runtime["cpu_set"],
        "--cpus",
        "2",
        "--memory",
        f"{runtime['memory_gib']}g",
        "--memory-swap",
        f"{runtime['memory_gib']}g",
        "--shm-size",
        "2g",
        "--tmpfs",
        "/tmp:rw,size=4g",
        "--network",
        "host",
        "--workdir",
        "/project",
    ]
    if stage != "cloud":
        argv += ["--gpus", f"device={uuid}"]
    for key, value in {
        "PYTHONPATH": "/project/src:/project/scripts",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": "/tmp/cache",
        "WANDB_CACHE_DIR": "/tmp/wandb-cache",
        "WANDB_CONSOLE": "off",
        "WANDB_SILENT": "true",
        "WANDB_DISABLE_CODE": "true",
        "WANDB_DOCKER": runtime["image"],
    }.items():
        argv += ["--env", f"{key}={value}"]
    for src, dst, suffix in [
        (base / "source", "/project", ",readonly"),
        (runtime["persistent_root"], runtime["persistent_root"], ",readonly"),
        (base / "artifacts", "/output", ""),
    ]:
        argv += ["--mount", f"type=bind,source={src},target={dst}{suffix}"]
    argv += [
        runtime["image"],
        "python",
        "-c",
        boot,
        "python",
        "-u",
        "scripts/run_loop_state_handoff.py",
        stage,
        "--config",
        CONFIG,
        "--base",
        "/output",
        "--arm",
        "state-mlp",
    ]
    return label, argv


class Controller:
    def __init__(self, base, token):
        self.base, self.token = base, token
        self.config = json.loads((base / "source" / CONFIG).read_text())
        validate(self.config)
        self.state = {"batch": BATCH, "state": "submitted", "stage": "environment"}

    def update(self, **values):
        self.state.update(values, updated_utc=utc())
        write(self.base / "controller-state.json", self.state)

    def admit(self, selected=None):
        runtime = self.config["runtime"]
        while True:
            snap = snapshot()
            reasons = {}
            for gpu in runtime["gpu_candidates"] if selected is None else [selected]:
                reasons[gpu] = available(snap, runtime, gpu)
                if not reasons[gpu]:
                    write(self.base / "resource-admission" / f"{self.state['stage']}.json", snap)
                    return gpu, snap["rows"][gpu]["uuid"]
            self.update(state="waiting_resource", waiting_reasons=reasons)
            time.sleep(runtime["poll_seconds"])

    def execute(self, stage, gpu=None, uuid=None):
        folder = self.base / "execution" / stage
        folder.mkdir(parents=True, exist_ok=False)
        label, argv = stage_command(self.config, self.base, stage, gpu, uuid)
        write(folder / "command.json", {"argv": argv, "gpu": gpu, "utc": utc()})
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120)
        write(folder / "container.json", {"id": result.stdout.strip(), "name": label})
        if stage in ("train", "cloud"):
            subprocess.run(
                [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
                input=credential_archive(self.token),
                check=True,
                capture_output=True,
                timeout=120,
            )
        subprocess.run([*DOCKER, "start", label], check=True, capture_output=True, timeout=120)
        self.update(state="running", stage=stage, gpu=gpu, gpu_uuid=uuid, container=label)
        with (folder / "stdout.log").open("w") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            result = subprocess.run(
                [*DOCKER, "wait", label], check=True, capture_output=True, text=True
            )
            follower.wait(timeout=60)
        code = int(result.stdout.strip())
        write(folder / "exit.json", {"code": code, "utc": utc()})
        if code:
            raise RuntimeError(f"{label} exited {code}; no automatic retry")

    def run(self):
        with (self.base / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (self.base / "controller-process.json").exists():
                raise FileExistsError("Already submitted; inspect receipts instead of resubmitting")
            verify_freeze(self.base)
            checks = json.loads((self.base / "engineering/cpu-checks.json").read_text())
            if checks.get("passed") is not True:
                raise ValueError("CPU contracts and lint must pass before queue submission")
            if (
                checks.get("freeze_sha256") != digest(self.base / "freeze.json")
                or checks.get("image") != self.config["runtime"]["image"]
            ):
                raise ValueError("CPU verification must cover this exact frozen source and image")
            image = subprocess.run(
                [
                    *DOCKER,
                    "image",
                    "inspect",
                    self.config["runtime"]["image"],
                    "--format",
                    "{{.Id}}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
            if image != self.config["runtime"]["image"]:
                raise ValueError("Fixed runtime image changed")
            credential_archive(self.token)
            write(
                self.base / "controller-process.json",
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "utc": utc(),
                    "controller_sha256": digest(__file__),
                    "freeze_sha256": digest(self.base / "freeze.json"),
                    "config_sha256": digest(self.base / "source" / CONFIG),
                },
            )
            self.update()
            gpu = uuid = None
            try:
                for stage in STAGES:
                    self.update(stage=stage)
                    if stage != "cloud":
                        selected, found_uuid = self.admit(gpu)
                        if uuid is not None and found_uuid != uuid:
                            raise RuntimeError("GPU UUID changed between stages")
                        gpu, uuid = selected, found_uuid
                    verify_freeze(self.base)
                    self.execute(stage, gpu, uuid)
                self.update(state="complete", scientific_updates=1500000)
            except BaseException as error:
                self.update(state="failed", error=repr(error))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    if args.action == "freeze":
        print(json.dumps(freeze(base)))
    else:
        if args.wandb_token_file is None:
            parser.error("run requires --wandb-token-file")
        Controller(base, args.wandb_token_file).run()


if __name__ == "__main__":
    os.umask(0o022)
    main()
