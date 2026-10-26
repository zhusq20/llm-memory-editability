#!/usr/bin/env python3
"""Freeze and queue six authorized LR runs on exclusively idle LM GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_memory_editability.lr_control import (  # noqa: E402
    BASE_SPEC,
    BATCH,
    INITIAL_SHA,
    PARAMETERS,
    decay_contract,
    validate,
)

DOCKER = ["docker", "--context", "lm-memory"]
STAGES = ("cuda-contracts", "preflight", "smoke", "smoke-audit", "train", "audit", "cloud-audit")
CUDA_TESTS = (
    "test_disabled_runner_matches_original_logits_loss_and_all_gradients "
    "or test_baseline_contract_rejects_different_initial_weights"
)
CPU_TESTS = (
    "not test_disabled_runner_matches_original_logits_loss_and_all_gradients "
    "and not test_baseline_contract_rejects_different_initial_weights"
)
SCRIPTS = (
    "run_lr_control.py",
    "launch_lr_control.py",
    "run_supervised_alignment.py",
    "run_discoloop_restore.py",
    "run_self_decode_alignment.py",
)
TESTS = (
    "test_lr_control.py",
    "test_supervised_alignment.py",
    "test_supervised_alignment_runner.py",
    "test_grokking_reproduction.py",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def docker(*args, context="lm-memory", check=True):
    return subprocess.run(
        ["docker", "--context", context, *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


def cpu_set(value):
    result = set()
    for part in value.split(","):
        if part:
            bounds = list(map(int, part.split("-")))
            result.update(range(bounds[0], bounds[-1] + 1))
    return result


def explicit_gpu_reservation(container, index, uuid):
    """An all-visible container grants access; actual CUDA PIDs establish use."""
    return any(
        {str(index), uuid} & set(request.get("DeviceIDs") or [])
        for request in container["HostConfig"].get("DeviceRequests") or []
    )


def resource_snapshot():
    containers = []
    for context in ("lm-memory", "default"):
        ids = docker("ps", "-q", context=context).stdout.split()
        if ids:
            rows = json.loads(docker("inspect", *ids, context=context).stdout)
            containers.extend({**row, "inspection_context": context} for row in rows)
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    gpus = {}
    for line in output.splitlines():
        index, uuid, used, free, total, utilization = (x.strip() for x in line.split(","))
        gpus[int(index)] = {
            "gpu": int(index),
            "uuid": uuid,
            "used_mib": int(used),
            "free_mib": int(free),
            "total_mib": int(total),
            "utilization": int(utilization),
        }
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    processes = [
        {"uuid": line.split(",")[0].strip(), "pid": int(line.split(",")[1])}
        for line in output.splitlines()
        if line.strip()
    ]
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    return {
        "utc": utc(),
        "gpus": gpus,
        "processes": processes,
        "containers": containers,
        "available_kib": available_kib,
    }


def admission(config, name, snapshot, gpu=None):
    runtime = config["runtime"]
    cpus = runtime["contracts_cpu_set"] if name == "contracts" else runtime["cpu_sets"][name]
    if not set(cpus) <= set(os.sched_getaffinity(0)):
        raise ValueError("Requested CPU set is outside host affinity")
    if snapshot["available_kib"] < (16 if name == "contracts" else runtime["memory_gib"]) * 1024**2:
        return None
    for item in snapshot["containers"]:
        pinning = item["HostConfig"].get("CpusetCpus", "")
        if pinning and set(cpus) & cpu_set(pinning):
            return None
    result = {
        "utc": utc(),
        "cpu_set": cpus,
        "available_kib": snapshot["available_kib"],
        "containers": [item["Name"] for item in snapshot["containers"]],
    }
    if gpu is None:
        return result
    row = snapshot["gpus"].get(gpu)
    if row is None or row["total_mib"] < runtime["min_total_gpu_mib"]:
        return None
    if (
        row["used_mib"] > runtime["gpu_memory_max_mib"]
        or row["free_mib"] < runtime["min_free_gpu_mib"]
        or row["utilization"] > runtime["gpu_utilization_max"]
        or any(p["uuid"] == row["uuid"] for p in snapshot["processes"])
        or any(explicit_gpu_reservation(c, gpu, row["uuid"]) for c in snapshot["containers"])
    ):
        return None
    return {**result, **row, "exclusive_idle_gpu": True, "preemption": False}


def freeze(config, batch, config_path):
    source, inputs = batch / "source", batch / "input-mount"
    if source.exists() or (batch / "controller-process.json").exists():
        raise FileExistsError("Existing batch: do not submit or freeze twice")
    source.mkdir()
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    relatives = [
        *(f"scripts/{name}" for name in SCRIPTS),
        *(f"tests/{name}" for name in TESTS),
        "pyproject.toml",
        "CONTRIBUTING.md",
        f"configs/{BATCH}.json",
        "configs/experiment-tracking-defaults.json",
        "configs/grokking-reproduction-development-v1.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
    ]
    for relative in relatives:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if sha(config_path) != sha(source / f"configs/{BATCH}.json"):
        raise ValueError("Submission config differs from frozen config")
    parent = batch.parent
    old_data = parent / config["input_sources"]["data"]
    for relative in (
        "data/complete.json",
        "data/world.npz",
        "data/panels.npz",
        "reference/gpt2/config.json",
    ):
        target = inputs / "data/discoloop-success-restore-v1/original" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_data / relative, target)
    initial = inputs / "initial/loop.pt"
    initial.parent.mkdir(parents=True)
    shutil.copy2(parent / config["input_sources"]["initial"], initial)
    references = {}
    for architecture, reuse in config["baseline_reuse"].items():
        old_batch = parent / reuse["batch"]
        root = old_batch / reuse["folder"]
        train, audit = read(root / "result-train.json"), read(root / "result-audit.json")
        cloud = read(root / "result-cloud-audit.json")
        old_config = read(old_batch / reuse["config"])
        if any(old_config["spec"][k] != v for k, v in BASE_SPEC.items()):
            raise ValueError("Reused 1e-4 base recipe differs")
        if (
            old_config["runtime"]["image"] != config["runtime"]["image"]
            or old_config["dropout_seed"] != config["dropout_seed"]
        ):
            raise ValueError("Reused baseline environment or dropout differs")
        checkpoint = root / reuse["checkpoint"]
        checkpoint_hash = sha(checkpoint)
        native_audit = read(root / reuse["training_folder"] / "audit.json")
        native_run = read(root / reuse["training_folder"] / "run.json")
        if (
            train["state"] != "complete"
            or train["step"] != 300000
            or train["budget"] != 300000
            or train["engineering_only"]
            or train["initial_model_sha256"] != INITIAL_SHA
            or train["parameters"] != PARAMETERS
            or not audit["passed"]
            or not cloud["passed"]
            or not native_audit["passed"]
            or checkpoint_hash != train["final_checkpoint_sha256"]
        ):
            raise ValueError("Reused baseline lacks a bound scientific endpoint/audit")
        if native_run["source_checkpoint_sha256"] != sha(initial):
            raise ValueError("Reused baseline original checkpoint bytes differ")
        old_runner = old_batch / "source/scripts/run_supervised_alignment.py"
        if sha(old_runner) != sha(source / "scripts/run_supervised_alignment.py"):
            raise ValueError("Original native training core differs")
        for module in (
            "grokking_reproduction.py",
            "grok_depth.py",
            "self_decode_alignment.py",
            "supervised_alignment.py",
            "bios_model.py",
        ):
            relative = Path("src/llm_memory_editability") / module
            if sha(old_batch / "source" / relative) != sha(source / relative):
                raise ValueError(f"Original model, stream or optimizer source differs: {module}")
        folder = batch / "baseline-reference" / architecture
        folder.mkdir(parents=True)
        for filename in (
            "result-train.json",
            "result-audit.json",
            "result-cloud-audit.json",
            "wandb-train.json",
        ):
            shutil.copy2(root / filename, folder / filename)
        for filename in ("run.json", "learning.json", "exposure.npz", "audit.json"):
            shutil.copy2(root / reuse["training_folder"] / filename, folder / filename)
        references[architecture] = {
            "training_restarted": False,
            "source": str(root),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "initial_checkpoint_sha256": sha(initial),
            "native_runner_sha256": sha(old_runner),
            "complete": train,
            "audit_passed": True,
            "cloud_verified": True,
            "wandb": read(root / "wandb-train.json"),
        }
    write(batch / "baseline-reuse.json", references)
    write(
        batch / "decay-contract.json",
        {name: decay_contract(job["lr"]) for name, job in config["jobs"].items()},
    )
    for folder in (source, inputs, batch / "baseline-reference"):
        for path in (folder, *folder.rglob("*")):
            path.chmod(0o755 if path.is_dir() else 0o644)
    files = [
        p
        for folder in (source, inputs, batch / "baseline-reference")
        for p in folder.rglob("*")
        if p.is_file()
    ]
    write(
        batch / "source-manifest.json",
        {
            "utc": utc(),
            "config_sha256": sha(config_path),
            "image": config["runtime"]["image"],
            "files": {str(p.relative_to(batch)): sha(p) for p in sorted(files)},
            "baseline_reuse_sha256": sha(batch / "baseline-reuse.json"),
            "decay_contract_sha256": sha(batch / "decay-contract.json"),
        },
    )


def verify(batch, *, inputs=True):
    manifest = read(batch / "source-manifest.json")
    for relative, expected in manifest["files"].items():
        if not inputs and not relative.startswith("source/"):
            continue
        if sha(batch / relative) != expected:
            raise ValueError(f"Frozen bytes changed: {relative}")
    if sha(batch / "baseline-reuse.json") != manifest["baseline_reuse_sha256"]:
        raise ValueError("Baseline reuse receipt changed")
    actual_image = json.loads(docker("image", "inspect", manifest["image"]).stdout)[0]["Id"]
    if actual_image != manifest["image"]:
        raise ValueError("Pinned image changed")


def command(config, batch, job, stage, token, row):
    name = f"{BATCH}-{job}-{stage}" if job != "contracts" else f"{BATCH}-contracts"
    cpus = row["cpu_set"]
    memory = 16 if job == "contracts" or stage == "cloud-audit" else config["runtime"]["memory_gib"]
    output = batch / ("contracts" if job == "contracts" else f"jobs/{job}")
    output.mkdir(parents=True, exist_ok=True)
    args = [
        *DOCKER,
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        str(len(cpus)),
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        f"{memory}g",
        "--memory-swap",
        f"{memory}g",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
    ]
    if "uuid" in row:
        args += ["--gpus", f"device={row['uuid']}"]
    for host, target, readonly in (
        (batch / "source", "/frozen", True),
        (batch / "input-mount", "/inputs", True),
        (batch / "baseline-reference", "/baselines", True),
        (output, "/output", False),
        (token.resolve(), "/run/secrets/wandb", True),
    ):
        args += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    for value in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "XDG_CACHE_HOME=/tmp/lm-lr-cache",
        "WANDB_CONFIG_DIR=/tmp/lm-lr-config",
        "WANDB_DISABLE_CODE=true",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
    ):
        args += ["--env", value]
    args += [
        config["runtime"]["image"],
        "bash",
        "-c",
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"\nexec "$@"',
        "lr-control-entry",
    ]
    if job == "contracts":
        args += [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *(f"/frozen/tests/{name}" for name in TESTS),
            "-k",
            CPU_TESTS,
        ]
    elif stage == "cuda-contracts":
        args += [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "/frozen/tests/test_supervised_alignment_runner.py",
            "-k",
            CUDA_TESTS,
        ]
    else:
        args += [
            "python",
            "/frozen/scripts/run_lr_control.py",
            stage,
            "--config",
            f"/frozen/configs/{BATCH}.json",
            "--job",
            job,
            "--out",
            "/output",
        ]
    return name, args


def execute(config, batch, job, stage, token, row, update):
    verify(batch, inputs=stage in ("preflight", "train", "audit"))
    folder = batch / "execution" / job / stage
    folder.mkdir(parents=True, exist_ok=False)
    name, args = command(config, batch, job, stage, token, row)
    write(folder / "claim.json", {"utc": utc(), "pid": os.getpid(), "container": name})
    write(folder / "command.json", {"command": args, "admission": row, "utc": utc()})
    launched = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
    docker_id = launched.stdout.strip()
    update(
        state="running",
        stage=stage,
        container=name,
        container_id=docker_id,
        gpu=row.get("gpu"),
        cpu_set=row["cpu_set"],
        utc=utc(),
    )
    started = time.monotonic()
    with (folder / "stdout.log").open("x") as log:
        follower = subprocess.Popen(
            [*DOCKER, "logs", "--follow", name], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while True:
                item = json.loads(docker("inspect", name).stdout)[0]
                state = item["State"]
                heartbeat = {
                    "container": name,
                    "running": state["Running"],
                    "elapsed_seconds": time.monotonic() - started,
                    "utc": utc(),
                }
                if job != "contracts":
                    path = batch / "jobs" / job / "training/status.json"
                    if path.exists():
                        heartbeat["training"] = read(path)
                    wandb = batch / "jobs" / job / "wandb-train.json"
                    if wandb.exists():
                        heartbeat["wandb"] = read(wandb)
                write(folder / "heartbeat.json", heartbeat)
                update(heartbeat=heartbeat)
                if not state["Running"]:
                    write(
                        folder / "exit.json",
                        {"code": state["ExitCode"], "state": state, "utc": utc()},
                    )
                    if state["ExitCode"] != 0:
                        raise RuntimeError(f"{name} exited {state['ExitCode']}; preserved")
                    return
                if time.monotonic() - started > config["runtime"]["timeout_hours"] * 3600:
                    write(
                        folder / "timeout-notice.json",
                        {
                            "utc": utc(),
                            "action": "stop only own named container after fixed hard timeout",
                        },
                    )
                    docker("stop", "--time", "30", name)
                    raise TimeoutError(f"Own container hard timeout: {name}")
                time.sleep(config["runtime"]["poll_seconds"])
        finally:
            try:
                follower.wait(timeout=45)
            except subprocess.TimeoutExpired:
                follower.terminate()  # Only the logging client, never an experiment.
                follower.wait(timeout=10)


def comparison(config, batch):
    baselines, panels = read(batch / "baseline-reuse.json"), {}
    for architecture in ("loop", "discoloop"):
        panels[architecture] = {"1e-4": baselines[architecture]["complete"]}
    for name, job in config["jobs"].items():
        root = batch / "jobs" / name
        train, audit, cloud = (
            read(root / f"result-{stage}.json") for stage in ("train", "audit", "cloud-audit")
        )
        if train["step"] != 300000 or not audit["passed"] or not cloud["passed"]:
            raise ValueError("All six fixed endpoints and independent/cloud audits required")
        panels[job["architecture"]][str(job["lr"])] = train
    write(
        batch / "comparison.json",
        {
            "utc": utc(),
            "panels": panels,
            "new_scientific_updates": 1800000,
            "new_engineering_updates": 192,
            "baseline_retrained": False,
            "worlds": 1,
            "initializations": 1,
            "unique_mechanism_established": False,
            "scope": "Six fixed 300k endpoints plus two reused controls; retain all seven pools "
            "and curves. Analytic AdamW shrinkage matched; data/FLOPs costs remain distinct.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    config = read(args.config)
    validate(config)
    if not args.execute:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    os.umask(0o022)
    root = Path(config["runtime"]["persistent_root"])
    if (
        not root.is_dir()
        or not root.resolve().is_relative_to("/ossfs/workspace")
        or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/"))
    ):
        raise ValueError("Actual persistent /ossfs mount required")
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Existing private W&B credential required")
    batch = root / BATCH
    if batch.is_symlink():
        raise ValueError("No redirected batch output")
    batch.mkdir(exist_ok=True)
    with (batch / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        required_space = config["runtime"]["minimum_persistent_free_gib"] * 1024**3
        if shutil.disk_usage(batch).free < required_space:
            raise RuntimeError("Insufficient persistent free space")
        freeze(config, batch, args.config)
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "config_sha256": sha(args.config),
                "source_manifest_sha256": sha(batch / "source-manifest.json"),
                "automatic_retry": False,
                "controller_source_sha256": sha(Path(__file__)),
            },
        )
        state_lock, resources_lock = threading.Lock(), threading.Lock()
        state = {
            "state": "submitted",
            "utc": utc(),
            "jobs": {
                name: {"state": "queued", "stage": "cuda-contracts"} for name in config["jobs"]
            },
        }

        def update(name, **values):
            with state_lock:
                if name == "contracts":
                    state["contracts"] = {**state.get("contracts", {}), **values}
                else:
                    state["jobs"][name].update(values)
                state["utc"] = utc()
                write(batch / "controller-state.json", state)
                write(batch / "science-schedule.json", state["jobs"])

        lock_root = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
        lock_root.mkdir(exist_ok=True)

        def wait_slot(name, stage):
            deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
            while time.monotonic() < deadline:
                with resources_lock:
                    snapshot = resource_snapshot()
                    if name == "contracts" or stage == "cloud-audit":
                        row = admission(config, name, snapshot)
                        if row is not None:
                            return None, row
                    else:
                        for gpu in config["runtime"]["gpu_preference"]:
                            held = (lock_root / f"gpu-{gpu}.lock").open("a")
                            try:
                                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                held.close()
                                continue
                            row = admission(config, name, snapshot, gpu)
                            if row is not None:
                                return held, row
                            held.close()
                    update(
                        name,
                        state="waiting_for_idle_resources",
                        stage=stage,
                        last_resource_check_utc=snapshot["utc"],
                        observed_gpu_memory={
                            str(k): v["used_mib"] for k, v in snapshot["gpus"].items()
                        },
                    )
                time.sleep(config["runtime"]["poll_seconds"])
            raise TimeoutError("Resource waiting timeout; existing tasks unchanged")

        def worker(name):
            completed = []
            try:
                for stage in STAGES:
                    held, row = wait_slot(name, stage)
                    try:
                        execute(
                            config,
                            batch,
                            name,
                            stage,
                            token,
                            row,
                            lambda **values: update(name, **values),
                        )
                    finally:
                        if held is not None:
                            held.close()
                    completed.append(stage)
                    update(name, completed_stages=completed, utc=utc())
                update(
                    name,
                    state="complete",
                    independent_audit_passed=True,
                    cloud_verified=True,
                    scientific_updates=300000,
                    utc=utc(),
                )
                return True
            except BaseException as error:
                update(
                    name,
                    state="needs_attention",
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    completed_stages=completed,
                    utc=utc(),
                )
                return False

        try:
            update("contracts", state="queued", stage="contracts")
            held, row = wait_slot("contracts", "contracts")
            execute(
                config,
                batch,
                "contracts",
                "contracts",
                token,
                row,
                lambda **values: update("contracts", **values),
            )
            update("contracts", state="passed", utc=utc())
            with state_lock:
                state["state"] = "running_or_waiting"
                write(batch / "controller-state.json", state)
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                outcomes = list(pool.map(worker, config["jobs"]))
            with state_lock:
                state["state"] = "complete" if all(outcomes) else "needs_attention"
                state["utc"] = utc()
                if all(outcomes):
                    comparison(config, batch)
                    write(batch / "completion-manifest.json", state)
                write(batch / "controller-state.json", state)
            if not all(outcomes):
                raise SystemExit(1)
        except BaseException as error:
            with state_lock:
                state.update(
                    state="needs_attention",
                    error=repr(error),
                    utc=utc(),
                    traceback=traceback.format_exc(),
                )
                write(batch / "controller-state.json", state)
            raise


if __name__ == "__main__":
    main()
