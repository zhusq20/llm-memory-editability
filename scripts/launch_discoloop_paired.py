#!/usr/bin/env python3
"""Prepare successful-base TTCD, gate a common reader, then run paired arms.

Default invocation is read-only. Execution uses frozen source and data, explicit
LM Docker containers and exclusive idle GPU8/9 admission. Foundation and common
gates precede dependent training. Failures are retained without automatic retry.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["docker", "--context", "lm-memory"]
BATCH = "ttcd-successful-base-v1"
ARMS = ("single", "stream")
COMMON_STAGES = ("preflight", "train", "train-audit", "gate", "cloud-audit")
ARM_STAGES = ("train", "train-audit", "calibrate", "evaluate", "audit", "cloud-audit")
STAGES = (
    "prepare",
    "preflight",
    "train",
    "train-audit",
    "gate",
    "calibrate",
    "evaluate",
    "audit",
    "cloud-audit",
)
SOURCE_SCRIPTS = (
    "run_ttcd_verified_base.py",
    "run_supervised_alignment.py",
    "run_ttcd_second_hop.py",
    "prepare_ttcd_scratch.py",
    "launch_discoloop_paired.py",
)
GPU_DEPENDENCIES = {8: "single", 9: "stream"}
PROTOCOLS = (
    "docs/development-artifacts/discoloop-success-restore-v1/protocol.md",
    "docs/development-artifacts/ttcd-successful-base-v1/protocol.md",
)


class ResourceUnavailable(RuntimeError):
    """A busy resource can be waited for without resubmitting an experiment."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def docker(*args, check=True, context="lm-memory"):
    return subprocess.run(
        ["docker", "--context", context, *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=45,
    )


def inspect_container(name):
    result = docker("inspect", name, check=False)
    if result.returncode:
        error = result.stderr.lower()
        if "no such object" not in error and "no such container" not in error:
            raise RuntimeError(f"Cannot inspect container {name}: {result.stderr.strip()}")
        return None
    return json.loads(result.stdout)[0]


def parse_cpu_set(value):
    cpus = set()
    for part in value.split(","):
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus


def check_cpu_sets(task_cpus, containers, available_cpus):
    requested = set(task_cpus)
    if not requested <= set(available_cpus):
        raise RuntimeError("Requested CPUs are outside this process's available affinity")
    for container in containers:
        if not container["State"]["Running"]:
            continue
        value = container["HostConfig"].get("CpusetCpus", "")
        if not value:
            raise RuntimeError(f"Cannot verify CPU isolation for {container['Name']}")
        if requested & parse_cpu_set(value):
            raise ResourceUnavailable(
                f"CPU set collides with running container {container['Name']}"
            )


def running_containers(contexts):
    containers = []
    for context in contexts:
        ids = docker("ps", "--quiet", context=context).stdout.split()
        if ids:
            rows = json.loads(docker("inspect", *ids, context=context).stdout)
            containers.extend({**row, "InspectionContext": context} for row in rows)
    return containers


def gpu_snapshot():
    rows = {}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        index, uuid, used, free, total, utilization = (value.strip() for value in line.split(","))
        rows[int(index)] = {
            "uuid": uuid,
            "memory": int(used),
            "free_mib": int(free),
            "total_mib": int(total),
            "utilization": int(utilization),
        }
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    processes = []
    for line in result.stdout.splitlines():
        if line.strip():
            uuid, pid = (value.strip() for value in line.split(","))
            processes.append({"uuid": uuid, "pid": int(pid)})
    return rows, processes


def host_memory_available_kib():
    return next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def monitor_container(container_name, log, deadline, update=lambda _: None, poll_seconds=30):
    follower = subprocess.Popen(
        [*DOCKER, "logs", "--follow", container_name], stdout=log, stderr=log
    )
    waiter = subprocess.Popen(
        [*DOCKER, "wait", container_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                write_json(
                    Path(log.name).with_name("hard-timeout.json"),
                    {
                        "container": container_name,
                        "notice_utc": utc_now(),
                        "action": "stop own container",
                    },
                )
                print(f"Hard timeout: stopping own container {container_name}", flush=True)
                docker("stop", "--time", "30", container_name, check=False)
                raise TimeoutError(f"Hard timeout for own container {container_name}")
            try:
                stdout, stderr = waiter.communicate(timeout=min(poll_seconds, remaining))
                if waiter.returncode:
                    raise RuntimeError(f"Docker wait failed: {stderr.strip()}")
                return int(stdout.strip())
            except subprocess.TimeoutExpired:
                state = inspect_container(container_name)
                if state is None:
                    raise RuntimeError("Own monitored container disappeared") from None
                stats = docker("stats", "--no-stream", "--format", "{{json .}}", container_name)
                update(
                    {
                        "heartbeat_utc": utc_now(),
                        "container_running": state["State"]["Running"],
                        "container_resources": json.loads(stats.stdout)
                        if stats.stdout.strip()
                        else {},
                        "log_bytes": Path(log.name).stat().st_size,
                        "remaining_seconds": remaining,
                    }
                )
    finally:
        if waiter.poll() is None:
            waiter.terminate()  # Monitoring client only; never the experiment.
            waiter.communicate(timeout=10)
        try:
            follower.wait(timeout=45)
        except subprocess.TimeoutExpired:
            follower.terminate()
            follower.wait(timeout=10)


def container_uses_gpu(container, gpu, uuid):
    for request in container["HostConfig"].get("DeviceRequests") or []:
        if request.get("Count") == -1:
            return True
        if any(identifier in (str(gpu), uuid) for identifier in request.get("DeviceIDs") or []):
            return True
    return False


def validate_config(config):
    if config["batch"] != BATCH or set(config["models"]) != set(ARMS):
        raise ValueError("Only the authorized successful-base single/stream matrix may run")
    if config["common"]["steps"] != 10000 or config["training"]["steps"] != 10000:
        raise ValueError("Budgets are common10k plus two paired10k branches")
    if config["phi"] != 7.2 or config["continual"]["train_length"] != 8:
        raise ValueError("Preserve the successful Wang support and eight-fact training length")
    if (
        config["continual"]["panel_source_path"]
        != f"results/{BATCH}/preparation/continual-panel.json"
    ):
        raise ValueError("Preparation must publish its panel inside the new batch")
    if config["continual"]["panel_path"] != "/frozen/configs/continual-panel.json":
        raise ValueError("Science reads the independently frozen prepared panel")
    if Path(config["data_dir"]) != Path("data") / BATCH:
        raise ValueError("Prepared adaptation data must use its own new directory")
    if Path(config["model_config"]).parent != Path(config["data_dir"]):
        raise ValueError("The copied model config belongs in the new adaptation data directory")
    for key in ("original_data_dir", "original_model_config"):
        path = Path(config[key])
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parts[:2] != ("data", "discoloop-success-restore-v1")
        ):
            raise ValueError("Raw inputs must come from the independently restored foundation tree")
    for key in ("checkpoint", "gate_receipt"):
        path = Path(config["foundation"][key])
        if ".." in path.parts or not path.is_relative_to("results/discoloop-success-restore-v1"):
            raise ValueError("Foundation inputs must remain in their separate verified batch")
    if config["foundation"]["step"] != 300000:
        raise ValueError("The verified foundation is its fixed300k endpoint")
    for phase in ("common", *ARMS):
        section = config["common"] if phase == "common" else config["models"][phase]
        if Path(section["checkpoint"]) != Path("results") / BATCH / phase / "training/parent.pt":
            raise ValueError("Every phase needs its own isolated terminal checkpoint")
    runtime = config["runtime"]
    if runtime["gpus"]["single"] != 8 or runtime["gpus"]["stream"] != 9:
        raise ValueError("The paired arms are explicitly assigned GPU8 and GPU9")
    if runtime.get("shared_gpu_allowlist"):
        raise ValueError("Successful-base adaptation does not authorize GPU sharing")
    if set(runtime["cpu_sets"]["single"]) & set(runtime["cpu_sets"]["stream"]):
        raise ValueError("Parallel arm CPU sets must be disjoint")
    for phase in ("common", *ARMS):
        cpus = runtime["cpu_sets"][phase]
        if len(cpus) != runtime["cpu_threads"] or len(set(cpus)) != len(cpus):
            raise ValueError("Every phase needs a complete unique CPU set")
    if runtime.get("docker_context", "lm-memory") != "lm-memory":
        raise ValueError("Use only the explicit LM Docker context")
    if not runtime["image"].startswith("sha256:") or len(runtime["image"]) != 71:
        raise ValueError("Pin the exact reviewed dependency image")
    if (
        runtime["tracking"] != "online"
        or runtime.get("uid", 1015) != 1015
        or runtime.get("gid", 1016) != 1016
    ):
        raise ValueError("Formal LM stages require online tracking and uid1015:gid1016")
    if runtime.get("minimum_persistent_free_gib", 200) < 200:
        raise ValueError("Reserve at least200GiB before this state-retaining experiment")


def persistent_batch(config):
    root = Path(config["runtime"]["persistent_root"])
    if not root.is_dir() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise RuntimeError("The required persistent mount is unavailable")
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Formal LM outputs must remain below /ossfs/workspace")
    batch = root / BATCH
    if batch.is_symlink():
        raise ValueError("The new output batch must not be a symlink")
    return batch


def result_host_path(batch, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "results":
        raise ValueError("Result inputs must use reviewed project-relative results paths")
    return batch.parent / Path(*path.parts[1:])


def verify_foundation(batch, config):
    spec = config["foundation"]
    gate = result_host_path(batch, spec["gate_receipt"])
    checkpoint = result_host_path(batch, spec["checkpoint"])
    if not gate.is_file() or not checkpoint.is_file():
        raise ResourceUnavailable(
            "Waiting for the foundation's actual successful gate and checkpoint"
        )
    receipt = read_json(gate)
    if receipt.get("passed") is not True:
        raise RuntimeError(
            "The foundation gate failed; no common reader or write training is allowed"
        )
    digest = sha256(checkpoint)
    if receipt[spec.get("gate_checkpoint_field", "checkpoint_sha256")] != digest:
        raise ValueError("The foundation gate binds another checkpoint")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "gate": str(gate),
        "gate_sha256": sha256(gate),
        "passed": True,
    }


def raw_inputs(config):
    return [
        Path(config["original_data_dir"]) / name
        for name in ("world.npz", "panels.npz", "complete.json")
    ] + [Path(config["original_model_config"])]


def prepared_inputs(config):
    return [
        Path(config["data_dir"]) / name
        for name in (
            "world.npz",
            "panels.npz",
            "complete.json",
            "training-split.json",
            "ttcd-input-manifest.json",
        )
    ] + [Path(config["model_config"])]


def source_hashes(source):
    return {
        str(path.relative_to(source)): sha256(path)
        for path in sorted(source.rglob("*"))
        if path.is_file()
    }


def verify_files(source, entries):
    for relative, expected in entries.items():
        if sha256(source / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def freeze_preparation(batch, config_path, foundation):
    """Freeze science code before graph preparation; preserve that first manifest."""
    source, manifest = batch / "source", batch / "preparation-source-manifest.json"
    if manifest.exists():
        original = read_json(manifest)
        if original["config_sha256"] != sha256(config_path) or original["foundation"] != foundation:
            raise ValueError("Prepared configuration or foundation identity changed")
        verify_files(source, original["files"])
        if (batch / "source-manifest.json").exists():
            verify_files(source, read_json(batch / "source-manifest.json")["files"])
        return
    source.mkdir(exist_ok=False)
    for directory in ("src", "tests"):
        shutil.copytree(
            ROOT / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    (source / "scripts").mkdir()
    (source / "configs").mkdir()
    for name in SOURCE_SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, source / "scripts" / name)
    shutil.copy2(config_path, source / "configs/experiment.json")
    shutil.copy2(config_path, source / f"configs/{BATCH}.json")
    shutil.copy2(
        ROOT / "configs/experiment-tracking-defaults.json",
        source / "configs/experiment-tracking-defaults.json",
    )
    # Test fixture only; no science stage reads this historical template.
    shutil.copy2(
        ROOT / "configs/ttcd-second-hop-scratch-v1.json",
        source / "configs/ttcd-second-hop-scratch-v1.json",
    )
    for name in ("pyproject.toml", "CONTRIBUTING.md"):
        shutil.copy2(ROOT / name, source / name)
    for relative in PROTOCOLS:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for relative in raw_inputs(read_json(config_path)):
        target = source / "raw-input-backup" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    source.chmod(0o755)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    write_json(
        manifest,
        {
            "schema": "ttcd-verified-preparation-freeze-v1",
            "foundation": foundation,
            "config_sha256": sha256(config_path),
            "protocols": {
                relative: {
                    "sha256": sha256(source / relative),
                    "bytes": (source / relative).stat().st_size,
                }
                for relative in PROTOCOLS
            },
            "files": source_hashes(source),
        },
    )


def setup_raw_overlay(batch, config):
    root = batch / "input-mount"
    panel = result_host_path(batch, config["continual"]["panel_source_path"])
    panel.parent.mkdir(parents=True, exist_ok=True)
    panel.parent.chmod(0o755)
    for relative in raw_inputs(config):
        target, original = root / relative, batch / "source/raw-input-backup" / relative
        if target.is_symlink():
            raise ValueError("Input overlay files cannot be symlinks")
        target.parent.mkdir(parents=True, exist_ok=True)
        for directory in (root, *target.parent.parents, target.parent):
            if directory == root or directory.is_relative_to(root):
                if directory.is_symlink():
                    raise ValueError("Input overlay parents cannot be symlinks")
                directory.chmod(0o755)
        if target.exists():
            if sha256(target) != sha256(original):
                raise ValueError("Original input overlay bytes changed")
        else:
            shutil.copy2(original, target)
            target.chmod(0o644)
    return root / "data"


def freeze_prepared_inputs(batch, config):
    source, manifest = batch / "source", batch / "source-manifest.json"
    if manifest.exists():
        verify_files(source, read_json(manifest)["files"])
        verify_input_overlay(batch, config)
        return
    original = read_json(batch / "preparation-source-manifest.json")
    verify_files(source, original["files"])
    panel = result_host_path(batch, config["continual"]["panel_source_path"])
    shutil.copy2(panel, source / "configs/continual-panel.json")
    for relative in prepared_inputs(config):
        target = source / "prepared-input-backup" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(batch / "input-mount" / relative, target)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    write_json(
        manifest,
        {
            "schema": "ttcd-verified-science-freeze-v1",
            "preparation_manifest_sha256": sha256(batch / "preparation-source-manifest.json"),
            "config_sha256": original["config_sha256"],
            "foundation": original["foundation"],
            "protocols": original["protocols"],
            "inputs": {
                str(p): sha256(batch / "input-mount" / p)
                for p in raw_inputs(config) + prepared_inputs(config)
            },
            "panel_sha256": sha256(panel),
            "files": source_hashes(source),
        },
    )


def verify_input_overlay(batch, config):
    frozen = read_json(batch / "source-manifest.json")
    for relative, expected in frozen["inputs"].items():
        if sha256(batch / "input-mount" / relative) != expected:
            raise ValueError(f"Prepared input overlay changed: {relative}")
    if (
        sha256(result_host_path(batch, config["continual"]["panel_source_path"]))
        != frozen["panel_sha256"]
    ):
        raise ValueError("Prepared panel changed after scientific freeze")


def check_space(batch, config):
    usage = shutil.disk_usage(batch)
    required = config["runtime"].get("minimum_persistent_free_gib", 200) * 1024**3
    if usage.free < required:
        raise RuntimeError("Need at least200GiB persistent free space before this batch")
    write_json(
        batch / "space-admission.json",
        {"free_bytes": usage.free, "required_bytes": required, "utc": utc_now()},
    )


def admit_cpu(config, phase):
    containers = running_containers(["lm-memory"])
    check_cpu_sets(config["runtime"]["cpu_sets"][phase], containers, os.sched_getaffinity(0))
    available = host_memory_available_kib()
    if available < config["runtime"]["memory_gib"] * 1024**2:
        raise ResourceUnavailable("Insufficient host memory")
    return containers, available


def admit_gpu(batch, config, phase, gpu):
    dependency = batch.parent / "ttcd-cross-sample-v1" / GPU_DEPENDENCIES[gpu] / "audit.json"
    if not dependency.is_file() or read_json(dependency).get("passed") is not True:
        raise ResourceUnavailable(f"Waiting for GPU{gpu}'s corresponding prior science audit")
    containers, available = admit_cpu(config, phase)
    rows, processes = gpu_snapshot()
    if gpu not in rows:
        raise ValueError(f"GPU{gpu} is unavailable on this host")
    row = rows[gpu]
    if (
        row["memory"] > config["runtime"]["gpu_memory_max_mib"]
        or row["utilization"] > config["runtime"]["gpu_utilization_max"]
        or any(p["uuid"] == row["uuid"] for p in processes)
        or any(container_uses_gpu(c, gpu, row["uuid"]) for c in containers)
    ):
        raise ResourceUnavailable(f"GPU{gpu} occupied; sharing and preemption are forbidden")
    return {
        **row,
        "gpu": gpu,
        "phase": phase,
        "host_available_kib": available,
        "cpu_set": config["runtime"]["cpu_sets"][phase],
        "utc": utc_now(),
        "prior_science_audit_sha256": sha256(dependency),
        "gpu_sharing": False,
    }


def wait_gpu(batch, config, phase, deadline, update=lambda _: None):
    choices = [8, 9] if phase == "common" else [config["runtime"]["gpus"][phase]]
    directory = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    while time.monotonic() < deadline:
        reasons = {}
        for gpu in choices:
            held = (directory / f"gpu-{gpu}.lock").open("a")
            try:
                try:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ResourceUnavailable("Exclusive GPU lock held") from error
                return held, admit_gpu(batch, config, phase, gpu)
            except ResourceUnavailable as error:
                held.close()
                reasons[str(gpu)] = str(error)
            except BaseException:
                held.close()
                raise
        update({"state": "waiting_for_resources", "reasons": reasons})
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("Resource wait timeout; no existing task was stopped")


def build_command(config, phase, stage, batch, token_file, row=None):
    runtime, preparation = config["runtime"], stage == "prepare"
    out_phase = "preparation" if preparation else phase
    out = batch / out_phase
    name = f"{BATCH}-{out_phase}-{stage}"
    cpus = runtime["cpu_sets"]["common" if preparation else phase]
    command = [
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
        f"{runtime['memory_gib']}g",
        "--shm-size",
        "8g",
        "--network",
        "host",
        "--workdir",
        "/project",
    ]
    if not preparation:
        command += ["--gpus", f"device={row['uuid']}"]
    for host, target, readonly in [
        (ROOT, "/project", True),
        (batch.parent, "/project/results", True),
        (batch, f"/project/results/{BATCH}", not preparation),
        (batch / "source", "/frozen", True),
        (batch / "source", f"/project/results/{BATCH}/source", True),
        (batch / "input-mount/data", "/project/data", not preparation),
    ]:
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    if preparation:
        original = Path(config["original_data_dir"]).parent
        command += [
            "--mount",
            f"type=bind,source={batch / 'input-mount' / original},"
            f"target=/project/{original},readonly",
        ]
        command += [
            "--mount",
            f"type=bind,source={batch / 'input-mount' / original},"
            f"target=/project/results/{BATCH}/input-mount/{original},readonly",
        ]
    else:
        command += ["--mount", f"type=bind,source={out},target=/project/results/{BATCH}/{phase}"]
        command += [
            "--mount",
            f"type=bind,source={batch / 'input-mount'},"
            f"target=/project/results/{BATCH}/input-mount,readonly",
        ]
    command += [
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--env",
        "PYTHONPATH=/frozen/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "HOME=/tmp/lm-home",
        "--env",
        "XDG_CACHE_HOME=/tmp/lm-home/.cache",
        "--env",
        "XDG_CONFIG_HOME=/tmp/lm-home/.config",
        "--env",
        f"OMP_NUM_THREADS={len(cpus)}",
        "--env",
        f"MKL_NUM_THREADS={len(cpus)}",
    ]
    if token_file.is_file():
        command += [
            "--mount",
            f"type=bind,source={token_file.resolve()},target=/run/secrets/wandb,readonly",
        ]
    command += [
        runtime["image"],
        "bash",
        "-c",
        'mkdir -p "$HOME"\nif [ -f /run/secrets/wandb ]; then\n'
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"; fi\nexec "$@"',
        "ttcd-verified-entry",
        "python",
        "/frozen/scripts/run_ttcd_verified_base.py",
        stage,
        "--config",
        "/frozen/configs/experiment.json",
        "--model",
        phase,
        "--device",
        "cpu" if preparation else "cuda:0",
        "--out",
        f"/project/results/{BATCH}/{out_phase}",
        "--tracking",
        "online",
    ]
    return name, command


def run_stage(batch, config, phase, stage, token_file, deadline, row=None, update=lambda _: None):
    out = batch / ("preparation" if stage == "prepare" else phase)
    out.mkdir(exist_ok=True)
    out.chmod(0o755)
    with (out / f"claim-{stage}.json").open("x") as stream:
        json.dump({"stage": stage, "phase": phase, "pid": os.getpid(), "utc": utc_now()}, stream)
    name, command = build_command(config, phase, stage, batch, token_file, row)
    if inspect_container(name) is not None:
        raise FileExistsError("The stage container already exists; no automatic retry")
    write_json(out / f"command-{stage}.json", {"argv": command, "utc": utc_now()})
    write_json(out / f"admission-{stage}.json", row or {"device": "cpu", "phase": phase})
    update({"state": "launching", "stage": stage, "container": name})
    with (out / f"{stage}.log").open("x") as log:
        launched = subprocess.run(command, capture_output=True, text=True, timeout=60)
        log.write(launched.stderr)
        log.flush()
    if launched.returncode:
        write_json(
            out / f"exit-{stage}.json", {"returncode": launched.returncode, "launch_failed": True}
        )
        raise RuntimeError(f"Stage launch failed: {launched.stderr.strip()}")
    update(
        {
            "state": "running",
            "stage": stage,
            "container": name,
            "container_id": launched.stdout.strip(),
        }
    )
    with (out / f"{stage}.log").open("a") as log:
        code = monitor_container(name, log, deadline, update)
    write_json(
        out / f"exit-{stage}.json", {"returncode": code, "container": name, "utc": utc_now()}
    )
    if stage in ("gate", "calibrate") and code == 0:
        if read_json(out / f"result-{stage}.json").get("passed") is not True:
            return 3
    return code


def run_phase(batch, config, phase, stages, token_file):
    out = batch / phase
    out.mkdir(exist_ok=True)
    out.chmod(0o755)
    state = {"phase": phase, "stages": stages, "automatic_retry": False}

    def update(value):
        state.update(value)
        write_json(out / "controller-state.json", {**state, "updated_utc": utc_now()})

    try:
        with (out / "controller.lock").open("a") as phase_lock:
            fcntl.flock(phase_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if phase != "common":
                require_common_gate(batch, config)
            held, row = wait_gpu(
                batch,
                config,
                phase,
                time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600,
                update,
            )
            with held:
                deadline = time.monotonic() + config["runtime"]["timeout_hours"] * 3600
                for index, stage in enumerate(stages):
                    verify_input_overlay(batch, config)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Phase hard timeout before next stage")
                    if index:
                        row = admit_gpu(batch, config, phase, row["gpu"])
                    code = run_stage(batch, config, phase, stage, token_file, deadline, row, update)
                    if code:
                        update(
                            {
                                "state": "gate_failed"
                                if stage in ("gate", "calibrate")
                                else "needs_attention",
                                "stage": stage,
                                "returncode": code,
                            }
                        )
                        return code
            update({"state": "complete", "finished_utc": utc_now()})
            return 0
    except Exception:
        failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
        update(failure)
        write_json(out / "controller-failure.json", failure)
        return 1


def require_common_gate(batch, config):
    gate = result_host_path(batch, config["common"]["gate_receipt"])
    checkpoint = result_host_path(batch, config["common"]["checkpoint"])
    if not gate.is_file() or not checkpoint.is_file():
        raise ValueError("A complete passing common gate is required before paired updates")
    receipt = read_json(gate)
    if receipt.get("passed") is not True or receipt["checkpoint_sha256"] != sha256(checkpoint):
        raise ValueError("The common gate must bind this exact complete parent")


def execute_pipeline(batch, config, token_file):
    state = {"state": "running", "automatic_retry": False}

    def update(value):
        state.update(value)
        write_json(batch / "controller-state.json", {**state, "updated_utc": utc_now()})

    deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
    while True:
        try:
            admit_cpu(config, "common")
            break
        except ResourceUnavailable as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU preparation resource wait expired") from error
            update({"state": "waiting_for_preparation_resources", "reason": str(error)})
            time.sleep(min(30, max(0, deadline - time.monotonic())))
    code = run_stage(
        batch,
        config,
        "common",
        "prepare",
        token_file,
        time.monotonic() + config["runtime"]["timeout_hours"] * 3600,
        update=update,
    )
    if code:
        update({"state": "needs_attention", "stage": "prepare", "returncode": code})
        return code
    freeze_prepared_inputs(batch, config)
    code = run_phase(batch, config, "common", list(COMMON_STAGES), token_file)
    if code:
        update(
            {
                "state": "needs_attention",
                "phase": "common",
                "returncode": code,
                "paired_scientific_updates": 0,
            }
        )
        return code
    require_common_gate(batch, config)
    # run_phase's with-held scope has ended: neither common's GPU reservation
    # nor a predecessor foundation lock can leak into the paired workers.
    update({"state": "paired_running", "phase": "paired"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            phase: pool.submit(run_phase, batch, config, phase, list(ARM_STAGES), token_file)
            for phase in ARMS
        }
        exits = {phase: future.result() for phase, future in futures.items()}
    update(
        {
            "state": "complete" if not any(exits.values()) else "needs_attention",
            "exit_codes": exits,
            "finished_utc": utc_now(),
        }
    )
    return 1 if any(exits.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pipeline", *STAGES), nargs="?", default="pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--model", choices=("common", *ARMS), default="common")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    plan = {
        "batch": BATCH,
        "executed": args.execute,
        "docker_context": "lm-memory",
        "image": config["runtime"]["image"],
        "prepare_device": "cpu",
        "common_stages": list(COMMON_STAGES),
        "paired_stages": list(ARM_STAGES),
        "common_gpu_preference": [8, 9],
        "paired_gpus": {p: config["runtime"]["gpus"][p] for p in ARMS},
        "minimum_persistent_free_gib": config["runtime"].get("minimum_persistent_free_gib", 200),
        "foundation_gate_required": True,
        "common_gate_required_before_paired": True,
        "automatic_retry": False,
        "gpu_sharing": False,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not args.config.resolve().is_relative_to(ROOT):
        raise ValueError("The reviewed configuration must remain inside the project")
    if not args.wandb_token_file.is_file() or args.wandb_token_file.stat().st_mode & 0o077:
        raise ValueError("Provide the private local online W&B token file")
    batch = persistent_batch(config)
    batch.mkdir(parents=True, exist_ok=True)
    batch.chmod(0o755)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            batch / "controller-process.json", {"pid": os.getpid(), "utc": utc_now(), **plan}
        )
        try:
            docker("info")
            image = json.loads(docker("image", "inspect", config["runtime"]["image"]).stdout)[0][
                "Id"
            ]
            if image != config["runtime"]["image"]:
                raise ValueError("Selected dependency image differs from frozen contract")
            deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
            while True:
                try:
                    foundation = verify_foundation(batch, config)
                    break
                except ResourceUnavailable as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Foundation gate wait expired") from error
                    write_json(
                        batch / "controller-state.json",
                        {"state": "waiting_for_foundation", "reason": str(error), "utc": utc_now()},
                    )
                    time.sleep(min(30, max(0, deadline - time.monotonic())))
            check_space(batch, config)
            freeze_preparation(batch, args.config.resolve(), foundation)
            setup_raw_overlay(batch, config)
            if args.stage == "pipeline":
                code = execute_pipeline(batch, config, args.wandb_token_file)
            elif args.stage == "prepare":
                code = run_stage(
                    batch,
                    config,
                    "common",
                    "prepare",
                    args.wandb_token_file,
                    time.monotonic() + config["runtime"]["timeout_hours"] * 3600,
                )
                if not code:
                    freeze_prepared_inputs(batch, config)
            else:
                verify_input_overlay(batch, config)
                code = run_phase(batch, config, args.model, [args.stage], args.wandb_token_file)
            if code:
                raise SystemExit(code)
        except Exception:
            failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
            write_json(batch / "controller-state.json", failure)
            write_json(batch / "controller-failure.json", failure)
            raise


if __name__ == "__main__":
    main()
