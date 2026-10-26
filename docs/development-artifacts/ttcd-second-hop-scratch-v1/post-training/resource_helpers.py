#!/usr/bin/env python3
"""Run the paired TTCD cross-sample arms in isolated LM containers.

The default only prints a plan. Explicit --execute freezes the reviewed source,
verifies the shared original DiscoLoop 20k checkpoint, and starts each arm on
GPU, or shares GPU8/9 only with the explicitly authorized original containers.
A failed stage is preserved and is never automatically retried.
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
ARMS = ("single", "stream")
ALLOWED_SHARED = {
    "single": ["ttcd-second-hop-scratch-v1-discoloop-train-restart1"],
    "stream": ["ttcd-second-hop-scratch-v1-loop-train"],
}
STAGES = ("preflight", "train", "train-audit", "calibrate", "evaluate", "audit", "cloud-audit")
INPUTS = ("world.npz", "panels.npz", "complete.json", "training-split.json")
SOURCE_SCRIPTS = (
    "run_ttcd_continual.py",
    "run_ttcd_second_hop.py",
    "prepare_ttcd_scratch.py",
    "launch_ttcd_continual.py",
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


def validate_config(config):
    lineage = config["parent"]
    if config["batch"] != "ttcd-cross-sample-v1":
        raise ValueError("This controller is scoped to the authorized cross-sample batch")
    if lineage["parent_batch"] != "ttcd-second-hop-scratch-v1":
        raise ValueError("Only the original scratch batch may supply the parent")
    if lineage.get("parent_model", "discoloop") != "discoloop":
        raise ValueError("Both paired arms must inherit the same DiscoLoop parent")
    if lineage["checkpoint_step"] != 20000 or lineage["steps"] != 10000:
        raise ValueError("Authorized arms are exactly step 20000 plus 10000 updates")
    if lineage["parent_source_dir"] != "/parent-frozen":
        raise ValueError("Unexpected parent source mount")
    if lineage["parent_manifest_path"] != "/parent-source-manifest.json":
        raise ValueError("Unexpected parent manifest mount")
    for field in ("parent_manifest_sha256", "parent_config_sha256", "checkpoint_sha256"):
        value = lineage[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"A fixed SHA256 is required for {field}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"Invalid SHA256 for {field}") from error
    if not lineage["parent_image_sha256"].startswith("sha256:"):
        raise ValueError("The parent image must be pinned by image ID")
    expected = f"results/{lineage['parent_batch']}/discoloop/training/checkpoint-0020000.pt"
    if set(config["models"]) != set(ARMS):
        raise ValueError("The complete authorized matrix is single and stream")
    runtime = config["runtime"]
    if set(runtime["gpus"]) != set(ARMS) or set(runtime["cpu_sets"]) != set(ARMS):
        raise ValueError("Both arms need explicit GPU and CPU assignments")
    if len(set(runtime["gpus"].values())) != len(ARMS):
        raise ValueError("The paired arms require separate GPUs; sharing is not authorized")
    if runtime.get("shared_gpu_allowlist"):
        if runtime["shared_gpu_allowlist"] != ALLOWED_SHARED:
            raise ValueError(
                "GPU sharing is limited to the explicitly authorized original containers"
            )
        if runtime["gpus"] != {"single": 8, "stream": 9}:
            raise ValueError("Authorized sharing is only single on GPU8 and stream on GPU9")
        if runtime["shared_gpu_min_free_mib"] < 16384:
            raise ValueError("Shared GPU admission requires at least 16384MiB free")
    if set(runtime["cpu_sets"]["single"]) & set(runtime["cpu_sets"]["stream"]):
        raise ValueError("The paired CPU allocations collide")
    if runtime.get("uid", 1015) != 1015 or runtime.get("gid", 1016) != 1016:
        raise ValueError("The persistent output owner must remain uid1015:gid1016")
    if runtime["tracking"] not in ("online", "disabled"):
        raise ValueError("Use online W&B tracking or explicit disabled development mode")
    if (
        runtime["timeout_hours"] <= 0
        or runtime.get("resource_wait_timeout_hours", 12) <= 0
        or runtime["memory_gib"] <= 0
    ):
        raise ValueError("Positive timeout and host memory limits are required")
    for name in ARMS:
        cpus = runtime["cpu_sets"][name]
        if len(cpus) != runtime["cpu_threads"] or len(set(cpus)) != len(cpus):
            raise ValueError("Each arm needs its own complete CPU set")
        if lineage["checkpoints"][name] != expected:
            raise ValueError("Both arms must branch from the exact same DiscoLoop checkpoint")
        output = Path(config["models"][name]["checkpoint"])
        if not output.is_relative_to(Path("results") / config["batch"] / name):
            raise ValueError("Each checkpoint must stay in its own new arm directory")
        if ".." in output.parts:
            raise ValueError("Output paths must not escape their own arm directory")
    contexts = runtime.get("docker_inspection_contexts", ["lm-memory"])
    if not contexts or "lm-memory" not in contexts or len(contexts) != len(set(contexts)):
        raise ValueError("Read-only Docker inspection contexts must include lm-memory once")
    for path in (Path(config["data_dir"]), Path(config["model_config"])):
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
            raise ValueError("Immutable inputs must be relative paths beneath data/")
    if Path(config["model_config"]).parent != Path(config["data_dir"]):
        raise ValueError("The model configuration must be inside the inherited data directory")
    panel_source = Path(config["continual"]["panel_source_path"])
    if (
        panel_source.is_absolute()
        or ".." in panel_source.parts
        or panel_source.parts[0] != "configs"
        or config["continual"]["panel_path"] != "/frozen/configs/continual-panel.json"
    ):
        raise ValueError("The predeclared panel must be frozen from the reviewed configs directory")


def persistent_batch(config):
    root = Path(config["runtime"]["persistent_root"])
    if not root.is_dir() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise RuntimeError("Prepare the required /ossfs persistent mount before formal execution")
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Formal LM outputs must remain below /ossfs/workspace")
    batch = root / config["batch"]
    if batch.is_symlink():
        raise ValueError("The new persistent batch must not be a symlink")
    return batch


def verify_parent(batch, config):
    lineage = config["parent"]
    parent = batch.parent / lineage["parent_batch"]
    manifest = parent / "source-manifest.json"
    if sha256(manifest) != lineage["parent_manifest_sha256"]:
        raise ValueError("Parent source manifest differs from the pinned original")
    frozen = read_json(manifest)
    if frozen["config_sha256"] != lineage["parent_config_sha256"]:
        raise ValueError("Parent configuration digest differs from the pinned original")
    for relative, digest in frozen["files"].items():
        path = parent / "source" / relative
        if not path.resolve().is_relative_to((parent / "source").resolve()):
            raise ValueError("Parent manifest contains a path outside its frozen source")
        if sha256(path) != digest:
            raise ValueError(f"Parent frozen source changed: {relative}")
    if sha256(parent / "source/configs/experiment.json") != lineage["parent_config_sha256"]:
        raise ValueError("Parent frozen configuration changed")
    verify_checkpoint(parent, config)
    return parent


def verify_checkpoint(parent, config):
    path = parent / "discoloop/training/checkpoint-0020000.pt"
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError("The complete original 20k checkpoint is not available")
    digest = sha256(path)
    if digest != config["parent"]["checkpoint_sha256"]:
        raise ValueError("The exact original 20k checkpoint identity changed")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def freeze_source(batch, config_path, parent):
    """Freeze once; existing frozen bytes are verified and never refreshed."""
    source = batch / "source"
    manifest = batch / "source-manifest.json"
    if manifest.exists():
        frozen = read_json(manifest)
        if sha256(config_path) != frozen["config_sha256"]:
            raise ValueError("Configuration changed; use a separately named amended batch")
        for relative, expected in frozen["files"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        return source / "configs/experiment.json"
    source.mkdir(exist_ok=False)
    source.chmod(0o755)
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(
        ROOT / "tests", source / "tests", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (source / "scripts").mkdir()
    (source / "configs").mkdir()
    for name in SOURCE_SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, source / "scripts" / name)
    shutil.copy2(config_path, source / "configs/experiment.json")
    config = read_json(config_path)
    shutil.copy2(
        ROOT / config["continual"]["panel_source_path"], source / "configs/continual-panel.json"
    )
    for name in ("experiment-tracking-defaults.json",):
        shutil.copy2(ROOT / "configs" / name, source / "configs" / name)
    for name in ("pyproject.toml", "CONTRIBUTING.md"):
        shutil.copy2(ROOT / name, source / name)
    backup = source / "input-backup"
    backup.mkdir()
    backup.chmod(0o755)
    for name in (*INPUTS, "model-config.json"):
        original = parent / "source/input-backup" / name
        # The parent manifest was already verified. Copy exact parent inputs;
        # never traverse the private live data directories to prepare mounts.
        shutil.copy2(original, backup / name)
        if sha256(backup / name) != sha256(original):
            raise ValueError(f"Inherited frozen input copy changed: {name}")
    write_json(
        source / "controller-contract.json",
        {
            "schema": "ttcd-cross-sample-controller-v1",
            "stages": list(STAGES),
            "calibration_exit3_stops_arm": True,
            "parent_checkpoint": verify_checkpoint(parent, config),
            "gpu_sharing": bool(config["runtime"].get("shared_gpu_allowlist")),
            "allowed_shared_containers": config["runtime"].get("shared_gpu_allowlist", {}),
            "automatic_retry": False,
            "locks": "new batch/arm claims; exclusive gpu-N or short shared-admission-gpu-N locks",
            "hard_timeout_action": "stop only own currently monitored named container",
            "image": config["parent"]["parent_image_sha256"],
            "runtime": config["runtime"],
        },
    )
    # root_squash daemons must traverse our new source, never private live inputs.
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    write_json(
        manifest,
        {
            "config_sha256": sha256(config_path),
            "parent_manifest_sha256": config["parent"]["parent_manifest_sha256"],
            "checkpoint_sha256": config["parent"]["checkpoint_sha256"],
            "controller_contract_sha256": sha256(source / "controller-contract.json"),
            "files": {
                str(path.relative_to(source)): sha256(path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            },
        },
    )
    return source / "configs/experiment.json"


def prepare_input_mount(batch, config):
    """Prepare a traversable data overlay from immutable parent input backups."""
    data_dir = Path(config["data_dir"])
    model_config = Path(config["model_config"])
    for path in (data_dir, model_config):
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
            raise ValueError("The input overlay supports only relative data/ paths")
    if model_config.parent != data_dir:
        raise ValueError("The model configuration must remain in the inherited data directory")
    mount_root = batch / "input-mount"
    directory = mount_root / data_dir
    expected_files = {data_dir / name: batch / "source/input-backup" / name for name in INPUTS}
    expected_files[model_config] = batch / "source/input-backup/model-config.json"
    expected = {
        "schema": "ttcd-cross-sample-input-mount-v1",
        "frozen_source_manifest_sha256": sha256(batch / "source-manifest.json"),
        "files": {str(relative): sha256(original) for relative, original in expected_files.items()},
    }
    manifest = batch / "input-mount-manifest.json"
    if manifest.exists() and read_json(manifest) != expected:
        raise ValueError("The recorded input mount differs from the immutable frozen inputs")
    for parent in (*reversed(directory.parents), directory):
        if parent != mount_root and not parent.is_relative_to(mount_root):
            continue
        if parent.is_symlink():
            raise ValueError("Input overlay directories must not be symlinks")
        parent.mkdir(exist_ok=True)
        parent.chmod(0o755)
    for relative, original in expected_files.items():
        target = mount_root / relative
        if target.is_symlink():
            raise ValueError("Input overlay files must not be symlinks")
        if target.exists():
            if sha256(target) != expected["files"][str(relative)]:
                raise ValueError(f"Input mount bytes changed: {relative}")
        elif manifest.exists():
            raise ValueError(f"Recorded input mount file is missing: {relative}")
        else:
            shutil.copy2(original, target)
            target.chmod(0o644)
            if sha256(target) != expected["files"][str(relative)]:
                raise ValueError(f"Input mount copy differs from its frozen input: {relative}")
    if not manifest.exists():
        write_json(manifest, expected)
    return mount_root / "data"


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


def check_idle_gpu(gpu, runtime, snapshot):
    rows, processes = snapshot
    if gpu not in rows:
        raise ValueError(f"Requested GPU {gpu} does not exist on this host")
    row = rows[gpu]
    if (
        row["memory"] > runtime["gpu_memory_max_mib"]
        or row["utilization"] > runtime["gpu_utilization_max"]
        or any(process["uuid"] == row["uuid"] for process in processes)
    ):
        raise ResourceUnavailable(f"GPU {gpu} is occupied; no sharing, preemption or reassignment")
    return row


def container_pids(container):
    result = docker("top", container["Id"], "-eo", "pid")
    pids = {int(line.strip()) for line in result.stdout.splitlines()[1:] if line.strip()}
    pids.add(int(container["State"]["Pid"]))
    return pids


def check_shared_gpu(gpu, runtime, snapshot, allowed_pids):
    rows, processes = snapshot
    if gpu not in rows:
        raise ValueError(f"Requested GPU {gpu} does not exist on this host")
    row = rows[gpu]
    unrelated = [
        p["pid"] for p in processes if p["uuid"] == row["uuid"] and p["pid"] not in allowed_pids
    ]
    if unrelated:
        raise ResourceUnavailable(f"GPU {gpu} has unrelated CUDA processes: {unrelated}")
    if row["free_mib"] < runtime["shared_gpu_min_free_mib"]:
        raise ResourceUnavailable(f"GPU {gpu} lacks reserved free memory for a shared arm")
    return row


def host_memory_available_kib():
    return next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def admit_job(config, name):
    runtime = config["runtime"]
    contexts = runtime.get("docker_inspection_contexts", ["lm-memory"])
    containers = running_containers(contexts)
    check_cpu_sets(runtime["cpu_sets"][name], containers, os.sched_getaffinity(0))
    available_kib = host_memory_available_kib()
    if available_kib < runtime["memory_gib"] * 1024**2:
        raise ResourceUnavailable("Insufficient available host memory for another arm")
    snapshot = gpu_snapshot()
    allowed_pids = set()
    shared_containers = []
    shared = runtime.get("shared_gpu_allowlist", {})
    if shared:
        for container in containers:
            if container["Name"].lstrip("/") in shared[name]:
                if container["InspectionContext"] != "lm-memory":
                    raise ValueError("Only the named LM containers may share a GPU")
                if container["Image"] != config["parent"]["parent_image_sha256"]:
                    raise ValueError(
                        "Whitelisted original image differs from the pinned parent image"
                    )
                pids = container_pids(container)
                allowed_pids.update(pids)
                shared_containers.append({"name": container["Name"], "host_pids": sorted(pids)})
        row = check_shared_gpu(runtime["gpus"][name], runtime, snapshot, allowed_pids)
    else:
        row = check_idle_gpu(runtime["gpus"][name], runtime, snapshot)
    return {
        **row,
        "admission_utc": utc_now(),
        "cpu_set": runtime["cpu_sets"][name],
        "host_available_kib": available_kib,
        "inspection_contexts": contexts,
        "gpu_sharing": bool(shared),
        "allowed_shared_containers": shared.get(name, []),
        "observed_shared_containers": shared_containers,
        "allowed_host_pids": sorted(allowed_pids),
        "running_containers": [
            {
                "name": container["Name"],
                "context": container["InspectionContext"],
                "cpu_set": container["HostConfig"].get("CpusetCpus", ""),
            }
            for container in containers
        ],
        "observed_gpu_processes": [p for p in snapshot[1] if p["uuid"] == row["uuid"]],
    }


def wait_for_resources(config, name, gpu_lock, deadline, update=lambda _: None, poll_seconds=30):
    """Wait for exclusive admission before creating any scientific stage claim.

    Only expected occupancy is queued. An unreachable daemon, NVML failure or
    invalid CPU allocation fails immediately rather than masquerading as busy.
    The returned admission lock remains held until the caller releases it.
    """
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Resource wait timeout; no experiment was launched or stopped")
        acquired = False
        try:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as error:
                raise ResourceUnavailable("Exclusive GPU lock is held by another LM job") from error
            return admit_job(config, name)
        except ResourceUnavailable as error:
            if acquired:
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            update(
                {
                    "state": "waiting_for_resources",
                    "gpu": config["runtime"]["gpus"][name],
                    "reason": str(error),
                    "resource_wait_deadline_monotonic": deadline,
                }
            )
            time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def reserve_stage(out, stage):
    if stage == "train" and (out / "training").exists():
        raise FileExistsError("Training outputs already exist; no retry or overwrite")
    with (out / f"claim-{stage}.json").open("x") as stream:
        json.dump({"stage": stage, "pid": os.getpid(), "claimed_utc": utc_now()}, stream)
        stream.write("\n")


def build_command(config, name, stage, image, row, out, batch, parent, token_file):
    runtime = config["runtime"]
    container_name = f"{config['batch']}-{name}-{stage}"
    command = [
        *DOCKER,
        "run",
        "--detach",
        "--name",
        container_name,
        "--gpus",
        f"device={row['uuid']}",
        "--user",
        f"{runtime.get('uid', 1015)}:{runtime.get('gid', 1016)}",
        "--cpus",
        str(runtime["cpu_threads"]),
        "--cpuset-cpus",
        ",".join(map(str, runtime["cpu_sets"][name])),
        "--memory",
        f"{runtime['memory_gib']}g",
        "--shm-size",
        "2g",
        "--network",
        "host",
        "--workdir",
        "/project",
        "--mount",
        f"type=bind,source={ROOT},target=/project,readonly",
        "--mount",
        f"type=bind,source={batch.parent},target=/project/results,readonly",
        "--mount",
        f"type=bind,source={batch / 'input-mount/data'},target=/project/data,readonly",
        "--mount",
        f"type=bind,source={batch / 'source'},target=/frozen,readonly",
        "--mount",
        f"type=bind,source={parent / 'source'},target=/parent-frozen,readonly",
        "--mount",
        f"type=bind,source={parent / 'source-manifest.json'},"
        "target=/parent-source-manifest.json,readonly",
        "--mount",
        f"type=bind,source={out},target=/project/results/{config['batch']}/{name}",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--env",
        "PYTHONPATH=/frozen/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HOME=/tmp/lm-home",
        "--env",
        "XDG_CACHE_HOME=/tmp/lm-home/.cache",
        "--env",
        "XDG_CONFIG_HOME=/tmp/lm-home/.config",
        "--env",
        f"OMP_NUM_THREADS={runtime['cpu_threads']}",
        "--env",
        f"MKL_NUM_THREADS={runtime['cpu_threads']}",
    ]
    if token_file.is_file():
        command += [
            "--mount",
            f"type=bind,source={token_file.resolve()},target=/run/secrets/wandb,readonly",
        ]
    command += [
        image,
        "bash",
        "-c",
        'mkdir -p "$HOME"\nif [ -f /run/secrets/wandb ]; then\n'
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"; fi\nexec "$@"',
        "ttcd-continual-entry",
        "python",
        "/frozen/scripts/run_ttcd_continual.py",
        stage,
        "--config",
        "/frozen/configs/experiment.json",
        "--model",
        name,
        "--device",
        "cuda:0",
        "--out",
        f"/project/results/{config['batch']}/{name}",
        "--tracking",
        runtime["tracking"],
    ]
    return container_name, command


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


def run_arm(config, name, stages, image, batch, parent, token_file):
    out = batch / name
    if out.is_symlink():
        raise ValueError("An arm output directory must not be a symlink")
    out.mkdir(exist_ok=True)
    out.chmod(0o755)
    state_path = out / "controller-state.json"
    resource_deadline = (
        time.monotonic() + config["runtime"].get("resource_wait_timeout_hours", 12) * 3600
    )
    state = {"model": name}

    def update(value):
        state.update(value)
        write_json(state_path, {**state, "updated_utc": utc_now()})

    try:
        with (out / "controller.lock").open("a") as arm_lock:
            fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = verify_checkpoint(parent, config)
            write_json(out / "parent-checkpoint-receipt.json", receipt)
            lock_dir = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
            lock_dir.mkdir(exist_ok=True)
            gpu = config["runtime"]["gpus"][name]
            shared = bool(config["runtime"].get("shared_gpu_allowlist"))
            lock_name = f"shared-admission-gpu-{gpu}.lock" if shared else f"gpu-{gpu}.lock"
            with (lock_dir / lock_name).open("a") as gpu_lock:
                first_admission = wait_for_resources(
                    config, name, gpu_lock, resource_deadline, update
                )
                deadline = time.monotonic() + config["runtime"]["timeout_hours"] * 3600
                for stage_index, stage in enumerate(stages):
                    prepare_input_mount(batch, config)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Hard timeout before starting next stage")
                    if stage_index == 0:
                        row = first_admission
                    elif shared:
                        row = wait_for_resources(
                            config, name, gpu_lock, min(resource_deadline, deadline), update
                        )
                    else:
                        row = admit_job(config, name)
                    reserve_stage(out, stage)
                    container_name = f"{config['batch']}-{name}-{stage}"
                    if inspect_container(container_name) is not None:
                        raise FileExistsError("The arm container already exists; no resubmission")
                    write_json(out / f"admission-{stage}.json", row)
                    _, command = build_command(
                        config, name, stage, image, row, out, batch, parent, token_file
                    )
                    # Arguments contain only the secret file path, never its bytes.
                    write_json(out / f"command-{stage}.json", {"argv": command, "utc": utc_now()})
                    details = {
                        "stage": stage,
                        "container": container_name,
                        "gpu": gpu,
                        "gpu_uuid": row["uuid"],
                        "cpu_set": config["runtime"]["cpu_sets"][name],
                        "image": image,
                    }
                    update({"state": "launching", **details})
                    with (out / f"{stage}.log").open("x") as log:
                        launched = subprocess.run(
                            command, capture_output=True, text=True, timeout=60
                        )
                        log.write(launched.stderr)
                        log.flush()
                        if launched.returncode:
                            write_json(
                                out / f"exit-{stage}.json",
                                {
                                    "returncode": launched.returncode,
                                    "launch_failed": True,
                                    "finished_utc": utc_now(),
                                    **details,
                                },
                            )
                            raise RuntimeError(f"Arm launch failed: {launched.stderr.strip()}")
                    container_id = launched.stdout.strip()
                    if shared:
                        # Original gpu-N locks are untouched. The short shared
                        # admission lock covers inspect plus launch, not training.
                        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    update({"state": "running", "container_id": container_id, **details})
                    with (out / f"{stage}.log").open("a") as log:
                        code = monitor_container(container_name, log, deadline, update)
                    write_json(
                        out / f"exit-{stage}.json",
                        {
                            "returncode": code,
                            "container_id": container_id,
                            "memory_gib": config["runtime"]["memory_gib"],
                            "finished_utc": utc_now(),
                            **details,
                        },
                    )
                    if code:
                        update(
                            {
                                "state": "calibration_failed"
                                if stage == "calibrate" and code == 3
                                else "needs_attention",
                                "returncode": code,
                            }
                        )
                        return code
            update({"state": "complete", "stages": stages})
            return 0
    except Exception:
        failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
        update(failure)
        write_json(out / "controller-failure.json", failure)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pipeline", *STAGES), nargs="?", default="pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ttcd-cross-sample-v1.json")
    parser.add_argument("--model", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--image", default="lm-memory-ttcd:torch2.8-cu128")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=Path.home() / ".cache/llm-memory-ttcd/wandb-token",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    models = list(ARMS) if args.model == "both" else [args.model]
    stages = list(STAGES) if args.stage == "pipeline" else [args.stage]
    plan = {
        "batch": config["batch"],
        "stages": stages,
        "docker_context": "lm-memory",
        "image": config["parent"]["parent_image_sha256"],
        "jobs": [
            {
                "model": name,
                "gpu": config["runtime"]["gpus"][name],
                "cpu_set": config["runtime"]["cpu_sets"][name],
                "parent_checkpoint": config["parent"]["checkpoints"][name],
                "parent_checkpoint_sha256": config["parent"]["checkpoint_sha256"],
            }
            for name in models
        ],
        "executed": args.execute,
        "gpu_sharing": bool(config["runtime"].get("shared_gpu_allowlist")),
        "allowed_shared_containers": config["runtime"].get("shared_gpu_allowlist", {}),
        "automatic_retry": False,
        "persistent_root": config["runtime"]["persistent_root"],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not args.config.resolve().is_relative_to(ROOT):
        raise ValueError("Put the reviewed configuration inside the mounted project")
    if config["runtime"]["tracking"] != "disabled":
        if not args.wandb_token_file.is_file():
            raise FileNotFoundError(
                "Provide the local private W&B credential file before execution"
            )
        if args.wandb_token_file.stat().st_mode & 0o077:
            raise ValueError("W&B credential file must not be accessible to group/others")
    batch = persistent_batch(config)
    parent = verify_parent(batch, config)
    docker("info")
    image = json.loads(docker("image", "inspect", args.image).stdout)[0]["Id"]
    if image != config["parent"]["parent_image_sha256"]:
        raise ValueError("The selected image differs from the parent's fixed image")
    batch.mkdir(parents=True, exist_ok=True)
    batch.chmod(0o755)
    with (batch / "controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_source(batch, args.config.resolve(), parent)
        prepare_input_mount(batch, config)
        process = {
            "pid": os.getpid(),
            "models": models,
            "stage": args.stage,
            "image": image,
            "started_utc": utc_now(),
            "source_manifest": str(batch / "source-manifest.json"),
            "input_mount_manifest": str(batch / "input-mount-manifest.json"),
            "automatic_retry": False,
            "gpu_sharing": bool(config["runtime"].get("shared_gpu_allowlist")),
        }
        write_json(batch / "controller-process.json", process)
        write_json(batch / "controller-state.json", {"state": "running", **plan})
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
            futures = {
                name: pool.submit(
                    run_arm, config, name, stages, image, batch, parent, args.wandb_token_file
                )
                for name in models
            }
            results = {name: future.result() for name, future in futures.items()}
        write_json(
            batch / "controller-state.json",
            {
                "state": "complete" if not any(results.values()) else "needs_attention",
                "stages": stages,
                "exit_codes": results,
                "finished_utc": utc_now(),
            },
        )
        if any(results.values()):
            raise SystemExit(
                "An arm stopped; preserved logs explain the failure. No retry occurred."
            )


if __name__ == "__main__":
    main()
