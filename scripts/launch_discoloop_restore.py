#!/usr/bin/env python3
"""Restore the successful DiscoLoop foundation in a new isolated batch.

Default invocation is read-only. Explicit execution waits for the matching
cross-sample arm's successful science audit and an unoccupied GPU8/9, then runs
only foundation stages. A downstream launcher requires an explicit separate
configuration and a passing foundation gate. No stage is automatically retried.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["docker", "--context", "lm-memory"]
BATCH = "discoloop-success-restore-v1"
ORIGINAL_IMAGE = "sha256:0087c5381ef5286d30b022825ab773f044fa15c161e5999ff5b47fbab681b18a"
STAGES = ("preflight", "smoke", "smoke-audit", "train", "audit", "foundation-gate", "cloud-audit")
RELOAD_STAGES = ("preflight", "audit", "foundation-gate", "cloud-audit")
SOURCE_SCRIPTS = (
    "run_discoloop_restore.py",
    "run_supervised_alignment.py",
    "launch_discoloop_restore.py",
)
GPU_DEPENDENCIES = {8: "single", 9: "stream"}
PROTOCOLS = (
    "docs/development-artifacts/discoloop-success-restore-v1/protocol.md",
    "docs/development-artifacts/ttcd-successful-base-v1/protocol.md",
)
DOWNSTREAM_SCRIPTS = (
    "scripts/run_ttcd_verified_base.py",
    "scripts/run_supervised_alignment.py",
    "scripts/run_ttcd_second_hop.py",
    "scripts/prepare_ttcd_scratch.py",
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


def pipeline_stages(config):
    mode = config.get("restore", {}).get("mode", "rebuild")
    return list(RELOAD_STAGES if mode == "reload" else STAGES)


def validate_config(config):
    if config["batch"] != BATCH or set(config["arms"]) != {"discoloop"}:
        raise ValueError("Only the new one-arm DiscoLoop foundation batch is authorized")
    spec = config["spec"]
    expected = dict(
        unique_layers=4,
        repeats=2,
        phi=7.2,
        weight_decay=0.1,
        initialization=791101,
        stream_seed=792101,
        lr=1e-4,
        warmup=2000,
        batch_size=512,
        steps=300000,
    )
    if any(spec[key] != value for key, value in expected.items()):
        raise ValueError("The original successful static training recipe must be preserved")
    arm = config["arms"]["discoloop"]
    if (
        arm["kind"] != "none"
        or arm["coefficient"] != 0
        or not arm.get("discoloop")
        or arm.get("first_round_kv", False)
        or arm.get("scope", "supervised") != "supervised"
        or arm.get("discoloop_alpha", 1.0) != 1.0
        or arm.get("discoloop_temperature", 1.0) != 1.0
        or arm.get("discoloop_eps", 1e-6) != 1e-6
    ):
        raise ValueError("Foundation training is the original DiscoLoop static CE arm")
    if config.get("restore", {}).get("mode", "rebuild") not in ("rebuild", "reload"):
        raise ValueError("Choose explicit rebuild or reload mode; there is no automatic fallback")
    input_root = Path(config["input_root"])
    if input_root != Path("data") / BATCH / "original":
        raise ValueError("Use the new batch's independent original-data restoration tree")
    checkpoint = Path(config["initial_checkpoint"])
    expected_checkpoint = input_root / "development/loop4x2-phi7.2-wd0.1/checkpoint-0000000.pt"
    if checkpoint != expected_checkpoint:
        raise ValueError("The pinned original step-zero source belongs in the restored data tree")
    nodes = config["evaluation_nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != 300000:
        raise ValueError("Evaluation nodes must include the exact fixed 300k endpoint")
    runtime = config["runtime"]
    if not runtime["image"].startswith("sha256:") or len(runtime["image"]) != 71:
        raise ValueError("Pin the selected foundation environment by complete image SHA256")
    if runtime.get("docker_context", "lm-memory") != "lm-memory":
        raise ValueError("All new experiments must use the explicit lm-memory Docker context")
    if runtime["gpu_preference"] != [8, 9]:
        raise ValueError("Foundation scheduling is scoped to GPU8/9 after their science audits")
    if runtime.get("shared_gpu_allowlist"):
        raise ValueError("Foundation GPU sharing is not authorized")
    cpus = runtime["cpu_set"]
    if not cpus or len(cpus) != len(set(cpus)):
        raise ValueError("Assign a nonempty unique foundation CPU set")
    if runtime.get("cpu_threads", len(cpus)) != len(cpus):
        raise ValueError("The CPU quota and pinned CPU set must agree")
    if runtime["memory_gib"] < 48 or runtime["memory_gib"] > 96:
        raise ValueError("Foundation host memory must remain within the reviewed 48–96GiB range")
    if runtime["min_total_gpu_mib"] < 90000 or runtime["min_free_gpu_mib"] < 80000:
        raise ValueError("Foundation requires the reviewed large-memory GPU admission floor")
    if runtime.get("tracking", "online") != "online":
        raise ValueError("Formal restoration stages require online W&B tracking")
    if runtime.get("uid", 1015) != 1015 or runtime.get("gid", 1016) != 1016:
        raise ValueError("Persistent LM outputs retain uid1015:gid1016 ownership")
    if runtime.get("timeout_hours", 12) <= 0 or runtime.get("wait_timeout_hours", 24) <= 0:
        raise ValueError("Positive experiment and resource-wait timeouts are required")
    if runtime.get("dependency_batch", "ttcd-cross-sample-v1") != "ttcd-cross-sample-v1":
        raise ValueError("The reviewed predecessor is the active cross-sample batch")


def persistent_batch(config):
    root = Path(config["runtime"]["persistent_root"])
    if not root.is_dir() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise RuntimeError("Prepare the required /ossfs persistent mount before execution")
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Formal LM outputs must remain below /ossfs/workspace")
    batch = root / BATCH
    if batch.is_symlink():
        raise ValueError("The new persistent batch must not be a symlink")
    return batch


def input_files(config):
    root = Path(config["input_root"])
    paths = [
        root / name
        for name in (
            "data/complete.json",
            "data/world.npz",
            "data/panels.npz",
            "reference/gpt2/config.json",
        )
    ]
    paths.append(Path(config["initial_checkpoint"]))
    if config.get("restore", {}).get("mode", "rebuild") == "reload":
        recovered = Path(config["restore"]["recovered_checkpoint"])
        if not recovered.is_relative_to(root) or ".." in recovered.parts:
            raise ValueError("Recovered weights must live in the new immutable input tree")
        paths.append(recovered)
    return paths


def freeze_source(batch, config_path):
    """Freeze source, tests and exact restored input bytes once."""
    source, manifest = batch / "source", batch / "source-manifest.json"
    if manifest.exists():
        frozen = read_json(manifest)
        if sha256(config_path) != frozen["config_sha256"]:
            raise ValueError(
                "Frozen configuration changed; a separately recorded amendment is required"
            )
        for relative, expected in frozen["files"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        return source / "configs/experiment.json"
    config = read_json(config_path)
    files = input_files(config)
    # Check all inputs before creating a partial freeze, including the pinned
    # step-zero checkpoint prepared separately from scientific training.
    for relative in files:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"Required restored input is missing: {relative}")
    source.mkdir(exist_ok=False)
    source.chmod(0o755)
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
    # Test fixture only; the science CLI always reads configs/experiment.json.
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
    backup = source / "input-backup"
    for relative in files:
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        if sha256(target) != sha256(ROOT / relative):
            raise ValueError(f"Restored input changed while freezing: {relative}")
    write_json(
        source / "controller-contract.json",
        {
            "schema": "discoloop-success-restore-controller-v1",
            "stages": pipeline_stages(config),
            "mode": config.get("restore", {}).get("mode", "rebuild"),
            "original_image": ORIGINAL_IMAGE,
            "selected_image": config["runtime"]["image"],
            "original_environment_identity_claimed": config["runtime"]["image"] == ORIGINAL_IMAGE,
            "predecessor": {str(gpu): arm for gpu, arm in GPU_DEPENDENCIES.items()},
            "predecessor_release": "local science audit passed; no GPU container or CUDA PID",
            "gpu_sharing": False,
            "automatic_retry": False,
            "foundation_gate_failure_stops_pipeline": True,
            "reader_or_writer_before_gate": False,
            "hard_timeout_action": "stop only own monitored named container",
            "runtime": config["runtime"],
        },
    )
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    write_json(
        manifest,
        {
            "config_sha256": sha256(config_path),
            "protocols": {
                relative: {
                    "sha256": sha256(source / relative),
                    "bytes": (source / relative).stat().st_size,
                }
                for relative in PROTOCOLS
            },
            "inputs": {str(relative): sha256(backup / relative) for relative in files},
            "files": {
                str(path.relative_to(source)): sha256(path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            },
        },
    )
    return source / "configs/experiment.json"


def prepare_input_mount(batch, config):
    manifest = batch / "input-mount-manifest.json"
    frozen = read_json(batch / "source-manifest.json")
    expected = {
        "schema": "discoloop-success-restore-input-mount-v1",
        "source_manifest_sha256": sha256(batch / "source-manifest.json"),
        "files": frozen["inputs"],
    }
    if manifest.exists() and read_json(manifest) != expected:
        raise ValueError("Input overlay identity differs from the frozen source")
    root = batch / "input-mount"
    for relative, digest in frozen["inputs"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
            raise ValueError("Input overlays support only reviewed relative data/ paths")
        target, original = root / path, batch / "source/input-backup" / path
        if target.is_symlink() or any(
            p.is_symlink() for p in target.parents if p.is_relative_to(root)
        ):
            raise ValueError("Input overlay paths must not be symlinks")
        target.parent.mkdir(parents=True, exist_ok=True)
        for directory in (root, *target.parent.parents, target.parent):
            if directory == root or directory.is_relative_to(root):
                directory.chmod(0o755)
        if target.exists():
            if sha256(target) != digest:
                raise ValueError(f"Input overlay bytes changed: {relative}")
        elif manifest.exists():
            raise ValueError(f"Recorded input overlay file is missing: {relative}")
        else:
            shutil.copy2(original, target)
            target.chmod(0o644)
            if sha256(target) != digest:
                raise ValueError(f"Input overlay copy differs: {relative}")
    if not manifest.exists():
        write_json(manifest, expected)
    return root / "data"


def predecessor_receipt(batch, config, gpu):
    arm = GPU_DEPENDENCIES[gpu]
    predecessor = batch.parent / config["runtime"].get("dependency_batch", "ttcd-cross-sample-v1")
    path = predecessor / arm / "audit.json"
    if not path.is_file():
        raise ResourceUnavailable(f"Waiting for GPU{gpu} predecessor {arm}'s local science audit")
    result = read_json(path)
    if result.get("passed") is not True:
        raise ResourceUnavailable(f"GPU{gpu} predecessor science audit did not pass")
    return {"arm": arm, "path": str(path), "sha256": sha256(path), "passed": True}


def container_uses_gpu(container, gpu, uuid):
    for request in container["HostConfig"].get("DeviceRequests") or []:
        if request.get("Count") == -1:
            return True
        if any(identifier in (str(gpu), uuid) for identifier in request.get("DeviceIDs") or []):
            return True
    return False


def admit_job(batch, config, gpu):
    receipt = predecessor_receipt(batch, config, gpu)
    runtime = config["runtime"]
    containers = running_containers(["lm-memory"])
    check_cpu_sets(runtime["cpu_set"], containers, os.sched_getaffinity(0))
    available = host_memory_available_kib()
    if available < runtime["memory_gib"] * 1024**2:
        raise ResourceUnavailable("Insufficient host memory for the foundation")
    rows, processes = gpu_snapshot()
    if gpu not in rows:
        raise ValueError(f"Requested GPU{gpu} does not exist")
    row = rows[gpu]
    if row["total_mib"] < runtime["min_total_gpu_mib"]:
        raise ValueError(f"GPU{gpu} is smaller than the reviewed memory requirement")
    if (
        row["free_mib"] < runtime["min_free_gpu_mib"]
        or row["memory"] > runtime.get("gpu_memory_max_mib", 1024)
        or row["utilization"] > runtime.get("gpu_utilization_max", 5)
        or any(process["uuid"] == row["uuid"] for process in processes)
        or any(container_uses_gpu(c, gpu, row["uuid"]) for c in containers)
    ):
        raise ResourceUnavailable(f"GPU{gpu} is occupied; no sharing, preemption or reassignment")
    return {
        **row,
        "gpu": gpu,
        "cpu_set": runtime["cpu_set"],
        "host_available_kib": available,
        "predecessor_audit": receipt,
        "admission_utc": utc_now(),
        "gpu_sharing": False,
        "running_lm_containers": [c["Name"] for c in containers],
        "observed_compute_pids": [p for p in processes if p["uuid"] == row["uuid"]],
    }


def wait_for_slot(batch, config, deadline, update=lambda _: None, poll_seconds=30):
    """Return an exclusive GPU lock plus admission; never retry a stage."""
    directory = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Resource wait timeout; no existing experiment was stopped")
        reasons = {}
        for gpu in config["runtime"]["gpu_preference"]:
            held = (directory / f"gpu-{gpu}.lock").open("a")
            try:
                try:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ResourceUnavailable("Exclusive GPU lock is held") from error
                admission = admit_job(batch, config, gpu)
                return held, admission
            except ResourceUnavailable as error:
                reasons[str(gpu)] = str(error)
                held.close()
            except BaseException:
                held.close()
                raise
        update({"state": "waiting_for_resources", "reasons": reasons})
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


def reserve_stage(batch, stage):
    with (batch / f"claim-{stage}.json").open("x") as stream:
        json.dump({"stage": stage, "pid": os.getpid(), "claimed_utc": utc_now()}, stream)
        stream.write("\n")


def build_command(config, stage, row, batch, token_file):
    runtime = config["runtime"]
    name = f"{BATCH}-{stage}"
    cpus = runtime["cpu_set"]
    command = [
        *DOCKER,
        "run",
        "--detach",
        "--name",
        name,
        "--gpus",
        f"device={row['uuid']}",
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
        "--mount",
        f"type=bind,source={ROOT},target=/project,readonly",
        "--mount",
        f"type=bind,source={batch.parent},target=/project/results,readonly",
        "--mount",
        f"type=bind,source={batch},target=/project/results/{BATCH}",
        "--mount",
        f"type=bind,source={batch / 'source'},target=/frozen,readonly",
        "--mount",
        f"type=bind,source={batch / 'source'},target=/project/results/{BATCH}/source,readonly",
        "--mount",
        f"type=bind,source={batch / 'input-mount'},"
        f"target=/project/results/{BATCH}/input-mount,readonly",
        "--mount",
        f"type=bind,source={batch / 'source-manifest.json'},"
        f"target=/project/results/{BATCH}/source-manifest.json,readonly",
        "--mount",
        f"type=bind,source={batch / 'input-mount/data'},target=/project/data,readonly",
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
        "discoloop-foundation-entry",
        "python",
        "/frozen/scripts/run_discoloop_restore.py",
        stage,
        "--config",
        "/frozen/configs/experiment.json",
        "--device",
        "cuda:0",
        "--out",
        f"/project/results/{BATCH}",
        "--tracking",
        "online",
    ]
    return name, command


def run_pipeline(batch, config, stages, token_file):
    state = {"batch": BATCH, "stages": stages, "automatic_retry": False}

    def update(value):
        state.update(value)
        write_json(batch / "controller-state.json", {**state, "updated_utc": utc_now()})

    try:
        wait_deadline = time.monotonic() + config["runtime"].get("wait_timeout_hours", 24) * 3600
        held, first_admission = wait_for_slot(batch, config, wait_deadline, update)
        with held:
            deadline = time.monotonic() + config["runtime"].get("timeout_hours", 12) * 3600
            gpu = first_admission["gpu"]
            for index, stage in enumerate(stages):
                prepare_input_mount(batch, config)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Foundation hard timeout before next stage")
                row = first_admission if index == 0 else admit_job(batch, config, gpu)
                reserve_stage(batch, stage)
                name, command = build_command(config, stage, row, batch, token_file)
                if inspect_container(name) is not None:
                    raise FileExistsError("Foundation container already exists; no resubmission")
                write_json(batch / f"admission-{stage}.json", row)
                write_json(batch / f"command-{stage}.json", {"argv": command, "utc": utc_now()})
                details = {"stage": stage, "container": name, "gpu": gpu, "gpu_uuid": row["uuid"]}
                update({"state": "launching", **details})
                with (batch / f"{stage}.log").open("x") as log:
                    launched = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    log.write(launched.stderr)
                    log.flush()
                if launched.returncode:
                    write_json(
                        batch / f"exit-{stage}.json",
                        {
                            "returncode": launched.returncode,
                            "launch_failed": True,
                            **details,
                        },
                    )
                    raise RuntimeError(f"Foundation launch failed: {launched.stderr.strip()}")
                update({"state": "running", "container_id": launched.stdout.strip(), **details})
                with (batch / f"{stage}.log").open("a") as log:
                    code = monitor_container(name, log, deadline, update)
                write_json(
                    batch / f"exit-{stage}.json",
                    {
                        "returncode": code,
                        "finished_utc": utc_now(),
                        **details,
                    },
                )
                if code:
                    update(
                        {
                            "state": "foundation_gate_failed"
                            if stage == "foundation-gate"
                            else "needs_attention",
                            "returncode": code,
                        }
                    )
                    return code
                if stage == "foundation-gate":
                    if read_json(batch / "result-foundation-gate.json").get("passed") is not True:
                        update({"state": "foundation_gate_failed", "returncode": 3})
                        return 3
            update({"state": "complete", "finished_utc": utc_now()})
            return 0
    except Exception:
        failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
        update(failure)
        write_json(batch / "controller-failure.json", failure)
        return 1


def downstream_contract(config_path, launcher):
    """Pin future science before the long foundation run starts."""
    config_path, launcher = config_path.resolve(), launcher.resolve()
    if not config_path.is_file() or not launcher.is_file():
        raise FileNotFoundError("Provide the independently reviewed downstream config and launcher")
    if not config_path.is_relative_to(ROOT) or not launcher.is_relative_to(ROOT):
        raise ValueError("The downstream contract must use reviewed files inside the project")
    paths = [
        path
        for path in (ROOT / "src").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    paths += [ROOT / relative for relative in (*DOWNSTREAM_SCRIPTS, *PROTOCOLS)]
    paths.append(launcher)
    downstream = read_json(config_path)
    if "original_data_dir" in downstream:
        paths += [
            Path(downstream["original_data_dir"]) / name
            for name in ("world.npz", "panels.npz", "complete.json")
        ]
        paths = [path if path.is_absolute() else ROOT / path for path in paths]
    if "original_model_config" in downstream:
        paths.append(ROOT / downstream["original_model_config"])
    entries = {}
    for path in sorted(set(paths)):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Pinned downstream scientific inputs must remain inside the project")
        entries[str(path.relative_to(ROOT))] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema": "discoloop-downstream-input-contract-v1",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "launcher": str(launcher),
        "launcher_sha256": sha256(launcher),
        "files": entries,
    }


def run_downstream(batch, config_path, launcher, token_file, expected_contract=None):
    """Only an explicitly supplied, separately frozen paired experiment may run."""
    gate = read_json(batch / "result-foundation-gate.json")
    exit_receipt = read_json(batch / "exit-foundation-gate.json")
    if gate.get("passed") is not True or exit_receipt["returncode"] != 0:
        raise ValueError("A passing local foundation gate is required before downstream work")
    if not config_path.is_file() or not launcher.is_file():
        raise FileNotFoundError("Provide the independently reviewed downstream config and launcher")
    if expected_contract is not None:
        if downstream_contract(config_path, launcher) != expected_contract:
            raise ValueError(
                "Downstream configuration, launcher or scientific source changed after startup"
            )
    command = [
        sys.executable,
        str(launcher),
        "pipeline",
        "--config",
        str(config_path),
        "--execute",
        "--wandb-token-file",
        str(token_file),
    ]
    write_json(
        batch / "downstream-handoff.json",
        {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "launcher": str(launcher),
            "launcher_sha256": sha256(launcher),
            "gate_sha256": sha256(batch / "result-foundation-gate.json"),
            "argv": command,
            "utc": utc_now(),
        },
    )
    completed = subprocess.run(command, check=False)
    write_json(
        batch / "downstream-exit.json", {"returncode": completed.returncode, "utc": utc_now()}
    )
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pipeline", *STAGES), nargs="?", default="pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--then-config", type=Path)
    parser.add_argument("--then-launcher", type=Path)
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=Path.home() / ".cache/llm-memory-ttcd/wandb-token",
    )
    args = parser.parse_args()
    if bool(args.then_config) != bool(args.then_launcher):
        parser.error(
            "Use --then-config and --then-launcher together for the explicit paired handoff"
        )
    config = read_json(args.config)
    validate_config(config)
    pinned_downstream = (
        downstream_contract(args.then_config, args.then_launcher) if args.then_config else None
    )
    stages = pipeline_stages(config) if args.stage == "pipeline" else [args.stage]
    if config.get("restore", {}).get("mode", "rebuild") == "reload" and any(
        stage in ("smoke", "smoke-audit", "train") for stage in stages
    ):
        raise ValueError("Reload mode never starts new training")
    plan = {
        "batch": BATCH,
        "stages": stages,
        "docker_context": "lm-memory",
        "image": config["runtime"]["image"],
        "original_image": ORIGINAL_IMAGE,
        "mode": config.get("restore", {}).get("mode", "rebuild"),
        "gpu_preference": config["runtime"]["gpu_preference"],
        "gpu_dependencies": {str(gpu): arm for gpu, arm in GPU_DEPENDENCIES.items()},
        "cpu_set": config["runtime"]["cpu_set"],
        "gpu_sharing": False,
        "automatic_retry": False,
        "executed": args.execute,
        "downstream_config": str(args.then_config) if args.then_config else None,
        "downstream_launcher": str(args.then_launcher) if args.then_launcher else None,
        "downstream_scientific_files": len(pinned_downstream["files"]) if pinned_downstream else 0,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not args.config.resolve().is_relative_to(ROOT):
        raise ValueError("Put the reviewed configuration inside the project")
    if not args.wandb_token_file.is_file() or args.wandb_token_file.stat().st_mode & 0o077:
        raise ValueError("Online W&B requires the local private token file")
    batch = persistent_batch(config)
    docker("info")
    observed = json.loads(docker("image", "inspect", config["runtime"]["image"]).stdout)[0]["Id"]
    if observed != config["runtime"]["image"]:
        raise ValueError("The selected pinned image differs")
    batch.mkdir(parents=True, exist_ok=True)
    batch.chmod(0o755)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if pinned_downstream:
            pin_path = batch / "downstream-input-contract.json"
            if pin_path.exists() and read_json(pin_path) != pinned_downstream:
                raise ValueError("The original downstream startup contract must not be replaced")
            if not pin_path.exists():
                write_json(pin_path, pinned_downstream)
        freeze_source(batch, args.config.resolve())
        prepare_input_mount(batch, config)
        write_json(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "started_utc": utc_now(),
                "stages": stages,
                "source_manifest": str(batch / "source-manifest.json"),
                "image": observed,
                "original_image": ORIGINAL_IMAGE,
                "automatic_retry": False,
                "gpu_sharing": False,
            },
        )
        code = run_pipeline(batch, config, stages, args.wandb_token_file)
        if code:
            raise SystemExit(code)
    if args.then_config:
        raise SystemExit(
            run_downstream(
                batch,
                args.then_config.resolve(),
                args.then_launcher.resolve(),
                args.wandb_token_file,
                pinned_downstream,
            )
        )


if __name__ == "__main__":
    main()
