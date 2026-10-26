#!/usr/bin/env python3
"""Wait for the exact TTCD 20k parents, then run explicitly authorized controls.

The default is a read-only plan. --execute runs independently monitored arms;
only the named original training containers may share their respective GPU.
Existing training and its GPU locks are never changed or retried.
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
STAGES = ("train", "train-audit", "calibrate", "evaluate", "audit")
PARENTS = {
    "discoloop": "ttcd-second-hop-scratch-v1-discoloop-train-restart1",
    "loop": "ttcd-second-hop-scratch-v1-loop-train",
}


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


def docker(*args, check=True):
    return subprocess.run([*DOCKER, *args], check=check, capture_output=True, text=True, timeout=45)


def persistent_batch(config):
    root = Path(config["runtime"]["persistent_root"])
    if not root.is_dir() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise RuntimeError("Prepare the required /ossfs persistent mount before formal execution")
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Formal LM outputs must remain below /ossfs/workspace")
    return root / config["batch"]


def validate_config(config):
    branch = config["branch"]
    if config["batch"] == branch["parent_batch"]:
        raise ValueError("The control batch must not overwrite its parent")
    if config["batch"] != "ttcd-second-hop-reader-control-v1":
        raise ValueError("This controller is scoped to the authorized reader-control batch")
    if branch["parent_batch"] != "ttcd-second-hop-scratch-v1":
        raise ValueError("Only the authorized scratch parent may be branched")
    if branch["checkpoint_step"] != 20000 or branch["steps"] != 10000:
        raise ValueError("Authorized branches are exactly step 20000 plus 10000 updates")
    if branch["objective"] != "old_atoms_old_chains_context_reader_only":
        raise ValueError("Unexpected branch objective")
    if branch["parent_source_dir"] != "/parent-frozen":
        raise ValueError("Unexpected parent source mount")
    if branch["parent_manifest_path"] != "/parent-source-manifest.json":
        raise ValueError("Unexpected parent manifest mount")
    runtime = config["runtime"]
    if runtime["gpus"] != {"discoloop": 8, "loop": 9}:
        raise ValueError("GPU assignment must remain DiscoLoop 8 and Loop 9")
    cpu_sets = runtime["cpu_sets"]
    if set(cpu_sets["discoloop"]) & set(cpu_sets["loop"]):
        raise ValueError("Control CPU allocations collide")
    for name in PARENTS:
        if runtime["shared_gpu_allowlist"][name] != [PARENTS[name]]:
            raise ValueError("GPU sharing is limited to the explicitly authorized parent")
        cpus = cpu_sets[name]
        if len(cpus) != runtime["cpu_threads"] or len(set(cpus)) != len(cpus):
            raise ValueError("Each control needs its own complete CPU set")
        expected = f"results/{branch['parent_batch']}/{name}/training/checkpoint-0020000.pt"
        if branch["checkpoints"][name] != expected:
            raise ValueError("Each control must branch from its own exact 20k checkpoint")
        output = Path(config["models"][name]["checkpoint"])
        if not output.is_relative_to(Path("results") / config["batch"] / name):
            raise ValueError("Control checkpoints must stay in their own new output directory")


def verify_parent(batch, config):
    parent = batch.parent / config["branch"]["parent_batch"]
    manifest = parent / "source-manifest.json"
    branch = config["branch"]
    if sha256(manifest) != branch["parent_manifest_sha256"]:
        raise ValueError("Parent source manifest differs from the pinned original")
    frozen = read_json(manifest)
    if frozen["config_sha256"] != branch["parent_config_sha256"]:
        raise ValueError("Parent configuration digest differs from the pinned original")
    for relative, digest in frozen["files"].items():
        path = parent / "source" / relative
        if not path.resolve().is_relative_to((parent / "source").resolve()):
            raise ValueError("Parent manifest contains a path outside its frozen source")
        if sha256(path) != digest:
            raise ValueError(f"Parent frozen source changed: {relative}")
    if sha256(parent / "source/configs/experiment.json") != branch["parent_config_sha256"]:
        raise ValueError("Parent frozen configuration changed")
    return parent


def freeze_source(batch, config_path, parent):
    """Freeze once; existing bytes are verified and never silently refreshed."""
    source = batch / "source"
    manifest = batch / "source-manifest.json"
    if manifest.exists():
        frozen = read_json(manifest)
        if sha256(config_path) != frozen["config_sha256"]:
            raise ValueError("Configuration changed; create a separately named amended batch")
        for relative, expected in frozen["files"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        return source / "configs/experiment.json"
    source.mkdir(exist_ok=False)
    source.chmod(0o755)  # NFS root_squash: Docker must traverse the source mount.
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (source / "scripts").mkdir()
    (source / "configs").mkdir()
    for name in ("run_ttcd_second_hop.py", "prepare_ttcd_scratch.py", "launch_ttcd_branch.py"):
        shutil.copy2(ROOT / "scripts" / name, source / "scripts" / name)
    shutil.copy2(config_path, source / "configs/experiment.json")
    shutil.copy2(
        ROOT / "configs/experiment-tracking-defaults.json",
        source / "configs/experiment-tracking-defaults.json",
    )
    config = read_json(config_path)
    backup = source / "input-backup"
    backup.mkdir()
    backup.chmod(0o755)  # NFS root_squash: the daemon must traverse file-bind parents.
    for name in ("world.npz", "panels.npz", "complete.json", "training-split.json"):
        current = ROOT / config["data_dir"] / name
        original = parent / "source/input-backup" / name
        if sha256(current) != sha256(original):
            raise ValueError(f"The paired parent input changed: {name}")
        shutil.copy2(original, backup / name)
    current = ROOT / config["model_config"]
    original = parent / "source/input-backup/model-config.json"
    if sha256(current) != sha256(original):
        raise ValueError("The paired parent model configuration changed")
    shutil.copy2(original, backup / "model-config.json")
    write_json(
        source / "controller-contract.json",
        {
            "schema": "ttcd-reader-control-controller-v1",
            "stages": list(STAGES),
            "calibration_exit3_stops_arm": True,
            "checkpoint_publication": "exact step20000 .pt published by atomic rename",
            "checkpoint_waiters": "independent per-model threads",
            "allowed_parent_containers": PARENTS,
            "gpu_sharing": "only host CUDA PIDs belonging to the named parent",
            "locks": "new batch/arm locks and short shared-admission locks; parent locks unchanged",
            "automatic_retry": False,
            "hard_timeout_action": "stop only own currently monitored named container",
            "image": config["branch"]["parent_image_sha256"],
            "runtime": config["runtime"],
        },
    )
    write_json(
        manifest,
        {
            "config_sha256": sha256(config_path),
            "parent_manifest_sha256": config["branch"]["parent_manifest_sha256"],
            "controller_contract_sha256": sha256(source / "controller-contract.json"),
            "files": {
                str(path.relative_to(source)): sha256(path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            },
        },
    )
    return source / "configs/experiment.json"


def inspect_container(name):
    result = docker("inspect", name, check=False)
    if result.returncode:
        # An unreachable daemon is not evidence that a parent container stopped.
        if "No such" not in result.stderr:
            raise RuntimeError(f"Cannot inspect container {name}: {result.stderr.strip()}")
        return None
    return json.loads(result.stdout)[0]


def prepare_input_mount(batch, config):
    """Copy exact frozen input bytes into a traversable, separate data overlay.

    The scientific source/configuration remains frozen. A directory overlay
    avoids traversing private live data parents during Docker's mount setup.
    """
    data_dir = Path(config["data_dir"])
    model_config = Path(config["model_config"])
    for path in (data_dir, model_config):
        if path.is_absolute() or ".." in path.parts or path.parts[0] != "data":
            raise ValueError("The authorized input overlay supports only relative data/ paths")
    if model_config.parent != data_dir:
        raise ValueError("The model configuration must remain inside the paired data directory")
    mount_root = batch / "input-mount"
    data_root = mount_root / "data"
    directory = mount_root / data_dir
    expected_files = {
        data_dir / name: batch / "source/input-backup" / name
        for name in ("world.npz", "panels.npz", "complete.json", "training-split.json")
    }
    expected_files[model_config] = batch / "source/input-backup/model-config.json"
    expected = {
        "schema": "ttcd-reader-control-input-mount-v1",
        "frozen_source_manifest_sha256": sha256(batch / "source-manifest.json"),
        "files": {str(relative): sha256(original) for relative, original in expected_files.items()},
    }
    manifest = batch / "input-mount-manifest.json"
    if manifest.exists() and read_json(manifest) != expected:
        raise ValueError("The recorded input mount differs from the immutable frozen inputs")
    # Apply traversal only to our new tree, never to the private original data.
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
            if sha256(target) != expected["files"][str(relative)]:
                raise ValueError(f"Input mount copy differs from its frozen input: {relative}")
    if not manifest.exists():
        write_json(manifest, expected)
    return data_root


def wait_for_checkpoint(path, parent_state, deadline, update=lambda _: None, poll_seconds=15):
    """Only final .pt names are published; the frozen parent uses atomic rename."""
    path = Path(path)
    while True:
        if path.is_file() and path.stat().st_size > 0:
            return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Hard timeout while waiting for the parent's published 20k checkpoint"
            )
        state = parent_state()
        if state is None or not state["State"]["Running"]:
            raise RuntimeError("Parent stopped before publishing the required 20k checkpoint")
        update({"state": "waiting_for_parent_20000", "checkpoint": str(path)})
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))


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
        raise RuntimeError("Requested control CPUs are outside this process's available affinity")
    for container in containers:
        if not container["State"]["Running"]:
            continue
        value = container["HostConfig"].get("CpusetCpus", "")
        if not value:
            raise RuntimeError(f"Cannot verify CPU isolation for {container['Name']}")
        if requested & parse_cpu_set(value):
            raise RuntimeError(
                f"Control CPU set collides with running container {container['Name']}"
            )


def gpu_snapshot():
    rows = {}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        index, uuid, free, total = (value.strip() for value in line.split(","))
        rows[int(index)] = {"uuid": uuid, "free_mib": int(free), "total_mib": int(total)}
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


def check_shared_gpu(gpu, runtime, snapshot, allowed_pids):
    rows, processes = snapshot
    if gpu not in rows:
        raise ValueError(f"Requested GPU {gpu} does not exist on this host")
    row = rows[gpu]
    unrelated = [
        process["pid"]
        for process in processes
        if process["uuid"] == row["uuid"] and process["pid"] not in allowed_pids
    ]
    if unrelated:
        raise RuntimeError(f"GPU {gpu} has unrelated CUDA processes: {unrelated}; refusing sharing")
    if row["free_mib"] < runtime["shared_gpu_min_free_mib"]:
        raise RuntimeError(f"GPU {gpu} lacks reserved free memory for another control")
    return row


def running_containers():
    ids = docker("ps", "--quiet").stdout.split()
    return json.loads(docker("inspect", *ids).stdout) if ids else []


def container_pids(container):
    result = docker("top", container["Id"], "-eo", "pid")
    pids = {int(line.strip()) for line in result.stdout.splitlines()[1:] if line.strip()}
    # Docker top reports host PIDs for all current container descendants.
    pids.add(int(container["State"]["Pid"]))
    return pids


def admit_shared_job(config, name, image):
    runtime = config["runtime"]
    containers = running_containers()
    check_cpu_sets(runtime["cpu_sets"][name], containers, os.sched_getaffinity(0))
    allowed_pids = set()
    for container in containers:
        if container["Name"].lstrip("/") in runtime["shared_gpu_allowlist"][name]:
            if container["Image"] != image:
                raise ValueError("Whitelisted parent image differs from the pinned image")
            allowed_pids.update(container_pids(container))
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available_kib < runtime["memory_gib"] * 1024**2:
        raise RuntimeError("Insufficient available host memory for another control")
    snapshot = gpu_snapshot()
    row = check_shared_gpu(runtime["gpus"][name], runtime, snapshot, allowed_pids)
    return {
        **row,
        "admission_utc": utc_now(),
        "allowed_parent": PARENTS[name],
        "allowed_host_pids": sorted(allowed_pids),
        "observed_gpu_processes": [p for p in snapshot[1] if p["uuid"] == row["uuid"]],
        "cpu_set": runtime["cpu_sets"][name],
        "host_available_kib": available_kib,
        "minimum_free_gpu_mib": runtime["shared_gpu_min_free_mib"],
    }


def reserve_stage(out, stage):
    if stage == "train" and (out / "training").exists():
        raise FileExistsError("Control training outputs already exist; no retry or overwrite")
    # An exclusive claim survives process failure, so a later invocation cannot
    # accidentally repeat the same stage or replace its scientific outputs.
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
        f"{os.getuid()}:{os.getgid()}",
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
        "ttcd-branch-entry",
        "python",
        "/frozen/scripts/run_ttcd_second_hop.py",
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


def monitor_container(container_name, log, deadline):
    follower = subprocess.Popen(
        [*DOCKER, "logs", "--follow", container_name], stdout=log, stderr=log
    )
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(container_name, 0)
        completed = subprocess.run(
            [*DOCKER, "wait", container_name],
            capture_output=True,
            text=True,
            timeout=remaining,
            check=True,
        )
        return int(completed.stdout.strip())
    except subprocess.TimeoutExpired:
        # Only this arm's own named container can be stopped at its hard timeout.
        write_json(
            Path(log.name).with_name("hard-timeout.json"),
            {"container": container_name, "notice_utc": utc_now(), "action": "stop own container"},
        )
        print(f"Hard timeout: stopping own control container {container_name}", flush=True)
        docker("stop", "--time", "30", container_name, check=False)
        raise
    finally:
        try:
            follower.wait(timeout=45)
        except subprocess.TimeoutExpired:
            follower.terminate()  # Log-following client only, never a training process.
            follower.wait(timeout=10)


def run_arm(config, name, stages, image, batch, parent, token_file):
    out = batch / name
    out.mkdir(exist_ok=True)
    out.chmod(0o755)
    state_path = out / "controller-state.json"
    deadline = time.monotonic() + config["runtime"]["timeout_hours"] * 3600

    def update(value):
        write_json(state_path, {"model": name, "updated_utc": utc_now(), **value})

    try:
        with (out / "controller.lock").open("a") as arm_lock:
            fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if "train" in stages:
                checkpoint = parent / name / "training/checkpoint-0020000.pt"
                receipt = wait_for_checkpoint(
                    checkpoint, lambda: inspect_container(PARENTS[name]), deadline, update
                )
                write_json(out / "parent-checkpoint-receipt.json", receipt)
            for stage in stages:
                prepare_input_mount(batch, config)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Hard timeout before starting next stage")
                reserve_stage(out, stage)
                container_name = f"{config['batch']}-{name}-{stage}"
                if inspect_container(container_name) is not None:
                    raise FileExistsError("The control container already exists; no resubmission")
                lock_dir = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
                lock_dir.mkdir(exist_ok=True)
                # This separate short lock serializes intentional shared admission;
                # the original parent's held gpu-N.lock is left completely alone.
                gpu = config["runtime"]["gpus"][name]
                with (lock_dir / f"shared-admission-gpu-{gpu}.lock").open("a") as admission:
                    fcntl.flock(admission, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    row = admit_shared_job(config, name, image)
                    write_json(out / f"admission-{stage}.json", row)
                    _, command = build_command(
                        config, name, stage, image, row, out, batch, parent, token_file
                    )
                    update(
                        {
                            "state": "launching",
                            "stage": stage,
                            "container": container_name,
                            "gpu": gpu,
                            "gpu_uuid": row["uuid"],
                            "cpu_set": config["runtime"]["cpu_sets"][name],
                            "image": image,
                        }
                    )
                    with (out / f"{stage}.log").open("x") as log:
                        launched = subprocess.run(
                            command, capture_output=True, text=True, timeout=60
                        )
                        log.write(launched.stderr)
                        log.flush()
                        if launched.returncode:
                            raise RuntimeError(f"Control launch failed: {launched.stderr.strip()}")
                    container_id = launched.stdout.strip()
                update(
                    {
                        "state": "running",
                        "stage": stage,
                        "container": container_name,
                        "container_id": container_id,
                        "gpu": gpu,
                        "gpu_uuid": row["uuid"],
                        "cpu_set": config["runtime"]["cpu_sets"][name],
                        "image": image,
                    }
                )
                with (out / f"{stage}.log").open("a") as log:
                    code = monitor_container(container_name, log, deadline)
                write_json(
                    out / f"exit-{stage}.json",
                    {
                        "returncode": code,
                        "gpu": gpu,
                        "gpu_uuid": row["uuid"],
                        "cpu_set": config["runtime"]["cpu_sets"][name],
                        "memory_gib": config["runtime"]["memory_gib"],
                        "container": container_name,
                        "container_id": container_id,
                        "image": image,
                        "finished_utc": utc_now(),
                    },
                )
                if code:
                    update(
                        {
                            "state": "calibration_failed"
                            if stage == "calibrate" and code == 3
                            else "needs_attention",
                            "stage": stage,
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
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ttcd-second-hop-reader-control-v1.json"
    )
    parser.add_argument("--model", choices=("discoloop", "loop", "both"), default="both")
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
    models = list(PARENTS) if args.model == "both" else [args.model]
    stages = list(STAGES) if args.stage == "pipeline" else [args.stage]
    plan = {
        "batch": config["batch"],
        "stages": stages,
        "docker_context": "lm-memory",
        "image": config["branch"]["parent_image_sha256"],
        "jobs": [
            {
                "model": name,
                "gpu": config["runtime"]["gpus"][name],
                "cpu_set": config["runtime"]["cpu_sets"][name],
                "parent_checkpoint": config["branch"]["checkpoints"][name],
                "shared_parent": PARENTS[name],
            }
            for name in models
        ],
        "executed": args.execute,
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
    if image != config["branch"]["parent_image_sha256"]:
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
                "A control stopped; preserved logs explain the failure. No retry occurred."
            )


if __name__ == "__main__":
    main()
