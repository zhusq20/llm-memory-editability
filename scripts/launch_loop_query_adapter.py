#!/usr/bin/env python3
"""Freeze and run one300k stateless Q-adapter control alongside the existing run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from resume_loop_query_control_transport import secret_archive

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-query-adapter-development-v1"
PARENT = "loop-query-control-development-v1"
DOCKER = ["docker", "--context", "lm-memory"]
STAGES = ("contracts", "preflight", "train", "audit", "cloud-audit")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docker(*args):
    return subprocess.run([*DOCKER, *args], check=True, capture_output=True, text=True, timeout=60)


def validate(config):
    rt = config["runtime"]
    if (
        config["batch"] != BATCH
        or config["spec"]["steps"] != 300000
        or set(config["arms"]) != {"query_adapter"}
        or rt["docker_context"] != "lm-memory"
        or rt["gpu"] != 4
        or rt["cpu_set"] != [84, 85, 86, 87]
        or rt["memory_gib"] != 96
        or rt["tracking"] != "online"
    ):
        raise ValueError("Only the authorized one300k stateless adapter with isolated CPUs")
    expected = [f"{PARENT}-train-capture-v3", f"{PARENT}-audit-capture-v3"]
    if rt["shared_gpu_peer_batch"] != PARENT or rt["shared_gpu_peer_containers"] != expected:
        raise ValueError("Only the exact existing same-project control run may co-locate")
    if config["arms"]["query_adapter"] != {
        "kind": "none",
        "coefficient": 0.0,
        "scope": "supervised",
        "first_round_kv": False,
        "discoloop": False,
        "adapter_dim": 48,
        "adapter_seed": 20261016,
    }:
        raise ValueError("Fixed nonlinear shared48 bottleneck; no other mechanisms or labels")
    if rt["minimum_free_gpu_mib"] < 16000 or rt["peer_max_gpu_memory_mib"] > 16000:
        raise ValueError("Keep original conservative memory limits")


def cpu_set(value):
    result = set()
    for part in value.split(","):
        if part:
            ends = list(map(int, part.split("-")))
            result.update(range(ends[0], ends[-1] + 1))
    return result


def resources(config):
    rt = config["runtime"]
    ids = docker("ps", "-q").stdout.split()
    containers = json.loads(docker("inspect", *ids).stdout) if ids else []
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    rows = [[value.strip() for value in line.split(",")] for line in raw.splitlines()]
    gpu = next(row for row in rows if int(row[0]) == rt["gpu"])
    wanted_cpus = set(rt["cpu_set"])
    if not wanted_cpus <= set(os.sched_getaffinity(0)):
        raise ValueError("CPU affinity does not include requested disjoint CPUs")
    peers = []
    peer_pids = set()
    for item in containers:
        value = item["HostConfig"].get("CpusetCpus", "")
        if not value:
            raise RuntimeError(f"Unknown CPU ownership: {item['Name']}")
        if wanted_cpus & cpu_set(value):
            return None
        owns = any(
            request.get("Count") == -1
            or {str(rt["gpu"]), gpu[1]} & set(request.get("DeviceIDs") or [])
            for request in item["HostConfig"].get("DeviceRequests") or []
        )
        if owns:
            name = item["Name"].lstrip("/")
            if name not in rt["shared_gpu_peer_containers"]:
                return None
            peers.append(name)
            for line in docker("top", name, "-eo", "pid").stdout.splitlines()[1:]:
                if line.strip().isdigit():
                    peer_pids.add(int(line.strip()))
    process_text = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    gpu_pids = {
        int(line.split(",")[1])
        for line in process_text.splitlines()
        if line.split(",")[0].strip() == gpu[1]
    }
    if not gpu_pids <= peer_pids or int(gpu[3]) < rt["minimum_free_gpu_mib"]:
        return None
    if peers:
        if int(gpu[2]) > rt["peer_max_gpu_memory_mib"]:
            return None
    elif int(gpu[2]) > 1024 or int(gpu[4]) > 5:
        return None
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < rt["memory_gib"] * 1024**2:
        return None
    return {
        "gpu": rt["gpu"],
        "uuid": gpu[1],
        "gpu_rows": rows,
        "shared_with": peers,
        "peer_gpu_pids": sorted(gpu_pids),
        "cpus": rt["cpu_set"],
        "utc": utc(),
    }


def freeze(config, batch):
    parent = batch.parent / PARENT
    original = read(parent / f"source/configs/{PARENT}.json")
    if read(parent / "controller-state.json")["state"] == "needs_attention":
        raise ValueError("Parent needs attention; do not silently change comparison")
    if any(
        value != original["spec"][key] for key, value in config["spec"].items() if key != "name"
    ):
        raise ValueError("Training recipe must exactly match the running controller")
    for key in (
        "dropout_seed",
        "evaluation_nodes",
        "checkpoint_nodes",
        "heartbeat_steps",
        "evaluation_batch_size",
        "audit_loss_atol",
        "audit_loss_rtol",
    ):
        if config[key] != original[key]:
            raise ValueError(f"Parent training/evaluation convention changed: {key}")
    if config["runtime"]["image"] != original["runtime"]["image"]:
        raise ValueError("Comparison requires same pinned image")
    source = batch / "source"
    source.mkdir(exist_ok=False)
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in (
        f"configs/{BATCH}.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
        "scripts/run_loop_query_adapter.py",
        "scripts/run_supervised_alignment.py",
        "scripts/launch_loop_query_adapter.py",
        "scripts/resume_loop_query_control_transport.py",
        "tests/test_loop_query_adapter.py",
        "pyproject.toml",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(parent / f"source/configs/{PARENT}.json", source / f"configs/{PARENT}.json")
    shutil.copytree(parent / "input-mount", batch / "input-mount")
    frozen_parent = read(parent / "freeze.json")
    for name in (
        "src/llm_memory_editability/grokking_reproduction.py",
        "src/llm_memory_editability/grok_depth.py",
        "src/llm_memory_editability/bios_model.py",
        "src/llm_memory_editability/supervised_alignment.py",
        "scripts/run_supervised_alignment.py",
    ):
        if sha(source / name) != frozen_parent["files"][f"source/{name}"]:
            raise ValueError(f"Shared training/scoring implementation changed: {name}")
    for path in (batch / "input-mount").rglob("*"):
        if path.is_file() and sha(path) != frozen_parent["files"][str(path.relative_to(batch))]:
            raise ValueError("The parent's exact input bytes must be reused")
    if (
        sha(batch / "input-mount/initial/loop.pt")
        != config["paired_control"]["initial_source_sha256"]
    ):
        raise ValueError("Wrong exact step-zero checkpoint")
    write(
        batch / "paired-control.json",
        {
            "batch": PARENT,
            "freeze_sha256": sha(parent / "freeze.json"),
            "config": original,
            "source_model_sha256": frozen_parent["files"][
                "source/src/llm_memory_editability/loop_query_control.py"
            ],
            "initial_checkpoint_sha256": sha(batch / "input-mount/initial/loop.pt"),
            "world_sha256": sha(
                batch / "input-mount/data/discoloop-success-restore-v1/original/data/world.npz"
            ),
            "panels_sha256": sha(
                batch / "input-mount/data/discoloop-success-restore-v1/original/data/panels.npz"
            ),
            "controller_restarted": False,
        },
    )
    (batch / "run").mkdir()
    # Prepare standard bind permissions only inside this new batch.
    for folder in (batch, source, batch / "input-mount", batch / "run"):
        folder.chmod(0o755)
        if folder.name in ("source", "input-mount"):
            for path in folder.rglob("*"):
                if path.is_dir():
                    path.chmod(0o755)
                elif path.is_file():
                    path.chmod(0o644)
    paths = [
        p for folder in (source, batch / "input-mount") for p in folder.rglob("*") if p.is_file()
    ]
    write(
        batch / "freeze.json",
        {
            "image": config["runtime"]["image"],
            "utc": utc(),
            "files": {str(p.relative_to(batch)): sha(p) for p in sorted(paths)},
        },
    )


def verify(batch):
    freeze = read(batch / "freeze.json")
    for relative, expected in freeze["files"].items():
        if sha(batch / relative) != expected:
            raise ValueError(f"Frozen file changed: {relative}")
    if json.loads(docker("image", "inspect", freeze["image"]).stdout)[0]["Id"] != freeze["image"]:
        raise ValueError("Fixed image changed")


def command(config, batch, stage, row=None):
    rt = config["runtime"]
    name = f"{BATCH}-{stage}"
    args = [
        *DOCKER,
        "create",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        "4",
        "--cpuset-cpus",
        ",".join(map(str, rt["cpu_set"])),
        "--memory",
        "96g",
        "--memory-swap",
        "96g",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
    ]
    if row is not None:
        args += ["--gpus", f"device={row['uuid']}"]
    for host, target, readonly in (
        (batch / "source", "/frozen", True),
        (batch / "input-mount", "/inputs", True),
        (batch / "run", "/output", False),
    ):
        args += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    for env in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "XDG_CACHE_HOME=/tmp/lm-adapter-cache",
        "WANDB_CONFIG_DIR=/tmp/lm-adapter-config",
        "WANDB_CACHE_DIR=/tmp/lm-adapter-wandb-cache",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
    ):
        args += ["--env", env]
    args += [
        rt["image"],
        "python",
        "-c",
        "import os,sys;from pathlib import Path;"
        "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
        "os.execvp(sys.argv[1],sys.argv[1:])",
        "python",
    ]
    if stage == "contracts":
        args += [
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "/frozen/tests/test_loop_query_adapter.py",
        ]
    else:
        args += [
            "/frozen/scripts/run_loop_query_adapter.py",
            stage,
            "--config",
            f"/frozen/configs/{BATCH}.json",
            "--out",
            "/output",
        ]
    return name, args


def execute(config, batch, stage, row):
    verify(batch)
    name, args = command(config, batch, stage, row)
    folder = batch / "execution" / stage
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "command.json", {"argv": args, "utc": utc()})
    try:
        created = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as error:
        write(
            folder / "create-failure.json",
            {"stderr": error.stderr, "returncode": error.returncode, "utc": utc()},
        )
        raise
    subprocess.run(
        [*DOCKER, "cp", "--archive", "-", f"{name}:/"],
        input=secret_archive(),
        check=True,
        capture_output=True,
        timeout=60,
    )
    docker("start", name)
    write(
        folder / "launch.json",
        {"container": name, "id": created.stdout.strip(), "utc": utc(), "resource": row},
    )
    if row and row["shared_with"]:
        note = {
            "peer_batch": PARENT,
            "new_batch": BATCH,
            "container": name,
            "utc": utc(),
            "gpu": 4,
            "peer_containers": row["shared_with"],
            "cpus": [84, 85, 86, 87],
            "scientific_source_or_recipe_changed": False,
            "exclusive_timing_claim": False,
            "reason": "Authorized additional300k parameter-matched Q-adapter control "
            "on available same-project GPU memory",
        }
        write(batch / f"co-location-{stage}.json", note)
        # Record external scheduling without editing parent science or measurements.
        write(batch.parent / PARENT / f"schedule-amendment-{BATCH}-{stage}.json", note)
    started = time.monotonic()
    with (folder / "stdout.log").open("x") as log:
        follower = subprocess.Popen(
            [*DOCKER, "logs", "--follow", name], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while True:
                state = json.loads(docker("inspect", name).stdout)[0]["State"]
                write(
                    folder / "heartbeat.json",
                    {
                        "container": name,
                        "state": state,
                        "utc": utc(),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
                if not state["Running"]:
                    write(folder / "exit.json", {"code": state["ExitCode"], "utc": utc()})
                    if state["ExitCode"]:
                        raise RuntimeError(f"{stage} failed; no automatic retry")
                    break
                if time.monotonic() - started > config["runtime"]["timeout_hours"] * 3600:
                    write(
                        folder / "timeout-notice.json",
                        {"stop_only_own_container": name, "utc": utc()},
                    )
                    docker("stop", "--time", "30", name)
                    raise TimeoutError("Own container hard timeout")
                telemetry = subprocess.run(
                    [
                        "nvidia-smi",
                        "--id=4",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout.strip()
                with (folder / "gpu-resources.jsonl").open("a") as resources_log:
                    resources_log.write(
                        json.dumps({"utc": utc(), "device_total": telemetry, "shared_device": True})
                        + "\n"
                    )
                time.sleep(30)
        finally:
            try:
                follower.wait(timeout=45)
            except subprocess.TimeoutExpired:
                follower.terminate()
                follower.wait(timeout=10)


def compare(config, batch):
    parent = batch.parent / PARENT
    deadline = time.monotonic() + config["runtime"]["comparison_wait_hours"] * 3600
    while not (parent / "completion-manifest.json").exists():
        state = read(parent / "controller-state.json")
        if state["state"] == "needs_attention":
            raise RuntimeError("Parent verification needs attention; no unsupported comparison")
        if time.monotonic() > deadline:
            raise TimeoutError("Waiting for parent audited endpoint")
        write(
            batch / "controller-state.json",
            {"state": "waiting_for_paired_control_audit", "utc": utc()},
        )
        time.sleep(30)
    complete = read(parent / "completion-manifest.json")
    if not complete.get("independent_audit_passed") or not complete.get("cloud_verified"):
        raise ValueError("Parent endpoint, audit and cloud verification must pass separately")
    control = read(parent / "run/result-train.json")
    adapter = read(batch / "run/result-train.json")
    if control["step"] != 300000 or adapter["step"] != 300000:
        raise ValueError("Both fixed endpoints must be300k")
    panels = {
        name: {
            "query_control": metrics["accuracy"],
            "query_adapter": adapter["endpoint"]["metrics"][name]["accuracy"],
            "control_minus_adapter_pp": 100
            * (metrics["accuracy"] - adapter["endpoint"]["metrics"][name]["accuracy"]),
        }
        for name, metrics in control["endpoint"]["metrics"].items()
    }
    for key in (
        "examples",
        "counts",
        "atomic_epochs",
        "composition_epochs",
        "effective_input_tokens",
        "supervised_tokens",
        "executed_input_tokens",
        "masked_batch_slots",
    ):
        if control["endpoint"][key] != adapter["endpoint"][key]:
            raise ValueError(f"Actual exposure differs: {key}")
    write(
        batch / "comparison.json",
        {
            "fixed_step": 300000,
            "panels": panels,
            "parameters": {"control": control["parameters"], "adapter": adapter["parameters"]},
            "actual_exposure_exact": True,
            "parent_retrained": False,
            "new_scientific_updates": 300000,
            "initializations": 1,
            "worlds": 1,
            "mechanism_proven": False,
            "limits": "Near-parameter-matched ordinary adapter comparison; input skip, "
            "LN/RMS inputs, update cadence and compute differ. Persistence alone is not isolated. "
            "Shared GPU timing is not exclusive throughput.",
            "parent_train_sha256": sha(parent / "run/result-train.json"),
            "adapter_train_sha256": sha(batch / "run/result-train.json"),
            "utc": utc(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    validate(config)
    if not args.execute:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    root = Path(config["runtime"]["persistent_root"])
    if not root.resolve().is_relative_to("/ossfs/workspace") or not any(
        p.is_mount() for p in (root, *root.parents) if p != Path("/")
    ):
        raise ValueError("Actual persistent LM mount required")
    batch = root / BATCH
    if batch.is_symlink():
        raise ValueError("No redirected output batch")
    batch.mkdir(exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("Do not duplicate submission")
        if shutil.disk_usage(batch).free < 20 * 1024**3:
            raise RuntimeError("Need20GiB persistent space")
        freeze(config, batch)
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "freeze_sha256": sha(batch / "freeze.json"),
                "automatic_retry": False,
            },
        )
        state = {"state": "running", "utc": utc()}
        try:
            for stage in STAGES:
                deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
                row = None
                while row is None:
                    row = resources(config)
                    if row is None:
                        state.update(state="waiting_for_resources", stage=stage, utc=utc())
                        write(batch / "controller-state.json", state)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Resource waiting timeout")
                        time.sleep(30)
                write(batch / f"resource-admission-{stage}.json", row)
                state.update(state="running", stage=stage, utc=utc())
                write(batch / "controller-state.json", state)
                execute(
                    config, batch, stage, None if stage in ("contracts", "cloud-audit") else row
                )
                if (
                    stage == "preflight"
                    and read(batch / "run/result-preflight.json")["passed"] is not True
                ):
                    raise ValueError("Full-size zero-update preflight must pass")
            compare(config, batch)
            state.update(
                state="complete",
                new_scientific_updates=300000,
                independent_audit_passed=True,
                cloud_verified=True,
                paired_comparison_complete=True,
                utc=utc(),
            )
            write(batch / "controller-state.json", state)
            write(batch / "completion-manifest.json", state)
        except BaseException as error:
            state.update(
                state="needs_attention",
                error=repr(error),
                traceback=traceback.format_exc(),
                utc=utc(),
            )
            write(batch / "controller-state.json", state)
            raise


if __name__ == "__main__":
    main()
