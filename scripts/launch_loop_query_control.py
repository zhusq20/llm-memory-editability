#!/usr/bin/env python3
"""Freeze, validate, run and independently audit one authorized 300k experiment."""

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

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-query-control-development-v1"
DOCKER = ["docker", "--context", "lm-memory"]


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


def resources(config):
    ids = docker("ps", "-q").stdout.split()
    containers = json.loads(docker("inspect", *ids).stdout) if ids else []
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [[x.strip() for x in line.split(",")] for line in raw.stdout.splitlines()]
    wanted = config["runtime"]["gpu"]
    row = next(r for r in rows if int(r[0]) == wanted)
    if int(row[2]) > 1024 or int(row[3]) > 5:
        return None
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    if any(line.split(",")[0].strip() == row[1] for line in processes.splitlines()):
        return None
    wanted_cpus = set(config["runtime"]["cpu_set"])
    if not wanted_cpus <= set(os.sched_getaffinity(0)):
        raise ValueError("Requested CPUs outside available affinity")
    for item in containers:
        for request in item["HostConfig"].get("DeviceRequests") or []:
            if request.get("Count") == -1 or {str(wanted), row[1]} & set(
                request.get("DeviceIDs") or []
            ):
                return None
        cpu_string = item["HostConfig"].get("CpusetCpus", "")
        if not cpu_string:
            raise ValueError(f"Cannot verify CPU ownership for {item['Name']}")
        used = set()
        for part in cpu_string.split(","):
            bounds = list(map(int, part.split("-")))
            used.update(range(bounds[0], bounds[-1] + 1))
        if used & wanted_cpus:
            return None
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < config["runtime"]["memory_gib"] * 1024**2:
        return None
    return {
        "gpu": wanted,
        "uuid": row[1],
        "gpu_rows": rows,
        "cpus": sorted(wanted_cpus),
        "containers": [item["Name"] for item in containers],
        "utc": utc(),
    }


def freeze(config, batch):
    if (batch / "source").exists():
        raise FileExistsError("Existing frozen source: preserve previous attempt")
    source = batch / "source"
    source.mkdir()
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in (
        f"configs/{BATCH}.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
        f"docs/development-artifacts/{BATCH}/source-plan.md",
        "scripts/run_loop_query_control.py",
        "scripts/run_supervised_alignment.py",
        "scripts/launch_loop_query_control.py",
        "tests/test_loop_query_control.py",
        "pyproject.toml",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    root = batch.parent
    old_data = (
        root / "ttcd-successful-base-v1/input-mount/data/discoloop-success-restore-v1/original"
    )
    data = batch / "input-mount/data/discoloop-success-restore-v1/original"
    for relative in (
        "data/complete.json",
        "data/world.npz",
        "data/panels.npz",
        "reference/gpt2/config.json",
    ):
        target = data / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_data / relative, target)
    initial = batch / "input-mount/initial/loop.pt"
    initial.parent.mkdir(parents=True)
    original_initial = root / "ttcd-architecture-comparison-v1/input-mount/initial/loop.pt"
    shutil.copy2(original_initial, initial)
    baseline = root / config["baseline_reuse"]["batch"]
    train = read(baseline / config["baseline_reuse"]["train_result"])
    audit = read(baseline / config["baseline_reuse"]["audit_result"])
    checkpoint = baseline / config["baseline_reuse"]["checkpoint"]
    if (
        train["step"] != 300000
        or audit["passed"] is not True
        or audit["checkpoint_sha256"] != sha(checkpoint)
    ):
        raise ValueError("Reused baseline must have a bound, passing terminal audit")
    for key, value in config["spec"].items():
        if (
            key != "name"
            and read(baseline / "source/configs/foundation-loop.json")["spec"][key] != value
        ):
            raise ValueError(f"Baseline recipe differs: {key}")
    write(
        batch / "baseline-reuse.json",
        {
            "training_restarted": False,
            "train_result": train,
            "audit_result": audit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha(checkpoint),
            "initial_checkpoint_sha256": sha(initial),
            "initial_original_sha256": sha(original_initial),
            "limits": "Same observed graph, initialization, data stream and recipe; "
            "separate execution; "
            "controller adds parameters, state and compute. No multi-seed generality claim.",
        },
    )
    files = [
        p for folder in (source, batch / "input-mount") for p in folder.rglob("*") if p.is_file()
    ]
    write(
        batch / "freeze.json",
        {
            "utc": utc(),
            "image": config["runtime"]["image"],
            "files": {str(p.relative_to(batch)): sha(p) for p in sorted(files)},
        },
    )


def verify(batch):
    manifest = read(batch / "freeze.json")
    for relative, expected in manifest["files"].items():
        if sha(batch / relative) != expected:
            raise ValueError(f"Frozen source/input changed: {relative}")
    if (
        json.loads(docker("image", "inspect", manifest["image"]).stdout)[0]["Id"]
        != manifest["image"]
    ):
        raise ValueError("Fixed image mismatch")


def command(config, batch, stage, token, row=None):
    runtime = config["runtime"]
    name = f"{BATCH}-{stage}"
    args = [
        *DOCKER,
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        "4",
        "--cpuset-cpus",
        ",".join(map(str, runtime["cpu_set"])),
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
        (token.resolve(), "/run/secrets/wandb", True),
    ):
        args += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    for env in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "XDG_CACHE_HOME=/tmp/lm-control-cache",
        "WANDB_CONFIG_DIR=/tmp/lm-control-config",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        "WANDB_DISABLE_CODE=true",
    ):
        args += ["--env", env]
    # The credential is read only within the container, never included in arguments/logs.
    args += [
        runtime["image"],
        "bash",
        "-c",
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"\nexec "$@"',
        "query-control-entry",
    ]
    if stage == "contracts":
        args += [
            "python",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "/frozen/tests/test_loop_query_control.py",
        ]
    else:
        args += [
            "python",
            "/frozen/scripts/run_loop_query_control.py",
            stage,
            "--config",
            f"/frozen/configs/{BATCH}.json",
            "--out",
            "/output",
        ]
    return name, args


def execute(config, batch, stage, token, row):
    verify(batch)
    name, args = command(config, batch, stage, token, row)
    folder = batch / "execution" / stage
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "command.json", {"command": args, "utc": utc()})
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
    started = time.monotonic()
    heartbeat = folder / "heartbeat.json"
    with (folder / "stdout.log").open("x") as log:
        follower = subprocess.Popen(
            [*DOCKER, "logs", "--follow", name], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while True:
                item = json.loads(docker("inspect", name).stdout)[0]
                state = item["State"]
                write(
                    heartbeat,
                    {
                        "container": name,
                        "running": state["Running"],
                        "elapsed_seconds": time.monotonic() - started,
                        "utc": utc(),
                    },
                )
                if not state["Running"]:
                    write(
                        folder / "exit.json",
                        {"code": state["ExitCode"], "state": state, "utc": utc()},
                    )
                    if state["ExitCode"] != 0:
                        raise RuntimeError(
                            f"{stage} failed with exit code {state['ExitCode']}; retained"
                        )
                    break
                if time.monotonic() - started > config["runtime"]["timeout_hours"] * 3600:
                    write(
                        folder / "timeout-notice.json",
                        {"action": "stop only own container", "utc": utc()},
                    )
                    docker("stop", "--time", "30", name)
                    raise TimeoutError(f"Own container hard timeout: {name}")
                time.sleep(30)
        finally:
            try:
                follower.wait(timeout=45)
            except subprocess.TimeoutExpired:
                follower.terminate()
                follower.wait(timeout=10)


def compare(config, batch):
    before = read(batch / "baseline-reuse.json")["train_result"]
    after = read(batch / "run/result-train.json")
    panels = {
        name: {
            "baseline": values["accuracy"],
            "query_control": after["endpoint"]["metrics"][name]["accuracy"],
            "difference_pp": 100
            * (after["endpoint"]["metrics"][name]["accuracy"] - values["accuracy"]),
        }
        for name, values in before["endpoint"]["metrics"].items()
    }
    write(
        batch / "comparison.json",
        {
            "panels": panels,
            "fixed_step": 300000,
            "baseline_retrained": False,
            "new_scientific_updates": 300000,
            "utc": utc(),
            "mechanism_proven": False,
            "independent_worlds": 1,
            "initializations": 1,
            "note": "Unmatched additional parameters/state/compute; causal content-control "
            "separation not established by endpoint scores.",
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
    if not args.execute:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return
    if (
        config["batch"] != BATCH
        or config["runtime"]["docker_context"] != "lm-memory"
        or config["runtime"]["gpu"] != 4
        or config["spec"]["steps"] != 300000
    ):
        raise ValueError("Only the authorized one-arm GPU4/300k plan")
    root = Path(config["runtime"]["persistent_root"])
    if not root.resolve().is_relative_to("/ossfs/workspace") or not root.is_dir():
        raise ValueError("Persistent LM mount required")
    if not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise ValueError("Persistent mount unavailable")
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Existing private tracking credential required")
    batch = root / BATCH
    if batch.is_symlink():
        raise ValueError("Batch output cannot be redirected")
    batch.mkdir(exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("Existing controller record: no duplicate submission")
        if shutil.disk_usage(batch).free < 20 * 1024**3:
            raise RuntimeError("Need 20GiB persistent free space")
        freeze(config, batch)
        (batch / "run").mkdir()
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "source_manifest_sha256": sha(batch / "freeze.json"),
                "automatic_retry": False,
            },
        )
        state = {"state": "running", "stage": "resource_admission", "utc": utc()}
        write(batch / "controller-state.json", state)
        try:
            for stage in ("contracts", "preflight", "train", "audit", "cloud-audit"):
                deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
                row = None
                while row is None:
                    row = resources(config)
                    if row is None:
                        state.update(stage=stage, state="waiting_for_idle_resources", utc=utc())
                        write(batch / "controller-state.json", state)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Resource waiting timeout; no preemption")
                        time.sleep(30)
                write(batch / f"resource-admission-{stage}.json", row)
                state.update(stage=stage, state="running", utc=utc())
                write(batch / "controller-state.json", state)
                execute(
                    config,
                    batch,
                    stage,
                    token,
                    None if stage in ("contracts", "cloud-audit") else row,
                )
            compare(config, batch)
            state.update(
                state="complete",
                utc=utc(),
                new_scientific_updates=300000,
                independent_audit_passed=True,
                cloud_verified=True,
            )
            write(batch / "completion-manifest.json", state)
            write(batch / "controller-state.json", state)
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
