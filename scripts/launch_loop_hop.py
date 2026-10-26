#!/usr/bin/env python3
"""Freeze and queue eight isolated LM jobs without preempting any other project."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-hop-development-v1"
DOCKER = ["docker", "--context", "lm-memory"]
TOKEN = Path.home() / ".cache/llm-memory-ttcd/wandb-token"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def docker(*args, context="lm-memory", check=True):
    return subprocess.run(
        ["docker", "--context", context, *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


def validate_config(config):
    expected = {
        (architecture, layers, rounds)
        for architecture in ("loop", "discoloop")
        for layers, rounds in ((1, 2), (1, 3), (1, 4), (2, 2))
    }
    observed = {
        (arm["architecture"], arm["unique_layers"], arm["repeats"])
        for arm in config["arms"].values()
    }
    if config["batch"] != BATCH or len(config["arms"]) != 8 or observed != expected:
        raise ValueError("Only the authorized eight-condition matrix may run")
    if config["training"]["steps"] != 300000 or config["training"]["batch_size"] != 512:
        raise ValueError("Fixed common 300k update budget and batch512 required")
    nodes = config["evaluation_nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != 300000:
        raise ValueError("Evaluation nodes must include the fixed endpoint")
    rt = config["runtime"]
    if rt["docker_context"] != "lm-memory" or not rt["image"].startswith("sha256:"):
        raise ValueError("Use only the fixed LM Docker image")
    cpus = [*rt["utility_cpus"], *(v for slot in rt["cpu_slots"] for v in slot)]
    if len(cpus) != len(set(cpus)) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Own task CPU slots must be disjoint and available")


def prepare(config, batch):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from llm_memory_editability.loop_hop_data import prepare as prepare_data

    batch.mkdir(parents=True, exist_ok=False)
    # The host's private umask must not prevent the root-squashed Docker
    # daemon from traversing this NEW synthetic-data batch's bind roots.
    batch.chmod(0o755)
    source = batch / "source"
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in (
        f"configs/{BATCH}.json",
        "configs/experiment-tracking-defaults.json",
        "scripts/run_supervised_alignment.py",
        "scripts/run_loop_hop.py",
        "scripts/launch_loop_hop.py",
        "scripts/report_loop_hop.py",
        "tests/test_loop_hop.py",
        "tests/test_supervised_alignment.py",
        f"docs/development-artifacts/{BATCH}/protocol.md",
        "pyproject.toml",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    metadata = prepare_data(config, batch / "inputs/data")
    for folder in (source, batch / "inputs"):
        folder.chmod(0o755)
        for item in folder.rglob("*"):
            item.chmod(0o755 if item.is_dir() else 0o644)
    (batch / "output").mkdir()
    (batch / "output").chmod(0o755)
    files = [p for folder in (source, batch / "inputs") for p in folder.rglob("*") if p.is_file()]
    write(
        batch / "freeze.json",
        {
            "utc": utc(),
            "image": config["runtime"]["image"],
            "files": {str(p.relative_to(batch)): sha(p) for p in sorted(files)},
            "scientific_updates": 8 * 300000,
            "engineering_smoke_updates": 8 * config["smoke_steps"],
            "data": metadata,
        },
    )
    write(batch / "controller-state.json", {"state": "prepared", "utc": utc(), "completed": []})


def verify(batch):
    freeze = read(batch / "freeze.json")
    for relative, expected in freeze["files"].items():
        if sha(batch / relative) != expected:
            raise ValueError(f"Frozen content changed: {relative}")
    if json.loads(docker("image", "inspect", freeze["image"]).stdout)[0]["Id"] != freeze["image"]:
        raise ValueError("Image digest mismatch")


def inventory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = {}
    for line in result.stdout.splitlines():
        i, uuid, used, free, utilization = [v.strip() for v in line.split(",")]
        rows[int(i)] = {
            "gpu": int(i),
            "uuid": uuid,
            "used_mib": int(used),
            "free_mib": int(free),
            "utilization": int(utilization),
        }
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    busy = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    allocated = {}
    lm_cpu_sets = []
    for context in ("default", "lm-memory"):
        ids = docker("ps", "--quiet", context=context).stdout.split()
        if not ids:
            continue
        containers = json.loads(docker("inspect", *ids, context=context).stdout)
        for container in containers:
            name = container["Name"].lstrip("/")
            for request in container["HostConfig"].get("DeviceRequests") or []:
                # An old general-purpose container with all GPUs visible is not
                # an exclusive reservation. Actual processes still block it.
                for value in request.get("DeviceIDs") or []:
                    allocated.setdefault(str(value), []).append(
                        {"context": context, "container": name}
                    )
            cpus = container["HostConfig"].get("CpusetCpus", "")
            if context == "lm-memory" and cpus and not name.startswith(BATCH):
                occupied = set()
                for part in cpus.split(","):
                    if "-" in part:
                        begin, end = map(int, part.split("-"))
                        occupied.update(range(begin, end + 1))
                    else:
                        occupied.add(int(part))
                lm_cpu_sets.append(occupied)
    for row in rows.values():
        row["process_present"] = row["uuid"] in busy
        row["explicit_allocations"] = allocated.get(str(row["gpu"]), []) + allocated.get(
            row["uuid"], []
        )
    return rows, lm_cpu_sets


def admissible(row, runtime):
    return (
        not row["process_present"]
        and not row["explicit_allocations"]
        and row["used_mib"] <= runtime["idle_gpu_max_used_mib"]
        and row["utilization"] <= runtime["idle_gpu_max_utilization"]
    )


def secret_archive():
    if not TOKEN.is_file() or TOKEN.stat().st_mode & 0o077:
        raise ValueError("Private existing W&B credential required")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            item = tarfile.TarInfo(name)
            item.type, item.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(item)
        content = TOKEN.read_bytes()
        item = tarfile.TarInfo("run/secrets/wandb")
        item.size, item.mode, item.uid, item.gid = len(content), 0o400, 1015, 1016
        archive.addfile(item, io.BytesIO(content))
    return buffer.getvalue()


def container_command(config, batch, stage, arm=None, row=None, cpus=None, smoke=False):
    rt = config["runtime"]
    suffix = f"-{arm}" if arm else ""
    if smoke:
        suffix += "-smoke"
    revision = rt.get("launch_revision", 1)
    name = f"{BATCH}-v{revision}-{stage}{suffix}"
    command = [
        *DOCKER,
        "create",
        "--name",
        name,
        "--user",
        f"{rt['uid']}:{rt['gid']}",
        "--cpus",
        str(len(cpus)),
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        f"{rt['memory_gib']}g",
        "--memory-swap",
        f"{rt['memory_gib']}g",
        "--shm-size",
        "4g",
        "--tmpfs",
        "/tmp:rw,size=4g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
    ]
    if row:
        command += ["--gpus", f"device={row['uuid']}"]
    for host, target, readonly in (
        (batch / "source", "/frozen", True),
        (batch / "inputs", "/inputs", True),
        (batch / "output", "/output", False),
    ):
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    if stage == "tracking":
        # Preserve the batch name in W&B identity/group while sharing the same
        # artifacts. Host PID visibility is used only for own trainer metrics.
        command += [
            "--pid",
            "host",
            "--mount",
            f"type=bind,source={batch / 'output'},target=/tracking/{BATCH}",
        ]
    for env in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "XDG_CACHE_HOME=/tmp/lm-hop-cache",
        "WANDB_CONFIG_DIR=/tmp/lm-hop-wandb-config",
        "WANDB_CACHE_DIR=/tmp/lm-hop-wandb-cache",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
    ):
        command += ["--env", env]
    command += [rt["image"]]
    if stage in ("tracking", "report"):
        command += [
            "python",
            "-c",
            "import os,sys;from pathlib import Path;"
            "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
            "os.execvp(sys.argv[1],sys.argv[1:])",
            "python",
        ]
    else:
        command += ["python"]
    if stage == "contracts":
        command += [
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "/frozen/tests/test_loop_hop.py",
            "/frozen/tests/test_supervised_alignment.py",
        ]
    elif stage == "tracking":
        command += [
            "-m",
            "llm_memory_editability.curve_tracking",
            "--root",
            f"/tracking/{BATCH}",
            "--runs-dir",
            f"/tracking/{BATCH}/runs",
            "--mode",
            "online",
        ]
    elif stage == "report":
        command += [
            "/frozen/scripts/report_loop_hop.py",
            "--config",
            f"/frozen/configs/{BATCH}.json",
            "--base",
            "/output",
            "--cloud",
        ]
    else:
        command += [
            "/frozen/scripts/run_loop_hop.py",
            stage,
            "--config",
            f"/frozen/configs/{BATCH}.json",
            "--base",
            "/output",
            "--arm",
            arm,
        ]
        if smoke:
            command += ["--smoke"]
    return name, command


def start(config, batch, stage, arm=None, row=None, cpus=None, smoke=False):
    verify(batch)
    cpus = cpus or config["runtime"]["utility_cpus"]
    name, command = container_command(config, batch, stage, arm, row, cpus, smoke)
    folder = batch / "execution" / name
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "command.json", {"argv": command, "utc": utc()})
    created = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    if stage in ("tracking", "report"):
        subprocess.run(
            [*DOCKER, "cp", "--archive", "-", f"{name}:/"],
            input=secret_archive(),
            check=True,
            capture_output=True,
            timeout=60,
        )
    docker("start", name)
    inspected = json.loads(docker("inspect", name).stdout)[0]
    record = {
        "container": name,
        "id": created.stdout.strip(),
        "utc": utc(),
        "resource": row,
        "cpus": cpus,
        "host_pid": inspected["State"]["Pid"],
    }
    write(folder / "launch.json", record)
    if arm and row:
        write(
            batch / "output/runtime" / f"{arm}-{stage}{'-smoke' if smoke else ''}.json",
            {
                "host_pid": record["host_pid"],
                "physical_gpu": row["gpu"],
                "uuid": row["uuid"],
                "cpus": cpus,
                "context": "lm-memory",
            },
        )
    return name


def wait_container(config, batch, name):
    folder = batch / "execution" / name
    started = time.monotonic()
    with (folder / "stdout.log").open("x") as output:
        follower = subprocess.Popen(
            [*DOCKER, "logs", "--follow", name], stdout=output, stderr=subprocess.STDOUT
        )
        try:
            while True:
                state = json.loads(docker("inspect", name).stdout)[0]["State"]
                write(folder / "heartbeat.json", {"state": state, "utc": utc()})
                if not state["Running"]:
                    write(folder / "exit.json", {"code": state["ExitCode"], "utc": utc()})
                    if state["ExitCode"]:
                        raise RuntimeError(f"Container failed: {name}; no automatic retry")
                    return
                if time.monotonic() - started > config["runtime"]["stage_timeout_hours"] * 3600:
                    write(folder / "timeout.json", {"own_container": name, "utc": utc()})
                    docker("stop", "--time", "30", name)
                    raise TimeoutError(name)
                time.sleep(15)
        finally:
            follower.terminate()
            follower.wait(timeout=30)


def execute(config, batch, stage, **options):
    name = start(config, batch, stage, **options)
    wait_container(config, batch, name)
    return name


def worker(config, batch, phase, arm, row, cpus):
    if phase == "engineering":
        execute(config, batch, "preflight", arm=arm, row=row, cpus=cpus)
        execute(config, batch, "smoke", arm=arm, row=row, cpus=cpus)
        execute(config, batch, "audit", arm=arm, row=row, cpus=cpus, smoke=True)
    else:
        execute(config, batch, "train", arm=arm, row=row, cpus=cpus)
        execute(config, batch, "audit", arm=arm, row=row, cpus=cpus)
        execute(config, batch, "trace", arm=arm, row=row, cpus=cpus)
    return arm


def schedule(config, batch, phase):
    rt = config["runtime"]
    pending, running, completed, failed = list(config["arms"]), {}, [], {}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=rt["maximum_parallel"]) as pool:
        while pending or running:
            for future, resource in list(running.items()):
                if not future.done():
                    continue
                del running[future]
                try:
                    completed.append(future.result())
                except Exception:
                    failed[resource["arm"]] = traceback.format_exc()
            rows, foreign_cpus = inventory()
            occupied_gpus = {resource["gpu"] for resource in running.values()}
            occupied_slots = {resource["slot"] for resource in running.values()}
            for gpu in rt["gpu_preference"]:
                if not pending or len(running) >= rt["maximum_parallel"]:
                    break
                if gpu in occupied_gpus or not admissible(rows[gpu], rt):
                    continue
                slots = [
                    slot
                    for slot, cpus in enumerate(rt["cpu_slots"])
                    if slot not in occupied_slots
                    and not any(set(cpus) & foreign for foreign in foreign_cpus)
                ]
                if not slots:
                    continue
                # Two snapshots prevent transient deallocation being mistaken
                # for an available device. Explicit reservations always block.
                time.sleep(2)
                refreshed, _ = inventory()
                if not admissible(refreshed[gpu], rt):
                    continue
                slot, arm = slots[0], pending.pop(0)
                future = pool.submit(
                    worker, config, batch, phase, arm, refreshed[gpu], rt["cpu_slots"][slot]
                )
                running[future] = {"arm": arm, "gpu": gpu, "slot": slot}
                occupied_slots.add(slot)
                occupied_gpus.add(gpu)
            state = {
                "state": "running" if running else "waiting_resources",
                "phase": phase,
                "utc": utc(),
                "pending": pending,
                "running": list(running.values()),
                "completed": completed,
                "failed": failed,
                "resource_snapshot": list(rows.values()),
                "science_started": phase == "science" and bool(running or completed or failed),
            }
            for resource in running.values():
                write(
                    batch / "output/resources" / f"{resource['arm']}.json",
                    {"utc": utc(), "phase": phase, **rows[resource["gpu"]]},
                )
            with (batch / "resource-history.jsonl").open("a") as history:
                history.write(
                    json.dumps(
                        {
                            "utc": utc(),
                            "phase": phase,
                            "own_jobs": list(running.values()),
                            "gpus": list(rows.values()),
                        }
                    )
                    + "\n"
                )
            write(batch / "controller-state.json", state)
            write(batch / "output/controller-state.json", state)
            if (
                not running
                and pending
                and time.monotonic() - started > rt["wait_timeout_hours"] * 3600
            ):
                raise TimeoutError("No available unreserved GPU; queued work was not submitted")
            if phase == "engineering" and failed and not running:
                raise RuntimeError("Engineering failed; no scientific training may start")
            if pending or running:
                time.sleep(20)
    return completed, failed


def controller(config, batch):
    verify(batch)
    lock = (batch / "execution.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(
        batch / "controller-process.json",
        {"pid": os.getpid(), "utc": utc(), "argv": __import__("sys").argv},
    )
    try:
        execute(config, batch, "contracts")
        schedule(config, batch, "engineering")
        # Pairing is checked BEFORE any scientific update across all eight arms.
        for layers, rounds in ((1, 2), (1, 3), (1, 4), (2, 2)):
            pair = [
                read(batch / "output/preflight" / f"{arch}-l{layers}-r{rounds}.json")
                for arch in ("loop", "discoloop")
            ]
            if pair[0]["initial_model_sha256"] != pair[1]["initial_model_sha256"]:
                raise ValueError("Paired initial parameters differ")
        tracker = start(config, batch, "tracking")
        completed, failed = schedule(config, batch, "science")
        state = {
            "state": "finished_with_failures" if failed else "complete",
            "phase": "science",
            "utc": utc(),
            "completed": completed,
            "failed": failed,
        }
        write(batch / "controller-state.json", state)
        write(batch / "output/controller-state.json", state)
        wait_container(config, batch, tracker)
        execute(config, batch, "report")
    except BaseException as error:
        state = {
            "state": "needs_attention",
            "utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        write(batch / "controller-failure.json", state)
        write(batch / "controller-state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "snapshot"))
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    args = parser.parse_args()
    config = read(args.config)
    registry = Path(config["runtime"]["persistent_root"]) / "loop-hop-current.json"
    if args.command in ("prepare", "execute") and registry.exists():
        current = read(registry)
        if current.get("batch") != BATCH:
            raise SystemExit(
                "This preparation was superseded by loop-hop-unified-v1; "
                "use scripts/launch_loop_hop_depth.py. Historical frozen sources are unchanged."
            )
    validate_config(config)
    batch = Path(config["runtime"]["persistent_root"]) / BATCH
    if args.command == "prepare":
        prepare(config, batch)
    elif args.command == "execute":
        controller(config, batch)
    else:
        print(json.dumps(inventory()[0], indent=2))


if __name__ == "__main__":
    main()
