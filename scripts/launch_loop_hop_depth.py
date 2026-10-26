#!/usr/bin/env python3
"""Freeze eight authorized runs and admit them only to unused LM GPU resources."""

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

from resume_loop_query_control_transport import secret_archive

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-hop-unified-v1"
DOCKER = ["docker", "--context", "lm-memory"]
ENGINEERING = ("preflight", "smoke", "smoke-audit", "smoke-grid", "smoke-grid-audit")
SCIENCE = ("train", "audit", "grid", "grid-audit")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def docker(*args, context="lm-memory"):
    return subprocess.run(
        ["docker", "--context", context, *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def validate(config):
    expected = {
        f"{arch}-l{layers}-r{repeats}": {
            "architecture": "loop" if arch == "loop" else "discoloop",
            "unique_layers": layers,
            "repeats": repeats,
        }
        for layers, repeats in ((1, 2), (1, 3), (1, 4), (2, 2))
        for arch in ("loop", "disco")
    }
    if config["batch"] != BATCH or config["arms"] != expected:
        raise ValueError("Only the authorized eight-arm matrix may launch")
    if config["training"]["steps"] != 300000 or config["smoke_steps"] != 32:
        raise ValueError("The fixed development/scientific budgets changed")
    if config["runtime"]["docker_context"] != "lm-memory":
        raise ValueError("LM experiments must use the independent LM daemon")
    if config["runtime"]["tracking"] != "online":
        raise ValueError("New training requires online tracking")
    if config["runtime"]["gpu_candidates"] != [9]:
        raise ValueError("The user specified physical cuda9 only")
    if config["runtime"]["gpu_memory_fraction"] * 8 > 0.75:
        raise ValueError("Keep bounded same-batch GPU sharing")
    if config["evaluation_nodes"][-1] != config["training"]["steps"]:
        raise ValueError("The endpoint must be evaluated")


def blocked_gpus(rows, processes, containers, own_prefix=None):
    """Explicit per-device reservations include waiting external evaluation queues.

    An all-device utility container does not establish exclusive ownership;
    actual CUDA processes still block every device on which it is computing.
    """
    reasons = {index: [] for index in rows}
    by_uuid = {value["uuid"]: index for index, value in rows.items()}
    own_pids = {
        str(item["State"]["Pid"])
        for item in containers
        if own_prefix and item["Name"].lstrip("/").startswith(own_prefix)
    }
    for uuid, pid in processes:
        if pid in own_pids:
            continue
        if uuid in by_uuid:
            reasons[by_uuid[uuid]].append(f"active CUDA process {pid}")
    for container in containers:
        if own_prefix and container["Name"].lstrip("/").startswith(own_prefix):
            continue
        for request in container["HostConfig"].get("DeviceRequests") or []:
            for device in request.get("DeviceIDs") or []:
                index = by_uuid.get(device)
                if device.isdigit() and int(device) in rows:
                    index = int(device)
                if index is not None:
                    reasons[index].append(container["Name"])
    return reasons


def snapshot(own_prefix=None):
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
    rows = {}
    for line in raw.splitlines():
        index, uuid, used, free, utilization = [part.strip() for part in line.split(",")]
        rows[int(index)] = {
            "uuid": uuid,
            "used": int(used),
            "free": int(free),
            "utilization": int(utilization),
        }
    raw = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    processes = [
        tuple(part.strip() for part in line.split(",")) for line in raw.splitlines() if line
    ]
    containers = []
    for context in ("default", "lm-memory"):
        identifiers = docker("ps", "-q", context=context).stdout.split()
        if identifiers:
            containers.extend(json.loads(docker("inspect", *identifiers, context=context).stdout))
    return rows, blocked_gpus(rows, processes, containers, own_prefix)


def freeze(config, base):
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    source.mkdir()
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in (
        f"configs/{BATCH}.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
        f"docs/development-artifacts/{BATCH}/history-review.md",
        "pyproject.toml",
        "scripts/run_loop_hop.py",
        "scripts/run_loop_hop_depth.py",
        "scripts/run_supervised_alignment.py",
        "scripts/launch_loop_hop_depth.py",
        "scripts/resume_loop_query_control_transport.py",
        "tests/test_loop_hop_depth.py",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    origin = Path(config["input_origin"]["root"])
    for filename, key in (("world.npz", "world_sha256"), ("panels.npz", "panels_sha256")):
        if sha(origin / filename) != config["input_origin"][key]:
            raise ValueError("The selected common input changed")
    shutil.copytree(origin, base / "input")
    # Only this new batch's public science files acquire daemon traversal access.
    for path in [base, source, *source.rglob("*"), base / "input", *(base / "input").rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    write(
        base / "freeze.json",
        {
            "image": config["runtime"]["image"],
            "utc": utc(),
            "files": {
                str(p.relative_to(base)): sha(p)
                for folder in (source, base / "input")
                for p in sorted(folder.rglob("*"))
                if p.is_file()
            },
        },
    )


def verify(base):
    manifest = read(base / "freeze.json")
    for path, expected in manifest["files"].items():
        if sha(base / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    if (
        json.loads(docker("image", "inspect", manifest["image"]).stdout)[0]["Id"]
        != manifest["image"]
    ):
        raise ValueError("The fixed image identity changed")


class Controller:
    def __init__(self, config, base):
        self.config, self.base = config, base
        self.mutex = threading.RLock()
        self.leases = {}
        self.state = {
            "state": "preparing",
            "batch": BATCH,
            "arms": {name: {"state": "pending"} for name in config["arms"]},
        }

    def update(self, name=None, **values):
        with self.mutex:
            target = self.state if name is None else self.state["arms"][name]
            target.update(values)
            self.state["updated_utc"] = utc()
            write(self.base / "controller-state.json", self.state)

    def acquire(self, name):
        started = time.monotonic()
        rt = self.config["runtime"]
        while True:
            with self.mutex:
                rows, blocked = snapshot(f"{BATCH}-")
                for index in rt["gpu_candidates"]:
                    row = rows.get(index)
                    members = self.leases.get(index, set())
                    if row is None or blocked[index] or len(members) >= rt["maximum_parallel"]:
                        continue
                    if not members and (
                        row["used"] > rt["maximum_idle_gpu_used_mib"] or row["utilization"] > 5
                    ):
                        continue
                    if row["free"] < rt["minimum_free_gpu_mib"]:
                        continue
                    self.leases.setdefault(index, set()).add(name)
                    self.update(name, state="admitted", gpu=index, gpu_uuid=row["uuid"])
                    return index, row["uuid"]
                self.update(name, state="waiting_resources", blocked=blocked)
            if time.monotonic() - started > rt["waiting_timeout_hours"] * 3600:
                raise TimeoutError("Resource wait expired; no other task was changed")
            time.sleep(rt["poll_seconds"])

    def release(self, index, name):
        with self.mutex:
            self.leases[index].discard(name)
            if not self.leases[index]:
                self.leases.pop(index)

    def execute(self, name, stage, gpu=None):
        verify(self.base)
        folder = self.base / "execution" / name / stage
        if (folder / "exit.json").exists():
            if read(folder / "exit.json")["code"] == 0:
                return
            raise RuntimeError(f"Preserve failed {name}/{stage}; no automatic retry")
        folder.mkdir(parents=True, exist_ok=False)
        label = f"{BATCH}-{name}-{stage}"
        rt = self.config["runtime"]
        position = list(self.config["arms"]).index(name) if name in self.config["arms"] else 8
        cpus = [rt["cpu_first"] + 2 * position + i for i in range(2)]
        if not set(cpus) <= os.sched_getaffinity(0):
            raise ValueError("Requested CPU affinity is unavailable")
        args = [
            *DOCKER,
            "create",
            "--name",
            label,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpuset-cpus",
            ",".join(map(str, cpus)),
            "--cpus",
            "2",
            "--memory",
            "16g",
            "--memory-swap",
            "16g",
            "--shm-size",
            "2g",
            "--tmpfs",
            "/tmp:rw,size=4g",
            "--network",
            "host",
            "--workdir",
            "/frozen",
            "--env",
            "PYTHONPATH=/frozen/src:/frozen/scripts",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "TRANSFORMERS_OFFLINE=1",
            "--env",
            "RUFF_CACHE_DIR=/tmp/ruff-cache",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "WANDB_CONSOLE=off",
            "--env",
            "WANDB_SILENT=true",
            "--env",
            "WANDB_DISABLE_CODE=true",
            "--mount",
            f"type=bind,source={self.base / 'source'},target=/frozen,readonly",
            "--mount",
            f"type=bind,source={self.base},target=/output",
        ]
        args += ["--mount", f"type=bind,source={self.base / 'input'},target=/inputs,readonly"]
        if gpu is not None:
            args += ["--gpus", f"device={gpu[1]}"]
        if stage == "contracts":
            entry = [
                "python",
                "-c",
                (
                    "import json,subprocess;from pathlib import Path;"
                    "from llm_memory_editability.loop_hop_data import load,audit_data;"
                    f"c=json.loads(Path('configs/{BATCH}.json').read_text());"
                    "w,p,m=load('/inputs');audit_data(w,p,m);"
                    "assert all(m[k]==v for k,v in c['data'].items());"
                    "subprocess.run(['python','-m','pytest','-q','tests/test_loop_hop_depth.py',"
                    "'--junitxml=/output/contracts.xml'],check=True);"
                    "subprocess.run(['python','-m','ruff','check','scripts/run_loop_hop_depth.py',"
                    "'scripts/launch_loop_hop_depth.py','tests/test_loop_hop_depth.py'],check=True)"
                ),
            ]
        else:
            entry = [
                "python",
                "-u",
                "scripts/run_loop_hop_depth.py",
                stage,
                "--config",
                f"configs/{BATCH}.json",
                "--base",
                "/output",
                "--arm",
                name,
            ]
        tracked = stage in ("train", "cloud")
        bootstrap = "import os,sys;os.umask(0o022);"
        if tracked:
            bootstrap += (
                "from pathlib import Path;"
                "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
            )
        bootstrap += "os.execvp(sys.argv[1],sys.argv[1:])"
        args += [rt["image"], "python", "-c", bootstrap, *entry]
        write(
            folder / "command.json",
            {"argv": args, "utc": utc(), "freeze_sha256": sha(self.base / "freeze.json")},
        )
        created = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
        if tracked:
            subprocess.run(
                [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
                input=secret_archive(),
                check=True,
                capture_output=True,
                timeout=60,
            )
        docker("start", label)
        write(
            folder / "launch.json", {"container": label, "id": created.stdout.strip(), "utc": utc()}
        )
        if name in self.config["arms"]:
            write(
                self.base / "runtime" / f"{name}-{stage}.json",
                {
                    "gpu": gpu[0] if gpu else None,
                    "gpu_uuid": gpu[1] if gpu else None,
                    "cpus": cpus,
                    "container": label,
                    "gpu_memory_fraction": rt["gpu_memory_fraction"],
                    "shared_only_with_own_batch": True,
                    "utc": utc(),
                },
            )
            self.update(name, state="running", stage=stage, container=label)
        started = time.monotonic()
        with (folder / "stdout.log").open("x") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                while True:
                    state = json.loads(docker("inspect", label).stdout)[0]["State"]
                    write(folder / "heartbeat.json", {"state": state, "utc": utc()})
                    if not state["Running"]:
                        write(folder / "exit.json", {"code": state["ExitCode"], "utc": utc()})
                        if state["ExitCode"]:
                            raise RuntimeError(f"{name}/{stage} exited {state['ExitCode']}")
                        break
                    if time.monotonic() - started > rt["stage_timeout_hours"] * 3600:
                        write(folder / "timeout.json", {"own_container": label, "utc": utc()})
                        docker("stop", "--time", "30", label)
                        raise TimeoutError("Own stage hard timeout")
                    time.sleep(rt["poll_seconds"])
            finally:
                try:
                    follower.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    follower.terminate()
                    follower.wait(timeout=10)

    def worker(self, name, engineering):
        gpu = None
        try:
            gpu = self.acquire(name)
            for stage in ENGINEERING if engineering else SCIENCE:
                self.execute(name, stage, gpu)
            self.release(gpu[0], name)
            gpu = None
            if not engineering:
                self.execute(name, "cloud")
            self.update(name, state="engineering_passed" if engineering else "complete")
        except BaseException as error:
            self.update(
                name, state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            raise
        finally:
            if gpu is not None:
                self.release(gpu[0], name)

    def run(self):
        self.execute("all", "contracts")
        for engineering in (True, False):
            self.update(state="engineering" if engineering else "scientific_training")
            errors = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                jobs = {
                    pool.submit(self.worker, name, engineering): name
                    for name in self.config["arms"]
                }
                for future in concurrent.futures.as_completed(jobs):
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append({"arm": jobs[future], "error": repr(error)})
            if errors:
                self.update(state="needs_attention", errors=errors)
                return
            if engineering:
                write(
                    self.base / "engineering-complete.json",
                    {"passed": True, "utc": utc(), "arms": list(self.config["arms"])},
                )
        write(
            self.base / "completion-manifest.json",
            {
                "passed": True,
                "scientific_updates": 8 * self.config["training"]["steps"],
                "arms": {
                    name: read(self.base / "runs" / name / "cloud-audit.json")
                    for name in self.config["arms"]
                },
                "utc": utc(),
            },
        )
        self.update(state="complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read(ROOT / "configs" / f"{BATCH}.json")
    validate(config)
    if not args.execute:
        print(json.dumps({"batch": BATCH, "arms": config["arms"], "science_updates": 2400000}))
        return
    global_lock = (Path(config["runtime"]["persistent_root"]) / "loop-hop-execution.lock").open("a")
    fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for previous in config["supersedes"]:
        process_path = (
            Path(config["runtime"]["persistent_root"]) / previous / "controller-process.json"
        )
        if process_path.exists():
            pid = read(process_path)["pid"]
            command = Path(f"/proc/{pid}/cmdline")
            if command.exists() and b"launch_loop_hop" in command.read_bytes():
                raise RuntimeError(f"Previous same-scope controller still active: {previous}")
    write(
        Path(config["runtime"]["persistent_root"]) / "loop-hop-current.json",
        {
            "batch": BATCH,
            "gpu": 9,
            "pid": os.getpid(),
            "supersedes": config["supersedes"],
            "utc": utc(),
        },
    )
    base = Path(config["runtime"]["persistent_root"]) / BATCH
    if args.resume:
        verify(base)
    else:
        freeze(config, base)
    with (base / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(
            base / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "argv": sys.argv,
                "freeze_sha256": sha(base / "freeze.json"),
            },
        )
        controller = Controller(config, base)
        try:
            controller.run()
        except BaseException as error:
            controller.update(
                state="needs_attention", error=repr(error), traceback=traceback.format_exc()
            )
            raise


if __name__ == "__main__":
    main()
