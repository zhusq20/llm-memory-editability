#!/usr/bin/env python3
"""Run the transferred whitelist on three GPUs without changing frozen recipes."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import io
import json
import os
import queue
import socket
import subprocess
import tarfile
import threading
import time
from pathlib import Path

DOCKER = ["docker", "--context", "lm-memory"]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, check=True, **kwargs)


def credential_archive(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("A private credential file is required")
    value = path.read_bytes().strip()
    if not value:
        raise ValueError("Empty credential")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            item = tarfile.TarInfo(name)
            item.type, item.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(item)
        item = tarfile.TarInfo("run/secrets/wandb")
        item.size, item.mode = len(value), 0o400
        item.uid, item.gid = os.getuid(), os.getgid()
        archive.addfile(item, io.BytesIO(value))
    return buffer.getvalue()


def snapshot():
    rows = {}
    raw = command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    ).stdout
    for line in raw.splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(",")]
        rows[int(index)] = dict(
            uuid=uuid, used=int(used), free=int(free), utilization=int(utilization)
        )
    raw = command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=30,
    ).stdout
    processes = [line.split(",")[0].strip() for line in raw.splitlines() if line]
    containers = []
    for context in ("default", "lm-memory"):
        ids = command(
            [
                "docker",
                "--context",
                context,
                "ps",
                "-aq",
                "--filter",
                "status=running",
                "--filter",
                "status=paused",
                "--filter",
                "status=restarting",
                "--filter",
                "status=created",
            ],
            text=True,
            timeout=30,
        ).stdout.split()
        if ids:
            data = json.loads(
                command(
                    ["docker", "--context", context, "inspect", *ids], text=True, timeout=30
                ).stdout
            )
            for c in data:
                containers.append(
                    {
                        "id": c["Id"],
                        "name": c["Name"],
                        "context": context,
                        "state": c["State"],
                        "requests": c["HostConfig"].get("DeviceRequests"),
                        "restart": c["HostConfig"]["RestartPolicy"],
                    }
                )
    return {"utc": utc(), "rows": rows, "processes": processes, "containers": containers}


def blockers(snap, gpu, ignored_created):
    row = snap["rows"][gpu]
    reasons = []
    if row["used"] > 512 or row["free"] < 20000 or row["utilization"] > 5:
        reasons.append("GPU not idle")
    if row["uuid"] in snap["processes"]:
        reasons.append("active CUDA process")
    for c in snap["containers"]:
        state = c["state"]
        if (
            c["id"] in ignored_created
            and state["Status"] == "created"
            and state["Pid"] == 0
            and not state["Running"]
            and state["StartedAt"] == "0001-01-01T00:00:00Z"
            and c["restart"]["Name"] == "no"
        ):
            continue
        for request in c["requests"] or []:
            # Visibility of all GPUs is not exclusive ownership. Active CUDA PIDs
            # above cover processes in long-lived development containers.
            if any(d in (str(gpu), row["uuid"]) for d in request.get("DeviceIDs") or []):
                reasons.append(c["name"])
    return reasons


def validate_schedule(plan, manifest):
    tasks = manifest["training_jobs"] + manifest["evaluation_only_jobs"]
    expected = {t["job"] for t in tasks}
    ordered = [j for phase in plan["phases"] for j in phase]
    if len(ordered) != len(expected) or set(ordered) != expected:
        raise ValueError("Schedule must contain the exact transferred whitelist once")
    seeds = {t["job"] for t in manifest["training_jobs"] if t["condition"] in ("init-1", "init-2")}
    if set(plan["phases"][-1]) != seeds:
        raise ValueError("Both new initialization groups must be last")
    if plan["gpus"] != [5, 6, 8] or len(plan["cpu_sets"]) != 3:
        raise ValueError("Only the three authorized GPUs may be used")
    if any(t.get("steps", 1500000) != 1500000 for t in manifest["training_jobs"]):
        raise ValueError("Training budget changed")


class Controller:
    def __init__(self, base, token):
        hold = Path(base) / "training-hold.json"
        if hold.exists():
            raise RuntimeError(
                f"Loop queue is held by the user: {hold}. "
                "Collect completed results only; resuming training requires new authorization."
            )
        self.base, self.token = base, token
        self.plan = json.loads((base / "schedule.json").read_text())
        self.manifest = json.loads((base / "tasks.json").read_text())
        validate_schedule(self.plan, self.manifest)
        self.tasks = {
            t["job"]: t
            for t in self.manifest["training_jobs"] + self.manifest["evaluation_only_jobs"]
        }
        self.mutex = threading.RLock()
        self.state = {
            "state": "preflight",
            "host": socket.gethostname(),
            "jobs": {j: {"state": "pending"} for phase in self.plan["phases"] for j in phase},
        }

    def update(self, job=None, **values):
        with self.mutex:
            (self.state if job is None else self.state["jobs"][job]).update(values)
            self.state["updated_utc"] = utc()
            write(self.base / "controller-state.json", self.state)

    def admit(self, gpu, label):
        while True:
            snap = snapshot()
            reasons = blockers(snap, gpu, self.plan["ignored_created_ids"])
            if not reasons:
                if snap["rows"][gpu]["uuid"] != self.plan["gpu_uuids"][str(gpu)]:
                    raise RuntimeError("GPU identity changed")
                write(self.base / "resource-admission" / f"{label}.json", snap)
                return
            self.update(waiting_gpu=gpu, waiting_reasons=reasons)
            time.sleep(30)

    def execute(self, job, stage, lane, preflight=False):
        gpu = self.plan["gpus"][lane]
        task = self.tasks[job]
        output = self.base / "execution" / job / stage
        output.mkdir(parents=True, exist_ok=False)
        label = f"lqt-1021-{job}-{stage}"
        self.admit(gpu, f"{job}-{stage}")
        source = Path(task["source"])
        boot = "import os,sys;os.umask(0o022);"
        secret = stage in ("train", "cloud") or preflight
        if secret:
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
            self.plan["cpu_sets"][lane],
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
            "/project",
            "--gpus",
            f"device={self.plan['gpu_uuids'][str(gpu)]}",
        ]
        env = {
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
            "WANDB_DOCKER": self.plan["image"],
            "LOOP_PARENT_SOURCE": self.plan["parent_source"],
        }
        for key, value in env.items():
            argv += ["--env", f"{key}={value}"]
        for src, dst, mode in [
            (source, "/project", ",readonly"),
            (self.plan["persistent_root"], self.plan["persistent_root"], ",readonly"),
            (self.base, "/output", ""),
        ]:
            argv += ["--mount", f"type=bind,source={src},target={dst}{mode}"]
        argv += [self.plan["image"], "python", "-c", boot, "python", "-u"]
        if preflight:
            argv += [
                "/output/control/check_loop_transfer_environment.py",
                "--job",
                job,
                "--output",
                f"/output/preflight/gpu-{gpu}",
            ]
        else:
            argv += [
                "scripts/run_loop_robustness.py",
                stage,
                "--plan",
                str(Path(task["plan"]).relative_to(source)),
                "--job",
                job,
                "--base",
                "/output",
            ]
        write(output / "command.json", {"argv": argv, "utc": utc()})
        created = command(argv, text=True, timeout=120).stdout.strip()
        write(output / "container.json", {"id": created, "name": label})
        if secret:
            command(
                [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
                input=credential_archive(self.token),
                timeout=120,
            )
        command([*DOCKER, "start", label], timeout=120)
        if not preflight:
            self.update(job, state="running", stage=stage, gpu=gpu, container=label)
        with (output / "stdout.log").open("w") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            result = command([*DOCKER, "wait", label], text=True)
            follower.wait(timeout=60)
        code = int(result.stdout.strip())
        write(output / "exit.json", {"code": code, "utc": utc()})
        if code:
            raise RuntimeError(f"{label} exited {code}; retained, no automatic retry")

    def worker(self, lane, jobs, first=None):
        while True:
            if first:
                job, first = first, None
            else:
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    return
            try:
                task = self.tasks[job]
                if (
                    task["kind"] == "new"
                    and digest(Path(task["source"]) / task["config"]) != task["config_sha256"]
                ):
                    raise ValueError("Frozen configuration changed")
                for stage in task["stages"]:
                    self.execute(job, stage, lane)
                self.update(job, state="complete")
            except Exception as error:
                self.update(job, state="failed", error=str(error))

    def run(self):
        # Locks are held until completion, including the two stopped source queues.
        locks = []
        for path in [
            self.base / "controller.lock",
            *map(Path, self.plan["source_controller_locks"]),
        ]:
            lock = path.open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        if (self.base / "controller-process.json").exists():
            raise FileExistsError("Already submitted; inspect receipts, never implicitly resume")
        actual = json.loads(
            command([*DOCKER, "image", "inspect", self.plan["image"]], text=True).stdout
        )[0]["Id"]
        if actual != self.plan["image"]:
            raise ValueError("Runtime image changed")
        for path, expected in self.plan["frozen_files"].items():
            if digest(path) != expected:
                raise ValueError(f"Frozen file changed: {path}")
        credential_archive(self.token)
        write(
            self.base / "controller-process.json",
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "utc": utc(),
                "schedule_sha256": digest(self.base / "schedule.json"),
                "controller_sha256": digest(__file__),
            },
        )
        self.update()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(self.execute, job, "environment", lane, True)
                    for lane, job in enumerate(
                        ["old-content", "old-recipe-additive", "old-invariant-raw"]
                    )
                ]
                for future in futures:
                    future.result()
            self.update(state="running", environment_passed=True)
            for phase_index, phase in enumerate(self.plan["phases"]):
                self.update(phase=phase_index)
                jobs = queue.Queue()
                # First three assignments are explicit; subsequent work uses a shared queue.
                first = phase[:3]
                for job in phase[3:]:
                    jobs.put(job)
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    futures = [
                        pool.submit(
                            self.worker, lane, jobs, first[lane] if lane < len(first) else None
                        )
                        for lane in range(3)
                    ]
                    for future in futures:
                        future.result()
                if any(self.state["jobs"][j]["state"] != "complete" for j in phase):
                    raise RuntimeError("Phase has failed jobs; later phase remains unstarted")
            self.update(state="complete")
        except Exception as error:
            self.update(state="failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o022)
    Controller(args.base.resolve(), args.wandb_token_file).run()


if __name__ == "__main__":
    main()
