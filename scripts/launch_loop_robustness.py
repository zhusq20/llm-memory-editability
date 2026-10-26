#!/usr/bin/env python3
"""Queue the authorized robustness matrix, respecting external GPU owners."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
if os.environ.get("LOOP_PARENT_SOURCE"):
    sys.path.append(str(Path(os.environ["LOOP_PARENT_SOURCE"]) / "scripts"))
from launch_dual_gate import docker, sha, snapshot, utc  # noqa: E402
from launch_paper_update_gate import credential_archive  # noqa: E402

DOCKER = ["docker", "--context", "lm-memory"]
PREFIX = "lir-v1-"
FAILED_HEALTH_ID = "824bf0b9a897ee2eb770329589ef690d3286b21fbd090af2a7b65ecbcfc44ba3"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def failed_health_check(container):
    s = container["State"]
    return (
        container["Id"] == FAILED_HEALTH_ID
        and container["Name"] == "/table4-gpu4-health-20261005"
        and s["Status"] == "created"
        and s["Pid"] == 0
        and not s["Running"]
        and s["StartedAt"] == "0001-01-01T00:00:00Z"
        and "gpu requires reset" in s["Error"]
        and container["HostConfig"]["RestartPolicy"]["Name"] == "no"
    )


def resource_snapshot():
    rows, blocked = snapshot(own_prefix=PREFIX)
    by_uuid = {r["uuid"]: gpu for gpu, r in rows.items()}
    ignored = []
    for context in ("default", "lm-memory"):
        ids = docker(
            "ps",
            "-aq",
            "--filter",
            "status=created",
            "--filter",
            "status=paused",
            "--filter",
            "status=restarting",
            context=context,
        ).stdout.split()
        for c in json.loads(docker("inspect", *ids, context=context).stdout) if ids else []:
            if context == "default" and failed_health_check(c):
                ignored.append({"id": c["Id"], "name": c["Name"], "state": c["State"]})
                continue
            for request in c["HostConfig"].get("DeviceRequests") or []:
                for device in request.get("DeviceIDs") or []:
                    gpu = int(device) if device.isdigit() else by_uuid.get(device)
                    if gpu in rows:
                        blocked.setdefault(gpu, []).append(c["Name"])
    return rows, blocked, ignored


def eligible_gpu(row, blockers, own_jobs):
    return bool(
        row is not None
        and not blockers
        and own_jobs < 2
        and row["free"] >= 20000
        and (own_jobs > 0 or (row["used"] <= 512 and row["utilization"] <= 5))
    )


class Controller:
    def __init__(self, base, token, plan_path=None):
        global PREFIX
        hold = Path(base) / "training-hold.json"
        if hold.exists():
            raise RuntimeError(
                f"Loop queue is held by the user: {hold}. "
                "Collect completed results only; resuming training requires new authorization."
            )
        self.base, self.token = base, token
        self.plan_path = plan_path or ROOT / "configs/loop-invariant-robustness-v1.json"
        self.plan = json.loads(self.plan_path.read_text())
        PREFIX = self.plan.get("container_prefix", PREFIX)
        self.mutex = threading.RLock()
        self.claims = {}
        self.state = {
            "batch": self.plan["batch"],
            "state": "running",
            "jobs": {j: {"state": "pending"} for j in self.plan["execution_order"]},
        }

    def update(self, name=None, **values):
        with self.mutex:
            target = self.state if name is None else self.state["jobs"][name]
            target.update(values)
            self.state["updated_utc"] = utc()
            write(self.base / "controller-state.json", self.state)

    def acquire(self, job):
        while True:
            with self.mutex:
                rows, blocked, ignored = resource_snapshot()
                candidates = sorted(
                    self.plan["runtime"]["gpu_candidates"],
                    key=lambda gpu: (self.claims.get(gpu, 0), gpu),
                )
                for gpu in candidates:
                    row = rows.get(gpu)
                    own = self.claims.get(gpu, 0)
                    if not eligible_gpu(row, blocked.get(gpu), own):
                        continue
                    self.claims[gpu] = own + 1
                    self.update(
                        job,
                        state="admitted",
                        gpu=gpu,
                        gpu_uuid=row["uuid"],
                        own_batch_gpu_sharing=bool(own),
                    )
                    write(
                        self.base / "resource-admission" / f"{job}.json",
                        {
                            "utc": utc(),
                            "gpu": gpu,
                            "row": row,
                            "other_own_jobs": own,
                            "ignored_failed_health_check": ignored,
                        },
                    )
                    return gpu, row["uuid"]
                self.update(job, state="waiting_for_idle_gpu")
            time.sleep(30)

    def command(self, job, stage, uuid, lane):
        rt = self.plan["runtime"]
        label = f"{PREFIX}{job}-{stage}"
        boot = "import os,sys;os.umask(0o022);"
        if stage in ("train", "cloud"):
            boot += (
                "from pathlib import Path;"
                "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
            )
        boot += "os.execvp(sys.argv[1],sys.argv[1:])"
        command = [
            *DOCKER,
            "create",
            "--name",
            label,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--cpuset-cpus",
            rt["cpu_sets"][lane],
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
            f"device={uuid}",
        ]
        for key, value in {
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
            "WANDB_DOCKER": rt["image"],
            "LOOP_PARENT_SOURCE": self.plan.get("parent_source", str(ROOT)),
        }.items():
            command += ["--env", f"{key}={value}"]
        command += [
            "--mount",
            f"type=bind,source={ROOT},target=/project,readonly",
            "--mount",
            f"type=bind,source={rt['persistent_root']},target={rt['persistent_root']},readonly",
            "--mount",
            f"type=bind,source={self.base},target=/output",
            rt["image"],
            "python",
            "-c",
            boot,
            "python",
            "-u",
            "scripts/run_loop_robustness.py",
            stage,
            "--plan",
            str(self.plan_path.relative_to(ROOT)),
            "--job",
            job,
            "--base",
            "/output",
        ]
        return label, command

    def execute(self, job, stage, gpu, uuid, lane):
        folder = self.base / "execution" / job / stage
        folder.mkdir(parents=True, exist_ok=False)
        label, command = self.command(job, stage, uuid, lane)
        write(folder / "command.json", {"argv": command, "utc": utc()})
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        if stage in ("train", "cloud"):
            subprocess.run(
                [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
                input=credential_archive(self.token),
                check=True,
                capture_output=True,
                timeout=120,
            )
        subprocess.run([*DOCKER, "start", label], check=True, capture_output=True, timeout=120)
        self.update(job, state="running", stage=stage, container=label, gpu=gpu)
        with (folder / "stdout.log").open("w") as log:
            follower = subprocess.Popen(
                [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            waited = subprocess.run(
                [*DOCKER, "wait", label], capture_output=True, text=True, check=True
            )
            follower.wait(timeout=60)
        code = int(waited.stdout.strip())
        write(folder / "exit.json", {"code": code, "utc": utc()})
        if code:
            raise RuntimeError(f"{label} exited {code}; failure retained, no automatic retry")

    def worker(self, jobs, lane):
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            gpu = None
            try:
                existing = self.plan["jobs"][job].get("existing_depth")
                if existing:
                    self.update(job, state="reused_running_depth", container=existing["container"])
                    while True:
                        state = json.loads(docker("inspect", existing["container"]).stdout)[0][
                            "State"
                        ]
                        if not state["Running"]:
                            break
                        time.sleep(15)
                    complete = Path(existing["path"]) / "complete.json"
                    if state["ExitCode"] != 0 or not complete.is_file():
                        raise RuntimeError("Existing depth run failed; retained without retry")
                    write(
                        self.base / "execution" / job / "reused-depth.json",
                        {
                            "state": state,
                            "complete": str(complete),
                            "sha256": sha(complete),
                            "original_execution": existing,
                            "utc": utc(),
                        },
                    )
                    self.update(job, state="complete", cloud_independent_check=False)
                    continue
                if job != "old-invariant":
                    while self.state["jobs"]["old-invariant"]["state"] in (
                        "pending",
                        "waiting_for_idle_gpu",
                    ):
                        time.sleep(2)
                gpu, uuid = self.acquire(job)
                for stage in self.plan["jobs"][job]["stages"]:
                    if stage == "train":
                        replay = self.base / "depth/old-invariant/replay.json"
                        while not replay.exists():
                            if self.state["jobs"]["old-invariant"]["state"] == "failed":
                                raise RuntimeError("Old-checkpoint replay failed; training blocked")
                            self.update(job, state="waiting_for_environment_replay", stage=stage)
                            time.sleep(15)
                        if not json.loads(replay.read_text())["passed"]:
                            raise RuntimeError("Environment replay did not pass")
                    self.execute(job, stage, gpu, uuid, lane)
                self.update(
                    job,
                    state="complete",
                    cloud_independent_check=self.plan["jobs"][job]["kind"] == "new",
                )
            except Exception as error:
                self.update(job, state="failed", error=repr(error))
            finally:
                with self.mutex:
                    if gpu is not None:
                        self.claims[gpu] -= 1
                jobs.task_done()

    def run(self):
        lock = (self.base / "controller.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process_path = self.base / "controller-process.json"
        if process_path.exists():
            raise FileExistsError("Batch already submitted; no duplicate or implicit resume")
        manifest = json.loads((self.base / "freeze.json").read_text())
        if (
            json.loads(docker("image", "inspect", manifest["image"]).stdout)[0]["Id"]
            != manifest["image"]
        ):
            raise ValueError("Frozen image identity changed")

        def verify_entry(item):
            relative, expected = item
            if sha(self.base / relative) != expected:
                raise ValueError(f"Frozen file changed: {relative}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(verify_entry, manifest["files"].items()))
        credential_archive(self.token)
        write(
            process_path,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "utc": utc(),
                "source": str(ROOT),
                "freeze_sha256": sha(self.base / "freeze.json"),
            },
        )
        jobs = queue.Queue()
        for job in self.plan["execution_order"]:
            jobs.put(job)
        self.update()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self.worker, jobs, lane) for lane in range(4)]
            for future in futures:
                future.result()
        done = all(j["state"] == "complete" for j in self.state["jobs"].values())
        self.update(state="complete" if done else "finished_with_failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--wandb-token-file", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    Controller(
        args.base.resolve(), args.wandb_token_file, args.plan.resolve() if args.plan else None
    ).run()


if __name__ == "__main__":
    main()
