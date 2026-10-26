#!/usr/bin/env python3
"""Queue evaluation after existing train/grid success, without changing those jobs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from launch_paper_update_gate import credential_archive  # noqa: E402

from llm_memory_editability.experiment_tracking import read_json, write_json  # noqa: E402

DOCKER = ["docker", "--context", "lm-memory"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def prerequisite_state(parent, arm):
    """Wait for BOTH original stages, and reject a recorded failure."""
    for stage in ("train", "grid"):
        p = Path(parent) / "execution" / arm / stage / "exit.json"
        if not p.exists():
            return "waiting"
        if read_json(p)["code"] != 0:
            return "failed"
    return "ready"


def gpu_free(uuid):
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    )
    if uuid in active.splitlines():
        return False
    ids = subprocess.check_output([*DOCKER, "ps", "-q"], text=True).split()
    if ids:
        containers = json.loads(subprocess.check_output([*DOCKER, "inspect", *ids], text=True))
        for c in containers:
            for request in c["HostConfig"].get("DeviceRequests") or []:
                if uuid in (request.get("DeviceIDs") or []) or request.get("Count") == -1:
                    return False
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for row in rows.splitlines():
        identity, memory, utilization = [s.strip() for s in row.split(",")]
        if identity == uuid:
            return int(memory) < 1024 and int(utilization) < 10
    raise ValueError("Scheduled GPU UUID is unavailable")


def run_container(config, output, name, lane, mode="evaluate"):
    label = f"{config['batch']}-{name}"
    folder = output / "execution" / name
    folder.mkdir(parents=True, exist_ok=False)
    gpu_args = ["--gpus", f"device={lane['uuid']}"] if mode == "evaluate" else []
    boot = "import os,sys;os.umask(0o022);os.execvp(sys.argv[1],sys.argv[1:])"
    cmd = [
        *DOCKER,
        "create",
        "--name",
        label,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpuset-cpus",
        lane["cpus"],
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
        *gpu_args,
        "--workdir",
        "/project",
        "--env",
        "PYTHONPATH=/project/src:/project/scripts",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "OMP_NUM_THREADS=2",
        "--env",
        "MKL_NUM_THREADS=2",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "MPLCONFIGDIR=/tmp/matplotlib",
        "--env",
        "WANDB_CACHE_DIR=/tmp/wandb",
        "--env",
        "WANDB_SILENT=true",
        "--env",
        "WANDB_CONSOLE=off",
        "--env",
        "WANDB_DISABLE_CODE=true",
        "--mount",
        f"type=bind,source={ROOT},target=/project,readonly",
        "--mount",
        f"type=bind,source={config['persistent_root']},target={config['persistent_root']},readonly",
        "--mount",
        f"type=bind,source={output},target=/output",
        config["image"],
        "python",
        "-c",
        boot,
        "python",
        "-u",
        "scripts/evaluate_rule_gate_rounds.py",
        mode,
        "--config",
        "/output/config.json",
        "--output",
        "/output",
    ]
    if mode == "evaluate":
        cmd += ["--arm", name]
    write_json(folder / "command.json", {"argv": cmd, "utc": utc()})
    subprocess.run(cmd, capture_output=True, check=True, timeout=120)
    if mode == "evaluate":
        subprocess.run(
            [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
            input=credential_archive(config["credential_file"]),
            capture_output=True,
            check=True,
            timeout=120,
        )
    subprocess.run([*DOCKER, "start", label], capture_output=True, check=True, timeout=120)
    write_json(
        folder / "launch.json",
        {"container": label, "utc": utc(), "gpu": lane["gpu"] if gpu_args else None},
    )
    with (folder / "stdout.log").open("w") as log:
        follow = subprocess.Popen(
            [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            waited = subprocess.run(
                [*DOCKER, "wait", label], text=True, capture_output=True, check=True
            )
            code = int(waited.stdout.strip())
            try:
                follow.wait(timeout=10)
            except subprocess.TimeoutExpired:
                follow.terminate()
        finally:
            if follow.poll() is None:
                follow.terminate()
    write_json(folder / "exit.json", {"code": code, "utc": utc()})
    if code:
        raise RuntimeError(f"{name} exited {code}; retained logs and container, no automatic retry")


def main():
    os.umask(0o022)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    config = read_json(output / "config.json")
    lockfile = (output / "execution.lock").open("a")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "controller-state.json").exists():
        raise FileExistsError("Controller already submitted; inspect state, do not resubmit")
    state = {
        "batch": config["batch"],
        "state": "waiting_for_training_and_grid",
        "arms": {name: {"state": "queued"} for name in config["arms"]},
    }
    mutex = threading.Lock()

    def update(name, **values):
        with mutex:
            state["arms"].setdefault(name, {}).update(values)
            if any(arm.get("state") == "running" for arm in state["arms"].values()):
                state["state"] = "evaluating"
            state["updated_utc"] = utc()
            write_json(output / "controller-state.json", state)

    def worker(lane):
        try:
            for name in lane["arms"]:
                update(name, state="waiting_for_predecessor", gpu=lane["gpu"])
                while True:
                    predecessor = prerequisite_state(config["parent_execution"], lane["parent_arm"])
                    if predecessor == "failed":
                        raise RuntimeError("Original train/grid failed; evaluation will not run")
                    if predecessor == "ready" and gpu_free(lane["uuid"]):
                        break
                    time.sleep(config["poll_seconds"])
                update(name, state="running", gpu=lane["gpu"])
                run_container(config, output, name, lane)
                update(name, state="complete", gpu=lane["gpu"])
        except Exception as exc:
            update(name, state="failed", error=str(exc))

    write_json(
        output / "controller-process.json",
        {"pid": os.getpid(), "host": os.uname().nodename, "utc": utc(), "argv": sys.argv},
    )
    threads = [threading.Thread(target=worker, args=(lane,)) for lane in config["lanes"]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    try:
        if not all(
            state["arms"].get(name, {}).get("state") == "complete" for name in config["arms"]
        ):
            raise RuntimeError("At least one evaluation failed; aggregate report not generated")
        folder = output / "execution" / "report"
        folder.mkdir(parents=True, exist_ok=False)
        command = [
            "taskset",
            "-c",
            config["report_cpus"],
            sys.executable,
            str(ROOT / "scripts/evaluate_rule_gate_rounds.py"),
            "report",
            "--config",
            str(output / "config.json"),
            "--output",
            str(output),
        ]
        env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        env["LD_LIBRARY_PATH"] = "/lib64" + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        write_json(folder / "command.json", {"argv": command, "utc": utc(), "gpu": None})
        with (folder / "stdout.log").open("w") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(folder / "exit.json", {"code": completed.returncode, "utc": utc()})
        completed.check_returncode()
        state["state"] = "complete"
    except Exception as exc:
        state.update(state="needs_attention", error=str(exc))
    state["updated_utc"] = utc()
    write_json(output / "controller-state.json", state)
    write_json(
        output / "controller-exit.json",
        {"code": 0 if state["state"] == "complete" else 1, "utc": utc()},
    )
    return 0 if state["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
