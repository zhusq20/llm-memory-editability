"""Prepare or explicitly schedule the conditional 12-run context control matrix."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from llm_memory_editability.bios_context_control import context_sources, validate_study
from llm_memory_editability.bios_cross import CONDITIONS
from llm_memory_editability.bios_data import write_json

ROOT = Path(__file__).resolve().parents[1]
MODULE = "llm_memory_editability.bios_context_control"


def jobs(output, config, python=sys.executable):
    return [
        {
            "id": f"width-256/world-{world}-seed-{seed}-{condition}",
            "world": world,
            "seed": seed,
            "condition": condition,
            "output": str(
                (Path(output) / "width-256" / f"world-{world}-seed-{seed}-{condition}").resolve()
            ),
            "command": [
                python,
                "-m",
                MODULE,
                "--world",
                str(world),
                "--seed",
                str(seed),
                "--condition",
                condition,
                "--config",
                str(Path(config).resolve()),
                "--output",
                str(
                    (
                        Path(output) / "width-256" / f"world-{world}-seed-{seed}-{condition}"
                    ).resolve()
                ),
            ],
        }
        for world in (0, 1)
        for seed in (0, 1)
        for condition in CONDITIONS
    ]


def worker_alive(pid, destination):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return MODULE.encode() in command and str(Path(destination).resolve()).encode() in command
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bios-context-control-v1.json")
    parser.add_argument("--output", default="results/bios-mechanism-dev-v1/p3-context")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument(
        "--execute", action="store_true", help="Explicitly schedule this conditional branch"
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    config, output = (ROOT / args.config).resolve(), (ROOT / args.output).resolve()
    study = json.loads(config.read_text())
    validate_study(study)
    matrix = jobs(output, config)
    if not args.execute:
        print(
            json.dumps(
                {"state": "prepared_only", "learning_runs": 12, "edit_cases": 0, "jobs": matrix},
                indent=2,
            )
        )
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or args.slots_per_gpu < 1:
        raise ValueError("Specify unique healthy GPUs and positive concurrency")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = {
        "protocol": "v2.8-p3-context-control",
        "sources": context_sources(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "learning_runs": 12,
        "edit_cases": 0,
        "jobs": matrix,
    }
    contract_path = output / "launch-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Context control launch matrix changed")
    write_json(contract_path, contract)
    pending, active, complete, failed = [], [], [], []
    for job in matrix:
        dest = Path(job["output"])
        if (dest / "complete.json").exists():
            complete.append(job["id"])
            continue
        launch_path = dest / "launch.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
        if launch and worker_alive(launch["pid"], dest):
            if launch["gpu"] not in args.gpus:
                raise ValueError("Live context worker outside requested GPU pool")
            active.append(
                {
                    "job": job,
                    "gpu": launch["gpu"],
                    "pid": launch["pid"],
                    "started": launch["started"],
                    "process": None,
                    "log": None,
                }
            )
        else:
            pending.append(job)
    while pending or active:
        for item in active[:]:
            process = item["process"]
            if process is None:
                if worker_alive(item["pid"], item["job"]["output"]):
                    continue
                code = None
            else:
                code = process.poll()
                if code is None:
                    continue
            if item["log"]:
                item["log"].close()
            dest = Path(item["job"]["output"])
            good = code in (0, None) and (dest / "complete.json").exists()
            (complete if good else failed).append(item["job"]["id"])
            write_json(
                dest / "execution.json",
                {"returncode": code, "gpu": item["gpu"], "elapsed": time.time() - item["started"]},
            )
            active.remove(item)
        if not failed:
            for gpu in args.gpus:
                while pending and sum(item["gpu"] == gpu for item in active) < args.slots_per_gpu:
                    if context_sources() != contract["sources"]:
                        raise ValueError("Context sources changed during execution")
                    if hashlib.sha256(config.read_bytes()).hexdigest() != contract["config_sha256"]:
                        raise ValueError("Context config changed during execution")
                    job = pending.pop(0)
                    dest = Path(job["output"])
                    dest.mkdir(parents=True, exist_ok=True)
                    env = {
                        **os.environ,
                        "PYTHONPATH": str(ROOT / "src"),
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "OMP_NUM_THREADS": "2",
                    }
                    log = (dest / "run.log").open("a")
                    process = subprocess.Popen(
                        job["command"], env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    started = time.time()
                    write_json(
                        dest / "launch.json",
                        {
                            "pid": process.pid,
                            "gpu": gpu,
                            "command": job["command"],
                            "started": started,
                        },
                    )
                    active.append(
                        {
                            "job": job,
                            "gpu": gpu,
                            "pid": process.pid,
                            "started": started,
                            "process": process,
                            "log": log,
                        }
                    )
        write_json(
            output / "status.json",
            {
                "state": "failed" if failed else "running",
                "complete": complete,
                "failed": failed,
                "active": [
                    {"id": item["job"]["id"], "pid": item["pid"], "gpu": item["gpu"]}
                    for item in active
                ],
                "pending": len(pending),
                "updated": time.time(),
            },
        )
        if failed and not active:
            raise RuntimeError("A context worker failed; pending jobs were not launched")
        if pending or active:
            time.sleep(5)
    write_json(
        output / "status.json",
        {
            "state": "complete",
            "complete": complete,
            "learning_runs": len(complete),
            "edit_cases": 0,
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    main()
