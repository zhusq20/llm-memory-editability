"""Schedule the 36 frozen P2 continuations, or print commands for external scheduling."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from llm_memory_editability.bios_cross_continue import continuation_sources, file_hash
from llm_memory_editability.bios_data import write_json

ROOT = Path(__file__).resolve().parents[1]
MODULE = "llm_memory_editability.bios_cross_continue"


def jobs(parent_root, output_root, python=sys.executable):
    result = []
    for branch, lr, seeds in (("common", 0.0001, (0, 1)), ("sensitivity", 0.0003, (0,))):
        for width in (128, 256):
            for seed in seeds:
                for world in (0, 1):
                    for condition in ("company", "project", "neither"):
                        name = f"world-{world}-seed-{seed}-{condition}"
                        parent = (Path(parent_root) / f"width-{width}" / name).resolve()
                        output = (Path(output_root) / branch / f"width-{width}" / name).resolve()
                        result.append(
                            {
                                "id": f"{branch}/width-{width}/{name}",
                                "parent": str(parent),
                                "output": str(output),
                                "branch": branch,
                                "width": width,
                                "world": world,
                                "seed": seed,
                                "condition": condition,
                                "lr": lr,
                                "command": [
                                    python,
                                    "-m",
                                    MODULE,
                                    "--parent",
                                    str(parent),
                                    "--output",
                                    str(output),
                                    "--lr",
                                    str(lr),
                                ],
                            }
                        )
    return result


def worker_alive(pid, destination):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return MODULE.encode() in command and str(Path(destination).resolve()).encode() in command
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", default="results/bios-cross-scale-dev-v1")
    parser.add_argument("--output", default="results/bios-mechanism-dev-v1/p2")
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--slots-per-gpu", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = (ROOT / args.output).resolve()
    matrix = jobs(ROOT / args.parent_root, out)
    if args.dry_run:
        print(json.dumps({"learning_runs": 36, "edit_cases": 144, "jobs": matrix}, indent=2))
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or args.slots_per_gpu < 1:
        raise ValueError("Specify unique healthy GPUs and positive concurrency")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = {
        "protocol": "v2.8-development-p2",
        "sources": continuation_sources(),
        "runner_sha256": file_hash(__file__),
        "learning_runs": 36,
        "edit_cases": 144,
        "jobs": matrix,
    }
    contract_path = out / "launch-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("P2 matrix contract changed")
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
                raise ValueError("Live P2 worker outside requested GPU pool")
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
                    if continuation_sources() != contract["sources"]:
                        raise ValueError("P2 sources changed during execution")
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
            out / "status.json",
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
            raise RuntimeError("P2 worker failed; pending jobs were not launched")
        if pending or active:
            time.sleep(5)
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "complete": complete,
            "learning_runs": len(complete),
            "edit_cases": len(complete) * 4,
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    main()
