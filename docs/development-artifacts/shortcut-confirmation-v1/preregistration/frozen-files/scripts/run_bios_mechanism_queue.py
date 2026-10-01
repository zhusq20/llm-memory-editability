"""Schedule immutable per-model commands without changing research batches."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from llm_memory_editability.bios_data import write_json


def alive(pid, output):
    try:
        return str(Path(output).resolve()).encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or args.slots_per_gpu < 1:
        raise ValueError("Unique GPU IDs and positive slot count required")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    manifest = Path(args.manifest).resolve()
    raw = manifest.read_bytes()
    jobs = json.loads(raw)["jobs"]
    state_dir = manifest.parent
    lock = (state_dir / (manifest.stem + ".lock")).open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if len({j["key"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job key")
    active, pending, complete, failed = [], [], [], []
    for job in jobs:
        destination = Path(job["output"]).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        job["output"] = str(destination)
        if Path(job["completion"]).exists():
            complete.append(job["key"])
            continue
        launch_path = destination / "queue-launch.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
        if launch and alive(launch["pid"], job["output"]):
            if launch["command"] != job["command"] or launch["gpu"] not in args.gpus:
                raise ValueError("Existing worker has a different command or GPU pool")
            active.append({**job, **launch, "process": None, "log": None})
        else:
            pending.append(job)
    digest = hashlib.sha256(raw).hexdigest()
    status_path = state_dir / (manifest.stem + "-status.json")
    while pending or active:
        for item in active[:]:
            code = item["process"].poll() if item["process"] else None
            running = code is None if item["process"] else alive(item["pid"], item["output"])
            if running:
                continue
            if item["log"]:
                item["log"].close()
            good = code in (0, None) and Path(item["completion"]).exists()
            (complete if good else failed).append(item["key"])
            write_json(
                Path(item["output"]) / "queue-execution.json",
                {
                    "returncode": code,
                    "gpu": item["gpu"],
                    "success": good,
                    "elapsed_seconds": time.time() - item["started"],
                },
            )
            active.remove(item)
        if not failed:
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != digest:
                raise ValueError("Queue manifest changed during execution")
            for gpu in args.gpus:
                while pending and sum(i["gpu"] == gpu for i in active) < args.slots_per_gpu:
                    job = pending.pop(0)
                    log = (Path(job["output"]) / "queue-worker.log").open("a")
                    env = {
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "PYTHONPATH": str(root / "src"),
                        "OMP_NUM_THREADS": "2",
                    }
                    process = subprocess.Popen(
                        job["command"],
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    launch = {
                        "pid": process.pid,
                        "gpu": gpu,
                        "started": time.time(),
                        "command": job["command"],
                        "manifest_sha256": digest,
                    }
                    write_json(Path(job["output"]) / "queue-launch.json", launch)
                    active.append({**job, **launch, "process": process, "log": log})
        state = "failed" if failed else ("running" if active or pending else "complete")
        write_json(
            status_path,
            {
                "state": state,
                "manifest_sha256": digest,
                "complete": complete,
                "failed": failed,
                "pending": len(pending),
                "updated": time.time(),
                "active": [{"key": i["key"], "gpu": i["gpu"], "pid": i["pid"]} for i in active],
            },
        )
        if failed and not active:
            raise RuntimeError("Worker failed; pending jobs retained in queue")
        if active or pending:
            time.sleep(5)
    # Also write a terminal state when every job was already complete on entry.
    write_json(
        status_path,
        {
            "state": "complete",
            "manifest_sha256": digest,
            "complete": complete,
            "failed": [],
            "pending": 0,
            "active": [],
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    main()
