"""Run the frozen crossover matrix on explicitly selected healthy GPUs."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from llm_memory_editability.bios_cross import make_cross_world, save_world
from llm_memory_editability.bios_cross_train import source_hashes
from llm_memory_editability.bios_data import write_json

ROOT = Path(__file__).resolve().parents[1]


def worker_alive(pid, destination):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return (
            b"llm_memory_editability.bios_cross_train" in command
            and str(destination.resolve()).encode() in command
        )
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--output", default="results/bios-cross-dev-v1")
    parser.add_argument("--config", default="configs/bios-cross-development-v1.json")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or args.slots_per_gpu < 1:
        raise ValueError("Unique GPUs and positive concurrency required")
    os.chdir(ROOT)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = (ROOT / args.config).resolve()
    study = json.loads(config_path.read_text())
    contract = {"study": study, "sources": source_hashes()}
    frozen = out / "launch-contract.json"
    if frozen.exists() and json.loads(frozen.read_text()) != contract:
        raise ValueError("Frozen contract changed")
    write_json(frozen, contract)
    data_root = ROOT / "data/bios-cross-v1"
    data_root.mkdir(parents=True, exist_ok=True)
    with (data_root / ".prepare.lock").open("w") as data_lock:
        fcntl.flock(data_lock, fcntl.LOCK_EX)
        for world in study["worlds"]:
            save_world(make_cross_world(world), data_root / f"world-{world}")
    jobs = [(w, s, c) for s in study["seeds"] for w in study["worlds"] for c in study["conditions"]]
    pending, active, complete, failed = [], [], [], []
    for w, s, c in jobs:
        name = f"world-{w}-seed-{s}-{c}"
        if (out / name / "complete.json").exists():
            complete.append(name)
        else:
            launch_path = out / name / "launch.json"
            launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
            if launch and worker_alive(launch["pid"], out / name):
                if launch["gpu"] not in args.gpus:
                    raise ValueError("Live worker outside the requested GPU pool")
                active.append(
                    {
                        "name": name,
                        "gpu": launch["gpu"],
                        "pid": launch["pid"],
                        "process": None,
                        "log": None,
                        "started": launch["started"],
                    }
                )
            else:
                pending.append((w, s, c, name))
    while pending or active:
        for item in active[:]:
            if item["process"] is None:
                if worker_alive(item["pid"], out / item["name"]):
                    continue
                code = None
            else:
                code = item["process"].poll()
                if code is None:
                    continue
            if item["log"] is not None:
                item["log"].close()
            name = item["name"]
            good = code in (0, None) and (out / name / "complete.json").exists()
            (complete if good else failed).append(name)
            write_json(
                out / name / "execution.json",
                {"returncode": code, "gpu": item["gpu"], "elapsed": time.time() - item["started"]},
            )
            active.remove(item)
        if not failed:
            for gpu in args.gpus:
                while pending and sum(item["gpu"] == gpu for item in active) < args.slots_per_gpu:
                    if source_hashes() != contract["sources"]:
                        raise ValueError("Sources changed during matrix execution")
                    w, s, c, name = pending.pop(0)
                    dest = out / name
                    dest.mkdir(exist_ok=True)
                    command = [
                        sys.executable,
                        "-m",
                        "llm_memory_editability.bios_cross_train",
                        "--world",
                        str(w),
                        "--seed",
                        str(s),
                        "--condition",
                        c,
                        "--config",
                        str(config_path),
                        "--output",
                        str(dest),
                    ]
                    env = {
                        **os.environ,
                        "PYTHONPATH": str(ROOT / "src"),
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "OMP_NUM_THREADS": "2",
                    }
                    log = (dest / "run.log").open("a")
                    process = subprocess.Popen(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    started = time.time()
                    write_json(
                        dest / "launch.json",
                        {"pid": process.pid, "gpu": gpu, "command": command, "started": started},
                    )
                    active.append(
                        {
                            "name": name,
                            "gpu": gpu,
                            "process": process,
                            "pid": process.pid,
                            "log": log,
                            "started": started,
                        }
                    )
        write_json(
            out / "status.json",
            {
                "state": "failed" if failed else "running",
                "complete": complete,
                "failed": failed,
                "active": [{"name": i["name"], "gpu": i["gpu"], "pid": i["pid"]} for i in active],
                "pending": len(pending),
                "updated": time.time(),
            },
        )
        if failed and not active:
            raise RuntimeError("Worker failed; pending jobs retained in status")
        if pending or active:
            time.sleep(5)
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "complete": complete,
            "learning_runs": len(complete),
            "edit_cases": len(complete) * 8,
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    main()
