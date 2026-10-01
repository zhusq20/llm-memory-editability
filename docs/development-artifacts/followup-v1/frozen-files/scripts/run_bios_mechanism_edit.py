"""Schedule P1 counterfactual edits; optional frozen-embedding arm is opt-in."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_mechanism_edit import (
    ARMS,
    OPTIONAL_ARM,
    load_study,
    source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]


def matrix(study):
    jobs = []
    for width in study["widths"]:
        parent_root = (
            ROOT / "results/bios-cross-dev-v1"
            if width == 768
            else ROOT / f"results/bios-cross-scale-dev-v1/width-{width}"
        )
        for seed in study["seeds"]:
            for world in study["worlds"]:
                for condition in study["conditions"]:
                    name = f"world-{world}-seed-{seed}-{condition}"
                    jobs.append((f"width-{width}/{name}", parent_root / name))
    return jobs


def worker_alive(pid, destination):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return (
            b"llm_memory_editability.bios_mechanism_edit" in command
            and str(destination.resolve()).encode() in command
        )
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--output", default="results/bios-mechanism-edit-dev-v1")
    parser.add_argument("--config")
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, OPTIONAL_ARM))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    study = load_study(args.config, args.arms)
    jobs = matrix(study)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "models": len(jobs),
                    "new_edit_cases": len(jobs) * 2 * len(study["arms"]),
                    "jobs": [name for name, _ in jobs],
                },
                indent=2,
            )
        )
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or args.slots_per_gpu < 1:
        raise ValueError("Unique GPUs and positive concurrency are required")
    os.chdir(ROOT)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = {
        "study": study,
        "sources": source_hashes(),
        "jobs": [{"name": n, "baseline": str(p)} for n, p in jobs],
    }
    contract_path = out / "launch-contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Frozen P1 scheduler contract changed")
    write_json(contract_path, contract)
    study_path = out / "study.json"
    write_json(study_path, study)
    pending, active, complete, failed = [], [], [], []
    for name, baseline in jobs:
        dest = out / name
        if (dest / "complete.json").exists():
            complete.append(name)
            continue
        launch_path = dest / "launch.json"
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
        if launch and worker_alive(launch["pid"], dest):
            if launch["gpu"] not in args.gpus:
                raise ValueError("Live worker is outside this GPU pool")
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
            pending.append((name, baseline))
    while pending or active:
        for item in active[:]:
            process = item["process"]
            if process is None:
                if worker_alive(item["pid"], out / item["name"]):
                    continue
                code = None
            else:
                code = process.poll()
                if code is None:
                    continue
            if item["log"] is not None:
                item["log"].close()
            dest = out / item["name"]
            good = code in (None, 0) and (dest / "complete.json").exists()
            (complete if good else failed).append(item["name"])
            write_json(
                dest / "execution.json",
                {"returncode": code, "gpu": item["gpu"], "elapsed": time.time() - item["started"]},
            )
            active.remove(item)
        if not failed:
            for gpu in args.gpus:
                while pending and sum(i["gpu"] == gpu for i in active) < args.slots_per_gpu:
                    if source_hashes() != contract["sources"]:
                        raise ValueError("P1 sources changed during execution")
                    name, baseline = pending.pop(0)
                    dest = out / name
                    dest.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        "-m",
                        "llm_memory_editability.bios_mechanism_edit",
                        "--baseline",
                        str(baseline),
                        "--output",
                        str(dest),
                        "--config",
                        str(study_path),
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
                        {"pid": process.pid, "gpu": gpu, "started": started, "command": command},
                    )
                    active.append(
                        {
                            "name": name,
                            "gpu": gpu,
                            "pid": process.pid,
                            "process": process,
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
            raise RuntimeError("A P1 worker failed; remaining jobs have not been launched")
        if pending or active:
            time.sleep(5)
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "complete": complete,
            "new_edit_cases": len(complete) * 2 * len(study["arms"]),
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    main()
