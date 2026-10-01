"""Adopt live study workers and fill multiple independent slots on each GPU.

Scientific configuration and training sources stay frozen. This dispatcher only
changes placement/concurrency, preserves stage order, and never kills a worker.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_bios_capacity import ROOT, jobs

from llm_memory_editability.bios_data import write_json


def worker_alive(pid, destination):
    try:
        proc = Path(f"/proc/{pid}")
        args = proc.joinpath("cmdline").read_bytes().split(b"\0")
        return (
            str(destination.resolve()).encode() in args
            and b"llm_memory_editability.bios_organization_train" in args
        )
    except FileNotFoundError:
        return False


def choose_gpu(job, active, gpus, slots, large_limit=1):
    eligible = []
    for gpu in gpus:
        resident = [r for r in active.values() if r["gpu"] == gpu]
        if len(resident) >= slots:
            continue
        if job["width"] >= 768 and sum(r["job"]["width"] >= 768 for r in resident) >= large_limit:
            continue
        # Fill least occupied GPUs first with an explicit large-model concurrency limit.
        eligible.append((len(resident), sum(r["job"]["width"] ** 2 for r in resident), gpu))
    return min(eligible)[-1] if eligible else None


def verify_complete(destination, job, config):
    path = destination / "complete.json"
    if not path.exists():
        return False
    done = json.loads(path.read_text())
    actual = json.loads((destination / "config.json").read_text())
    if done["status"] != "complete" or done["step"] != config["steps"]:
        raise ValueError(f"Invalid completion {destination}")
    expected = {
        "condition": job["condition"],
        "seed": job["seed"],
        "world_seed": job["world"],
        "lr": job["lr"],
        "qa_split": "all" if job["stage"] == "capacity" else "half",
        "steps": config["steps"],
    }
    if any(actual[k] != v for k, v in expected.items()) or actual["model"]["width"] != job["width"]:
        raise ValueError(f"Completion identity mismatch {destination}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--large-models-per-gpu", type=int, default=1)
    parser.add_argument("--output", default="results/bios-capacity-dev-v1")
    args = parser.parse_args()
    if (
        args.slots_per_gpu < 1
        or args.large_models_per_gpu < 1
        or len(set(args.gpus)) != len(args.gpus)
    ):
        raise ValueError("Positive slots and unique GPU IDs required")
    out = ROOT / args.output
    lock = (out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((out / "study-config.json").read_text())
    if config != json.loads((ROOT / "configs/bios-capacity-development-v1.json").read_text()):
        raise ValueError("Scientific study configuration changed")
    frozen = json.loads((out / "launch-sources.json").read_text())
    for filename, digest in frozen.items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {filename}")
    record = {
        "started": time.time(),
        "pid": os.getpid(),
        "gpus": args.gpus,
        "slots_per_gpu": args.slots_per_gpu,
        "max_large_models_per_gpu": args.large_models_per_gpu,
        "dispatcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_sources_unchanged": True,
    }
    write_json(out / "parallel-dispatcher.json", record)
    write_json(
        out / "runner-pid.json", {"pid": os.getpid(), "state": "parallel dispatcher running"}
    )
    ledger = []
    ledger_path = out / f"parallel-ledger-{os.getpid()}.json"
    for stage in config["stage_order"]:
        all_jobs = {j["name"]: j for j in jobs(config, stage)}
        active, done, failures = {}, set(), []
        # Adopt workers from the previous dispatcher without replaying any updates.
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                command = proc.joinpath("cmdline").read_bytes().split(b"\0")
                if b"llm_memory_editability.bios_organization_train" not in command:
                    continue
                destination = Path(command[command.index(b"--output") + 1].decode())
                if (
                    destination.parent.resolve() != out.resolve()
                    or destination.name not in all_jobs
                ):
                    continue
                launch = json.loads((destination / "launch.json").read_text())
                if launch["gpu"] not in args.gpus:
                    raise ValueError("Live worker uses GPU outside configured pool")
                active[destination.name] = {
                    "job": all_jobs[destination.name],
                    "pid": int(proc.name),
                    "gpu": launch["gpu"],
                    "started": launch["started"],
                    "process": None,
                }
            except (FileNotFoundError, ProcessLookupError):
                continue
        for name, job in all_jobs.items():
            if name not in active and verify_complete(out / name, job, config):
                done.add(name)
        pending = [j for name, j in all_jobs.items() if name not in active and name not in done]
        while pending or active:
            for name, worker in list(active.items()):
                process = worker["process"]
                if process is not None:
                    alive = process.poll() is None
                else:
                    alive = worker_alive(worker["pid"], out / name)
                if alive:
                    continue
                success = verify_complete(out / name, worker["job"], config)
                result = {
                    **worker["job"],
                    "pid": worker["pid"],
                    "gpu": worker["gpu"],
                    "status": "complete" if success else "failed",
                    "elapsed": time.time() - worker["started"],
                    "adopted": process is None,
                }
                write_json(out / name / "execution.json", result)
                ledger.append(result)
                write_json(ledger_path, ledger)
                print(json.dumps(result), flush=True)
                done.add(name) if success else failures.append(name)
                del active[name]
            if not failures:
                for job in list(pending):
                    gpu = choose_gpu(
                        job, active, args.gpus, args.slots_per_gpu, args.large_models_per_gpu
                    )
                    if gpu is None:
                        continue
                    destination = out / job["name"]
                    destination.mkdir(exist_ok=True)
                    command = [
                        sys.executable,
                        "-m",
                        "llm_memory_editability.bios_organization_train",
                        "--world",
                        str(ROOT / f"data/bios-organization-v1/world-{job['world']}"),
                        "--output",
                        str(destination),
                        "--condition",
                        job["condition"],
                        "--seed",
                        str(job["seed"]),
                        "--width",
                        str(job["width"]),
                        "--heads",
                        str(job["width"] // config["head_dimension"]),
                        "--layers",
                        str(config["layers"]),
                        "--lr",
                        str(job["lr"]),
                        "--steps",
                        str(config["steps"]),
                        "--study",
                        "capacity-v1",
                        "--qa-split",
                        "all" if stage == "capacity" else "half",
                        "--threads",
                        "2",
                    ]
                    if (destination / "resume.pt").exists():
                        command.append("--resume")
                    started = time.time()
                    with (destination / "run.log").open("a") as log:
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    write_json(
                        destination / "launch.json",
                        {
                            "command": command,
                            "gpu": gpu,
                            "started": started,
                            "pid": process.pid,
                            "dispatcher": os.getpid(),
                        },
                    )
                    active[job["name"]] = {
                        "job": job,
                        "gpu": gpu,
                        "started": started,
                        "pid": process.pid,
                        "process": process,
                    }
                    pending.remove(job)
            write_json(
                out / "status.json",
                {
                    "stage": stage,
                    "status": "running" if not failures else "failed_draining",
                    "expected": len(all_jobs),
                    "completed": len(done),
                    "failed": len(failures),
                    "active": [
                        {"run": n, "gpu": w["gpu"], "pid": w["pid"]} for n, w in active.items()
                    ],
                    "pending": len(pending),
                    "slots_per_gpu": args.slots_per_gpu,
                    "updated": time.time(),
                },
            )
            if failures and not active:
                raise RuntimeError(f"Failures retained; next stage will not start: {failures}")
            if pending or active:
                time.sleep(5)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/summarize_bios_capacity.py"), "--root", str(out)],
            cwd=ROOT,
            check=True,
        )
    audit = json.loads((out / "audit.json").read_text())
    if (
        audit["completed"] != config["capacity_runs"] + config["holdout_runs"]
        or not audit["passed"]
    ):
        raise ValueError("Final count or audit failure")
    write_json(
        out / "status.json",
        {
            "status": "complete",
            "capacity_runs": config["capacity_runs"],
            "holdout_runs": config["holdout_runs"],
            "updated": time.time(),
        },
    )


if __name__ == "__main__":
    main()
