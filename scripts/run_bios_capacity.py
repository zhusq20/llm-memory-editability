"""Run frozen width scan, then heldout QA, with an auditable persistent queue."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

from llm_memory_editability.bios_data import write_json

ROOT = Path(__file__).resolve().parents[1]


def jobs(config, stage):
    widths = config["widths"] if stage == "capacity" else config["holdout_widths"]
    rates = config["learning_rates"] if stage == "capacity" else [config["holdout_learning_rate"]]
    for rate in rates:
        for width in widths + (
            [config["control_width"]]
            if stage == "capacity" and rate == config["control_learning_rate"]
            else []
        ):
            for world in config["worlds"]:
                for seed in config["initialization_seeds"]:
                    for condition in config["conditions"]:
                        yield {
                            "stage": stage,
                            "width": width,
                            "lr": rate,
                            "world": world,
                            "seed": seed,
                            "condition": condition,
                            "name": (
                                f"{stage}-w{width}-lr{rate:g}-world{world}-seed{seed}-{condition}"
                            ),
                        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="results/bios-capacity-dev-v1")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = ROOT / "configs/bios-capacity-development-v1.json"
    config = json.loads(config_path.read_text())
    frozen = out / "study-config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError("Study configuration changed")
    write_json(frozen, config)
    sources = list((ROOT / "src/llm_memory_editability").glob("*.py")) + [
        Path(__file__),
        config_path,
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    provenance = out / "launch-sources.json"
    if provenance.exists() and json.loads(provenance.read_text()) != hashes:
        raise ValueError("Source files changed since launch")
    write_json(provenance, hashes)
    gpu_pool = queue.Queue()
    for gpu in args.gpus:
        gpu_pool.put(gpu)

    def execute(job):
        destination = out / job["name"]
        destination.mkdir(exist_ok=True)
        complete = destination / "complete.json"
        if complete.exists():
            done = json.loads(complete.read_text())
            if done["status"] != "complete" or done["step"] != config["steps"]:
                raise ValueError(f"Invalid completion: {destination}")
            return {**job, "status": "reused"}
        gpu = gpu_pool.get()
        try:
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
                "all" if job["stage"] == "capacity" else "half",
                "--threads",
                "2",
            ]
            if (destination / "resume.pt").exists():
                command.append("--resume")
            started = time.time()
            write_json(
                destination / "launch.json", {"command": command, "gpu": gpu, "started": started}
            )
            with (destination / "run.log").open("a") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record = {
                **job,
                "gpu": gpu,
                "returncode": result.returncode,
                "elapsed": time.time() - started,
                "status": "complete" if result.returncode == 0 and complete.exists() else "failed",
            }
            write_json(destination / "execution.json", record)
            return record
        finally:
            gpu_pool.put(gpu)

    for stage in config["stage_order"]:
        records = []
        pending = list(jobs(config, stage))
        write_json(
            out / "status.json",
            {"stage": stage, "status": "running", "expected": len(pending), "completed": 0},
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [executor.submit(execute, job) for job in pending]
            for future in concurrent.futures.as_completed(futures):
                record = future.result()
                records.append(record)
                write_json(out / f"{stage}-ledger.json", records)
                write_json(
                    out / "status.json",
                    {
                        "stage": stage,
                        "status": "running",
                        "expected": len(pending),
                        "completed": len(records),
                        "failed": sum(r["status"] == "failed" for r in records),
                    },
                )
                print(json.dumps(record), flush=True)
        if any(r["status"] == "failed" for r in records):
            raise RuntimeError(f"{stage} has failed runs; next stage not started")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/summarize_bios_capacity.py"), "--root", str(out)],
        cwd=ROOT,
        check=True,
    )
    audit = json.loads((out / "audit.json").read_text())
    if audit["completed"] != config["capacity_runs"] + config["holdout_runs"]:
        raise RuntimeError("Unexpected completed run count")
    write_json(
        out / "status.json",
        {
            "status": "complete",
            "capacity_runs": config["capacity_runs"],
            "holdout_runs": config["holdout_runs"],
        },
    )


if __name__ == "__main__":
    main()
