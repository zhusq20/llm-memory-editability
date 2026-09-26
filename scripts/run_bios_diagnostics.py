"""Complete the fixed stage/endpoint measurement matrix without retraining."""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from llm_memory_editability.bios_data import write_json


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-dev-v1")
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    args = parser.parse_args()
    root = Path(args.root)
    steps = (2640, 5280, 13280) if args.device == "cuda" else (13280,)
    jobs = []
    for run in sorted(root.glob("world-*-seed-*-*")):
        world = json.loads((run / "config.json").read_text())["world_seed"]
        for step in steps:
            name = f"organization-{step}" if args.device == "cuda" else f"organization-cpu-{step}"
            jobs.append({"run": run.name, "world": world, "step": step, "output": str(run / name)})
    status = {
        "status": "running",
        "started_utc": now(),
        "device": args.device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(),
        "jobs": jobs,
    }
    status_path = root / f"diagnostics-{args.device}-status.json"
    write_json(status_path, status)
    for job in jobs:
        out = Path(job["output"])
        measurement = out / "measurements.json"
        if measurement.exists():
            existing = json.loads(measurement.read_text())
            if len(existing["probes"]) != 320 or existing["step"] != job["step"]:
                raise ValueError(f"Incomplete or mismatched existing measurement: {measurement}")
            job["status"] = "already_complete"
            write_json(status_path, status)
            continue
        command = [
            sys.executable,
            "-u",
            "-m",
            "llm_memory_editability.bios_measure",
            "--world",
            f"data/bios-work-v1/world-{job['world']}",
            "--checkpoint",
            str(root / job["run"] / f"model-{job['step']}.pt"),
            "--output",
            str(out),
            "--device",
            args.device,
        ]
        if args.device == "cuda":
            command.append("--behavior")
        job.update(status="running", started_utc=now(), command=command)
        write_json(status_path, status)
        print(json.dumps({"event": "start", **job}), flush=True)
        with (root / job["run"] / f"{out.name}.log").open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        job.update(
            status="complete" if result.returncode == 0 else "failed",
            ended_utc=now(),
            returncode=result.returncode,
        )
        write_json(status_path, status)
        print(json.dumps({"event": "finish", **job}), flush=True)
        if result.returncode:
            status.update(status="failed", ended_utc=now())
            write_json(status_path, status)
            raise SystemExit(result.returncode)
    status.update(status="complete", ended_utc=now())
    write_json(status_path, status)


if __name__ == "__main__":
    main()
