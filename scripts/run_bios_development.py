"""Run the frozen symbolic development batch, one process per selected GPU."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from llm_memory_editability.bios_data import write_json


def worker(gpu, jobs, args):
    records = []
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"}
    for world, seed, order in jobs:
        name = f"world-{world}-seed-{seed}-{order}"
        run = Path(args.output) / name
        run.mkdir(parents=True, exist_ok=True)
        common = ["--world", f"data/bios-work-v1/world-{world}"]
        if args.phase == "learning":
            if (run / "complete.json").exists():
                records.append({"run": name, "status": "already_complete"})
                continue
            command = [
                sys.executable,
                "-m",
                "llm_memory_editability.bios_train",
                *common,
                "--output",
                str(run),
                "--order",
                order,
                "--seed",
                str(seed),
                "--steps",
                "13280",
            ]
            log = run / "train.log"
        elif args.phase == "editing":
            command = [
                sys.executable,
                "-m",
                "llm_memory_editability.bios_edit",
                *common,
                "--checkpoint",
                str(run / "model-13280.pt"),
                "--output",
                str(run / "edits"),
                "--branches",
            ]
            log = run / "edit.log"
        else:
            for step in (2640, 5280, 13280):
                destination = run / f"organization-{step}"
                if (destination / "measurements.json").exists():
                    continue
                command = [
                    sys.executable,
                    "-m",
                    "llm_memory_editability.bios_measure",
                    *common,
                    "--checkpoint",
                    str(run / f"model-{step}.pt"),
                    "--output",
                    str(destination),
                ]
                with (run / f"measure-{step}.log").open("w") as stream:
                    result = subprocess.run(
                        command, env=environment, stdout=stream, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    records.append(
                        {"run": name, "status": "failed", "stage": "measure", "step": step}
                    )
                    break
            else:
                records.append({"run": name, "status": "complete", "stage": "measure"})
            continue
        print(
            json.dumps({"event": "start", "gpu": gpu, "run": name, "phase": args.phase}), flush=True
        )
        with log.open("w") as stream:
            result = subprocess.run(
                command, env=environment, stdout=stream, stderr=subprocess.STDOUT
            )
        records.append(
            {
                "run": name,
                "status": "complete" if result.returncode == 0 else "failed",
                "returncode": result.returncode,
                "phase": args.phase,
            }
        )
        print(json.dumps(records[-1]), flush=True)
        write_json(Path(args.output) / f"worker-{gpu}-{args.phase}.json", records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="results/bios-dev-v1")
    parser.add_argument("--phase", choices=["learning", "editing", "measurement"], required=True)
    args = parser.parse_args()
    jobs = [(world, seed, order) for seed in (0, 1) for world in (0, 1) for order in ("SA", "AS")]
    assignments = [jobs[i :: len(args.gpus)] for i in range(len(args.gpus))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, batch, args)
            for gpu, batch in zip(args.gpus, assignments, strict=True)
        ]
        results = [record for future in futures for record in future.result()]
    write_json(Path(args.output) / f"batch-{args.phase}.json", results)
    if any(record["status"] == "failed" for record in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
