"""Independent initialization of the fixed entity-representation contrast."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import diagnose_realworld_entities as experiment

SOURCE = Path(__file__).resolve()
experiment.SEED = 916001
experiment.ROOT = experiment.common.PROJECT / "results/realworld-loop-entity-replica-v1"
experiment.ART = (
    experiment.common.PROJECT / "docs/development-artifacts/realworld-loop-entity-replica-v1"
)


def pipeline(gpu, jobs):
    while True:
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = {int(a): int(b) for a, b in (line.split(",") for line in memory.splitlines())}
        if used[gpu] < 100:
            break
        time.sleep(10)
    for arch, condition in jobs:
        out = experiment.ROOT / "development" / f"{arch}-{condition}"
        out.mkdir(parents=True, exist_ok=True)
        for mode in ["worker", "audit"]:
            with (out / f"{mode}.log").open("a") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(SOURCE),
                        mode,
                        "--gpu",
                        str(gpu),
                        "--arch",
                        arch,
                        "--condition",
                        condition,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["worker", "audit", "pipeline"])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--arch", choices=["standard8", "loop4x2"])
    parser.add_argument("--condition", choices=["entity", "natural"])
    args = parser.parse_args()
    if args.command == "pipeline":
        jobs = {
            3: [("standard8", "natural")],
            4: [("loop4x2", "natural")],
            7: [("standard8", "entity"), ("loop4x2", "entity")],
        }[args.gpu]
        pipeline(args.gpu, jobs)
    else:
        {"worker": experiment.worker, "audit": experiment.audit}[args.command](
            args.arch, args.condition, args.gpu
        )
