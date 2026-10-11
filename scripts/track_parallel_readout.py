"""Track a single zero-update readout process without changing its calculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}/{key}" if prefix else key))
    elif isinstance(value, int | float | bool) and value is not None:
        result[prefix] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == "--" else args.command
    config = json.loads(args.config.read_text())
    task = next(task for task in config["execution"]["tasks"] if task["name"] == args.task)
    import wandb

    identity = hashlib.sha256(f"{config['batch']}/{args.task}".encode()).hexdigest()[:12]
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        id=identity,
        resume="never",
        name=f"{config['batch']}-{args.task}",
        group=config["batch"],
        job_type=task["kind"],
        dir=str(args.out),
        config={
            "batch": config["batch"],
            "task": args.task,
            "kind": task["kind"],
            "new_optimizer_updates": 0,
            "physical_gpu": task["gpu"],
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "command": command,
        },
    )
    (args.out / "tracking.json").write_text(
        json.dumps({"run_id": run.id, "url": run.url, "new_optimizer_updates": 0}, indent=2) + "\n"
    )
    start = time.monotonic()
    process = subprocess.Popen(command)
    while process.poll() is None:
        run.log({"wall_seconds": time.monotonic() - start, "new_optimizer_updates": 0})
        time.sleep(5)
    code = process.wait()
    run.summary.update(
        {"exit_code": code, "wall_seconds": time.monotonic() - start, "new_optimizer_updates": 0}
    )
    summary = args.out / "summary.json"
    if code == 0 and summary.exists():
        run.summary.update(flatten(json.loads(summary.read_text()), "readout"))
    run.finish(exit_code=code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
