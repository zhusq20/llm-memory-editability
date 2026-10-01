"""Run the frozen confirmation matrix with bounded GPU concurrency."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def worker(gpu, names, config, mode):
    cfg = json.loads(Path(config).read_text())
    logs = Path(cfg["source_lock"]).parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name in names:
        run = Path(cfg["output_root"]) / name
        if mode == "train":
            commands = [
                [
                    sys.executable,
                    "scripts/run_text_pretrain.py",
                    "run",
                    "--config",
                    config,
                    "--run",
                    name,
                ]
            ]
        else:
            deadline = time.monotonic() + 3600
            while not (run / "complete.json").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(name)
                time.sleep(1)
            commands = [
                [
                    sys.executable,
                    "scripts/analyze_text_pretrain.py",
                    "--run",
                    str(run),
                    "--output",
                    str(Path(cfg["source_lock"]).parent / "mechanism" / name / str(node)),
                    "--node",
                    str(node),
                    "--position",
                    "3",
                ]
                for node in (16000, 32000)
            ]
        with (logs / f"{name}-{mode}.log").open("a") as log:
            for command in commands:
                subprocess.run(
                    command,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        print(json.dumps(dict(run=name, mode=mode, status="complete")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/text-pretrain-confirmation-v1.json")
    parser.add_argument("--mode", choices=["train", "mechanism"], default="train")
    args = parser.parse_args()
    names = list(json.loads(Path(args.config).read_text())["runs"])
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(worker, gpu, names[i::6], args.config, args.mode)
            for i, gpu in enumerate((2, 2, 2, 5, 5, 5))
        ]
        for future in futures:
            future.result()
