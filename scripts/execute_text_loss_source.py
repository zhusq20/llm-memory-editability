"""Execute a frozen loss-source matrix using a bounded number of GPU workers."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def worker(names, config, gpu):
    cfg = json.loads(Path(config).read_text())
    logs = Path(cfg["lock"]).parent / "logs"
    logs.mkdir(exist_ok=True, parents=True)
    for name in names:
        with (logs / (name + ".log")).open("a") as f:
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run_text_loss_source.py",
                    "run",
                    "--config",
                    config,
                    "--run",
                    name,
                ],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(name + " complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu", type=int, default=2)
    args = p.parse_args()
    names = list(json.loads(Path(args.config).read_text())["runs"])
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(worker, names[i :: args.workers], args.config, args.gpu)
            for i in range(args.workers)
        ]
        for f in futures:
            f.result()
