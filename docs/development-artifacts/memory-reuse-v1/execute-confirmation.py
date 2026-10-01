"""Run the fixed 12-cell matrix in separate processes on physical GPU 2."""

import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

CONFIG = "configs/memory-reuse-confirmation-v1.json"
ROOT = Path("docs/development-artifacts/memory-reuse-v1/confirmation")
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
config = json.loads(Path(CONFIG).read_text())
tasks = list(itertools.product(config["worlds"], config["initializations"], config["arms"]))


def execute(task):
    world, init, arm = task
    name = f"w{world}-i{init}-{arm}"
    command = [
        sys.executable,
        "scripts/run_memory_reuse.py",
        "run",
        "--config",
        CONFIG,
        "--world",
        str(world),
        "--init",
        str(init),
        "--arm",
        arm,
    ]
    with (LOGS / f"{name}.log").open("a") as log:
        completed = subprocess.run(
            command,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="2"),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = dict(
        run=name, returncode=completed.returncode, utc=datetime.now(timezone.utc).isoformat()
    )
    (LOGS / f"{name}.process.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


(ROOT / "execution-start.json").write_text(
    json.dumps(
        dict(
            utc=datetime.now(timezone.utc).isoformat(), tasks=tasks, max_workers=3, physical_gpu=2
        ),
        indent=2,
    )
    + "\n"
)
with ThreadPoolExecutor(max_workers=3) as pool:
    records = list(pool.map(execute, tasks))
(ROOT / "execution-completion.json").write_text(json.dumps(records, indent=2) + "\n")
if any(row["returncode"] != 0 for row in records):
    raise SystemExit(1)
