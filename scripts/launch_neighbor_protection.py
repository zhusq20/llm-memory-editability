#!/usr/bin/env python3
"""Launch one container per neighbor-protection-development-v1 arm on a chosen GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from launch_untied_discoloop import DOCKER, PERSISTENT, ROOT, credential_archive

CONFIG = "configs/neighbor-protection-development-v1.json"


def gpu_memory(index):
    row = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-used-mib", type=int, default=512)
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=Path.home() / ".cache/lm-memory-ce-ablation/wandb-token",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text())
    if args.arm not in config["arms"]:
        raise SystemExit(f"Unknown arm {args.arm}")
    if gpu_memory(args.gpu) > args.max_used_mib:
        raise SystemExit(f"GPU {args.gpu} is busy; nothing started")
    out = PERSISTENT / config["batch"]
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o777)
    if (out / args.arm).exists():
        raise SystemExit("Arm output exists; preserve it and relaunch under a new batch")
    runtime = config["runtime"]
    name = f"np1-{args.arm}"
    script = (
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"\n'
        f"python scripts/run_neighbor_protection.py --arm {args.arm} --out /outputs\n"
    )
    command = [
        *DOCKER,
        "create",
        "--name",
        name,
        "--gpus",
        f"device={args.gpu}",
        "--cpus",
        str(runtime["cpus"]),
        "--memory",
        f"{runtime['memory_gib']}g",
        "--shm-size",
        "8g",
        "--network",
        "host",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-w",
        "/repo",
        "-e",
        "HOME=/tmp",
        "-e",
        "PYTHONPATH=/repo/src",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "WANDB_CONFIG_DIR=/tmp/wandb-config",
        "-e",
        "WANDB_CACHE_DIR=/tmp/wandb-cache",
        "-e",
        "WANDB_DISABLE_CODE=true",
        "--mount",
        f"type=bind,source={ROOT},target=/repo,readonly",
        "--mount",
        f"type=bind,source={PERSISTENT},target=/ossfs/workspace/llm-memory-editability,readonly",
        "--mount",
        f"type=bind,source={out},target=/outputs",
        runtime["image"],
        "bash",
        "-c",
        script,
    ]
    subprocess.run(command, check=True)
    subprocess.run(
        [*DOCKER, "cp", "-", f"{name}:/"],
        input=credential_archive(args.wandb_token_file),
        check=True,
    )
    subprocess.run([*DOCKER, "start", name], check=True)
    print(json.dumps({"container": name, "gpu": args.gpu, "out": str(out / args.arm)}))


if __name__ == "__main__":
    main()
