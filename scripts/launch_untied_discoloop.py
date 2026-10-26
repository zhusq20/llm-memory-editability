#!/usr/bin/env python3
"""Launch untied-discoloop-development-v1 (train then fresh-process audit) in one container."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERSISTENT = Path(
    "/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability"
)
DOCKER = ["docker", "--context", "lm-memory"]
CONFIG = "configs/untied-discoloop-development-v1.json"


def credential_archive(path):
    """Tar stream for docker cp, so the W&B key never appears in argv or logs."""
    path = Path(path)
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("A private W&B credential file with mode 0600 or 0400 is required")
    value = path.read_bytes().strip()
    if not value:
        raise ValueError("Empty tracking credential")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(info)
        info = tarfile.TarInfo("run/secrets/wandb")
        info.size, info.mode, info.uid, info.gid = len(value), 0o400, os.getuid(), os.getgid()
        archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


def gpu_is_free(index):
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for row in rows:
        gpu, used, utilization = (int(part) for part in row.split(","))
        if gpu == index:
            return used <= 512 and utilization <= 5
    raise ValueError(f"GPU {index} not found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="untied-discoloop-v1")
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=Path.home() / ".cache/lm-memory-ce-ablation/wandb-token",
    )
    args = parser.parse_args()
    config_path = ROOT / CONFIG
    config = json.loads(config_path.read_text())
    runtime = config["runtime"]
    if not gpu_is_free(runtime["gpu"]):
        raise SystemExit(f"GPU {runtime['gpu']} is busy; nothing started")
    out = PERSISTENT / config["batch"] / args.name
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o777)
    mounts = [
        (ROOT, "/repo", True),
        (PERSISTENT / "ttcd-successful-base-v1/input-mount/data", "/inputs/data", True),
        (
            PERSISTENT / "ttcd-architecture-comparison-v1/input-mount/initial",
            "/inputs/initial",
            True,
        ),
        (out, "/outputs", False),
    ]
    script = (
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"\n'
        "set -e\n"
        f"python scripts/run_untied_discoloop.py train --config {CONFIG} --out /outputs\n"
        f"python scripts/run_untied_discoloop.py audit --config {CONFIG} --out /outputs\n"
    )
    command = [
        *DOCKER,
        "create",
        "--name",
        args.name,
        "--gpus",
        f"device={runtime['gpu']}",
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
    ]
    for source, target, read_only in mounts:
        flag = ",readonly" if read_only else ""
        command += ["--mount", f"type=bind,source={source},target={target}{flag}"]
    command += [runtime["image"], "bash", "-c", script]
    subprocess.run(command, check=True)
    subprocess.run(
        [*DOCKER, "cp", "-", f"{args.name}:/"],
        input=credential_archive(args.wandb_token_file),
        check=True,
    )
    if not gpu_is_free(runtime["gpu"]):
        subprocess.run([*DOCKER, "rm", args.name], check=True)
        raise SystemExit(f"GPU {runtime['gpu']} became busy; container removed")
    subprocess.run([*DOCKER, "start", args.name], check=True)
    print(json.dumps({"container": args.name, "out": str(out), "gpu": runtime["gpu"]}))


if __name__ == "__main__":
    main()
