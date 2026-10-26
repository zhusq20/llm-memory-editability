#!/usr/bin/env python3
"""Freeze and submit the explicitly authorized four-GPU comparison once."""

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import tarfile
from pathlib import Path

from lm_resource_snapshot import snapshot

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-fourway-terminal-20261024"
DOCKER = ["docker", "--context", "lm-memory"]
ASSIGNMENTS = {"invariant": 6, "loop": 4, "discoloop": 7, "infiloop": 8}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def call(argv, **kwargs):
    return subprocess.run(argv, check=True, capture_output=True, timeout=120, **kwargs)


def requested(record, gpu, uuid):
    return any(
        r.get("Count") == -1
        or str(gpu) in (r.get("DeviceIDs") or [])
        or uuid in (r.get("DeviceIDs") or [])
        for r in record["requests"] or []
    )


def admit(baseline, own_ids):
    current = snapshot()
    baseline_containers = {c["id"]: c for c in baseline["containers"]}
    for gpu in ASSIGNMENTS.values():
        row = current["rows"][gpu]
        if row["uuid"] != baseline["rows"][str(gpu)]["uuid"]:
            raise RuntimeError("GPU identity changed")
        if row["used"] > 512 or row["free"] < 26000 or row["uuid"] in current["processes"]:
            raise RuntimeError(f"GPU {gpu} is not idle; nothing else will start")
        for c in current["containers"]:
            if c["id"] in own_ids or not requested(c, gpu, row["uuid"]):
                continue
            previous = baseline_containers.get(c["id"])
            if previous is None or (c["state"]["Status"], c["requests"]) != (
                previous["state"]["Status"],
                previous["requests"],
            ):
                raise RuntimeError(f"New/changed reservation on GPU {gpu}: {c['name']}")
    return current


def credential_archive(path):
    secret = path.read_bytes().strip()
    if len(secret) < 20:
        raise ValueError("Missing W&B credential")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        directory = tarfile.TarInfo("run/secrets")
        directory.type, directory.mode = tarfile.DIRTYPE, 0o755
        tar.addfile(directory)
        entry = tarfile.TarInfo("run/secrets/wandb-key")
        entry.uid, entry.gid, entry.mode, entry.size = os.getuid(), os.getgid(), 0o400, len(secret)
        tar.addfile(entry, io.BytesIO(secret))
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != "sn4622129202":
        raise ValueError("This authorization specifies the current host's GPUs 4/6/7/8")
    local = ROOT / "results" / BATCH
    persistent = Path("/ossfs/workspace/llm-memory-editability") / BATCH
    if (local / "runtime/launch.json").exists() or (persistent / "runtime/launch.json").exists():
        raise FileExistsError("Submission exists; inspect it rather than launching duplicates")
    baseline = json.loads((local / "runtime/resource-before.json").read_text())
    authorization = json.loads((local / "runtime/authorization.json").read_text())
    if authorization["formal_gpu_indices"] != [4, 6, 7, 8]:
        raise ValueError("Missing explicit permission for the reserved idle GPUs")
    config_path = ROOT / "configs" / f"{BATCH}.json"
    config = json.loads(config_path.read_text())
    before = admit(baseline, set())
    names = set(
        json.loads((ROOT / "results/loop-terminal-cuda-20261024/source-manifest.json").read_text())
    )
    names.update(
        str(p.relative_to(ROOT)) for p in (ROOT / "src/llm_memory_editability").glob("*.py")
    )
    names.update(
        [
            "scripts/run_loop_fourway.py",
            "scripts/launch_loop_fourway.py",
            "scripts/lm_resource_snapshot.py",
            f"configs/{BATCH}.json",
            "tests/test_loop_comparison.py",
        ]
    )
    names.update(str(p.relative_to(ROOT)) for p in (local / "references").iterdir())
    names.add(f"results/{BATCH}/protocol.md")
    manifest = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)
    }
    directories = {"project"}
    for name in names:
        directories.update(str(p) for p in Path("project", name).parents if str(p) != ".")
    archive = local / "runtime/source.tar"
    with tarfile.open(archive, "w") as tar:
        for directory in sorted(directories, key=lambda p: (p.count("/"), p)):
            entry = tarfile.TarInfo(directory)
            entry.type, entry.mode = tarfile.DIRTYPE, 0o555
            tar.addfile(entry)
        for name in sorted(names):
            content = (ROOT / name).read_bytes()
            entry = tarfile.TarInfo("project/" + name)
            entry.mode, entry.size = 0o444, len(content)
            tar.addfile(entry, io.BytesIO(content))
    write(local / "runtime/source-manifest.json", manifest)
    write(local / "runtime/admission-before-create.json", before)
    if not args.execute:
        print(json.dumps({"ready": True, "assignments": ASSIGNMENTS, "updates_per_arm": 1500000}))
        return
    private_archive = credential_archive(args.wandb_token_file)
    persistent.mkdir(parents=True, exist_ok=False)
    for name in ("runtime", "references"):
        shutil.copytree(local / name, persistent / name)
    shutil.copy2(local / "protocol.md", persistent / "protocol.md")
    shutil.copy2(config_path, persistent / "config.json")
    launch = {
        "batch": BATCH,
        "state": "preparing",
        "assignments": ASSIGNMENTS,
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "containers": {},
    }

    def save():
        for base in (local, persistent):
            write(base / "runtime/launch.json", launch)

    save()
    for arm, gpu in ASSIGNMENTS.items():
        uuid = baseline["rows"][str(gpu)]["uuid"]
        command = DOCKER + [
            "create",
            "--name",
            f"{BATCH}-{arm}",
            "--label",
            f"lm-memory.batch={BATCH}",
            "--gpus",
            "device=" + uuid,
            "--cpus",
            "2",
            "--memory",
            "16g",
            "--shm-size",
            "1g",
            "--pids-limit",
            "256",
            "--network",
            "host",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=2g",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--workdir",
            "/project",
            "--env",
            "PYTHONPATH=/project/src:/project/scripts",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "XDG_CACHE_HOME=/tmp/cache",
            "--env",
            "CUDA_CACHE_PATH=/tmp/cuda-cache",
            "--env",
            "WANDB_CACHE_DIR=/tmp/wandb-cache",
            "--env",
            "WANDB_CONFIG_DIR=/tmp/wandb-config",
            "--env",
            "OMP_NUM_THREADS=2",
            "--env",
            "MKL_NUM_THREADS=2",
            "--env",
            f"LM_HOST_GPU_INDEX={gpu}",
            "--env",
            f"LM_HOST_GPU_UUID={uuid}",
            "--mount",
            "type=bind,source=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/loop-hop-development-v1/inputs/data,target=/input,readonly",
            "--mount",
            f"type=bind,source={persistent},target=/output",
            config["runtime"]["image"],
            "python",
            "-u",
            "scripts/run_loop_fourway.py",
            "--config",
            f"configs/{BATCH}.json",
            "--base",
            "/output",
            "--arm",
            arm,
        ]
        cid = call(command, text=True).stdout.strip()
        launch["containers"][arm] = {
            "id": cid,
            "gpu": gpu,
            "uuid": uuid,
            "state": "created",
            "command": command,
        }
        save()
        call(DOCKER + ["cp", "--archive", "-", cid + ":/"], input=archive.read_bytes())
        call(DOCKER + ["cp", "--archive", "-", cid + ":/"], input=private_archive)
    own = {c["id"] for c in launch["containers"].values()}
    write(persistent / "runtime/admission-before-start.json", admit(baseline, own))
    for arm, item in launch["containers"].items():
        call(DOCKER + ["start", item["id"]], text=True)
        item["state"] = "started"
        save()
        print(
            json.dumps({"arm": arm, "gpu": item["gpu"], "state": "started", "id": item["id"]}),
            flush=True,
        )
    launch["state"] = "started"
    save()


if __name__ == "__main__":
    main()
