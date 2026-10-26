#!/usr/bin/env python3
"""Freeze, check and start the authorized two shared-GPU controls once."""

import argparse
import hashlib
import io
import json
import os
import shutil
import socket
import tarfile
from pathlib import Path

from launch_loop_fourway import DOCKER, call, credential_archive, requested, write
from lm_resource_snapshot import snapshot

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-feedback-controls-20261024"
LOCAL = ROOT / "results" / BATCH
PERSISTENT = Path("/ossfs/workspace/llm-memory-editability") / BATCH


def gpu_row(snap, index):
    return snap["rows"].get(str(index), snap["rows"].get(index))


def check_shared(current, baseline, authorization, own_ids=()):
    """Reject new processes/reservations and retain every original baseline PID."""
    old = {c["id"]: c for c in baseline["containers"]}
    now = {c["id"]: c for c in current["containers"]}
    for cid, expected in authorization["baseline_containers"].items():
        actual = now.get(cid)
        if actual is None or any(
            actual["state"][k] != expected["state"][k] for k in ("Status", "Pid", "StartedAt")
        ):
            raise RuntimeError(
                "An original baseline stopped or restarted; inspect before admission"
            )
        if actual["state"]["Status"] != "running":
            raise RuntimeError("Expected each of the four original baselines to remain running")
    own_pids = {now[cid]["state"]["Pid"] for cid in own_ids if cid in now}
    assignments = authorization["assignments"]
    if len(assignments) != 2 or len(set(assignments.values())) != 2:
        raise ValueError("Exactly two distinct shared GPUs are required")
    for gpu in assignments.values():
        row, previous_row = gpu_row(current, gpu), gpu_row(baseline, gpu)
        if row["uuid"] != previous_row["uuid"]:
            raise RuntimeError("GPU identity changed")
        # Covers two 0.16 allocator caps plus 8 GiB even before reclaiming any
        # memory currently held by the original task (cards are about 95 GiB).
        if row["free"] < 40000:
            raise RuntimeError(f"Insufficient shared memory headroom on GPU {gpu}")
        original_pids = {c["state"]["Pid"] for c in authorization["baseline_containers"].values()}
        allowed = {p["pid"] for p in baseline["process_details"] if p["uuid"] == row["uuid"]}
        if not allowed <= original_pids:
            raise RuntimeError("The selected baseline snapshot includes unrelated CUDA users")
        allowed |= own_pids
        for p in current["process_details"]:
            if p["uuid"] == row["uuid"] and p["pid"] not in allowed:
                raise RuntimeError(f"New CUDA process on GPU {gpu}: {p['pid']}")
        for c in current["containers"]:
            if c["id"] in own_ids or not requested(c, gpu, row["uuid"]):
                continue
            previous = old.get(c["id"])
            if previous is None or (c["state"]["Status"], c["requests"], c["state"]["Pid"]) != (
                previous["state"]["Status"],
                previous["requests"],
                previous["state"]["Pid"],
            ):
                raise RuntimeError(f"New/changed reservation on GPU {gpu}: {c['name']}")
    return current


def freeze():
    if PERSISTENT.exists() or (LOCAL / "runtime/source-manifest.json").exists():
        raise FileExistsError("Prepared batch exists; never overwrite its source")
    names = set(
        json.loads(
            (
                ROOT / "results/loop-fourway-terminal-20261024/runtime/source-manifest.json"
            ).read_text()
        )
    )
    names.update(
        [
            "scripts/run_loop_feedback_controls.py",
            "scripts/launch_loop_feedback_controls.py",
            "scripts/check_loop_feedback_cuda.py",
            "tests/test_loop_feedback_shared_launch.py",
            f"configs/{BATCH}.json",
            f"results/{BATCH}/protocol.md",
        ]
    )
    manifest = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)
    }
    dirs = {"project"}
    for name in names:
        dirs.update(str(p) for p in Path("project", name).parents if str(p) != ".")
    archive = LOCAL / "runtime/source.tar"
    with tarfile.open(archive, "w") as tar:
        for directory in sorted(dirs, key=lambda p: (p.count("/"), p)):
            entry = tarfile.TarInfo(directory)
            entry.type, entry.mode = tarfile.DIRTYPE, 0o555
            tar.addfile(entry)
        for name in sorted(names):
            data = (ROOT / name).read_bytes()
            entry = tarfile.TarInfo("project/" + name)
            entry.mode, entry.size = 0o444, len(data)
            tar.addfile(entry, io.BytesIO(data))
    write(LOCAL / "runtime/source-manifest.json", manifest)
    write(
        LOCAL / "runtime/archive.json",
        {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
    )
    PERSISTENT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(LOCAL / "runtime", PERSISTENT / "runtime")
    shutil.copy2(ROOT / "configs" / f"{BATCH}.json", PERSISTENT / "config.json")
    shutil.copy2(LOCAL / "protocol.md", PERSISTENT / "protocol.md")
    shutil.copy2(LOCAL / "validation.json", PERSISTENT / "cpu-validation-before-authorization.json")
    (PERSISTENT / "preflight").mkdir()


def create_command(config, arm, gpu, uuid, stage):
    suffix = "-preflight" if stage == "preflight" else ""
    command = DOCKER + [
        "create",
        "--name",
        f"{BATCH}-{arm}{suffix}",
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
        "none" if stage == "preflight" else "host",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        "/project",
    ]
    environment = {
        "PYTHONPATH": "/project/src:/project/scripts",
        "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_CACHE_HOME": "/tmp/cache",
        "CUDA_CACHE_PATH": "/tmp/cuda-cache",
        "WANDB_CACHE_DIR": "/tmp/wandb-cache",
        "WANDB_CONFIG_DIR": "/tmp/wandb-config",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "LM_HOST_GPU_INDEX": str(gpu),
        "LM_HOST_GPU_UUID": uuid,
    }
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    input_root = Path(config["runtime"]["persistent_root"]) / "loop-hop-development-v1/inputs/data"
    command += [
        "--mount",
        f"type=bind,source={input_root},target=/input,readonly",
        "--mount",
        f"type=bind,source={PERSISTENT},target=/output",
        config["runtime"]["image"],
        "python",
        "-u",
        "scripts/check_loop_feedback_cuda.py"
        if stage == "preflight"
        else "scripts/run_loop_feedback_controls.py",
        "--config",
        f"configs/{BATCH}.json",
        "--arm",
        arm,
    ]
    if stage == "preflight":
        command += ["--output", f"/output/preflight/{arm}.json"]
    else:
        command += ["--base", "/output"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "launch"))
    parser.add_argument("--wandb-token-file", type=Path)
    args = parser.parse_args()
    authorization = json.loads((LOCAL / "runtime/authorization.json").read_text())
    if socket.gethostname() != authorization["host"]:
        raise RuntimeError("Authorization is for the recorded host")
    if authorization["updates_per_arm"] != 1500000:
        raise ValueError("Expected two authorized 1,500,000-update runs")
    baseline = json.loads((LOCAL / "runtime/resource-before.json").read_text())
    config = json.loads((ROOT / "configs" / f"{BATCH}.json").read_text())
    if set(authorization["assignments"]) != set(config["arms"]):
        raise ValueError("GPU assignments do not match the two controls")
    before = check_shared(snapshot(), baseline, authorization)
    if args.stage == "prepare":
        freeze()
        write(LOCAL / "runtime/admission-before-prepare.json", before)
        print(json.dumps({"prepared": True, "assignments": authorization["assignments"]}))
        return
    archive = (PERSISTENT / "runtime/source.tar").read_bytes()
    expected = json.loads((PERSISTENT / "runtime/archive.json").read_text())["sha256"]
    if hashlib.sha256(archive).hexdigest() != expected:
        raise RuntimeError("Frozen archive changed")
    if (PERSISTENT / "config.json").read_bytes() != (
        ROOT / "configs" / f"{BATCH}.json"
    ).read_bytes():
        raise RuntimeError("Config changed after freezing")
    record_name = "preflight-launch.json" if args.stage == "preflight" else "launch.json"
    if any((base / "runtime" / record_name).exists() for base in (LOCAL, PERSISTENT)):
        raise FileExistsError("Submission already recorded; inspect rather than retry")
    secret = None
    if args.stage == "launch":
        for arm, gpu in authorization["assignments"].items():
            report = json.loads((PERSISTENT / "preflight" / f"{arm}.json").read_text())
            if not report["passed"] or report["gpu_index"] != gpu:
                raise RuntimeError("Each control must pass CUDA checks on its selected GPU")
            if report["peak_reserved_mib"] + 6144 >= report["allocator_cap_mib"]:
                raise RuntimeError("Missing 6 GiB of per-task headroom")
        if args.wandb_token_file is None:
            raise ValueError("Tracking credential file required for formal training")
        secret = credential_archive(args.wandb_token_file)
    record = {
        "batch": BATCH,
        "state": "preparing",
        "stage": args.stage,
        "source_archive_sha256": expected,
        "assignments": authorization["assignments"],
        "containers": {},
    }

    def save():
        for base in (LOCAL, PERSISTENT):
            write(base / "runtime" / record_name, record)

    save()
    own_ids = set()
    for arm, gpu in authorization["assignments"].items():
        check_shared(snapshot(), baseline, authorization, own_ids)
        uuid = gpu_row(baseline, gpu)["uuid"]
        command = create_command(config, arm, gpu, uuid, args.stage)
        cid = call(command, text=True).stdout.strip()
        own_ids.add(cid)
        record["containers"][arm] = {
            "id": cid,
            "gpu": gpu,
            "uuid": uuid,
            "state": "created",
            "command": command,
        }
        save()
        # Without --archive Docker keeps tar UID 0; --archive maps it to the
        # configured runtime user for stdin copies on this Docker version.
        call(DOCKER + ["cp", "-", cid + ":/"], input=archive)
        if secret is not None:
            call(DOCKER + ["cp", "--archive", "-", cid + ":/"], input=secret)
    for arm, item in record["containers"].items():
        current = check_shared(snapshot(), baseline, authorization, own_ids)
        write(PERSISTENT / "runtime" / f"admission-{args.stage}-{arm}.json", current)
        call(DOCKER + ["start", item["id"]], text=True)
        item["state"] = "started"
        save()
        print(json.dumps({"arm": arm, "gpu": item["gpu"], "id": item["id"], "started": True}))
    record["state"] = "started"
    save()


if __name__ == "__main__":
    main()
