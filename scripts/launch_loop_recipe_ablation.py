#!/usr/bin/env python3
"""Submit only four authorized ablations, preserving the six protected runs."""

import argparse
import fcntl
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
BATCH = "loop-recipe-ablation-20261025"
LOCAL = ROOT / "results" / BATCH
PERSISTENT = Path("/ossfs/workspace/llm-memory-editability") / BATCH
ARMS = {"supervision", "truncate", "supervision-truncate", "floor4"}


def gpu_row(snap, index):
    return snap["rows"].get(str(index), snap["rows"].get(index))


def child_of(pid, parent_pids):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid in parent_pids:
            return True
        seen.add(pid)
        try:
            line = next(
                x
                for x in Path(f"/proc/{pid}/status").read_text().splitlines()
                if x.startswith("PPid:")
            )
            pid = int(line.split()[1])
        except (OSError, StopIteration, ValueError):
            return False
    return False


def admit(current, baseline, auth, own_ids=()):
    old = {c["id"]: c for c in baseline["containers"]}
    now = {c["id"]: c for c in current["containers"]}
    protected = auth["baseline_containers"]
    if len(protected) != 6 or set(auth["assignments"]) != ARMS:
        raise ValueError("Exactly six protected runs and four new arms are required")
    if sorted(auth["assignments"].values()) != [4, 6, 7, 8]:
        raise ValueError("User explicitly specified GPUs 4/6/7/8")
    for cid, expected in protected.items():
        c = now.get(cid)
        if c is None or any(
            c["state"][k] != expected["state"][k] for k in ("Status", "Pid", "StartedAt")
        ):
            raise RuntimeError("Protected run stopped or restarted; inspect before admission")
        if c["state"]["Status"] != "running":
            raise RuntimeError("Protected run must remain running")
    original_pids = {c["state"]["Pid"] for c in protected.values()}
    own_pids = {now[cid]["state"]["Pid"] for cid in own_ids if cid in now}
    for gpu in auth["assignments"].values():
        row = gpu_row(current, gpu)
        if row["uuid"] != gpu_row(baseline, gpu)["uuid"]:
            raise RuntimeError("GPU identity changed")
        if row["free"] < 40000:
            raise RuntimeError(f"GPU {gpu} lacks 40000 MiB free headroom")
        old_pids = {p["pid"] for p in baseline["process_details"] if p["uuid"] == row["uuid"]}
        if not old_pids <= original_pids:
            raise RuntimeError("Baseline includes foreign CUDA processes")
        for p in current["process_details"]:
            if (
                p["uuid"] == row["uuid"]
                and p["pid"] not in original_pids | own_pids
                and not child_of(p["pid"], own_pids)
            ):
                raise RuntimeError(f"Foreign CUDA process on GPU {gpu}")
        for c in current["containers"]:
            if c["id"] in own_ids or not requested(c, gpu, row["uuid"]):
                continue
            previous = old.get(c["id"])
            if (
                previous is None
                or c["requests"] != previous["requests"]
                or any(
                    c["state"][k] != previous["state"][k] for k in ("Status", "Pid", "StartedAt")
                )
            ):
                raise RuntimeError(f"New or changed reservation on GPU {gpu}: {c['name']}")
    return current


def source_archive(archive):
    names = set(
        json.loads(
            (
                ROOT / "results/loop-fourway-terminal-20261024/runtime/source-manifest.json"
            ).read_text()
        )
    )
    names.update(
        [
            "scripts/run_loop_recipe_ablation.py",
            "scripts/launch_loop_recipe_ablation.py",
            "scripts/check_loop_recipe_ablation_cuda.py",
            "scripts/run_loop_recipe_pipeline.py",
            "src/llm_memory_editability/loop_recipe_ablation.py",
            "tests/test_loop_recipe_ablation.py",
            "tests/test_loop_recipe_launch.py",
            "tests/test_loop_terminal_training.py",
            "scripts/run_loop_feedback_controls.py",
            "configs/loop-feedback-controls-20261024.json",
            "configs/loop-native-handoff-development-v1.json",
            "pyproject.toml",
            f"configs/{BATCH}.json",
            f"results/{BATCH}/protocol.md",
        ]
    )
    manifest = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(names)}
    directories = {"project"}
    for name in names:
        directories.update(str(p) for p in Path("project", name).parents if str(p) != ".")
    with tarfile.open(archive, "w") as tar:
        for directory in sorted(directories, key=lambda p: (p.count("/"), p)):
            item = tarfile.TarInfo(directory)
            item.type, item.mode = tarfile.DIRTYPE, 0o555
            tar.addfile(item)
        for name in sorted(names):
            data = (ROOT / name).read_bytes()
            item = tarfile.TarInfo("project/" + name)
            item.mode, item.size = 0o444, len(data)
            tar.addfile(item, io.BytesIO(data))
    return manifest


def freeze():
    if PERSISTENT.exists():
        raise FileExistsError("Batch already prepared; never overwrite frozen evidence")
    archive = LOCAL / "runtime/source.tar"
    manifest = source_archive(archive)
    write(LOCAL / "runtime/source-manifest.json", manifest)
    write(
        LOCAL / "runtime/archive.json", {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    )
    PERSISTENT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(LOCAL / "runtime", PERSISTENT / "runtime")
    shutil.copy2(ROOT / "configs" / f"{BATCH}.json", PERSISTENT / "config.json")
    shutil.copy2(LOCAL / "protocol.md", PERSISTENT / "protocol.md")
    (PERSISTENT / "preflight").mkdir()


def create_command(config, arm, gpu, uuid, stage):
    name = f"{BATCH}-{arm}" + ("-preflight" if stage == "preflight" else "")
    cmd = DOCKER + [
        "create",
        "--name",
        name,
        "--label",
        f"lm-memory.batch={BATCH}",
        "--gpus",
        "device=" + uuid,
        "--cpus",
        "2",
        "--memory",
        "16g",
        "--memory-swap",
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
    env = {
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
    for key, value in env.items():
        cmd += ["--env", f"{key}={value}"]
    input_root = Path(config["runtime"]["persistent_root"]) / "loop-hop-development-v1/inputs/data"
    cmd += [
        "--mount",
        f"type=bind,source={input_root},target=/input,readonly",
        "--mount",
        f"type=bind,source={PERSISTENT},target=/output",
        config["runtime"]["image"],
        "python",
        "-u",
        "scripts/check_loop_recipe_ablation_cuda.py"
        if stage == "preflight"
        else "scripts/run_loop_recipe_pipeline.py",
        "--config",
        f"configs/{BATCH}.json",
        "--arm",
        arm,
    ]
    cmd += (
        ["--output", f"/output/preflight/{arm}.json"]
        if stage == "preflight"
        else ["--base", "/output"]
    )
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "launch"))
    parser.add_argument("--wandb-token-file", type=Path)
    args = parser.parse_args()
    lock = (LOCAL / "runtime/launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    auth = json.loads((LOCAL / "runtime/authorization.json").read_text())
    if (
        socket.gethostname() != auth["host"]
        or auth["updates_per_arm"] != 256000
        or auth["lr_schedule_steps"] != 1500000
    ):
        raise ValueError("Host, stopping budget or learning-rate budget differs from authorization")
    baseline = json.loads((LOCAL / "runtime/resource-before.json").read_text())
    config_path = ROOT / "configs" / f"{BATCH}.json"
    config = json.loads(config_path.read_text())
    if (
        set(config["arms"]) != ARMS
        or config["training"]["steps"] != 256000
        or config["training"]["lr_schedule_steps"] != 1500000
    ):
        raise ValueError("Only the four authorized short runs may start")
    hold = ROOT / "results/loop-queue-target-20261021/runtime/training-hold.json"
    if json.loads(hold.read_text())["state"] != "paused_by_user":
        raise RuntimeError("Old seven-run hold unexpectedly changed")
    before = admit(snapshot(), baseline, auth)
    if args.stage == "prepare":
        write(LOCAL / "runtime/admission-before-prepare.json", before)
        freeze()
        print(json.dumps({"prepared": True, "assignments": auth["assignments"]}))
        return
    bundle = {"archive": "source.tar", "manifest": "source-manifest.json"}
    amendment = PERSISTENT / "runtime/source-amendment.json"
    if amendment.exists():
        bundle = json.loads(amendment.read_text())
        if bundle["scientific_updates_before_amendment"] != 0:
            raise RuntimeError("This amendment is allowed only before any training")
    archive = (PERSISTENT / "runtime" / bundle["archive"]).read_bytes()
    expected = (
        bundle.get("sha256")
        or json.loads((PERSISTENT / "runtime/archive.json").read_text())["sha256"]
    )
    if (
        hashlib.sha256(archive).hexdigest() != expected
        or (PERSISTENT / "config.json").read_bytes() != config_path.read_bytes()
    ):
        raise RuntimeError("Frozen source or configuration changed")
    for name, digest in json.loads(
        (PERSISTENT / "runtime" / bundle["manifest"]).read_text()
    ).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Prepared source changed before launch: {name}")
    filename = "preflight-launch.json" if args.stage == "preflight" else "launch.json"
    if any((base / "runtime" / filename).exists() for base in (LOCAL, PERSISTENT)):
        raise FileExistsError("Stage already submitted; no automatic retry")
    secret = None
    if args.stage == "launch":
        for arm, gpu in auth["assignments"].items():
            report = json.loads((PERSISTENT / "preflight" / f"{arm}.json").read_text())
            if (
                not report["passed"]
                or report["gpu_index"] != gpu
                or report["peak_reserved_mib"] + 6144 >= report["allocator_cap_mib"]
            ):
                raise RuntimeError("CUDA preflight and per-task 6 GiB margin are required")
        if args.wandb_token_file is None:
            raise ValueError("W&B credential file is required")
        secret = credential_archive(args.wandb_token_file)
    record = {
        "batch": BATCH,
        "stage": args.stage,
        "state": "preparing",
        "source_archive_sha256": expected,
        "assignments": auth["assignments"],
        "containers": {},
        "updates_per_arm": 256000,
        "lr_schedule_steps": 1500000,
    }

    def save():
        for base in (LOCAL, PERSISTENT):
            write(base / "runtime" / filename, record)

    save()
    own_ids = set()
    for arm, gpu in auth["assignments"].items():
        admit(snapshot(), baseline, auth, own_ids)
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
        call(DOCKER + ["cp", "-", cid + ":/"], input=archive)
        if secret is not None:
            call(DOCKER + ["cp", "--archive", "-", cid + ":/"], input=secret)
    for arm, item in record["containers"].items():
        current = admit(snapshot(), baseline, auth, own_ids)
        write(PERSISTENT / "runtime" / f"admission-{args.stage}-{arm}.json", current)
        call(DOCKER + ["start", item["id"]], text=True)
        item["state"] = "started"
        save()
        print(json.dumps({"arm": arm, "gpu": item["gpu"], "started": True}), flush=True)
    record["state"] = "started"
    save()


if __name__ == "__main__":
    main()
