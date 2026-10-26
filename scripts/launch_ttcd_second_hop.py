#!/usr/bin/env python3
"""Plan or execute independent per-GPU LM Docker jobs, with no implicit launch.

Default is a read-only plan. --execute explicitly launches the selected stage.
The controller refuses occupied GPUs and never substitutes default Docker or
another project's daemon for the required lm-memory context.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def persistent_batch(config):
    root = Path(config["runtime"].get("persistent_root", "/ossfs/workspace/llm-memory-editability"))
    if not root.is_dir() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise RuntimeError("Prepare the required /ossfs persistent mount before formal execution")
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Formal LM outputs must remain below /ossfs/workspace")
    return root / config["batch"]


def freeze_source(batch, config_path):
    """Snapshot before any job; later stages verify rather than refresh it."""
    source = batch / "source"
    manifest = batch / "source-manifest.json"
    if manifest.exists():
        frozen = json.loads(manifest.read_text())
        for relative, expected in frozen["files"].items():
            if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source changed: {relative}")
        if hashlib.sha256(config_path.read_bytes()).hexdigest() != frozen["config_sha256"]:
            raise ValueError("Configuration changed; create a separately named amended batch")
        return source / "configs/experiment.json"
    source.mkdir(exist_ok=False)
    # The Docker daemon is host root, squashed by this server's NFS. It must
    # traverse the mount source; file contents stay under the invoking UID.
    source.chmod(0o755)
    shutil.copytree(
        ROOT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (source / "scripts").mkdir()
    (source / "configs").mkdir()
    for name in ("run_ttcd_second_hop.py", "launch_ttcd_second_hop.py"):
        shutil.copy2(ROOT / "scripts" / name, source / "scripts" / name)
    helper = ROOT / "scripts/prepare_ttcd_scratch.py"
    if helper.exists():
        shutil.copy2(helper, source / "scripts" / helper.name)
    shutil.copy2(config_path, source / "configs/experiment.json")
    config = json.loads(config_path.read_text())
    if "scratch" in config:
        backup = source / "input-backup"
        backup.mkdir()
        for name in ("world.npz", "panels.npz", "complete.json", "training-split.json"):
            shutil.copy2(Path(config["data_dir"]) / name, backup / name)
        shutil.copy2(config["model_config"], backup / "model-config.json")
    shutil.copy2(
        ROOT / "configs/experiment-tracking-defaults.json",
        source / "configs/experiment-tracking-defaults.json",
    )
    frozen = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "files": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
    }
    manifest.write_text(json.dumps(frozen, indent=2) + "\n")
    return source / "configs/experiment.json"


def gpu_snapshot():
    lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rows = {}
    for line in lines:
        index, uuid, memory, utilization = [s.strip() for s in line.split(",")]
        rows[int(index)] = {"uuid": uuid, "memory": int(memory), "utilization": int(utilization)}
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return rows, set(processes)


def check_gpu(gpu, runtime, snapshot):
    rows, processes = snapshot
    if gpu not in rows:
        raise ValueError(f"Requested GPU {gpu} does not exist on this host")
    row = rows[gpu]
    if (
        row["memory"] > runtime["gpu_memory_max_mib"]
        or row["utilization"] > runtime["gpu_utilization_max"]
        or row["uuid"] in processes
    ):
        raise RuntimeError(f"GPU {gpu} is occupied; no preemption or automatic reassignment")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("preflight", "train", "train-audit", "calibrate", "evaluate", "audit")
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ttcd-second-hop-scratch-v1.json"
    )
    parser.add_argument("--model", choices=("discoloop", "loop", "both"), default="both")
    parser.add_argument("--image", default="lm-memory-ttcd:torch2.8-cu128")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=Path.home() / ".cache/llm-memory-ttcd/wandb-token",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    models = list(config["models"]) if args.model == "both" else [args.model]
    plan = {
        "stage": args.stage,
        "docker_context": "lm-memory",
        "image": args.image,
        "jobs": [{"model": name, "gpu": config["runtime"]["gpus"][name]} for name in models],
        "executed": args.execute,
        "persistent_root": config["runtime"].get(
            "persistent_root", "/ossfs/workspace/llm-memory-editability"
        ),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not args.config.resolve().is_relative_to(ROOT):
        raise ValueError("Put the reviewed configuration inside the mounted project")
    if config["runtime"]["tracking"] != "disabled":
        if not args.wandb_token_file.is_file():
            raise FileNotFoundError(
                "Provide the local private W&B credential file before execution"
            )
        if args.wandb_token_file.stat().st_mode & 0o077:
            raise ValueError("W&B credential file must not be accessible to group/others")
    batch = persistent_batch(config)
    subprocess.run(["docker", "--context", "lm-memory", "info"], check=True, capture_output=True)
    image = json.loads(
        subprocess.run(
            [
                "docker",
                "--context",
                "lm-memory",
                "image",
                "inspect",
                args.image,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )[0]["Id"]
    runtime = config["runtime"]
    batch.mkdir(parents=True, exist_ok=True)
    batch.chmod(0o755)
    with (batch / "controller.lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen_config = freeze_source(batch, args.config.resolve())
        (batch / "controller-process.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": args.stage,
                    "models": models,
                    "image": image,
                    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "source_manifest": str(batch / "source-manifest.json"),
                },
                indent=2,
            )
            + "\n"
        )
        (batch / "controller-state.json").write_text(
            json.dumps({"state": "running", **plan}) + "\n"
        )
        available_kib = next(
            int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if available_kib < len(models) * runtime["memory_gib"] * 1024**2:
            raise RuntimeError("Insufficient available host memory for the requested parallel jobs")
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) < len(models) * runtime["cpu_threads"]:
            raise RuntimeError("Insufficient CPU affinity for separate task allocations")

        def worker(name):
            gpu = runtime["gpus"][name]
            lock_dir = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
            lock_dir.mkdir(exist_ok=True)
            with (lock_dir / f"gpu-{gpu}.lock").open("a") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                row = check_gpu(gpu, runtime, gpu_snapshot())
                out = batch / name
                out.mkdir(exist_ok=True)
                out.chmod(0o755)
                config_path = "/frozen/configs/experiment.json"
                offset = models.index(name) * runtime["cpu_threads"]
                task_cpus = cpus[offset : offset + runtime["cpu_threads"]]
                container_name = f"{config['batch']}-{name}-{args.stage}"
                command = [
                    "docker",
                    "--context",
                    "lm-memory",
                    "run",
                    "--name",
                    container_name,
                    "--gpus",
                    f"device={row['uuid']}",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--cpus",
                    str(runtime["cpu_threads"]),
                    "--cpuset-cpus",
                    ",".join(map(str, task_cpus)),
                    "--memory",
                    f"{runtime['memory_gib']}g",
                    "--shm-size",
                    "2g",
                    "--network",
                    "host",
                    "--workdir",
                    "/project",
                    "--mount",
                    f"type=bind,source={ROOT},target=/project,readonly",
                    "--mount",
                    f"type=bind,source={batch.parent},target=/project/results,readonly",
                    "--mount",
                    f"type=bind,source={frozen_config.parents[1]},target=/frozen,readonly",
                    "--mount",
                    f"type=bind,source={out},target=/project/results/{config['batch']}/{name}",
                    "--tmpfs",
                    "/tmp:rw,size=2g",
                    "--env",
                    "PYTHONPATH=/project/src",
                    "--env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "--env",
                    "HOME=/tmp/lm-home",
                    "--env",
                    "XDG_CACHE_HOME=/tmp/lm-home/.cache",
                    "--env",
                    "XDG_CONFIG_HOME=/tmp/lm-home/.config",
                ]
                for path in (
                    Path(config["data_dir"]),
                    Path(config["model_config"]).parent,
                    Path(config["models"][name]["checkpoint"]).parent,
                ):
                    path = path.resolve()
                    if not path.is_relative_to(ROOT):
                        command += ["--mount", f"type=bind,source={path},target={path},readonly"]
                if args.wandb_token_file.is_file():
                    command += [
                        "--mount",
                        f"type=bind,source={args.wandb_token_file.resolve()},target=/run/secrets/wandb,readonly",
                    ]
                command += [
                    image,
                    "bash",
                    "-c",
                    'mkdir -p "$HOME"\nif [ -f /run/secrets/wandb ]; then\n'
                    'export WANDB_API_KEY="$(cat /run/secrets/wandb)"; fi\nexec "$@"',
                    "ttcd-entry",
                    "python",
                    "/frozen/scripts/run_ttcd_second_hop.py",
                    args.stage,
                    "--config",
                    config_path,
                    "--model",
                    name,
                    "--device",
                    "cuda:0",
                    "--out",
                    f"/project/results/{config['batch']}/{name}",
                    "--tracking",
                    runtime["tracking"],
                ]
                with (out / f"{args.stage}.log").open("x") as log:
                    try:
                        completed = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=runtime["timeout_hours"] * 3600,
                        )
                    except subprocess.TimeoutExpired:
                        # Only this controller's named container; preserve it and its logs.
                        subprocess.run(
                            [
                                "docker",
                                "--context",
                                "lm-memory",
                                "stop",
                                "--time",
                                "30",
                                container_name,
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=45,
                            check=False,
                        )
                        raise
                (out / f"exit-{args.stage}.json").write_text(
                    json.dumps(
                        {
                            "returncode": completed.returncode,
                            "gpu": gpu,
                            "gpu_uuid": row["uuid"],
                            "image": image,
                            "cpu_set": task_cpus,
                            "memory_gib": runtime["memory_gib"],
                            "container_user": f"{os.getuid()}:{os.getgid()}",
                        }
                    )
                    + "\n"
                )
                return completed.returncode

        def guarded_worker(name):
            try:
                return worker(name)
            except Exception:
                out = batch / name
                out.mkdir(exist_ok=True)
                (out / f"controller-failure-{args.stage}.json").write_text(
                    json.dumps({"stage": args.stage, "traceback": traceback.format_exc()}) + "\n"
                )
                return 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
            results = list(pool.map(guarded_worker, models))
        (batch / "controller-state.json").write_text(
            json.dumps(
                {
                    "state": "complete" if not any(results) else "needs_attention",
                    "stage": args.stage,
                    "exit_codes": dict(zip(models, results, strict=True)),
                },
                indent=2,
            )
            + "\n"
        )
        if any(results):
            raise SystemExit(
                "A task failed or reading calibration did not pass; inspect preserved logs"
            )


if __name__ == "__main__":
    main()
