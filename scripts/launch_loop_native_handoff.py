#!/usr/bin/env python3
"""Run loop-native handoff stages, one LM Docker container per arm on its own idle GPU.

Follows the rule-gate launch: the repository working tree is mounted read-only, the
persistent workspace read-only for inputs, and only the batch output directory is
writable. The W&B credential is copied into a container only for the train stage.
Without ``--execute`` the planned containers are printed and nothing starts. Stages
run in order per arm; a failed stage stops that arm and is never retried.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from launch_gate_extension import snapshot  # noqa: E402  (also blocks created/paused reservations)
from launch_paper_update_gate import credential_archive  # noqa: E402

CONFIG = "configs/loop-native-handoff-development-v1.json"
DOCKER = ["docker", "--context", "lm-memory"]
STAGES = ("smoke", "smoke-grid", "train", "grid")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1) + "\n")


def admit(gpus, runtime, allow_reserved=False):
    """Return GPU UUIDs and any container reservations that the user chose to override.

    Active CUDA processes, used memory or utilization always refuse. Container
    reservations (created, paused or idle running containers that request the GPU)
    refuse unless the user designated these GPUs (``allow_reserved``); they are then
    recorded with the submission.
    """
    rows, blocked = snapshot()
    overridden = {}
    for gpu in gpus:
        row = rows.get(gpu)
        blockers = blocked.get(gpu, [])
        if (
            row is None
            or any(b.startswith("active CUDA process") for b in blockers)
            or row["used"] > runtime["maximum_idle_gpu_used_mib"]
            or row["utilization"] > 5
            or row["free"] < runtime["minimum_free_gpu_mib"]
        ):
            raise ValueError(f"GPU {gpu} is in use; nothing was started")
        if blockers and not allow_reserved:
            raise ValueError(f"GPU {gpu} is reserved by {blockers}; nothing was started")
        if blockers:
            overridden[gpu] = blockers
    return {gpu: rows[gpu]["uuid"] for gpu in gpus}, overridden


SOURCES = (
    CONFIG,
    "scripts/launch_loop_native_handoff.py",
    "scripts/run_loop_native_handoff.py",
    "scripts/run_loop_hop.py",
    "scripts/run_supervised_alignment.py",
    "src/llm_memory_editability/loop_native_handoff.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "src/llm_memory_editability/loop_hop_data.py",
    "tests/test_loop_native_handoff.py",
)


def source_hashes():
    import hashlib

    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}


def command(config, name, stage, uuid, cpus, output):
    runtime = config["runtime"]
    persistent = runtime["persistent_root"]
    boot = "import os,sys;os.umask(0o022);"
    if stage == "train":
        boot += (
            "from pathlib import Path;"
            "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
        )
    boot += "os.execvp(sys.argv[1],sys.argv[1:])"
    label = f"{config['batch']}-{name}-{stage}"
    args = [
        *DOCKER,
        "create",
        "--name",
        label,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--cpuset-cpus",
        cpus,
        "--cpus",
        "2",
        "--memory",
        f"{runtime['memory_gib']}g",
        "--memory-swap",
        f"{runtime['memory_gib']}g",
        "--shm-size",
        "2g",
        "--tmpfs",
        "/tmp:rw,size=4g",
        "--network",
        "host",
        "--workdir",
        "/project",
        "--gpus",
        f"device={uuid}",
    ]
    for key, value in (
        ("PYTHONPATH", "/project/src:/project/scripts"),
        ("PYTHONDONTWRITEBYTECODE", "1"),
        ("OMP_NUM_THREADS", "2"),
        ("MKL_NUM_THREADS", "2"),
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("XDG_CACHE_HOME", "/tmp/cache"),
        ("WANDB_CACHE_DIR", "/tmp/wandb-cache"),
        ("WANDB_CONSOLE", "off"),
        ("WANDB_SILENT", "true"),
        ("WANDB_DISABLE_CODE", "true"),
        ("WANDB_DOCKER", runtime["image"]),
    ):
        args += ["--env", f"{key}={value}"]
    args += [
        "--mount",
        f"type=bind,source={ROOT},target=/project,readonly",
        "--mount",
        f"type=bind,source={persistent},target={persistent},readonly",
        "--mount",
        f"type=bind,source={output},target=/output",
        runtime["image"],
        "python",
        "-c",
        boot,
        "python",
        "-u",
        "scripts/run_loop_native_handoff.py",
        stage,
        "--config",
        CONFIG,
        "--base",
        "/output",
        "--arm",
        name,
    ]
    return label, args


def run_stage(config, name, stage, gpu, uuid, cpus, output, token):
    label, args = command(config, name, stage, uuid, cpus, output)
    folder = output / "execution" / name / stage
    if folder.exists():
        raise FileExistsError(f"{folder} exists; stages are never retried")
    write(folder / "command.json", {"argv": args, "gpu": gpu, "utc": utc()})
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)
    if stage == "train":
        subprocess.run(
            [*DOCKER, "cp", "--archive", "-", f"{label}:/"],
            input=credential_archive(token),
            check=True,
            capture_output=True,
            timeout=120,
        )
    subprocess.run([*DOCKER, "start", label], check=True, capture_output=True, timeout=120)
    write(folder / "launch.json", {"container": label, "gpu": gpu, "gpu_uuid": uuid, "utc": utc()})
    with (folder / "stdout.log").open("w") as log:
        follower = subprocess.Popen(
            [*DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
        )
        waited = subprocess.run([*DOCKER, "wait", label], capture_output=True, text=True)
        try:
            follower.wait(timeout=30)
        except subprocess.TimeoutExpired:
            follower.terminate()
    code = int(waited.stdout.strip())
    write(folder / "exit.json", {"code": code, "utc": utc()})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--cpus", nargs="+", required=True, help="cpuset per arm, e.g. 12,13")
    parser.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--wandb-token-file", type=Path)
    parser.add_argument(
        "--allow-reserved",
        action="store_true",
        help="GPUs designated by the user: proceed past container reservations (recorded); "
        "active CUDA processes still refuse",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text())
    if not len(args.arms) == len(args.gpus) == len(args.cpus):
        raise ValueError("One GPU and one cpuset per arm")
    if len(set(args.gpus)) != len(args.gpus) or set(args.arms) - set(config["arms"]):
        raise ValueError("Distinct GPUs and configured arms are required")
    if "train" in args.stages:
        if config["status"].startswith("prepared"):
            raise PermissionError("Scientific training is not authorized in the configuration")
        credential_archive(args.wandb_token_file)
    output = args.output.resolve()
    plan = {
        "batch": config["batch"],
        "arms": dict(zip(args.arms, zip(args.gpus, args.cpus, strict=True), strict=True)),
        "stages": args.stages,
        "output": str(output),
    }
    if not args.execute:
        print(json.dumps(plan, indent=1))
        return
    output.mkdir(parents=True, exist_ok=True)
    if "train" in args.stages and (output / "controller-process.json").exists():
        raise FileExistsError("This batch was already submitted; no duplicate submission")
    uuids, overridden = admit(args.gpus, config["runtime"], args.allow_reserved)
    state = {"batch": config["batch"], "arms": {name: {"state": "starting"} for name in args.arms}}
    lock = threading.Lock()

    def update(name, **values):
        with lock:
            state["arms"][name].update(values)
            state["updated_utc"] = utc()
            write(output / "controller-state.json", state)

    def worker(name, gpu, cpus):
        try:
            for stage in args.stages:
                update(name, state="running", stage=stage, gpu=gpu)
                code = run_stage(
                    config, name, stage, gpu, uuids[gpu], cpus, output, args.wandb_token_file
                )
                if code:
                    update(name, state="failed", stage=stage, exit_code=code)
                    return
            update(name, state="complete", stages=args.stages)
        except Exception as error:  # noqa: BLE001 - recorded for the operator
            update(name, state="needs_attention", error=repr(error))

    process = {
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "utc": utc(),
        "plan": plan,
        "gpu_uuids": uuids,
        "authorization": config.get("authorization"),
        "overridden_reservations": overridden,
        "source": "repository working tree mounted read-only; sha256 at submission below",
        "source_sha256": source_hashes(),
    }
    write(output / "execution" / f"controller-process-{'-'.join(args.stages)}.json", process)
    if "train" in args.stages:
        write(output / "controller-process.json", process)
    threads = [
        threading.Thread(target=worker, args=(name, gpu, cpus))
        for name, gpu, cpus in zip(args.arms, args.gpus, args.cpus, strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    write(output / "execution" / f"controller-exit-{'-'.join(args.stages)}.json", state)


if __name__ == "__main__":
    os.umask(0o022)
    main()
