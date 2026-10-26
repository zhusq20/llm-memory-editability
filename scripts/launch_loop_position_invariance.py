#!/usr/bin/env python3
"""Run position-invariance arms, one LM Docker container per arm on its own GPU.

Same container policy as scripts/launch_loop_native_handoff.py (repository read-only,
persistent workspace read-only, only the batch output writable, W&B credential copied
in for the train stage only). Each arm waits until its GPU has no active CUDA process
and little memory in use, so an arm can queue behind work that is finishing. Stages
run in order per arm; a failed stage stops that arm and is never retried. Without
``--execute`` the plan is printed and nothing starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from launch_loop_native_handoff import admit, utc, write  # noqa: E402
from launch_paper_update_gate import credential_archive  # noqa: E402

CONFIG = "configs/loop-position-invariance-development-v1.json"
RUNNER = "scripts/run_loop_position_invariance.py"
DOCKER = ["docker", "--context", "lm-memory"]
STAGES = ("smoke", "smoke-grid", "train", "grid")
SOURCES = (
    CONFIG,
    RUNNER,
    "scripts/launch_loop_position_invariance.py",
    "scripts/run_loop_native_handoff.py",
    "scripts/run_loop_hop.py",
    "src/llm_memory_editability/loop_position_invariance.py",
    "src/llm_memory_editability/loop_native_handoff.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/loop_hop_data.py",
    "tests/test_loop_position_invariance.py",
)


def source_hashes():
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
    args = [*DOCKER, "create", "--name", label, "--user", f"{os.getuid()}:{os.getgid()}"]
    args += ["--cpuset-cpus", cpus, "--cpus", "2"]
    args += ["--memory", f"{runtime['memory_gib']}g", "--memory-swap", f"{runtime['memory_gib']}g"]
    args += ["--shm-size", "2g", "--tmpfs", "/tmp:rw,size=4g", "--network", "host"]
    args += ["--workdir", "/project", "--gpus", f"device={uuid}"]
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
        RUNNER,
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


def wait_for_gpu(gpu, runtime, allow_reserved, update, limit_hours=48):
    started = time.monotonic()
    while True:
        try:
            uuids, overridden = admit([gpu], runtime, allow_reserved)
            return uuids[gpu], overridden.get(gpu, [])
        except ValueError as busy:
            if time.monotonic() - started > limit_hours * 3600:
                raise TimeoutError(f"GPU {gpu} did not become free") from busy
            update(state="waiting_for_gpu", gpu=gpu, reason=str(busy))
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--cpus", nargs="+", required=True, help="cpuset per arm, e.g. 72,73")
    parser.add_argument("--stages", nargs="+", choices=STAGES, required=True)
    parser.add_argument("--wandb-token-file", type=Path)
    parser.add_argument("--allow-reserved", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text())
    if not len(args.arms) == len(args.gpus) == len(args.cpus):
        raise ValueError("One GPU and one cpuset per arm")
    if len(set(args.gpus)) != len(args.gpus) or set(args.arms) - set(config["arms"]):
        raise ValueError("Distinct GPUs and configured arms are required")
    if "train" in args.stages:
        if not config["status"].startswith("authorized"):
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
    state = {"batch": config["batch"], "arms": {name: {"state": "starting"} for name in args.arms}}
    lock = threading.Lock()

    def updater(name):
        def update(**values):
            with lock:
                state["arms"][name].update(values)
                state["updated_utc"] = utc()
                write(output / "controller-state.json", state)

        return update

    process = {
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "utc": utc(),
        "plan": plan,
        "authorization": config.get("authorization"),
        "allow_reserved": args.allow_reserved,
        "source": "repository working tree mounted read-only; sha256 at submission below",
        "source_sha256": source_hashes(),
    }
    write(output / "execution" / f"controller-process-{'-'.join(args.stages)}.json", process)
    if "train" in args.stages:
        write(output / "controller-process.json", process)

    def worker(name, gpu, cpus):
        update = updater(name)
        try:
            uuid, overridden = wait_for_gpu(gpu, config["runtime"], args.allow_reserved, update)
            update(state="admitted", gpu=gpu, gpu_uuid=uuid, overridden_reservations=overridden)
            for stage in args.stages:
                update(state="running", stage=stage)
                code = run_stage(
                    config, name, stage, gpu, uuid, cpus, output, args.wandb_token_file
                )
                if code:
                    update(state="failed", stage=stage, exit_code=code)
                    return
            update(state="complete", stages=args.stages)
        except Exception as error:  # noqa: BLE001 - recorded for the operator
            update(state="needs_attention", error=repr(error))

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
