"""Launcher for write-mode-readout-v1 (zero training, authorized 2026-10-20; immediate start
without freeze/hash/CUDA preflight). One GPU container per model (read-interface image), then one
CPU summary/figure container (plotting image).

Usage: launch_write_mode_readout.py gpu MODEL GPU_INDEX   |   launch_write_mode_readout.py summary
"""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/disk2_from_server2/siqizhu4/llm-memory-editability/scripts")
from launch_paper_update_gate import credential_archive  # noqa: E402

REPO = "/mnt/disk2_from_server2/siqizhu4/llm-memory-editability"
WS = Path("/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability")
KRID = WS / "knowledge-read-interface-development-v1"
HF = KRID / "hf-cache"
BATCH = WS / "write-mode-readout-v1"
OUT = BATCH / "execution"
IMAGE = "sha256:8eb44ee1611fe88f5a7113eb271cae9a0894fe0e95c988360b994f79e4017eba"  # read-interface
PLOT_IMAGE = "lm-memory-gated-depth:20261017-plotting-v1"
TOKEN = "/home/siqizhu4/.cache/lm-memory-ce-ablation/wandb-token"
DOCKER = ["docker", "--context", "lm-memory"]
CPUS = {"qwen3-1.7b": "56-63", "qwen3.5-2b": "64-71", None: "72-79"}

STAGE = sys.argv[1]
MODEL = sys.argv[2] if STAGE == "gpu" else None
GPU = int(sys.argv[3]) if STAGE == "gpu" else None
NAME = f"wmr-{MODEL or 'summary'}"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))


def gpu_row(index):
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()  # fmt: skip
    for r in rows:
        i, uuid, used = [x.strip() for x in r.split(",")]
        if int(i) == index:
            return uuid, int(used.split()[0]), rows
    raise SystemExit(f"no GPU {index}")


BOOT = (
    "import os,sys;os.umask(0o022);from pathlib import Path;"
    "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
    "os.execvp(sys.argv[1],sys.argv[1:])"
)
ENV = [
    "PYTHONPATH=/project/src:/project/scripts", "PYTHONDONTWRITEBYTECODE=1", "OMP_NUM_THREADS=8",
    "HOME=/tmp", "XDG_CACHE_HOME=/tmp/cache", "MPLCONFIGDIR=/tmp/mpl",
    "WANDB_CACHE_DIR=/tmp/wandb-cache", "WANDB_CONSOLE=off", "WANDB_SILENT=true",
    "WANDB_DISABLE_CODE=true", "HF_HOME=/hf", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
]  # fmt: skip


def main():
    BATCH.mkdir(parents=True, exist_ok=True)
    os.chmod(BATCH, 0o777)
    image = IMAGE if STAGE == "gpu" else PLOT_IMAGE
    cmd = ["python", "-u", "scripts/run_write_mode_readout.py"]
    cmd += ["--krid", "/krid", "--out", "/batch"]
    cmd += ["--model", MODEL] if STAGE == "gpu" else ["--summary"]
    folder = OUT / NAME
    write(folder / "controller-process.json", {
        "pid": os.getpid(), "utc": utc(), "stage": STAGE, "model": MODEL, "gpu": GPU,
        "image": image, "policy": "user authorized write-mode-readout-v1 (zero training) on "
        "2026-10-20; immediate start without freeze/hash/CUDA preflight",
    })  # fmt: skip
    env = [x for e in ENV + [f"WANDB_DOCKER={image}"] for x in ("--env", e)]
    mounts = [
        "--mount", f"type=bind,source={REPO},target=/project,readonly",
        "--mount", f"type=bind,source={BATCH},target=/batch",
        "--mount", f"type=bind,source={KRID},target=/krid,readonly",
        "--mount", f"type=bind,source={HF},target=/hf,readonly",
    ]  # fmt: skip
    uuid = gpu_row(GPU)[0] if GPU is not None else None
    args = [
        *DOCKER, "create", "--name", NAME, "--user", "1015:1016", "--cpuset-cpus", CPUS[MODEL],
        "--cpus", "8", "--memory", "96g", "--memory-swap", "96g", "--shm-size", "8g",
        "--tmpfs", "/tmp:rw,size=16g", "--network", "host", "--workdir", "/project",
        *(["--gpus", f"device={uuid}"] if uuid else []),
        *env, *mounts, image, "python", "-c", BOOT, *cmd,
    ]  # fmt: skip
    write(folder / "command.json", {"argv": args, "utc": utc()})
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([*DOCKER, "cp", "--archive", "-", f"{NAME}:/"], input=credential_archive(TOKEN),
                   check=True, capture_output=True, timeout=120)  # fmt: skip
    state = {"container": NAME, "image": image, "gpu": GPU, "cpus": CPUS[MODEL]}
    if GPU is not None:
        _, used, rows = gpu_row(GPU)
        write(folder / "nvidia-smi-before-start.json", {"utc": utc(), "rows": rows})
        if used > 1024:
            write(folder / "controller-state.json", {**state, "state": "refused", "used": used})
            raise SystemExit(f"GPU {GPU} busy ({used} MiB)")
    subprocess.run([*DOCKER, "start", NAME], check=True, capture_output=True, timeout=120)
    write(folder / "controller-state.json", {**state, "state": "running", "started_utc": utc(),
          "source": "repository working tree, read-only mount (no frozen snapshot)"})  # fmt: skip
    with (folder / "stdout.log").open("a") as log:
        follower = subprocess.Popen([*DOCKER, "logs", "--follow", NAME], stdout=log,
                                    stderr=subprocess.STDOUT)  # fmt: skip
        code = int(subprocess.run([*DOCKER, "wait", NAME], capture_output=True,
                                  text=True).stdout.strip())  # fmt: skip
        time.sleep(5)
        follower.terminate()
    write(folder / "controller-state.json", {**state, "state": "complete" if code == 0 else
          "failed", "exit_code": code, "finished_utc": utc()})  # fmt: skip


if __name__ == "__main__":
    main()
