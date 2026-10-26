"""Launcher for the two zero-training analyses authorized on 2026-10-20 (immediate start, no
freeze/hash/CUDA preflight): knowledge-reasoning-map-v1 (Qwen) and bios-write-address-v1 (bioS).
Each runs one GPU container (read-interface image) and then one CPU figure container (plotting
image). Usage: launch_zero_training_maps.py {krmap|bwa} GPU_INDEX [SUFFIX]"""

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/mnt/disk2_from_server2/siqizhu4/llm-memory-editability/scripts")
sys.path.insert(0, "/mnt/disk2_from_server2/siqizhu4/llm-memory-editability/src")
from launch_paper_update_gate import credential_archive  # noqa: E402

REPO = "/mnt/disk2_from_server2/siqizhu4/llm-memory-editability"
WS = Path("/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability")
HF = WS / "knowledge-read-interface-development-v1" / "hf-cache"
# read-interface:20261019
IMAGE = "sha256:8eb44ee1611fe88f5a7113eb271cae9a0894fe0e95c988360b994f79e4017eba"
PLOT_IMAGE = "lm-memory-gated-depth:20261017-plotting-v1"
TOKEN = "/home/siqizhu4/.cache/lm-memory-ce-ablation/wandb-token"
DOCKER = ["docker", "--context", "lm-memory"]
JOBS = {
    "krmap": dict(
        batch="knowledge-reasoning-map-v1",
        gpu_cmd=["python", "-u", "scripts/run_function_map.py", "--out", "/batch"],
        fig_cmd=["python", "-u", "scripts/run_function_map.py", "--out", "/batch", "--figures"],
        mounts=[f"type=bind,source={HF},target=/hf,readonly"],
        cpus="48-55",
    ),
    "krmap-fig": dict(  # figure stage only, after adding a figure
        batch="knowledge-reasoning-map-v1",
        gpu_cmd=None,
        fig_cmd=["python", "-u", "scripts/run_function_map.py", "--out", "/batch", "--figures"],
        mounts=[],
        cpus="24-31",
    ),
    "bwa-fig": dict(  # figure stage only, after fixing the figure code
        batch="bios-write-address-v1",
        gpu_cmd=None,
        fig_cmd=["python", "-u", "scripts/run_bios_address.py", "--old", "/old", "--out", "/batch",
                 "--figures"],
        mounts=[f"type=bind,source={WS / 'bios-write-address-v1' / 'inputs'},target=/old,readonly"],
        cpus="24-31",
    ),
    "bwa2": dict(  # helper: world 142702 only, no figures; the full bwa run reuses its files
        batch="bios-write-address-v1",
        gpu_cmd=["python", "-u", "scripts/run_bios_address.py", "--old", "/old", "--out", "/batch",
                 "--worlds", "142702"],
        fig_cmd=None,
        mounts=[f"type=bind,source={WS / 'bios-write-address-v1' / 'inputs'},target=/old,readonly"],
        cpus="32-39",
    ),
    "bwa": dict(
        batch="bios-write-address-v1",
        gpu_cmd=["python", "-u", "scripts/run_bios_address.py", "--old", "/old", "--out", "/batch"],
        fig_cmd=[
            "python",
            "-u",
            "scripts/run_bios_address.py",
            "--old",
            "/old",
            "--out",
            "/batch",
            "--figures",
        ],  # fmt: skip
        mounts=[f"type=bind,source={WS / 'bios-write-address-v1' / 'inputs'},target=/old,readonly"],
        cpus="40-47",
    ),
}
JOB = sys.argv[1]
GPU = int(sys.argv[2])
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""
SPEC = JOBS[JOB]
BATCH = WS / SPEC["batch"]
OUT = BATCH / "execution"
state = {"batch": SPEC["batch"], "jobs": {}}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1))


def update(name, **v):
    state["jobs"].setdefault(name, {}).update(v)
    state["updated_utc"] = utc()
    write(OUT / f"controller-state{SUFFIX}.json", state)


def gpu_uuid(index):
    rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
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
COMMON_ENV = [
    "PYTHONPATH=/project/src:/project/scripts",
    "PYTHONDONTWRITEBYTECODE=1",
    "OMP_NUM_THREADS=8",
    "HOME=/tmp",
    "XDG_CACHE_HOME=/tmp/cache",
    "MPLCONFIGDIR=/tmp/mpl",
    "WANDB_CACHE_DIR=/tmp/wandb-cache",
    "WANDB_CONSOLE=off",
    "WANDB_SILENT=true",
    "WANDB_DISABLE_CODE=true",
    "HF_HOME=/hf",
    "HF_HUB_OFFLINE=1",
    "TRANSFORMERS_OFFLINE=1",
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
]


def run(name, image, cmd, gpu):
    label = f"{JOB}-{name}{SUFFIX}"
    env = [x for e in COMMON_ENV + [f"WANDB_DOCKER={image}"] for x in ("--env", e)]
    mounts = ["--mount", f"type=bind,source={REPO},target=/project,readonly",
              "--mount", f"type=bind,source={BATCH},target=/batch"]  # fmt: skip
    for m in SPEC["mounts"]:
        mounts += ["--mount", m]
    uuid = gpu_uuid(GPU)[0] if gpu else None
    args = [
        *DOCKER, "create", "--name", label, "--user", "1015:1016", "--cpuset-cpus", SPEC["cpus"],
        "--cpus", "8", "--memory", "96g", "--memory-swap", "96g", "--shm-size", "8g",
        "--tmpfs", "/tmp:rw,size=16g", "--network", "host", "--workdir", "/project",
        *(["--gpus", f"device={uuid}"] if gpu else []),
        *env, *mounts, image, "python", "-c", BOOT, *cmd,
    ]  # fmt: skip
    folder = OUT / f"{name}{SUFFIX}"
    write(folder / "command.json", {"argv": args, "utc": utc()})
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=120)
    subprocess.run([*DOCKER, "cp", "--archive", "-", f"{label}:/"], input=credential_archive(TOKEN),
                   check=True, capture_output=True, timeout=120)  # fmt: skip
    if gpu:
        _, used, rows = gpu_uuid(GPU)
        write(folder / "nvidia-smi-before-start.json", {"utc": utc(), "rows": rows})
        if used > 1024:
            update(name, state="refused", reason=f"GPU {GPU} busy ({used} MiB) at start")
            raise SystemExit(f"GPU {GPU} busy")
    subprocess.run([*DOCKER, "start", label], check=True, capture_output=True, timeout=120)
    update(
        name,
        state="running",
        container=label,
        image=image,
        gpu=GPU if gpu else None,
        cpus=SPEC["cpus"],
        started_utc=utc(),
        source="repository working tree, read-only mount (no frozen snapshot, user request)",
    )
    with (folder / "stdout.log").open("a") as log:
        follower = subprocess.Popen([*DOCKER, "logs", "--follow", label], stdout=log,
                                    stderr=subprocess.STDOUT)  # fmt: skip
        code = int(
            subprocess.run([*DOCKER, "wait", label], capture_output=True, text=True).stdout.strip()
        )
        time.sleep(5)
        follower.terminate()
    update(name, state="complete" if code == 0 else "failed", exit_code=code, finished_utc=utc())
    return code


BATCH.mkdir(parents=True, exist_ok=True)
os.chmod(BATCH, 0o777)
write(OUT / f"controller-process{SUFFIX}.json", {
    "pid": os.getpid(), "utc": utc(), "job": JOB, "gpu": GPU,
    "policy": f"user authorized {SPEC['batch']} (zero training) on 2026-10-20; immediate start "
              "without freeze/hash/CUDA preflight",
    "image": IMAGE, "plot_image": PLOT_IMAGE,
})  # fmt: skip
code = run("gpu", IMAGE, SPEC["gpu_cmd"], True) if SPEC["gpu_cmd"] else 0
if code == 0 and SPEC["fig_cmd"]:
    run("fig", PLOT_IMAGE, SPEC["fig_cmd"], False)
write(OUT / f"controller-exit{SUFFIX}.json", {"utc": utc(), "jobs": state["jobs"]})
