"""Isolated LM containers, validation, tracking and final replay for one new batch."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import netrc
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path("/ossfs/workspace/research-recovery/self-decode-alignment-development-v1")
CONFIG = "configs/self-decode-alignment-development-v1.json"
DOCKER = "/workspaces/lm-memory-tools/docker/docker"
PYTHON = "/tmp/lm-perk-method-venv/bin/python"
SCRIPTS = [
    "execute_self_decode_alignment.py",
    "run_self_decode_alignment.py",
    "diagnose_self_decode_alignment.py",
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def docker(args, **kwargs):
    return subprocess.run([DOCKER, "--context", "lm-memory", *args], check=True, **kwargs)


def config():
    return read(BASE / "source" / CONFIG)


def resources():
    cfg = config()
    ids = docker(["ps", "-q"], capture_output=True, text=True).stdout.split()
    existing = (
        json.loads(docker(["inspect", *ids], capture_output=True, text=True).stdout) if ids else []
    )
    wanted = set(map(str, cfg["runtime"]["gpus"].values())) | {"6"}
    occupied_cpus = set()
    for item in existing:
        for req in item["HostConfig"].get("DeviceRequests") or []:
            if req.get("Count") == -1 or wanted.intersection(req.get("DeviceIDs") or []):
                raise RuntimeError(f"Requested GPU owned by {item['Name']}")
        for part in item["HostConfig"].get("CpusetCpus", "").split(","):
            if not part:
                continue
            ends = list(map(int, part.split("-")))
            occupied_cpus.update(range(ends[0], ends[-1] + 1))
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_rows = [list(map(int, line.split(","))) for line in result.stdout.splitlines()]
    if any(str(i) in wanted and (used > 1024 or util > 5) for i, used, util in gpu_rows):
        raise RuntimeError("Requested GPU has active memory or utilization")
    cpus = sorted(set(os.sched_getaffinity(0)) - occupied_cpus)[16:72]
    if len(cpus) != 56:
        raise RuntimeError("Need 56 free CPU cores after reserving 16")
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 7 * 48 * 1024**2:
        raise RuntimeError("Need 336 GiB available memory")
    result = {
        "created_utc": utc(),
        "gpu_rows": gpu_rows,
        "containers": [c["Name"] for c in existing],
        "cpus": {str(i): cpus[8 * i : 8 * (i + 1)] for i in range(7)},
    }
    write(BASE / f"resources-{time.time_ns()}.json", result)
    return result


def prepare():
    destination = BASE / "source"
    destination.mkdir(exist_ok=False)
    shutil.copytree(
        PROJECT / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in [
        CONFIG,
        "pyproject.toml",
        *[f"scripts/{s}" for s in SCRIPTS],
        "tests/test_self_decode_alignment.py",
        "tests/test_self_decode_alignment_runner.py",
        "docs/development-artifacts/self-decode-alignment-development-v1/protocol.md",
    ]:
        path = PROJECT / relative
        if not path.exists():
            if relative.endswith("test_self_decode_alignment_runner.py"):
                continue
            raise FileNotFoundError(path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    cfg = config()
    root = Path(cfg["input_root"])
    inputs = [root / "data" / name for name in ("complete.json", "world.npz", "panels.npz")]
    inputs += [root / "reference/gpt2/config.json"]
    inputs += [
        root / "development" / cfg["source_run"] / f"checkpoint-{n:07d}.pt"
        for n in [0, *cfg["diagnostics"]["nodes"]]
    ]
    frozen = {
        "created_utc": utc(),
        "source": {
            str(p.relative_to(destination)): sha(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
        "inputs": {str(p): sha(p) for p in inputs},
        "image": cfg["runtime"]["image"],
        "authorization": cfg["authorization"],
    }
    write(BASE / "freeze.json", frozen)
    print(json.dumps({"sources": len(frozen["source"]), "inputs": len(inputs)}), flush=True)


def verify():
    frozen = read(BASE / "freeze.json")
    for relative, value in frozen["source"].items():
        if sha(BASE / "source" / relative) != value:
            raise RuntimeError(f"Frozen source changed: {relative}")
    for path, value in frozen["inputs"].items():
        if sha(path) != value:
            raise RuntimeError(f"Frozen input changed: {path}")
    actual = json.loads(
        docker(["image", "inspect", frozen["image"]], capture_output=True, text=True).stdout
    )[0]["Id"]
    if actual != frozen["image"]:
        raise RuntimeError("Image mismatch")


def command(name, gpu, cpus, arguments):
    source = BASE / "source"
    return [
        DOCKER,
        "--context",
        "lm-memory",
        "run",
        "--name",
        name,
        "--gpus",
        f"device={gpu}",
        "--cpus",
        "8",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        "48g",
        "--memory-swap",
        "48g",
        "--shm-size",
        "8g",
        "--network",
        "none",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--mount",
        f"type=bind,source={BASE},target={BASE}",
        "--mount",
        f"type=bind,source={source},target={source},readonly",
        "--mount",
        "type=bind,source=/ossfs/workspace/llm-memory-editability,target=/ossfs/workspace/llm-memory-editability,readonly",
        "--mount",
        "type=bind,source=/ossfs/workspace/research-recovery/hf-review-20261010-6c580455,target=/ossfs/workspace/research-recovery/hf-review-20261010-6c580455,readonly",
        "-e",
        f"PYTHONPATH={source / 'src'}",
        "-e",
        "PYTHONUNBUFFERED=1",
        "--workdir",
        str(source),
        config()["runtime"]["image"],
        "python",
        *arguments,
    ]


def task(stage, arm, gpu, cpus, arguments):
    name = f"lm-selfdecode-v1-{stage}-{arm}"
    directory = BASE / "execution" / stage / arm
    directory.mkdir(parents=True, exist_ok=False)
    cmd = command(name, gpu, cpus, arguments)
    write(directory / "command.json", {"command": cmd, "created_utc": utc()})
    with (directory / "stdout.log").open("w") as out:
        process = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        write(
            directory / "process.json",
            {"pid": process.pid, "container": name, "gpu": gpu, "created_utc": utc()},
        )
        try:
            code = process.wait(timeout=config()["runtime"]["hard_timeout_hours"] * 3600)
        except subprocess.TimeoutExpired:
            write(
                directory / "timeout.json",
                {"created_utc": utc(), "reason": "predeclared 12-hour hard timeout"},
            )
            docker(["stop", "--time", "30", name], capture_output=True, text=True)
            code = process.wait(timeout=60)
    inspect = docker(["inspect", name], capture_output=True, text=True)
    item = json.loads(inspect.stdout)[0]
    receipt = {
        "exit_code": code,
        "container_state": item["State"],
        "image": item["Image"],
        "finished_utc": utc(),
    }
    write(directory / "exit.json", receipt)
    if code:
        raise RuntimeError(f"{stage}/{arm} exited {code}; log retained at {directory}")
    return receipt


def arguments(mode, arm, smoke=False):
    return [
        str(BASE / "source/scripts/run_self_decode_alignment.py"),
        mode,
        "--config",
        str(BASE / "source" / CONFIG),
        "--base",
        str(BASE / "runs" / "smoke" if smoke else BASE / "runs"),
        "--arm",
        arm,
    ]


def validate():
    verify()
    allocation = resources()["cpus"]
    cfg = config()
    task(
        "tests",
        "contracts",
        6,
        allocation["6"],
        [
            "-m",
            "pytest",
            "-q",
            "tests/test_self_decode_alignment.py",
            *(
                ["tests/test_self_decode_alignment_runner.py"]
                if (BASE / "source/tests/test_self_decode_alignment_runner.py").exists()
                else []
            ),
        ],
    )

    def short(arm):
        gpu = cfg["runtime"]["gpus"][arm]
        # The smoke worker nests outputs under <base>/smoke/<arm>.
        task("smoke", arm, gpu, allocation[str(gpu)], arguments("smoke", arm))
        return task(
            "smoke-audit", arm, gpu, allocation[str(gpu)], arguments("audit", arm, smoke=True)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = dict(zip(cfg["arms"], pool.map(short, cfg["arms"]), strict=True))
    task(
        "diagnostic",
        "fixed-weights",
        6,
        allocation["6"],
        [
            str(BASE / "source/scripts/diagnose_self_decode_alignment.py"),
            "--config",
            str(BASE / "source" / CONFIG),
            "--output",
            str(BASE / "diagnostics"),
        ],
    )
    write(
        BASE / "validation.json",
        {"passed": True, "created_utc": utc(), "smoke_replays": outcomes, "science_started": False},
    )


def tracking_environment():
    auth = netrc.netrc("/ossfs/workspace/.lm-memory-credentials/wandb.netrc").authenticators(
        "api.wandb.ai"
    )
    if auth is None:
        raise RuntimeError("W&B credentials unavailable")
    return {**os.environ, "WANDB_API_KEY": auth[2], "WANDB_SILENT": "true"}


def factorial_summary():
    """Report the fixed endpoint of every arm and every original task panel."""
    cfg = config()
    endpoints = {arm: read(BASE / "runs" / arm / "complete.json") for arm in cfg["arms"]}
    panels = endpoints["baseline"]["endpoint"]["metrics"]
    contrasts = {}
    for panel in panels:
        a = {name: row["endpoint"]["metrics"][panel]["accuracy"] for name, row in endpoints.items()}
        contrasts[panel] = {
            "accuracy": a,
            "interaction_pp": 100
            * ((a["kv_alignment"] - a["kv"]) - (a["alignment"] - a["baseline"])),
            "joint_minus_kv_pp": 100 * (a["kv_alignment"] - a["kv"]),
            "joint_minus_alignment_pp": 100 * (a["kv_alignment"] - a["alignment"]),
            "alignment_minus_self_ce_pp": 100 * (a["alignment"] - a["self_ce"]),
            "kv_alignment_minus_kv_self_ce_pp": 100 * (a["kv_alignment"] - a["kv_self_ce"]),
        }
    result = {
        "created_utc": utc(),
        "fixed_step": cfg["spec"]["steps"],
        "primary_panel": "test_oi",
        "panels": contrasts,
        "scope": "One previously observed world and initialization; no independent-world inference",
        "costs": {
            arm: {
                key: row["endpoint"][key]
                for key in (
                    "examples",
                    "effective_input_tokens",
                    "supervised_tokens",
                    "estimated_matmul_training_flops",
                    "training_seconds",
                    "wall_seconds",
                )
            }
            for arm, row in endpoints.items()
        },
        "sources": {arm: sha(BASE / "runs" / arm / "complete.json") for arm in endpoints},
    }
    write(BASE / "factorial-summary.json", result)


def track(arm):
    import wandb

    os.environ.update(tracking_environment())
    cfg = config()
    directory = BASE / "tracking" / arm
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        group=cfg["batch"],
        name=f"{cfg['batch']}-{arm}",
        config={
            "batch": cfg["batch"],
            "arm": arm,
            "spec": cfg["spec"],
            "auxiliary": cfg["arms"][arm],
            "freeze_sha256": sha(BASE / "freeze.json"),
            "physical_gpu": cfg["runtime"]["gpus"][arm],
        },
        dir=str(directory),
        mode="online",
    )
    write(directory / "ready.json", {"id": run.id, "url": run.url, "created_utc": utc()})
    seen = set()
    progress_count = 0
    while True:
        path = BASE / "runs" / arm
        learning = path / "learning.json"
        if learning.exists():
            for row in read(learning):
                if row["step"] not in seen:
                    run.log(
                        {
                            "evaluation_step": row["step"],
                            **{k: v for k, v in row.items() if isinstance(v, int | float)},
                            **{
                                f"{panel}/{key}": value
                                for panel, values in row.get("metrics", {}).items()
                                for key, value in values.items()
                                if isinstance(value, int | float)
                            },
                        }
                    )
                    seen.add(row["step"])
        progress = path / "progress.jsonl"
        if progress.exists():
            lines = progress.read_text().splitlines()
            for line in lines[progress_count:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                run.log({f"train/{k}": v for k, v in row.items() if isinstance(v, int | float)})
                progress_count += 1
        telemetry = subprocess.run(
            [
                "nvidia-smi",
                f"--id={cfg['runtime']['gpus'][arm]}",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if telemetry.returncode == 0:
            utilization, memory, power = map(float, telemetry.stdout.strip().split(","))
            run.log(
                {
                    "gpu/utilization_percent": utilization,
                    "gpu/memory_mib": memory,
                    "gpu/power_watts": power,
                }
            )
        state = BASE / "task-state" / f"{arm}.json"
        if state.exists() and read(state)["state"] in ("complete", "failed"):
            final = read(state)
            run.summary.update(
                {
                    "execution_state": final["state"],
                    "final_step": cfg["spec"]["steps"]
                    if final["state"] == "complete"
                    else max(seen, default=0),
                    "independent_audit_passed": final["state"] == "complete",
                }
            )
            run.finish(exit_code=0 if final["state"] == "complete" else 1)
            write(directory / "finished.json", final)
            break
        time.sleep(15)


def controller():
    lock = (BASE / "execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    if not read(BASE / "validation.json")["passed"]:
        raise RuntimeError("Validation not passed")
    cfg = config()
    allocation = resources()["cpus"]
    write(
        BASE / "controller-state.json",
        {
            "state": "running",
            "created_utc": utc(),
            "arms": list(cfg["arms"]),
            "automatic_retry": False,
        },
    )

    def one(arm):
        gpu = cfg["runtime"]["gpus"][arm]
        tracking = BASE / "tracking" / arm
        tracking.mkdir(parents=True, exist_ok=False)
        with (tracking / "stdout.log").open("w") as log:
            tracker = subprocess.Popen(
                [
                    PYTHON,
                    str(BASE / "source/scripts/execute_self_decode_alignment.py"),
                    "track",
                    "--arm",
                    arm,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=tracking_environment(),
            )
        write(tracking / "process.json", {"pid": tracker.pid, "created_utc": utc()})
        try:
            deadline = time.monotonic() + 120
            while not (tracking / "ready.json").exists():
                if tracker.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"W&B tracker not ready for {arm}")
                time.sleep(2)
            task("train", arm, gpu, allocation[str(gpu)], arguments("train", arm))
            task("audit", arm, gpu, allocation[str(gpu)], arguments("audit", arm))
            result = {"state": "complete", "created_utc": utc()}
        except Exception as exc:
            result = {"state": "failed", "error": str(exc), "created_utc": utc()}
        write(BASE / "task-state" / f"{arm}.json", result)
        try:
            tracker.wait(timeout=120)
        except subprocess.TimeoutExpired:
            result["tracking"] = "still_running"
        return arm, result

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(pool.map(one, cfg["arms"]))
    state = (
        "scientific_runs_and_audits_complete"
        if all(v["state"] == "complete" for v in results.values())
        else "needs_attention"
    )
    write(
        BASE / "controller-state.json",
        {"state": state, "arms": results, "created_utc": utc(), "automatic_retry": False},
    )
    if state == "scientific_runs_and_audits_complete":
        import wandb

        factorial_summary()
        os.environ.update(tracking_environment())
        api = wandb.Api()
        cloud = {}
        for arm in cfg["arms"]:
            info = read(BASE / "tracking" / arm / "ready.json")
            remote = api.run(f"zhusq20/llm-memory-editability/{info['id']}")
            cloud[arm] = {
                "url": info["url"],
                "state": remote.state,
                "final_step": remote.summary.get("final_step"),
                "audit": remote.summary.get("independent_audit_passed"),
            }
        passed = all(
            v["state"] == "finished" and v["final_step"] == 300000 and v["audit"] is True
            for v in cloud.values()
        )
        write(
            BASE / "cloud-verification.json",
            {
                "passed": passed,
                "scope": "run state/final step/audit summary; not every history point",
                "runs": cloud,
                "created_utc": utc(),
            },
        )
        write(
            BASE / "completion-manifest.json",
            {
                "science_complete": True,
                "audits_passed": True,
                "cloud_verified": passed,
                "optimization_updates": 1800000,
                "created_utc": utc(),
            },
        )


def launch():
    marker = BASE / "controller-process.json"
    if marker.exists():
        raise RuntimeError("Existing submission retained; never launch twice")
    if not read(BASE / "validation.json")["passed"]:
        raise RuntimeError("Validation incomplete")
    cmd = [PYTHON, str(BASE / "source/scripts/execute_self_decode_alignment.py"), "controller"]
    with (BASE / "controller.log").open("x") as log:
        process = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    write(
        marker,
        {
            "pid": process.pid,
            "command": cmd,
            "created_utc": utc(),
            "project": "https://wandb.ai/zhusq20/llm-memory-editability",
        },
    )
    print(json.dumps(read(marker)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "validate", "launch", "controller", "track"))
    parser.add_argument("--arm")
    args = parser.parse_args()
    if args.mode == "track":
        track(args.arm)
    else:
        globals()[args.mode]()
