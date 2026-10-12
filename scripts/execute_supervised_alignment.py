"""Validate and schedule six isolated additions without preempting prior audits."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import execute_self_decode_alignment as shared

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path("/ossfs/workspace/research-recovery/supervised-alignment-development-v1")
CONFIG = "configs/supervised-alignment-development-v1.json"
DOCKER = shared.DOCKER
PYTHON = shared.PYTHON
read, write, sha, utc, docker = shared.read, shared.write, shared.sha, shared.utc, shared.docker
TESTS = [
    "tests/test_supervised_alignment.py",
    "tests/test_supervised_alignment_runner.py",
    "tests/test_supervised_alignment_controller.py",
    "tests/test_self_decode_alignment.py",
    "tests/test_self_decode_alignment_runner.py",
]


def config():
    return read(BASE / "source" / CONFIG)


def prepare():
    destination = BASE / "source"
    destination.mkdir(exist_ok=False)
    shutil.copytree(
        PROJECT / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for relative in [
        CONFIG,
        "pyproject.toml",
        "scripts/execute_self_decode_alignment.py",
        "scripts/run_self_decode_alignment.py",
        "scripts/run_supervised_alignment.py",
        "scripts/execute_supervised_alignment.py",
        "docs/development-artifacts/supervised-alignment-development-v1/protocol.md",
        *TESTS,
    ]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, target)
    cfg = config()
    old = Path(cfg["baseline_batch"])
    previous = read(old / "source/configs/self-decode-alignment-development-v1.json")
    for key, value in cfg["spec"].items():
        if key != "name" and previous["spec"][key] != value:
            raise ValueError(f"Reused baseline recipe differs: {key}")
    for key in ("input_root", "source_run", "dropout_seed", "evaluation_nodes"):
        if cfg[key] != previous[key]:
            raise ValueError(f"Reused baseline input differs: {key}")
    if cfg["runtime"]["image"] != previous["runtime"]["image"]:
        raise ValueError("Reused baseline image differs")
    root = Path(cfg["input_root"])
    inputs = [root / "data" / n for n in ("complete.json", "world.npz", "panels.npz")]
    inputs += [root / "reference/gpt2/config.json"]
    inputs += [root / "development" / cfg["source_run"] / "checkpoint-0000000.pt"]
    inputs += [
        old / "freeze.json",
        old / "source/configs/self-decode-alignment-development-v1.json",
    ]
    old_frozen = read(old / "freeze.json")
    shared_dependencies = {}
    for relative, expected in old_frozen["source"].items():
        if sha(old / "source" / relative) != expected:
            raise RuntimeError(f"Prior source changed: {relative}")
        if relative.startswith("src/") and relative.endswith(".py"):
            if sha(destination / relative) != expected:
                raise RuntimeError(f"Reused baseline dependency differs: {relative}")
            shared_dependencies[relative] = expected
    input_hashes = {str(p): sha(p) for p in inputs}
    for path, observed in input_hashes.items():
        if path in old_frozen["inputs"] and observed != old_frozen["inputs"][path]:
            raise RuntimeError(f"Reused baseline data/checkpoint changed: {path}")
    for name in ("discoloop-readable.txt", "discoloop-2607.00341v2.html", "source-manifest.json"):
        target = BASE / "sources" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(old / "sources" / name, target)
    frozen = {
        "created_utc": utc(),
        "source": {
            str(p.relative_to(destination)): sha(p)
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
        "inputs": input_hashes,
        "literature": {p.name: sha(p) for p in (BASE / "sources").iterdir()},
        "image": cfg["runtime"]["image"],
        "authorization": cfg["authorization"],
        "new_scientific_updates": 1800000,
        "baseline_shared_dependencies": shared_dependencies,
    }
    write(BASE / "freeze.json", frozen)
    print(json.dumps({"sources": len(frozen["source"]), "inputs": len(inputs)}), flush=True)


def verify():
    frozen = read(BASE / "freeze.json")
    for relative, expected in frozen["source"].items():
        if sha(BASE / "source" / relative) != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
    for path, expected in frozen["inputs"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Frozen input changed: {path}")
    old = Path(config()["baseline_batch"])
    for relative, expected in read(old / "freeze.json")["source"].items():
        if sha(old / "source" / relative) != expected:
            raise RuntimeError(f"Prior source changed: {relative}")
    observed = json.loads(
        docker(["image", "inspect", frozen["image"]], capture_output=True, text=True).stdout
    )[0]["Id"]
    if observed != frozen["image"]:
        raise RuntimeError("Fixed image mismatch")


def cpuset(value):
    result = set()
    for part in value.split(","):
        if part:
            ends = [int(n) for n in part.split("-")]
            result.update(range(ends[0], ends[-1] + 1))
    return result


def predecessor_released(gpu, cfg):
    """A gap between old training and old audit never releases its reservation."""
    previous = Path(cfg["baseline_batch"])
    mapping = read(previous / "source/configs/self-decode-alignment-development-v1.json")
    for arm, assigned in mapping["runtime"]["gpus"].items():
        if int(assigned) == gpu:
            marker = previous / "task-state" / f"{arm}.json"
            return marker.exists() and read(marker)["state"] in ("complete", "failed")
    return True


def snapshot():
    ids = docker(["ps", "-q"], capture_output=True, text=True).stdout.split()
    containers = (
        json.loads(docker(["inspect", *ids], capture_output=True, text=True).stdout) if ids else []
    )
    rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    gpus = {}
    for row in rows:
        index, uuid, memory, utilization = [part.strip() for part in row.split(",")]
        gpus[int(index)] = {"uuid": uuid, "memory": int(memory), "utilization": int(utilization)}
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    process_uuids = {line.split(",")[0].strip() for line in processes.splitlines() if line.strip()}
    occupied_cpus, occupied_gpus = set(), set()
    for item in containers:
        host = item["HostConfig"]
        occupied_cpus.update(cpuset(host.get("CpusetCpus", "")))
        for request in host.get("DeviceRequests") or []:
            if request.get("Count") == -1:
                occupied_gpus.update(gpus)
            for identifier in request.get("DeviceIDs") or []:
                for index, row in gpus.items():
                    if identifier in (str(index), row["uuid"]):
                        occupied_gpus.add(index)
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    return {
        "created_utc": utc(),
        "gpus": gpus,
        "process_uuids": sorted(process_uuids),
        "occupied_cpus": sorted(occupied_cpus),
        "occupied_gpus": sorted(occupied_gpus),
        "containers": [c["Name"] for c in containers],
        "memory_available_kib": available_kib,
    }


def acquire_slot(held):
    cfg, observed = config(), snapshot()
    locks = BASE.parent / "gpu-allocation-locks"
    locks.mkdir(exist_ok=True)
    for gpu in cfg["runtime"]["gpu_preference"]:
        if gpu in held or gpu in observed["occupied_gpus"]:
            continue
        if not predecessor_released(gpu, cfg):
            continue
        row = observed["gpus"][gpu]
        if row["memory"] > 1024 or row["utilization"] > 5:
            continue
        if row["uuid"] in observed["process_uuids"]:
            continue
        cpus = list(range(16 + 8 * gpu, 24 + 8 * gpu))
        if not set(cpus).issubset(os.sched_getaffinity(0)):
            continue
        if set(cpus).intersection(observed["occupied_cpus"]):
            continue
        if observed["memory_available_kib"] < 48 * 1024**2:
            continue
        handle = (locks / f"gpu-{gpu}.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        return {"gpu": gpu, "cpus": cpus, "lock": handle, "observed": observed}
    return None


def command(name, slot, arguments):
    source = BASE / "source"
    mounts = [
        (BASE, False),
        (source, True),
        (Path(config()["baseline_batch"]), True),
        (Path("/ossfs/workspace/llm-memory-editability"), True),
        (Path("/ossfs/workspace/research-recovery/hf-review-20261010-6c580455"), True),
    ]
    cmd = [
        DOCKER,
        "--context",
        "lm-memory",
        "run",
        "--name",
        name,
        "--gpus",
        f"device={slot['gpu']}",
        "--cpus",
        "8",
        "--cpuset-cpus",
        ",".join(map(str, slot["cpus"])),
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
    ]
    for path, readonly in mounts:
        cmd += [
            "--mount",
            f"type=bind,source={path},target={path}" + (",readonly" if readonly else ""),
        ]
    return cmd + [
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


def task(stage, arm, slot, arguments):
    name = f"lm-supervised-v1-{stage}-{arm}"
    directory = BASE / "execution" / stage / arm
    directory.mkdir(parents=True, exist_ok=False)
    cmd = command(name, slot, arguments)
    write(directory / "command.json", {"command": cmd, "created_utc": utc()})
    with (directory / "stdout.log").open("w") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        write(
            directory / "process.json", {"pid": process.pid, "container": name, "gpu": slot["gpu"]}
        )
        try:
            code = process.wait(timeout=config()["runtime"]["hard_timeout_hours"] * 3600)
        except subprocess.TimeoutExpired:
            write(directory / "timeout.json", {"created_utc": utc(), "limit_hours": 12})
            docker(["stop", "--time", "30", name], capture_output=True, text=True)
            code = process.wait(timeout=60)
    item = json.loads(docker(["inspect", name], capture_output=True, text=True).stdout)[0]
    result = {
        "exit_code": code,
        "container_state": item["State"],
        "image": item["Image"],
        "finished_utc": utc(),
    }
    write(directory / "exit.json", result)
    if code:
        raise RuntimeError(f"{stage}/{arm} exited {code}; all output retained")
    return result


def arguments(mode, arm=None, smoke=False):
    args = [
        str(BASE / "source/scripts/run_supervised_alignment.py"),
        mode,
        "--config",
        str(BASE / "source" / CONFIG),
        "--base",
        str(BASE / "runs"),
    ]
    if arm is not None:
        args += ["--arm", arm]
    if smoke:
        args += ["--smoke"]
    return args


def schedule(items, worker, stage):
    pending, running, results = list(items), {}, {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config()["runtime"]["max_parallel"]
    ) as pool:
        while pending or running:
            for future in list(running):
                if future.done():
                    arm, slot = running.pop(future)
                    try:
                        results[arm] = future.result()
                    except Exception as exc:
                        results[arm] = {
                            "state": "failed",
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    finally:
                        slot["lock"].close()
            while pending and len(running) < config()["runtime"]["max_parallel"]:
                slot = acquire_slot({slot["gpu"] for _, slot in running.values()})
                if slot is None:
                    break
                arm = pending.pop(0)
                write(
                    BASE / "allocations" / stage / f"{arm}.json",
                    {
                        "gpu": slot["gpu"],
                        "cpus": slot["cpus"],
                        "observed": slot["observed"],
                        "created_utc": utc(),
                    },
                )
                running[pool.submit(worker, arm, slot)] = (arm, slot)
            write(
                BASE / f"{stage}-schedule.json",
                {
                    "pending": pending,
                    "running": {
                        arm: {"gpu": slot["gpu"], "cpus": slot["cpus"]}
                        for arm, slot in running.values()
                    },
                    "results": results,
                    "updated_utc": utc(),
                    "automatic_retry": False,
                },
            )
            if pending or running:
                time.sleep(config()["runtime"]["poll_seconds"])
    return results


def validate():
    lock = (BASE / "validation.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()

    def engineering(name, slot):
        if name == "contracts":
            return task("tests", name, slot, ["-m", "pytest", "-q", *TESTS])
        return task("compatibility", name, slot, arguments("compatibility"))

    checks = schedule(["contracts", "baseline"], engineering, "validation")
    if any(v.get("state") == "failed" for v in checks.values()):
        write(BASE / "validation.json", {"passed": False, "checks": checks, "created_utc": utc()})
        raise RuntimeError("Engineering validation failed; preserved without automatic retry")
    if not read(BASE / "runs/compatibility.json")["passed"]:
        raise RuntimeError("Baseline compatibility did not pass")

    def short(arm, slot):
        task("smoke", arm, slot, arguments("smoke", arm))
        return task("smoke-audit", arm, slot, arguments("audit", arm, smoke=True))

    outcomes = schedule(config()["arms"], short, "smoke")
    passed = all(v.get("exit_code") == 0 for v in outcomes.values())
    write(
        BASE / "validation.json",
        {
            "passed": passed,
            "checks": checks,
            "smokes": outcomes,
            "created_utc": utc(),
            "science_started": False,
        },
    )
    if not passed:
        raise RuntimeError("Full-size smoke/reload failed; retained without automatic retry")


def track(arm):
    """Use the established tracking transport, with this batch's dynamic allocation."""
    shared.BASE = BASE

    def tracking_config():
        cfg = config()
        cfg["runtime"]["gpus"] = {arm: read(BASE / "allocations/science" / f"{arm}.json")["gpu"]}
        return cfg

    shared.config = tracking_config
    shared.track(arm)


def science(arm, slot):
    directory = BASE / "tracking" / arm
    tracker = None
    try:
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "stdout.log").open("w") as log:
            tracker = subprocess.Popen(
                [
                    PYTHON,
                    str(BASE / "source/scripts/execute_supervised_alignment.py"),
                    "track",
                    "--arm",
                    arm,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=shared.tracking_environment(),
            )
        write(directory / "process.json", {"pid": tracker.pid, "created_utc": utc()})
        deadline = time.monotonic() + 120
        while not (directory / "ready.json").exists():
            if tracker.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("W&B tracking failed to become ready")
            time.sleep(2)
        task("train", arm, slot, arguments("train", arm))
        task("audit", arm, slot, arguments("audit", arm))
        result = {"state": "complete", "created_utc": utc()}
    except Exception as exc:
        result = {"state": "failed", "error": str(exc), "created_utc": utc()}
    write(BASE / "task-state" / f"{arm}.json", result)
    if tracker is not None:
        try:
            tracker.wait(timeout=120)
        except subprocess.TimeoutExpired:
            result["tracking"] = "still_running"
    return result


def factorial_summary():
    cfg = config()
    previous = Path(cfg["baseline_batch"]) / "runs" / cfg["baseline_arm"]
    if not (previous / "audit.json").exists() or not read(previous / "audit.json")["passed"]:
        write(
            BASE / "comparison-state.json",
            {"state": "waiting_for_baseline_audit", "created_utc": utc()},
        )
        return False
    endpoints = {arm: read(BASE / "runs" / arm / "complete.json") for arm in cfg["arms"]}
    endpoints["baseline"] = read(previous / "complete.json")
    if any(row["step"] != cfg["spec"]["steps"] for row in endpoints.values()):
        raise ValueError("Endpoint budgets differ")
    comparisons = {}
    for panel in endpoints["baseline"]["endpoint"]["metrics"]:
        a = {name: row["endpoint"]["metrics"][panel]["accuracy"] for name, row in endpoints.items()}
        comparisons[panel] = {
            "accuracy": a,
            "interaction_pp": 100
            * ((a["same_kv_alignment"] - a["same_kv"]) - (a["alignment"] - a["baseline"])),
            "joint_minus_kv_pp": 100 * (a["same_kv_alignment"] - a["same_kv"]),
            "joint_minus_alignment_pp": 100 * (a["same_kv_alignment"] - a["alignment"]),
            "alignment_minus_label_ce_pp": 100 * (a["alignment"] - a["label_ce"]),
            "kv_alignment_minus_kv_label_ce_pp": 100
            * (a["same_kv_alignment"] - a["same_kv_label_ce"]),
            "discoloop_minus_baseline_pp": 100 * (a["discoloop"] - a["baseline"]),
            "alignment_minus_discoloop_pp": 100 * (a["alignment"] - a["discoloop"]),
        }
    write(
        BASE / "factorial-summary.json",
        {
            "created_utc": utc(),
            "primary_panel": "test_oi",
            "fixed_step": 300000,
            "panels": comparisons,
            "baseline_source": str(previous),
            "baseline_complete_sha256": sha(previous / "complete.json"),
            "baseline_audit_sha256": sha(previous / "audit.json"),
            "costs": {
                arm: {k: v for k, v in row["endpoint"].items() if k != "metrics"}
                for arm, row in endpoints.items()
            },
            "scope": "One observed graph and initialization; no independent-world inference",
        },
    )
    return True


def cloud_verify():
    import wandb

    os.environ.update(shared.tracking_environment())
    api, cloud = wandb.Api(), {}
    for arm in config()["arms"]:
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
            "runs": cloud,
            "scope": "state/final step/audit summary, not every history point",
            "created_utc": utc(),
        },
    )
    return passed


def await_comparison():
    cfg = config()
    marker = Path(cfg["baseline_batch"]) / "task-state" / f"{cfg['baseline_arm']}.json"
    deadline = time.monotonic() + cfg["runtime"]["hard_timeout_hours"] * 3600
    while True:
        if factorial_summary():
            write(BASE / "comparison-state.json", {"state": "complete", "created_utc": utc()})
            return True
        if marker.exists() and read(marker)["state"] in ("failed", "complete"):
            write(
                BASE / "comparison-state.json",
                {
                    "state": "baseline_audit_unavailable",
                    "created_utc": utc(),
                    "reason": "Predecessor terminal without passed baseline audit; no restart",
                },
            )
            return False
        if time.monotonic() > deadline:
            write(
                BASE / "comparison-state.json",
                {
                    "state": "needs_attention",
                    "created_utc": utc(),
                    "reason": "12-hour comparison wait expired; science artifacts retained",
                },
            )
            return False
        time.sleep(cfg["runtime"]["poll_seconds"])


def controller():
    lock = (BASE / "execution.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()
    if not read(BASE / "validation.json")["passed"]:
        raise RuntimeError("Validation incomplete")
    write(
        BASE / "controller-state.json",
        {
            "state": "running",
            "arms": list(config()["arms"]),
            "created_utc": utc(),
            "automatic_retry": False,
        },
    )
    outcomes = schedule(config()["arms"], science, "science")
    complete = all(v.get("state") == "complete" for v in outcomes.values())
    write(
        BASE / "controller-state.json",
        {
            "state": "scientific_runs_and_audits_complete" if complete else "needs_attention",
            "arms": outcomes,
            "updated_utc": utc(),
            "automatic_retry": False,
        },
    )
    if complete:
        cloud = cloud_verify()
        comparison = await_comparison()
        write(
            BASE / "completion-manifest.json",
            {
                "science_complete": True,
                "audits_passed": True,
                "comparison_complete": comparison,
                "cloud_verified": cloud,
                "optimization_updates": 1800000,
                "created_utc": utc(),
            },
        )


def launch():
    marker = BASE / "controller-process.json"
    launch_lock = (BASE / "launch.lock").open("a+")
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if marker.exists():
        raise RuntimeError("Existing submission retained; never launch twice")
    if not read(BASE / "validation.json")["passed"]:
        raise RuntimeError("Validation incomplete")
    verify()
    cmd = [PYTHON, str(BASE / "source/scripts/execute_supervised_alignment.py"), "controller"]
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
