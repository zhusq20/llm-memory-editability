"""Run the frozen zero-update readout matrix on independent idle LM GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RECOVERY = Path("/ossfs/workspace/research-recovery")
ARCHIVE = Path("/ossfs/workspace/llm-memory-editability")
DOCKER = "/workspaces/lm-memory-tools/docker/docker"
CREDENTIALS = Path("/ossfs/workspace/.lm-memory-credentials/wandb.netrc")
IMAGE = "sha256:0087c5381ef5286d30b022825ab773f044fa15c161e5999ff5b47fbab681b18a"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def docker(args, **kwargs):
    return subprocess.run([DOCKER, "--context", "lm-memory", *args], **kwargs)


def cpus(text):
    values = set()
    for part in text.split(","):
        if part:
            start, _, end = part.partition("-")
            values.update(range(int(start), int(end or start) + 1))
    return values


def validate_config(cfg):
    if cfg.get("new_optimizer_updates") != 0:
        raise ValueError("This executor permits zero optimizer updates only")
    if cfg.get("batch") != "parallel-readouts-development-v1":
        raise ValueError("Executor is scoped to the reviewable development readout batch")
    lanes = cfg["execution"]["lanes"]
    tasks = cfg["execution"]["tasks"]
    if len({t["name"] for t in tasks}) != len(tasks):
        raise ValueError("Task names must be unique")
    if sorted(lanes) != ["1", "2", "3", "4", "5", "6", "7"]:
        raise ValueError("Only the seven originally idle GPUs 1--7 are authorized")
    all_cpu = [int(core) for lane in lanes.values() for core in lane]
    if len(all_cpu) != len(set(all_cpu)) or any(len(lane) != 8 for lane in lanes.values()):
        raise ValueError("Eight exclusive CPU cores are required per lane")
    if len(tasks) != 12 or any(str(t["gpu"]) not in lanes for t in tasks):
        raise ValueError("Exactly four restoration and eight phrasing tasks are frozen")
    if (
        sum(t["kind"] == "restoration" for t in tasks) != 4
        or sum(t["kind"] == "phrasing" for t in tasks) != 8
    ):
        raise ValueError("The two scientific families must retain four and eight tasks")
    return cfg


def resources(cfg):
    validate_config(cfg)
    intended = set(map(int, cfg["execution"]["lanes"]))
    ids = docker(["ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
    running = (
        json.loads(docker(["inspect", *ids], capture_output=True, text=True, check=True).stdout)
        if ids
        else []
    )
    occupied = set()
    for row in running:
        host = row["HostConfig"]
        occupied.update(cpus(host.get("CpusetCpus", "")))
        for request in host.get("DeviceRequests") or []:
            selected = set(map(int, request.get("DeviceIDs") or []))
            if (
                request.get("Count") == -1
                or (request.get("Count", 0) > 0 and not selected)
                or selected & intended
            ):
                raise RuntimeError(f"GPU collision: {row['Name']}")
    allocated = {int(x) for lane in cfg["execution"]["lanes"].values() for x in lane}
    # This host has 96 physical cores; PERK also owns the sibling threads.
    occupied |= {x + 96 if x < 96 else x - 96 for x in occupied}
    if allocated & occupied or not allocated <= set(os.sched_getaffinity(0)):
        raise RuntimeError("CPU allocation collides with an existing task")
    used = list(
        map(
            int,
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).split(),
        )
    )
    if any(used[gpu] > 500 for gpu in intended):
        raise RuntimeError("An intended GPU is already in use")
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < len(intended) * 48 * 1024 * 1024:
        raise RuntimeError("Insufficient host memory for the frozen lanes")
    identity = json.loads(
        docker(["image", "inspect", IMAGE], capture_output=True, text=True, check=True).stdout
    )[0]["Id"]
    if identity != IMAGE:
        raise ValueError("Fixed image differs")
    return {
        "created_utc": utc(),
        "gpu_memory_used_mib": used,
        "lanes": cfg["execution"]["lanes"],
        "available_memory_kib": available,
        "image": identity,
    }


def freeze(config):
    cfg = validate_config(read(config))
    base = RECOVERY / cfg["batch"]
    dest = base / "frozen"
    if dest.exists():
        raise FileExistsError("Never overwrite a freeze")
    dest.mkdir(parents=True)
    shutil.copytree(
        PROJECT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in cfg["execution"]["source_files"]:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, target)
    shutil.copy2(config, dest / "config.json")
    write(
        dest / "freeze.json",
        {
            "batch": cfg["batch"],
            "created_utc": utc(),
            "image": IMAGE,
            "source_files": {
                str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob("*")) if p.is_file()
            },
            "inputs": {str(p): sha(p) for p in map(Path, cfg["execution"]["input_files"])},
            "new_optimizer_updates": 0,
            "evidence_class": "post_hoc_development_readouts",
        },
    )
    link = PROJECT / "results" / cfg["batch"]
    if not link.exists():
        link.symlink_to(base, target_is_directory=True)
    return base


def verify(base):
    base = Path(base)
    frozen = base / "frozen"
    record = read(frozen / "freeze.json")
    for name, expected in record["source_files"].items():
        if sha(frozen / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in record["inputs"].items():
        if sha(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return validate_config(read(frozen / "config.json"))


def task_command(base, task, stage, smoke=False):
    base = Path(base)
    cfg = read(base / "frozen/config.json")
    if stage not in ("science", "audit") or task not in cfg["execution"]["tasks"]:
        raise ValueError("Task/stage outside the frozen matrix")
    out = base / ("preflight" if smoke else "runs") / task["name"]
    out.mkdir(parents=True, exist_ok=True)
    args = [
        word.format(
            config=str(base / "frozen/config.json"),
            out=str(out),
            audit_out=str(out / "audit"),
            frozen=str(base / "frozen"),
        )
        for word in task[f"{stage}_arguments"]
    ]
    if smoke and stage == "science":
        args += task["smoke_arguments"]
    if smoke and stage == "audit":
        args += task.get("audit_smoke_arguments", [])
    name = f"lm-par-readouts-{task['name']}-{stage}" + ("-smoke" if smoke else "")
    gpu = task["gpu"]
    lane = cfg["execution"]["lanes"][str(gpu)]
    script = str(base / "frozen" / task["script"])
    child = ["python", "-u", script, *args, "--device", "cuda:0"]
    if stage == "science" and not smoke:
        child = [
            "python",
            "-u",
            str(base / "frozen/scripts/track_parallel_readout.py"),
            "--config",
            str(base / "frozen/config.json"),
            "--task",
            task["name"],
            "--out",
            str(out),
            "--",
            *child,
        ]
    cmd = [
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        f"device={gpu}",
        "--cpus",
        "8",
        "--cpuset-cpus",
        ",".join(map(str, lane)),
        "--memory",
        "48g",
        "--memory-swap",
        "48g",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--network",
        "host",
    ]
    for source, target, readonly in [
        (RECOVERY, RECOVERY, True),
        (ARCHIVE, ARCHIVE, True),
        (base, base, False),
        (base / "frozen", base / "frozen", True),
        (CREDENTIALS, Path("/root/.netrc"), True),
    ]:
        cmd += [
            "--mount",
            f"type=bind,source={source},target={target}" + (",readonly" if readonly else ""),
        ]
    for value in [
        f"PYTHONPATH={base / 'frozen/src'}",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "TOKENIZERS_PARALLELISM=false",
        "WANDB_SILENT=true",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
    ]:
        cmd += ["--env", value]
    cmd += ["--workdir", str(base / "frozen"), IMAGE, *child]
    return cmd, name, out


def execute_task(base, task, smoke=False):
    receipts = []
    for stage in ("science", "audit"):
        cmd, name, out = task_command(base, task, stage, smoke)
        result = docker(cmd, capture_output=True, text=True)
        receipt = {
            "stage": stage,
            "task": task["name"],
            "created_utc": utc(),
            "physical_gpu": task["gpu"],
            "container": name,
            "command": cmd,
        }
        if result.returncode:
            receipt.update(state="launch_failed", exit_code=result.returncode, error=result.stderr)
        else:
            container = result.stdout.strip()
            started = time.monotonic()
            receipt.update(state="running", container_id=container)
            write(out / f"{stage}-process.json", receipt)
            while True:
                state = json.loads(
                    docker(
                        ["inspect", "--format", "{{json .State}}", container],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                )
                if not state["Running"]:
                    break
                if (
                    time.monotonic() - started
                    > read(Path(base) / "frozen/config.json")["execution"]["hard_timeout_seconds"]
                ):
                    docker(
                        ["stop", "--time", "30", container],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    receipt["hard_timeout"] = True
                time.sleep(5)
            logs = docker(["logs", container], capture_output=True, text=True)
            (out / f"{stage}.log").write_text(logs.stdout + logs.stderr)
            receipt.update(
                state="complete" if state["ExitCode"] == 0 else "failed",
                exit_code=state["ExitCode"],
                completed_utc=utc(),
            )
        write(out / f"{stage}-process.json", receipt)
        receipts.append(receipt)
        if receipt["exit_code"] != 0:
            return {"task": task["name"], "passed": False, "receipts": receipts}
        expected = out / ("complete.json" if stage == "science" else "audit/audit.json")
        evidence = read(expected)
        if evidence.get("passed") is not True:
            return {
                "task": task["name"],
                "passed": False,
                "receipts": receipts,
                "error": f"Invalid endpoint receipt: {expected}",
            }
    return {"task": task["name"], "passed": True, "receipts": receipts}


def controller(base):
    base = Path(base)
    cfg = verify(base)
    write(base / "resource-check.json", resources(cfg))
    results = []
    active = [task["name"] for task in cfg["execution"]["tasks"]]

    def update():
        write(
            base / "controller-state.json",
            {
                "state": "running"
                if active
                else (
                    "complete"
                    if all(row["passed"] for row in results)
                    else "finished_with_failures"
                ),
                "active": active,
                "completed": [r["task"] for r in results if r["passed"]],
                "failed": [r for r in results if not r["passed"]],
                "updated_utc": utc(),
            },
        )

    def lane(gpu):
        lane_results = []
        for task in cfg["execution"]["tasks"]:
            if task["gpu"] == gpu:
                try:
                    row = execute_task(base, task)
                except Exception as error:
                    row = {"task": task["name"], "passed": False, "error": repr(error)}
                lane_results.append(row)
        return lane_results

    update()
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(lane, int(gpu)) for gpu in cfg["execution"]["lanes"]]
        for future in concurrent.futures.as_completed(futures):
            for row in future.result():
                results.append(row)
                active.remove(row["task"])
            update()
    write(
        base / "completion-manifest.json",
        {
            "passed": all(r["passed"] for r in results),
            "tasks": results,
            "new_optimizer_updates": 0,
            "independent_container_audits": True,
            "cloud_verification": "separate_receipt",
            "completed_utc": utc(),
        },
    )


def launch(base):
    base = Path(base)
    cfg = verify(base)
    preflight = read(base / "preflight-passed.json")
    if preflight.get("passed") is not True or preflight.get("freeze_sha256") != sha(
        base / "frozen/freeze.json"
    ):
        raise ValueError(
            "Fixed-image contracts and CUDA plus independent replay must bind this freeze"
        )
    coverage = set()
    for receipt in preflight["receipts"]:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Preflight evidence changed")
        row = read(receipt["path"])
        if row.get("exit_code") != 0 or row.get("state") != "complete":
            raise ValueError("CUDA preflight or independent replay failed")
        task = next(t for t in cfg["execution"]["tasks"] if t["name"] == row["task"])
        coverage.add((task["kind"], row["stage"]))
    if coverage != {
        (kind, stage) for kind in ("restoration", "phrasing") for stage in ("science", "audit")
    }:
        raise ValueError("Both readout families require CUDA and fresh-container replay")
    for evidence in preflight.get("contract_evidence", []):
        if sha(evidence["path"]) != evidence["sha256"]:
            raise ValueError("Fixed-image contracts changed")
    if not preflight.get("contract_evidence"):
        raise ValueError("Fixed-image contract tests are required")
    with (base / "submission.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if (base / "controller-process.json").exists():
            raise RuntimeError("Batch already submitted; no duplicate launch or automatic retry")
        resources(cfg)
        command = [
            sys.executable,
            "-u",
            str(base / "frozen/scripts/execute_parallel_readouts.py"),
            "controller",
            "--base",
            str(base),
        ]
        with (base / "controller.log").open("ab") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(
            base / "controller-process.json",
            {
                "pid": process.pid,
                "command": command,
                "created_utc": utc(),
                "wandb_project": "https://wandb.ai/zhusq20/llm-memory-editability",
            },
        )
    return {"pid": process.pid, "base": str(base)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "smoke", "launch", "controller"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.config))
    elif args.action == "verify":
        verify(args.base)
        print("verified")
    elif args.action == "launch":
        print(json.dumps(launch(args.base)))
    elif args.action == "controller":
        controller(args.base)
    else:
        cfg = verify(args.base)
        resources(cfg)
        task = next(task for task in cfg["execution"]["tasks"] if task["name"] == args.task)
        print(json.dumps(execute_task(args.base, task, smoke=True)))


if __name__ == "__main__":
    main()
