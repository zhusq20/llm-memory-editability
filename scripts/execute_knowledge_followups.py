"""Freeze and run six independent development lanes on idle LM GPUs 2--7.

Each lane runs once and then enters a new container for its audit. Failed stages
are retained and never retried. This controller cannot submit an existing batch.
"""

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
DOCKER = "/workspaces/lm-memory-tools/docker/docker"
CREDENTIALS = Path("/ossfs/workspace/.lm-memory-credentials/wandb.netrc")
IMAGE = "sha256:0087c5381ef5286d30b022825ab773f044fa15c161e5999ff5b47fbab681b18a"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def docker(args, **kwargs):
    return subprocess.run([DOCKER, "--context", "lm-memory", *args], **kwargs)


def parse_cpus(text):
    result = set()
    for block in text.split(","):
        if not block:
            continue
        a, _, b = block.partition("-")
        result.update(range(int(a), int(b or a) + 1))
    return result


def resources(gpus):
    if gpus != list(range(2, 8)):
        raise ValueError("This development matrix uses only GPUs 2--7")
    ids = docker(["ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
    rows = (
        json.loads(docker(["inspect", *ids], capture_output=True, text=True, check=True).stdout)
        if ids
        else []
    )
    occupied = set()
    for row in rows:
        host = row["HostConfig"]
        occupied.update(parse_cpus(host.get("CpusetCpus", "")))
        for request in host.get("DeviceRequests") or []:
            assigned = set(map(int, request.get("DeviceIDs") or []))
            unknown_assignment = request.get("Count", 0) > 0 and not assigned
            if request.get("Count") == -1 or unknown_assignment or assigned.intersection(gpus):
                raise RuntimeError(f"GPU collision: {row['Name']}")
    used = list(
        map(
            int,
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            ).split(),
        )
    )
    if any(used[gpu] > 500 for gpu in gpus):
        raise RuntimeError("An intended GPU has an existing allocation")
    free = sorted(set(os.sched_getaffinity(0)) - occupied)
    if len(free) < 48:
        raise RuntimeError("Need 48 unoccupied CPU cores")
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 6 * 48 * 1024 * 1024:
        raise RuntimeError("Need 288 GiB currently available host memory")
    image = json.loads(
        docker(["image", "inspect", IMAGE], capture_output=True, text=True, check=True).stdout
    )[0]["Id"]
    if image != IMAGE:
        raise ValueError("Immutable image identity differs")
    return {
        "gpus": gpus,
        "cpus": [free[i * 8 : (i + 1) * 8] for i in range(6)],
        "gpu_memory_used_mib": used,
        "image": image,
        "created_utc": utc(),
    }


def freeze(config_path):
    cfg = read(config_path)
    base = RECOVERY / cfg["batch"]
    if (base / "frozen").exists():
        raise FileExistsError("Do not overwrite an existing freeze")
    frozen = base / "frozen"
    frozen.mkdir(parents=True)
    shutil.copytree(
        PROJECT / "src", frozen / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in cfg["execution"]["source_files"]:
        dest = frozen / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / name, dest)
    shutil.copy2(config_path, frozen / "config.json")
    inputs = {str(p): sha(p) for p in map(Path, cfg["execution"]["input_files"])}
    source_hashes = {
        str(p.relative_to(frozen)): sha(p) for p in sorted(frozen.rglob("*")) if p.is_file()
    }
    write(
        frozen / "freeze.json",
        {
            "batch": cfg["batch"],
            "image": IMAGE,
            "source_files": source_hashes,
            "inputs": inputs,
            "created_utc": utc(),
            "development_only": True,
            "existing_batches_untouched": True,
        },
    )
    link = PROJECT / "results" / cfg["batch"]
    if not link.exists():
        link.symlink_to(base, target_is_directory=True)
    return base


def verify(base):
    frozen = Path(base) / "frozen"
    meta = read(frozen / "freeze.json")
    for name, expected in meta["source_files"].items():
        if sha(frozen / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for name, expected in meta["inputs"].items():
        if sha(name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return read(frozen / "config.json")


def command(base, index, stage, cpus, gpu, smoke=False):
    if gpu not in range(2, 8) or stage not in ("science", "audit") or index not in range(6):
        raise ValueError("Task is outside the authorized six-lane matrix")
    if len(cpus) != 8 or len(set(cpus)) != 8 or (not smoke and gpu != index + 2):
        raise ValueError("Need eight exclusive CPUs and the frozen lane GPU")
    base = Path(base)
    cfg = read(base / "frozen/config.json")
    task = cfg["execution"]["tasks"][index]
    amendment_path = base / "runtime-amendment.json"
    amendment = read(amendment_path) if amendment_path.exists() else {}
    tag = amendment.get("attempt_tag")
    name = f"lm-{cfg['execution']['container_prefix']}-{index}-{stage}"
    if tag:
        name += f"-{tag}"
    if smoke:
        name += "-smoke"
    preflight = base / "preflight" / tag if tag else base / "preflight"
    out = (preflight / "smoke" if smoke else base / "runs") / task["name"]
    out.mkdir(parents=True, exist_ok=True)
    args = [
        "python",
        "-u",
        str(base / "frozen" / task["script"]),
        "--config",
        str(base / "frozen/config.json"),
        *task["arguments"],
        "--out",
        str(out),
        "--device",
        "cuda:0",
    ]
    if smoke:
        args.append("--smoke")
    if stage == "audit":
        args.extend(arg.format(out=str(out)) for arg in task.get("audit_arguments", ["--audit"]))
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
        ",".join(map(str, cpus)),
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
        "--mount",
        f"type=bind,source={RECOVERY},target={RECOVERY},readonly",
        "--mount",
        f"type=bind,source={base},target={base}",
        "--mount",
        f"type=bind,source={base / 'frozen'},target={base / 'frozen'},readonly",
        "--mount",
        f"type=bind,source={CREDENTIALS},target=/root/.netrc,readonly",
        "--env",
        f"PYTHONPATH={base / 'frozen/src'}",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "TOKENIZERS_PARALLELISM=false",
        "--env",
        "WANDB_SILENT=true",
        "--workdir",
        str(base / "frozen"),
        IMAGE,
        *args,
    ]
    extra_mounts = []
    for path in cfg["execution"]["input_files"]:
        target = Path(path).resolve()
        if not target.is_relative_to(RECOVERY):
            extra_mounts.append(target.parent)
    insertion = cmd.index(IMAGE)
    mounts = []
    for target in sorted(set(extra_mounts)):
        mounts.extend(["--mount", f"type=bind,source={target},target={target},readonly"])
    cmd[insertion:insertion] = mounts
    return cmd, name, out


def lane(base, index, cpus, gpu, smoke=False):
    cfg = read(Path(base) / "frozen/config.json")
    task = cfg["execution"]["tasks"][index]
    reports = []
    for stage in ("science", "audit"):
        cmd, name, out = command(base, index, stage, cpus, gpu, smoke)
        result = docker(cmd, capture_output=True, text=True)
        if result.returncode:
            report = {
                "stage": stage,
                "state": "launch_failed",
                "error": result.stderr,
                "exit_code": result.returncode,
                "created_utc": utc(),
            }
            write(out / f"{stage}-process.json", report)
            return {"task": task["name"], "passed": False, "reports": [*reports, report]}
        identity = result.stdout.strip()
        report = {
            "stage": stage,
            "state": "running",
            "container": name,
            "container_id": identity,
            "physical_gpu": gpu,
            "cpus": cpus,
            "started_utc": utc(),
        }
        write(out / f"{stage}-process.json", report)
        while True:
            state = json.loads(
                docker(
                    ["inspect", "--format", "{{json .State}}", identity],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )
            if not state["Running"]:
                break
            time.sleep(10)
        logs = docker(["logs", identity], capture_output=True, text=True)
        (out / f"{stage}.log").write_text(logs.stdout + logs.stderr)
        report.update(
            state="complete" if state["ExitCode"] == 0 else "failed",
            exit_code=state["ExitCode"],
            completed_utc=utc(),
        )
        write(out / f"{stage}-process.json", report)
        reports.append(report)
        if state["ExitCode"] != 0:
            return {"task": task["name"], "passed": False, "reports": reports}
    return {"task": task["name"], "passed": True, "reports": reports}


def controller(base):
    base = Path(base)
    cfg = verify(base)
    predecessor = cfg["execution"].get("after_batch")
    if predecessor:
        dependency = RECOVERY / predecessor / "completion-manifest.json"
        write(
            base / "controller-state.json",
            {
                "state": "waiting_for_predecessor",
                "predecessor": predecessor,
                "active": [],
                "updated_utc": utc(),
            },
        )
        while not dependency.exists():
            time.sleep(10)
        if read(dependency).get("passed") is not True:
            write(
                base / "controller-state.json",
                {
                    "state": "predecessor_failed",
                    "predecessor": predecessor,
                    "active": [],
                    "updated_utc": utc(),
                },
            )
            return
    allocation = resources(list(range(2, 8)))
    write(base / "resource-check.json", allocation)
    result = []
    pending = [task["name"] for task in cfg["execution"]["tasks"]]
    if len(pending) != 6:
        raise ValueError("Frozen matrix must contain exactly six tasks")
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(lane, base, i, allocation["cpus"][i], i + 2): pending[i] for i in range(6)
        }
        write(
            base / "controller-state.json",
            {
                "state": "running",
                "active": pending,
                "completed": [],
                "failed": [],
                "updated_utc": utc(),
            },
        )
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                row = future.result()
            except Exception as error:
                row = {"task": name, "passed": False, "error": repr(error)}
            result.append(row)
            pending.remove(name)
            write(
                base / "controller-state.json",
                {
                    "state": "running"
                    if pending
                    else (
                        "complete" if all(r["passed"] for r in result) else "finished_with_failures"
                    ),
                    "active": list(pending),
                    "completed": [r["task"] for r in result if r["passed"]],
                    "failed": [r for r in result if not r["passed"]],
                    "updated_utc": utc(),
                },
            )
    pairing = {}
    if cfg.get("study_kind") == "planned_U1_development_only" and all(r["passed"] for r in result):
        for role in ("first", "second"):
            hashes = [
                sha(base / "runs" / f"{role}-{scope}" / "case-bank.json")
                for scope in ("joint", "round1", "round2")
            ]
            pairing[role] = {"passed": len(set(hashes)) == 1, "case_bank_hashes": hashes}
    passed = all(r["passed"] for r in result) and all(r["passed"] for r in pairing.values())
    if not passed:
        write(
            base / "controller-state.json",
            {
                "state": "finished_with_failures",
                "active": [],
                "tasks": result,
                "paired_case_banks": pairing,
                "updated_utc": utc(),
            },
        )
    write(
        base / "completion-manifest.json",
        {
            "passed": passed,
            "tasks": result,
            "paired_case_banks": pairing,
            "development_only": True,
            "independent_container_audits": True,
            "wandb_cloud_verification": "Separate receipt required",
            "completed_utc": utc(),
        },
    )


def launch(base):
    base = Path(base)
    verify(base)
    preflight_path = base / "preflight-passed.json"
    if not preflight_path.exists():
        raise ValueError("Real CUDA preflight and fresh-process replay are required")
    preflight = read(preflight_path)
    if preflight.get("passed") is not True or preflight.get("freeze_sha256") != sha(
        base / "frozen/freeze.json"
    ):
        raise ValueError("Preflight must bind the actual frozen batch")
    if not preflight.get("receipts"):
        raise ValueError("Real CUDA and replay evidence are required")
    for receipt in preflight["receipts"]:
        if sha(receipt["path"]) != receipt["sha256"]:
            raise ValueError("CUDA preflight receipt changed")
        row = read(receipt["path"])
        if row.get("exit_code") != 0 or row.get("state") != "complete":
            raise ValueError("CUDA science/audit process did not complete successfully")
    if {read(r["path"]).get("stage") for r in preflight["receipts"]} != {"science", "audit"}:
        raise ValueError("Both CUDA science and fresh-process audit are required")
    with (base / "submission.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if (base / "controller-process.json").exists():
            raise RuntimeError("This batch has already been submitted; no automatic retry")
        if not read(base / "frozen/config.json")["execution"].get("after_batch"):
            resources(list(range(2, 8)))
        script = base / "orchestration/execute_knowledge_followups.py"
        if not script.exists():
            script = base / "frozen/scripts/execute_knowledge_followups.py"
        args = [sys.executable, "-u", str(script), "controller", "--base", str(base)]
        with (base / "controller.log").open("ab") as log:
            proc = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(
            base / "controller-process.json",
            {
                "pid": proc.pid,
                "command": args,
                "created_utc": utc(),
                "wandb_project": "https://wandb.ai/zhusq20/llm-memory-editability",
            },
        )
    print(json.dumps({"pid": proc.pid, "base": str(base)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "launch", "controller", "smoke"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.config))
    elif args.action == "verify":
        verify(args.base)
        print("verified")
    elif args.action == "launch":
        launch(args.base)
    elif args.action == "controller":
        controller(args.base)
    else:
        verify(args.base)
        allocation = resources(list(range(2, 8)))
        print(json.dumps(lane(args.base, args.index, allocation["cpus"][0], 2, smoke=True)))


if __name__ == "__main__":
    main()
