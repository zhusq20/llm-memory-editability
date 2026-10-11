"""Stage the authorized PERK writer comparison behind its existing prerequisite.

One controller waits for the original audited comparison, runs reader/write
calibration, full-reader CUDA smoke/replay, then freezes and launches four paired
writer trainings and one common fixed baseline. A failed gate stops the pipeline;
it never retries a scientific stage, restarts the parent, or selects by test data.
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
BASE = Path("/ossfs/workspace/research-recovery/perk-writer-comparison-v1")
DOCKER = "/workspaces/lm-memory-tools/docker/docker"
CREDENTIALS = Path("/ossfs/workspace/.lm-memory-credentials/wandb.netrc")
SOURCE_FILES = (
    "docs/development-artifacts/perk-writer-comparison-v1/protocol.md",
    "configs/perk-writer-comparison-v1.json",
    "src/llm_memory_editability/perk_writer_data.py",
    "src/llm_memory_editability/perk_writer_control.py",
    "src/llm_memory_editability/perk_writer_experiment.py",
    "src/llm_memory_editability/perk_reproduction.py",
    "scripts/run_perk_writer_experiment.py",
    "scripts/check_perk_writer_dependencies.py",
    "scripts/execute_perk_writer_comparison.py",
    "scripts/analyze_perk_writer_comparison.py",
    "tests/test_perk_writer_data.py",
    "tests/test_perk_writer_control.py",
    "tests/test_perk_writer_dependencies.py",
    "tests/test_perk_writer_comparison.py",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def docker(args, **kwargs):
    return subprocess.run([DOCKER, "--context", "lm-memory", *args], **kwargs)


def scientific_tasks(plan):
    if plan["conditions"] != ["atomic", "compositional", "fixed"]:
        raise ValueError("The authorized three conditions changed")
    if plan["checkpoint_rule"] != "fixed_final_budget":
        raise ValueError("Test-dependent checkpoint selection is not permitted")
    if plan["outer_updates_per_arm"] != plan["counts"]["train"] * plan["outer_epochs"]:
        raise ValueError("Outer update budget differs from the material stream")
    tasks = [
        (condition, seed)
        for seed in plan["training_seeds"]
        for condition in ("atomic", "compositional")
    ]
    tasks.append(("fixed", plan["training_seeds"][0]))
    if len(tasks) != 5 or len(set(tasks)) != 5:
        raise ValueError("Exactly four paired trainings and one common baseline are required")
    return tasks


def prepare(base):
    if base.exists():
        raise FileExistsError(base)
    plan = read(PROJECT / "configs/perk-writer-comparison-v1.json")
    scientific_tasks(plan)
    base.mkdir(parents=True)
    source = base / "source-preparation"
    shutil.copytree(PROJECT / "src", source / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for relative in SOURCE_FILES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, target)
    for name in ("pyproject.toml", "README.md"):
        shutil.copy2(PROJECT / name, source / name)
    write(base / "plan-preparation.json", plan)
    sys.path.insert(0, str(source / "src"))
    from llm_memory_editability.perk_writer_data import save_pools

    data = save_pools(base / "data", plan["counts"], plan["data_seed"])
    receipt = {
        "created_utc": utc(),
        "user_authorized": True,
        "scientific_recipe_frozen": False,
        "scientific_jobs_started": False,
        "source_files": {
            str(path.relative_to(source)): sha(path)
            for path in source.rglob("*")
            if path.is_file() and path.suffix in {".py", ".json", ".toml", ".md"}
        },
        "plan_sha256": sha(base / "plan-preparation.json"),
        "data": data,
    }
    write(base / "preparation-manifest.json", receipt)
    write(
        base / "controller-state.json",
        {"state": "prepared", "created_utc": utc(), "scientific_training": "not_started"},
    )
    return receipt


def verify_preparation(base):
    receipt = read(base / "preparation-manifest.json")
    if sha(base / "plan-preparation.json") != receipt["plan_sha256"]:
        raise ValueError("Prepared plan changed")
    for name, expected in receipt["source_files"].items():
        if sha(base / "source-preparation" / name) != expected:
            raise ValueError(f"Prepared source changed: {name}")
    for split, entry in receipt["data"]["files"].items():
        if sha(base / "data" / f"{split}.jsonl") != entry["sha256"]:
            raise ValueError(f"Prepared data changed: {split}")


def parse_cpus(value):
    result = set()
    for part in value.split(","):
        if part:
            first, _, last = part.partition("-")
            result.update(range(int(first), int(last or first) + 1))
    return result


def resources(gpu, cpus, memory_gib):
    ids = docker(["ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
    running = (
        json.loads(docker(["inspect", *ids], capture_output=True, text=True, check=True).stdout)
        if ids
        else []
    )
    occupied = set()
    for row in running:
        host = row["HostConfig"]
        occupied |= parse_cpus(host.get("CpusetCpus", ""))
        for request in host.get("DeviceRequests") or []:
            devices = request.get("DeviceIDs") or []
            if (
                request.get("Count") == -1
                or (request.get("Count", 0) > 0 and not devices)
                or str(gpu) in devices
            ):
                raise RuntimeError(f"GPU {gpu} collision with {row['Name']}")
    occupied |= {core + 96 if core < 96 else core - 96 for core in occupied}
    if set(cpus) & occupied:
        raise RuntimeError("CPU allocation collides with another running task")
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).split()
    if int(used[gpu]) > 500:
        raise RuntimeError(f"GPU {gpu} is occupied outside the Docker allocation")
    available = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < memory_gib * 1024 * 1024:
        raise RuntimeError("Insufficient available host memory")
    return {
        "created_utc": utc(),
        "gpu": gpu,
        "cpus": cpus,
        "memory_gib": memory_gib,
        "running_names": [row["Name"] for row in running],
    }


def run_container(base, plan, stage_name, arguments, gpu=1, cpus=None):
    cpus = cpus or list(range(16, 24))
    root = base / "execution" / stage_name
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    write(root / "resources.json", resources(gpu, cpus, plan["execution"]["memory_gib"]))
    source = base / "source-preparation"
    name = f"lm-writer-v1-{stage_name}"
    command = [
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        f"device={gpu}",
        "--cpus",
        str(len(cpus)),
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        f"{plan['execution']['memory_gib']}g",
        "--memory-swap",
        f"{plan['execution']['memory_gib']}g",
        "--shm-size",
        "8g",
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--network",
        "host",
        "--mount",
        "type=bind,source=/ossfs/workspace/research-recovery,target=/ossfs/workspace/research-recovery,readonly",
        "--mount",
        f"type=bind,source={base},target={base}",
        "--mount",
        f"type=bind,source={source},target={source},readonly",
        "--mount",
        f"type=bind,source={CREDENTIALS},target=/root/.netrc,readonly",
    ]
    for key, value in {
        "PYTHONPATH": str(source / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_SILENT": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PERK_WRITER_CUDA_CONTRACT": "1",
        "PERK_AUTHOR_SOURCE": plan["source"],
    }.items():
        command.extend(["--env", f"{key}={value}"])
    command.extend(["--workdir", str(source), plan["image"], "python", "-u", *arguments])
    write(
        root / "command.json",
        {"created_utc": utc(), "container": name, "command": command, "image": plan["image"]},
    )
    launched = docker(command, capture_output=True, text=True)
    if launched.returncode:
        write(
            root / "exit.json",
            {
                "exit_code": launched.returncode,
                "created_utc": utc(),
                "stderr": launched.stderr,
                "stage": "container_create",
            },
        )
        raise RuntimeError(f"Container creation failed: {name}")
    write(root / "container.json", {"container_id": launched.stdout.strip(), "name": name})
    waited = docker(["wait", name], capture_output=True, text=True, check=True)
    code = int(waited.stdout.strip())
    with (root / "container.log").open("w") as handle:
        docker(["logs", name], stdout=handle, stderr=subprocess.STDOUT, check=True)
    state = json.loads(
        docker(["inspect", name], capture_output=True, text=True, check=True).stdout
    )[0]
    receipt = {
        "created_utc": utc(),
        "exit_code": code,
        "container": name,
        "container_id": state["Id"],
        "image": state["Image"],
        "state": state["State"],
        "command_sha256": sha(root / "command.json"),
    }
    write(root / "exit.json", receipt)
    if code:
        raise RuntimeError(f"Stage {stage_name} exited {code}; original attempt retained")
    return receipt


def science_freeze(base, plan):
    verify_preparation(base)
    calibration = read(base / "calibration-result.json")
    if not calibration["passed"]:
        raise ValueError("Calibration did not pass")
    for condition in ("atomic", "compositional", "fixed"):
        root = base / "smoke" / f"{condition}-seed317"
        if read(root / "audit.json").get("passed") is not True:
            raise ValueError("CUDA development replay did not pass")
        completed = read(root / "science-complete.json")
        if completed["outer_updates"] != (0 if condition == "fixed" else 2):
            raise ValueError("CUDA development update count changed")
    plan = {
        **plan,
        "status": "scientific_recipe_frozen",
        "selected_inner_rate": calibration["selected_inner_rate"],
    }
    destination = base / "frozen"
    destination.mkdir()
    write(destination / "plan.json", plan)
    parent = Path(plan["prerequisite"])
    selection = read(parent / "runs/perk/checkpoint-selection.json")
    receipt = {
        "created_utc": utc(),
        "image": plan["image"],
        "preparation_sha256": sha(base / "preparation-manifest.json"),
        "prerequisite_status_sha256": sha(base / "dependency-status.json"),
        "parent_completion_sha256": sha(parent / "completion-manifest.json"),
        "parent_freeze_sha256": sha(parent / "frozen/freeze.json"),
        "parent_checkpoint_files": selection["files"],
        "calibration_sha256": sha(base / "calibration-result.json"),
        "scientific_plan_sha256": sha(destination / "plan.json"),
        "data_manifest_sha256": sha(base / "data/manifest.json"),
        "source_files": read(base / "preparation-manifest.json")["source_files"],
    }
    write(destination / "freeze.json", receipt)
    return plan


def pipeline(base):
    marker = base / "pipeline.lock"
    with marker.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base / "controller-started.json").exists():
            raise RuntimeError("This pipeline has already been submitted")
        write(base / "controller-started.json", {"pid": os.getpid(), "created_utc": utc()})
        verify_preparation(base)
        if not read(base / "implementation-validation.json").get("passed"):
            raise ValueError("Prepared implementation contracts were not validated")
        plan = read(base / "plan-preparation.json")
        source = base / "source-preparation"
        try:
            while True:
                output = base / "dependency-status.json"
                checked = subprocess.run(
                    [
                        sys.executable,
                        str(source / "scripts/check_perk_writer_dependencies.py"),
                        "--base",
                        plan["prerequisite"],
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                )
                if checked.returncode not in (0, 2) or not output.exists():
                    raise RuntimeError("Prerequisite checker failed")
                status = read(output)
                if status["invalid"]:
                    raise RuntimeError(f"Invalid prerequisite evidence: {status['invalid']}")
                if status["ready_for_calibration"]:
                    break
                write(
                    base / "controller-state.json",
                    {
                        "state": "waiting_for_original_perk",
                        "updated_utc": utc(),
                        "pending": status["pending"],
                        "scientific_training": "not_started",
                        "parent_updates": status["arms"].get("perk", {}).get("completed_updates"),
                    },
                )
                time.sleep(plan["execution"]["poll_seconds"])
            run_args = [str(source / "scripts/run_perk_writer_experiment.py")]
            common = ["--plan", str(base / "plan-preparation.json"), "--base", str(base)]
            write(
                base / "controller-state.json",
                {
                    "state": "calibrating",
                    "updated_utc": utc(),
                    "scientific_training": "not_started",
                },
            )
            run_container(base, plan, "calibration", run_args + ["calibrate", *common])
            write(
                base / "controller-state.json",
                {
                    "state": "cuda_development",
                    "updated_utc": utc(),
                    "scientific_training": "not_started",
                },
            )

            def smoke(condition):
                args = ["--condition", condition, "--seed", "317", "--stage", "smoke"]
                run_container(
                    base,
                    plan,
                    f"smoke-{condition}",
                    run_args + ["run", *common, *args, "--updates", "2", "--limit-worlds", "2"],
                    gpu={"atomic": 1, "compositional": 2, "fixed": 3}[condition],
                    cpus=list(
                        range(
                            {"atomic": 16, "compositional": 24, "fixed": 32}[condition],
                            {"atomic": 24, "compositional": 32, "fixed": 40}[condition],
                        )
                    ),
                )
                run_container(
                    base,
                    plan,
                    f"smoke-replay-{condition}",
                    run_args + ["replay", *common, *args],
                    gpu={"atomic": 1, "compositional": 2, "fixed": 3}[condition],
                    cpus=list(
                        range(
                            {"atomic": 16, "compositional": 24, "fixed": 32}[condition],
                            {"atomic": 24, "compositional": 32, "fixed": 40}[condition],
                        )
                    ),
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(smoke, ("atomic", "compositional", "fixed")))
            plan = science_freeze(base, plan)
            common = ["--plan", str(base / "frozen/plan.json"), "--base", str(base)]
            write(
                base / "controller-state.json",
                {
                    "state": "scientific_training",
                    "updated_utc": utc(),
                    "scientific_training": "in_progress",
                },
            )

            def scientific(row):
                index, (condition, seed) = row
                gpu, cores = index + 1, list(range(16 + index * 8, 24 + index * 8))
                args = ["--condition", condition, "--seed", str(seed)]
                run_container(
                    base,
                    plan,
                    f"science-{condition}-{seed}",
                    run_args + ["run", *common, *args],
                    gpu,
                    cores,
                )
                run_container(
                    base,
                    plan,
                    f"replay-{condition}-{seed}",
                    run_args + ["replay", *common, *args],
                    gpu,
                    cores,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(scientific, enumerate(scientific_tasks(plan))))
            write(
                base / "controller-state.json",
                {
                    "state": "analysis_and_cloud_audit",
                    "updated_utc": utc(),
                    "scientific_training": "complete",
                },
            )
            analysis = [
                str(source / "scripts/analyze_perk_writer_comparison.py"),
                "--base",
                str(base),
            ]
            run_container(base, plan, "analysis-cloud", analysis)
            write(
                base / "controller-state.json",
                {"state": "complete", "updated_utc": utc(), "scientific_training": "complete"},
            )
        except Exception as error:
            calibrated = base / "calibration-result.json"
            state = (
                "calibration_needs_attention"
                if calibrated.exists() and not read(calibrated)["passed"]
                else "failed"
            )
            write(
                base / "controller-state.json",
                {
                    "state": state,
                    "updated_utc": utc(),
                    "error": f"{type(error).__name__}: {error}",
                    "automatic_retry": False,
                    "scientific_training": "see_per_task_receipts",
                },
            )
            raise


def launch(base):
    if (base / "controller-process.json").exists():
        raise RuntimeError("Controller registration already exists")
    command = [
        sys.executable,
        str(base / "source-preparation/scripts/execute_perk_writer_comparison.py"),
        "pipeline",
        "--base",
        str(base),
    ]
    with (base / "controller.log").open("w") as handle:
        process = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
    write(
        base / "controller-process.json",
        {
            "controller_pid": process.pid,
            "command": command,
            "created_utc": utc(),
            "wandb_project": "https://wandb.ai/zhusq20/llm-memory-editability",
        },
    )
    return {
        "pid": process.pid,
        "registered": True,
        "scientific_training": "awaits_prerequisite_and_calibration",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "launch", "pipeline", "verify-preparation", "validate"]
    )
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.base)))
    elif args.mode == "launch":
        print(json.dumps(launch(args.base)))
    elif args.mode == "pipeline":
        pipeline(args.base)
    elif args.mode == "validate":
        import xml.etree.ElementTree as ET

        verify_preparation(args.base)
        plan = read(args.base / "plan-preparation.json")
        result = run_container(
            args.base,
            plan,
            "implementation-contracts",
            [
                "-m",
                "pytest",
                "tests/test_perk_writer_data.py",
                "tests/test_perk_writer_control.py",
                "tests/test_perk_writer_dependencies.py",
                "tests/test_perk_writer_comparison.py",
                "-q",
                "--junitxml=" + str(args.base / "implementation-contracts.xml"),
            ],
        )
        cases = list(ET.parse(args.base / "implementation-contracts.xml").iter("testcase"))
        cuda_cases = [
            case for case in cases if case.get("name", "").startswith("test_actual_cuda_matrix")
        ]
        if len(cuda_cases) != 1 or any(list(case) for case in cuda_cases):
            raise ValueError("Actual CUDA writer contract was skipped or failed")
        write(
            args.base / "implementation-validation.json",
            {
                "passed": True,
                "receipt": result,
                "science_or_reader_calibration": False,
                "real_cuda_writer_contract": True,
                "tests": len(cases),
                "skipped": sum(case.find("skipped") is not None for case in cases),
            },
        )
        print(json.dumps({"passed": True, "real_cuda_writer_contract": True}))
    else:
        verify_preparation(args.base)
        print("Prepared source and data hashes passed")


if __name__ == "__main__":
    main()
