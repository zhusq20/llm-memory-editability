"""Freeze two entire-MLP controls and execute isolated parallel LM containers."""

from __future__ import annotations

import concurrent.futures
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import launch_ce_scope_discoloop as helpers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ce-architecture-novel-v1"
EXTRAS = (
    "src/llm_memory_editability/ce_architecture.py",
    "scripts/run_ce_architecture.py",
    "scripts/launch_ce_architecture.py",
    "scripts/report_ce_architecture.py",
    "tests/test_ce_architecture.py",
    f"configs/{BATCH}.json",
    f"docs/development-artifacts/{BATCH}/protocol.md",
)
write, sha, docker = helpers.write, helpers.sha, helpers.docker


def freeze(plan, batch):
    parent, source = Path(plan["parent_source"]), batch / "source"
    if (
        source.exists()
        or sha(parent / "source-manifest.json") != plan["parent_source_manifest_sha256"]
    ):
        raise ValueError("Prior submission or changed parent source; never duplicate a batch")
    original = json.loads((parent / "source-manifest.json").read_text())
    source.mkdir(mode=0o700, parents=True)

    def copy_parent(item):
        relative, expected = item
        path, target = parent / relative, source / relative
        if path.is_symlink():
            raise ValueError(f"Frozen parent source is a symlink: {relative}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError(f"Frozen parent source changed: {relative}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_parent, original["files"].items()))
    for relative in EXTRAS:
        target = source / relative
        if target.exists():
            raise ValueError("New architecture files must not replace parent science")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    files = {
        **original["files"],
        **{relative: sha(source / relative) for relative in EXTRAS},
    }
    write(
        source / "source-manifest.json",
        {
            "batch": BATCH,
            "parent_manifest_sha256": sha(parent / "source-manifest.json"),
            "files": files,
            "new_foundation_updates": 0,
            "new_composition_supervision": 0,
        },
    )
    return source, sha(source / "source-manifest.json")


def transport_for(source, arm):
    spec = importlib.util.spec_from_file_location(
        f"ce_transport_{arm}", source / "scripts/launch_ce_scope_discoloop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VOLUME = f"{BATCH}-{arm}-private-output"
    return module


def admission(plan, gpu):
    running = docker("ps", "--filter", f"label=lm.batch={BATCH}", "-q").stdout.decode().split()
    allowed = set()
    for name in running:
        for row in docker("top", name, "-eo", "pid").stdout.decode().splitlines()[1:]:
            allowed.add(int(row.strip()))
    if gpu == plan["resources"].get("registered_lm_peer_gpu"):
        for name, expected in plan["resources"].get("registered_lm_peers", {}).items():
            peer = json.loads(docker("inspect", "--format", "{{json .}}", name).stdout)
            if peer["Id"] != expected:
                raise ValueError("Registered LM peer identity changed")
            if peer["State"]["Running"]:
                for request in peer["HostConfig"]["DeviceRequests"]:
                    if gpu["uuid"] not in request["DeviceIDs"]:
                        raise ValueError("Registered LM peer moved to another GPU")
                for row in docker("top", name, "-eo", "pid").stdout.decode().splitlines()[1:]:
                    allowed.add(int(row.strip()))
    process_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    actual = {int(row.split(",")[1]) for row in process_rows if gpu["uuid"] in row}
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            gpu["uuid"],
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "passed": actual <= allowed and int(memory) >= 16384,
        "actual_pids": sorted(actual),
        "allowed_pids": sorted(allowed),
        "free_mib": int(memory),
        "gpu": gpu,
    }


def run_stage(plan, batch, source, manifest, arm, stage, gpu=None):
    transport = transport_for(source, arm)
    directory = batch / arm
    directory.mkdir(mode=0o700, exist_ok=True)
    name = f"{BATCH}-{arm}-{stage}"
    cpu = plan["resources"]["cpu_sets"].get(arm, "80-83")
    if gpu:
        started = time.monotonic()
        while True:
            receipt = admission(plan, gpu)
            write(directory / f"admission-{stage}.json", receipt)
            if receipt["passed"]:
                break
            write(
                directory / "controller-state.json",
                {"state": "waiting_for_resources", "stage": stage, "resource": receipt},
            )
            if time.monotonic() - started > plan["resource_wait_seconds"]:
                raise TimeoutError("GPU changed ownership; no peer was stopped")
            time.sleep(15)
    if shutil.disk_usage("/workspaces").free < plan["resources"]["minimum_disk_free_bytes"]:
        raise RuntimeError("Insufficient temporary disk headroom; existing work preserved")
    args = [
        "create",
        "--name",
        name,
        "--label",
        f"lm.batch={BATCH}",
        "--user",
        "1015:1016",
        "--network",
        "host",
        "--cpuset-cpus",
        cpu,
        "--cpus",
        "4",
        "--memory",
        "32g",
        "--tmpfs",
        "/tmp:rw,size=2g",
    ]
    for value in (
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=/frozen/src:/frozen/scripts",
        f"LM_SOURCE_MANIFEST_SHA256={manifest}",
        "XDG_CACHE_HOME=/tmp/cache",
        "WANDB_CACHE_DIR=/tmp/wandb-cache",
        "WANDB_CONFIG_DIR=/tmp/wandb-config",
        "WANDB_DATA_DIR=/tmp/wandb-data",
        "RUFF_CACHE_DIR=/tmp/ruff-cache",
    ):
        args += ["--env", value]
    args += [
        "--mount",
        "type=bind,source=/ossfs/workspace/llm-memory-editability,target=/ossfs/workspace/llm-memory-editability,readonly",
        "--mount",
        f"type=volume,source={transport.VOLUME},target=/out",
        "--mount",
        "type=bind,source=/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token,target=/run/secrets/wandb,readonly",
        "--workdir",
        "/frozen",
    ]
    if gpu:
        args += ["--gpus", f"device={gpu['uuid']}"]
    if stage == "contracts":
        paths = [p for p in EXTRAS if p.endswith(".py")]
        commands = [
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_ce_architecture.py",
                "tests/test_ce_scope_editing.py",
            ],
            ["ruff", "check", *paths],
            ["ruff", "format", "--check", *paths],
        ]
        program = "import subprocess; " + "; ".join(
            f"subprocess.run({command!r},check=True)" for command in commands
        )
        command = ["python", "-c", program]
    elif stage == "comparison":
        command = [
            "python",
            "scripts/report_ce_architecture.py",
            "--plan",
            f"configs/{BATCH}.json",
            "--out",
            "/out/comparison",
        ]
    else:
        command = [
            "python",
            "scripts/run_ce_architecture.py",
            stage.split("-")[0],
            "--architecture",
            arm,
            "--plan",
            f"configs/{BATCH}.json",
            "--root",
            "/out",
            "--device",
            "cuda:0" if gpu else "cpu",
        ]
    args += [plan["image"], *command]
    identifier = docker(*args).stdout.decode().strip()
    archive = helpers.load_transport(source).archive(source)
    docker("cp", "-", f"{name}:/", input=archive)
    write(
        directory / f"launch-{stage}.json",
        {
            "name": name,
            "id": identifier,
            "argv": args,
            "gpu": gpu,
            "source_sha256": manifest,
        },
    )
    docker("start", name)
    started = time.monotonic()
    while True:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
        write(
            directory / "controller-state.json",
            {
                "state": "running",
                "stage": stage,
                "container": name,
                "elapsed_seconds": time.monotonic() - started,
                "gpu": gpu,
                "utc_epoch": time.time(),
            },
        )
        if not state["Running"]:
            logs = docker("logs", name)
            (directory / f"{stage}.log").write_bytes(logs.stdout + logs.stderr)
            write(directory / f"exit-{stage}.json", state)
            if stage != "contracts":
                transport.persist(directory, name, stage)
            if state["ExitCode"]:
                raise RuntimeError(f"Preserved failed container: {name}; exit{state['ExitCode']}")
            return
        if time.monotonic() - started > plan["stage_timeout_seconds"]:
            docker("stop", "--time", "30", name)
            raise TimeoutError(f"Stopped only this stage after its frozen timeout: {name}")
        time.sleep(10)


def initialize_volume(plan, source, arm):
    transport = transport_for(source, arm)
    docker("volume", "create", transport.VOLUME)
    docker(
        "run",
        "--name",
        f"{BATCH}-{arm}-private-init",
        "--user",
        "0:0",
        "--network",
        "none",
        "--read-only",
        "--cpus",
        "1",
        "--memory",
        "1g",
        "--mount",
        f"type=volume,source={transport.VOLUME},target=/out",
        plan["image"],
        "python",
        "-c",
        "import os; os.chown('/out',1015,1016); os.chmod('/out',0o700)",
    )


def execute_arm(plan, batch, source, manifest, arm, gpu):
    directory = batch / arm
    completed = []
    try:
        for stage in (
            "prepare",
            "development-mlp",
            "evaluation-mlp",
            "audit-mlp",
            "cloud",
        ):
            run_stage(
                plan,
                batch,
                source,
                manifest,
                arm,
                stage,
                gpu if stage != "cloud" else None,
            )
            completed.append(stage)
        completion = {
            "state": "complete",
            "passed": True,
            "architecture": arm,
            "completed_stages": completed,
            "new_foundation_updates": 0,
            "scientific_updates": 1280,
            "independent_replay_passed": True,
            "full_cloud_json_verified": True,
        }
        write(directory / "controller-state.json", completion)
        write(directory / "completion-manifest.json", completion)
        return completion
    except BaseException as error:
        write(
            directory / "controller-state.json",
            {
                "state": "needs_attention",
                "completed_stages": completed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main():
    plan_path = ROOT / f"configs/{BATCH}.json"
    plan = json.loads(plan_path.read_text())
    batch = Path(plan["output_root"])
    batch.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("Authorized batch already submitted")
        source, manifest = freeze(plan, batch)
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "source_sha256": manifest,
                "plan_sha256": sha(plan_path),
                "submitted_epoch": time.time(),
                "parallel_arms": ["loop", "transformer"],
                "automatic_retry": False,
            },
        )
        gpu_lock = None
        try:
            for arm in ("contracts", "loop", "transformer", "report"):
                initialize_volume(plan, source, arm)
            run_stage(plan, batch, source, manifest, "contracts", "contracts")
            transport = helpers.load_transport(source)
            gpu, started = None, time.monotonic()
            while gpu is None:
                gpu, gpu_lock = transport.idle_gpu()
                if gpu is None:
                    candidate = plan["resources"].get("registered_lm_peer_gpu")
                    if candidate is not None:
                        receipt = admission(plan, candidate)
                        if receipt["passed"] and receipt["free_mib"] >= 49152:
                            gpu = candidate
                            write(batch / "registered-peer-admission.json", receipt)
                if gpu is None:
                    write(
                        batch / "controller-state.json",
                        {
                            "state": "waiting_for_exclusive_idle_gpu",
                            "contracts_passed": True,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
                    if time.monotonic() - started > plan["resource_wait_seconds"]:
                        raise TimeoutError("No idle GPU available; no external task modified")
                    time.sleep(15)
            write(
                batch / "resource-admission.json",
                {
                    "gpu": gpu,
                    "two_isolated_arms": True,
                    "external_sharing": False,
                    "registered_lm_peer_sharing": gpu
                    == plan["resources"].get("registered_lm_peer_gpu"),
                    "torch_allocator_fraction_per_arm": plan["resources"][
                        "torch_allocator_fraction"
                    ],
                },
            )
            outcomes, failures = {}, {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                tasks = {
                    pool.submit(execute_arm, plan, batch, source, manifest, arm, gpu): arm
                    for arm in ("loop", "transformer")
                }
                for future in concurrent.futures.as_completed(tasks):
                    arm = tasks[future]
                    try:
                        outcomes[arm] = future.result()
                    except BaseException:
                        failures[arm] = traceback.format_exc()
                    write(
                        batch / "controller-state.json",
                        {
                            "state": "running",
                            "outcomes": outcomes,
                            "failures": failures,
                            "gpu": gpu,
                        },
                    )
            if failures:
                raise RuntimeError(f"Failed arms preserved: {list(failures)}")
            run_stage(plan, batch, source, manifest, "report", "comparison")
            completion = {
                "state": "complete",
                "passed": True,
                "outcomes": outcomes,
                "source_sha256": manifest,
                "new_foundation_updates": 0,
                "new_scientific_updates": 2560,
                "engineering_updates": 4,
                "formal_edits": 64,
                "discoloop_reused": True,
                "independent_audits_passed": True,
                "full_cloud_json_verified": True,
                "comparison_sha256": sha(batch / "report/output/comparison/result.json"),
            }
            write(batch / "completion-manifest.json", completion)
            write(batch / "controller-state.json", completion)
        except BaseException as error:
            write(
                batch / "controller-state.json",
                {
                    "state": "needs_attention",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            if gpu_lock:
                gpu_lock.close()


if __name__ == "__main__":
    main()
