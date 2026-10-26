"""Freeze and run the authorized continuous CE experiment without repeat writes."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BATCH = "ce-cross-sample-double-new-v1"
ARCHITECTURES = ("transformer", "loop", "discoloop")
INPUT_ROOT = Path("/ossfs/workspace/llm-memory-editability")
TOKEN = Path("/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token")
DEFAULT_CPU_SETS = {"transformer": "80-83", "loop": "84-87", "discoloop": "88-91"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docker(*arguments, **kwargs):
    return subprocess.run(
        ["docker", "--context", "lm-memory", *arguments],
        check=True,
        capture_output=True,
        timeout=60,
        **kwargs,
    )


def resources(plan):
    value = plan["execution"].get("resources", {})
    return value if isinstance(value, dict) else {}


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Source path escaped its root: {value}")
    return path


def extra_files(plan):
    values = plan["execution"].get("extra_files")
    if not values or len(values) != len(set(values)):
        raise ValueError("A unique complete execution.extra_files list must be frozen")
    paths = tuple(str(safe_relative(value)) for value in values)
    required = {
        "scripts/launch_ce_cross_sample.py",
        "scripts/run_ce_cross_sample.py",
        "scripts/report_ce_cross_sample.py",
        f"configs/{BATCH}.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
        plan["panel"],
    }
    if not required <= set(paths):
        raise ValueError(f"Frozen extras omit required files: {sorted(required - set(paths))}")
    return paths


def verify_source(source, expected_manifest=None):
    manifest_path = source / "source-manifest.json"
    if expected_manifest and sha(manifest_path) != expected_manifest:
        raise ValueError("The source manifest changed after freezing")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = source / safe_relative(relative)
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    actual = {
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() and path != source / "source-manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Unmanifested files are present in frozen source")
    return sha(manifest_path)


def freeze(plan, batch, *, repository=ROOT):
    execution = plan["execution"]
    parent = Path(execution.get("parent_source", plan.get("parent_source", "")))
    expected_parent = execution.get(
        "parent_source_manifest_sha256", plan.get("parent_source_manifest_sha256")
    )
    if not expected_parent or sha(parent / "source-manifest.json") != expected_parent:
        raise ValueError("The original parent manifest must match the authorized source")
    extras = extra_files(plan)
    revision = execution.get("source_revision", "source")
    if not isinstance(revision, str) or not re.fullmatch(r"source(?:-v[1-9][0-9]*)?", revision):
        raise ValueError("Invalid immutable source revision")
    source = batch / revision
    receipt_path = batch / (
        "freeze-receipt.json" if revision == "source" else f"freeze-receipt-{revision}.json"
    )
    if source.exists():
        receipt = json.loads(receipt_path.read_text())
        manifest_sha = verify_source(source, receipt["source_sha256"])
        if receipt["parent_manifest_sha256"] != expected_parent:
            raise ValueError("A frozen batch cannot change its parent source")
        for relative in extras:
            if sha(repository / relative) != sha(source / relative):
                raise ValueError(f"Working file differs from already frozen source: {relative}")
        return source, manifest_sha
    original = json.loads((parent / "source-manifest.json").read_text())
    if set(extras) & set(original["files"]):
        raise ValueError("New files must not replace any frozen parent code")
    expected_count = execution.get("parent_source_file_count")
    if expected_count is not None and len(original["files"]) != expected_count:
        raise ValueError("Parent source file count changed")
    source.mkdir(mode=0o700, parents=True)

    def copy_parent(item):
        relative, expected = item
        path = parent / safe_relative(relative)
        if path.is_symlink() or sha(path) != expected:
            raise ValueError(f"Original frozen parent changed: {relative}")
        target = source / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != expected:
            raise ValueError(f"Copied parent source changed: {relative}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy_parent, original["files"].items()))
    for relative in extras:
        origin, target = repository / relative, source / relative
        if origin.is_symlink() or not origin.is_file():
            raise ValueError(f"New source must be a regular file: {relative}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    write(
        source / "source-manifest.json",
        {
            "batch": BATCH,
            "parent_manifest_sha256": expected_parent,
            "parent_file_count": len(original["files"]),
            "extra_files": extras,
            "files": {
                **original["files"],
                **{relative: sha(source / relative) for relative in extras},
            },
            "new_foundation_updates": 0,
            "new_composition_supervision": 0,
        },
    )
    manifest_sha = verify_source(source)
    write(
        receipt_path,
        {
            "source_sha256": manifest_sha,
            "parent_manifest_sha256": expected_parent,
            "source": str(source),
            "source_revision": revision,
            "utc_epoch": time.time(),
        },
    )
    return source, manifest_sha


def archive(source):
    """Transport only validated files; container source is root-owned and read-only."""
    verify_source(source)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as stream:
        paths = [Path("."), *sorted(source.rglob("*"))]
        for path in paths:
            relative = (
                Path("frozen") if path == Path(".") else Path("frozen") / path.relative_to(source)
            )
            member = tarfile.TarInfo(str(relative))
            member.uid = member.gid = 0
            member.uname = member.gname = "root"
            if path == Path(".") or path.is_dir():
                member.type, member.mode = tarfile.DIRTYPE, 0o555
                stream.addfile(member)
            elif path.is_file() and not path.is_symlink():
                data = path.read_bytes()
                member.mode, member.size = 0o444, len(data)
                stream.addfile(member, io.BytesIO(data))
            else:
                raise ValueError("Unsupported object in frozen source")
    return buffer.getvalue()


def gpu_inventory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pids = {}
    for line in processes:
        if line.strip():
            uuid, pid = (value.strip() for value in line.split(","))
            pids.setdefault(uuid, []).append(int(pid))
    inventory = []
    for line in result.stdout.splitlines():
        index, uuid, used, free, utilization = (value.strip() for value in line.split(","))
        inventory.append(
            {
                "index": int(index),
                "uuid": uuid,
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_percent": int(utilization),
                "compute_pids": pids.get(uuid, []),
            },
        )
    return inventory


def idle(gpu, plan):
    return (
        not gpu["compute_pids"]
        and gpu["used_mib"] <= 1024
        and gpu["utilization_percent"] <= 5
        and gpu["free_mib"] >= resources(plan).get("minimum_gpu_free_mib", 16384)
    )


def acquire_gpu(plan):
    inventory = gpu_inventory()
    preferred = resources(plan).get("preferred_gpu_indices", [4, 6, 9])
    inventory.sort(key=lambda gpu: (gpu["index"] not in preferred, gpu["index"]))
    for gpu in inventory:
        if not idle(gpu, plan):
            continue
        directory = Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}")
        directory.mkdir(mode=0o700, exist_ok=True)
        lock = (directory / f"gpu-{gpu['index']}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # A process could have appeared between the first snapshot and lock admission.
            actual = next(row for row in gpu_inventory() if row["uuid"] == gpu["uuid"])
            if idle(actual, plan):
                return actual, lock
        except BlockingIOError:
            pass
        except BaseException:
            lock.close()
            raise
        lock.close()
    return None, None


def validate_batch(batch):
    expected = INPUT_ROOT.resolve() / BATCH
    if batch.name != BATCH or batch.resolve() != expected:
        raise ValueError("Persistent batch path escaped its authorized project location")


def runtime_batch(plan):
    validate_batch(Path(plan["execution"]["persistent_root"]))
    return Path("/persist") / BATCH


def private_output(batch, arm):
    validate_batch(batch)
    if arm not in (*ARCHITECTURES, "contracts", "report"):
        raise ValueError("Output architecture escaped the authorized batch")
    output = batch / arm / "output"
    if output.resolve() != batch.resolve() / arm / "output":
        raise ValueError("Output symlink escaped the authorized batch")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = output.stat()
    if (identity.st_uid, identity.st_gid) != (1015, 1016):
        raise ValueError("Persistent output must belong to runtime UID 1015:GID 1016")
    if identity.st_mode & 0o077:
        raise ValueError("Persistent output permissions must be 0700")
    return output


def command_for(plan, stage, arm):
    if arm not in (*ARCHITECTURES, "contracts", "report"):
        raise ValueError("Output architecture escaped the authorized batch")
    if stage == "contracts":
        paths = [path for path in extra_files(plan) if path.endswith(".py")]
        tests = plan["execution"].get(
            "contract_tests", [path for path in paths if path.startswith("tests/")]
        )
        if not tests:
            raise ValueError("Frozen contract tests are required")
        commands = [
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
            ["ruff", "check", *paths],
            ["ruff", "format", "--check", *paths],
        ]
        program = "import subprocess; " + "; ".join(
            f"subprocess.run({command!r},check=True)" for command in commands
        )
        return ["python", "-c", program]
    if stage == "report":
        return [
            "python",
            "scripts/report_ce_cross_sample.py",
            "--plan",
            f"configs/{BATCH}.json",
            "--batch",
            str(runtime_batch(plan)),
            "--out",
            str(runtime_batch(plan) / "report/output"),
        ]
    return [
        "python",
        "scripts/run_ce_cross_sample.py",
        stage,
        "--architecture",
        arm,
        "--plan",
        f"configs/{BATCH}.json",
        "--root",
        str(runtime_batch(plan) / arm / "output"),
        "--device",
        "cpu" if stage == "cloud" else "cuda:0",
    ]


def container_arguments(plan, batch, manifest, arm, stage, gpu):
    name = f"{BATCH}-{arm}-{stage}"
    private_output(batch, arm)
    cpu = resources(plan).get("cpu_sets", DEFAULT_CPU_SETS).get(arm, "80-83")
    args = [
        "create",
        "--name",
        name,
        "--label",
        f"lm.batch={BATCH}",
        "--label",
        f"lm.architecture={arm}",
        "--label",
        f"lm.stage={stage}",
        "--user",
        "1015:1016",
        "--network",
        "host",
        "--cpuset-cpus",
        cpu,
        "--cpus",
        "4",
        "--memory",
        resources(plan).get("memory", "32g"),
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
        f"type=bind,source={INPUT_ROOT},target={INPUT_ROOT},readonly",
        "--mount",
        f"type=bind,source={INPUT_ROOT},target=/persist",
        "--mount",
        f"type=bind,source={TOKEN},target=/run/secrets/wandb,readonly",
        "--workdir",
        "/frozen",
    ]
    if gpu:
        args += ["--gpus", f"device={gpu['uuid']}"]
    return name, [*args, plan["execution"]["image"], *command_for(plan, stage, arm)]


def completed_stage(directory, stage, manifest):
    exit_path = directory / f"exit-{stage}.json"
    launch_path = directory / f"launch-{stage}.json"
    if not launch_path.exists() and not exit_path.exists():
        return False
    if launch_path.exists() and exit_path.exists():
        launch, state = json.loads(launch_path.read_text()), json.loads(exit_path.read_text())
        if launch["source_sha256"] == manifest and state["ExitCode"] == 0:
            return True
    raise FileExistsError(f"Stage has existing or failed evidence; never repeat it: {stage}")


def run_stage(plan, batch, source, manifest, arm, stage, *, gpu=None, timeout=None):
    directory = batch / arm
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if completed_stage(directory, stage, manifest):
        return json.loads((directory / f"duration-{stage}.json").read_text())["elapsed_seconds"]
    if timeout is not None and timeout <= 0:
        raise TimeoutError("The aggregate architecture GPU budget is exhausted")
    if gpu:
        waiting = time.monotonic()
        while True:
            actual = next(row for row in gpu_inventory() if row["uuid"] == gpu["uuid"])
            receipt = {"passed": idle(actual, plan), "gpu": actual, "utc_epoch": time.time()}
            write(directory / f"admission-{stage}.json", receipt)
            if receipt["passed"]:
                break
            write(
                directory / "controller-state.json", {"state": "waiting_for_resources", **receipt}
            )
            if time.monotonic() - waiting > resources(plan).get("resource_wait_seconds", 14400):
                raise TimeoutError("GPU acquired external work; no process was preempted")
            time.sleep(15)
    verify_source(source, manifest)
    name, args = container_arguments(plan, batch, manifest, arm, stage, gpu)
    identifier = docker(*args).stdout.decode().strip()
    write(
        directory / f"launch-{stage}.json",
        {
            "name": name,
            "id": identifier,
            "argv": args,
            "source_sha256": manifest,
            "gpu": gpu,
            "created_epoch": time.time(),
            "automatic_retry": False,
        },
    )
    payload = archive(source)
    docker("cp", "-", f"{name}:/", input=payload)
    write(
        directory / f"source-deployment-{stage}.json",
        {
            "source_sha256": manifest,
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
    )
    docker("start", name)
    started = time.monotonic()
    deadline = (
        timeout
        if timeout is not None
        else plan["execution"].get("cpu_stage_timeout_seconds", 14400)
    )
    timed_out = False
    while True:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
        elapsed = time.monotonic() - started
        write(
            directory / "controller-state.json",
            {
                "state": "running",
                "stage": stage,
                "container": name,
                "container_state": state,
                "elapsed_seconds": elapsed,
                "gpu": gpu,
                "utc_epoch": time.time(),
            },
        )
        if not state["Running"]:
            logs = docker("logs", name)
            (directory / f"{stage}.log").write_bytes(logs.stdout + logs.stderr)
            write(directory / f"exit-{stage}.json", state)
            write(
                directory / f"duration-{stage}.json",
                {"elapsed_seconds": elapsed, "timed_out": timed_out},
            )
            if timed_out:
                raise TimeoutError(f"Stopped only this batch container after timeout: {name}")
            if state["ExitCode"]:
                raise RuntimeError(
                    f"Failed container preserved without retry: {name}; exit {state['ExitCode']}"
                )
            return elapsed
        if elapsed > deadline:
            docker("stop", "--time", "30", name)
            timed_out = True
        time.sleep(10)


def execute_arm(plan, batch, source, manifest, arm):
    directory = batch / arm
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    completed, gpu, gpu_lock = [], None, None
    waiting = time.monotonic()
    try:
        while gpu is None:
            gpu, gpu_lock = acquire_gpu(plan)
            if gpu is None:
                write(
                    directory / "controller-state.json", {"state": "waiting_for_exclusive_idle_gpu"}
                )
                if time.monotonic() - waiting > resources(plan).get("resource_wait_seconds", 14400):
                    raise TimeoutError("No idle GPU available; external tasks were preserved")
                time.sleep(15)
        write(directory / "resource-admission.json", {"gpu": gpu, "exclusive_gpu": True})
        remaining = plan["execution"]["timeout_seconds_per_architecture"]
        durations = {}
        for stage in ("prepare", "evaluate", "audit"):
            durations[stage] = run_stage(
                plan, batch, source, manifest, arm, stage, gpu=gpu, timeout=remaining
            )
            remaining -= durations[stage]
            completed.append(stage)
        gpu_lock.close()
        gpu_lock = None
        run_stage(plan, batch, source, manifest, arm, "cloud")
        completed.append("cloud")
        completion = {
            "state": "complete",
            "passed": True,
            "architecture": arm,
            "completed_stages": completed,
            "scientific_updates": plan["budget"]["new_scientific_updates_per_architecture"],
            "new_foundation_updates": 0,
            "engineering_updates": plan["execution"].get("engineering_updates_per_architecture", 4),
            "independent_replay_passed": True,
            "full_cloud_json_verified": True,
            "gpu_durations_seconds": durations,
            "gpu": gpu,
            "source_sha256": manifest,
        }
        write(directory / "completion-manifest.json", completion)
        write(directory / "controller-state.json", completion)
        return completion
    except BaseException as error:
        write(
            directory / "controller-state.json",
            {
                "state": "needs_attention",
                "completed_stages": completed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
            },
        )
        raise
    finally:
        if gpu_lock is not None:
            gpu_lock.close()


def preflight(plan, batch):
    if plan["batch"] != BATCH or tuple(plan["architectures"]) != ARCHITECTURES:
        raise ValueError("The authorized matrix must contain exactly the three architectures")
    if plan["budget"]["new_scientific_updates_total"] != 13440:
        raise ValueError("The authorized total write budget changed")
    if plan["budget"]["new_scientific_updates_per_architecture"] != 4480:
        raise ValueError("The authorized architecture write budget changed")
    validate_batch(batch)
    minimum = plan["execution"]["minimum_free_persistent_bytes"]
    free = shutil.disk_usage(batch).free
    if free < minimum:
        raise RuntimeError(f"Insufficient persistent storage: {free} available, {minimum} required")
    minimum_temporary = resources(plan).get("minimum_free_temporary_bytes", 2 * 1024**3)
    if shutil.disk_usage("/workspaces").free < minimum_temporary:
        raise RuntimeError("Insufficient temporary disk headroom")
    selected = set()
    for arm in ARCHITECTURES:
        cpu_set = resources(plan).get("cpu_sets", DEFAULT_CPU_SETS)[arm]
        cpus = set()
        for segment in cpu_set.split(","):
            start, _, finish = segment.partition("-")
            cpus.update(range(int(start), int(finish or start) + 1))
        if selected & cpus or not cpus <= os.sched_getaffinity(0):
            raise ValueError("Architecture CPU sets overlap or are unavailable")
        selected |= cpus
    if not TOKEN.is_file():
        raise FileNotFoundError("The existing read-only W&B token mount is missing")
    write(
        batch / "preflight.json",
        {"persistent_free_bytes": free, "cpu_sets": sorted(selected), "utc_epoch": time.time()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / f"configs/{BATCH}.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--contracts-only", action="store_true")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    batch = Path(plan["execution"]["persistent_root"])
    batch.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("Authorized science already submitted; inspect retained state")
        source, manifest = freeze(plan, batch)
        if args.freeze_only:
            return
        preflight(plan, batch)
        run_stage(plan, batch, source, manifest, "contracts", "contracts")
        write(batch / "contracts.json", {"passed": True, "source_sha256": manifest})
        if args.contracts_only:
            return
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "source_sha256": manifest,
                "source_revision": source.name,
                "plan_sha256": sha(plan_path),
                "submitted_epoch": time.time(),
                "parallel_architectures": ARCHITECTURES,
                "automatic_retry": False,
            },
        )
        outcomes, failures = {}, {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                tasks = {
                    pool.submit(execute_arm, plan, batch, source, manifest, arm): arm
                    for arm in ARCHITECTURES
                }
                for future in concurrent.futures.as_completed(tasks):
                    arm = tasks[future]
                    try:
                        outcomes[arm] = future.result()
                    except BaseException:
                        failures[arm] = traceback.format_exc()
                    write(
                        batch / "controller-state.json",
                        {"state": "running", "outcomes": outcomes, "failures": failures},
                    )
            if failures:
                raise RuntimeError(f"Failed architectures preserved: {sorted(failures)}")
            run_stage(plan, batch, source, manifest, "report", "report")
            verification_path = batch / "report/output/cloud-verification.json"
            verification = json.loads(verification_path.read_text())
            if verification.get("passed") is not True:
                raise ValueError("The report cloud verification did not pass")
            completion = {
                "state": "complete",
                "passed": True,
                "outcomes": outcomes,
                "source_sha256": manifest,
                "new_foundation_updates": 0,
                "new_scientific_updates": plan["budget"]["new_scientific_updates_total"],
                "engineering_updates": sum(
                    item["engineering_updates"] for item in outcomes.values()
                ),
                "independent_audits_passed": True,
                "full_cloud_json_verified": True,
                "report_cloud_verification_sha256": sha(verification_path),
                "report_output": str(batch / "report/output"),
            }
            write(batch / "completion-manifest.json", completion)
            write(batch / "controller-state.json", completion)
        except BaseException as error:
            write(
                batch / "controller-state.json",
                {
                    "state": "needs_attention",
                    "outcomes": outcomes,
                    "failures": failures,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
