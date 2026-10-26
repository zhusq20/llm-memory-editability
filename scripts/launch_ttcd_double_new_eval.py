#!/usr/bin/env python3
"""Queue supplementary two-new-fact evaluation without any new training.

Default invocation is a read-only plan. Explicit execution copies the parent's
exact repaired source, prepares a graph-only panel on CPU, and waits for both
original arms' local and cloud completion before independently using idle GPUs.
Failed stages are preserved and never automatically retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ["docker", "--context", "lm-memory"]
BATCH = "ttcd-double-new-eval-v1"
PARENT = "ttcd-successful-base-v1"
ARMS = ("single", "stream")
STAGES = ("preflight", "evaluate", "audit", "cloud-audit")
PARENT_STAGES = ("train", "train-audit", "calibrate", "evaluate", "audit", "cloud-audit")
PREFERENCES = {"single": [4, 7], "stream": [7, 4]}
PROTOCOL = "docs/development-artifacts/ttcd-double-new-eval-v1/protocol.md"
OVERLAYS = (
    "src/llm_memory_editability/ttcd_double_new_eval.py",
    "scripts/run_ttcd_double_new_eval.py",
    "scripts/launch_ttcd_double_new_eval.py",
    "scripts/reschedule_ttcd_verified.py",
    "tests/test_ttcd_double_new_controller.py",
    "docs/development-artifacts/ttcd-double-new-eval-v1/verify_cuda.py",
    PROTOCOL,
)


class DependencyPending(RuntimeError):
    """A missing original completion receipt can be waited for read-only."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_module(path, name):
    """Import resource helpers without creating bytecode in frozen parents."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_resources(parent, batch):
    controller = load_module(
        parent / "source-v2/scripts/launch_discoloop_paired.py", "double_new_parent_resources"
    )
    policy = load_module(
        batch / "source/scripts/reschedule_ttcd_verified.py", "double_new_idle_policy"
    )
    return controller, policy


def validate_settings(settings):
    if settings["batch"] != BATCH or settings["parent_batch"] != PARENT:
        raise ValueError(
            "This controller only adds the authorized evaluation to its unchanged parent"
        )
    if (
        settings["pair_seed"] != 20261015
        or settings["pairs_per_type"] != 4
        or settings["rotations"] != 32
    ):
        raise ValueError("Preserve the predeclared graph-only pair selection and order budget")
    runtime = settings["runtime"]
    if runtime.get("docker_context", "lm-memory") != "lm-memory":
        raise ValueError("Use only the explicit LM Docker context")
    if not runtime["image"].startswith("sha256:") or len(runtime["image"]) != 71:
        raise ValueError("Pin the exact parent dependency image")
    if runtime.get("tracking", "online") != "online":
        raise ValueError("Formal supplementary evaluation requires online tracking")
    if runtime["memory_gib"] != 96 or runtime["cpu_threads"] != 4:
        raise ValueError("Use the reviewed96GiB and four-CPU evaluation allocation")
    expected = {"single": list(range(32, 36)), "stream": list(range(36, 40))}
    if runtime["cpu_sets"] != expected:
        raise ValueError("Evaluation CPU sets must stay separate from the parent's24–31")
    if runtime.get("shared_gpu_allowlist"):
        raise ValueError("Supplementary evaluation does not authorize GPU sharing")
    if runtime.get("uid", 1015) != 1015 or runtime.get("gid", 1016) != 1016:
        raise ValueError("Persistent LM ownership remains uid1015:gid1016")
    if min(runtime["timeout_hours"], runtime["wait_timeout_hours"]) <= 0:
        raise ValueError("Positive execution and waiting timeouts are required")
    if minimum_space(settings) < 200:
        raise ValueError("Require at least200GiB persistent free space")


def minimum_space(settings):
    runtime = settings["runtime"]
    return runtime.get(
        "minimum_persistent_free_gib", runtime.get("minspace_gib", runtime.get("minspace", 200))
    )


def persistent_batch(settings):
    root = Path(settings["runtime"]["persistent_root"])
    if not root.is_dir() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise RuntimeError("The persistent LM mount is unavailable")
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Formal LM outputs must stay below /ossfs/workspace")
    batch = root / BATCH
    if batch.is_symlink():
        raise ValueError("Supplementary outputs must not be redirected by a symlink")
    return batch


def safe_source(source, relative):
    path = source / relative
    if not path.resolve().is_relative_to(source.resolve()):
        raise ValueError("A frozen manifest path escaped its source directory")
    return path


def verify_parent(parent, expected_manifest_sha256=None):
    manifest_path = parent / "source-v2-manifest.json"
    if expected_manifest_sha256 and sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Parent repaired source manifest changed")
    manifest = read_json(manifest_path)
    source = parent / "source-v2"
    if sha256(source / "configs/experiment.json") != manifest["config_sha256"]:
        raise ValueError("The original parent configuration changed")
    for relative, expected in manifest["files"].items():
        if sha256(safe_source(source, relative)) != expected:
            raise ValueError(f"Parent repaired source changed: {relative}")
    for relative, expected in manifest["inputs"].items():
        if sha256(safe_source(parent / "input-mount", relative)) != expected:
            raise ValueError(f"Parent immutable input changed: {relative}")
    return manifest


def freeze_source(batch, parent, settings_path):
    if batch.resolve() == parent.resolve() or batch.is_symlink():
        raise ValueError("Supplementary freezing must use its own real batch directory")
    batch.chmod(0o755)
    source, output = batch / "source", batch / "source-manifest.json"
    if output.exists():
        frozen = read_json(output)
        if frozen["settings_sha256"] != sha256(settings_path):
            raise ValueError("Supplementary settings changed after freezing")
        verify_parent(parent, frozen["parent_source_v2_manifest_sha256"])
        for relative, expected in frozen["files"].items():
            if sha256(safe_source(source, relative)) != expected:
                raise ValueError(f"Supplementary frozen source changed: {relative}")
        return frozen
    parent_manifest = verify_parent(parent)
    settings = read_json(settings_path)
    if settings.get("parent_source_v2_manifest_sha256") and settings[
        "parent_source_v2_manifest_sha256"
    ] != sha256(parent / "source-v2-manifest.json"):
        raise ValueError("Parent source differs from the reviewed supplementary settings")
    base_config = read_json(parent / "source-v2/configs/experiment.json")
    if settings["runtime"]["image"] != base_config["runtime"]["image"]:
        raise ValueError("Supplementary evaluation must use the unchanged parent image")
    additions = list(OVERLAYS)
    additions += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "tests").glob("test_ttcd_double_new_eval*.py"))
    ]
    if any(relative in parent_manifest["files"] for relative in additions):
        raise ValueError("New supplementary files must not overwrite any existing science")
    source.mkdir(exist_ok=False)
    for relative in parent_manifest["files"]:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(safe_source(parent / "source-v2", relative), destination)
    for relative in additions:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for name in ("double-new-eval.json", f"{BATCH}.json"):
        destination = source / "configs" / name
        if destination.exists():
            raise ValueError("Supplementary settings cannot overwrite a parent file")
        shutil.copy2(settings_path, destination)
    # Check every original byte after the overlay, including identity CODE_FILES.
    for relative, expected in parent_manifest["files"].items():
        if sha256(source / relative) != expected:
            raise ValueError(f"A parent scientific byte changed during copying: {relative}")
    source.chmod(0o755)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    shutil.copy2(parent / "source-v2-manifest.json", batch / "parent-source-v2-manifest.json")
    frozen = {
        "schema": "ttcd-double-new-evaluation-freeze-v1",
        "parent_batch": PARENT,
        "parent_source_v2_manifest_sha256": sha256(parent / "source-v2-manifest.json"),
        "parent_config_sha256": parent_manifest["config_sha256"],
        "settings_sha256": sha256(settings_path),
        "original_files": parent_manifest["files"],
        "overlays": additions,
        "new_base_parameter_updates": 0,
        "protocol": {
            "sha256": sha256(source / PROTOCOL),
            "bytes": (source / PROTOCOL).stat().st_size,
        },
        "files": {
            str(p.relative_to(source)): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()
        },
    }
    write_json(output, frozen)
    return frozen


def verify_snapshot(batch, parent, panel_required=False):
    frozen = read_json(batch / "source-manifest.json")
    verify_parent(parent, frozen["parent_source_v2_manifest_sha256"])
    for relative, expected in frozen["files"].items():
        if sha256(safe_source(batch / "source", relative)) != expected:
            raise ValueError(f"Supplementary source changed: {relative}")
    if panel_required:
        panel = read_json(batch / "panel-manifest.json")
        if panel["settings_sha256"] != frozen["settings_sha256"] or panel["panel_sha256"] != sha256(
            batch / "panel.json"
        ):
            raise ValueError("The graph-only supplementary panel changed")


def result_path(parent, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[0] != "results":
        raise ValueError("Parent checkpoints must retain their original relative result paths")
    return parent.parent / Path(*path.parts[1:])


def dependency_receipt(parent, base_config, arm):
    directory = parent / arm
    common_gate = result_path(parent, base_config["common"]["gate_receipt"])
    common_checkpoint = result_path(parent, base_config["common"]["checkpoint"])
    checkpoint = result_path(parent, base_config["models"][arm]["checkpoint"])
    paths = [
        common_gate,
        common_checkpoint,
        checkpoint,
        directory / "audit.json",
        directory / "result-cloud-audit.json",
    ]
    for stage in PARENT_STAGES:
        path = directory / f"exit-{stage}.json"
        paths.append(path)
        if path.is_file() and read_json(path).get("returncode") != 0:
            raise RuntimeError(
                f"Original {arm}/{stage} failed; supplementary evaluation cannot proceed"
            )
    for path in (common_gate, directory / "audit.json", directory / "result-cloud-audit.json"):
        if path.is_file() and read_json(path).get("passed") is not True:
            raise RuntimeError(f"Required original gate/audit failed: {path.name}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise DependencyPending(f"Waiting for original {arm} local and cloud completion: {missing}")
    if read_json(common_gate)["checkpoint_sha256"] != sha256(common_checkpoint):
        raise ValueError("The passing common gate binds another checkpoint")
    return {
        "arm": arm,
        "passed": True,
        "files": {str(path): sha256(path) for path in paths},
        "new_base_parameter_updates": 0,
    }


def resource_config(settings):
    runtime = dict(settings["runtime"])
    runtime.setdefault("gpu_memory_max_mib", 1024)
    runtime.setdefault("gpu_utilization_max", 5)
    return {"runtime": runtime}


def wait_dependencies(parent, base_config, arm, deadline, update=lambda _: None):
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Original completion wait expired; no parent job was changed")
        try:
            receipts, pending = {}, []
            for name in ARMS:
                try:
                    receipts[name] = dependency_receipt(parent, base_config, name)
                except DependencyPending as error:
                    pending.append(str(error))
            if pending:
                raise DependencyPending("; ".join(pending))
            return receipts[arm]
        except DependencyPending as error:
            update({"state": "waiting_for_parent_completion", "reason": str(error)})
            time.sleep(min(30, max(0, deadline - time.monotonic())))


def admit_gpu(settings, arm, gpu, helpers, policy):
    config = resource_config(settings)
    containers, available = helpers.admit_cpu(config, arm)
    rows, processes = helpers.gpu_snapshot()
    row = policy.idle_row(helpers, config, arm, gpu, rows, processes, containers, available)
    return {**row, "resource_amendment": BATCH, "new_base_parameter_updates": 0}


def wait_gpu(settings, arm, helpers, policy, deadline, update=lambda _: None):
    directory = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    while time.monotonic() < deadline:
        rows, _ = helpers.gpu_snapshot()
        choices = list(dict.fromkeys([*PREFERENCES[arm], *sorted(rows)]))
        reasons = {}
        for gpu in choices:
            held = (directory / f"gpu-{gpu}.lock").open("a")
            try:
                try:
                    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise helpers.ResourceUnavailable("Exclusive GPU lock held") from error
                return held, admit_gpu(settings, arm, gpu, helpers, policy)
            except helpers.ResourceUnavailable as error:
                held.close()
                reasons[str(gpu)] = str(error)
            except BaseException:
                held.close()
                raise
        update({"state": "waiting_for_resources", "reasons": reasons})
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("No exclusively idle GPU before the supplementary resource deadline")


def build_command(settings, arm, stage, batch, parent, token_file, row=None):
    preparation = stage == "prepare"
    name = f"{BATCH}-{'preparation' if preparation else arm}-{stage}"
    out = batch / ("preparation" if preparation else arm)
    runtime = settings["runtime"]
    cpus = runtime["cpu_sets"][arm]
    command = [
        *DOCKER,
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        str(runtime["cpu_threads"]),
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        f"{runtime['memory_gib']}g",
        "--shm-size",
        "8g",
        "--network",
        "host",
        "--workdir",
        "/project",
    ]
    if not preparation:
        command += [
            "--gpus",
            f"device={row['uuid']}",
            "--env",
            f"LM_ACTUAL_GPU_INDEX={row['gpu']}",
            "--env",
            f"LM_ACTUAL_GPU_UUID={row['uuid']}",
        ]
    for host, target, readonly in [
        (ROOT, "/project", True),
        (batch.parent, "/project/results", True),
        (batch, f"/project/results/{BATCH}", not preparation),
        (batch / "source", "/frozen", True),
        (batch / "source", f"/project/results/{BATCH}/source", True),
        (parent / "input-mount/data", "/project/data", True),
    ]:
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    if not preparation:
        command += ["--mount", f"type=bind,source={out},target=/project/results/{BATCH}/{arm}"]
    command += [
        "--tmpfs",
        "/tmp:rw,size=8g",
        "--env",
        "PYTHONPATH=/frozen/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "HOME=/tmp/lm-home",
        "--env",
        "XDG_CACHE_HOME=/tmp/lm-home/.cache",
        "--env",
        "XDG_CONFIG_HOME=/tmp/lm-home/.config",
        "--env",
        f"OMP_NUM_THREADS={runtime['cpu_threads']}",
        "--env",
        f"MKL_NUM_THREADS={runtime['cpu_threads']}",
    ]
    if token_file.is_file():
        command += [
            "--mount",
            f"type=bind,source={token_file.resolve()},target=/run/secrets/wandb,readonly",
        ]
    command += [
        runtime["image"],
        "bash",
        "-c",
        'mkdir -p "$HOME"\nif [ -f /run/secrets/wandb ]; then\n'
        'export WANDB_API_KEY="$(cat /run/secrets/wandb)"; fi\nexec "$@"',
        "ttcd-double-new-entry",
        "python",
        "/frozen/scripts/run_ttcd_double_new_eval.py",
        stage,
        "--base-config",
        "/frozen/configs/experiment.json",
        "--settings",
        "/frozen/configs/double-new-eval.json",
        "--model",
        arm,
        "--out",
        f"/project/results/{BATCH}/{'preparation' if preparation else arm}",
        "--panel",
        f"/project/results/{BATCH}/panel.json",
        "--tracking",
        "online",
        "--device",
        "cpu" if preparation else "cuda:0",
    ]
    return name, command


def run_stage(
    settings,
    arm,
    stage,
    batch,
    parent,
    token_file,
    helpers,
    deadline,
    row=None,
    update=lambda _: None,
):
    out = batch / ("preparation" if stage == "prepare" else arm)
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o755)
    verify_snapshot(batch, parent, panel_required=stage != "prepare")
    if stage in STAGES[1:]:
        result = out / "result-preflight.json"
        exit_path = out / "exit-preflight.json"
        if (
            not result.is_file()
            or read_json(result).get("passed") is not True
            or not exit_path.is_file()
            or read_json(exit_path).get("returncode") != 0
        ):
            raise ValueError("Formal evaluation requires a passing development CUDA preflight")
    with (out / f"claim-{stage}.json").open("x") as claim:
        json.dump({"pid": os.getpid(), "stage": stage, "arm": arm, "utc": utc_now()}, claim)
    name, command = build_command(settings, arm, stage, batch, parent, token_file, row)
    if helpers.inspect_container(name) is not None:
        raise FileExistsError("The supplementary stage container exists; no automatic retry")
    write_json(out / f"command-{stage}.json", {"argv": command, "utc": utc_now()})
    write_json(
        out / f"admission-{stage}.json",
        row or {"device": "cpu", "cpu_set": settings["runtime"]["cpu_sets"][arm]},
    )
    update({"state": "launching", "stage": stage, "container": name})
    with (out / f"{stage}.log").open("x") as log:
        launched = subprocess.run(command, capture_output=True, text=True, timeout=60)
        log.write(launched.stderr)
        log.flush()
    if launched.returncode:
        write_json(
            out / f"exit-{stage}.json", {"returncode": launched.returncode, "launch_failed": True}
        )
        raise RuntimeError(f"Supplementary stage launch failed: {launched.stderr.strip()}")
    update(
        {
            "state": "running",
            "stage": stage,
            "container": name,
            "container_id": launched.stdout.strip(),
        }
    )
    with (out / f"{stage}.log").open("a") as log:
        container_code = helpers.monitor_container(name, log, deadline, update)
    code = container_code
    if code == 0 and stage in ("prepare", "preflight", "audit", "cloud-audit"):
        result = out / f"result-{stage}.json"
        if not result.is_file() or read_json(result).get("passed") is not True:
            code = 3
    write_json(
        out / f"exit-{stage}.json",
        {
            "returncode": code,
            "container_returncode": container_code,
            "container": name,
            "utc": utc_now(),
        },
    )
    return code


def record_panel(batch):
    result = read_json(batch / "preparation/result-prepare.json")
    if result.get("passed") is not True:
        raise ValueError("Graph-only preparation did not pass")
    frozen = read_json(batch / "source-manifest.json")
    write_json(
        batch / "panel-manifest.json",
        {
            "panel_sha256": sha256(batch / "panel.json"),
            "settings_sha256": frozen["settings_sha256"],
            "new_base_parameter_updates": 0,
            "bytes": (batch / "panel.json").stat().st_size,
            "utc": utc_now(),
        },
    )


def run_arm(settings, arm, stages, batch, parent, token_file, helpers, policy, base_config):
    out = batch / arm
    out.mkdir(exist_ok=True)
    out.chmod(0o755)
    state = {
        "arm": arm,
        "stages": stages,
        "new_base_parameter_updates": 0,
        "automatic_retry": False,
    }

    def update(value):
        state.update(value)
        write_json(out / "controller-state.json", {**state, "updated_utc": utc_now()})

    try:
        with (out / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            wait_deadline = time.monotonic() + settings["runtime"]["wait_timeout_hours"] * 3600
            receipt = wait_dependencies(parent, base_config, arm, wait_deadline, update)
            parent_receipts = {name: dependency_receipt(parent, base_config, name) for name in ARMS}
            if parent_receipts[arm] != receipt:
                raise ValueError("Original parent completion identity changed")
            write_json(out / "parent-completion-receipt.json", receipt)
            write_json(out / "all-parent-completion-receipts.json", parent_receipts)
            held, row = wait_gpu(settings, arm, helpers, policy, wait_deadline, update)
            with held:
                deadline = time.monotonic() + settings["runtime"]["timeout_hours"] * 3600
                for index, stage in enumerate(stages):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Supplementary hard timeout before next stage")
                    if any(
                        dependency_receipt(parent, base_config, name) != parent_receipts[name]
                        for name in ARMS
                    ):
                        raise ValueError("Original parent completion identity changed")
                    if index:
                        row = admit_gpu(settings, arm, row["gpu"], helpers, policy)
                    code = run_stage(
                        settings,
                        arm,
                        stage,
                        batch,
                        parent,
                        token_file,
                        helpers,
                        deadline,
                        row,
                        update,
                    )
                    if code:
                        update({"state": "needs_attention", "stage": stage, "returncode": code})
                        return code
            update({"state": "complete", "finished_utc": utc_now()})
            return 0
    except Exception:
        failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
        update(failure)
        write_json(out / "controller-failure.json", failure)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("pipeline", "prepare", *STAGES), nargs="?", default="pipeline"
    )
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--model", choices=(*ARMS, "both"), default="both")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    settings = read_json(args.config)
    validate_settings(settings)
    arms = list(ARMS) if args.model == "both" else [args.model]
    plan = {
        "batch": BATCH,
        "parent_batch": PARENT,
        "executed": args.execute,
        "docker_context": "lm-memory",
        "image": settings["runtime"]["image"],
        "prepare_device": "cpu",
        "stages": list(STAGES) if args.stage == "pipeline" else [args.stage],
        "arms": arms,
        "preferences": {arm: PREFERENCES[arm] for arm in arms},
        "fallback": "Any physically idle GPU meeting the parent's90000/80000MiB policy",
        "parent_local_and_cloud_completion_required": True,
        "all_parent_arms_completion_required": True,
        "new_base_parameter_updates": 0,
        "automatic_retry": False,
        "gpu_sharing": False,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    if not args.config.resolve().is_relative_to(ROOT):
        raise ValueError("Keep reviewed supplementary settings inside the workspace")
    if not args.wandb_token_file.is_file() or args.wandb_token_file.stat().st_mode & 0o077:
        raise ValueError("Provide the private existing online W&B credential file")
    batch = persistent_batch(settings)
    parent = batch.parent / PARENT
    batch.mkdir(parents=True, exist_ok=True)
    batch.chmod(0o755)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            usage = shutil.disk_usage(batch)
            if usage.free < minimum_space(settings) * 1024**3:
                raise RuntimeError("Insufficient supplementary persistent space: require200GiB")
            write_json(
                batch / "space-admission.json",
                {"free_bytes": usage.free, "minimum_gib": minimum_space(settings)},
            )
            freeze_source(batch, parent, args.config.resolve())
            helpers, policy = load_resources(parent, batch)
            helpers.docker("info")
            image = json.loads(
                helpers.docker("image", "inspect", settings["runtime"]["image"]).stdout
            )[0]["Id"]
            if image != settings["runtime"]["image"]:
                raise ValueError("The supplementary image differs from its frozen parent")
            base_config = read_json(batch / "source/configs/experiment.json")
            write_json(
                batch / "controller-process.json", {"pid": os.getpid(), "utc": utc_now(), **plan}
            )
            if args.stage in ("pipeline", "prepare"):
                deadline = time.monotonic() + settings["runtime"]["wait_timeout_hours"] * 3600
                while True:
                    try:
                        helpers.admit_cpu(resource_config(settings), "single")
                        break
                    except helpers.ResourceUnavailable as error:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("CPU preparation wait expired") from error
                        write_json(
                            batch / "controller-state.json",
                            {"state": "waiting_for_preparation_resources", "reason": str(error)},
                        )
                        time.sleep(min(30, max(0, deadline - time.monotonic())))
                code = run_stage(
                    settings,
                    "single",
                    "prepare",
                    batch,
                    parent,
                    args.wandb_token_file,
                    helpers,
                    time.monotonic() + settings["runtime"]["timeout_hours"] * 3600,
                )
                if code:
                    write_json(
                        batch / "controller-state.json",
                        {
                            "state": "needs_attention",
                            "stage": "prepare",
                            "returncode": code,
                            "new_base_parameter_updates": 0,
                        },
                    )
                    raise SystemExit(code)
                record_panel(batch)
                if args.stage == "prepare":
                    write_json(
                        batch / "controller-state.json",
                        {"state": "prepared", "new_base_parameter_updates": 0},
                    )
                    return
            verify_snapshot(batch, parent, panel_required=True)
            stages = list(STAGES) if args.stage == "pipeline" else [args.stage]
            write_json(
                batch / "controller-state.json",
                {
                    "state": "waiting_for_parent_completion",
                    "arms": arms,
                    "new_base_parameter_updates": 0,
                },
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
                futures = {
                    arm: pool.submit(
                        run_arm,
                        settings,
                        arm,
                        stages,
                        batch,
                        parent,
                        args.wandb_token_file,
                        helpers,
                        policy,
                        base_config,
                    )
                    for arm in arms
                }
                exits = {arm: future.result() for arm, future in futures.items()}
            write_json(
                batch / "controller-state.json",
                {
                    "state": "complete" if not any(exits.values()) else "needs_attention",
                    "exit_codes": exits,
                    "new_base_parameter_updates": 0,
                    "utc": utc_now(),
                },
            )
            if any(exits.values()):
                raise SystemExit(1)
        except Exception:
            failure = {
                "state": "needs_attention",
                "traceback": traceback.format_exc(),
                "new_base_parameter_updates": 0,
            }
            write_json(batch / "controller-state.json", failure)
            write_json(batch / "controller-failure.json", failure)
            raise


if __name__ == "__main__":
    main()
