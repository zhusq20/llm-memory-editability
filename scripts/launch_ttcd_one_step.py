#!/usr/bin/env python3
"""Freeze and queue one-step TTCD evaluations of three audited existing endpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-one-step-second-hop-v1"
ARCHITECTURES = ("transformer", "loop", "discoloop")
STAGES = ("preflight", "calibrate", "evaluate", "audit", "cloud-audit")
ADDITIONS = (
    "src/llm_memory_editability/ttcd_one_step.py",
    "src/llm_memory_editability/ttcd_eval_speedup.py",
    "src/llm_memory_editability/ttcd_double_new_eval.py",
    "scripts/run_ttcd_one_step.py",
    "scripts/launch_ttcd_one_step.py",
    "tests/test_ttcd_one_step.py",
    "tests/test_ttcd_one_step_controller.py",
    f"docs/development-artifacts/{BATCH}/protocol.md",
)


def load_resources():
    spec = importlib.util.spec_from_file_location(
        "one_step_resource_helpers", ROOT / "scripts/launch_discoloop_paired.py"
    )
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


resources = load_resources()
read_json, write_json, sha256 = resources.read_json, resources.write_json, resources.sha256


def digest(content):
    return hashlib.sha256(content).hexdigest()


def validate_plan(plan):
    if (
        plan["batch"] != BATCH
        or plan["new_base_parameter_updates"] != 0
        or plan["parent_phase"] != "stream"
        or plan["writer_steps"] != 1
        or plan["parent_writer_steps"] != 4
        or set(plan["architectures"]) != set(ARCHITECTURES)
    ):
        raise ValueError("Only one-step evaluation of the three existing stream endpoints")
    for arm, spec in plan["architectures"].items():
        expected = (
            {
                "parent_batch": "ttcd-successful-base-v1",
                "source_directory": "source-v2",
                "source_manifest": "source-v2-manifest.json",
                "parent_config": "configs/experiment.json",
                "endpoint_directory": "stream",
            }
            if arm == "discoloop"
            else {
                "parent_batch": "ttcd-architecture-comparison-v1",
                "source_directory": "source",
                "source_manifest": "source-manifest.json",
                "parent_config": f"configs/architecture-{arm}.json",
                "endpoint_directory": f"{arm}/stream",
            }
        )
        if spec != expected:
            raise ValueError("Reuse the fixed audited terminal endpoint and original source")
    runtime = plan["runtime"]
    if (
        runtime["docker_context"] != "lm-memory"
        or runtime["tracking"] != "online"
        or runtime["cpu_threads"] != 4
        or runtime["memory_gib"] != 96
        or runtime["uid"] != 1015
        or runtime["gid"] != 1016
        or runtime["cpu_sets"]
        != {
            "transformer": list(range(12, 16)),
            "loop": list(range(16, 20)),
            "discoloop": list(range(20, 24)),
        }
        or runtime["gpu_preferences"]
        != {
            "transformer": [0, 1, 2],
            "loop": [1, 2, 0],
            "discoloop": [2, 0, 1],
        }
        or runtime["minimum_persistent_free_gib"] < 250
        or runtime["gpu_memory_max_mib"] > 1024
        or runtime["gpu_utilization_max"] > 5
        or runtime["timeout_hours"] != 48
        or runtime["wait_timeout_hours"] != 48
        or not runtime["image"].startswith("sha256:")
        or len(runtime["image"]) != 71
    ):
        raise ValueError("Keep the reviewed exclusive resources, image and fixed timeouts")


def safe_path(root, relative):
    """Reject symlinks at every component, including a redirected source directory."""
    root, relative = Path(root), Path(relative)
    if root.is_symlink() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Frozen paths cannot escape or redirect their source")
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"Symlink in frozen path: {relative}")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Frozen path escapes its root")
    return current


def verify_entries(root, entries):
    for relative, expected in entries.items():
        path = safe_path(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen bytes changed: {relative}")


def freeze(batch, plan, config_path, workspace=ROOT):
    """Copy original parent bytes once; only add new files, never overwrite parent files."""
    if (batch / "source-manifest.json").exists():
        raise FileExistsError("Existing submission must not be frozen or restarted again")
    records = {}
    for arm in ARCHITECTURES:
        spec = plan["architectures"][arm]
        parent = safe_path(batch.parent, spec["parent_batch"])
        parent_source = safe_path(parent, spec["source_directory"])
        parent_manifest_path = safe_path(parent, spec["source_manifest"])
        original = read_json(parent_manifest_path)
        verify_entries(parent_source, original["files"])
        if spec["parent_config"] not in original["files"]:
            raise ValueError("Original parent configuration is absent from its frozen manifest")
        parent_config = read_json(parent_source / spec["parent_config"])
        if (
            parent_config["runtime"]["image"] != plan["runtime"]["image"]
            or parent_config["writer"]["steps"] != plan["parent_writer_steps"]
            or parent_config["models"]["stream"]["step"] != 10000
            or parent_config["models"]["stream"]["checkpoint"]
            != f"results/{spec['parent_batch']}/{spec['endpoint_directory']}/training/parent.pt"
        ):
            raise ValueError("Parent dependency image, writer or exact endpoint changed")
        inputs = original.get("inputs", original.get("parent_immutable_inputs"))
        if not inputs:
            raise ValueError("Parent source must identify the immutable data")
        data_parent = parent if arm == "discoloop" else batch.parent / "ttcd-successful-base-v1"
        verify_entries(data_parent / "input-mount", inputs)
        panel_relative = "configs/continual-panel.json"
        if sha256(parent_source / panel_relative) != plan["panel_sha256"]:
            raise ValueError("All architectures must reuse the exact same frozen fact panel")
        directory = safe_path(batch, arm)
        directory.mkdir(exist_ok=False)
        source = directory / "source"
        source.mkdir()
        for relative in original["files"]:
            target = safe_path(source, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((parent_source / relative).read_bytes())
        for relative in ADDITIONS:
            content = safe_path(workspace, relative).read_bytes()
            target = safe_path(source, relative)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f"Cannot replace frozen parent science: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        target = source / "configs/one-step.json"
        if target.exists():
            raise FileExistsError("One-step settings must be a new file")
        target.write_bytes(config_path.read_bytes())
        files = resources.source_hashes(source)
        verify_entries(source, original["files"])
        manifest = {
            "schema": "ttcd-one-step-source-v1",
            "utc": resources.utc_now(),
            "architecture": arm,
            "parent_batch": spec["parent_batch"],
            "parent_source_directory": spec["source_directory"],
            "parent_manifest": str(parent_manifest_path),
            "parent_manifest_sha256": sha256(parent_manifest_path),
            "parent_files": original["files"],
            "parent_config": spec["parent_config"],
            "parent_inputs": inputs,
            "data_directory": str(data_parent / "input-mount/data"),
            "files": files,
            "host_permissions_changed": False,
            "base_parameter_updates": 0,
        }
        write_json(directory / "source-manifest.json", manifest)
        records[arm] = sha256(directory / "source-manifest.json")
    write_json(
        batch / "source-manifest.json",
        {
            "schema": "ttcd-one-step-freeze-v1",
            "config_sha256": sha256(config_path),
            "arms": records,
        },
    )
    return records


def verify_snapshot(batch, arm):
    index = read_json(batch / "source-manifest.json")
    path = safe_path(batch, f"{arm}/source-manifest.json")
    if sha256(path) != index["arms"][arm]:
        raise ValueError("Per-architecture source manifest changed")
    manifest = read_json(path)
    verify_entries(batch / arm / "source", manifest["files"])
    verify_entries(batch / arm / "source", manifest["parent_files"])
    parent_manifest = Path(manifest["parent_manifest"])
    if sha256(parent_manifest) != manifest["parent_manifest_sha256"]:
        raise ValueError("Parent manifest identity changed")
    data_directory = Path(manifest["data_directory"])
    verify_entries(data_directory.parent, manifest["parent_inputs"])
    return manifest


def endpoint_receipt(batch, plan, arm):
    """Wait only for training and its independent replay, not old four-step evaluation."""
    spec = plan["architectures"][arm]
    parent = safe_path(batch.parent, spec["parent_batch"])
    endpoint = safe_path(parent, spec["endpoint_directory"])
    if arm == "discoloop":
        exit_paths = [endpoint / f"exit-{stage}.json" for stage in ("train", "train-audit")]
        state_paths = [endpoint / f"controller-state{suffix}.json" for suffix in ("", "-v2")]
    else:
        exit_paths = [
            parent / arm / "stages" / f"stream-{stage}" / "exit.json"
            for stage in ("train", "train-audit")
        ]
        state_paths = [parent / arm / "controller-state.json"]
    for path in exit_paths:
        if path.exists() and read_json(path).get("returncode", read_json(path).get("code")) != 0:
            raise RuntimeError(f"Parent stage failed: {path}; no automatic retry")
    required = [endpoint / "training/parent.pt", endpoint / "training/audit.json", *exit_paths]
    if not all(path.is_file() for path in required):
        for path in state_paths:
            if path.exists() and read_json(path).get("state") in ("needs_attention", "failed"):
                raise RuntimeError(f"Parent controller needs attention: {path}")
        return None
    for path in required:
        safe_path(parent, path.relative_to(parent))
    audit = read_json(required[1])
    checkpoint_sha = sha256(required[0])
    if audit.get("passed") is not True or audit.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Parent audit failed or does not bind the exact checkpoint")
    if audit.get("phase") != "stream":
        raise ValueError("Only the audited stream endpoint may be reused")
    return {
        "passed": True,
        "architecture": arm,
        "checkpoint": str(required[0]),
        "checkpoint_sha256": checkpoint_sha,
        "audit": str(required[1]),
        "audit_sha256": sha256(required[1]),
        "exit_receipts": {str(path): sha256(path) for path in exit_paths},
        "utc": resources.utc_now(),
    }


def wait_endpoint(batch, plan, arm, update):
    deadline = time.monotonic() + plan["runtime"]["wait_timeout_hours"] * 3600
    while time.monotonic() < deadline:
        receipt = endpoint_receipt(batch, plan, arm)
        if receipt:
            write_json(batch / arm / "parent-endpoint.json", receipt)
            return receipt
        update({"state": "waiting_for_parent_endpoint", "parent_phase": "stream10000"})
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("Existing stream endpoint did not finish its independent training audit")


def runtime_archive(batch, arm):
    manifest = verify_snapshot(batch, arm)
    files = {
        f"frozen/{relative}": (batch / arm / "source" / relative).read_bytes()
        for relative in manifest["files"]
    }
    files["frozen/source-manifest.json"] = (batch / arm / "source-manifest.json").read_bytes()
    directories = {
        str(parent) for name in files for parent in Path(name).parents if str(parent) != "."
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
            info = tarfile.TarInfo(directory)
            info.type, info.mode = tarfile.DIRTYPE, 0o555
            info.uid = info.gid = 0
            archive.addfile(info)
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o444
            info.uid = info.gid = 0
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def acquire_resources(plan, arm, stage, update, lock_directory=None):
    """Hold CPU and GPU locks until this stage exits; never share or stop another task."""
    runtime = plan["runtime"]
    directory = lock_directory or Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    deadline = time.monotonic() + runtime["wait_timeout_hours"] * 3600
    cpu_only = stage in ("cloud-audit", "aggregate")
    while time.monotonic() < deadline:
        held = contextlib.ExitStack()
        try:
            for cpu in runtime["cpu_sets"][arm]:
                lock = held.enter_context((directory / f"cpu-{cpu}.lock").open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise resources.ResourceUnavailable("Exclusive CPU lock held") from error
            containers, available = resources.admit_cpu(plan, arm)
            if cpu_only:
                return held, {"device": "cpu", "cpu_set": runtime["cpu_sets"][arm]}
            rows, processes = resources.gpu_snapshot()
            reasons = {}
            for gpu in dict.fromkeys([*runtime["gpu_preferences"][arm], *sorted(rows)]):
                gpu_lock = (directory / f"gpu-{gpu}.lock").open("a")
                try:
                    try:
                        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise resources.ResourceUnavailable("Exclusive GPU lock held") from error
                    if gpu not in rows:
                        raise resources.ResourceUnavailable(f"GPU{gpu} absent")
                    row = rows[gpu]
                    if (
                        row["total_mib"] < 90000
                        or row["free_mib"] < 80000
                        or row["memory"] > runtime["gpu_memory_max_mib"]
                        or row["utilization"] > runtime["gpu_utilization_max"]
                        or any(p["uuid"] == row["uuid"] for p in processes)
                        or any(
                            resources.container_uses_gpu(c, gpu, row["uuid"]) for c in containers
                        )
                    ):
                        raise resources.ResourceUnavailable(f"GPU{gpu} occupied")
                    held.enter_context(gpu_lock)
                    return held, {
                        **row,
                        "gpu": gpu,
                        "cpu_set": runtime["cpu_sets"][arm],
                        "host_available_kib": available,
                        "gpu_sharing": False,
                        "utc": resources.utc_now(),
                    }
                except resources.ResourceUnavailable as error:
                    gpu_lock.close()
                    reasons[str(gpu)] = str(error)
                except BaseException:
                    gpu_lock.close()
                    raise
            raise resources.ResourceUnavailable(json.dumps(reasons))
        except resources.ResourceUnavailable as error:
            held.close()
            update({"state": "waiting_for_resources", "stage": stage, "reason": str(error)})
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        except BaseException:
            held.close()
            raise
    raise TimeoutError("No exclusive idle resource became available; no task was preempted")


def build_command(plan, arm, stage, batch, token, row):
    runtime = plan["runtime"]
    spec = plan["architectures"][arm]
    out = batch / ("aggregate" if stage == "aggregate" else f"{arm}/evaluation")
    name = f"{BATCH}-{arm}-{stage}"
    command = [
        *resources.DOCKER,
        "create",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        "4",
        "--cpuset-cpus",
        ",".join(map(str, runtime["cpu_sets"][arm])),
        "--memory",
        "96g",
        "--shm-size",
        "8g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
    ]
    if "gpu" in row:
        command += ["--gpus", f"device={row['uuid']}"]
    data_parent = (
        batch.parent / "ttcd-successful-base-v1"
        if arm != "discoloop"
        else batch.parent / spec["parent_batch"]
    )
    for host, target, readonly in (
        (data_parent / "input-mount/data", "/inputs/data", True),
        (data_parent / "input-mount/data", "/frozen/data", True),
        (batch.parent, "/frozen/results", True),
        (out, "/aggregate" if stage == "aggregate" else "/output", False),
        (token.resolve(), "/run/secrets/wandb", True),
    ):
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    if stage == "aggregate":
        command += ["--mount", f"type=bind,source={batch},target=/comparison,readonly"]
    command += ["--tmpfs", "/tmp:rw,size=8g"]
    for env in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "HOME=/tmp/lm-home",
        "XDG_CACHE_HOME=/tmp/lm-home/.cache",
        "XDG_CONFIG_HOME=/tmp/lm-home/.config",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        f"WANDB_TAGS=ttcd-one-step,{arm},{stage}",
        f"LM_TTCD_SOURCE_MANIFEST_SHA256={sha256(batch / arm / 'source-manifest.json')}",
    ):
        command += ["--env", env]
    if "gpu" in row:
        command += [
            "--env",
            f"LM_ACTUAL_GPU_INDEX={row['gpu']}",
            "--env",
            f"LM_ACTUAL_GPU_UUID={row['uuid']}",
        ]
    command += [
        runtime["image"],
        "bash",
        "-c",
        'mkdir -p "$HOME"\nexport WANDB_API_KEY="$(cat /run/secrets/wandb)"\nexec "$@"',
        "ttcd-one-step-entry",
        "python",
        "/frozen/scripts/run_ttcd_one_step.py",
        stage,
        "--parent-config",
        f"/frozen/{spec['parent_config']}",
        "--settings",
        "/frozen/configs/one-step.json",
        "--architecture",
        arm,
        "--out",
        "/aggregate" if stage == "aggregate" else "/output",
        "--tracking",
        "online",
        "--device",
        "cuda:0" if "gpu" in row else "cpu",
    ]
    if stage == "aggregate":
        command += ["--comparison-root", "/comparison"]
    return name, command


def launch_copied_runtime(command, archive, receipt_directory, *, copy_runner=subprocess.run):
    """Create → copy immutable root-owned source → start, retaining every failure."""
    name = command[command.index("--name") + 1]
    receipt = {
        "container": name,
        "archive_sha256": digest(archive),
        "state": "creating",
        "host_permissions_changed": False,
    }
    write_json(receipt_directory / "transport.json", receipt)
    try:
        created = resources.docker(*command[3:])
    except subprocess.CalledProcessError as error:
        receipt.update(
            state="create_failed",
            returncode=error.returncode,
            stderr=error.stderr,
        )
        write_json(receipt_directory / "transport.json", receipt)
        raise
    receipt.update(state="created", container_id=created.stdout.strip())
    write_json(receipt_directory / "transport.json", receipt)
    copied = copy_runner(
        [*resources.DOCKER, "cp", "-", f"{name}:/"],
        input=archive,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if copied.returncode:
        receipt.update(state="copy_failed", returncode=copied.returncode)
        write_json(receipt_directory / "transport.json", receipt)
        raise RuntimeError(f"Frozen source copy failed: {copied.stderr.decode(errors='replace')}")
    receipt["state"] = "copied"
    write_json(receipt_directory / "transport.json", receipt)
    resources.docker("start", name)
    receipt["state"] = "started"
    write_json(receipt_directory / "transport.json", receipt)
    return created


def run_stage(plan, arm, stage, batch, token, update):
    archive = runtime_archive(batch, arm)
    directory = batch / ("aggregate-stage" if stage == "aggregate" else f"{arm}/stages/{stage}")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "claim.json").open("x") as stream:
        json.dump({"stage": stage, "utc": resources.utc_now(), "pid": os.getpid()}, stream)
    out = batch / ("aggregate" if stage == "aggregate" else f"{arm}/evaluation")
    out.mkdir(parents=True, exist_ok=True)
    held, row = acquire_resources(plan, arm, stage, update)
    with held:
        name, command = build_command(plan, arm, stage, batch, token, row)
        if resources.inspect_container(name) is not None:
            raise FileExistsError(
                "Stage container exists; no duplicate submission or automatic retry"
            )
        write_json(directory / "admission.json", row)
        write_json(directory / "command.json", {"argv": command})
        update({"state": "launching", "stage": stage, "container": name, "resource": row})
        try:
            created = launch_copied_runtime(command, archive, directory)
            update({"state": "running", "container_id": created.stdout.strip()})
            with (directory / "container.log").open("x") as log:
                code = resources.monitor_container(
                    name, log, time.monotonic() + plan["runtime"]["timeout_hours"] * 3600, update
                )
            write_json(directory / "container-state.json", resources.inspect_container(name))
            write_json(directory / "exit.json", {"returncode": code, "utc": resources.utc_now()})
            if code:
                raise RuntimeError(f"{arm}/{stage} exited {code}; no automatic retry")
            result_path = out / f"result-{stage}.json"
            result = read_json(result_path)
            if result.get("passed") is False:
                raise ValueError(f"Mandatory {stage} validation failed")
            write_json(
                directory / "result.json",
                {"path": str(result_path), "sha256": sha256(result_path), "passed": True},
            )
            links = {
                path.stem: read_json(path).get("url")
                for path in out.glob("wandb-*.json")
                if read_json(path).get("url")
            }
            update({"state": "stage_complete", "stage": stage, "wandb_links": links})
        except BaseException:
            if not (directory / "exit.json").exists():
                write_json(
                    directory / "exit.json",
                    {
                        "returncode": None,
                        "state": "launch_or_monitor_failed",
                        "traceback": traceback.format_exc(),
                    },
                )
            raise


def run_arm(plan, arm, batch, token):
    state = {"architecture": arm, "state": "starting", "completed_stages": []}

    def update(values):
        state.update(values)
        state["utc"] = resources.utc_now()
        write_json(batch / arm / "controller-state.json", state)

    try:
        endpoint = wait_endpoint(batch, plan, arm, update)
        update({"parent_endpoint": endpoint})
        for stage in STAGES:
            run_stage(plan, arm, stage, batch, token, update)
            state["completed_stages"].append(stage)
        update({"state": "complete", "new_base_parameter_updates": 0})
        return state
    except BaseException:
        update({"state": "needs_attention", "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    plan = read_json(args.config)
    validate_plan(plan)
    if not args.execute:
        print(
            json.dumps(
                {"executed": False, "plan": plan, "stages": STAGES, "aggregate": True}, indent=2
            )
        )
        return 0
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private W&B credential file")
    root = Path(plan["runtime"]["persistent_root"])
    if not root.is_dir() or not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Outputs must use the LM persistent mount")
    if not any(path.is_mount() for path in (root, *root.parents) if path != Path("/")):
        raise ValueError("Persistent mount is unavailable")
    batch = safe_path(root, BATCH)
    batch.mkdir(exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError(
                "Controller already submitted; inspect its state without restarting"
            )
        resources.check_space(batch, plan)
        freeze(batch, plan, args.config)
        write_json(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": resources.utc_now(),
                "config_sha256": sha256(args.config),
                "source_manifest_sha256": sha256(batch / "source-manifest.json"),
                "automatic_retry": False,
                "new_base_parameter_updates": 0,
            },
        )
        state = {"state": "running", "pid": os.getpid(), "outcomes": {}, "failures": {}}

        def update(values):
            state.update(values)
            state["utc"] = resources.utc_now()
            write_json(batch / "controller-state.json", state)

        update({})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run_arm, plan, arm, batch, token): arm for arm in ARCHITECTURES}
            for future in concurrent.futures.as_completed(futures):
                arm = futures[future]
                try:
                    state["outcomes"][arm] = future.result()
                except BaseException:
                    state["failures"][arm] = traceback.format_exc()
                update({})
        if state["failures"]:
            update({"state": "needs_attention"})
            return 1
        try:
            run_stage(plan, "transformer", "aggregate", batch, token, update)
        except BaseException:
            update({"state": "needs_attention", "traceback": traceback.format_exc()})
            return 1
        update({"state": "complete", "new_base_parameter_updates": 0})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
