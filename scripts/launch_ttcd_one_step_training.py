#!/usr/bin/env python3
"""Freeze three one-step stream continuations and queue their paired evaluations."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-one-step-training-v1"
ARCHITECTURES = ("transformer", "loop", "discoloop")
STAGES = (
    "prepare",
    "preflight",
    "train",
    "train-audit",
    "train-cloud-audit",
    "eval-preflight",
    "calibrate",
    "evaluate",
    "audit",
    "cloud-audit",
)
CPU_STAGES = ("prepare", "train-cloud-audit", "cloud-audit", "aggregate")
ADDITIONS = (
    "src/llm_memory_editability/ttcd_one_step.py",
    "src/llm_memory_editability/ttcd_eval_speedup.py",
    "src/llm_memory_editability/ttcd_train_speedup.py",
    "src/llm_memory_editability/ttcd_double_new_eval.py",
    "src/llm_memory_editability/ttcd_merge_hop_panel.py",
    "src/llm_memory_editability/ttcd_merge_hops.py",
    "src/llm_memory_editability/ttcd_one_step_training.py",
    "src/llm_memory_editability/ttcd_one_step_evaluation.py",
    "scripts/run_ttcd_one_step.py",
    "scripts/launch_ttcd_one_step.py",
    "scripts/run_ttcd_merge_hops.py",
    "scripts/launch_ttcd_merge_hops.py",
    "scripts/run_ttcd_one_step_training.py",
    "scripts/launch_ttcd_one_step_training.py",
    "tests/test_ttcd_one_step.py",
    "tests/test_ttcd_merge_hops.py",
    "tests/test_ttcd_merge_hop_panel.py",
    "tests/test_ttcd_one_step_training.py",
    "tests/test_ttcd_one_step_training_cli.py",
    "tests/test_ttcd_one_step_evaluation.py",
    "tests/test_ttcd_one_step_training_controller.py",
    f"docs/development-artifacts/{BATCH}/protocol.md",
)


def load_parent_controller():
    spec = importlib.util.spec_from_file_location(
        "one_step_training_transport", ROOT / "scripts/launch_ttcd_one_step.py"
    )
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


transport = load_parent_controller()
resources = transport.resources
read_json, write_json, sha256 = resources.read_json, resources.write_json, resources.sha256
safe_path, verify_entries = transport.safe_path, transport.verify_entries
runtime_archive, launch_copied_runtime = transport.runtime_archive, transport.launch_copied_runtime


def validate_plan(plan):
    if (
        plan["batch"] != BATCH
        or plan["writer_steps"] != 1
        or plan["training_writer_steps"] != 1
        or plan["training_steps"] != 10000
        or plan["new_base_parameter_updates"] != 30000
        or plan.get("new_foundation_updates") != 0
        or plan.get("new_common_updates") != 0
        or plan.get("training_methods") != ["stream"]
        or plan["methods"] != ["single", "continuous", "independent_merge"]
        or plan["roles"] != ["first", "second"]
        or set(plan["architectures"]) != set(ARCHITECTURES)
    ):
        raise ValueError("Exactly three one-step stream models and three test-time write methods")
    for arm, spec in plan["architectures"].items():
        expected = (
            {
                "parent_batch": "ttcd-successful-base-v1",
                "source_directory": "source-v2",
                "source_manifest": "source-v2-manifest.json",
                "parent_config": "configs/experiment.json",
            }
            if arm == "discoloop"
            else {
                "parent_batch": "ttcd-architecture-comparison-v1",
                "source_directory": "source",
                "source_manifest": "source-manifest.json",
                "parent_config": f"configs/architecture-{arm}.json",
            }
        )
        if spec != expected:
            raise ValueError("Reuse the original frozen sources and common endpoints")
    runtime = plan["runtime"]
    expected_cpus = {
        "transformer": list(range(64, 68)),
        "loop": list(range(68, 72)),
        "discoloop": list(range(72, 76)),
    }
    if (
        runtime["docker_context"] != "lm-memory"
        or runtime["tracking"] != "online"
        or runtime["cpu_threads"] != 4
        or runtime["memory_gib"] != 96
        or runtime["uid"] != 1015
        or runtime["gid"] != 1016
        or runtime["cpu_sets"] != expected_cpus
        or runtime["minimum_persistent_free_gib"] < 400
        or runtime["timeout_hours"] != 48
        or runtime["wait_timeout_hours"] != 96
        or runtime.get("minimum_shared_free_mib", 0) < 32768
        or runtime.get("maximum_own_containers_per_gpu", 0) not in (1, 2, 3)
        or runtime.get("gpu_excluded") != [8]
        or runtime.get("gpu_allowed") != [5, 6]
        or runtime.get("allow_own_batch_sharing") is not True
        or not runtime["image"].startswith("sha256:")
        or len(runtime["image"]) != 71
    ):
        raise ValueError("Keep the reviewed image, CPU isolation and memory/time budgets")
    if set(runtime["gpu_preferences"]) != set(ARCHITECTURES) or any(
        not value or any(gpu not in (5, 6) for gpu in value)
        for value in runtime["gpu_preferences"].values()
    ):
        raise ValueError("The user authorized only CUDA5 and CUDA6 for this batch")


def task_path(workspace, arm):
    return Path(workspace) / f"configs/{BATCH}-{arm}.json"


def verify_task(task, arm, parent_config, parent_config_path, batch):
    if (
        task["batch"] != BATCH
        or task["architecture_name"] != arm
        or task["training_method"] != "stream"
        or task["parent_config"] != "configs/parent.json"
        or task["parent_config_sha256"] != sha256(parent_config_path)
        or task["training_config"]["writer"]["steps"] != 1
        or task["training_config"]["training"]["steps"] != 10000
    ):
        raise ValueError("Task must retain the exact parent identity and use one-step stream")
    for name in ("checkpoint", "gate"):
        expected = task[f"parent_common_{name}_sha256"]
        relative = task.get(f"parent_common_{name}")
        if relative is None:
            common = parent_config.get("common", {})
            relative = common.get("checkpoint" if name == "checkpoint" else "gate_receipt")
        if relative is None:
            raise ValueError("The original common artifacts must be explicit")
        path = Path(relative)
        if not path.is_absolute():
            if path.parts[:1] == ("results",):
                path = batch.parent / Path(*path.parts[1:])
            else:
                path = batch.parent / path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Original common {name} identity changed")


def freeze(batch, plan, config_path, workspace=ROOT):
    """Retain all parent bytes; add an explicit one-step task without editing the parent."""
    if (batch / "source-manifest.json").exists():
        raise FileExistsError("This batch is already frozen; no duplicate freeze or restart")
    records = {}
    for arm in ARCHITECTURES:
        spec = plan["architectures"][arm]
        parent = safe_path(batch.parent, spec["parent_batch"])
        parent_source = safe_path(parent, spec["source_directory"])
        parent_manifest = safe_path(parent, spec["source_manifest"])
        original = read_json(parent_manifest)
        verify_entries(parent_source, original["files"])
        if spec["parent_config"] not in original["files"]:
            raise ValueError("Parent configuration is absent from the original manifest")
        parent_path = parent_source / spec["parent_config"]
        parent_config = read_json(parent_path)
        if (
            parent_config["writer"]["steps"] != 4
            or parent_config["runtime"]["image"] != plan["runtime"]["image"]
            or sha256(parent_source / "configs/continual-panel.json") != plan["panel_sha256"]
        ):
            raise ValueError("Original four-step control, immutable image or fact panel changed")
        task_file = task_path(workspace, arm)
        task = read_json(task_file)
        if task["parent_source_manifest_sha256"] != sha256(parent_manifest):
            raise ValueError("Task binds a different original source manifest")
        verify_task(task, arm, parent_config, parent_path, batch)
        inputs = original.get("inputs", original.get("parent_immutable_inputs"))
        if not inputs:
            raise ValueError("The original source must identify immutable inputs")
        data_parent = batch.parent / "ttcd-successful-base-v1"
        verify_entries(data_parent / "input-mount", inputs)
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
                    raise ValueError(f"Cannot replace original parent science: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        for relative, content in (
            ("configs/parent.json", parent_path.read_bytes()),
            ("configs/task.json", task_file.read_bytes()),
            ("configs/one-step-training.json", Path(config_path).read_bytes()),
        ):
            target = source / relative
            if target.exists():
                raise FileExistsError(f"New settings path already exists: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        verify_entries(source, original["files"])
        manifest = {
            "schema": "ttcd-one-step-training-source-v1",
            "utc": resources.utc_now(),
            "architecture": arm,
            "parent_batch": spec["parent_batch"],
            "parent_source_directory": spec["source_directory"],
            "parent_manifest": str(parent_manifest),
            "parent_manifest_sha256": sha256(parent_manifest),
            "parent_files": original["files"],
            "parent_config": spec["parent_config"],
            "parent_inputs": inputs,
            "data_directory": str(data_parent / "input-mount/data"),
            "files": resources.source_hashes(source),
            "task_sha256": sha256(task_file),
            "parent_common_checkpoint_sha256": task["parent_common_checkpoint_sha256"],
            "parent_common_gate_sha256": task["parent_common_gate_sha256"],
            "host_permissions_changed": False,
            "base_parameter_updates": 10000,
            "writer_steps": 1,
        }
        write_json(directory / "source-manifest.json", manifest)
        records[arm] = sha256(directory / "source-manifest.json")
    write_json(
        batch / "source-manifest.json",
        {
            "schema": "ttcd-one-step-training-freeze-v1",
            "config_sha256": sha256(config_path),
            "arms": records,
            "base_parameter_updates": 30000,
        },
    )
    return records


class _GpuLease:
    def __init__(self, pool, gpu, arm):
        self.pool, self.gpu, self.arm, self.closed = pool, gpu, arm, False

    def close(self):
        if not self.closed:
            self.pool.release(self.gpu, self.arm)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CohortSlots:
    """Hold a global GPU lock while several local architecture stages share its GPU."""

    def __init__(self, directory=None):
        self.directory = directory or Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
        self.guard = threading.RLock()
        self.handles, self.arms = {}, {}

    def take(self, gpu, arm, maximum=3):
        with self.guard:
            if arm not in ARCHITECTURES or any(arm in arms for arms in self.arms.values()):
                raise ValueError("An architecture may hold only one active GPU stage")
            if len(self.arms.get(gpu, set())) >= maximum:
                raise resources.ResourceUnavailable("The per-GPU shared-stage limit is reached")
            if gpu not in self.handles:
                self.directory.mkdir(parents=True, exist_ok=True)
                handle = (self.directory / f"gpu-{gpu}.lock").open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    handle.close()
                    raise
                self.handles[gpu] = handle
                self.arms[gpu] = set()
            self.arms[gpu].add(arm)
            return _GpuLease(self, gpu, arm)

    def release(self, gpu, arm):
        with self.guard:
            self.arms[gpu].remove(arm)
            if not self.arms[gpu]:
                self.handles.pop(gpu).close()
                self.arms.pop(gpu)


SLOTS = CohortSlots()


def gpu_cohort(batch, plan, gpu, row, processes, containers):
    """Account for every CUDA process; only copied runtimes from this batch may share."""
    members, known, arms = [], set(), set()
    for container in containers:
        if not resources.container_uses_gpu(container, gpu, row["uuid"]):
            continue
        name = container["Name"].lstrip("/")
        arm = next((a for a in ARCHITECTURES if name.startswith(f"{BATCH}-{a}-")), None)
        if arm is None or arm in arms:
            raise resources.ResourceUnavailable("GPU contains another project or duplicate arm")
        config, host = container["Config"], container["HostConfig"]
        manifest_sha = sha256(batch / arm / "source-manifest.json")
        stage = name.split(f"{arm}-", 1)[1]
        output = output_directory(batch, arm, stage)
        mounts = {item["Destination"]: item for item in container["Mounts"]}
        requests = host.get("DeviceRequests") or []
        if (
            config.get("User") != "1015:1016"
            or config.get("Image") != plan["runtime"]["image"]
            or "/frozen/scripts/run_ttcd_one_step_training.py" not in config.get("Cmd", [])
            or f"LM_TTCD_SOURCE_MANIFEST_SHA256={manifest_sha}" not in config.get("Env", [])
            or len(requests) != 1
            or requests[0].get("DeviceIDs") != [row["uuid"]]
            or "/frozen" in mounts
            or "/output" not in mounts
            or Path(mounts["/output"]["Source"]).resolve() != output.resolve()
            or not mounts["/output"]["RW"]
        ):
            raise resources.ResourceUnavailable("A candidate batch container differs in identity")
        pid_text = resources.docker("top", container["Id"], "-eo", "pid").stdout
        pids = {int(line.strip()) for line in pid_text.splitlines()[1:] if line.strip().isdigit()}
        known.update(pids)
        arms.add(arm)
        members.append({"name": name, "id": container["Id"], "host_pids": sorted(pids)})
    unknown = [p["pid"] for p in processes if p["uuid"] == row["uuid"] and p["pid"] not in known]
    if unknown:
        raise resources.ResourceUnavailable(f"GPU{gpu} has outside CUDA processes: {unknown}")
    if not members and (row["memory"] > 1024 or row["utilization"] > 5):
        raise resources.ResourceUnavailable("GPU has unaccounted memory or activity")
    return members


def acquire_resources(plan, arm, stage, update, batch, slots=None):
    runtime, slots = plan["runtime"], slots or SLOTS
    slots.directory.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + runtime["wait_timeout_hours"] * 3600
    while time.monotonic() < deadline:
        held = contextlib.ExitStack()
        try:
            for cpu in runtime["cpu_sets"][arm]:
                handle = held.enter_context((slots.directory / f"cpu-{cpu}.lock").open("a"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise resources.ResourceUnavailable("CPU lease is occupied") from error
            containers, available = resources.admit_cpu(plan, arm)
            if stage in CPU_STAGES:
                return held, {"device": "cpu", "cpu_set": runtime["cpu_sets"][arm]}
            rows, processes = resources.gpu_snapshot()
            choices = list(dict.fromkeys([*runtime["gpu_preferences"][arm], *sorted(rows)]))
            choices = [gpu for gpu in choices if gpu in runtime["gpu_allowed"]]
            reasons = {}
            # A completely idle card takes priority over sharing a card in our own cohort.
            for sharing in (False, True):
                for gpu in choices:
                    lease = None
                    try:
                        if gpu not in rows or rows[gpu]["total_mib"] < 90000:
                            raise resources.ResourceUnavailable("GPU is absent or too small")
                        row = rows[gpu]
                        cohort = gpu_cohort(batch, plan, gpu, row, processes, containers)
                        if bool(cohort) != sharing:
                            continue
                        if row["free_mib"] < runtime["minimum_shared_free_mib"]:
                            raise resources.ResourceUnavailable("Need32GiB free before a new stage")
                        lease = slots.take(gpu, arm, runtime["maximum_own_containers_per_gpu"])
                        # Shared slot acquisitions are serialized; reject capacity reserved by
                        # a stage that has not created its Docker container yet.
                        if len(slots.arms[gpu]) > len(cohort) + 1:
                            raise resources.ResourceUnavailable("Another stage is launching here")
                        held.enter_context(lease)
                        return held, {
                            **row,
                            "gpu": gpu,
                            "cpu_set": runtime["cpu_sets"][arm],
                            "host_available_kib": available,
                            "cohort": cohort,
                            "gpu_sharing": sharing,
                            "minimum_free_mib": runtime["minimum_shared_free_mib"],
                            "wallclock_is_exclusive_gpu_benchmark": False,
                            "utc": resources.utc_now(),
                        }
                    except BlockingIOError:
                        if lease is not None:
                            lease.close()
                        reasons[str(gpu)] = "GPU lock held by another controller"
                    except resources.ResourceUnavailable as error:
                        if lease is not None:
                            lease.close()
                        reasons[str(gpu)] = str(error)
            raise resources.ResourceUnavailable(json.dumps(reasons))
        except resources.ResourceUnavailable as error:
            held.close()
            update({"state": "waiting_for_resources", "stage": stage, "reason": str(error)})
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        except BaseException:
            held.close()
            raise
    raise TimeoutError("No safe resource slot became available; no task was preempted")


def output_directory(batch, arm, stage, report_directory=None):
    if stage == "aggregate":
        if report_directory not in (None, "report", "control-report"):
            raise ValueError("Aggregate outputs must use a distinct reviewed report directory")
        return batch / (report_directory or "report")
    if report_directory is not None:
        raise ValueError("Only aggregate stages may choose a report directory")
    return (
        batch
        / arm
        / ("training-run" if stage.startswith("train") or stage == "preflight" else "evaluation")
    )


def build_command(plan, arm, stage, batch, token, row, report_directory=None, report_variant=None):
    runtime = plan["runtime"]
    out = output_directory(batch, arm, stage, report_directory)
    if report_variant not in (None, "control-aggregate") or (
        report_variant is not None
        and (stage != "aggregate" or report_directory != "control-report")
    ):
        raise ValueError("The control aggregate requires its separate output and stage receipts")
    name = f"{BATCH}-{arm}-{report_variant or stage}"
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
    data = batch.parent / "ttcd-successful-base-v1/input-mount/data"
    mounts = [
        (data, "/inputs/data", True),
        (data, "/frozen/data", True),
        (batch.parent, "/frozen/results", True),
        (batch / arm, f"/frozen/results/{BATCH}/{arm}", False),
        (out, "/output", False),
        (token.resolve(), "/run/secrets/wandb", True),
    ]
    if stage == "aggregate":
        mounts.append((batch, "/comparison", True))
    for host, target, readonly in mounts:
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
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
        f"LM_GPU_SHARING={str(row.get('gpu_sharing', False)).lower()}",
        "LM_EXCLUSIVE_GPU_BENCHMARK=false",
        f"WANDB_TAGS=ttcd-one-step-training,{arm},{stage}",
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
        "ttcd-one-step-training-entry",
        "python",
        "/frozen/scripts/run_ttcd_one_step_training.py",
        stage,
        "--task",
        "/frozen/configs/task.json",
        "--settings",
        "/frozen/configs/one-step-training.json",
        "--architecture",
        arm,
        "--out",
        "/output",
        "--tracking",
        "online",
        "--device",
        "cuda:0" if "gpu" in row else "cpu",
    ]
    if stage == "aggregate":
        command += ["--comparison-root", "/comparison"]
    return name, command


def run_stage(plan, arm, stage, batch, token, update, report_directory=None, report_variant=None):
    archive = runtime_archive(batch, arm)
    directory = batch / (
        f"{report_variant or 'aggregate'}-stage"
        if stage == "aggregate"
        else f"{arm}/stages/{stage}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "claim.json").open("x") as stream:
        json.dump({"stage": stage, "utc": resources.utc_now(), "pid": os.getpid()}, stream)
    out = output_directory(batch, arm, stage, report_directory)
    out.mkdir(parents=True, exist_ok=True)
    held, row = acquire_resources(plan, arm, stage, update, batch)
    with held:
        if report_directory is None and report_variant is None:
            name, command = build_command(plan, arm, stage, batch, token, row)
        else:
            name, command = build_command(
                plan, arm, stage, batch, token, row, report_directory, report_variant
            )
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
        for stage in STAGES:
            run_stage(plan, arm, stage, batch, token, update)
            state["completed_stages"].append(stage)
        update({"state": "complete", "new_base_parameter_updates": 10000})
        return state
    except BaseException:
        update({"state": "needs_attention", "traceback": traceback.format_exc()})
        raise


def four_step_control_receipt(batch):
    """Wait for independently audited old controls, preserving the completed main report."""
    parent = safe_path(batch.parent, "ttcd-merge-hops-v1")
    receipts, missing = {}, False
    for arm in ARCHITECTURES:
        arm_missing = False
        for stage in ("audit", "cloud-audit"):
            result = safe_path(parent, f"{arm}/evaluation/result-{stage}.json")
            exit_path = safe_path(parent, f"{arm}/stages/{stage}/exit.json")
            if exit_path.exists() and read_json(exit_path).get("returncode") != 0:
                raise RuntimeError(f"Old four-step control {arm}/{stage} failed")
            if result.exists() and read_json(result).get("passed") is not True:
                raise ValueError(f"Old four-step control {arm}/{stage} validation failed")
            if not (result.is_file() and exit_path.is_file()):
                missing = True
                arm_missing = True
            else:
                receipts.update({str(path): sha256(path) for path in (result, exit_path)})
        state_path = safe_path(parent, f"{arm}/controller-state.json")
        if (
            arm_missing
            and state_path.exists()
            and read_json(state_path).get("state") in ("failed", "needs_attention")
        ):
            raise RuntimeError(f"Old four-step control {arm} needs attention")
    return None if missing else {"passed": True, "receipts": receipts, "utc": resources.utc_now()}


def wait_four_step_controls(plan, batch, update):
    deadline = time.monotonic() + plan["runtime"]["wait_timeout_hours"] * 3600
    while time.monotonic() < deadline:
        receipt = four_step_control_receipt(batch)
        if receipt is not None:
            write_json(batch / "four-step-control-receipt.json", receipt)
            return receipt
        update({"state": "main_complete_waiting_four_step_control"})
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        "Main experiment is complete; old four-step controls did not finish within96h"
    )


def finish_reports(plan, batch, token, update):
    run_stage(plan, "transformer", "aggregate", batch, token, update)
    main = batch / "report/result-aggregate.json"
    evidence = {"main_report": {"path": str(main), "sha256": sha256(main)}}
    update(evidence)
    wait_four_step_controls(plan, batch, update)
    if read_json(main).get("four_step_training_control", {}).get("status") != "complete":
        run_stage(
            plan,
            "transformer",
            "aggregate",
            batch,
            token,
            update,
            report_directory="control-report",
            report_variant="control-aggregate",
        )
        control = batch / "control-report/result-aggregate.json"
        if read_json(control).get("four_step_training_control", {}).get("status") != "complete":
            raise ValueError(
                "Audited four-step controls are absent from the separate control report"
            )
    else:
        control = main
    evidence["control_report"] = {"path": str(control), "sha256": sha256(control)}
    update(evidence)
    return evidence


def execute(plan, batch, token):
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
        reports = finish_reports(plan, batch, token, update)
    except BaseException:
        update({"state": "needs_attention", "traceback": traceback.format_exc()})
        return 1
    update({"state": "complete", "new_base_parameter_updates": 30000, **reports})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    plan = read_json(args.config)
    validate_plan(plan)
    if not (args.execute or args.freeze_only):
        print(json.dumps({"executed": False, "plan": plan, "stages": STAGES}, indent=2))
        return 0
    token = args.wandb_token_file
    if args.execute and (not token.is_file() or token.stat().st_mode & 0o077):
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
            raise FileExistsError("Controller already submitted; inspect it without restarting")
        resources.check_space(batch, plan)
        if (batch / "source-manifest.json").exists():
            if read_json(batch / "source-manifest.json")["config_sha256"] != sha256(args.config):
                raise ValueError("Previously frozen configuration changed")
            for arm in ARCHITECTURES:
                transport.verify_snapshot(batch, arm)
        else:
            freeze(batch, plan, args.config)
        if args.freeze_only:
            print(json.dumps({"frozen": True, "batch": str(batch), "executed": False}))
            return 0
        write_json(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": resources.utc_now(),
                "config_sha256": sha256(args.config),
                "source_manifest_sha256": sha256(batch / "source-manifest.json"),
                "automatic_retry": False,
                "new_base_parameter_updates": 30000,
            },
        )
        return execute(plan, batch, token)


if __name__ == "__main__":
    raise SystemExit(main())
