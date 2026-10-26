#!/usr/bin/env python3
"""Adopt frozen TTCD pipelines under the user's GPU4 sharing amendment.

This host adapter changes resource admission only. Existing common training is
monitored without restarting it; the original gates and scientific commands
remain authoritative. Default invocation only prints a read-only plan.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = "ttcd-successful-base-v1"
SUPPLEMENT = "ttcd-double-new-eval-v1"
AMENDMENT = "gpu4-parallel-20261015"


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def load_parent(batch, workspace):
    repair = load_module(
        batch / "source-v2/scripts/repair_ttcd_verified_preflight.py", "shared_gpu4_repair"
    )
    repair.ROOT = workspace
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        repair.controller = repair.load_controller(
            batch / "source/scripts/launch_discoloop_paired.py"
        )
    finally:
        sys.dont_write_bytecode = previous
    repair.controller.ROOT = workspace
    return repair


def stopped_controller(batch, target):
    path = batch / AMENDMENT / "stop-host-controller.json"
    if not path.is_file():
        raise ValueError("A preserved stop record is required before replacing host monitoring")
    stop = read_json(path)
    pid = stop["old_pid"]
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise ValueError("Invalid previous host controller identity")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("The previous host controller is still alive")
    if stop.get("no_model_restart") is not True or not stop.get("reason"):
        raise ValueError(
            "The resource amendment must preserve models and record user authorization"
        )
    if target == "parent":
        if (
            stop.get("phase", "common") != "common"
            or stop.get("stage") != "train"
            or stop.get("container") != f"{PARENT}-common-train-v2"
            or not stop.get("container_id")
            or stop.get("deadline_remaining_seconds", 0) <= 0
            or "science_updates_snapshot" not in stop
        ):
            raise ValueError("Only the existing common/train container may be adopted")
    return stop


def remaining_deadline(stop):
    timestamp = stop.get("stopped_utc", stop.get("utc"))
    if not timestamp:
        raise ValueError("The preserved training deadline requires the stop timestamp")
    recorded = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if recorded.tzinfo is None:
        raise ValueError("The stop timestamp must include its UTC timezone")
    elapsed = max(0, (datetime.datetime.now(datetime.timezone.utc) - recorded).total_seconds())
    remaining = stop["deadline_remaining_seconds"] - elapsed
    if remaining <= 0:
        raise TimeoutError("The original common phase deadline has expired; it cannot be reset")
    return time.monotonic() + remaining


def require_unclaimed(batch, phases, stages):
    for phase in phases:
        for stage in stages:
            for prefix in ("claim", "result", "exit"):
                if (batch / phase / f"{prefix}-{stage}.json").exists():
                    raise FileExistsError(f"Preserve existing {phase}/{prefix}-{stage}; no retry")


def validate_parent(repair, batch, config, config_path, receipt_path, stop):
    helpers = repair.controller
    helpers.validate_config(config)
    repair.verify_repaired_source(batch, config, config_path, receipt_path)
    preflight = batch / "common/preflight-v2"
    if (
        read_json(preflight / "exit-preflight.json").get("returncode") != 0
        or read_json(preflight / "result-preflight.json").get("passed") is not True
    ):
        raise ValueError("The existing repaired preflight must already have passed")
    remaining = tuple(s for s in helpers.COMMON_STAGES if s not in ("preflight", "train"))
    require_unclaimed(batch, ["common"], remaining)
    require_unclaimed(batch, helpers.ARMS, helpers.ARM_STAGES)
    if (batch / "common/exit-train.json").exists():
        raise FileExistsError(
            "Preserve the existing common train exit; no automatic adoption retry"
        )
    if not (batch / "common/claim-train.json").is_file():
        raise ValueError("The original common training claim is missing")
    command = read_json(batch / "common/command-train.json")["argv"]
    if command[command.index("--name") + 1] != stop["container"]:
        raise ValueError("The stop record names another common training command")
    container = helpers.inspect_container(stop["container"])
    if (
        container is None
        or container["Id"] != stop["container_id"]
        or not container["State"]["Running"]
        or container["Config"]["Image"] != config["runtime"]["image"]
    ):
        raise ValueError("The exact original common training container is not running")
    return remaining_deadline(stop)


def add_resource_metadata(command, image, row, amendment_path):
    command = list(command)
    if row.get("gpu") != 4 or not row.get("uuid"):
        raise ValueError("The user authorized sharing GPU4 only")
    position = command.index(image)
    command[position:position] = [
        "--mount",
        f"type=bind,source={amendment_path},target=/run/experiment/gpu4-sharing.json,readonly",
        "--env",
        "LM_ACTUAL_GPU_INDEX=4",
        "--env",
        f"LM_ACTUAL_GPU_UUID={row['uuid']}",
        "--env",
        "LM_RESOURCE_SCHEDULE_PATH=/run/experiment/gpu4-sharing.json",
        "--env",
        "WANDB_TAGS=ttcd-continual,gpu4-shared,user-authorized",
    ]
    return command


def guard_monitor(helpers, resources):
    original = helpers.monitor_container

    def monitor(name, log, deadline, update=lambda _: None, **kwargs):
        def guarded_update(value):
            resources.verify_gpu4_cohort(helpers)
            update(value)

        resources.verify_gpu4_cohort(helpers)
        return original(name, log, deadline, guarded_update, **kwargs)

    helpers.monitor_container = monitor


def install_parent_resources(repair, resources, amendment_path, stop, deadline):
    helpers = repair.controller
    original_build, original_phase = repair.build_command, repair.run_phase

    def admit(batch, config, phase, gpu):
        if gpu != 4:
            raise ValueError("No fallback or sharing beyond GPU4 was authorized")
        return resources.admission(helpers, config["runtime"], phase)

    def wait(batch, config, phase, deadline, update=lambda _: None):
        row = admit(batch, config, phase, 4)
        update({"actual_gpu": 4, "actual_gpu_uuid": row["uuid"], "gpu_sharing": True})
        return contextlib.nullcontext(), row

    def build(config, phase, stage, batch, token_file, row):
        name, command = original_build(config, phase, stage, batch, token_file, row)
        return name, add_resource_metadata(command, config["runtime"]["image"], row, amendment_path)

    def phase(batch, config, config_path, receipt_path, phase, token_file):
        if phase != "common":
            return original_phase(batch, config, config_path, receipt_path, phase, token_file)
        return adopt_running_common(
            repair, resources, batch, config, config_path, receipt_path, token_file, stop, deadline
        )

    helpers.admit_gpu, helpers.wait_gpu = admit, wait
    repair.build_command, repair.run_phase = build, phase
    guard_monitor(helpers, resources)


def adopt_running_common(
    repair, resources, batch, config, config_path, receipt_path, token_file, stop, deadline
):
    helpers, out = repair.controller, batch / "common"
    state = {
        "phase": "common",
        "state": "adopting_existing_train",
        "stage": "train",
        "container": stop["container"],
        "container_id": stop["container_id"],
        "gpu_sharing": True,
        "automatic_retry": False,
        "model_restarted": False,
    }

    def update(value):
        state.update(value)
        helpers.write_json(
            out / "controller-state-v2.json", {**state, "updated_utc": helpers.utc_now()}
        )

    try:
        with (out / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            repair.verify_repaired_source(batch, config, config_path, receipt_path)
            cohort = resources.verify_gpu4_cohort(helpers)
            helpers.write_json(
                batch / AMENDMENT / "adopted-common-admission.json",
                {
                    **read_json(out / "admission-train.json"),
                    "gpu": 4,
                    "uuid": cohort["row"]["uuid"],
                    "gpu_sharing": True,
                    "adopted_existing_container": True,
                    "cpu_readmission": False,
                    "cohort": cohort["containers"],
                    "original_admission_sha256": sha256(out / "admission-train.json"),
                },
            )
            update({"remaining_seconds": deadline - time.monotonic()})
            with (batch / AMENDMENT / "adopted-common-train.log").open("x") as log:
                code = helpers.monitor_container(stop["container"], log, deadline, update)
            with (out / "exit-train.json").open("x") as stream:
                json.dump(
                    {
                        "returncode": code,
                        "container": stop["container"],
                        "container_id": stop["container_id"],
                        "adopted": True,
                        "utc": helpers.utc_now(),
                    },
                    stream,
                    indent=2,
                )
            if code:
                update({"state": "needs_attention", "returncode": code})
                return code
            for stage in helpers.COMMON_STAGES:
                if stage in ("preflight", "train"):
                    continue
                repair.verify_repaired_source(batch, config, config_path, receipt_path)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Original common phase deadline before its next stage")
                row = helpers.admit_gpu(batch, config, "common", 4)
                code = repair.run_stage(
                    batch, config, "common", stage, token_file, deadline, row, update
                )
                if code:
                    update({"state": "needs_attention", "stage": stage, "returncode": code})
                    return code
            update({"state": "complete", "finished_utc": helpers.utc_now()})
            return 0
    except Exception:
        failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
        update(failure)
        helpers.write_json(out / "controller-failure-v2.json", failure)
        return 1


def install_supplement_resources(controller, helpers, resources, amendment_path):
    original_build = controller.build_command

    def admit(settings, arm, gpu, ignored_helpers, policy):
        if gpu != 4:
            raise ValueError("Supplementary sharing is restricted to GPU4")
        return resources.admission(helpers, settings["runtime"], arm)

    def wait(settings, arm, ignored_helpers, policy, deadline, update=lambda _: None):
        row = admit(settings, arm, 4, helpers, policy)
        update({"actual_gpu": 4, "actual_gpu_uuid": row["uuid"], "gpu_sharing": True})
        return contextlib.nullcontext(), row

    def build(settings, arm, stage, batch, parent, token_file, row=None):
        name, command = original_build(settings, arm, stage, batch, parent, token_file, row)
        return name, add_resource_metadata(
            command, settings["runtime"]["image"], row, amendment_path
        )

    controller.admit_gpu, controller.wait_gpu, controller.build_command = admit, wait, build
    guard_monitor(helpers, resources)


@contextlib.contextmanager
def wait_cohort_lock(resources, deadline, update):
    while time.monotonic() < deadline:
        held = contextlib.ExitStack()
        try:
            held.enter_context(resources.acquire_gpu4_lock())
        except BlockingIOError:
            held.close()
            update({"state": "waiting_for_parent_gpu4_cohort_lock"})
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        else:
            with held:
                yield
            return
    raise TimeoutError("The parent GPU4 cohort lock was not released before the wait deadline")


def record_amendment(batch, helpers, target, stop, resources_path):
    directory = batch / AMENDMENT
    source = batch / ("source-v2" if target == "parent" else "source")
    manifest = batch / ("source-v2-manifest.json" if target == "parent" else "source-manifest.json")
    record = {
        "target": target,
        "pid": os.getpid(),
        "utc": helpers.utc_now(),
        "reason": "Explicit user authorization: run both arms concurrently on GPU4",
        "gpu": 4,
        "gpu_sharing": True,
        "automatic_retry": False,
        "scientific_source_changed": False,
        "scientific_config_changed": False,
        "model_restarted": False,
        "preparation_repeated": False,
        "controller_sha256": sha256(Path(__file__)),
        "resources_sha256": sha256(resources_path),
        "stop_record_sha256": sha256(directory / "stop-host-controller.json"),
        "source_manifest_sha256": sha256(manifest),
        "scientific_config_sha256": sha256(source / "configs/experiment.json"),
        "old_pid": stop["old_pid"],
        "wallclock_is_exclusive_gpu_benchmark": False,
    }
    if target == "supplement":
        record["settings_sha256"] = sha256(source / "configs/double-new-eval.json")
        record["panel_sha256"] = sha256(batch / "panel.json")
    path = directory / "sharing-amendment.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
    path.chmod(0o644)
    helpers.write_json(directory / "controller-process.json", record)
    return path


def execute_parent(workspace, token, resources, resources_path):
    config = read_json(workspace / f"configs/{PARENT}.json")
    batch = Path(config["runtime"]["persistent_root"]) / PARENT
    repair = load_parent(batch, workspace)
    helpers = repair.controller
    config_path = batch / "source-v2/configs/experiment.json"
    if sha256(workspace / f"configs/{PARENT}.json") != sha256(config_path):
        raise ValueError("The scientific parent configuration changed")
    receipt = batch / "preflight-repair-20261015/engineering-receipt.json"
    with contextlib.ExitStack() as stack:
        for name in ("controller.lock", "resume-controller-v2.lock", "resource-controller.lock"):
            lock = stack.enter_context((batch / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = stopped_controller(batch, "parent")
        deadline = validate_parent(repair, batch, config, config_path, receipt, stop)
        stack.enter_context(resources.acquire_gpu4_lock())
        resources.verify_gpu4_cohort(helpers)
        amendment = record_amendment(batch, helpers, "parent", stop, resources_path)
        install_parent_resources(repair, resources, amendment, stop, deadline)
        return repair.execute_pipeline(batch, config, config_path, receipt, token)


def execute_supplement(workspace, token, resources, resources_path):
    live = read_json(workspace / f"configs/{SUPPLEMENT}.json")
    batch = Path(live["runtime"]["persistent_root"]) / SUPPLEMENT
    parent = batch.parent / PARENT
    controller = load_module(
        batch / "source/scripts/launch_ttcd_double_new_eval.py", "shared_gpu4_supplement"
    )
    controller.ROOT = workspace
    settings = read_json(batch / "source/configs/double-new-eval.json")
    controller.validate_settings(settings)
    if sha256(workspace / f"configs/{SUPPLEMENT}.json") != sha256(
        batch / "source/configs/double-new-eval.json"
    ):
        raise ValueError("The supplementary settings changed")
    helpers, policy = controller.load_resources(parent, batch)
    base = read_json(batch / "source/configs/experiment.json")
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stop = stopped_controller(batch, "supplement")
        controller.verify_snapshot(batch, parent, panel_required=True)
        if read_json(batch / "preparation/result-prepare.json").get("passed") is not True:
            raise ValueError("Graph-only preparation must already have passed")
        require_unclaimed(batch, controller.ARMS, controller.STAGES)
        amendment = record_amendment(batch, helpers, "supplement", stop, resources_path)
        install_supplement_resources(controller, helpers, resources, amendment)
        helpers.write_json(
            batch / AMENDMENT / "controller-state.json", {"state": "waiting_for_parent_completion"}
        )
        wait_deadline = time.monotonic() + settings["runtime"]["wait_timeout_hours"] * 3600
        controller.wait_dependencies(parent, base, "single", wait_deadline)
        # Parent owns this GPU4 cohort lock until both original arms are finished.
        with wait_cohort_lock(
            resources,
            wait_deadline,
            lambda value: helpers.write_json(batch / AMENDMENT / "controller-state.json", value),
        ):
            resources.verify_gpu4_cohort(helpers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                tasks = {
                    arm: pool.submit(
                        controller.run_arm,
                        settings,
                        arm,
                        list(controller.STAGES),
                        batch,
                        parent,
                        token,
                        helpers,
                        policy,
                        base,
                    )
                    for arm in controller.ARMS
                }
                exits = {arm: task.result() for arm, task in tasks.items()}
        helpers.write_json(
            batch / AMENDMENT / "controller-state.json",
            {
                "state": "complete" if not any(exits.values()) else "needs_attention",
                "exit_codes": exits,
                "scientific_training_updates": 0,
                "gpu_sharing": True,
            },
        )
        return 1 if any(exits.values()) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("parent", "supplement"), required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "target": args.target,
                "executed": args.execute,
                "gpu": 4,
                "gpu_sharing": True,
                "scientific_source_changed": False,
                "scientific_config_changed": False,
                "model_restarted": False,
                "preparation_repeated": False,
                "automatic_retry": False,
                "both_parent_arms_before_supplement": True,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return 0
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private online tracking credential file")
    workspace = args.workspace.resolve()
    resources_path = workspace / "scripts/ttcd_gpu4_resources.py"
    resources = load_module(resources_path, "ttcd_gpu4_shared_resources")
    execute = execute_parent if args.target == "parent" else execute_supplement
    return execute(workspace, token, resources, resources_path)


if __name__ == "__main__":
    raise SystemExit(main())
