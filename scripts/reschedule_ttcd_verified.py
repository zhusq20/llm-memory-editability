#!/usr/bin/env python3
"""Run the frozen TTCD pipeline on other exclusively idle GPUs.

This host scheduling amendment leaves both scientific source snapshots and the
experiment configuration unchanged. The frozen repair controller still owns
preflight, common capability gates, paired training and independent audits.
Default invocation prints a read-only plan.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-successful-base-v1"
AMENDMENT = "resource-reschedule-20261015"
PREFERENCES = {"common": [4, 7], "single": [4, 7], "stream": [7, 4]}


def load_repair(batch, workspace):
    path = batch / "source-v2/scripts/repair_ttcd_verified_preflight.py"
    spec = importlib.util.spec_from_file_location("frozen_ttcd_repair", path)
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    repair.ROOT = workspace
    repair.controller = repair.load_controller(batch / "source/scripts/launch_discoloop_paired.py")
    return repair


def validate_reschedule(repair, batch, config, config_path, receipt):
    repair.validate_zero_updates(batch)
    repair.verify_repaired_source(batch, config, config_path, receipt)
    if (batch / "common/preflight-v2/claim-preflight.json").exists():
        raise ValueError("Repaired preflight already claimed; no automatic retry")
    stop = repair.controller.read_json(batch / AMENDMENT / "stop-waiting-controller.json")
    if (
        stop["scientific_updates"] != 0
        or stop["previous_state"]["state"] != "waiting_for_resources"
    ):
        raise ValueError("Only the preserved zero-update waiting controller may be replaced")
    try:
        os.kill(stop["previous_pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("Previous waiting controller is still alive")
    audits = {}
    for arm in repair.controller.ARMS:
        path = batch.parent / "ttcd-cross-sample-v1" / arm / "audit.json"
        if repair.controller.read_json(path).get("passed") is not True:
            raise ValueError("Preserved prior continual-learning audits must have passed")
        audits[arm] = repair.controller.sha256(path)
    return audits


def idle_row(controller, config, phase, gpu, rows, processes, containers, available):
    if gpu not in rows:
        raise controller.ResourceUnavailable(f"GPU{gpu} is absent")
    row = rows[gpu]
    if (
        row["total_mib"] < 90000
        or row["free_mib"] < 80000
        or row["memory"] > config["runtime"]["gpu_memory_max_mib"]
        or row["utilization"] > config["runtime"]["gpu_utilization_max"]
        or any(p["uuid"] == row["uuid"] for p in processes)
        or any(controller.container_uses_gpu(c, gpu, row["uuid"]) for c in containers)
    ):
        raise controller.ResourceUnavailable(f"GPU{gpu} occupied; no sharing or preemption")
    return {
        **row,
        "gpu": gpu,
        "phase": phase,
        "cpu_set": config["runtime"]["cpu_sets"][phase],
        "host_available_kib": available,
        "utc": controller.utc_now(),
        "gpu_sharing": False,
        "resource_amendment": AMENDMENT,
    }


def install_resource_policy(repair, amendment_path, lock_directory=None):
    controller = repair.controller
    lock_directory = lock_directory or Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"

    def admit_gpu(batch, config, phase, gpu):
        containers, available = controller.admit_cpu(config, phase)
        rows, processes = controller.gpu_snapshot()
        return idle_row(controller, config, phase, gpu, rows, processes, containers, available)

    def wait_gpu(batch, config, phase, deadline, update=lambda _: None):
        lock_directory.mkdir(exist_ok=True)
        while time.monotonic() < deadline:
            rows, _ = controller.gpu_snapshot()
            choices = list(dict.fromkeys([*PREFERENCES[phase], *sorted(rows)]))
            reasons = {}
            for gpu in choices:
                held = (lock_directory / f"gpu-{gpu}.lock").open("a")
                try:
                    try:
                        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise controller.ResourceUnavailable("Exclusive GPU lock held") from error
                    row = admit_gpu(batch, config, phase, gpu)
                    update({"actual_gpu": gpu, "actual_gpu_uuid": row["uuid"]})
                    return held, row
                except controller.ResourceUnavailable as error:
                    held.close()
                    reasons[str(gpu)] = str(error)
                except BaseException:
                    held.close()
                    raise
            update({"state": "waiting_for_resources", "reasons": reasons})
            time.sleep(min(30, max(0, deadline - time.monotonic())))
        raise TimeoutError("No exclusively idle GPU before the resource wait deadline")

    original_build = repair.build_command

    def build_command(config, phase, stage, batch, token_file, row):
        name, command = original_build(config, phase, stage, batch, token_file, row)
        index = command.index(config["runtime"]["image"])
        command[index:index] = [
            "--mount",
            f"type=bind,source={amendment_path},target=/run/experiment/resource-schedule.json,readonly",
            "--env",
            f"LM_ACTUAL_GPU_INDEX={row['gpu']}",
            "--env",
            f"LM_ACTUAL_GPU_UUID={row['uuid']}",
            "--env",
            "LM_RESOURCE_SCHEDULE_PATH=/run/experiment/resource-schedule.json",
            "--env",
            f"WANDB_TAGS=ttcd-continual,resource-rescheduled,gpu-{row['gpu']}",
        ]
        return name, command

    controller.admit_gpu = admit_gpu
    controller.wait_gpu = wait_gpu
    repair.build_command = build_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    plan = {
        "batch": BATCH,
        "executed": args.execute,
        "preferred_gpus": PREFERENCES,
        "fallback": "Any exclusively idle GPU with >=90000MiB total and >=80000MiB free",
        "common_gate_before_paired_arms": True,
        "scientific_config_changed": False,
        "scientific_source_changed": False,
        "automatic_retry": False,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    workspace = args.workspace.resolve()
    config_path = workspace / f"configs/{BATCH}.json"
    config = json.loads(config_path.read_text())
    batch = Path(config["runtime"]["persistent_root"]) / BATCH
    repair = load_repair(batch, workspace)
    controller = repair.controller
    controller.validate_config(config)
    receipt = batch / "preflight-repair-20261015/engineering-receipt.json"
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private online tracking credential file")
    directory = batch / AMENDMENT
    with contextlib.ExitStack() as stack:
        for name in ("controller.lock", "resume-controller-v2.lock", "resource-controller.lock"):
            lock = stack.enter_context((batch / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audits = validate_reschedule(repair, batch, config, config_path, receipt)
        controller.check_space(batch, config)
        with (directory / "claim-controller.json").open("x") as claim:
            json.dump({"pid": os.getpid(), "utc": controller.utc_now()}, claim)
        frozen_script = directory / "controller.py"
        if frozen_script.exists():
            raise FileExistsError("Preserve existing resource controller source")
        shutil.copyfile(Path(__file__).resolve(), frozen_script)
        frozen_script.chmod(0o644)
        amendment = {
            **plan,
            "reason": "User requested other idle GPUs; TTCD common gate precedes paired learning",
            "utc": controller.utc_now(),
            "pid": os.getpid(),
            "source": str(frozen_script),
            "controller_sha256": controller.sha256(frozen_script),
            "config_sha256": controller.sha256(config_path),
            "source_v2_manifest_sha256": controller.sha256(batch / "source-v2-manifest.json"),
            "engineering_receipt_sha256": controller.sha256(receipt),
            "previous_controller_record": str(directory / "stop-waiting-controller.json"),
            "prior_audit_sha256": audits,
            "scientific_updates_at_reschedule": 0,
            "actual_gpu_authority": (
                "Per-stage admission files and Docker device UUID; "
                "original config retains historical GPU8/9"
            ),
        }
        amendment_path = directory / "schedule-amendment.json"
        with amendment_path.open("x") as stream:
            json.dump(amendment, stream, indent=2)
        amendment_path.chmod(0o644)
        root_amendment = batch / "schedule-amendment.json"
        with root_amendment.open("x") as stream:
            json.dump(
                {"current": str(amendment_path), "sha256": controller.sha256(amendment_path)},
                stream,
            )
        controller.write_json(directory / "controller-process.json", amendment)
        controller.write_json(batch / "controller-process-resource.json", amendment)
        install_resource_policy(repair, amendment_path)
        return repair.execute_pipeline(batch, config, config_path, receipt, token)


if __name__ == "__main__":
    raise SystemExit(main())
