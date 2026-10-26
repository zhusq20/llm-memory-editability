"""Reassign only an unstarted DiscoLoop after the other two arms fully finish.

The science source and containers remain the original immutable source-v2.
This host-only scheduling amendment never stops a model container or repeats
any started stage. Run only if the original controller remains resource-waiting.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = Path("/ossfs/workspace/llm-memory-editability/ce-cross-sample-double-new-v1")


def read(path):
    return json.loads(Path(path).read_text())


def eligible(batch):
    """Fail closed before changing the original scheduler or launching anything."""
    outcomes = {}
    for arm in ("transformer", "loop"):
        completed = read(batch / arm / "completion-manifest.json")
        if completed.get("state") != "complete" or completed.get("passed") is not True:
            raise ValueError("Both running architectures must finish all four stages first")
        if completed.get("scientific_updates") != 4480:
            raise ValueError("Unexpected completed science budget")
        outcomes[arm] = completed
    directory = batch / "discoloop"
    waiting = read(directory / "controller-state.json")
    if waiting.get("state") not in ("waiting_for_resources", "waiting_for_exclusive_idle_gpu"):
        raise ValueError("DiscoLoop must still be waiting, with no active science")
    if (directory / "launch-evaluate.json").exists():
        raise FileExistsError("A started DiscoLoop write can never be resubmitted")
    prepared = read(directory / "output/prepare/result.json")
    if not prepared.get("passed") or prepared.get("scientific_optimizer_updates") != 0:
        raise ValueError("DiscoLoop preparation must already have passed")
    if (batch / "report/launch-report.json").exists():
        raise FileExistsError("An existing comparison may not be replaced")
    return outcomes, waiting


def main():
    original = read(BATCH / "controller-process.json")
    source = BATCH / original["source_revision"]
    spec = importlib.util.spec_from_file_location(
        "frozen_ce_cross_sample_controller", source / "scripts/launch_ce_cross_sample.py"
    )
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    manifest = controller.verify_source(source, original["source_sha256"])
    plan_path = source / "configs/ce-cross-sample-double-new-v1.json"
    if controller.sha(plan_path) != original["plan_sha256"]:
        raise ValueError("Frozen science configuration changed")
    plan = read(plan_path)
    outcomes, waiting = eligible(BATCH)
    active = controller.docker("ps", "--format", "{{.Names}}").stdout.decode().splitlines()
    if any(name.startswith(controller.BATCH + "-") for name in active):
        raise RuntimeError("Never hand off while this batch has a running model container")
    old_pid = original["pid"]
    command = Path(f"/proc/{old_pid}/cmdline").read_bytes().split(b"\0")
    if not any(value.endswith(b"scripts/launch_ce_cross_sample.py") for value in command):
        raise ValueError("The original PID no longer identifies this batch controller")
    intervention = BATCH / "scheduling-intervention"
    intervention.mkdir(mode=0o700, exist_ok=False)
    controller.write(
        intervention / "amendment.json",
        {
            "reason": (
                "External work reclaimed the admitted GPU after preparation; reselect an idle card"
            ),
            "original_controller": original,
            "waiting_state": waiting,
            "original_disco_resource_admission": read(BATCH / "discoloop/resource-admission.json"),
            "completed_architectures": outcomes,
            "source_sha256": manifest,
            "host_scheduler_sha256": controller.sha(Path(__file__)),
            "scientific_source_changed": False,
            "model_containers_stopped": 0,
            "repeated_scientific_updates": 0,
            "utc_epoch": time.time(),
        },
    )
    # Only this idle scheduler is terminated. Docker containers are independent.
    os.kill(old_pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    with (BATCH / "controller.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("The original controller did not release its lock") from None
                time.sleep(0.2)
        eligible(BATCH)
        controller.write(
            intervention / "controller-process.json",
            {
                "pid": os.getpid(),
                "source_sha256": manifest,
                "previous_idle_scheduler_signalled": old_pid,
                "signal": "SIGTERM",
                "original_controller_lock_released": True,
                "model_containers_stopped": 0,
                "utc_epoch": time.time(),
            },
        )
        try:
            outcomes["discoloop"] = controller.execute_arm(
                plan, BATCH, source, manifest, "discoloop"
            )
            controller.run_stage(plan, BATCH, source, manifest, "report", "report")
            cloud = BATCH / "report/output/cloud-verification.json"
            if read(cloud).get("passed") is not True:
                raise ValueError("The report full cloud JSON must verify")
            completion = {
                "state": "complete",
                "passed": True,
                "outcomes": outcomes,
                "source_sha256": manifest,
                "new_foundation_updates": 0,
                "new_scientific_updates": 13440,
                "engineering_updates": sum(row["engineering_updates"] for row in outcomes.values()),
                "independent_audits_passed": True,
                "full_cloud_json_verified": True,
                "report_cloud_verification_sha256": controller.sha(cloud),
                "report_output": str(cloud.parent),
                "scheduling_amendment": str(intervention / "amendment.json"),
            }
            controller.write(BATCH / "completion-manifest.json", completion)
            controller.write(BATCH / "controller-state.json", completion)
        except BaseException as error:
            controller.write(
                intervention / "failure.json",
                {"error": repr(error), "traceback": traceback.format_exc()},
            )
            raise


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
