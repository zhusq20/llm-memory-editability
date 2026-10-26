#!/usr/bin/env python3
"""Recover only the diagnosed pre-container permission failure of one-step training."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
from pathlib import Path

BATCH = "ttcd-one-step-training-v1"
ROOT = Path("/ossfs/workspace/llm-memory-editability") / BATCH
ARMS = ("transformer", "loop", "discoloop")
OLD_PID = 3161452
CONTROLLER_SHA256 = "1b8cd58da0c66e32bd93b469c42542df1679ce00c7ee64b17690e29f57af5f65"
REPAIR = "launch-repair-20261016"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def process_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_controller(batch):
    path = batch / "transformer/source/scripts/launch_ttcd_one_step_training.py"
    if path.is_symlink() or sha256(path) != CONTROLLER_SHA256:
        raise ValueError("The exact frozen controller bytes must remain unchanged")
    spec = importlib.util.spec_from_file_location("frozen_one_step_training_repair", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module, path


def permission_directories(batch):
    """Only seven public traversal/output directories may change; source stays private."""
    batch = Path(batch)
    if batch.name != BATCH:
        raise ValueError("Only this newly submitted batch may receive the repair")
    directories = [batch]
    for arm in ARMS:
        directories.extend((batch / arm, batch / arm / "evaluation"))
    for path in directories:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Repair target must be an existing real directory: {path}")
        if not path.resolve().is_relative_to(batch.resolve()):
            raise ValueError("Repair directory escaped the new batch")
    return directories


def inspect_failure(batch, controller, is_running=process_running):
    """Fail closed unless every arm stopped before Docker create and before science."""
    permission_directories(batch)
    process = read_json(batch / "controller-process.json")
    state = read_json(batch / "controller-state.json")
    if process.get("pid") != OLD_PID or is_running(OLD_PID):
        raise RuntimeError("The original controller must have exited; no concurrent recovery")
    if state.get("state") != "needs_attention" or state.get("outcomes"):
        raise ValueError("Only the original failed startup may be repaired")
    if set(state.get("failures", {})) != set(ARMS):
        raise ValueError("All three original pre-container failures must be retained")
    if sha256(batch / "source-manifest.json") != process["source_manifest_sha256"]:
        raise ValueError("Original frozen source manifest changed")
    diagnostic = read_json(batch / "launch-create-diagnostic.json")
    if (
        diagnostic.get("returncode") != 1
        or diagnostic.get("model_started") is not False
        or "permission denied" not in diagnostic.get("stderr", "")
        or str(batch / "transformer") not in diagnostic.get("stderr", "")
    ):
        raise ValueError("The recorded failure must be the diagnosed new-batch mount permission")
    for name in ("report", "control-report"):
        output = batch / name
        if output.exists() and any(path.is_file() for path in output.rglob("*")):
            raise ValueError("A scientific aggregate already exists; no recovery may overwrite it")
    for arm in ARMS:
        controller.transport.verify_snapshot(batch, arm)
        arm_state = read_json(batch / arm / "controller-state.json")
        if (
            arm_state.get("state") != "needs_attention"
            or arm_state.get("completed_stages") != []
            or arm_state.get("stage") != "prepare"
        ):
            raise ValueError("An arm advanced beyond the known zero-update failure")
        stage = batch / arm / "stages/prepare"
        transport = read_json(stage / "transport.json")
        name = f"{BATCH}-{arm}-prepare"
        if (
            transport.get("state") != "creating"
            or transport.get("container_id")
            or transport.get("container") != name
            or controller.resources.inspect_container(name) is not None
        ):
            raise ValueError("A container was created; this pre-container repair does not apply")
        if {path.name for path in stage.parent.iterdir()} != {"prepare"}:
            raise ValueError("Later stage claims exist; no model restart is authorized")
        if (batch / arm / "training-run/training").exists():
            raise ValueError("Training artifacts exist; no duplicate model updates")
        for output_name in ("evaluation", "training-run"):
            output = batch / arm / output_name
            if output.exists() and any(path.is_file() for path in output.rglob("*")):
                raise ValueError("Scientific output already exists; no result may be overwritten")
    for container in controller.resources.running_containers(["lm-memory"]):
        if container["Name"].lstrip("/").startswith(f"{BATCH}-"):
            raise RuntimeError("A container from this batch is already running")
    plan_path = batch / "transformer/source/configs/one-step-training.json"
    plan = read_json(plan_path)
    if sha256(plan_path) != process["config_sha256"]:
        raise ValueError("Frozen science settings differ from the failed original submission")
    controller.validate_plan(plan)
    return {
        "plan": plan,
        "old_process": process,
        "controller_source_sha256": CONTROLLER_SHA256,
        "source_manifest_sha256": process["source_manifest_sha256"],
        "model_started": False,
        "base_parameter_updates_before_repair": 0,
        "permission_targets": [str(path) for path in permission_directories(batch)],
    }


def archive_failed_attempt(batch):
    """Move failed claims aside only once; never remove or overwrite prior evidence."""
    archive = batch / REPAIR / "failed-attempts"
    archive.parent.mkdir(exist_ok=True)
    archive.mkdir(exist_ok=False)
    for relative in (
        "controller-process.json",
        "controller-state.json",
        "launch-create-diagnostic.json",
        *(f"{arm}/controller-state.json" for arm in ARMS),
    ):
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(batch / relative, destination)
    for arm in ARMS:
        destination = archive / arm / "stages/prepare"
        destination.parent.mkdir(parents=True, exist_ok=True)
        (batch / arm / "stages/prepare").rename(destination)
    return archive


def repair_permissions(batch):
    records = []
    for path in permission_directories(batch):
        before = stat.S_IMODE(path.stat().st_mode)
        path.chmod(0o755)
        records.append({"path": str(path), "before": oct(before), "after": "0o755"})
    return records


def install_shared_resources(controller, snapshot_path):
    """Load only the verified host scheduling amendment, before any worker starts."""
    snapshot_path = Path(snapshot_path)
    amendment = snapshot_path.parent
    manifest = read_json(amendment / "source-manifest.json")
    if (
        manifest.get("engineering_passed") is not True
        or manifest.get("external_snapshot_sha256") != sha256(snapshot_path)
        or manifest.get("controller_source_sha256") != CONTROLLER_SHA256
    ):
        raise ValueError("Shared resources require the verified frozen host amendment")
    for relative, expected in manifest["files"].items():
        if sha256(amendment / "source" / relative) != expected:
            raise ValueError("The frozen host scheduling amendment changed")
    policy_path = Path(__file__).parent / "ttcd_one_step_gpu56_resources.py"
    if policy_path != amendment / "source/scripts/ttcd_one_step_gpu56_resources.py":
        raise ValueError("Run the frozen recovery entry alongside its scheduling policy")
    spec = importlib.util.spec_from_file_location("one_step_gpu56_sharing", policy_path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    policy.install(controller, read_json(snapshot_path))
    return {
        "source_manifest": str(amendment / "source-manifest.json"),
        "source_manifest_sha256": sha256(amendment / "source-manifest.json"),
        "external_snapshot": str(snapshot_path),
        "external_snapshot_sha256": sha256(snapshot_path),
        "allowed_gpus": [5, 6],
        "external_processes_are_read_only": True,
    }


def execute_repair(batch, controller, controller_path, token, scheduling=None):
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the original private W&B credential file")
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = inspect_failure(batch, controller)
        os.umask(0o022)
        archive = archive_failed_attempt(batch)
        permissions = repair_permissions(batch)
        for arm in ARMS:
            controller.transport.verify_snapshot(batch, arm)
        record = {
            "pid": os.getpid(),
            "utc": controller.resources.utc_now(),
            "old_pid": OLD_PID,
            "controller_source": str(controller_path),
            "controller_source_sha256": CONTROLLER_SHA256,
            "source_manifest_sha256": evidence["source_manifest_sha256"],
            "config_sha256": evidence["old_process"]["config_sha256"],
            "repair_wrapper": str(Path(__file__).resolve()),
            "repair_wrapper_sha256": sha256(__file__),
            "failed_attempts": str(archive),
            "base_parameter_updates_before_repair": 0,
            "new_base_parameter_updates": 30000,
            "automatic_retry": False,
            "recovery_reason": (
                "Diagnosed Docker create mount traversal denied; no container/model started"
            ),
            "future_umask": "0o022",
            "permissions": permissions,
            "resource_scheduling_amendment": scheduling,
        }
        controller.resources.write_json(batch / REPAIR / "controller-process.json", record)
        controller.resources.write_json(batch / "controller-process.json", record)
        return controller.execute(evidence["plan"], batch, token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--sharing-snapshot", type=Path)
    parser.add_argument(
        "--wandb-token-file",
        type=Path,
        default=Path.home() / ".cache/llm-memory-ttcd/wandb-token",
    )
    args = parser.parse_args()
    controller, path = load_controller(ROOT)
    if not args.execute:
        print(
            json.dumps({"executed": False, "evidence": inspect_failure(ROOT, controller)}, indent=2)
        )
        return 0
    scheduling = (
        install_shared_resources(controller, args.sharing_snapshot)
        if args.sharing_snapshot is not None
        else None
    )
    return execute_repair(ROOT, controller, path, args.wandb_token_file, scheduling)


if __name__ == "__main__":
    raise SystemExit(main())
