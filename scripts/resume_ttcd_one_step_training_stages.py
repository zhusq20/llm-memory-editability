#!/usr/bin/env python3
"""Resume only zero-update train launches, retaining passed prepare/CUDA preflight."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

BATCH = "ttcd-one-step-training-v1"
ROOT = Path("/ossfs/workspace/llm-memory-editability") / BATCH
ARMS = ("transformer", "loop", "discoloop")
OLD_PID = 3200759
CONTROLLER_SHA256 = "1b8cd58da0c66e32bd93b469c42542df1679ce00c7ee64b17690e29f57af5f65"
POLICY_SHA256 = "aca660bddbf9a0f077e823548acc4509d44ff32c35e0a56b7324939fffe44304"
POLICY_MANIFEST_SHA256 = "c7476b7677eb54a579b242aae98e7e111b8a017634cdfa3a3a6d1e1a2615c12e"
REPAIR = "launch-stage-repair-20261016"
OUTPUT_ERROR = "Terminal checkpoint must use this phase's isolated output"


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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_frozen(batch):
    controller_path = batch / "transformer/source/scripts/launch_ttcd_one_step_training.py"
    policy_root = batch / "launch-repair-20261016"
    policy_path = policy_root / "source/scripts/ttcd_one_step_gpu56_resources.py"
    if sha256(controller_path) != CONTROLLER_SHA256:
        raise ValueError("Original science controller changed")
    if (
        sha256(policy_root / "source-manifest.json") != POLICY_MANIFEST_SHA256
        or sha256(policy_path) != POLICY_SHA256
    ):
        raise ValueError("Previously validated sharing policy changed")
    index = read_json(policy_root / "source-manifest.json")
    snapshot_path = policy_root / "external-gpu-processes.json"
    if sha256(snapshot_path) != index["external_snapshot_sha256"]:
        raise ValueError("Frozen external identity bindings changed")
    controller = load_module("stage_repair_original_controller", controller_path)
    policy = load_module("stage_repair_original_policy", policy_path)
    snapshot = read_json(snapshot_path)
    policy.validate_snapshot(snapshot)
    return controller, policy, snapshot, controller_path, policy_path


def verify_repair_source(batch):
    root = batch / REPAIR
    manifest = root / "source-manifest.json"
    index = read_json(manifest)
    files = index["files"]
    required = {
        "scripts/resume_ttcd_one_step_training_stages.py",
        "tests/test_ttcd_one_step_training_stages.py",
    }
    if not required <= set(files) or index.get("engineering_passed") is not True:
        raise ValueError("Freeze and verify the separate recovery wrapper/tests before execution")
    for relative, expected in files.items():
        path = root / "source" / relative
        if path.is_symlink() or not path.resolve().is_relative_to((root / "source").resolve()):
            raise ValueError("Recovery source escaped its frozen directory")
        if sha256(path) != expected:
            raise ValueError("Recovery source bytes changed")
    if sha256(__file__) != files["scripts/resume_ttcd_one_step_training_stages.py"]:
        raise ValueError("Run the newly frozen recovery wrapper bytes")
    return {"path": str(manifest), "sha256": sha256(manifest)}


def inspect_failure(batch, controller, is_running=process_running):
    process = read_json(batch / "controller-process.json")
    state = read_json(batch / "controller-state.json")
    if process.get("pid") != OLD_PID or is_running(OLD_PID):
        raise RuntimeError("The failed prior controller must be stopped before stage recovery")
    if state.get("state") != "needs_attention" or state.get("outcomes"):
        raise ValueError("Only the diagnosed failed stage launch may be recovered")
    if sha256(batch / "source-manifest.json") != process["source_manifest_sha256"]:
        raise ValueError("Frozen science source changed")
    plan_path = batch / "transformer/source/configs/one-step-training.json"
    if sha256(plan_path) != process["config_sha256"]:
        raise ValueError("Frozen science plan changed")
    plan = read_json(plan_path)
    controller.validate_plan(plan)
    kept = {}
    for arm in ARMS:
        controller.transport.verify_snapshot(batch, arm)
        arm_state = read_json(batch / arm / "controller-state.json")
        if arm_state.get("state") != "needs_attention" or arm_state.get("completed_stages") != [
            "prepare",
            "preflight",
        ]:
            raise ValueError("Each arm must have only prepare and CUDA preflight complete")
        stages = batch / arm / "stages"
        if {p.name for p in stages.iterdir()} != {"prepare", "preflight", "train"}:
            raise ValueError("A later science stage already exists")
        for stage, output in (("prepare", "evaluation"), ("preflight", "training-run")):
            exit_path = stages / stage / "exit.json"
            result_path = batch / arm / output / f"result-{stage}.json"
            result = read_json(result_path)
            if read_json(exit_path).get("returncode") != 0 or result.get("passed") is not True:
                raise ValueError("All existing prepare/preflight successes must remain valid")
            if stage == "preflight" and (
                result.get("writer_steps") != 1
                or result.get("all_four_losses_exact") is not True
                or result.get("all_parameter_gradients_exact") is not True
                or result.get("all_fast_states_exact") is not True
                or result.get("base_parameter_updates") != 0
                or result.get("optimizer_updates") != 0
            ):
                raise ValueError("Reuse only the exact zero-update one-step CUDA preflight")
            kept.update({str(path): sha256(path) for path in (exit_path, result_path)})
        out = batch / arm / "training-run"
        training = out / "training"
        if training.exists() and (
            training.is_symlink() or not training.is_dir() or any(training.iterdir())
        ):
            raise ValueError(
                "Training directory must be empty; existing model updates cannot restart"
            )
        if list(out.rglob("*.pt")) or list(out.rglob("metrics.jsonl")):
            raise ValueError("A checkpoint or scientific training metric exists")
        if any(out.glob("result-train*.json")) or any(out.glob("status-train*.json")):
            raise ValueError("A scientific training result already exists")
        if any((batch / arm / "evaluation").glob("result-*.json")) and {
            p.name for p in (batch / arm / "evaluation").glob("result-*.json")
        } != {"result-prepare.json"}:
            raise ValueError("Evaluation advanced beyond graph preparation")
        container = controller.resources.inspect_container(f"{BATCH}-{arm}-train")
        if arm == "loop":
            if container is not None or (stages / "train/transport.json").exists():
                raise ValueError("Loop train must have failed before container creation")
            if "CalledProcessError" not in arm_state.get(
                "traceback", ""
            ) or "top" not in arm_state.get("traceback", ""):
                raise ValueError("Loop failure must be the diagnosed Docker-top exit race")
        else:
            failure = read_json(out / "failure-train.json")
            if (
                read_json(stages / "train/exit.json").get("returncode") != 1
                or OUTPUT_ERROR not in failure.get("traceback", "")
                or container is None
                or container["State"]["Running"]
                or container["State"]["ExitCode"] != 1
            ):
                raise ValueError(
                    "Train must have exited only on the diagnosed output-path assertion"
                )
    if any(
        c["Name"].lstrip("/").startswith(f"{BATCH}-")
        for c in controller.resources.running_containers(["lm-memory"])
    ):
        raise RuntimeError("This batch still has a running container")
    return {
        "plan": plan,
        "old_process": process,
        "preserved_successes": kept,
        "base_parameter_updates": 0,
    }


def install_stage_repairs(controller, policy, snapshot):
    original_cohort, original_command = controller.gpu_cohort, controller.build_command

    def cohort(*args, **kwargs):
        try:
            return original_cohort(*args, **kwargs)
        except subprocess.CalledProcessError as error:
            stderr = (
                error.stderr.decode(errors="replace")
                if isinstance(error.stderr, bytes)
                else (error.stderr or "")
            )
            command = error.cmd if isinstance(error.cmd, (list, tuple)) else []
            if "top" in command and any(
                text in stderr.lower()
                for text in (
                    "is not running",
                    "container not running",
                    "no such process",
                    "no such container",
                )
            ):
                raise controller.resources.ResourceUnavailable(
                    "A cohort container exited during Docker top; refresh admission"
                ) from error
            raise

    def command(plan, arm, stage, batch, token, row, *args, **kwargs):
        name, argv = original_command(plan, arm, stage, batch, token, row, *args, **kwargs)
        argv = list(argv)
        if stage == "preflight" or stage.startswith("train"):
            argv[argv.index("--out") + 1] = f"/frozen/results/{BATCH}/{arm}/training-run"
        if stage == "train":
            name += "-output-v2"
            argv[argv.index("--name") + 1] = name
        return name, argv

    controller.gpu_cohort = cohort
    controller.build_command = command
    policy.install(controller, snapshot)
    remaining = tuple(stage for stage in controller.STAGES if stage not in ("prepare", "preflight"))

    def run_arm(plan, arm, batch, token):
        state = {
            "architecture": arm,
            "state": "resuming",
            "completed_stages": ["prepare", "preflight"],
            "recovery": REPAIR,
        }

        def update(values):
            state.update(values)
            state["utc"] = controller.resources.utc_now()
            controller.write_json(batch / arm / "controller-state.json", state)

        try:
            for stage in remaining:
                controller.run_stage(plan, arm, stage, batch, token, update)
                state["completed_stages"].append(stage)
            update({"state": "complete", "new_base_parameter_updates": 10000})
            return state
        except BaseException:
            update({"state": "needs_attention", "traceback": traceback.format_exc()})
            raise

    controller.run_arm = run_arm


def archive_failed_launches(batch):
    for arm in ARMS:
        training = batch / arm / "training-run/training"
        if training.exists() and (training.is_symlink() or any(training.iterdir())):
            raise ValueError("Never archive a nonempty training directory for a fresh restart")
    archive = batch / REPAIR / "failed-attempts"
    archive.parent.mkdir(exist_ok=True)
    archive.mkdir(exist_ok=False)
    for relative in (
        "controller-process.json",
        "controller-state.json",
        "schedule-amendment.json",
        *(f"{arm}/controller-state.json" for arm in ARMS),
    ):
        path = batch / relative
        if path.exists():
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for arm in ARMS:
        target = archive / arm / "stages/train"
        target.parent.mkdir(parents=True, exist_ok=True)
        (batch / arm / "stages/train").rename(target)
        for name in (
            "failure-train.json",
            "environment-train.json",
            "wandb-train.json",
            "wandb",
            "training",
        ):
            path = batch / arm / "training-run" / name
            if not path.exists():
                continue
            if name == "training" and any(path.iterdir()):
                raise ValueError("Never archive a nonempty training directory for a fresh restart")
            target = archive / arm / "training-run" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
    return archive


def execute_repair(batch, controller, policy, snapshot, token, repair_source):
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the original private W&B token")
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = inspect_failure(batch, controller)
        install_stage_repairs(controller, policy, snapshot)
        os.umask(0o022)
        archive = archive_failed_launches(batch)
        record = {
            "pid": os.getpid(),
            "utc": controller.resources.utc_now(),
            "old_pid": OLD_PID,
            "controller_source_sha256": CONTROLLER_SHA256,
            "controller_source": str(
                batch / "transformer/source/scripts/launch_ttcd_one_step_training.py"
            ),
            "resource_policy_sha256": POLICY_SHA256,
            "resource_policy": str(
                batch / "launch-repair-20261016/source/scripts/ttcd_one_step_gpu56_resources.py"
            ),
            "source_manifest_sha256": evidence["old_process"]["source_manifest_sha256"],
            "config_sha256": evidence["old_process"]["config_sha256"],
            "recovery_source": repair_source,
            "failed_attempts": str(archive),
            "preserved_successes": evidence["preserved_successes"],
            "base_parameter_updates_before_recovery": 0,
            "new_base_parameter_updates": 30000,
            "automatic_retry": False,
            "science_source_changed": False,
            "completed_stages_reused": ["prepare", "preflight"],
            "gpu_allowed": [5, 6],
            "output_mapping": "/frozen/results/ttcd-one-step-training-v1/{arm}/training-run",
        }
        controller.write_json(batch / REPAIR / "controller-process.json", record)
        controller.write_json(batch / "controller-process.json", record)
        controller.write_json(
            batch / "schedule-amendment.json", {"recovery": str(batch / REPAIR), **record}
        )
        return controller.execute(evidence["plan"], batch, token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    controller, policy, snapshot, _, _ = load_frozen(ROOT)
    if not args.execute:
        print(
            json.dumps({"executed": False, "evidence": inspect_failure(ROOT, controller)}, indent=2)
        )
        return 0
    source = verify_repair_source(ROOT)
    return execute_repair(ROOT, controller, policy, snapshot, args.wandb_token_file, source)


if __name__ == "__main__":
    raise SystemExit(main())
