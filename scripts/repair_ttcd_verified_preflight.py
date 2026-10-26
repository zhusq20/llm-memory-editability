#!/usr/bin/env python3
"""Freeze a validated preflight repair, then resume the unchanged TTCD plan.

Default invocation only prints a plan. No foundation training, preparation,
historical source or failed preflight is overwritten. Execution is gated by
hash-bound CPU contracts and a real trained-foundation preflight receipt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-successful-base-v1"
REVISION = "v2"
PATCHABLE_SCIENCE = {
    "src/llm_memory_editability/ttcd_editing.py",
    "src/llm_memory_editability/ttcd_verified_base.py",
}
ADDITIONS = (
    "scripts/repair_ttcd_verified_preflight.py",
    # Historical continual CLI is a CPU regression-test dependency only.
    # Successful-base science continues to use run_ttcd_verified_base.py.
    "scripts/run_ttcd_continual.py",
    "tests/test_ttcd_verified_preflight_repair.py",
)


def load_controller(path=None):
    path = path or ROOT / "scripts/launch_discoloop_paired.py"
    spec = importlib.util.spec_from_file_location("verified_preflight_original_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


controller = load_controller()


def validate_zero_updates(batch):
    """Refuse recovery once any adaptation training has been claimed or entered."""
    for phase in ("common", *controller.ARMS):
        for relative in ("training", "claim-train.json", "command-train.json", "result-train.json"):
            if (batch / phase / relative).exists():
                raise ValueError(f"Adaptation training already entered: {phase}/{relative}")
    failure = batch / "common/failure-preflight.json"
    exit_path = batch / "common/exit-preflight.json"
    if not failure.is_file() or not exit_path.is_file():
        raise ValueError("The original failed preflight evidence must remain present")
    if controller.read_json(exit_path).get("returncode") == 0:
        raise ValueError("Recovery applies only to the preserved failed original preflight")
    return {"common_updates": 0, "single_updates": 0, "stream_updates": 0}


def verify_lineage(batch, config, config_path):
    manifest_path = batch / "source-manifest.json"
    manifest = controller.read_json(manifest_path)
    if controller.sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("The original scientific configuration cannot change")
    controller.verify_files(batch / "source", manifest["files"])
    controller.verify_input_overlay(batch, config)
    foundation = controller.verify_foundation(batch, config)
    if foundation != manifest["foundation"]:
        raise ValueError("Foundation weights/gate differ from the original freeze")
    return manifest, foundation


def validate_engineering_receipt(batch, config, config_path, receipt_path, *, workspace=True):
    manifest, foundation = verify_lineage(batch, config, config_path)
    receipt = controller.read_json(receipt_path)
    checks = receipt.get("checks", {})
    if receipt.get("passed") is not True or any(
        checks.get(key) is not True
        for key in ("cpu_scientific_contracts", "trained_foundation_preflight")
    ):
        raise ValueError(
            "Passing CPU scientific contracts and trained-foundation preflight required"
        )
    required = {
        "original_source_manifest_sha256": controller.sha256(batch / "source-manifest.json"),
        "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
        "config_sha256": manifest["config_sha256"],
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Engineering receipt binds another source, config or foundation")
    if receipt.get("scientific_updates") != 0:
        raise ValueError("This repair must precede every adaptation scientific update")
    patched = receipt.get("patched_files", {})
    if not patched or not set(patched).intersection(PATCHABLE_SCIENCE):
        raise ValueError("Engineering receipt must identify the narrow causal-reader repair")
    for relative, expected in patched.items():
        path = Path(relative)
        allowed_test = len(path.parts) == 2 and path.parts[0] == "tests" and path.suffix == ".py"
        if (
            path.is_absolute()
            or ".." in path.parts
            or (relative not in PATCHABLE_SCIENCE and not allowed_test)
        ):
            raise ValueError(f"Repair cannot change this scientific file: {relative}")
        if workspace and controller.sha256(ROOT / path) != expected:
            raise ValueError(
                f"Workspace repair differs from the validated engineering bytes: {relative}"
            )
    evidence = receipt.get("evidence_files", {})
    if not evidence:
        raise ValueError("Engineering checks require persistent hash-bound evidence")
    for name, expected in evidence.items():
        path = Path(name)
        if not path.is_absolute() or controller.sha256(path) != expected:
            raise ValueError(f"Engineering evidence differs: {name}")
    return receipt, manifest, foundation


def freeze_repair(batch, config, config_path, receipt_path):
    """Copy original source and overlay only already-validated narrow changes."""
    zero = validate_zero_updates(batch)
    receipt, old, foundation = validate_engineering_receipt(
        batch, config, config_path, receipt_path
    )
    source, manifest_path = batch / "source-v2", batch / "source-v2-manifest.json"
    if manifest_path.exists():
        frozen = controller.read_json(manifest_path)
        if frozen["engineering_receipt_sha256"] != controller.sha256(receipt_path):
            raise ValueError("A different repair is already frozen; preserve it")
        controller.verify_files(source, frozen["files"])
        return frozen
    if source.exists():
        raise FileExistsError("Preserve the partial repaired source; no automatic replacement")
    repair = batch / "preflight-repair-20261015"
    repair.mkdir(exist_ok=True)
    shutil.copytree(batch / "source", source)
    changes = {}
    for relative, expected in receipt["patched_files"].items():
        target, original = source / relative, batch / "source" / relative
        before = original.read_text() if original.is_file() else ""
        after = (ROOT / relative).read_text()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        diff_path = repair / "diffs" / (relative.replace("/", "__") + ".patch")
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff_path.write_text(
            "".join(
                difflib.unified_diff(
                    before.splitlines(True),
                    after.splitlines(True),
                    fromfile="source/" + relative,
                    tofile="source-v2/" + relative,
                )
            )
        )
        changes[relative] = {
            "before_sha256": old["files"].get(relative),
            "after_sha256": expected,
            "diff_sha256": controller.sha256(diff_path),
            "diff": str(diff_path),
        }
    for relative in ADDITIONS:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    source.chmod(0o755)
    for path in source.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    frozen = {
        "schema": "ttcd-verified-preflight-repair-v2",
        "revision": REVISION,
        "original_source_manifest_sha256": controller.sha256(batch / "source-manifest.json"),
        "engineering_receipt": str(receipt_path),
        "engineering_receipt_sha256": controller.sha256(receipt_path),
        "config_sha256": old["config_sha256"],
        "foundation": foundation,
        "inputs": old["inputs"],
        "panel_sha256": old["panel_sha256"],
        "changes": changes,
        "controller_additions": list(ADDITIONS),
        "files": controller.source_hashes(source),
        "zero_updates_at_repair": zero,
        "utc": controller.utc_now(),
        "scientific_config_changed": False,
    }
    controller.write_json(manifest_path, frozen)
    controller.write_json(
        repair / "repair-amendment.json",
        {
            "reason": "Original trained-weight preflight exposed causal-reader BF16 mismatch "
            "before adaptation updates",
            "original_source_retained": True,
            "original_failure_retained": True,
            "foundation_retrained": False,
            "preparation_repeated": False,
            "configuration_and_inputs_changed": False,
            "new_source_manifest_sha256": controller.sha256(manifest_path),
            "engineering_receipt_sha256": controller.sha256(receipt_path),
            "changes": changes,
            "zero_updates": zero,
            "utc": controller.utc_now(),
        },
    )
    # The previous freeze remains authoritative historical evidence.
    controller.verify_files(batch / "source", old["files"])
    return frozen


def verify_repaired_source(batch, config, config_path, receipt_path):
    receipt, original, _ = validate_engineering_receipt(
        batch, config, config_path, receipt_path, workspace=False
    )
    frozen = controller.read_json(batch / "source-v2-manifest.json")
    if (
        frozen["engineering_receipt_sha256"] != controller.sha256(receipt_path)
        or frozen["inputs"] != original["inputs"]
        or frozen["panel_sha256"] != original["panel_sha256"]
    ):
        raise ValueError("Repaired lineage changed after freezing")
    controller.verify_files(batch / "source-v2", frozen["files"])
    for relative, expected in receipt["patched_files"].items():
        if frozen["files"].get(relative) != expected:
            raise ValueError("Repaired source is not the independently validated science")


def output_directory(batch, phase, stage):
    return batch / phase / "preflight-v2" if stage == "preflight" else batch / phase


def build_command(config, phase, stage, batch, token_file, row):
    name, original = controller.build_command(config, phase, stage, batch, token_file, row)
    command = list(original)
    command[command.index("--name") + 1] = name + "-v2"
    for index, value in enumerate(command):
        if value.startswith("type=bind,source=" + str(batch / "source") + ","):
            command[index] = value.replace(
                "source=" + str(batch / "source") + ",",
                "source=" + str(batch / "source-v2") + ",",
                1,
            )
    if stage == "preflight":
        command[command.index("--out") + 1] += "/preflight-v2"
    return name + "-v2", command


def run_stage(batch, config, phase, stage, token_file, deadline, row, update):
    out = output_directory(batch, phase, stage)
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o755)
    with (out / f"claim-{stage}.json").open("x") as stream:
        json.dump(
            {
                "stage": stage,
                "phase": phase,
                "pid": os.getpid(),
                "revision": REVISION,
                "utc": controller.utc_now(),
            },
            stream,
        )
    name, command = build_command(config, phase, stage, batch, token_file, row)
    if controller.inspect_container(name) is not None:
        raise FileExistsError("Repair stage container already exists; no automatic retry")
    controller.write_json(
        out / f"command-{stage}.json", {"argv": command, "utc": controller.utc_now()}
    )
    controller.write_json(out / f"admission-{stage}.json", row)
    update({"state": "launching", "stage": stage, "container": name})
    with (out / f"{stage}.log").open("x") as log:
        launched = subprocess.run(command, capture_output=True, text=True, timeout=60)
        log.write(launched.stderr)
        log.flush()
    if launched.returncode:
        controller.write_json(
            out / f"exit-{stage}.json", {"returncode": launched.returncode, "launch_failed": True}
        )
        raise RuntimeError(f"Repair stage launch failed: {launched.stderr.strip()}")
    update(
        {
            "state": "running",
            "stage": stage,
            "container": name,
            "container_id": launched.stdout.strip(),
        }
    )
    with (out / f"{stage}.log").open("a") as log:
        code = controller.monitor_container(name, log, deadline, update)
    controller.write_json(
        out / f"exit-{stage}.json",
        {"returncode": code, "container": name, "utc": controller.utc_now()},
    )
    if (
        stage in ("gate", "calibrate", "preflight")
        and code == 0
        and controller.read_json(out / f"result-{stage}.json").get("passed") is not True
    ):
        return 3
    return code


def run_phase(batch, config, config_path, receipt_path, phase, token_file):
    out = batch / phase
    out.mkdir(exist_ok=True)
    out.chmod(0o755)
    stages = controller.COMMON_STAGES if phase == "common" else controller.ARM_STAGES
    state = {"phase": phase, "stages": list(stages), "revision": REVISION, "automatic_retry": False}

    def update(value):
        state.update(value)
        controller.write_json(
            out / "controller-state-v2.json", {**state, "updated_utc": controller.utc_now()}
        )

    try:
        with (out / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if phase != "common":
                controller.require_common_gate(batch, config)
            held, row = controller.wait_gpu(
                batch,
                config,
                phase,
                time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600,
                update,
            )
            with held:
                deadline = time.monotonic() + config["runtime"]["timeout_hours"] * 3600
                for index, stage in enumerate(stages):
                    verify_repaired_source(batch, config, config_path, receipt_path)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Phase hard timeout before launching its next stage")
                    if index:
                        row = controller.admit_gpu(batch, config, phase, row["gpu"])
                    code = run_stage(batch, config, phase, stage, token_file, deadline, row, update)
                    if code:
                        update({"state": "needs_attention", "stage": stage, "returncode": code})
                        return code
            update({"state": "complete", "finished_utc": controller.utc_now()})
            return 0
    except Exception:
        failure = {"state": "needs_attention", "traceback": traceback.format_exc()}
        update(failure)
        controller.write_json(out / "controller-failure-v2.json", failure)
        return 1


def execute_pipeline(batch, config, config_path, receipt_path, token_file):
    state = {"state": "running", "revision": REVISION, "automatic_retry": False}

    def update(value):
        state.update(value)
        controller.write_json(
            batch / "controller-state-v2.json", {**state, "updated_utc": controller.utc_now()}
        )

    code = run_phase(batch, config, config_path, receipt_path, "common", token_file)
    if code:
        update(
            {
                "state": "needs_attention",
                "phase": "common",
                "returncode": code,
                "paired_scientific_updates": 0,
            }
        )
        return code
    controller.require_common_gate(batch, config)
    update({"state": "paired_running"})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {
            phase: pool.submit(
                run_phase, batch, config, config_path, receipt_path, phase, token_file
            )
            for phase in controller.ARMS
        }
        exits = {phase: task.result() for phase, task in tasks.items()}
    update(
        {
            "state": "complete" if not any(exits.values()) else "needs_attention",
            "exit_codes": exits,
            "finished_utc": controller.utc_now(),
        }
    )
    return 1 if any(exits.values()) else 0


def main():
    global controller
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--engineering-receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    config = controller.read_json(args.config)
    controller.validate_config(config)
    plan = {
        "batch": BATCH,
        "revision": REVISION,
        "executed": args.execute,
        "engineering_receipt_required": str(args.engineering_receipt),
        "source": "source-v2",
        "common_stages": list(controller.COMMON_STAGES),
        "paired_stages": list(controller.ARM_STAGES),
        "preflight_output": "common/preflight-v2",
        "original_source_and_failure_retained": True,
        "foundation_retraining": False,
        "preparation_repeated": False,
        "configuration_changed": False,
        "paired_gpus": {"single": 8, "stream": 9},
        "prior_audits_and_actual_idle_required": True,
        "gpu_sharing": False,
        "automatic_retry": False,
        "freeze_only": args.freeze_only,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    if not args.config.resolve().is_relative_to(ROOT):
        raise ValueError("Use the reviewed configuration inside the project")
    if not args.freeze_only and (
        not args.wandb_token_file.is_file() or args.wandb_token_file.stat().st_mode & 0o077
    ):
        raise ValueError("Provide the existing private online tracking credential file")
    batch = controller.persistent_batch(config)
    controller = load_controller(batch / "source/scripts/launch_discoloop_paired.py")
    with (
        (batch / "controller.lock").open("a") as original_lock,
        (batch / "resume-controller-v2.lock").open("a") as resume_lock,
    ):
        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(resume_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = freeze_repair(batch, config, args.config, args.engineering_receipt)
        if args.freeze_only:
            return 0
        controller.check_space(batch, config)
        claim = batch / "resume-controller-claim-v2.json"
        with claim.open("x") as stream:
            json.dump(
                {
                    "pid": os.getpid(),
                    "utc": controller.utc_now(),
                    "source_manifest_sha256": controller.sha256(batch / "source-v2-manifest.json"),
                    "engineering_receipt_sha256": frozen["engineering_receipt_sha256"],
                },
                stream,
            )
        controller.write_json(
            batch / "controller-process-v2.json",
            {"pid": os.getpid(), "utc": controller.utc_now(), **plan},
        )
        return execute_pipeline(
            batch, config, args.config, args.engineering_receipt, args.wandb_token_file
        )


if __name__ == "__main__":
    raise SystemExit(main())
