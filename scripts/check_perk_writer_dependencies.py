"""Check the PERK prerequisite for writer calibration without launching a job.

The roadmap requires the original comparison and independent audits to finish
before the writer extension is frozen. This check reads the original receipts
and rechecks their source, checkpoint and prediction hashes. Passing it means
calibration may begin; it is not a frozen scientific recipe or a launch command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

DEFAULT_BASE = Path("/ossfs/workspace/research-recovery/perk-paper-reproduction-v1")
ARMS = ("perk", "fticr")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_executor(base, frozen):
    relative = "scripts/execute_perk_reproduction.py"
    path = base / "frozen/source" / relative
    if sha(path) != frozen["source_files"][relative]:
        raise ValueError("Original frozen execution source changed")
    # SourceFileLoader may write __pycache__ into the immutable original batch.
    module = ModuleType("_perk_writer_prerequisite_executor")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def inspect_dependencies(base):
    base = Path(base).resolve()
    result = {
        "batch": "perk-writer-control-preparation-v1",
        "prerequisite": str(base),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_calibration": False,
        "scientific_extension_frozen": False,
        "launch_performed": False,
        "pending": [],
        "invalid": [],
        "arms": {},
    }
    state_path = base / "controller-state.json"
    if not state_path.exists():
        result["pending"].append("original controller state is absent")
        return result
    try:
        state = read(state_path)
        result["controller_state"] = state["state"]
        result["active"] = [
            {"arm": row["arm"], "stage": row["stage"], "container": row["container"]}
            for row in state.get("active", [])
        ]
        if state.get("failed") or state["state"] == "finished_with_failures":
            result["invalid"].append("original execution has recorded failures")
        if state["state"] != "complete" or state.get("scientific_training") != "complete":
            result["pending"].append("original comparison is not complete")
        for arm in ARMS:
            root = base / "runs" / arm
            progress = root / "train/training-progress.json"
            completion = root / "train/completion.json"
            result["arms"][arm] = {
                "training_complete": completion.exists(),
                "independent_audit_present": (root / "audit.json").exists(),
            }
            if progress.exists():
                row = read(progress)
                result["arms"][arm]["completed_updates"] = row.get(
                    "completed_outer_updates", row.get("optimizer_steps")
                )
            for relative in (
                "train/completion.json",
                "native-training-endpoint.json",
                "checkpoint-selection.json",
                "audit.json",
            ):
                if not (root / relative).exists():
                    result["pending"].append(f"{arm}/{relative} is absent")
        for relative in ("completion-manifest.json", "wandb-cloud-verification.json"):
            if not (base / relative).exists():
                result["pending"].append(f"{relative} is absent")
        if result["pending"] or result["invalid"]:
            return result

        frozen_path = base / "frozen/freeze.json"
        freeze_sha = sha(frozen_path)
        frozen = read(frozen_path)
        executor = frozen_executor(base, frozen)
        executor.verify_freeze(base, inputs=False)
        manifest = read(base / "completion-manifest.json")
        if not all(
            manifest.get(key) is True
            for key in (
                "passed",
                "scientific_training",
                "independent_full_test_audits",
                "cloud_verified",
            )
        ):
            raise ValueError("Original completion manifest did not pass all stages")
        if manifest["freeze_sha256"] != freeze_sha:
            raise ValueError("Original completion is bound to another freeze")
        if manifest["controller_state_sha256"] != sha(state_path):
            raise ValueError("Original completion is bound to another controller endpoint")
        cloud = read(base / "wandb-cloud-verification.json")
        if cloud.get("passed") is not True:
            raise ValueError("Original cloud verification did not pass")
        for arm in ARMS:
            root = base / "runs" / arm
            completion_path = root / "train/completion.json"
            completed = read(completion_path)
            native = read(root / "native-training-endpoint.json")
            selection = read(root / "checkpoint-selection.json")
            if (
                native.get("training_complete") is not True
                or native["completion_sha256"] != sha(completion_path)
                or selection["selected_by"] != "validation_loss_only"
                or completed["best_checkpoint"] != selection["checkpoint"]
            ):
                raise ValueError(f"{arm} completed training or checkpoint selection changed")
            audit = read(root / "audit.json")
            if audit.get("passed") is not True or audit["freeze_sha256"] != freeze_sha:
                raise ValueError(f"{arm} audit did not pass the original freeze")
            if not audit["checkpoint_files"]:
                raise ValueError(f"{arm} audit has no checkpoint files")
            if audit["checkpoint_files"] != selection["files"]:
                raise ValueError(f"{arm} audit differs from validation-selected checkpoint")
            for path, expected in audit["checkpoint_files"].items():
                if sha(path) != expected:
                    raise ValueError(f"{arm} audited checkpoint changed")
            replay = executor.compare_replay(root / "evaluation", root / "replay", arm, 999)
            for key in ("rows", "predictions_sha256", "generated_tokens", "metrics"):
                if audit[key] != replay[key]:
                    raise ValueError(f"{arm} audit no longer matches original predictions")
            updates = completed.get("global_step", completed.get("optimizer_steps"))
            cloud_arm = cloud["runs"][arm]
            if (
                cloud_arm.get("finished") is not True
                or cloud_arm.get("exact_step_verified") is not True
                or cloud_arm["local_optimizer_steps"] != updates
                or cloud_arm["cloud_final_step"] not in (updates, updates - 1)
            ):
                raise ValueError(f"{arm} cloud endpoint differs from completed training")
            result["arms"][arm].update(
                completed_updates=updates, audited_rows=replay["rows"], metrics=replay["metrics"]
            )
        result["ready_for_calibration"] = True
        result["freeze_sha256"] = freeze_sha
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["invalid"].append(f"{type(error).__name__}: {error}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect_dependencies(args.base)
    if args.output:
        if args.output.resolve().is_relative_to(args.base.resolve()):
            parser.error("Output must be outside the original reproduction batch")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ready_for_calibration"] else 2


if __name__ == "__main__":
    sys.exit(main())
