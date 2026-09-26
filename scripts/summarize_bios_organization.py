"""Audit the A/B/C organization development study, including incomplete runs.

Only descriptive comparisons are reported. Supports are repeated edits of one
old model; they are averaged within that model before condition aggregation.
"""

import argparse
import csv
import itertools
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_data import (
    N_BASE,
    N_QUERIES,
    array_hash,
    load_world,
    paired_edit,
    rng_for,
    write_json,
)

CONDITIONS = ("A", "B", "C")
WORLDS = (0, 1)
SEEDS = (0, 1)
SCOPES = ("mlp", "all")
KINDS = ("coherent", "exception")
SUPPORTS = (0, 1)
TRAIN_STEPS = 14336
EDIT_STEPS = 512
TRAIN_CHECKPOINTS = (0, 896, 1792, 3584, 7168, 10752, 14336)
EDIT_CHECKPOINTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
RETENTION_GROUPS = ("0", "1", "2", "3", "2_base", "2_derived")
EDIT_FIELDS = ("E", "E_root", "E_member", "D", "U_accuracy", "E_target_value_nll")


def read_json(path, warnings):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: {type(error).__name__}: {error}")
        return None


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else None


def matches(left, right):
    """Compare JSON metrics, preserving the distinction between NA and failure."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() <= right.keys()
            and all(matches(value, right[key]) for key, value in left.items())
        )
    if left is None or isinstance(left, bool):
        return left is right
    if isinstance(left, (int, float)):
        return isinstance(right, (int, float)) and math.isclose(
            left, right, rel_tol=1e-6, abs_tol=1e-7
        )
    return left == right


def quality_pair(left, right):
    """Prerequisite, not evidence that organization caused an editing difference."""
    fields = ("base_accuracy", "derived_accuracy", "value_nll")
    if not all(left.get(f) is not None and right.get(f) is not None for f in fields):
        return None
    return bool(
        min(
            left["base_accuracy"],
            right["base_accuracy"],
            left["derived_accuracy"],
            right["derived_accuracy"],
        )
        >= 0.99
        and abs(left["base_accuracy"] - right["base_accuracy"]) <= 0.005 + 1e-12
        and abs(left["derived_accuracy"] - right["derived_accuracy"]) <= 0.005 + 1e-12
        and abs(left["value_nll"] - right["value_nll"]) <= 0.1 + 1e-12
    )


def study_manifest(root, warnings):
    expected_path = Path(__file__).parents[1] / "configs/bios-organization-development-v1.json"
    expected = json.loads(expected_path.read_text())
    saved = read_json(root / "study-config.json", warnings)
    errors = []
    if saved is not None and saved != expected:
        errors.append("study-config.json differs from the frozen organization development plan")
    if root.exists() and any(root.glob("world-*/config.json")) and saved is None:
        errors.append("study-config.json is missing or unreadable for an existing study")
    return expected, errors


def score_learning(world, arrays):
    correct = (arrays["prediction"] == world.answers) & arrays["ended"]
    nll = arrays["value_nll"]
    masks = {
        "employer": world.relation == 0,
        "company_default": world.relation == 1,
        "actual_nonexception": (world.relation == 2)
        & ~world.exceptions[np.maximum(world.person, 0)],
        "actual_old_exception": (world.relation == 2)
        & world.exceptions[np.maximum(world.person, 0)],
        "independent_attributes": np.isin(world.relation, [3, 4, 5, 6]),
    }
    return {
        "base_accuracy": float(correct[:N_BASE].mean()),
        "derived_accuracy": float(correct[N_BASE:].mean()),
        "value_nll": float(nll[:N_BASE].mean()),
        "nontermination_rate": float((~arrays["ended"]).mean()),
        "strata": {k: float(correct[v].mean()) for k, v in masks.items()},
        "learning_threshold_passed": bool(
            correct[:N_BASE].mean() >= 0.99 and correct[N_BASE:].mean() >= 0.99
        ),
    }, correct


def prediction_arrays(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = dict(source)
    for field in ("prediction", "ended", "correct", "value_nll"):
        if arrays[field].shape != (N_QUERIES,):
            raise ValueError(f"{field} must have shape ({N_QUERIES},)")
    if arrays["ended"].dtype != bool or arrays["correct"].dtype != bool:
        raise ValueError("ended and correct must be boolean arrays")
    if not np.isfinite(arrays["value_nll"]).all():
        raise ValueError("nonfinite saved answer NLL")
    return arrays


def inspect_learning(directory, world, seed, condition, warnings, truth=None, manifest=None):
    manifest = manifest or study_manifest(Path("/nonexistent-organization-study"), [])[0]
    config = read_json(directory / "config.json", warnings) or {}
    complete = read_json(directory / "complete.json", warnings)
    learning = read_json(directory / "learning.json", warnings) or []
    learning = sorted(learning, key=lambda p: p["step"])
    latest = learning[-1] if learning else {}
    reported_complete = (directory / "complete.json").exists()
    errors, checkpoint_audit = [], []
    planned = manifest["world_steps"]
    if directory.exists():
        expected = {
            "world_seed": world,
            "seed": seed,
            "condition": condition,
            "steps": planned,
            "run_kind": "full",
            "protocol": manifest["protocol"],
            "organization_seed": manifest["organization_seed"],
            "lr": manifest["learning_rate"],
            "optimizer": manifest["optimizer"],
            "weight_decay": manifest["weight_decay"],
            "gradient_clip_norm": manifest["gradient_clip"],
            "warmup_epochs": manifest["warmup_epochs"],
            "documents_per_step": manifest["document_batch_size"],
            "derived_queries_per_step": manifest["derived_batch_size"],
            "facts_per_document": 7,
            "cross_document_attention": False,
            "document_weight": 0.8,
            "derived_weight": 0.2,
            "checkpoints": list(TRAIN_CHECKPOINTS),
            "planned_supervised_tokens": planned * 280,
            "planned_input_tokens_including_padding": planned * 840,
            "planned_total_presentations": planned * 140,
        }
        for field, value in expected.items():
            if not matches(value, config.get(field)):
                errors.append(f"config missing/mismatched {field}")
        for field in ("width", "layers", "heads", "context"):
            if config.get("model", {}).get(field) != manifest["model"][field]:
                errors.append(f"config model missing/mismatched {field}")
        if truth is None:
            errors.append("truth unavailable; learning predictions cannot be verified")
        else:
            for field, array in (
                ("truth", truth.answers),
                ("prompts", truth.prompts),
                ("lengths", truth.lengths),
            ):
                if config.get(f"{field}_sha256") != array_hash(array):
                    errors.append(f"config {field} hash differs from canonical world")
            if config.get("model", {}).get("vocab_size") != truth.vocab_size:
                errors.append("config vocabulary size differs from canonical world")
        for field in ("model_initial_sha256", "derived_schedule_sha256", "core_source_sha256"):
            if not config.get(field):
                errors.append(f"config missing provenance field {field}")
        if truth is not None and config:
            try:
                from llm_memory_editability.bios_organization import make_documents

                with np.load(directory / "schedule.npz", allow_pickle=False) as source:
                    schedule = dict(source)
                documents = make_documents(truth, condition, manifest["organization_seed"])
                if not np.array_equal(schedule["documents"], documents):
                    errors.append("saved document schedule differs from canonical organization")
                rng = rng_for(world, 702)
                derived = np.concatenate(
                    [rng.permutation(2048) + N_BASE for _ in range(math.ceil(planned * 28 / 2048))]
                )
                derived = derived[: planned * 28].reshape(planned, 28)
                if not np.array_equal(schedule["derived"], derived):
                    errors.append("saved derived schedule differs from canonical common stream")
                for field in ("documents", "derived"):
                    key = "documents_sha256" if field == "documents" else "derived_schedule_sha256"
                    if config.get(key) != array_hash(schedule[field]):
                        errors.append(f"config {field} hash differs from saved schedule")
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
                errors.append(f"saved schedules unavailable: {error}")
    if len({p["step"] for p in learning}) != len(learning):
        errors.append("duplicate learning checkpoint steps")
    if reported_complete:
        if not complete or complete.get("status") != "complete":
            errors.append("invalid learning completion marker")
        elif complete.get("step") != planned or not matches(latest, complete.get("final", {})):
            errors.append("learning completion final differs from final recorded checkpoint")
        if complete:
            for field in ("exposure_sha256", "slot_exposure_sha256"):
                if complete.get(field) != latest.get(field):
                    errors.append(f"learning completion {field} differs from final checkpoint")
            if not complete.get("model_final_sha256"):
                errors.append("learning completion missing final model hash")
        if [p["step"] for p in learning] != list(TRAIN_CHECKPOINTS):
            errors.append("completed learning trajectory lacks the prescribed checkpoint set")
        saved_steps = sorted(
            int(p.stem.split("-")[-1]) for p in directory.glob("predictions-*.npz")
        )
        if saved_steps != list(TRAIN_CHECKPOINTS):
            errors.append("completed learning predictions lack the prescribed checkpoint set")
        for step in TRAIN_CHECKPOINTS:
            if not (directory / f"model-{step}.pt").exists():
                errors.append(f"step {step}: missing model checkpoint")
    if truth is not None:
        for point in learning:
            step = point["step"]
            point_errors = []
            try:
                arrays = prediction_arrays(directory / f"predictions-{step}.npz")
                metrics, correct = score_learning(truth, arrays)
                if not np.array_equal(correct, arrays["correct"]):
                    point_errors.append("saved correctness differs from full-answer scoring")
                for field, value in metrics.items():
                    if not matches(value, point.get(field)):
                        point_errors.append(f"metric mismatch: {field}")
                for name in ("exposure", "slot_exposure", "lr_weighted_exposure"):
                    expected_shape = (N_BASE, 7) if name == "slot_exposure" else (N_QUERIES,)
                    if arrays[name].shape != expected_shape:
                        point_errors.append(f"invalid {name} shape")
                    elif array_hash(arrays[name]) != point.get(f"{name}_sha256"):
                        point_errors.append(f"{name} hash differs from saved arrays")
                if step == planned:
                    expected_exposure = np.full(N_QUERIES, planned // 128, dtype=np.int64)
                    expected_exposure[:N_BASE][truth.relation[:N_BASE] == 1] *= 32
                    expected_exposure[N_BASE:] = planned * 28 // 2048
                    if not np.array_equal(arrays["exposure"], expected_exposure):
                        point_errors.append("endpoint per-fact exposure differs from plan")
                    expected_slots = np.repeat(expected_exposure[:N_BASE, None] // 7, 7, axis=1)
                    if not np.array_equal(arrays["slot_exposure"], expected_slots):
                        point_errors.append(
                            "endpoint per-fact/per-position exposure differs from plan"
                        )
                checkpoint_audit.append({"step": step, "passed": not point_errors})
            except (
                OSError,
                ValueError,
                KeyError,
                IndexError,
                EOFError,
                zipfile.BadZipFile,
            ) as error:
                point_errors.append(str(error))
                checkpoint_audit.append({"step": step, "passed": False})
            errors.extend(f"step {step}: {message}" for message in point_errors)
    completed = bool(reported_complete and not errors)
    reached = [p for p in learning if p.get("learning_threshold_passed") is True]
    first = reached[0] if reached and (directory / "learning.json").exists() else {}
    row = {
        "run": directory.name,
        "world": world,
        "seed": seed,
        "condition": condition,
        "status": "complete"
        if completed
        else (
            "audit_failed"
            if reported_complete
            else "failed"
            if (directory / "failure.json").exists()
            else "in_progress"
            if directory.exists()
            else "not_started"
        ),
        "complete": completed,
        "reported_complete": reported_complete,
        "audit_passed": not errors if reported_complete else None,
        "audit_errors": errors,
        "checkpoint_audit": checkpoint_audit,
        "planned_steps": planned,
        "step": latest.get("step"),
        **{
            field: latest.get(field)
            for field in (
                "base_accuracy",
                "derived_accuracy",
                "value_nll",
                "train_seconds",
                "train_matmul_flops_estimate",
            )
        },
        "old_exception_accuracy": latest.get("strata", {}).get("actual_old_exception"),
        "first_observed_learning_step": first.get("step"),
        "first_observed_learning_matmul_flops_estimate": first.get("train_matmul_flops_estimate"),
    }
    return row, learning


def cross_condition_audit(root, learning, trajectories, warnings):
    """Compare actual aligned checkpoints without requiring unfinished arms to exist."""
    configs = {
        r["run"]: read_json(root / r["run"] / "config.json", warnings) or {} for r in learning
    }
    lookup = {(r["world"], r["seed"], r["condition"]): r for r in learning}
    invariant = (
        "model_initial_sha256",
        "truth_sha256",
        "prompts_sha256",
        "lengths_sha256",
        "derived_schedule_sha256",
        "core_source_sha256",
        "model",
        "lr",
        "warmup_epochs",
        "planned_supervised_tokens",
        "planned_input_tokens_including_padding",
        "planned_total_presentations",
        "planned_train_matmul_flops_estimate",
    )
    rows = []
    for world, seed, (left, right) in itertools.product(
        WORLDS, SEEDS, itertools.combinations(CONDITIONS, 2)
    ):
        lr, rr = lookup[world, seed, left], lookup[world, seed, right]
        lc, rc = configs[lr["run"]], configs[rr["run"]]
        errors = []
        available = bool(lc and rc)
        if available:
            for field in invariant:
                if field not in lc or field not in rc or lc[field] != rc[field]:
                    errors.append(f"cross-condition missing/mismatched {field}")
            if lc.get("documents_sha256") == rc.get("documents_sha256"):
                errors.append("different organization conditions have identical document hashes")
        lp = {p["step"]: p for p in trajectories[world, seed, left]}
        rp = {p["step"]: p for p in trajectories[world, seed, right]}
        aligned = []
        for step in sorted(lp.keys() & rp.keys()):
            checks = {
                field: bool(lp[step].get(field) and lp[step].get(field) == rp[step].get(field))
                for field in (
                    "exposure_sha256",
                    "slot_exposure_sha256",
                    "lr_weighted_exposure_sha256",
                )
            }
            aligned.append({"step": step, **checks})
            errors.extend(
                f"step {step}: cross-condition mismatched {field}"
                for field, passed in checks.items()
                if not passed
            )
        for run in (lr, rr):
            if errors:
                run["audit_errors"].extend(f"{left}/{right}: {message}" for message in errors)
                if run["reported_complete"]:
                    run.update(complete=False, audit_passed=False, status="audit_failed")
        rows.append(
            {
                "world": world,
                "seed": seed,
                "left": left,
                "right": right,
                "configs_available": available,
                "passed": not errors if available else None,
                "errors": errors,
                "aligned_checkpoints": aligned,
            }
        )
    # All model/world runs must use one frozen core implementation, not merely each A/B/C triplet.
    sources = [c.get("core_source_sha256") for c in configs.values() if c]
    if sources and any(source != sources[0] for source in sources):
        warnings.append("Core source hashes differ between study runs")
        for row in learning:
            if configs[row["run"]]:
                row["audit_errors"].append("core source differs somewhere within the frozen batch")
                if row["reported_complete"]:
                    row.update(complete=False, audit_passed=False, status="audit_failed")
    return rows


def audit_case(directory, world, warnings, meta=None, old_endpoint=None):
    """Recompute all saved predictions, and check their recorded score and final marker."""
    from llm_memory_editability.bios_edit import score_edit

    trajectory = read_json(directory / "trajectory.json", warnings) or []
    complete = read_json(directory / "complete.json", warnings)
    config = read_json(directory / "config.json", warnings) or {}
    recorded = {p["step"]: p for p in trajectory}
    errors, recomputed = [], []
    claimed_complete = (directory / "complete.json").exists()
    if meta is None:
        parsed = re.fullmatch(r"support-(\d+)-(coherent|exception)-(mlp|all)", directory.name)
        meta = {"support": int(parsed[1]), "kind": parsed[2], "scope": parsed[3]} if parsed else {}
    expected = {
        **{k: meta.get(k) for k in ("scope", "kind", "support")},
        "steps": EDIT_STEPS,
        "lr": 3e-5,
        "retention_kl_weight": 1.0,
        "window_zero_based": [3, 4, 5],
        "branch": None,
        "branch_parameters": 0,
        "selection": "prespecified_unmatched",
    }
    if not meta or meta.get("support") not in SUPPORTS:
        errors.append("edit directory identity is not in the frozen study")
    for field, value in expected.items():
        if field not in config or not matches(value, config[field]):
            errors.append(f"edit config missing/mismatched {field}")
        if claimed_complete and (
            not complete or field not in complete or not matches(value, complete[field])
        ):
            errors.append(f"edit completion missing/mismatched {field}")
    if claimed_complete and (not complete or complete.get("status") != "complete"):
        errors.append("invalid edit completion marker")
    if len(recorded) != len(trajectory):
        errors.append("duplicate edit trajectory checkpoint steps")
    try:
        with np.load(directory / "sets.npz", allow_pickle=False) as source:
            sets = dict(source)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        return {
            "passed": False,
            "errors": [f"sets unavailable: {error}"],
            "checkpoints": [],
            "sets": None,
            "predictions": None,
        }
    prediction_files = {int(p.stem.split("-")[-1]): p for p in directory.glob("predictions-*.npz")}
    if claimed_complete:
        if sorted(recorded) != list(EDIT_CHECKPOINTS):
            errors.append("completed edit trajectory lacks the prescribed checkpoint set")
        if sorted(prediction_files) != list(EDIT_CHECKPOINTS):
            errors.append("completed edit predictions lack the prescribed checkpoint set")
    try:
        canonical = paired_edit(world, support=meta["support"], k=1)
        for field in ("E", "D", "heldout", "replay", "strata"):
            if not np.array_equal(sets[field], canonical[field]):
                errors.append(f"saved edit set differs from canonical paired edit: {field}")
        if not np.array_equal(sets["target"], canonical[meta["kind"]]):
            errors.append("saved edit target differs from canonical paired edit")
    except (ValueError, KeyError, IndexError, AttributeError) as error:
        errors.append(f"canonical edit validation unavailable: {error}")
    baseline = None
    try:
        old_endpoint = old_endpoint or directory.parent.parent / f"predictions-{TRAIN_STEPS}.npz"
        baseline = prediction_arrays(old_endpoint)
        baseline_correct = (baseline["prediction"] == world.answers) & baseline["ended"]
        if not np.array_equal(sets["old_correct"], baseline_correct):
            errors.append("old-correct denominator differs from the old-model endpoint")
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
        errors.append(f"old-model endpoint unavailable: {error}")
    for step in sorted(set(recorded) - prediction_files.keys()):
        errors.append(f"step {step}: recorded metrics have no saved predictions")
    final_predictions = None
    for step, path in sorted(prediction_files.items()):
        try:
            arrays = prediction_arrays(path)
            correct = (arrays["prediction"] == sets["target"]) & arrays["ended"]
            if not np.array_equal(correct, arrays["correct"]):
                errors.append(f"step {step}: saved correctness differs from full-answer scoring")
            arrays["correct"] = correct
            if step == 0 and not np.array_equal(
                sets["old_correct"],
                (arrays["prediction"] == world.answers) & arrays["ended"],
            ):
                errors.append("step 0: old-correct denominator differs from baseline predictions")
            if step == 0 and baseline is not None:
                for field in ("prediction", "ended"):
                    if not np.array_equal(arrays[field], baseline[field]):
                        errors.append(f"step 0: {field} differs from the old-model endpoint")
            metrics = score_edit(world, sets, sets["target"], arrays, sets["old_correct"])
            if step in recorded:
                for field, value in metrics.items():
                    if not matches(value, recorded[step].get(field)):
                        errors.append(f"step {step}: metric mismatch: {field}")
            elif complete:
                errors.append(f"step {step}: saved predictions have no trajectory entry")
            point = {**recorded.get(step, {}), "step": step, **metrics}
            recomputed.append(point)
            final_predictions = arrays
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile) as error:
            errors.append(f"step {step}: unreadable or inconsistent arrays: {error}")
    if not recomputed:
        errors.append("no readable saved prediction checkpoints")
    if complete and recomputed:
        final = recomputed[-1]
        if final["step"] != complete.get("steps", EDIT_STEPS):
            errors.append("completion marker does not match saved final prediction step")
        if not matches(
            {k: v for k, v in final.items() if k != "down_calibration"}, complete.get("final", {})
        ):
            errors.append("completion final metrics differ from saved predictions/trajectory")
        first = next((p["step"] for p in recomputed if p["joint_pass"] is True), None)
        if first != complete.get("first_observed_joint_step"):
            errors.append("first observed joint pass differs from recomputed trajectory")
        if complete.get("joint_evaluable") is not (final["joint_pass"] is not None):
            errors.append("joint evaluability differs from retention denominators")
    return {
        "passed": not errors,
        "errors": errors,
        "checkpoints": recomputed,
        "sets": sets,
        "predictions": final_predictions,
    }


def inspect_edit(directory, meta, world, warnings):
    complete = read_json(directory / "complete.json", warnings)
    case = {
        **meta,
        "directory": str(directory),
        "reported_complete": (directory / "complete.json").exists(),
        "complete": bool(
            complete
            and complete.get("status") == "complete"
            and complete.get("final", {}).get("step") == EDIT_STEPS
        ),
    }
    case["status"] = (
        "complete"
        if case["complete"]
        else (
            "failed"
            if (directory / "failure.json").exists()
            else "in_progress"
            if directory.exists()
            else "not_started"
        )
    )
    audit = None
    if directory.exists() and world is not None and (directory / "sets.npz").exists():
        audit = audit_case(directory, world, warnings, meta=meta)
        run_config = read_json(directory.parent / "run.json", warnings) or {}
        old_config = read_json(directory.parent.parent / "config.json", warnings) or {}
        expected_run = {
            "world_step": TRAIN_STEPS,
            "steps": EDIT_STEPS,
            "window": 3,
            "lr": 3e-5,
            "retention": 1.0,
            "supports": [0, 1],
            "scopes": ["mlp", "all"],
            "branches": False,
        }
        for field, value in expected_run.items():
            if field not in run_config or not matches(value, run_config[field]):
                audit["errors"].append(f"editing run metadata missing/mismatched {field}")
        expected_checkpoint = directory.parent.parent / f"model-{TRAIN_STEPS}.pt"
        if Path(run_config.get("checkpoint", "")).resolve() != expected_checkpoint.resolve():
            audit["errors"].append("editing run checkpoint path differs from expected old endpoint")
        for name in ("bios_edit.py", "bios_model.py", "bios_data.py", "bios_train.py"):
            source_key = f"src/llm_memory_editability/{name}"
            old_hash = old_config.get("source_sha256", {}).get(source_key)
            edit_hash = run_config.get("source_sha256", {}).get(source_key)
            if not old_hash or old_hash != edit_hash:
                audit["errors"].append(f"editing source differs from old-model provenance: {name}")
        audit["passed"] = not audit["errors"]
    case.update(
        audit_passed=(
            bool(audit and audit["passed"] and case["complete"])
            if case["reported_complete"]
            else None
        ),
        audit_errors=audit["errors"] if audit else [],
        audited_checkpoints=len(audit["checkpoints"]) if audit else 0,
    )
    if case["reported_complete"] and case["audit_passed"] is not True:
        case["status"] = "audit_failed"
        if audit is None:
            case["audit_errors"].append("completion cannot be audited without truth and saved sets")
    points = audit["checkpoints"] if audit else []
    final = points[-1] if points else {}
    case.update(step=final.get("step"), **{field: final.get(field) for field in EDIT_FIELDS})
    case["joint_final_passed"] = final.get("joint_pass")
    case["first_observed_joint_step"] = next(
        (p["step"] for p in points if p["joint_pass"] is True), None
    )
    case["joint_ever_passed"] = (
        True
        if case["first_observed_joint_step"] is not None
        else False
        if any(p["joint_pass"] is not None for p in points)
        else None
    )
    case["final"] = final
    if audit and audit["predictions"] is not None:
        from llm_memory_editability.bios_analysis import propagation_errors

        case["propagation"] = propagation_errors(world, audit["sets"], audit["predictions"])
    return case, audit


def aggregate_edits(cases):
    """Only two-support model means enter condition means; expose all denominators."""
    aggregates, models = [], []
    for condition, scope, kind in itertools.product(CONDITIONS, SCOPES, KINDS):
        selected = [
            c for c in cases if (c["condition"], c["scope"], c["kind"]) == (condition, scope, kind)
        ]
        ready = [c for c in selected if c["complete"] and c["audit_passed"] is True]
        grouped = defaultdict(list)
        for case in ready:
            grouped[(case["world"], case["seed"])].append(case)
        model_means = []
        for (world, seed), group in sorted(grouped.items()):
            if {c["support"] for c in group} != set(SUPPORTS):
                continue
            row = {
                "condition": condition,
                "scope": scope,
                "kind": kind,
                "world": world,
                "seed": seed,
                "n_supports": len(group),
                **{f"mean_{f}": mean(c[f] for c in group) for f in EDIT_FIELDS},
            }
            for g in RETENTION_GROUPS:
                rates = [c["final"]["U_heldout_destruction"][g]["rate"] for c in group]
                row[f"mean_U{g}_damage"] = (
                    mean(rates) if all(r is not None for r in rates) else None
                )
            models.append(row)
            model_means.append(row)
        row = {
            "condition": condition,
            "scope": scope,
            "kind": kind,
            "expected_models": len(WORLDS) * len(SEEDS),
            "expected_cases": len(WORLDS) * len(SEEDS) * len(SUPPORTS),
            "completed_cases": sum(c["complete"] for c in selected),
            "audited_complete_cases": len(ready),
            "complete_two_support_models": len(model_means),
            "joint_final_evaluable": sum(c["joint_final_passed"] is not None for c in ready),
            "joint_final_passed": sum(c["joint_final_passed"] is True for c in ready),
            "joint_ever_evaluable": sum(c["joint_ever_passed"] is not None for c in ready),
            "joint_ever_passed": sum(c["joint_ever_passed"] is True for c in ready),
            **{f"mean_{f}": mean(m[f"mean_{f}"] for m in model_means) for f in EDIT_FIELDS},
        }
        for g in RETENTION_GROUPS:
            values = [m[f"mean_U{g}_damage"] for m in model_means]
            row[f"models_U{g}_evaluable"] = sum(v is not None for v in values)
            row[f"mean_U{g}_damage"] = mean(values)
        row["propagation_counts"] = dict(
            sum((Counter(c.get("propagation", {}).get("counts", {})) for c in ready), Counter())
        )
        row["propagation_strata"] = {
            name: dict(
                sum(
                    (
                        Counter(c.get("propagation", {}).get("strata", {}).get(name, {}))
                        for c in ready
                    ),
                    Counter(),
                )
            )
            for name in (
                "old_exception",
                "updated_personal_equals_default",
                "updated_personal_differs_from_default",
            )
        }
        aggregates.append(row)
    return aggregates, models


def paired_results(learning, cases, audits):
    lookup = {(r["world"], r["seed"], r["condition"]): r for r in learning}
    edit_lookup = {
        (c["world"], c["seed"], c["condition"], c["scope"], c["kind"], c["support"]): c
        for c in cases
        if c["complete"] and c["audit_passed"] is True
    }
    learning_pairs, edit_pairs = [], []
    for world, seed, (left, right) in itertools.product(
        WORLDS, SEEDS, itertools.combinations(CONDITIONS, 2)
    ):
        lrun, rrun = lookup[world, seed, left], lookup[world, seed, right]
        available = lrun["complete"] and rrun["complete"] and lrun["step"] == rrun["step"]
        quality = quality_pair(lrun, rrun) if available else None
        meta = {
            "world": world,
            "seed": seed,
            "left": left,
            "right": right,
            "comparable_complete_old_models": available,
            "old_knowledge_quality_eligible": quality,
        }
        learning_pairs.append(
            {
                **meta,
                **{
                    f"left_minus_right_{field}": lrun[field] - rrun[field] if available else None
                    for field in ("base_accuracy", "derived_accuracy", "value_nll")
                },
            }
        )
        for scope, kind, support in itertools.product(SCOPES, KINDS, SUPPORTS):
            lc = edit_lookup.get((world, seed, left, scope, kind, support))
            rc = edit_lookup.get((world, seed, right, scope, kind, support))
            if lc is None or rc is None:
                continue
            row = {
                **meta,
                "scope": scope,
                "kind": kind,
                "support": support,
                **{f"left_minus_right_{f}": lc[f] - rc[f] for f in EDIT_FIELDS},
            }
            la, ra = audits[lc["directory"]], audits[rc["directory"]]
            same = all(
                np.array_equal(la["sets"][f], ra["sets"][f])
                for f in ("E", "D", "target", "heldout", "replay", "strata")
            )
            row["identical_edit_sets"] = same
            row["eligible_edit_comparison"] = bool(quality and same)
            if same:
                from llm_memory_editability.bios_analysis import retention_counts

                known = la["sets"]["old_correct"] & ra["sets"]["old_correct"]
                lr = retention_counts(la["sets"], la["predictions"]["correct"], known)
                rr = retention_counts(ra["sets"], ra["predictions"]["correct"], known)
                row["common_known_retention"] = {
                    g: {
                        "known": lr[g]["known"],
                        "left_damage": lr[g]["rate"],
                        "right_damage": rr[g]["rate"],
                    }
                    for g in RETENTION_GROUPS
                }
            edit_pairs.append(row)
    return learning_pairs, edit_pairs


def csv_dump(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            for row in rows
        )


def aggregate_paired_edits(pairs):
    grouped = defaultdict(list)
    for pair in pairs:
        if pair["identical_edit_sets"]:
            grouped[
                tuple(pair[k] for k in ("world", "seed", "left", "right", "scope", "kind"))
            ].append(pair)
    models = []
    for key, group in grouped.items():
        if {p["support"] for p in group} != set(SUPPORTS):
            continue
        row = dict(zip(("world", "seed", "left", "right", "scope", "kind"), key, strict=True))
        row.update(
            n_supports=len(group),
            old_knowledge_quality_eligible=all(
                p["old_knowledge_quality_eligible"] is True for p in group
            ),
        )
        row.update(
            {
                f"left_minus_right_{field}": mean(p[f"left_minus_right_{field}"] for p in group)
                for field in EDIT_FIELDS
            }
        )
        models.append(row)
    aggregates = []
    for (left, right), scope, kind, subset in itertools.product(
        itertools.combinations(CONDITIONS, 2), SCOPES, KINDS, ("all", "quality_matched")
    ):
        selected = [
            p
            for p in models
            if (p["left"], p["right"], p["scope"], p["kind"]) == (left, right, scope, kind)
            and (subset == "all" or p["old_knowledge_quality_eligible"])
        ]
        aggregates.append(
            {
                "left": left,
                "right": right,
                "scope": scope,
                "kind": kind,
                "subset": subset,
                "n_model_pairs": len(selected),
                "n_worlds": len({p["world"] for p in selected}),
                "expected_model_pairs": 4,
                **{
                    f"mean_left_minus_right_{field}": mean(
                        p[f"left_minus_right_{field}"] for p in selected
                    )
                    for field in EDIT_FIELDS
                },
            }
        )
    return models, aggregates


def organization_measurements(root, learning, warnings):
    """Average intervention recipients within each model before summarizing models."""
    rows, measurements = [], []
    for run in learning:
        path = root / run["run"] / "organization-final" / "measurements.json"
        value = read_json(path, warnings)
        if not value:
            continue
        if not run["complete"] or value.get("step") != run["planned_steps"]:
            warnings.append(f"{path}: excluded from endpoint geometry: old model is incomplete")
            continue
        meta = {k: run[k] for k in ("run", "world", "seed", "condition")}
        meta.update(step=value["step"], precision=value.get("precision", "unspecified"))
        measurements.append({**meta, "n_probes": len(value.get("probes", []))})
        rows.extend(
            {**meta, "type": "geometry", "intervention": None, **point}
            for point in value.get("geometry", [])
        )
        groups = defaultdict(list)
        for point in value.get("probes", []):
            groups[point["layer"], point["intervention"]].append(point)
        for (layer, intervention), points in groups.items():
            row = {
                **meta,
                "type": "probe",
                "layer": layer,
                "intervention": intervention,
                "recipient_probes": len(points),
            }
            for field in (
                "same_company_nonexception",
                "old_exception",
                "independent_attribute",
                "unrelated_company",
                "selective_shared_score",
            ):
                row[field] = mean(p.get(field) for p in points)
            for group in (
                "same_company_nonexception",
                "old_exception",
                "independent_attribute",
                "unrelated_company",
            ):
                for field in (
                    "donor_generated_delta",
                    "baseline_old_accuracy",
                    "changed_old_accuracy",
                ):
                    row[f"generation_{group}_{field}"] = mean(
                        p.get("generation", {}).get(group, {}).get(field) for p in points
                    )
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("condition", "type", "layer", "intervention", "precision"))
        groups[key].append(row)
    aggregates = []
    metadata = {
        "run",
        "world",
        "seed",
        "step",
        "condition",
        "type",
        "layer",
        "intervention",
        "precision",
        "recipient_probes",
    }
    for key, group in groups.items():
        row = dict(
            zip(("condition", "type", "layer", "intervention", "precision"), key, strict=True)
        )
        row["n_models"] = len(group)
        fields = set().union(*(p.keys() for p in group)) - metadata
        row.update({f"mean_{field}": mean(p.get(field) for p in group) for field in sorted(fields)})
        aggregates.append(row)
    return measurements, rows, aggregates


def percent(value):
    return "NA" if value is None else f"{100 * value:.2f}%"


def markdown_report(report):
    counts = report["counts"]
    lines = [
        "# 数据组织 A/B/C 开发实验",
        "",
        f"旧模型完成 {counts['completed_old_models']}/{counts['expected_old_models']}；"
        f"编辑完成 {counts['completed_edits']}/{counts['expected_edits']}；"
        f"通过预测复核的已完成编辑 {counts['audited_complete_edits']}。",
        "",
        "A 为关系链组织，B 为人物组织，C 为打散组织。以下均为描述性结果；"
        "每个条件最多 4 个世界×初始化组合。同一模型上的 2 个支持集是重复编辑案例，"
        "不构成独立旧模型，也不把 96 次编辑当成 96 个独立样本。",
        "",
        "## 学习",
        "",
        "终点平均值仅使用完成预定预算的模型；部分轨迹的最新检查点见 learning-summary.csv。"
        "首次达标是首次观测到基础与组合查询均≥99%的检查点，并非精确达标时间。",
        "",
        "| 条件 | 完成模型 | 基础准确率 | 组合准确率 | 旧例外准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["learning_aggregates"]:
        lines.append(
            f"| {row['condition']} | {row['complete_models']}/4 | "
            f"{percent(row['mean_base_accuracy'])} | "
            f"{percent(row['mean_derived_accuracy'])} | "
            f"{percent(row['mean_old_exception_accuracy'])} |"
        )
    eligible = sum(p["old_knowledge_quality_eligible"] is True for p in report["learning_pairs"])
    comparable = sum(p["comparable_complete_old_models"] for p in report["learning_pairs"])
    lines.extend(
        [
            "",
            f"可比较的已完成条件配对 {comparable}/12，其中旧知识质量可比 {eligible}。"
            "门槛为两侧基础与组合准确率均≥99%，各准确率差≤0.5个百分点，"
            "基础答案 NLL 差≤0.1。总体比较保留组织对学习程度和编辑的共同影响；"
            "未通过旧知识匹配时，不能将编辑差异单独解释为相同掌握程度下的表征可编辑性差异。"
            "质量匹配子集属于处理后条件化，不能据此证明因果中介。"
            "edit-paired-aggregates.csv 同时报告全部配对与质量匹配配对，均先在模型内平均支持集。",
            "",
            "## 编辑",
            "",
            "E 是直接修改，D 是必要传播；下表 E/D 与最终联合通过均取 512 步编辑终点。"
            "均值先在同一旧模型内平均两个支持集，"
            "再跨完整模型平均；尚缺支持集的模型不进入该均值。联合通过要求根事实全对、"
            "成员事实与 D 各≥95%，且预先划定、未供编辑器使用的保持测试池 U_heldout 中，"
            "0/1/2/3 及 2_base/2_derived 各分层的旧正确事实破坏率≤1%。NA 不算失败。"
            "完整保持集 U_full 独立报告，不参与当前联合门槛；联合通过不保证 U_full 每层损伤≤1%。",
            "",
            "| 条件 | 方法 | 更新 | 完成案例 | 两支持集完整模型 | E | D | "
            "最终联合通过/可评价 | 曾联合通过/可评价 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["edit_aggregates"]:
        lines.append(
            f"| {row['condition']} | {row['scope']} | {row['kind']} | "
            f"{row['completed_cases']}/{row['expected_cases']} | "
            f"{row['complete_two_support_models']}/4 | {percent(row['mean_E'])} | "
            f"{percent(row['mean_D'])} | {row['joint_final_passed']}/"
            f"{row['joint_final_evaluable']} | {row['joint_ever_passed']}/"
            f"{row['joint_ever_evaluable']} |"
        )
    lines.extend(
        [
            "",
            "保持集各分层的 known/broken/rate 保留在 edit-cases.json；"
            "条件配对还报告两侧编辑前共同答对的保持集，避免比较不同损伤分母。"
            "不报告显著性检验，不基于完成较快或表现较好的案例挑选结论。",
            "",
            "## 传播错误诊断",
            "",
            "错误互斥地依次归为：未结束、旧公司默认、个人实际城市、其他。"
            "同时分开统计旧例外、新实际城市等于默认、不同于默认三组。"
            "这些错误类型描述输出行为，不能单独证明内部计算机制。",
            "",
        ]
    )
    available = [
        r for r in report["edit_aggregates"] if r["kind"] == "exception" and r["propagation_counts"]
    ]
    if not available:
        lines.append("尚无通过复核的已完成例外编辑，暂不能给出传播结论。")
    for row in available:
        counts = row["propagation_counts"]
        wrong = sum(counts.values()) - counts.get("correct", 0)
        personal = counts.get("personal_city_instead_of_company_default", 0)
        strata = row["propagation_strata"]
        formatted = []
        for name, label in (
            ("updated_personal_equals_default", "实际=默认"),
            ("updated_personal_differs_from_default", "实际≠默认"),
        ):
            n = strata[name].get("n", 0)
            value = strata[name].get("correct", 0) / n if n else None
            formatted.append(f"{label} D={percent(value)}（n={n}）")
        lines.append(
            f"- {row['condition']}/{row['scope']}：{'；'.join(formatted)}；"
            f"误用个人实际城市 {personal}/{wrong} 个 D 错误。"
        )
    lines.extend(
        [
            "",
            "## 复核与限制",
            "",
            f"复核了 {report['counts']['audited_checkpoints']} 个保存的编辑检查点，"
            f"有 {report['counts']['edit_audit_failures']} 个案例存在复核错误。"
            "逐检查点重新计算自由生成答案的 E/D/U、损伤分母与联合达标；"
            "复核失败或缺少真值文件的完成案例不进入编辑均值。",
            "旧模型同样复核完整预算、配置身份、全部预定学习检查点、保存预测和逐事实曝光；"
            "cross-condition-audit.csv 检查相同初始化、真值、组合清单、源码与对齐检查点的"
            "曝光/位置/学习率加权曝光一致。运行中的未完成产物单独显示，不能视为实验失败。",
            "每条个人事实有 16 次位于文档首槽，另 6/7 次呈现带有前文。七轮循环平衡绝对位置，"
            "并未平衡所有相对顺序：默认城市在实际城市前的比例为 6/7，三组共同使用此安排。"
            "同一 GPU 存在并行作业，墙钟耗时含资源竞争；学习效率优先比较步数和 FLOPs。",
        ]
    )
    if report["warnings"]:
        lines.append("")
        lines.extend(f"- {message}" for message in report["warnings"])
    lines.extend(
        [
            "",
            "## 表征诊断",
            "",
            f"完成终点表征测量 {len(report.get('organization_measurements', []))}/12。"
            "organization-models.csv 保留每个模型的层级几何与干预指标，"
            "organization-aggregates.csv 在条件、层、干预、精度均相同的情况下汇总。"
            "这些是探索性诊断；公司均值干预影响独立属性时，不能据此声称形成纯公司表征。",
        ]
    )
    return "\n".join(lines) + "\n"


def plot_report(output, trajectories, report):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for axis, field in zip(axes, ("base_accuracy", "derived_accuracy"), strict=True):
        for condition in CONDITIONS:
            by_step = defaultdict(list)
            for (_world, _seed, arm), points in trajectories.items():
                if arm == condition:
                    for point in points:
                        by_step[point["step"]].append(point[field])
            steps = sorted(by_step)
            if steps:
                axis.plot(
                    steps,
                    [100 * mean(by_step[s]) for s in steps],
                    "o-",
                    color=colors[condition],
                    markersize=3,
                    label=condition,
                )
                axis.fill_between(
                    steps,
                    [100 * min(by_step[s]) for s in steps],
                    [100 * max(by_step[s]) for s in steps],
                    color=colors[condition],
                    alpha=0.12,
                )
        axis.set(title=field, xlabel="Training steps", ylabel="Accuracy (%)", ylim=(-2, 102))
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()
    fig.suptitle("Available checkpoints; mean and observed range (counts may vary by step)")
    fig.savefig(output / "learning-curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for axis, scope in zip(axes, SCOPES, strict=True):
        for condition in CONDITIONS:
            for kind, marker in (("coherent", "o"), ("exception", "x")):
                row = next(
                    r
                    for r in report["edit_aggregates"]
                    if (r["condition"], r["scope"], r["kind"]) == (condition, scope, kind)
                )
                if row["mean_E"] is not None:
                    axis.scatter(
                        100 * row["mean_E"],
                        100 * row["mean_D"],
                        color=colors[condition],
                        marker=marker,
                        label=f"{condition}/{kind} (models={row['complete_two_support_models']})",
                    )
        axis.set(
            title=scope, xlabel="Final E (%)", ylabel="Final D (%)", xlim=(-2, 102), ylim=(-2, 102)
        )
        axis.axhline(95, color="gray", linestyle=":", linewidth=0.7)
        axis.grid(alpha=0.2)
        if axis.collections:
            axis.legend(fontsize=7)
    fig.suptitle("Completed two-support model means; E and D do not imply joint success")
    fig.savefig(output / "editing-final.png", dpi=150)
    plt.close(fig)
    return True


def run(args):
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "report"
    output.mkdir(parents=True, exist_ok=True)
    warnings, learning, cases, trajectories, audits = [], [], [], {}, {}
    manifest, manifest_errors = study_manifest(root, warnings)
    warnings.extend(manifest_errors)
    worlds = {}
    for world in WORLDS:
        try:
            worlds[world] = load_world(Path(args.world_root) / f"world-{world}")
        except (OSError, ValueError, KeyError) as error:
            worlds[world] = None
            warnings.append(f"world-{world} truth unavailable: {error}")
    for world, seed, condition in itertools.product(WORLDS, SEEDS, CONDITIONS):
        directory = root / f"world-{world}-seed-{seed}-{condition}"
        row, points = inspect_learning(
            directory, world, seed, condition, warnings, truth=worlds[world], manifest=manifest
        )
        if manifest_errors:
            row["audit_errors"].extend(manifest_errors)
            if row["reported_complete"]:
                row.update(complete=False, audit_passed=False, status="audit_failed")
        learning.append(row)
        trajectories[world, seed, condition] = points
        for scope, kind, support in itertools.product(SCOPES, KINDS, SUPPORTS):
            case_directory = directory / "edits" / f"support-{support}-{kind}-{scope}"
            meta = {
                "run": directory.name,
                "world": world,
                "seed": seed,
                "condition": condition,
                "scope": scope,
                "kind": kind,
                "support": support,
            }
            case, audit = inspect_edit(case_directory, meta, worlds[world], warnings)
            cases.append(case)
            if audit:
                audits[str(case_directory)] = audit
    matching_audit = cross_condition_audit(root, learning, trajectories, warnings)
    learning_lookup = {r["run"]: r for r in learning}
    for case in cases:
        case["old_model_audit_passed"] = learning_lookup[case["run"]]["complete"]
        if case["complete"] and not case["old_model_audit_passed"]:
            case["audit_passed"] = False
            case["audit_errors"].append("old model has not passed full-study completion audit")
            if case["directory"] in audits:
                audits[case["directory"]]["passed"] = False
    aggregates, model_means = aggregate_edits(cases)
    learning_pairs, edit_pairs = paired_results(learning, cases, audits)
    paired_models, paired_aggregates = aggregate_paired_edits(edit_pairs)
    measurements, measurement_rows, measurement_aggregates = organization_measurements(
        root, learning, warnings
    )
    learning_aggregates = []
    for condition in CONDITIONS:
        group = [r for r in learning if r["condition"] == condition and r["complete"]]
        learning_aggregates.append(
            {
                "condition": condition,
                "expected_models": 4,
                "complete_models": len(group),
                **{
                    f"mean_{field}": mean(r[field] for r in group)
                    for field in (
                        "base_accuracy",
                        "derived_accuracy",
                        "old_exception_accuracy",
                        "value_nll",
                        "train_seconds",
                        "train_matmul_flops_estimate",
                    )
                },
            }
        )
    report = {
        "study": "bios-organization-dev-v1",
        "root": str(root.resolve()),
        "counts": {
            "expected_old_models": 12,
            "completed_old_models": sum(r["complete"] for r in learning),
            "reported_complete_old_models": sum(r["reported_complete"] for r in learning),
            "learning_audit_failures": sum(r["audit_passed"] is False for r in learning),
            "expected_edits": 96,
            "completed_edits": sum(c["complete"] for c in cases),
            "audited_complete_edits": sum(
                c["complete"] and c["audit_passed"] is True for c in cases
            ),
            "audited_checkpoints": sum(c["audited_checkpoints"] for c in cases),
            "edit_audit_failures": sum(
                c["reported_complete"] and c["audit_passed"] is False for c in cases
            ),
        },
        "learning": learning,
        "learning_aggregates": learning_aggregates,
        "learning_pairs": learning_pairs,
        "cross_condition_audit": matching_audit,
        "edit_aggregates": aggregates,
        "edit_model_means": model_means,
        "edit_pairs": edit_pairs,
        "edit_paired_model_means": paired_models,
        "edit_paired_aggregates": paired_aggregates,
        "organization_measurements": measurements,
        "organization_aggregates": measurement_aggregates,
        "warnings": warnings,
        "inference": "Descriptive only; supports are repeated edits within an old model.",
    }
    write_json(output / "summary.json", report)
    write_json(output / "edit-cases.json", cases)
    write_json(
        output / "audit.json",
        [
            {
                "directory": path,
                "passed": audit["passed"],
                "errors": audit["errors"],
                "checkpoint_steps": [p["step"] for p in audit["checkpoints"]],
            }
            for path, audit in audits.items()
        ],
    )
    for name, rows in (
        ("learning-summary", learning),
        ("learning-aggregates", learning_aggregates),
        ("learning-pairs", learning_pairs),
        ("cross-condition-audit", matching_audit),
        ("edit-summary", cases),
        ("edit-aggregates", aggregates),
        ("edit-model-means", model_means),
        ("edit-pairs", edit_pairs),
        ("edit-paired-model-means", paired_models),
        ("edit-paired-aggregates", paired_aggregates),
        ("organization-models", measurement_rows),
        ("organization-aggregates", measurement_aggregates),
    ):
        csv_dump(output / f"{name}.csv", rows)
    (output / "report.md").write_text(markdown_report(report))
    if not args.no_plots:
        plot_report(output, trajectories, report)
    print(json.dumps({"output": str(output.resolve()), **report["counts"]}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-organization-dev-v1")
    parser.add_argument("--world-root", default="data/bios-organization-v1")
    parser.add_argument("--output")
    parser.add_argument("--no-plots", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
