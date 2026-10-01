"""Audit P1 from saved predictions and report paired, unreduced outcomes.

The common response probes keep answer labels fixed across counterfactual
arms. Own-world accuracy and retention are reported separately because their
correct answers and unchanged pools differ between arms.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.bios_cross import CHAINS, edit_pair, make_cross_world
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_mechanism_edit import (
    OPTIONAL_ARM,
    REFERENCE_ARMS,
    _contract_hash,
    arm_metrics,
    file_hash,
    make_arm,
    sampling_streams,
    source_hashes,
)
from llm_memory_editability.bios_organization_train import state_hash

META = ("width", "world", "seed", "condition", "chain", "arm", "source", "step")
CONTRASTS = (
    ("balanced-minus-uniform", "class-balanced-exception", "exception"),
    ("root-minus-rehearsal", "root-only", "old-fact-rehearsal"),
    ("actual-minus-rehearsal", "actual-only", "old-fact-rehearsal"),
    ("conflict-minus-root", "exception", "root-only"),
    ("conflict-minus-actual", "exception", "actual-only"),
    ("coherent-minus-conflict", "coherent", "exception"),
    ("frozen-embedding-minus-mlp", OPTIONAL_ARM, "exception"),
)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def response_probes(world, pair, arrays):
    """Fixed counterfactual labels; matching a label alone does not identify a route."""
    prediction, ended = arrays["prediction"], arrays["ended"]
    result = {}
    chain = int(np.isin(pair["D"], world.derived_ids[1]).all())
    for label, ids in (
        ("probe", pair["D"]),
        ("probe_heldout", np.intersect1d(pair["D"], world.heldout_ids[chain])),
        ("probe_conflict", pair["conflict_D"]),
        ("probe_conflict_heldout", np.intersect1d(pair["conflict_D"], world.heldout_ids[chain])),
    ):
        actual_ids = world.actual_ids[chain, world.person[ids]]
        candidates = {
            "new_default": pair["exception"][ids],
            "new_actual": pair["exception"][actual_ids],
            "old_default": world.answers[ids],
        }
        hits = []
        for target, values in candidates.items():
            hit = (prediction[ids] == values) & ended[ids]
            hits.append(hit)
            result[f"{label}_{target}"] = {
                "n": len(ids),
                "correct": int(hit.sum()),
                "accuracy": float(hit.mean()) if len(ids) else None,
            }
        matches = np.stack(hits).sum(0) if len(ids) else np.array([], dtype=int)
        result[f"{label}_labels"] = {
            "n": len(ids),
            "ambiguous_hits": int((matches > 1).sum()),
            "other_answer": int(((matches == 0) & ended[ids]).sum()),
            "termination_errors": int((~ended[ids]).sum()),
        }
    return result


def _assert_values(actual, expected, context):
    """Check a recomputed metric subset without admitting stale recorded scores."""
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"Missing recorded metric {context}/{key}")
        observed = actual[key]
        if isinstance(value, dict):
            _assert_values(observed, value, f"{context}/{key}")
        elif isinstance(value, float):
            if observed is None or not math.isclose(observed, value, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError(f"Recorded metric differs: {context}/{key}")
        elif observed != value:
            raise ValueError(f"Recorded metric differs: {context}/{key}")


def _read_predictions(path, target):
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in ("prediction", "ended", "correct", "value_nll")}
    if any(value.shape != target.shape for value in arrays.values()):
        raise ValueError(f"Prediction dimensions changed: {path}")
    if arrays["ended"].dtype != bool or arrays["correct"].dtype != bool:
        raise ValueError(f"Correctness/termination masks are not boolean: {path}")
    expected = (arrays["prediction"] == target) & arrays["ended"]
    if not np.array_equal(arrays["correct"], expected):
        raise ValueError(f"Stored correctness differs from free generation: {path}")
    if not np.isfinite(arrays["value_nll"]).all():
        raise ValueError(f"Nonfinite prediction NLL: {path}")
    return arrays


def _validate_sets(path, spec, before, choices=None):
    with np.load(path) as saved:
        for key, expected in (*spec.items(), ("old_correct", before)):
            if key not in saved or not np.array_equal(saved[key], expected):
                raise ValueError(f"Saved set differs: {path}/{key}")
        if choices is not None:
            for key, expected in zip(("edit_sampling", "replay_sampling"), choices, strict=True):
                if not np.array_equal(saved[key], expected):
                    raise ValueError(f"Saved stream differs: {path}/{key}")


def _validate_updates(directory, spec, choices, study, complete):
    stats = json.loads((directory / "update-stats.json").read_text())
    if len(stats) != study["edit_steps"]:
        raise ValueError("A completed run is missing update statistics")
    expected_root = spec["S_root_mask"][choices[0]].sum(1)
    for step, record in enumerate(stats):
        if record["step"] != step + 1 or record["root_samples"] != expected_root[step]:
            raise ValueError("Recorded update order or class sample count changed")
        if record["actual_samples"] != study["batch_size"] - expected_root[step]:
            raise ValueError("Actual sample count changed")
        for key in ("loss", "ce", "kl", "total_gradient_norm"):
            if not math.isfinite(record[key]):
                raise ValueError("Nonfinite update statistics")
        if record["gradient_clipped"] != (record["total_gradient_norm"] > 1):
            raise ValueError("Gradient clipping indicator differs")
        if step in study["gradient_steps"]:
            for kind in ("root", "actual"):
                norm = record[f"{kind}_ce_gradient_norm"]
                if norm < 0 or not math.isfinite(norm):
                    raise ValueError("Missing or invalid class gradient diagnostic")
    if complete["root_samples"] != int(expected_root.sum()):
        raise ValueError("Completion sample count differs")
    if complete["actual_samples"] != len(stats) * study["batch_size"] - int(expected_root.sum()):
        raise ValueError("Completion actual sample count differs")
    if complete["clipped_updates"] != sum(point["gradient_clipped"] for point in stats):
        raise ValueError("Completion clipping count differs")


def _validate_weights(directory, complete, contract):
    final = torch.load(directory / "model-final.pt", map_location="cpu", weights_only=False)
    if final["step"] != contract["study"]["edit_steps"]:
        raise ValueError("Final weight step differs")
    if state_hash(final["model"]) != complete["final_model_sha256"]:
        raise ValueError("Final model hash differs")
    del final
    restart = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    if restart["contract_sha256"] != _contract_hash(contract):
        raise ValueError("Restart contract differs")
    if restart["step"] != contract["study"]["edit_steps"] or not restart["optimizer"]["state"]:
        raise ValueError("Restart optimizer or step is incomplete")
    if state_hash(restart["model"]) != complete["final_model_sha256"]:
        raise ValueError("Restart and final weights differ")
    for value in restart["optimizer"]["state"].values():
        if int(value["step"]) != restart["step"]:
            raise ValueError("Optimizer state does not cover the complete edit budget")


def _write_csv(path, rows, meta=META):
    keys = set().union(*(row.keys() for row in rows)) if rows else set(meta)
    fields = [key for key in meta if key in keys] + sorted(keys - set(meta))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ("width", "world", "seed", "condition", "chain", "step"))
        arms = groups.setdefault(key, {})
        if row["arm"] in arms:
            raise ValueError("Duplicate model/arm/checkpoint outcome")
        arms[row["arm"]] = row
    output = []
    outcomes = sorted({key for row in rows for key in row if key.endswith(("_accuracy", "_rate"))})
    for key, arms in sorted(groups.items()):
        identity = dict(
            zip(("width", "world", "seed", "condition", "chain", "step"), key, strict=True)
        )
        for name, treatment, reference in CONTRASTS:
            if treatment not in arms or reference not in arms:
                continue
            row = {**identity, "contrast": name, "treatment": treatment, "reference": reference}
            for metric in outcomes:
                left, right = arms[treatment].get(metric), arms[reference].get(metric)
                row[f"delta_{metric}"] = (
                    left - right if left is not None and right is not None else None
                )
            output.append(row)
        if {"exception", "root-only", "actual-only", "old-fact-rehearsal"}.issubset(arms):
            row = {
                **identity,
                "contrast": "root-by-actual-interaction",
                "treatment": "exception-root-only",
                "reference": "actual-only-rehearsal",
            }
            # Only fixed-label outcomes have a shared causal-response interpretation here.
            for metric in outcomes:
                if metric.startswith("probe_"):
                    values = [
                        arms[arm][metric]
                        for arm in ("exception", "root-only", "actual-only", "old-fact-rehearsal")
                    ]
                    row[f"delta_{metric}"] = (
                        values[0] - values[1] - values[2] + values[3]
                        if all(value is not None for value in values)
                        else None
                    )
            output.append(row)
    return output


def summarize(root, require_complete=False, verify_weights=False):
    root = Path(root).resolve()
    launch = json.loads((root / "launch-contract.json").read_text())
    study, sources = launch["study"], launch["sources"]
    if sources != source_hashes():
        raise ValueError("The frozen P1 implementation has changed")
    worlds = {seed: make_cross_world(seed) for seed in study["worlds"]}
    rows, missing, complete_models, complete_edits, complete_refs, weights_verified = (
        [],
        [],
        0,
        0,
        0,
        0,
    )
    for job in launch["jobs"]:
        directory = root / job["name"]
        model_contract_path = directory / "contract.json"
        if (
            not model_contract_path.exists()
            or not (directory / "baseline-predictions.npz").exists()
        ):
            missing.append(job["name"])
            continue
        model_contract = json.loads(model_contract_path.read_text())
        if model_contract["study"] != study or model_contract["sources"] != sources:
            raise ValueError("Model contract differs from the launched matrix")
        parent = model_contract["parent"]
        if parent["directory"] != str(Path(job["baseline"]).resolve()):
            raise ValueError("Parent directory differs from launched matrix")
        if file_hash(Path(parent["directory"]) / "config.json") != parent["config_sha256"]:
            raise ValueError("Parent configuration has changed")
        world = worlds[parent["world"]]
        baseline = _read_predictions(directory / "baseline-predictions.npz", world.answers)
        parent_arrays = _read_predictions(
            Path(parent["directory"]) / f"predictions-{parent['step']}.npz", world.answers
        )
        for key in ("prediction", "ended", "correct"):
            if not np.array_equal(baseline[key], parent_arrays[key]):
                raise ValueError("Baseline predictions differ from the historical parent")
        if (directory / "complete.json").exists():
            model_complete = json.loads((directory / "complete.json").read_text())
            if model_complete["contract_sha256"] != _contract_hash(model_contract):
                raise ValueError("Model completion contract differs")
            if model_complete["edit_cases"] != 2 * len(study["arms"]):
                raise ValueError("Model completion case count differs")
            complete_models += 1
        for chain, chain_name in enumerate(CHAINS):
            pair = edit_pair(world, chain)
            choices = sampling_streams(world, chain, study["edit_steps"], study["batch_size"])
            for arm in (*REFERENCE_ARMS, *study["arms"]):
                is_reference = arm in REFERENCE_ARMS
                name = f"{chain_name}-{arm}" + ("-mlp" if is_reference else "")
                case = directory / ("references" if is_reference else "edits") / name
                if not (case / "contract.json").exists() or not (case / "trajectory.json").exists():
                    missing.append(f"{job['name']}/{name}")
                    continue
                spec = make_arm(world, chain, arm, pair)
                contract = json.loads((case / "contract.json").read_text())
                if contract["parent"] != parent:
                    raise ValueError("Case parent differs from model parent")
                _validate_sets(
                    case / "sets.npz", spec, baseline["correct"], None if is_reference else choices
                )
                complete = None
                if is_reference:
                    predictions = Path(contract["source"])
                    expected_source = Path(parent["directory"]) / "edits" / name
                    if predictions != expected_source:
                        raise ValueError("Reused reference directory differs")
                    if not (predictions / "complete.json").exists():
                        raise ValueError("Reused reference is incomplete")
                else:
                    predictions = case
                    expected = {
                        "study": study,
                        "sources": sources,
                        "chain": chain_name,
                        "arm": arm,
                        "sets_sha256": {key: array_hash(value) for key, value in spec.items()},
                        "edit_sampling_sha256": array_hash(choices[0]),
                        "replay_sampling_sha256": array_hash(choices[1]),
                        "old_correct_sha256": array_hash(baseline["correct"]),
                    }
                    _assert_values(contract, expected, str(case))
                    if (case / "complete.json").exists():
                        complete = json.loads((case / "complete.json").read_text())
                        if complete["contract_sha256"] != _contract_hash(contract):
                            raise ValueError("Case completion contract differs")
                        _validate_updates(case, spec, choices, study, complete)
                        if not all(
                            (case / file).exists() for file in ("resume.pt", "model-final.pt")
                        ):
                            raise ValueError("Completed edit is missing restart/final weights")
                        if verify_weights:
                            _validate_weights(case, complete, contract)
                            weights_verified += 1
                        complete_edits += 1
                timeline = json.loads((case / "trajectory.json").read_text())
                by_step = {point["step"]: point for point in timeline}
                if len(by_step) != len(timeline) or set(by_step) - set(study["edit_checkpoints"]):
                    raise ValueError("Duplicate or unplanned evaluation checkpoints")
                if is_reference or complete:
                    if set(by_step) != set(study["edit_checkpoints"]):
                        raise ValueError("A completed trajectory is missing planned checkpoints")
                    if is_reference:
                        complete_refs += 1
                for step, recorded in sorted(by_step.items()):
                    path = predictions / f"predictions-{step}.npz"
                    arrays = _read_predictions(path, spec["target"])
                    if is_reference and file_hash(path) != contract["files"][path.name]:
                        raise ValueError("Reused prediction file hash changed")
                    measured = arm_metrics(world, spec, arrays, baseline["correct"])
                    _assert_values(recorded, measured, str(path))
                    if complete and step == study["edit_steps"]:
                        _assert_values(complete["final"], measured, str(case / "complete.json"))
                    row = {key: parent[key] for key in ("width", "world", "seed", "condition")}
                    row.update(
                        chain=chain_name,
                        arm=arm,
                        source="reference" if is_reference else "new",
                        step=step,
                    )
                    row.update(flatten(measured))
                    row.update(flatten(response_probes(world, pair, arrays)))
                    rows.append(row)
    expected_models = len(launch["jobs"])
    expected_edits = expected_models * 2 * len(study["arms"])
    expected_refs = expected_models * 4
    done = (complete_models, complete_edits, complete_refs) == (
        expected_models,
        expected_edits,
        expected_refs,
    )
    audit = {
        "state": "complete" if done else "partial",
        "passed_available_predictions": True,
        "expected_models": expected_models,
        "complete_models": complete_models,
        "expected_new_edits": expected_edits,
        "complete_new_edits": complete_edits,
        "expected_reference_edits": expected_refs,
        "complete_reference_edits": complete_refs,
        "prediction_checkpoints_recomputed": len(rows),
        "expected_prediction_checkpoints": (expected_edits + expected_refs)
        * len(study["edit_checkpoints"]),
        "weights_and_optimizer_states_verified": weights_verified,
        "weight_verification_requested": verify_weights,
        "missing": missing,
        "sources": sources,
        "config": study,
        "limitations": [
            "Two reused development worlds",
            "Fixed-label response contrasts do not alone identify internal routing",
            "Retention pools differ across truth arms; supervised unchanged facts are separate",
            "No D/U targets enter editor optimization",
            "New-arm elapsed training time includes four class-gradient diagnostics; "
            "it is not an exact compute comparison with archived references",
        ],
    }
    if require_complete and not done:
        raise ValueError(
            f"P1 is incomplete: {complete_models}/{expected_models} models, "
            f"{complete_edits}/{expected_edits} new edits"
        )
    return rows, paired_rows(rows), audit


def report(rows, pairs, audit):
    final_step = audit["config"]["edit_steps"]
    terminal = [row for row in rows if row["step"] == final_step]
    lines = [
        "# P1 counterfactual editing",
        "",
        f"Status: {audit['state']}; "
        f"{audit['complete_new_edits']}/{audit['expected_new_edits']} new edits and "
        f"{audit['complete_reference_edits']}/{audit['expected_reference_edits']} references.",
        "",
        "Values below are equal-weight case means, not independent-query estimates. "
        "Changed-fact success is undefined for rehearsal. D is undefined when no derived "
        "truth changes. S always contains 93 supervised facts.",
        "",
        "| Width | Arm | Cases | Changed E | Heldout D | Conflict heldout D | "
        "Unseen U damage | Unchanged probe damage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    def number(values):
        values = [value for value in values if value is not None]
        return f"{100 * np.mean(values):.3f}%" if values else "—"

    for width, arm in sorted({(row["width"], row["arm"]) for row in terminal}):
        cases = [row for row in terminal if row["width"] == width and row["arm"] == arm]
        values = [
            number([row[key] for row in cases])
            for key in (
                "E_changed_accuracy",
                "D_heldout_accuracy",
                "D_conflict_heldout_accuracy",
                "U_unseen_rate",
                "U_D_probe_unchanged_rate",
            )
        ]
        lines.append(f"| {width} | {arm} | {len(cases)} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Balanced supervision versus uniform supervision",
            "",
            "Positive D differences favor balancing; positive U differences mean more damage. "
            "Keep each model/chain paired. No E-matched postselection is used.",
            "",
            "| Width | Paired cases | ΔE (pp) | Δconflict heldout D (pp) | Δunseen U (pp) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for width in sorted({row["width"] for row in terminal}):
        selected = [
            row
            for row in pairs
            if row["width"] == width
            and row["step"] == final_step
            and row["contrast"] == "balanced-minus-uniform"
        ]
        if selected:
            means = [
                f"{100 * np.mean([row[key] for row in selected]):+.3f}"
                for key in (
                    "delta_E_changed_accuracy",
                    "delta_D_conflict_heldout_accuracy",
                    "delta_U_unseen_rate",
                )
            ]
            lines.append(f"| {width} | {len(selected)} | " + " | ".join(means) + " |")
    lines.extend(
        [
            "",
            "The paired CSV also contains root-only/actual-only response contrasts and the "
            "factorial interaction. These use fixed new-default/new-actual/old-default labels on "
            "the same probe IDs; own-world accuracy differences alone are not interpreted as "
            "routing effects. Rehearsal is active old-fact supervision, not an unchanged model. "
            "Step zero supplies the unchanged baseline.",
            "",
            "`editing.csv` preserves all counts, known denominators, heldout versus supervised "
            "retention, cross-chain and independent-fact damage, and all planned checkpoints. "
            "`paired-editing.csv` retains width/world/initialization/organization/chain. "
            "No E/D/U composite or confirmatory significance is reported.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-mechanism-dev-v1/p1")
    parser.add_argument("--output")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-weights", action="store_true")
    args = parser.parse_args()
    rows, pairs, audit = summarize(args.root, args.require_complete, args.verify_weights)
    output = Path(args.output or args.root)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "editing.csv", rows)
    _write_csv(
        output / "paired-editing.csv",
        pairs,
        (
            "width",
            "world",
            "seed",
            "condition",
            "chain",
            "step",
            "contrast",
            "treatment",
            "reference",
        ),
    )
    write_json(output / "audit.json", audit)
    (output / "report.md").write_text(report(rows, pairs, audit))
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "state",
                    "complete_models",
                    "complete_new_edits",
                    "complete_reference_edits",
                    "prediction_checkpoints_recomputed",
                    "weights_and_optimizer_states_verified",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
