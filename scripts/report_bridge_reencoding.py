#!/usr/bin/env python3
"""Summarize fixed-weight re-encoding, with development-only alpha selection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def mean(values):
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def flatten_metrics(groups):
    result = {}
    for group, label in (
        ("common_atomic", "atomic"),
        ("train_composite", "train"),
        ("familiar_test", "familiar"),
        ("strict_test", "strict"),
    ):
        metrics = groups[group]
        for key in ("n", "accuracy", "answer_accuracy", "answer_nll", "format_accuracy"):
            result[f"{label}_{key}"] = metrics.get(key)
        if label in ("familiar", "strict"):
            fixed = metrics["fixed_baseline_atoms_subset"]
            for key in ("n", "coverage", "accuracy"):
                result[f"{label}_fixed_atoms_{key}"] = fixed[key]
            for key in (
                "atomic_correct_coverage",
                "decoded_bridge_correct_coverage",
                "selected_bridge_correct_coverage",
                "alternative_accidental_gold_match",
                "autonomous_two_calls",
            ):
                result[f"{label}_{key}"] = metrics.get(key)
            for subset in ("self_bridge_correct_subset", "common_atoms_and_self_bridge_subset"):
                for key in ("n", "coverage", "accuracy"):
                    result[f"{label}_{subset}_{key}"] = metrics.get(subset, {}).get(key)
    return result


def load_runs(root, phase):
    records, pending, identities = [], [], set()
    root = Path(root)
    for path in sorted(root.rglob("run.json")):
        if not (path.parent / "metrics.json").exists():
            continue
        manifest = read_json(path)
        if "cases" not in manifest or "parent_spec" not in manifest:
            continue
        parent = manifest["parent_spec"]
        declared = manifest.get("spec", {}).get("phase", parent.get("phase", ""))
        actual_phase = "development" if declared.startswith("development") else declared
        if actual_phase != phase:
            continue
        arm = manifest.get("spec", {}).get("arm", parent["arm"])
        arm = "aligned" if arm == "aligned_0.3" else arm
        identity = (int(parent["world"]), int(parent["initialization"]), arm)
        if identity in identities:
            raise ValueError(f"Duplicate parent model (not an independent replicate): {identity}")
        identities.add(identity)
        complete = path.parent / "complete.json"
        audited = complete.exists() and read_json(complete).get("independently_reloaded", False)
        if not audited:
            pending.append(str(path.parent))
        records.append(
            {
                "run": str(path.parent),
                "world": identity[0],
                "initialization": identity[1],
                "arm": arm,
                "parent_arm": parent["arm"],
                "audited": audited,
                "checkpoint_sha256": manifest.get("source", {}).get("parent_checkpoint"),
                "cases": manifest["cases"],
                "metrics": read_json(path.parent / "metrics.json"),
            }
        )
    return records, pending


def endpoint_rows(records):
    rows = []
    for record in records:
        baseline_atomic = record["metrics"]["baseline"]["common_atomic"]["accuracy"]
        for name, case in record["cases"].items():
            if name not in record["metrics"]:
                raise ValueError(f"Missing case {name}: {record['run']}")
            row = {key: record[key] for key in ("run", "world", "initialization", "arm", "audited")}
            row.update(case=name, **case)
            row["values"] = flatten_metrics(record["metrics"][name])
            row["values"]["atomic_delta_from_baseline"] = (
                row["values"]["atomic_accuracy"] - baseline_atomic
            )
            rows.append(row)
    return rows


def hierarchical_mean(rows):
    """Average initializations within worlds, then give each world equal weight."""
    worlds = defaultdict(list)
    for row in rows:
        worlds[row["world"]].append(row)
    names = sorted({key for row in rows for key in row["values"]})
    per_world = []
    for world, group in sorted(worlds.items()):
        per_world.append(
            {
                "world": world,
                "initializations": sorted({row["initialization"] for row in group}),
                "models": len(group),
                "values": {key: mean([row["values"].get(key) for row in group]) for key in names},
            }
        )
    return {
        "worlds": len(worlds),
        "models": len(rows),
        "per_world": per_world,
        "values": {key: mean([row["values"][key] for row in per_world]) for key in names},
    }


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["variant"], row["condition"], row["alpha"])].append(row)
    return [
        {
            "arm": key[0],
            "variant": key[1],
            "condition": key[2],
            "alpha": key[3],
            **hierarchical_mean(group),
        }
        for key, group in sorted(groups.items())
    ]


def development_decision(rows):
    # Neither strict scores nor other training arms enter this selection.
    candidates = [
        r
        for r in rows
        if r["arm"] == "bridge_ce"
        and r["variant"] == "norm_matched"
        and r["condition"] == "self_decode"
    ]
    identities = {(r["world"], r["initialization"]) for r in candidates}
    scores = []
    for alpha in sorted({r["alpha"] for r in candidates}):
        group = [r for r in candidates if r["alpha"] == alpha]
        complete = {(r["world"], r["initialization"]) for r in group} == identities
        eligible = complete and all(
            r["values"]["atomic_delta_from_baseline"] >= -0.01 - 1e-12 for r in group
        )
        score = hierarchical_mean(group)["values"]["familiar_accuracy"]
        scores.append(
            {
                "alpha": alpha,
                "eligible": eligible,
                "complete_across_development_models": complete,
                "familiar_accuracy_world_equal": score,
                "minimum_atomic_delta": min(
                    r["values"]["atomic_delta_from_baseline"] for r in group
                ),
            }
        )
    eligible = [score for score in scores if score["eligible"]]
    chosen = (
        min(eligible, key=lambda s: (-s["familiar_accuracy_world_equal"], s["alpha"]))
        if eligible
        else None
    )
    return {
        "phase": "development",
        "status": "selected" if chosen else "no_eligible_alpha",
        "selected_alpha": chosen["alpha"] if chosen else None,
        "selection_arm": "bridge_ce",
        "selection_variant": "norm_matched",
        "selection_condition": "self_decode",
        "atomic_drop_tolerance": 0.01,
        "rule": "highest familiar holdout accuracy with <=.01 atomic drop; ties use smaller alpha",
        "strict_used_for_selection": False,
        "other_training_arms_used_for_selection": False,
        "candidates": scores,
    }


def fixed_alpha_from_config(path):
    config = read_json(path)
    values = []
    for node in (config, config.get("confirmation", {}), config.get("selection", {})):
        for key in ("fixed_alpha", "confirmation_alpha", "selected_alpha", "alpha"):
            if node.get(key) is not None:
                values.append(float(node[key]))
        if "alphas" in node and len(node["alphas"]) == 1:
            values.append(float(node["alphas"][0]))
    if len(set(values)) != 1:
        raise ValueError("Config must specify one unambiguous frozen alpha")
    return values[0]


def norm_sensitivity(rows):
    paired = defaultdict(dict)
    for row in rows:
        if row["condition"] == "baseline":
            continue
        key = (row["world"], row["initialization"], row["arm"], row["condition"], row["alpha"])
        paired[key][row["variant"]] = row
    result = []
    for key, variants in sorted(paired.items()):
        if {"paper_unit", "norm_matched"} <= variants.keys():
            result.append(
                {
                    "world": key[0],
                    "initialization": key[1],
                    "arm": key[2],
                    "condition": key[3],
                    "alpha": key[4],
                    "paper_unit_minus_norm_matched": {
                        name: variants["paper_unit"]["values"][name]
                        - variants["norm_matched"]["values"][name]
                        for name in ("atomic_accuracy", "familiar_accuracy", "strict_accuracy")
                    },
                }
            )
    return result


def report(root, phase, out=None, alpha=None, config=None):
    if phase not in ("development", "confirmation"):
        raise ValueError("phase must be development or confirmation")
    records, pending = load_runs(root, phase)
    if not records:
        raise ValueError(f"No matching re-encoding runs found under {root}")
    rows = endpoint_rows(records)
    if phase == "development":
        if alpha is not None or config is not None:
            raise ValueError("Development alpha must be selected by the frozen rule")
        decision = development_decision(rows)
    else:
        if alpha is None and config is None:
            raise ValueError("Confirmation requires --alpha or --config; no score-based selection")
        from_config = fixed_alpha_from_config(config) if config is not None else None
        if alpha is not None and from_config is not None and float(alpha) != from_config:
            raise ValueError("Explicit alpha conflicts with the configuration")
        alpha = float(alpha) if alpha is not None else from_config
        if not 0 <= alpha <= 1:
            raise ValueError("Frozen alpha must be in [0,1]")
        decision = {
            "phase": phase,
            "status": "fixed",
            "selected_alpha": alpha,
            "selection_source": str(config) if config is not None else "explicit --alpha",
            "strict_used_for_selection": False,
            "confirmation_scores_used_for_selection": False,
        }
    cells = defaultdict(set)
    for record in records:
        cells[(record["world"], record["initialization"])].add(record["arm"])
    missing_arms = [
        {
            "world": key[0],
            "initialization": key[1],
            "missing_arms": sorted({"baseline", "bridge_ce", "aligned"} - arms),
        }
        for key, arms in sorted(cells.items())
        if not {"baseline", "bridge_ce", "aligned"} <= arms
    ]
    selected_alpha = decision["selected_alpha"]
    missing_selected = []
    if selected_alpha is not None:
        for record in records:
            if not any(
                case["condition"] == "self_decode"
                and case["variant"] == "norm_matched"
                and case["alpha"] == selected_alpha
                for case in record["cases"].values()
            ):
                missing_selected.append(record["run"])
    ready = not pending and not missing_arms and not missing_selected
    decision.update(
        ready_for_frozen_comparison=ready,
        pending_audits=pending,
        missing_arm_cells=missing_arms,
        missing_selected_alpha_runs=missing_selected,
    )
    if not ready:
        decision["status"] = "incomplete"
    selected_rows = [
        row for row in rows if row["condition"] == "baseline" or row["alpha"] == selected_alpha
    ]
    summary = {
        "phase": phase,
        "independent_worlds": len({r["world"] for r in records}),
        "parent_models": len(records),
        "evaluated_cases": len(rows),
        "aggregation": "initializations nested in world; worlds equally weighted",
        "replication_unit": "world; alpha=0 and intervention cases are repeated measurements",
        "atomic_subset": "fixed baseline correctness of both necessary atoms",
        "all_endpoints": rows,
        "all_aggregates": aggregate(rows),
        "selected_alpha": selected_alpha,
        "selected_comparison": aggregate(selected_rows),
        "norm_sensitivity": norm_sensitivity(rows),
        "pending_audits": pending,
    }
    out = Path(out) if out is not None else Path(root) / f"report-{phase}"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("summary", summary), ("decision", decision)):
        (out / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return summary, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("development", "confirmation"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    summary, decision = report(**vars(args))
    print(
        f"{args.phase}: {summary['parent_models']} models in "
        f"{summary['independent_worlds']} worlds; alpha={decision['selected_alpha']}; "
        f"status={decision['status']}"
    )
    for row in summary["selected_comparison"]:
        if row["variant"] != "norm_matched":
            continue
        values = row["values"]
        print(
            f"{row['arm']:10s} {row['condition']:12s} a={row['alpha']:g} "
            f"atomic={100 * values['atomic_accuracy']:.2f}% "
            f"familiar={100 * values['familiar_accuracy']:.2f}% "
            f"strict={100 * values['strict_accuracy']:.2f}% "
            f"fixed-atom-coverage={100 * values['familiar_fixed_atoms_coverage']:.2f}%/"
            f"{100 * values['strict_fixed_atoms_coverage']:.2f}%"
        )


if __name__ == "__main__":
    main()
