#!/usr/bin/env python3
"""Audit and report every registered same-bridge endpoint, without selecting runs.

Saved donor evaluations are balanced within recipient queries. Initializations
are averaged within worlds, then worlds receive equal weight. Figure whiskers
show world ranges, not confidence intervals. No model forward passes are run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("attention", "mlp", "both", "full")
ARCHITECTURES = ("c1", "c2", "cd", "l1", "l2")
METRICS = (
    "answer_accuracy",
    "complete_accuracy",
    "eos_accuracy",
    "target_probability",
    "target_log_probability",
    "target_margin",
)
COUNTS = (
    "n_recipients",
    "selected_recipients",
    "total_recipients",
    "coverage",
    "n_donor_evaluations",
)
GROUPS = (
    "all",
    "paired",
    "id_only",
    "ood_only",
    "neither",
    "paired_original_atomics_correct",
    "paired_baseline_failed",
    "paired_baseline_correct",
    "paired_all_atomics_correct",
    "paired_both_counterfactual_correct",
)
CONDITIONS = (
    "baseline",
    *("self_" + component for component in COMPONENTS),
    *("original_prefix_" + component for component in COMPONENTS),
    *(
        family + "_" + condition
        for family in ("id", "ood")
        for condition in ("baseline", "counterfactual", "self", *COMPONENTS)
    ),
)
KEYS = ("phase", "architecture", "step", "kind", "group", "condition", "component")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def expected_runs(config, root=ROOT):
    expected = []
    for phase, relative in config["historical_configs"].items():
        historical = read_json(root / relative)
        entries = []
        for run, overrides in historical["runs"].items():
            spec = {**historical["base"], **overrides}
            if spec["hops"] == 2:
                entries.append({"run": run, "phase": phase, "spec": spec})
        if len(entries) != config["expected_runs"][phase]:
            raise ValueError(f"Registered {phase} endpoint count differs from follow-up config")
        expected.extend(entries)
    if len({(r["phase"], r["run"]) for r in expected}) != len(expected):
        raise ValueError("Duplicate registered endpoints")
    return expected


def identity(entry):
    spec = entry["spec"]
    return {
        "run": entry["run"],
        "phase": entry["phase"],
        "architecture": spec["architecture"],
        "world_seed": spec["world_seed"],
        "initialization": spec["initialization"],
        "step": spec["steps"],
    }


def balanced_summary(values, indices, selected):
    """Independent recount of saved values, giving each recipient one vote."""
    indices = np.asarray(indices, dtype=np.int64)
    selected = np.asarray(selected, dtype=bool)
    n = len(selected)
    if indices.ndim != 1 or (len(indices) and (indices.min() < 0 or indices.max() >= n)):
        raise ValueError("Invalid saved recipient indices")
    count = np.bincount(indices, minlength=n)
    mask = selected & (count > 0)
    result = {
        "n_recipients": int(mask.sum()),
        "selected_recipients": int(selected.sum()),
        "total_recipients": n,
        "coverage": float(mask.mean()) if n else None,
        "n_donor_evaluations": int(selected[indices].sum()),
    }
    for metric in METRICS:
        vector = np.asarray(values[metric], dtype=np.float64)
        if vector.shape != indices.shape or not np.isfinite(vector).all():
            raise ValueError(f"Invalid saved metric: {metric}")
        sums = np.bincount(indices, weights=vector, minlength=n)
        result[metric] = float((sums[mask] / count[mask]).mean()) if mask.any() else None
    return result


def compare_summary(actual, declared, label):
    if set(declared) != set((*METRICS, *COUNTS)):
        raise ValueError(f"Incomplete metric schema: {label}")
    for field, value in actual.items():
        other = declared[field]
        if value is None:
            match = other is None
        else:
            match = (
                other is not None
                and np.isfinite(other)
                and np.isclose(value, other, rtol=1e-10, atol=1e-12)
            )
        if not match:
            raise ValueError(f"Saved predictions disagree with report: {label}.{field}")


def verify_raw_scores(path, report):
    """Recount every subgroup and paired contrast from saved metric vectors."""
    if set(report["conditions"]) != set(CONDITIONS) or set(report["scores"]) != set(GROUPS):
        raise ValueError("Incomplete condition/group matrix")
    expected_contrasts = {group + ":" + component for group in GROUPS for component in COMPONENTS}
    if set(report["contrasts"]) != expected_contrasts:
        raise ValueError("Incomplete paired contrast matrix")
    with np.load(path, allow_pickle=False) as saved:
        n = report["n_original_queries"]
        selected = {group: saved["subset_" + group] for group in GROUPS}
        if any(mask.dtype != np.bool_ or mask.shape != (n,) for mask in selected.values()):
            raise ValueError("Saved subset masks disagree with recipient pool")
        if not selected["all"].all() or int(selected["paired"].sum()) != report["n_common_queries"]:
            raise ValueError("Saved full/common recipient counts disagree with report")
        values = {
            condition: {metric: saved[condition + "_" + metric] for metric in METRICS}
            for condition in CONDITIONS
        }
        indices = {condition: saved[condition + "_recipient_indices"] for condition in CONDITIONS}
        for group, mask in selected.items():
            if set(report["scores"][group]) != set(CONDITIONS):
                raise ValueError("Incomplete scores within a subgroup")
            for condition in CONDITIONS:
                compare_summary(
                    balanced_summary(values[condition], indices[condition], mask),
                    report["scores"][group][condition],
                    group + "." + condition,
                )
        pid, pod = saved["pair_id_indices"], saved["pair_ood_indices"]
        pair_indices = saved["pair_recipient_indices"]
        if len(pair_indices) != report["n_pairs"]:
            raise ValueError("Saved donor-pair count disagrees with report")
        for component in COMPONENTS:
            contrasts = {
                "id_minus_ood": {
                    metric: values["id_" + component][metric][pid]
                    - values["ood_" + component][metric][pod]
                    for metric in METRICS
                },
                "baseline_adjusted_id_minus_ood": {
                    metric: (
                        values["id_" + component][metric][pid] - values["id_baseline"][metric][pid]
                    )
                    - (
                        values["ood_" + component][metric][pod]
                        - values["ood_baseline"][metric][pod]
                    )
                    for metric in METRICS
                },
            }
            for group, mask in selected.items():
                declared = report["contrasts"][group + ":" + component]
                if set(declared) != set(contrasts):
                    raise ValueError("Incomplete contrast types")
                for name, measured in contrasts.items():
                    compare_summary(
                        balanced_summary(measured, pair_indices, mask),
                        declared[name],
                        group + ":" + component + "." + name,
                    )


def endpoint_records(report, entry):
    base, records = identity(entry), []
    for group, scores in report["scores"].items():
        for condition, metrics in scores.items():
            component = next((c for c in COMPONENTS if condition.endswith("_" + c)), "none")
            records.append(
                {
                    **base,
                    "kind": "score",
                    "group": group,
                    "condition": condition,
                    "component": component,
                    **metrics,
                }
            )
        for family in ("id", "ood"):
            matched = scores[family + "_baseline"]
            for component in COMPONENTS:
                patched = scores[family + "_" + component]
                if any(patched[key] != matched[key] for key in COUNTS):
                    raise ValueError("Patch and matched baseline have different coverage")
                difference = {
                    metric: patched[metric] - matched[metric]
                    if patched[metric] is not None and matched[metric] is not None
                    else None
                    for metric in METRICS
                }
                records.append(
                    {
                        **base,
                        "kind": "contrast",
                        "group": group,
                        "condition": family + "_minus_baseline",
                        "component": component,
                        **{key: patched[key] for key in COUNTS},
                        **difference,
                    }
                )
    for key, contrasts in report["contrasts"].items():
        group, component = key.split(":")
        for name, metrics in contrasts.items():
            records.append(
                {
                    **base,
                    "kind": "contrast",
                    "group": group,
                    "condition": name,
                    "component": component,
                    **metrics,
                }
            )
    return records


def collect(config, config_path, *, root=ROOT, allow_partial=False):
    expected = expected_runs(config, root)
    artifact = root / "docs/development-artifacts" / config["experiment"]
    rows, manifest, matrix = [], [], []
    wanted = {(entry["phase"], entry["run"]) for entry in expected}
    for phase in config["historical_configs"]:
        directory = root / config["output_root"] / phase
        if directory.exists():
            for child in directory.iterdir():
                if (child / "complete.json").exists() and (phase, child.name) not in wanted:
                    raise ValueError(f"Unregistered completed endpoint: {child}")
    phase_locks = {}
    for phase in config["historical_configs"]:
        path = artifact / config.get("locks", {}).get(phase, phase + "-lock.json")
        if not path.exists():
            continue
        lock = read_json(path)
        if lock["phase"] != phase or lock["config_sha256"] != digest(config_path):
            raise ValueError("Frozen phase/config identity mismatch")
        entries = [entry for entry in expected if entry["phase"] == phase]
        if [e["run"] for e in entries] != [e["run"] for e in lock["runs"]]:
            raise ValueError("Frozen endpoint matrix differs from registration")
        for relative, expected_hash in lock["analysis_source_hashes"].items():
            for source in (root / relative, artifact / (path.stem + "-source") / relative):
                if digest(source) != expected_hash:
                    raise ValueError(f"Analysis source differs from freeze: {source}")
        phase_locks[phase] = (lock, digest(path))
    for entry in expected:
        base = identity(entry)
        directory = root / config["output_root"] / entry["phase"] / entry["run"]
        complete_path = directory / "complete.json"
        if not complete_path.exists() or entry["phase"] not in phase_locks:
            matrix.append(
                {
                    **base,
                    "state": "missing",
                    "reason": "phase_not_frozen"
                    if entry["phase"] not in phase_locks
                    else "incomplete_output"
                    if directory.exists()
                    else "not_started",
                }
            )
            continue
        lock, lock_hash = phase_locks[entry["phase"]]
        frozen = next(item for item in lock["runs"] if item["run"] == entry["run"])
        if frozen["spec"] != entry["spec"]:
            raise ValueError("Frozen historical specification differs from registration")
        hashes = {}
        for name, expected_hash in frozen["inputs"].items():
            path = Path(frozen["directory"]) / name
            hashes[str(path)] = digest(path)
            if hashes[str(path)] != expected_hash:
                raise ValueError(f"Historical input changed: {path}")
        for name, field in (
            ("donors_file", "donors_sha256"),
            ("donor_audit_file", "donor_audit_sha256"),
        ):
            path = root / frozen[name]
            hashes[str(path)] = digest(path)
            if hashes[str(path)] != frozen[field]:
                raise ValueError("Frozen donor selection changed")
        complete = read_json(complete_path)
        if (
            not complete["passed"]
            or complete["run"] != entry["run"]
            or complete["lock_sha256"] != lock_hash
        ):
            raise ValueError("Failed or mismatched endpoint completion")
        if set(complete["output_hashes"]) != {"metadata.json", "predictions.npz", "report.json"}:
            raise ValueError("Incomplete output hash manifest")
        for name, expected_hash in complete["output_hashes"].items():
            path = directory / name
            hashes[str(path)] = digest(path)
            if hashes[str(path)] != expected_hash:
                raise ValueError(f"Completed output changed: {path}")
        metadata, report = (
            read_json(directory / name) for name in ("metadata.json", "report.json")
        )
        for value in (metadata, report):
            if (
                value["run"] != entry["run"]
                or value["phase"] != entry["phase"]
                or value["spec"] != entry["spec"]
            ):
                raise ValueError("Output endpoint identity mismatch")
        if (
            metadata["lock_sha256"] != lock_hash
            or metadata["analysis_source_hashes"] != lock["analysis_source_hashes"]
            or metadata["donors_sha256"] != frozen["donors_sha256"]
        ):
            raise ValueError("Output provenance differs from phase lock")
        engineering = report["engineering"]
        for field in (
            "all_self_and_original_prefix_conditions_equal_baseline",
            "both_equals_full_first_execution_layer",
            "parameters_unchanged",
            "historical_answer_eos_reproduced",
            "eos_uses_generated_answer",
        ):
            if engineering[field] is not True:
                raise ValueError(f"Endpoint engineering check failed: {field}")
        if (
            engineering["training_updates"] != 0
            or engineering["mlp_recomputed_after_mixing"]
            or engineering["model_state_sha256_before"] != engineering["model_state_sha256_after"]
        ):
            raise ValueError("Intervention execution violated the frozen contract")
        verify_raw_scores(directory / "predictions.npz", report)
        rows.extend(endpoint_records(report, entry))
        hashes[str(complete_path)] = digest(complete_path)
        manifest.append(
            {
                **base,
                "lock_sha256": lock_hash,
                "artifact_hashes": hashes,
                "raw_query_balanced_recount_passed": True,
                "engineering": engineering,
                "donor_audit": report["donor_audit"],
            }
        )
        matrix.append({**base, "state": "complete", "reason": "all_hashes_and_raw_scores_verified"})
    missing = [row for row in matrix if row["state"] != "complete"]
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete registered matrix: {len(missing)}/{len(expected)} endpoints missing"
        )
    return rows, expected, manifest, matrix


def aggregate(rows, expected):
    """Nest initialization means within world means; retain undefined denominators."""
    seen, world_groups = set(), defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in (*KEYS, "world_seed", "initialization"))
        if key in seen:
            raise ValueError("Duplicate endpoint condition/world/initialization")
        seen.add(key)
        world_groups[tuple(row[field] for field in (*KEYS, "world_seed"))].append(row)
    worlds = []
    for members in world_groups.values():
        first = members[0]
        row = {key: first[key] for key in (*KEYS, "world_seed")}
        row["initializations"] = sorted(member["initialization"] for member in members)
        row["n_models"] = len(members)
        for metric in (*METRICS, *COUNTS):
            values = [member[metric] for member in members if member[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + "_initializations_with_denominator"] = len(values)
        worlds.append(row)
    grouped = defaultdict(list)
    for row in worlds:
        grouped[tuple(row[field] for field in KEYS)].append(row)
    summary = []
    for members in grouped.values():
        first = members[0]
        row = {key: first[key] for key in KEYS}
        registered = {
            (e["spec"]["world_seed"], e["spec"]["initialization"])
            for e in expected
            if e["phase"] == row["phase"] and e["spec"]["architecture"] == row["architecture"]
        }
        observed = {
            (member["world_seed"], seed) for member in members for seed in member["initializations"]
        }
        row.update(
            {
                "expected_worlds": len({w for w, _ in registered}),
                "observed_worlds": len(members),
                "expected_models": len(registered),
                "observed_models": len(observed),
                "missing_world_initializations": sorted(registered - observed),
                "registration_complete": observed == registered,
                "metrics": {},
            }
        )
        for metric in (*METRICS, *COUNTS):
            values = [member[metric] for member in members if member[metric] is not None]
            row["metrics"][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "worlds_with_denominator": len(values),
            }
        summary.append(row)
    return worlds, summary


def save_csv(path, rows):
    flattened = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "metrics"}
        for metric, stats in row.get("metrics", {}).items():
            flat.update({metric + "_" + key: value for key, value in stats.items()})
        flattened.append(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in flat.items()
            }
        )
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def plot(groups, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [
        phase
        for phase in ("development", "confirmation")
        if any(r["phase"] == phase for r in groups)
    ]
    if not phases:
        return []
    lookup = {
        tuple(
            r[key] for key in ("phase", "architecture", "kind", "group", "condition", "component")
        ): r
        for r in groups
    }
    paths = []

    def series(axis, phase, kind, group, condition, component, metric, label, color, offset=0):
        x, y, lower, upper = [], [], [], []
        for index, arch in enumerate(ARCHITECTURES):
            row = lookup.get((phase, arch, kind, group, condition, component))
            if row is None or row["metrics"][metric]["mean"] is None:
                continue
            stats = row["metrics"][metric]
            scale = (
                100
                if metric in ("answer_accuracy", "complete_accuracy", "eos_accuracy", "coverage")
                else 1
            )
            x.append(index + offset)
            y.append(scale * stats["mean"])
            lower.append(scale * (stats["mean"] - stats["min"]))
            upper.append(scale * (stats["max"] - stats["mean"]))
        if x:
            axis.errorbar(
                x,
                y,
                yerr=[lower, upper],
                marker="o",
                capsize=3,
                linewidth=1.2,
                markersize=4,
                label=label,
                color=color,
            )

    specifications = (
        ("paired-outcomes", ("complete_accuracy", "target_probability", "target_margin")),
        ("component-contrasts", ("complete_accuracy", "target_probability", "target_margin")),
        ("coverage", ("coverage", "coverage")),
    )
    for name, metrics in specifications:
        fig, axes = plt.subplots(
            len(phases),
            len(metrics),
            figsize=(12, 3.8 * len(phases)),
            squeeze=False,
            constrained_layout=True,
        )
        for i, phase in enumerate(phases):
            for j, metric in enumerate(metrics):
                axis = axes[i, j]
                if name == "paired-outcomes":
                    for condition, label, color, component, offset in (
                        ("id_baseline", "Matched OOD recipient", "#777777", "none", -0.15),
                        ("id_full", "ID prefix: full", "#2878B5", "full", -0.05),
                        ("ood_full", "OOD prefix: full", "#D77A20", "full", 0.05),
                        ("id_counterfactual", "Changed-head mixed query", "#26966A", "none", 0.15),
                    ):
                        series(
                            axis,
                            phase,
                            "score",
                            "paired",
                            condition,
                            component,
                            metric,
                            label,
                            color,
                            offset,
                        )
                elif name == "component-contrasts":
                    for component, color, offset in zip(
                        COMPONENTS,
                        ("#2878B5", "#D77A20", "#26966A", "#9B5AA4"),
                        (-0.15, -0.05, 0.05, 0.15),
                        strict=True,
                    ):
                        series(
                            axis,
                            phase,
                            "contrast",
                            "paired",
                            "id_minus_ood",
                            component,
                            metric,
                            component.upper(),
                            color,
                            offset,
                        )
                    axis.axhline(0, color="#777777", linewidth=0.8)
                elif j == 0:
                    for condition, label, color in (
                        ("id_baseline", "ID donor available", "#2878B5"),
                        ("ood_baseline", "OOD donor available", "#D77A20"),
                    ):
                        series(axis, phase, "score", "all", condition, "none", metric, label, color)
                    series(
                        axis,
                        phase,
                        "score",
                        "paired",
                        "baseline",
                        "none",
                        metric,
                        "Both donor families",
                        "#26966A",
                    )
                else:
                    for group, label, color in (
                        ("paired", "Unfiltered paired", "#777777"),
                        ("paired_original_atomics_correct", "Recipient atoms correct", "#2878B5"),
                        ("paired_all_atomics_correct", "All pair atoms correct", "#D77A20"),
                        (
                            "paired_both_counterfactual_correct",
                            "Both full queries correct",
                            "#26966A",
                        ),
                    ):
                        series(
                            axis, phase, "score", group, "baseline", "none", metric, label, color
                        )
                axis.set_title(
                    phase
                    + " | "
                    + (
                        "eligible pools"
                        if name == "coverage" and j == 0
                        else "conditional subsets"
                        if name == "coverage"
                        else metric.replace("_", " ")
                    )
                )
                axis.set_xticks(range(len(ARCHITECTURES)), [a.upper() for a in ARCHITECTURES])
                axis.set_ylabel(
                    "Coverage of full OOD pool (%)"
                    if name == "coverage"
                    else "Percentage points"
                    if name == "component-contrasts" and j == 0
                    else "Accuracy (%)"
                    if metric == "complete_accuracy"
                    else "Probability difference"
                    if name == "component-contrasts" and j == 1
                    else "Target probability"
                    if metric == "target_probability"
                    else "Target-margin difference"
                    if name == "component-contrasts"
                    else "Target minus strongest other logit"
                )
                axis.grid(axis="y", alpha=0.18)
                axis.spines[["top", "right"]].set_visible(False)
                if name == "coverage":
                    axis.set_ylim(-1, 101)
                axis.legend(fontsize=7, frameon=False)
        fig.suptitle(
            "Same bridge | "
            + name.replace("-", " ")
            + "\nQuery-balanced donors; seeds within world, then equal worlds; "
            "whiskers = world range",
            fontsize=11,
        )
        for suffix in ("png", "pdf"):
            path = out / (name + "." + suffix)
            fig.savefig(path, dpi=200)
            paths.append(str(path))
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/grok-loop-same-bridge-v1.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/development-artifacts/grok-loop-same-bridge-v1/report",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Development calibration only; preserve all missing registered endpoints",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    rows, expected, manifest, matrix = collect(
        config, args.config, allow_partial=args.allow_partial
    )
    worlds, groups = aggregate(rows, expected)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("endpoints", rows),
        ("worlds", worlds),
        ("contrasts", [row for row in rows if row["kind"] == "contrast"]),
        ("world-contrasts", [row for row in worlds if row["kind"] == "contrast"]),
        ("run-matrix", matrix),
    ):
        save_csv(args.out / (name + ".csv"), values)
    figures = [] if args.no_plots else plot(groups, args.out)
    complete = all(row["state"] == "complete" for row in matrix)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "frozen intervention follow-up on previously observed endpoints",
        "statistical_unit": (
            "donors/pairs within recipient; initializations within world; equal worlds"
        ),
        "world_range_is_confidence_interval": False,
        "primary_comparison": (
            "paired full-residual ID minus OOD; matched baseline effects also retained"
        ),
        "components": list(COMPONENTS),
        "groups": groups,
        "expected_endpoints": len(expected),
        "completed_endpoints": len(manifest),
        "complete_registered_matrix": complete,
        "partial_calibration": args.allow_partial and not complete,
        "missing": [row for row in matrix if row["state"] != "complete"],
        "new_forward_passes": 0,
        "new_training_steps": 0,
        "config_sha256": digest(args.config),
        "reporter_sha256": digest(Path(__file__)),
        "report_contract_tests_sha256": digest(ROOT / "tests/test_grok_loop_same_bridge_report.py"),
        "execution_revision": config.get("execution_revision", "unversioned"),
        "figures": figures,
    }
    save_json(args.out / "summary.json", summary)
    save_json(args.out / "completion-matrix.json", matrix)
    save_json(args.out / "source-manifest.json", manifest)
    outputs = {
        path.name: digest(path)
        for path in args.out.iterdir()
        if path.is_file() and path.name != "output-hashes.json"
    }
    save_json(args.out / "output-hashes.json", outputs)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "expected_endpoints",
                    "completed_endpoints",
                    "complete_registered_matrix",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
