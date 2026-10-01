#!/usr/bin/env python3
"""Report all loop interventions and fixed-checkpoint recurrence scans.

Initializations are averaged within each world, then worlds receive equal
weight. Development and confirmation, checkpoints, training recipes, donor
seeds and analysis implementations remain separate. No layer or recurrence
count is selected by performance; missing donors have undefined success rates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED_KEYS = {"world_seed", "initialization", "stream_seed"}
ARCH_KEYS = {"architecture", "layers", "repeats"}
GROUP_KEYS = (
    "kind",
    "condition_id",
    "comparison_id",
    "phase",
    "hops",
    "architecture",
    "width",
    "unique_layers",
    "training_repeats",
    "init_scheme",
    "step",
    "split",
    "donor_seed",
    "analysis_version",
    "group",
    "condition",
    "family",
    "component",
    "execution_layer",
    "evaluated_repeats",
)
COMPONENTS = ("attention", "mlp", "both", "full")
COLORS = {"attention": "#2878B5", "mlp": "#D77A20", "both": "#26966A", "full": "#9B5AA4"}
MARKERS = {"attention": "o", "mlp": "s", "both": "^", "full": "D"}


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def common(spec, run_id, step):
    return {
        "condition_id": fingerprint({k: v for k, v in spec.items() if k not in SEED_KEYS}),
        "comparison_id": fingerprint(
            {k: v for k, v in spec.items() if k not in SEED_KEYS | ARCH_KEYS}
        ),
        "phase": spec["phase"],
        "hops": spec["hops"],
        "architecture": spec.get("architecture", f"b{spec['layers']}r{spec['repeats']}"),
        "width": spec["width"],
        "unique_layers": spec["layers"],
        "training_repeats": spec["repeats"],
        "init_scheme": spec["init_scheme"],
        "step": step,
        "world_seed": spec["world_seed"],
        "initialization": spec["initialization"],
        "run_id": run_id,
    }


def expected_runs(paths):
    runs = []
    seen = set()
    for path in paths:
        config = read_json(path)
        for run_id, overrides in config["runs"].items():
            spec = {**config["base"], **overrides}
            item = common(spec, run_id, spec["steps"])
            key = (item["condition_id"], item["world_seed"], item["initialization"])
            if key in seen:
                raise ValueError("Configurations register the same condition/world/seed twice")
            seen.add(key)
            runs.append({**item, "config": str(path)})
    return runs


def mechanism_records(summary, metadata, run_id):
    """Flatten every reported subgroup, layer and component, including controls."""
    hashes = metadata.get("analysis_source_hashes", {})
    versions = [
        value for path, value in hashes.items() if Path(path).name == "grok_loop_mechanism.py"
    ]
    if len(versions) > 1:
        raise ValueError("Ambiguous analysis implementation identity")
    base = {
        **common(metadata["spec"], run_id, metadata["step"]),
        "split": summary["split"],
        "donor_seed": metadata["donor_seed"],
        "analysis_version": versions[0] if versions else "unrecorded",
        "evaluated_repeats": -1,
    }
    rows = []
    for group, scores in summary["scores"].items():
        for condition, metrics in scores["conditions"].items():
            description = summary["conditions"][condition]
            rows.append(
                {
                    **base,
                    "kind": "intervention",
                    "group": group,
                    "condition": condition,
                    "family": description["family"] or "none",
                    "component": description.get("component", "none"),
                    "execution_layer": description.get("patch_layer", -1) + 1,
                    "metrics": dict(metrics),
                }
            )
    for family, values in summary["donor_coverage"].items():
        n, total = values["n"], values["total_n"]
        metrics = {
            "n": n,
            "total_n": total,
            "coverage": n / total if total else None,
            "missing_n": total - n,
        }
        metrics.update(
            {
                "counterfactual_" + key + "_n": values["counterfactual_split"].get(key, 0)
                for key in ("train", "reserved_test", "unused", "ood", "missing")
            }
        )
        rows.append(
            {
                **base,
                "kind": "donor_coverage",
                "group": "all_rows",
                "condition": family,
                "family": family,
                "component": "none",
                "execution_layer": 0,
                "metrics": metrics,
            }
        )
    for family, values in summary["atomic_preconditions"].items():
        metrics = {k: v for k, v in values.items() if k != "per_hop_complete_accuracy"}
        for hop, accuracy in enumerate(values["per_hop_complete_accuracy"] or []):
            metrics[f"hop_{hop + 1}_complete_accuracy"] = accuracy
        rows.append(
            {
                **base,
                "kind": "atomic_precondition",
                "group": "all_rows",
                "condition": family,
                "family": family,
                "component": "none",
                "execution_layer": 0,
                "metrics": metrics,
            }
        )
    return rows


def recurrence_records(summary):
    rows = []
    for run in summary["runs"]:
        spec = run["spec"]
        checkpoint_steps = {
            int(match.group(1))
            for path in run["checkpoint"]
            if (match := re.search(r"weights-(\d+)\.pt$", path))
        }
        if len(checkpoint_steps) != 1:
            raise ValueError("Recurrence scan must identify exactly one frozen checkpoint step")
        base = {
            **common(spec, run["run"], checkpoint_steps.pop()),
            "kind": "recurrence",
            "donor_seed": -1,
            "analysis_version": fingerprint(summary.get("source", {})),
            "group": "all_rows",
            "condition": "fixed_checkpoint_scan",
            "family": "none",
            "component": "none",
            "execution_layer": 0,
        }
        for measurement in run["measurements"]:
            for split in (
                "atomic",
                "id_atomic",
                "ood_atomic",
                "test_composite",
                "test_full_composite",
                "ood_composite",
            ):
                if split not in measurement:
                    continue
                metrics = dict(measurement[split])
                metrics["executed_depth"] = measurement["effective_depth"]
                rows.append(
                    {
                        **base,
                        "split": split,
                        "evaluated_repeats": measurement["repeats"],
                        "metrics": metrics,
                    }
                )
    return rows


def aggregate(records, expected=()):
    """No query pooling; undefined metrics remain missing, never become zero."""
    by_world = defaultdict(list)
    seen = set()
    for record in records:
        group = tuple(record[k] for k in GROUP_KEYS)
        key = (*group, record["world_seed"])
        identity = (*key, record["initialization"])
        if identity in seen:
            raise ValueError("Duplicate analysis for the same condition/world/initialization")
        seen.add(identity)
        by_world[key].append(record)
    world_rows = []
    for key, members in sorted(by_world.items()):
        metrics = sorted({name for member in members for name in member["metrics"]})
        values = {}
        counts = {}
        for metric in metrics:
            available = [
                m["metrics"].get(metric) for m in members if m["metrics"].get(metric) is not None
            ]
            values[metric] = float(np.mean(available)) if available else None
            counts[metric] = len(available)
        world_rows.append(
            {
                **dict(zip(GROUP_KEYS, key[:-1], strict=True)),
                "world_seed": key[-1],
                "initializations": sorted(m["initialization"] for m in members),
                "metrics": values,
                "initializations_with_denominator": counts,
            }
        )
    groups = defaultdict(list)
    expected_pairs = defaultdict(set)
    for item in expected:
        expected_pairs[item["condition_id"], item["step"]].add(
            (item["world_seed"], item["initialization"])
        )
    for row in world_rows:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)
    summary = []
    for key, members in sorted(groups.items()):
        item = dict(zip(GROUP_KEYS, key, strict=True))
        wanted = expected_pairs.get((item["condition_id"], item["step"]))
        observed = {(m["world_seed"], seed) for m in members for seed in m["initializations"]}
        metrics = {}
        for metric in sorted({name for member in members for name in member["metrics"]}):
            values = [
                m["metrics"].get(metric) for m in members if m["metrics"].get(metric) is not None
            ]
            metrics[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "worlds_with_denominator": len(values),
            }
        summary.append(
            {
                **item,
                "worlds": sorted(m["world_seed"] for m in members),
                "worlds_observed": len(members),
                "expected_worlds": len({w for w, _ in wanted}) if wanted is not None else None,
                "missing_world_initializations": sorted(wanted - observed)
                if wanted is not None
                else None,
                "registration_complete": not (wanted - observed) if wanted is not None else None,
                "metrics": metrics,
            }
        )
    return world_rows, summary


def deduplicate_records(records):
    """Overlapping scan files are provenance copies, not independent replicates."""
    unique = {}
    duplicate_count = 0
    for row in records:
        key = tuple(row[k] for k in (*GROUP_KEYS, "world_seed", "initialization"))
        if key in unique:
            if unique[key]["metrics"] != row["metrics"]:
                raise ValueError("Conflicting results for the same checkpoint and intervention")
            duplicate_count += 1
        else:
            unique[key] = row
    return list(unique.values()), duplicate_count


def discover(inputs, recurrence_inputs):
    records, manifests, pending = [], [], []
    paths = sorted({p.resolve() for root in inputs for p in Path(root).rglob("summary.json")})
    for path in paths:
        metadata_path = path.with_name("metadata.json")
        if not metadata_path.is_file():
            continue
        summary = read_json(path)
        if "scores" not in summary or "conditions" not in summary:
            continue
        status = path.with_name("status.json")
        if not status.exists() or read_json(status).get("state") != "complete":
            pending.append(str(path.parent))
            continue
        metadata = read_json(metadata_path)
        run_id = path.parent.parent.name
        records.extend(mechanism_records(summary, metadata, run_id))
        manifests.append(
            {
                **common(metadata["spec"], run_id, metadata["step"]),
                "kind": "mechanism",
                "split": summary["split"],
                "summary": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "engineering_checks": summary["engineering_checks"],
            }
        )
    recurrence_paths = set()
    for root in recurrence_inputs:
        root = Path(root)
        recurrence_paths.update(
            path.resolve() for path in ([root] if root.is_file() else root.rglob("summary.json"))
        )
    for path in sorted(recurrence_paths):
        summary = read_json(path)
        if not summary.get("runs") or "measurements" not in summary["runs"][0]:
            continue
        if "finished_utc" not in summary:
            pending.append(str(path))
            continue
        records.extend(recurrence_records(summary))
        manifests.append(
            {
                "kind": "recurrence",
                "summary": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records, manifests, pending


def completeness(expected, manifests, splits):
    observed = {
        (m["condition_id"], m["world_seed"], m["initialization"], m["step"], m["split"])
        for m in manifests
        if m["kind"] == "mechanism"
    }
    states = []
    for run in expected:
        for split in splits:
            key = (
                run["condition_id"],
                run["world_seed"],
                run["initialization"],
                run["step"],
                split,
            )
            states.append({**run, "split": split, "complete": key in observed})
    return {
        "expected_endpoints": len(states) if expected else None,
        "completed_expected_endpoints": sum(s["complete"] for s in states) if expected else None,
        "all_registered_endpoints_complete": all(s["complete"] for s in states)
        if expected
        else None,
        "missing": [s for s in states if not s["complete"]],
        "note": "Completeness is only against supplied configurations and requested splits; "
        "absent registration means unknown expected world/seed counts.",
    }


def csv_rows(rows):
    flattened = []
    for row in rows:
        result = {
            k: v for k, v in row.items() if k not in ("metrics", "initializations_with_denominator")
        }
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, dict):
                result.update({key + "_" + suffix: number for suffix, number in value.items()})
            else:
                result[key] = value
        for key, value in row.get("initializations_with_denominator", {}).items():
            result[key + "_initializations_with_denominator"] = value
        flattened.append(
            {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in result.items()
            }
        )
    return flattened


def write_csv(path, rows):
    rows = csv_rows(rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(axis, ylabel):
    axis.set_ylim(-2, 102)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.16)
    axis.set_axisbelow(True)


def _curve(axis, rows, metric, color, label, marker="o", linestyle="-"):
    rows = sorted(rows, key=lambda row: row["execution_layer"])
    if not rows:
        return
    stats = [row["metrics"].get(metric, {}) for row in rows]
    x = [row["execution_layer"] for row in rows]

    def values(key):
        return [100 * s[key] if s.get(key) is not None else np.nan for s in stats]

    axis.plot(
        x,
        values("mean"),
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=label,
        markersize=4,
    )
    axis.fill_between(x, values("min"), values("max"), color=color, alpha=0.12)


def plot_mechanisms(groups, output, expected=()):
    plt = _matplotlib()
    records = [r for r in groups if r["kind"] == "intervention" and r["group"] == "all_rows"]
    facets = defaultdict(list)
    for row in records:
        facets[
            (
                row["comparison_id"],
                row["phase"],
                row["hops"],
                row["step"],
                row["donor_seed"],
                row["analysis_version"],
            )
        ].append(row)
    figures = []
    for facet, members in sorted(facets.items()):
        comparison, phase, hops, step, donor_seed, version = facet
        arches = {r["architecture"]: (r["unique_layers"], r["training_repeats"]) for r in members}
        for item in expected:
            if item["comparison_id"] == comparison:
                arches[item["architecture"]] = item["unique_layers"], item["training_repeats"]
        arches = dict(sorted(arches.items()))
        id_splits = sorted({r["split"] for r in members if r["split"] != "ood_composite"}) or [
            "test_composite"
        ]
        for id_split in id_splits:
            for family, metric, title in (
                ("different", "target_complete_accuracy", "New target + EOS"),
                ("same", "original_complete_accuracy", "Same-bridge: original target + EOS"),
            ):
                fig, axes = plt.subplots(
                    len(arches),
                    2,
                    figsize=(10, 2.8 * len(arches)),
                    squeeze=False,
                    constrained_layout=True,
                )
                for row_index, (arch, (layers, repeats)) in enumerate(arches.items()):
                    for col, split in enumerate((id_split, "ood_composite")):
                        axis = axes[row_index, col]
                        selected = [
                            r for r in members if r["architecture"] == arch and r["split"] == split
                        ]
                        patches = [
                            r
                            for r in selected
                            if r["family"] == family and r["execution_layer"] > 0
                        ]
                        for component in COMPONENTS:
                            _curve(
                                axis,
                                [r for r in patches if r["component"] == component],
                                metric,
                                COLORS[component],
                                component,
                                MARKERS[component],
                            )
                        native = next(
                            (
                                r
                                for r in selected
                                if r["condition"] == family + "_counterfactual_input"
                            ),
                            None,
                        )
                        if native and native["metrics"][metric]["mean"] is not None:
                            axis.axhline(
                                native["metrics"][metric]["mean"] * 100,
                                color="#777777",
                                linestyle="--",
                                linewidth=1,
                                label="Full counterfactual input",
                            )
                        if family == "different":
                            unpatched = next(
                                (
                                    r
                                    for r in selected
                                    if r["condition"] == "different_matched_baseline"
                                ),
                                None,
                            )
                            if unpatched and unpatched["metrics"][metric]["mean"] is not None:
                                axis.axhline(
                                    unpatched["metrics"][metric]["mean"] * 100,
                                    color="#aaaaaa",
                                    linestyle=":",
                                    linewidth=1.4,
                                    label="Unpatched response scored on new target",
                                )
                        if patches:
                            sample = patches[0]
                            coverage = sample["metrics"]["coverage"]["mean"]
                            nw = sample["worlds_observed"]
                            ne = sample["expected_worlds"]
                            label = f"Worlds {nw}/{ne if ne is not None else '?'}; donor coverage "
                            label += f"{coverage:.1%}" if coverage is not None else "undefined"
                            axis.text(
                                0.02,
                                0.05,
                                label,
                                transform=axis.transAxes,
                                fontsize=8,
                                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                            )
                        else:
                            axis.text(
                                0.5,
                                0.5,
                                "No completed analysis",
                                ha="center",
                                transform=axis.transAxes,
                            )
                        axis.set_title(
                            f"{arch}: {layers} blocks × {repeats} repeats | "
                            + ("OOD" if col else "ID")
                        )
                        axis.set_xticks(range(1, layers * repeats + 1))
                        axis.set_xlabel("Executed layer (1-based)")
                        _style(axis, title + " (%)")
                handles, labels = [], []
                for axis in axes.flat:
                    hs, ls = axis.get_legend_handles_labels()
                    for h, label in zip(hs, ls, strict=True):
                        if label not in labels:
                            handles.append(h)
                            labels.append(label)
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="outside lower center",
                        ncol=3,
                        frameon=False,
                        fontsize=9,
                    )
                fig.suptitle(
                    f"{phase} | {hops}-hop | checkpoint {step:,} | {id_split}\n"
                    "Means: seeds within world, then equal worlds; "
                    "bands: world range, not confidence intervals",
                    fontsize=10,
                )
                name = (
                    f"mechanism-{family}-{phase}-h{hops}-s{step}-{comparison[:6]}"
                    f"-d{donor_seed}-{version[:6]}-{id_split}"
                )
                for suffix in ("png", "pdf"):
                    destination = output / f"{name}.{suffix}"
                    fig.savefig(destination, dpi=180)
                    figures.append(str(destination))
                plt.close(fig)
    return figures


def plot_recurrence(groups, output):
    plt = _matplotlib()
    facets = defaultdict(list)
    for row in groups:
        if row["kind"] == "recurrence":
            facets[row["condition_id"], row["step"], row["analysis_version"]].append(row)
    figures = []
    for (condition_id, step, version), members in sorted(facets.items()):
        first = members[0]
        splits = [
            s
            for s in (
                "atomic",
                "id_atomic",
                "ood_atomic",
                "test_composite",
                "test_full_composite",
                "ood_composite",
            )
            if any(r["split"] == s for r in members)
        ]
        fig, axes = plt.subplots(
            2, len(splits), figsize=(4 * len(splits), 6.5), squeeze=False, constrained_layout=True
        )
        for col, split in enumerate(splits):
            rows = sorted(
                (r for r in members if r["split"] == split), key=lambda r: r["evaluated_repeats"]
            )
            for index, (metric, label) in enumerate(
                (("answer_accuracy", "Answer accuracy"), ("accuracy", "Answer + EOS accuracy"))
            ):
                axis = axes[index, col]
                points = [{**row, "execution_layer": row["evaluated_repeats"]} for row in rows]
                _curve(axis, points, metric, "#2878B5", split)
                axis.axvline(
                    first["training_repeats"],
                    linestyle="--",
                    color="#D77A20",
                    label="Training repeats",
                )
                axis.set_xticks([r["evaluated_repeats"] for r in rows])
                axis.set_xlabel("Evaluation repeats at fixed weights")
                axis.set_title(split)
                _style(axis, label + " (%)")
                counts = [row["worlds_observed"] for row in rows]
                nw = (
                    str(min(counts))
                    if min(counts) == max(counts)
                    else f"{min(counts)}–{max(counts)}"
                )
                ne = first["expected_worlds"]
                axis.text(
                    0.02,
                    0.05,
                    f"Worlds {nw}/{ne if ne is not None else '?'}",
                    transform=axis.transAxes,
                    fontsize=8,
                )
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            f"{first['phase']} | {first['hops']}-hop | {first['architecture']} "
            f"| checkpoint {step:,}\n"
            "All evaluated repeats; training repeats remain the registered endpoint; "
            "bands show world range",
            fontsize=11,
        )
        name = (
            f"recurrence-{first['phase']}-h{first['hops']}-{first['architecture']}"
            f"-s{step}-{condition_id[:6]}-{version[:6]}"
        )
        for suffix in ("png", "pdf"):
            destination = output / f"{name}.{suffix}"
            fig.savefig(destination, dpi=180)
            figures.append(str(destination))
        plt.close(fig)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            ROOT / "results/grok-loop-v1/mechanism-development",
            ROOT / "results/grok-loop-v1/mechanism-confirmation",
        ],
    )
    parser.add_argument(
        "--recurrence", type=Path, nargs="+", help="Scan summary files or parent directories"
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Registration for completeness; repeatable",
    )
    parser.add_argument("--expected-splits", nargs="+", default=["test_composite", "ood_composite"])
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/development-artifacts/grok-loop-v1/mechanism-report",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    recurrence = (
        args.recurrence
        if args.recurrence is not None
        else list((ROOT / "results/grok-loop-v1").glob("recurrence-*/summary.json"))
    )
    records, manifests, pending = discover(args.inputs, recurrence)
    records, duplicates = deduplicate_records(records)
    expected = expected_runs(args.config)
    world_rows, grouped = aggregate(records, expected)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("model-results", records),
        ("world-results", world_rows),
        ("group-results", grouped),
    ):
        write_csv(args.out / f"{name}.csv", rows)
    write_csv(
        args.out / "coverage-and-prerequisites.csv",
        [
            r
            for r in grouped
            if r["kind"] in ("donor_coverage", "atomic_precondition")
            or r["condition"].endswith("counterfactual_input")
        ],
    )
    figures = (
        []
        if args.no_plots
        else plot_mechanisms(grouped, args.out, expected) + plot_recurrence(grouped, args.out)
    )
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "statistical_unit": (
            "Equal world means after averaging available initializations within each world."
        ),
        "bands": (
            "Observed world range, not confidence intervals; "
            "a single world has no between-world spread."
        ),
        "missing_denominators": (
            "No donor means undefined success; coverage and missing counts remain separate."
        ),
        "selection": (
            "Every component, executed layer and measured repeat count is reported; "
            "no best-test selection."
        ),
        "completeness": completeness(expected, manifests, args.expected_splits),
        "incomplete_analysis_artifacts": pending,
        "identical_duplicate_records_counted_once": duplicates,
        "records": len(records),
        "world_records": len(world_rows),
        "groups": grouped,
        "runs": manifests,
        "figures": figures,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "figures": len(figures),
                "output": str(args.out),
                "all_registered_endpoints_complete": summary["completeness"][
                    "all_registered_endpoints_complete"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
