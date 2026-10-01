#!/usr/bin/env python3
"""Report complete loop comparisons; initialization replicates nest within worlds."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grok_multihop_data import path_details

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"c1": "#999999", "c2": "#4477aa", "cd": "#228833", "l1": "#ee6677", "l2": "#aa3377"}
METRICS = (
    "atomic",
    "id_atomic",
    "ood_atomic",
    "train_composite",
    "test_full_composite",
    "ood_composite",
    "autonomous_calls",
)
PAIRS = (
    ("cd", "l2", "equal_executed_depth"),
    ("cd", "l1", "equal_executed_depth"),
    ("c2", "l2", "equal_parameters"),
    ("c1", "l1", "equal_parameters"),
)


def condition_key(spec, excluded=()):
    return json.dumps({k: v for k, v in spec.items() if k not in set(excluded)}, sort_keys=True)


def mean_across_worlds(rows, metric):
    """Average initialization repeats within each world, then weight worlds equally."""
    worlds = sorted({r["world_seed"] for r in rows})
    means = []
    for world in worlds:
        values = [r[metric] for r in rows if r["world_seed"] == world and r[metric] is not None]
        if values:
            means.append(float(np.mean(values)))
    return {
        "mean": float(np.mean(means)) if means else None,
        "min": min(means) if means else None,
        "max": max(means) if means else None,
        "worlds_with_denominator": len(means),
    }


def group_world_means(endpoints):
    groups = defaultdict(list)
    for row in endpoints:
        groups[(row["config"], row["condition_id"])].append(row)
    summary = []
    for (config, condition_id), rows in groups.items():
        item = {
            "config": config,
            "condition_id": condition_id,
            **{
                key: rows[0][key]
                for key in (
                    "phase",
                    "hops",
                    "architecture",
                    "init_scheme",
                    "parameters",
                    "effective_depth",
                )
            },
            "worlds": len({r["world_seed"] for r in rows}),
            "runs": len(rows),
        }
        for metric in METRICS:
            means = mean_across_worlds(rows, metric)
            item[metric] = means["mean"]
            item[metric + "_world_min"] = means["min"]
            item[metric + "_world_max"] = means["max"]
            item[metric + "_worlds_with_denominator"] = means["worlds_with_denominator"]
        summary.append(item)
    return summary


def first_observed_threshold(history, threshold):
    """First saved ID-probe checkpoint meeting the threshold; no persistence rule."""
    for row in sorted(history, key=lambda item: item["step"]):
        accuracy = row["test_composite"]["accuracy"]
        if accuracy is not None and accuracy >= threshold:
            return {
                "step": row["step"],
                "training_flops": row["estimated_training_flops"],
                "observed_accuracy": accuracy,
            }
    return {"step": None, "training_flops": None, "observed_accuracy": None}


def matched_architecture_pairs(runs):
    """Keep depth variants distinct, matching each pair by its intended constraint."""
    indexed = defaultdict(dict)
    excluded = {"architecture", "layers", "repeats"}
    for run in runs:
        spec = run["spec"]
        key = (run["config"], condition_key(spec, excluded), run["dataset_sha256"])
        identity = (spec["architecture"], spec["layers"], spec["repeats"])
        if identity in indexed[key]:
            raise ValueError("Duplicate run for paired condition and architecture depth")
        if run["endpoint"]["effective_depth"] != spec["layers"] * spec["repeats"]:
            raise ValueError("Endpoint effective depth differs from registered architecture")
        indexed[key][identity] = run
    for (config, matching_key, dataset), group in indexed.items():
        for left_arch, right_arch, comparison in PAIRS:
            left_runs = [run for (arch, _, _), run in group.items() if arch == left_arch]
            right_runs = [run for (arch, _, _), run in group.items() if arch == right_arch]
            for left in left_runs:
                for right in right_runs:
                    ls, rs = left["spec"], right["spec"]
                    left_depth, right_depth = (
                        ls["layers"] * ls["repeats"],
                        rs["layers"] * rs["repeats"],
                    )
                    if comparison == "equal_executed_depth" and left_depth != right_depth:
                        continue
                    if comparison == "equal_parameters" and ls["layers"] != rs["layers"]:
                        continue
                    condition = {
                        **json.loads(matching_key),
                        "comparison_depth": right_depth,
                        "left_unique_layers": ls["layers"],
                        "left_repeats": ls["repeats"],
                        "right_unique_layers": rs["layers"],
                        "right_repeats": rs["repeats"],
                    }
                    yield config, condition, dataset, left_arch, right_arch, comparison, left, right


def paired_comparisons(runs):
    """Use paired worlds/initializations and budget-only recorded-node selection."""
    pairs = []
    for (
        config,
        condition,
        dataset,
        left_arch,
        right_arch,
        comparison,
        left,
        right,
    ) in matched_architecture_pairs(runs):
        ends = [run["endpoint"] for run in (left, right)]
        if (
            comparison == "equal_executed_depth"
            and ends[0]["effective_depth"] != ends[1]["effective_depth"]
        ):
            raise ValueError("Registered equal-depth pair has unequal executed depth")
        if comparison == "equal_parameters" and ends[0]["parameters"] != ends[1]["parameters"]:
            raise ValueError("Registered equal-parameter pair has unequal parameter counts")
        cap = min(end["estimated_training_flops"] for end in ends)
        selected = []
        for run in (left, right):
            eligible = [row for row in run["history"] if row["estimated_training_flops"] <= cap]
            if not eligible:
                raise ValueError("No recorded node within common compute budget")
            selected.append(max(eligible, key=lambda row: row["step"]))

        def difference(rows, metric):
            values = [row[metric]["accuracy"] for row in rows]
            return values[1] - values[0] if all(value is not None for value in values) else None

        pair = {
            "config": config,
            "comparison": comparison,
            "pair": right_arch + "_minus_" + left_arch,
            "world_seed": left["spec"]["world_seed"],
            "initialization": left["spec"]["initialization"],
            "hops": left["spec"]["hops"],
            "phase": left["spec"]["phase"],
            "init_scheme": left["spec"]["init_scheme"],
            "dataset_sha256": dataset,
            "condition": condition,
            "comparison_depth": condition["comparison_depth"],
            "difference_direction": "right loop architecture minus left classic architecture",
            "common_training_flop_cap": cap,
            "endpoint_id_full_difference": difference(ends, "test_full_composite"),
            "endpoint_ood_difference": difference(ends, "ood_composite"),
            "common_budget_id_probe_difference": difference(selected, "test_composite"),
            "common_budget_ood_difference": difference(selected, "ood_composite"),
            "common_budget_id_measurement": "fixed ID probe; endpoint ID uses full reserve",
        }
        for side, arch, run, end, node in zip(
            ("left", "right"),
            (left_arch, right_arch),
            (left, right),
            ends,
            selected,
            strict=True,
        ):
            pair[side] = {
                "run": run["run"],
                "architecture": arch,
                "unique_layers": run["spec"]["layers"],
                "repeats": run["spec"]["repeats"],
                "effective_depth": end["effective_depth"],
                "endpoint_step": end["step"],
                "endpoint_flops": end["estimated_training_flops"],
                "endpoint_id_full_accuracy": end["test_full_composite"]["accuracy"],
                "endpoint_id_full_n": end["test_full_composite"]["n"],
                "endpoint_ood_accuracy": end["ood_composite"]["accuracy"],
                "endpoint_ood_n": end["ood_composite"]["n"],
                "common_budget_step": node["step"],
                "common_budget_flops": node["estimated_training_flops"],
                "unused_common_flops": cap - node["estimated_training_flops"],
                "common_budget_id_probe_accuracy": node["test_composite"]["accuracy"],
                "common_budget_id_probe_n": node["test_composite"]["n"],
                "common_budget_ood_accuracy": node["ood_composite"]["accuracy"],
                "common_budget_ood_n": node["ood_composite"]["n"],
            }
        pair["actual_common_budget_flop_difference"] = (
            selected[1]["estimated_training_flops"] - selected[0]["estimated_training_flops"]
        )
        pairs.append(pair)
    groups = defaultdict(list)
    for pair in pairs:
        key = (
            pair["config"],
            pair["pair"],
            condition_key(pair["condition"], {"world_seed", "initialization", "stream_seed"}),
        )
        groups[key].append(pair)
    summary = []
    for (config, name, condition), rows in groups.items():
        item = {
            "config": config,
            "pair": name,
            "condition": json.loads(condition),
            "worlds": len({row["world_seed"] for row in rows}),
            "paired_runs": len(rows),
        }
        for metric in (
            "endpoint_id_full_difference",
            "endpoint_ood_difference",
            "common_budget_id_probe_difference",
            "common_budget_ood_difference",
        ):
            item[metric] = mean_across_worlds(rows, metric)
        summary.append(item)
    return {"pairs": pairs, "world_summary": summary}


def save_csv(path, records):
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def conditional_metrics(world, pred, split, spec):
    rows = world[split]
    prefixes = (
        "all_atoms_correct",
        "multiple_targets_per_relation_sequence",
        "all_atoms_correct_and_multiple_targets",
    )
    if not len(rows):
        return {
            key: value
            for prefix in prefixes
            for key, value in (
                (prefix + "_n", 0),
                (prefix + "_coverage", None),
                (prefix + "_accuracy", None),
            )
        }
    if not np.array_equal(pred["atomic_target"], world["atomic"][:, -1]):
        raise ValueError("Atomic predictions do not align with world rows")
    if not np.array_equal(pred[split + "_target"], rows[:, -1]):
        raise ValueError("Composite predictions do not align with world rows")
    _, indices = path_details(rows, world["atomic"], spec["entities"], spec["relations"])
    atoms_correct = (pred["atomic_answer"] == world["atomic"][:, -1]) & (pred["atomic_stop"] == 1)
    mask = atoms_correct[indices].all(1)
    correct = (pred[split + "_answer"] == rows[:, -1]) & (pred[split + "_stop"] == 1)
    suffix_targets = defaultdict(set)
    for row in rows:
        suffix_targets[tuple(row[1:-1])].add(int(row[-1]))
    ambiguous = np.array([len(suffix_targets[tuple(row[1:-1])]) > 1 for row in rows])
    result = {}
    for prefix, selected in zip(prefixes, (mask, ambiguous, mask & ambiguous), strict=True):
        result.update(
            {
                prefix + "_n": int(selected.sum()),
                prefix + "_coverage": float(selected.mean()),
                prefix + "_accuracy": float(correct[selected].mean()) if selected.any() else None,
            }
        )
    return result


def learning_curve_styles(rows):
    """Disambiguate repeated architecture labels without changing plotted values."""
    conditions = {row["condition_id"]: row for row in rows}
    by_arch = defaultdict(list)
    for condition, row in conditions.items():
        by_arch[row["architecture"]].append((condition, row))
    styles = {}
    line_styles = ("-", "--", ":", "-.", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
    for arch, variants in by_arch.items():
        if len(variants) == 1:
            styles[variants[0][0]] = (arch.upper(), "-")
            continue
        variants.sort(
            key=lambda item: (
                item[1]["effective_depth"],
                item[1]["unique_layers"],
                item[1]["repeats"],
                item[1]["init_scheme"],
                item[0],
            )
        )
        for index, (condition, row) in enumerate(variants):
            label = (
                f"{arch.upper()} / D{row['effective_depth']} "
                f"(b×R={row['unique_layers']}×{row['repeats']}) / {row['init_scheme']}"
            )
            styles[condition] = (label, line_styles[index % len(line_styles)])
    return styles


def save_learning_plots(learning, out):
    paths = []
    for config in sorted({row["config"] for row in learning}):
        selected = [row for row in learning if row["config"] == config]
        for x_key, suffix, scale, xlabel in (
            ("step", "steps", 1000, "Training steps (thousands)"),
            ("training_flops", "flops", 1e12, "Estimated training matmul FLOPs (trillions)"),
        ):
            fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
            for col, hops in enumerate((2, 3, 4)):
                conditions = sorted(
                    {
                        (row["architecture"], row["condition_id"])
                        for row in selected
                        if row["hops"] == hops
                    }
                )
                styles = learning_curve_styles([row for row in selected if row["hops"] == hops])
                for arch, condition in conditions:
                    rows = [
                        row
                        for row in selected
                        if row["hops"] == hops and row["condition_id"] == condition
                    ]
                    coordinates = sorted({row[x_key] for row in rows})
                    label, linestyle = styles[condition]
                    for j, metric in enumerate(("id", "ood")):
                        stats = [
                            mean_across_worlds([row for row in rows if row[x_key] == x], metric)
                            for x in coordinates
                        ]
                        points, low, high = (
                            np.array([item[k] if item[k] is not None else np.nan for item in stats])
                            * 100
                            for k in ("mean", "min", "max")
                        )
                        x = np.array(coordinates) / scale
                        axes[j, col].plot(
                            x, points, color=COLORS[arch], label=label, linestyle=linestyle
                        )
                        axes[j, col].fill_between(x, low, high, color=COLORS[arch], alpha=0.12)
                for j, label in enumerate(("ID fixed probe", "OOD full pool")):
                    axes[j, col].set_title(f"{hops} hops | {label}")
                    axes[j, col].grid(alpha=0.2)
                    axes[j, col].set_ylim(-2, 102)
                    axes[j, col].set_xlabel(xlabel)
            axes[0, 0].set_ylabel("Answer + EOS accuracy (%)")
            axes[1, 0].set_ylabel("Answer + EOS accuracy (%)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
            fig.suptitle(
                "Equal world weights; initialization means within world; band = world min/max"
            )
            fig.tight_layout(rect=(0, 0.12 if handles else 0, 1, 0.96))
            path = out / (Path(config).stem + "-learning-" + suffix + ".png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    endpoints, learning, absent = [], [], []
    conditions, paired_runs = [], []
    for config in args.configs:
        cfg = json.loads((ROOT / config).read_text())
        for run_id, overrides in cfg["runs"].items():
            spec = {**cfg["base"], **overrides}
            directory = ROOT / cfg["output_root"] / spec["phase"] / run_id
            if not (directory / "complete.json").exists():
                absent.append(run_id)
                continue
            completed = json.loads((directory / "complete.json").read_text())
            if completed["spec"] != spec:
                raise ValueError(f"Completed run specification differs from config: {run_id}")
            end = completed["endpoint"]
            if end["step"] != spec["steps"]:
                raise ValueError(f"Incomplete run marked complete: {run_id}")
            condition_id = hashlib.sha256(
                condition_key(spec, {"world_seed", "initialization", "stream_seed"}).encode()
            ).hexdigest()[:16]
            identity = {
                "run": run_id,
                "condition_id": condition_id,
                "config": config,
                "phase": spec["phase"],
                "world_seed": spec["world_seed"],
                "initialization": spec["initialization"],
                "init_scheme": spec["init_scheme"],
                "hops": spec["hops"],
                "architecture": spec["architecture"],
                "unique_layers": spec["layers"],
                "repeats": spec["repeats"],
                "effective_depth": spec["layers"] * spec["repeats"],
            }
            record = {
                **identity,
                "steps": spec["steps"],
                "parameters": end["parameters"],
                "training_flops": end["estimated_training_flops"],
                "training_seconds": completed["training_seconds"],
                "evaluation_seconds": completed["evaluation_seconds"],
                "atomic_exposures": end["counts"]["atomic"],
                "composite_exposures": end["counts"]["composite"],
            }
            for name in (
                "atomic",
                "id_atomic",
                "ood_atomic",
                "train_composite",
                "test_composite",
                "test_full_composite",
                "ood_composite",
                "autonomous_calls",
            ):
                record[name] = end[name]["accuracy"]
                record[name + "_n"] = end[name]["n"]
            history = json.loads((directory / "learning.json").read_text())
            if [row["step"] for row in history] != spec["nodes"] or history[-1] != end:
                raise ValueError(f"Learning nodes or endpoint differ from registration: {run_id}")
            for threshold in (0.90, 0.95):
                hit = first_observed_threshold(history, threshold)
                prefix = f"T{int(threshold * 100)}_probe_first_observed"
                record[prefix + "_step"] = hit["step"]
                record[prefix + "_flops"] = hit["training_flops"]
                record[prefix + "_accuracy"] = hit["observed_accuracy"]
            endpoints.append(record)
            for row in history:
                learning.append(
                    {
                        **identity,
                        "step": row["step"],
                        "training_flops": row["estimated_training_flops"],
                        "atomic": row["atomic"]["accuracy"],
                        "id": row["test_composite"]["accuracy"],
                        "ood": row["ood_composite"]["accuracy"],
                    }
                )
            world = dict(np.load(directory / "world.npz"))
            world_meta = json.loads((directory / "world-metadata.json").read_text())
            paired_runs.append(
                {
                    "run": run_id,
                    "config": config,
                    "spec": spec,
                    "endpoint": end,
                    "history": history,
                    "dataset_sha256": world_meta["dataset_sha256"],
                }
            )
            pred = dict(np.load(directory / f"predictions-{spec['steps']:07d}.npz"))
            conditions.append(
                {
                    **identity,
                    **{
                        split: conditional_metrics(world, pred, split, spec)
                        for split in ("test_full_composite", "ood_composite")
                    },
                }
            )
    summary = group_world_means(endpoints)
    paired = paired_comparisons(paired_runs)
    save_csv(
        out / "paired-comparisons.csv",
        [
            {
                **{k: v for k, v in pair.items() if not isinstance(v, dict)},
                **{side + "_" + k: v for side in ("left", "right") for k, v in pair[side].items()},
            }
            for pair in paired["pairs"]
        ],
    )
    save_csv(out / "endpoints.csv", endpoints)
    save_csv(out / "learning.csv", learning)
    save_csv(out / "world-summary.csv", summary)
    write_json(
        out / "summary.json",
        {
            "generated_utc": utc(),
            "missing": absent,
            "complete_runs": len(endpoints),
            "groups": summary,
            "endpoints": endpoints,
            "conditional_subsets": conditions,
            "conditional_interpretation": (
                "model-specific known-atom subsets; report coverage; multi-target relation "
                "groups are defined descriptively within each evaluation split"
            ),
            "paired_comparisons": paired,
            "threshold_cost_definition": (
                "first observed checkpoint whose fixed ID probe answer+EOS accuracy reaches "
                "90% or 95%; no interpolation or consecutive-node requirement; null if not reached"
            ),
            "statistical_unit": "world; initialization seeds averaged within each world",
        },
    )
    save_learning_plots(learning, out)
    print(json.dumps({"complete": len(endpoints), "missing": len(absent), "summary": summary}))


if __name__ == "__main__":
    main()
