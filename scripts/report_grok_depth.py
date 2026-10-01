#!/usr/bin/env python3
"""Summarize completed or running small-model composition experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("atomic", "train_composite", "test_composite", "ood_composite", "two_calls")
NON_CONDITION = {
    "phase",
    "world_seed",
    "initialization",
    "stream_seed",
    "nodes",
    "weight_nodes",
    "run_id",
}
ARCHITECTURE = {"layers", "width", "heads"}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def stable_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def mean(values):
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def exposure(row, counts):
    atomic_n = counts.get("atomic", 0)
    composite_n = counts.get("train_composite", 0)
    return {
        "atomic_exposures": row.get("counts", {}).get("atomic", 0) / atomic_n if atomic_n else None,
        "composite_exposures": row.get("counts", {}).get("composite", 0) / composite_n
        if composite_n
        else None,
        "dataset_passes": row.get("examples", 0) / (atomic_n + composite_n)
        if atomic_n + composite_n
        else None,
    }


def t90(rows, counts, budget):
    """First recorded node in the first three-node run with exact accuracy >= .9."""
    for index in range(max(0, len(rows) - 2)):
        triple = rows[index : index + 3]
        values = [row.get("test_composite", {}).get("accuracy") for row in triple]
        if all(value is not None and value >= 0.9 for value in values):
            first, confirmed = triple[0], triple[-1]
            return {
                "reached": True,
                "step": first["step"],
                "confirmed_at_step": confirmed["step"],
                "previous_evaluation_step": rows[index - 1]["step"] if index else None,
                "examples": first.get("examples"),
                "estimated_training_flops": first.get("estimated_training_flops"),
                **exposure(first, counts),
                "last_observed_step": rows[-1]["step"],
                "budget_steps": budget,
            }
    return {
        "reached": False,
        "step": None,
        "confirmed_at_step": None,
        "previous_evaluation_step": None,
        "examples": None,
        "estimated_training_flops": None,
        "atomic_exposures": None,
        "composite_exposures": None,
        "dataset_passes": None,
        "last_observed_step": rows[-1]["step"] if rows else None,
        "budget_steps": budget,
    }


def atomic_split_endpoint(directory, row):
    """Score the pre-existing atomic split from saved predictions, without inference."""
    if row is None:
        return {"available": False, "reason": "No evaluation has been recorded."}
    predictions_path = directory / f"predictions-{row['step']:07d}.npz"
    world_path = directory / "world.npz"
    if not predictions_path.exists() or not world_path.exists():
        return {"available": False, "step": row["step"], "reason": "Saved arrays unavailable."}
    with np.load(world_path) as world, np.load(predictions_path) as predictions:
        atomic = world["atomic"]
        answer, stop = predictions["atomic_answer"], predictions["atomic_stop"]
        if (
            len(answer) != len(atomic)
            or len(stop) != len(atomic)
            or not np.array_equal(predictions["atomic_target"], atomic[:, -1])
        ):
            raise ValueError(f"Atomic predictions do not align with saved world: {directory}")
        result = {"available": True, "step": row["step"], "source": str(predictions_path)}
        all_rows = [tuple(values) for values in atomic.tolist()]
        for name in ("id_atomic", "ood_atomic"):
            members = {tuple(values) for values in world[name].tolist()}
            selected = np.asarray([values in members for values in all_rows], dtype=bool)
            if int(selected.sum()) != len(members):
                raise ValueError(f"Atomic split does not align with saved world: {directory}")
            n = int(selected.sum())
            correct = answer[selected] == atomic[selected, -1]
            result[name] = {
                "n": n,
                "answer_accuracy": float(correct.mean()) if n else None,
                "accuracy": float((correct & (stop[selected] == 1)).mean()) if n else None,
            }
    return result


def load_runs(results):
    runs = []
    for phase in ("development", "confirmation"):
        for meta_path in sorted((results / phase).glob("*/metadata.json")):
            directory = meta_path.parent
            metadata = read_json(meta_path)
            spec = metadata["spec"]
            raw_rows = read_json(directory / "learning.json", [])
            by_step = {row["step"]: row for row in raw_rows}
            rows = [by_step[step] for step in sorted(by_step)]
            completion = read_json(directory / "complete.json")
            audit = read_json(directory / "data-audit.json", {})
            world_meta = read_json(directory / "world-metadata.json", {})
            counts = audit.get("counts", world_meta.get("counts", {}))
            warnings = []
            if len(rows) != len(raw_rows):
                warnings.append("Duplicate evaluation steps; last record at each step retained.")
            budget = spec["steps"]
            complete = bool(
                completion
                and completion.get("endpoint", {}).get("step") == budget
                and budget in by_step
            )
            if completion and not complete:
                warnings.append("Completion marker does not match budget and learning log.")
            endpoint = by_step.get(budget) if complete else None
            last = rows[-1] if rows else None
            condition = {k: v for k, v in spec.items() if k not in NON_CONDITION}
            runs.append(
                {
                    "run_id": directory.name,
                    "phase": phase,
                    "status": "complete" if complete else "incomplete",
                    "directory": str(directory),
                    "spec": spec,
                    "condition": condition,
                    "condition_id": stable_key(condition),
                    "data_counts": counts,
                    "dataset_sha256": audit.get("dataset_sha256"),
                    "budget_steps": budget,
                    "last_observed_step": last["step"] if last else None,
                    "last_observation": last,
                    "fixed_budget_endpoint": endpoint,
                    "atomic_split_endpoint": atomic_split_endpoint(directory, endpoint or last),
                    "t90": t90(rows, counts, budget),
                    "warnings": warnings,
                    "learning": rows,
                }
            )
    return runs


def group_world_means(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["phase"], run["condition_id"]].append(run)
    result = []
    for (phase, condition_id), members in sorted(grouped.items()):
        worlds = defaultdict(list)
        for run in members:
            if run["status"] == "complete":
                worlds[run["spec"]["world_seed"]].append(run)
        per_world = []
        for world_seed, replicates in sorted(worlds.items()):
            metrics = {
                split: mean(
                    [
                        run["fixed_budget_endpoint"].get(split, {}).get("accuracy")
                        for run in replicates
                    ]
                )
                for split in SPLITS
            }
            per_world.append(
                {
                    "world_seed": world_seed,
                    "completed_runs": len(replicates),
                    "run_ids": [run["run_id"] for run in replicates],
                    "fixed_budget_accuracy": metrics,
                }
            )
        result.append(
            {
                "phase": phase,
                "condition_id": condition_id,
                "condition": members[0]["condition"],
                "run_count": len(members),
                "completed_run_count": sum(run["status"] == "complete" for run in members),
                "independent_completed_world_count": len(per_world),
                "worlds": per_world,
                "world_mean_fixed_budget_accuracy": {
                    split: mean([world["fixed_budget_accuracy"][split] for world in per_world])
                    for split in SPLITS
                },
                "t90_reached_runs": sum(run["t90"]["reached"] for run in members),
                "t90_not_reached_runs": sum(not run["t90"]["reached"] for run in members),
            }
        )
    return result


def matched_completed_pairs(runs):
    """Only match identical worlds, seed IDs, streams and non-architecture settings."""
    matched = defaultdict(list)
    for run in runs:
        if run["status"] != "complete":
            continue
        controls = {
            k: v
            for k, v in run["spec"].items()
            if k not in ARCHITECTURE | {"phase", "nodes", "weight_nodes", "run_id"}
        }
        matched[run["phase"], stable_key(controls)].append(run)
    for members in matched.values():
        ordered = sorted(
            members,
            key=lambda r: (r["spec"]["layers"], r["spec"]["width"], r["condition_id"], r["run_id"]),
        )
        for left, right in itertools.combinations(ordered, 2):
            if left["condition_id"] == right["condition_id"]:
                continue
            # A source/configuration drift must not silently pair different datasets.
            if left["dataset_sha256"] != right["dataset_sha256"]:
                continue
            yield left, right


def paired_differences(runs):
    differences = defaultdict(list)
    for left, right in matched_completed_pairs(runs):
        key = left["phase"], left["condition_id"], right["condition_id"]
        diff = {}
        for split in SPLITS:
            a = left["fixed_budget_endpoint"].get(split, {}).get("accuracy")
            b = right["fixed_budget_endpoint"].get(split, {}).get("accuracy")
            diff[split] = b - a if a is not None and b is not None else None
        differences[key].append(
            {
                "world_seed": left["spec"]["world_seed"],
                "left_run": left["run_id"],
                "right_run": right["run_id"],
                "accuracy_difference_right_minus_left": diff,
            }
        )
    output = []
    for (phase, left_id, right_id), pairs in sorted(differences.items()):
        worlds = defaultdict(list)
        for pair in pairs:
            worlds[pair["world_seed"]].append(pair)
        per_world = [
            {
                "world_seed": seed,
                "paired_runs": len(items),
                "accuracy_difference_right_minus_left": {
                    split: mean(
                        [item["accuracy_difference_right_minus_left"][split] for item in items]
                    )
                    for split in SPLITS
                },
            }
            for seed, items in sorted(worlds.items())
        ]
        output.append(
            {
                "phase": phase,
                "left_condition_id": left_id,
                "right_condition_id": right_id,
                "independent_world_count": len(per_world),
                "pairs": pairs,
                "worlds": per_world,
                "world_mean_accuracy_difference_right_minus_left": {
                    split: mean(
                        [item["accuracy_difference_right_minus_left"][split] for item in per_world]
                    )
                    for split in SPLITS
                },
                "interpretation": "Descriptive world-paired differences; no significance claim.",
            }
        )
    return output


def paired_equal_compute(runs):
    """Use the final registered node within a common FLOP cap, without interpolation."""
    groups = defaultdict(list)
    for left, right in matched_completed_pairs(runs):
        cap = min(run["fixed_budget_endpoint"]["estimated_training_flops"] for run in (left, right))
        arms = []
        for run in (left, right):
            eligible = [
                row
                for row in run["learning"]
                if row["estimated_training_flops"] <= cap and row["step"] <= run["budget_steps"]
            ]
            if not eligible:
                raise ValueError(
                    f"No registered evaluation at or below the FLOP cap: {run['run_id']}"
                )
            selected = max(eligible, key=lambda row: row["step"])
            actual_flops = selected["estimated_training_flops"]
            endpoint_flops = run["fixed_budget_endpoint"]["estimated_training_flops"]
            arms.append(
                {
                    "run_id": run["run_id"],
                    "step": selected["step"],
                    "budget_steps": run["budget_steps"],
                    "examples": selected.get("examples"),
                    **exposure(selected, run["data_counts"]),
                    "estimated_training_flops": actual_flops,
                    "full_budget_training_flops": endpoint_flops,
                    "unused_common_cap_flops": cap - actual_flops,
                    "unrepresented_full_budget_flops": endpoint_flops - actual_flops,
                    "accuracy": {
                        split: selected.get(split, {}).get("accuracy") for split in SPLITS
                    },
                }
            )
        a, b = arms
        diff = {
            split: b["accuracy"][split] - a["accuracy"][split]
            if b["accuracy"][split] is not None and a["accuracy"][split] is not None
            else None
            for split in SPLITS
        }
        groups[left["phase"], left["condition_id"], right["condition_id"]].append(
            {
                "world_seed": left["spec"]["world_seed"],
                "initialization": left["spec"]["initialization"],
                "stream_seed": left["spec"]["stream_seed"],
                "common_flop_cap": cap,
                "left": a,
                "right": b,
                "actual_flop_difference_right_minus_left": (
                    b["estimated_training_flops"] - a["estimated_training_flops"]
                ),
                "accuracy_difference_right_minus_left": diff,
            }
        )
    output = []
    for (phase, left_id, right_id), pairs in sorted(groups.items()):
        worlds = defaultdict(list)
        for pair in pairs:
            worlds[pair["world_seed"]].append(pair)
        per_world = [
            {
                "world_seed": seed,
                "paired_runs": len(items),
                "accuracy_difference_right_minus_left": {
                    split: mean(
                        [item["accuracy_difference_right_minus_left"][split] for item in items]
                    )
                    for split in SPLITS
                },
            }
            for seed, items in sorted(worlds.items())
        ]
        output.append(
            {
                "phase": phase,
                "left_condition_id": left_id,
                "right_condition_id": right_id,
                "independent_world_count": len(per_world),
                "pairs": pairs,
                "worlds": per_world,
                "world_mean_accuracy_difference_right_minus_left": {
                    split: mean(
                        [
                            world["accuracy_difference_right_minus_left"][split]
                            for world in per_world
                        ]
                    )
                    for split in SPLITS
                },
                "interpretation": "Common compute caps with discrete evaluation nodes. Actual "
                "compute can differ by the reported slack. Descriptive world-paired "
                "differences; no interpolation or significance claim.",
            }
        )
    return output


def save_csv(runs, path):
    fields = [
        "phase",
        "run_id",
        "status",
        "world_seed",
        "layers",
        "width",
        "initialization",
        "step",
        "budget_steps",
        "examples",
        "atomic_exposures",
        "composite_exposures",
        "dataset_passes",
        "estimated_training_flops",
        "parameters",
        "training_seconds",
        "evaluation_seconds",
        "effective_input_tokens",
        "supervised_tokens",
        "last_batch_loss",
    ] + [
        f"{split}_{metric}"
        for split in SPLITS
        for metric in ("n", "accuracy", "answer_accuracy", "nll")
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for point in run["learning"]:
                row = {field: point.get(field) for field in fields}
                row.update({key: run[key] for key in ("phase", "run_id", "status", "budget_steps")})
                row.update(
                    {
                        key: run["spec"][key]
                        for key in ("world_seed", "layers", "width", "initialization")
                    }
                )
                row.update(exposure(point, run["data_counts"]))
                for split in SPLITS:
                    row.update(
                        {
                            f"{split}_{metric}": point.get(split, {}).get(metric)
                            for metric in ("n", "accuracy", "answer_accuracy", "nll")
                        }
                    )
                writer.writerow(row)


def save_plots(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {
        "atomic": "All atomic facts",
        "train_composite": "Training compositions",
        "test_composite": "Held-out ID compositions",
        "ood_composite": "OOD compositions",
    }
    paths = []
    for phase in ("development", "confirmation"):
        members = [run for run in runs if run["phase"] == phase and run["learning"]]
        if not members:
            continue
        fig, axes = plt.subplots(4, 2, figsize=(13, 13), squeeze=False, sharey=True)
        palette_size = 10 if len(members) <= 10 else 20
        colors = plt.get_cmap("tab10" if palette_size == 10 else "tab20")
        for index, run in enumerate(members):
            color = colors(index % palette_size)
            label = f"{run['run_id']} ({run['status']})"
            rows = run["learning"]
            for column, xfield in enumerate(("atomic_exposures", "estimated_training_flops")):
                xs = [
                    exposure(row, run["data_counts"])[xfield] if column == 0 else row.get(xfield)
                    for row in rows
                ]
                for panel, split in enumerate(names):
                    ys = [row.get(split, {}).get("accuracy") for row in rows]
                    points = [
                        (x, y)
                        for x, y in zip(xs, ys, strict=True)
                        if x is not None and y is not None
                    ]
                    if not points:
                        continue
                    x, y = zip(*points, strict=True)
                    ax = axes[panel, column]
                    ax.plot(
                        x,
                        y,
                        label=label,
                        color=color,
                        linewidth=1.3,
                        linestyle="-" if run["status"] == "complete" else "--",
                    )
                    ax.plot(x[-1], y[-1], marker="o", color=color, markersize=3)
        for panel, (split, title) in enumerate(names.items()):
            for column in range(2):
                ax = axes[panel, column]
                ax.set_title(title)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xscale("symlog", linthresh=1 if column == 0 else 1e8)
                ax.set_xlim(left=0)
                ax.set_xlabel(
                    "Mean exposures per atomic fact"
                    if column == 0
                    else "Estimated training FLOPs (matrix operations)"
                )
                ax.set_ylabel("Exact accuracy (answer + EOS)")
                ax.grid(alpha=0.2)
                if split == "test_composite":
                    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=min(3, max(1, len(members))), fontsize=7
        )
        fig.suptitle(f"{phase.title()}: individual learning curves; incomplete runs are dashed")
        legend_height = min(0.30, 0.035 * ((len(members) + 2) // 3))
        fig.tight_layout(rect=(0, legend_height, 1, 0.975))
        for extension in ("png", "pdf"):
            path = output / f"learning-{phase}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    return paths


def save_confirmation_primary(runs, output):
    """Plot completed confirmation worlds; one world has one weight in the mean."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for run in runs:
        if run["phase"] == "confirmation" and run["status"] == "complete":
            groups[run["condition_id"]].append(run)
    if not groups:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for index, (_, members) in enumerate(
        sorted(
            groups.items(),
            key=lambda item: (item[1][0]["spec"]["layers"], item[1][0]["spec"]["width"]),
        )
    ):
        color = plt.get_cmap("tab10")(index % 10)
        by_world = defaultdict(list)
        for run in members:
            by_world[run["spec"]["world_seed"]].append(run)
        world_curves = []
        for replicates in by_world.values():
            by_step = defaultdict(list)
            for run in replicates:
                for row in run["learning"]:
                    accuracy = row.get("test_composite", {}).get("accuracy")
                    if accuracy is not None:
                        by_step[row["step"]].append(
                            {
                                "accuracy": accuracy,
                                "exposures": exposure(row, run["data_counts"])["atomic_exposures"],
                                "flops": row["estimated_training_flops"],
                            }
                        )
            world_curves.append(
                {
                    step: {
                        name: mean([point[name] for point in values])
                        for name in ("accuracy", "exposures", "flops")
                    }
                    for step, values in by_step.items()
                    if len(values) == len(replicates)
                }
            )
        common_steps = sorted(set.intersection(*(set(curve) for curve in world_curves)))
        spec = members[0]["spec"]
        layer_word = "layer" if spec["layers"] == 1 else "layers"
        label = f"{spec['layers']} {layer_word}, width {spec['width']} ({len(by_world)} worlds)"
        for column, xfield in enumerate(("exposures", "flops")):
            for curve in world_curves:
                steps = sorted(curve)
                axes[column].plot(
                    [curve[step][xfield] for step in steps],
                    [curve[step]["accuracy"] for step in steps],
                    color=color,
                    alpha=0.23,
                    linewidth=1,
                )
            axes[column].plot(
                [mean([curve[step][xfield] for curve in world_curves]) for step in common_steps],
                [
                    mean([curve[step]["accuracy"] for curve in world_curves])
                    for step in common_steps
                ],
                color=color,
                linewidth=2,
                label=label,
            )
    for column, ax in enumerate(axes):
        ax.set_xscale("symlog", linthresh=1 if column == 0 else 1e8)
        ax.set_xlim(left=0)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(
            "Mean exposures per atomic fact"
            if column == 0
            else "Estimated training FLOPs (matrix operations)"
        )
        ax.set_ylabel("Held-out ID exact accuracy (answer + EOS)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    world_count = len({run["spec"]["world_seed"] for members in groups.values() for run in members})
    fig.suptitle(f"Two-hop generalization across {world_count} independent worlds")
    fig.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = output / f"learning-confirmation-primary.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/grok-depth-v1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/development-artifacts/grok-depth-v1"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    save_csv(runs, args.output / "learning.csv")
    plots = save_plots(runs, args.output) + save_confirmation_primary(runs, args.output)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "results_directory": str(args.results),
        "run_count": len(runs),
        "completed_run_count": sum(run["status"] == "complete" for run in runs),
        "t90_definition": "First of three consecutive recorded ID exact-accuracy nodes >= 0.9; "
        "confirmation time is the third node. Null means not established within "
        "the observed budget; evaluation nodes do not identify an exact crossing.",
        "aggregation": "Fixed-budget results include complete runs only. Runs are averaged "
        "within each world before averaging independent worlds. Development and "
        "confirmation are separate. No p-values or significance claims.",
        "flops_definition": "Executed-shape matrix/attention FLOPs estimate from training logs; "
        "excludes elementwise operations and evaluation.",
        "runs": runs,
        "world_means": group_world_means(runs),
        "paired_differences": paired_differences(runs),
        "equal_compute_rule": "For each completed matched pair, use the smaller full-budget "
        "estimated training FLOPs as the common cap. For each arm select the last registered "
        "evaluation node at or below that cap; report actual compute and unused cap separately. "
        "No interpolation or accuracy-based node selection.",
        "paired_equal_compute": paired_equal_compute(runs),
        "plots": plots,
    }
    target = args.output / "summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(target)
    print(
        json.dumps(
            {
                "runs": len(runs),
                "complete": summary["completed_run_count"],
                "summary": str(target),
                "plots": plots,
            }
        )
    )


if __name__ == "__main__":
    main()
