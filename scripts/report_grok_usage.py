#!/usr/bin/env python3
"""Independently rescore saved fact-usage predictions and paired exposure arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("A", "B", "repA", "repB")
GROUPS = ("BG", "A", "B")
METRICS = ("accuracy", "answer_accuracy", "answer_probability")
COLORS = {"A": "#2166ac", "B": "#b2182b", "repA": "#67a9cf", "repB": "#ef8a62"}
ROLE_PAIRS = {
    "composition_used": (("A", "A"), ("B", "B")),
    "atomic_only": (("B", "A"), ("A", "B")),
    "matching_repetition": (("repA", "A"), ("repB", "B")),
    "other_repetition": (("repB", "A"), ("repA", "B")),
}
ROLE_LABELS = {
    "composition_used": "Facts used in composition training",
    "atomic_only": "Facts trained only as atomic queries",
    "matching_repetition": "Matching constituent-fact repetition",
    "other_repetition": "Repetition of the other fact cohort",
}
ROLE_COLORS = {
    "composition_used": "#2166ac",
    "atomic_only": "#b2182b",
    "matching_repetition": "#008837",
    "other_repetition": "#7b3294",
}


def rescore(predictions, name, rows):
    """Compute endpoint scores from each original answer, EOS and answer NLL."""
    n = len(rows)
    if not n:
        return {"n": 0, **{metric: None for metric in METRICS}, "nll": None}
    answer, stop, target, nll = (
        np.asarray(predictions[name + "_" + key]) for key in ("answer", "stop", "target", "nll")
    )
    if not (answer.shape == stop.shape == target.shape == (n,)) or nll.shape != (n, 2):
        raise ValueError(f"Invalid saved prediction shape: {name}")
    if not np.array_equal(target, rows[:, -1]):
        raise ValueError(f"Saved targets differ from fixed evaluation rows: {name}")
    if not np.isfinite(nll).all():
        raise ValueError(f"Nonfinite NLL: {name}")
    correct = answer == target
    return {
        "n": n,
        "answer_accuracy": float(correct.mean()),
        "accuracy": float((correct & (stop == 1)).mean()),
        "answer_probability": float(np.exp(-nll[:, 0].astype(np.float64)).mean()),
        "nll": float(nll.mean()),
        "eos_accuracy": float((stop == 1).mean()),
    }


def scheduled_counts(size, seed, salt, n_per_step, steps):
    """Independently count shuffled epochs without constructing giant streams."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), salt]))
    generated = 0
    permutation = np.empty(0, dtype=np.int64)
    result = {}
    for step in sorted(set(steps)):
        complete, remainder = divmod(step * n_per_step, size)
        needed = complete + bool(remainder)
        while generated < needed:
            permutation = rng.permutation(size)
            generated += 1
        counts = np.full(size, complete, dtype=np.int64)
        if remainder:
            counts[permutation[:remainder]] += 1
        result[step] = counts
    return result


def supplementary_scores(predictions, atomic, queries):
    """Keep the all-atomic-answers-correct subset separate from the full score."""
    if not len(queries):
        empty = {"n": 0, **{metric: None for metric in METRICS}, "nll": None}
        return {
            "test_atomic_answers_correct_subset": {**empty, "population_n": 0, "coverage": None},
            "autonomous_calls": {**empty, "all_intermediate_answers_and_eos_correct": None},
        }
    lookup = {(int(row[0]), int(row[1])): i for i, row in enumerate(atomic)}
    first = np.array([lookup[int(row[0]), int(row[1])] for row in queries])
    bridges = atomic[first, -1]
    second = np.array(
        [lookup[int(b), int(row[2])] for b, row in zip(bridges, queries, strict=True)]
    )
    correct_atomic = predictions["atomic_answer"] == atomic[:, -1]
    mask = correct_atomic[first] & correct_atomic[second]
    subset = {
        "conditional_" + key: predictions["test_composite_" + key][mask]
        for key in ("answer", "stop", "target", "nll")
    }
    conditional = rescore(subset, "conditional", queries[mask])
    conditional["population_n"] = len(queries)
    conditional["coverage"] = float(mask.mean()) if len(mask) else None
    final = predictions["autonomous_hop2_answer"] == queries[:, -1]
    eos = (predictions["autonomous_hop1_stop"] == 1) & (predictions["autonomous_hop2_stop"] == 1)
    autonomous = {
        "n": len(queries),
        "answer_accuracy": float(final.mean()),
        "accuracy": float((final & eos).mean()),
        "answer_probability": None,
        "nll": None,
        "all_intermediate_answers_and_eos_correct": float(
            (final & eos & (predictions["autonomous_hop1_answer"] == bridges)).mean()
        ),
    }
    return {"test_atomic_answers_correct_subset": conditional, "autonomous_calls": autonomous}


def expected_exposures(world, spec, steps):
    """Reconstruct per-fact occurrence totals from independently replayed RNGs."""
    kinds = world["table_row_kinds"]
    ids = world["table_fact_indices"]
    n_atomic = len(world["atomic"])
    repeat = spec["arm"].startswith("rep")
    sizes = [int((kinds == k).sum()) for k in (0, 1, 3 if repeat else 2)]
    role_size = sizes[2] // 2 if repeat else sizes[2]
    schedules = [
        scheduled_counts(size, spec["stream_seed"], salt, spec[nkey], steps)
        for size, salt, nkey in zip(
            [sizes[0], sizes[1], role_size],
            [146201, 146202, 146203],
            ["n_atomic", "n_background", "n_role"],
            strict=True,
        )
    ]
    result = {}
    for step in steps:
        table_counts = np.r_[
            schedules[0][step],
            schedules[1][step],
            np.repeat(schedules[2][step], 2) if repeat else schedules[2][step],
        ]
        exposure = np.zeros((4, n_atomic), dtype=np.int64)
        row_counts = np.zeros(4, dtype=np.int64)
        for kind in range(4):
            selected = kinds == kind
            row_counts[kind] = table_counts[selected].sum()
            facts = ids[selected].ravel()
            weights = np.repeat(table_counts[selected], 2)
            # Integer add.at avoids converting large counts through float64.
            valid = facts >= 0
            np.add.at(exposure[kind], facts[valid], weights[valid])
        result[step] = exposure, row_counts
    return result


def paired_effects(rows):
    """Same-query AA/BB contrasts; the two fact cohorts are weighted equally."""
    indexed = defaultdict(dict)
    for row in rows:
        if row["split"] in ("test_A_A", "test_B_B"):
            key = (row["phase"], row["world_seed"], row["initialization"], row["step"])
            indexed[key][row["arm"], row["split"]] = row
    effects = []
    for (phase, world, initialization, step), group in sorted(indexed.items()):
        for metric in METRICS:
            values = {
                f"{cohort}_{arm}": group.get((arm, f"test_{cohort}_{cohort}"), {}).get(metric)
                for cohort in ("A", "B")
                for arm in ARMS
            }

            def available(*keys, _values=values):
                return all(_values[key] is not None for key in keys)

            contrasts = {}
            if available("A_A", "A_B", "B_A", "B_B"):
                contrasts["main_role_A"] = values["A_A"] - values["A_B"]
                contrasts["main_role_B"] = values["B_B"] - values["B_A"]
                contrasts["main_role_equal_cohorts"] = (
                    contrasts["main_role_A"] + contrasts["main_role_B"]
                ) / 2
            if available("A_A", "A_repA"):
                contrasts["composition_minus_repetition_A"] = values["A_A"] - values["A_repA"]
            if available("B_B", "B_repB"):
                contrasts["composition_minus_repetition_B"] = values["B_B"] - values["B_repB"]
            if all(
                key in contrasts
                for key in ("composition_minus_repetition_A", "composition_minus_repetition_B")
            ):
                contrasts["composition_minus_repetition_equal_cohorts"] = (
                    contrasts["composition_minus_repetition_A"]
                    + contrasts["composition_minus_repetition_B"]
                ) / 2
            if all(value is not None for value in values.values()):
                repeated_role = (
                    values["A_repA"] - values["A_repB"] + values["B_repB"] - values["B_repA"]
                ) / 2
                contrasts["repetition_role_equal_cohorts"] = repeated_role
                contrasts["role_difference_in_differences"] = (
                    contrasts["main_role_equal_cohorts"] - repeated_role
                )
            for name, effect in contrasts.items():
                effects.append(
                    {
                        "phase": phase,
                        "world_seed": world,
                        "initialization": initialization,
                        "step": step,
                        "metric": metric,
                        "comparison": name,
                        "effect": effect,
                        **values,
                    }
                )
    return effects


def world_average_effects(effects):
    groups = defaultdict(list)
    for row in effects:
        key = tuple(row[key] for key in ("phase", "world_seed", "step", "metric", "comparison"))
        groups[key].append(row)
    output = []
    for key, values in sorted(groups.items()):
        item = dict(zip(("phase", "world_seed", "step", "metric", "comparison"), key, strict=True))
        item["initializations"] = len(values)
        for name in ("effect",) + tuple(f"{cohort}_{arm}" for cohort in ("A", "B") for arm in ARMS):
            observations = [row[name] for row in values if row[name] is not None]
            item[name] = float(np.mean(observations)) if len(observations) == len(values) else None
        output.append(item)
    return output


def align_learning_roles(rows):
    """Align each fact cohort with its training role, then average within worlds."""
    indexed = defaultdict(dict)
    for row in rows:
        if row["split"] in ("atomic_A", "atomic_B", "test_A_A", "test_B_B"):
            key = (row["phase"], row["world_seed"], row["initialization"], row["step"])
            indexed[key][row["arm"], row["split"]] = row
    within_world = defaultdict(list)
    for (phase, world, _initialization, step), measurements in indexed.items():
        for role, pairs in ROLE_PAIRS.items():
            for task in ("atomic", "composition"):
                for metric in METRICS:
                    values = []
                    for arm, cohort in pairs:
                        split = (
                            f"atomic_{cohort}" if task == "atomic" else f"test_{cohort}_{cohort}"
                        )
                        value = measurements.get((arm, split), {}).get(metric)
                        if value is not None:
                            values.append(value)
                    if len(values) == 2:
                        # Exactly half weight per cohort, regardless of its query count.
                        key = (phase, world, step, role, task, metric)
                        within_world[key].append(sum(values) / 2)
    output = []
    for key, values in sorted(within_world.items()):
        output.append(
            {
                **dict(
                    zip(("phase", "world_seed", "step", "role", "task", "metric"), key, strict=True)
                ),
                "initializations": len(values),
                "value": float(np.mean(values)),
            }
        )
    return output


def average_aligned_worlds(world_rows, expected_worlds):
    """Retain every node; full-world means require every configured world."""
    grouped = defaultdict(list)
    for row in world_rows:
        key = tuple(row[name] for name in ("phase", "step", "role", "task", "metric"))
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        phase = key[0]
        observed = {row["world_seed"] for row in values}
        required = set(expected_worlds.get(phase, observed))
        complete = observed == required
        output.append(
            {
                **dict(zip(("phase", "step", "role", "task", "metric"), key, strict=True)),
                "worlds": len(observed),
                "expected_worlds": len(required),
                "all_worlds_present": complete,
                "value": float(np.mean([row["value"] for row in values])) if complete else None,
                "world_min": min(row["value"] for row in values),
                "world_max": max(row["value"] for row in values),
            }
        )
    return output


def plot_aligned_learning(world_rows, phase_rows, expected_worlds, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    for phase in sorted({row["phase"] for row in world_rows}):
        phase_worlds = sorted(expected_worlds[phase])
        initialization_counts = {
            row["initializations"] for row in world_rows if row["phase"] == phase
        }
        initialization_text = (
            "1 initialization per world"
            if initialization_counts == {1}
            else "initializations averaged within each world"
        )
        for metric, file_label, y_label in (
            ("accuracy", "learning", "Answer + EOS accuracy (%)"),
            ("answer_probability", "probability", "Mean correct-answer probability (%)"),
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True, sharey=True)
            for ax, task, title in zip(
                axes,
                ("atomic", "composition"),
                ("Atomic recall of the same fact cohorts", "Unseen combinations of those facts"),
                strict=True,
            ):
                for role in ROLE_PAIRS:
                    selected = [
                        row
                        for row in world_rows
                        if row["phase"] == phase
                        and row["task"] == task
                        and row["metric"] == metric
                        and row["role"] == role
                    ]
                    for world in phase_worlds:
                        points = sorted(
                            (row for row in selected if row["world_seed"] == world),
                            key=lambda row: row["step"],
                        )
                        ax.plot(
                            [row["step"] for row in points],
                            [row["value"] * 100 for row in points],
                            color=ROLE_COLORS[role],
                            alpha=0.25,
                            lw=0.9,
                        )
                    means = sorted(
                        (
                            row
                            for row in phase_rows
                            if row["phase"] == phase
                            and row["task"] == task
                            and row["metric"] == metric
                            and row["role"] == role
                        ),
                        key=lambda row: row["step"],
                    )
                    ax.plot(
                        [row["step"] for row in means],
                        [
                            row["value"] * 100 if row["value"] is not None else np.nan
                            for row in means
                        ],
                        color=ROLE_COLORS[role],
                        lw=2.5,
                        ls="--" if role in ("atomic_only", "other_repetition") else "-",
                        label=ROLE_LABELS[role],
                    )
                ax.set_title(title)
                ax.set_ylim(-2, 103)
                ax.set_xlabel("Training updates")
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda value, _position: f"{value / 1000:g}k")
                )
                ax.grid(alpha=0.2)
            axes[0].set_ylabel(y_label)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.03),
                ncol=2,
                fontsize=9,
                frameon=False,
            )
            fig.suptitle(
                f"{phase.capitalize()}: fact recall and composition during training", fontsize=14
            )
            fig.text(
                0.5,
                0.015,
                "A/B cohorts weighted equally within worlds; "
                f"{len(phase_worlds)} {'world' if len(phase_worlds) == 1 else 'worlds'} "
                "weighted equally. Thin lines: individual worlds; "
                f"{initialization_text}. No confidence intervals.",
                ha="center",
                fontsize=8,
            )
            fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.27, wspace=0.13)
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"role-aligned-{file_label}-{phase}.{suffix}", dpi=200)
            plt.close(fig)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, effects, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combinations = sorted({(row["phase"], row["world_seed"]) for row in rows})
    for phase, world in combinations:
        selected = [row for row in rows if row["phase"] == phase and row["world_seed"] == world]
        for figure_name, splits, titles in (
            (
                "composition-learning",
                ["test_A_A", "test_B_B", "test_BG_BG"],
                ["Held-out A→A", "Held-out B→B", "Held-out BG→BG"],
            ),
            (
                "atomic-learning",
                ["atomic_A", "atomic_B", "atomic_BG"],
                ["Atomic A facts", "Atomic B facts", "Atomic background"],
            ),
        ):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey="row")
            for column, (split, title) in enumerate(zip(splits, titles, strict=True)):
                for row_index, metric in enumerate(("accuracy", "answer_probability")):
                    ax = axes[row_index, column]
                    for arm in ARMS:
                        by_step = defaultdict(list)
                        for row in selected:
                            if (
                                row["arm"] == arm
                                and row["split"] == split
                                and row[metric] is not None
                            ):
                                by_step[row["step"]].append(row[metric])
                        x = sorted(by_step)
                        y = [np.mean(by_step[step]) * 100 for step in x]
                        ax.plot(
                            x,
                            y,
                            label=arm,
                            color=COLORS[arm],
                            ls="--" if arm.startswith("rep") else "-",
                        )
                    ax.set_ylim(-2, 102)
                    ax.grid(alpha=0.2)
                    if row_index == 0:
                        ax.set_title(title)
                    else:
                        ax.set_xlabel("Training updates")
                    if column == 0:
                        ax.set_ylabel(
                            "Answer + EOS (%)"
                            if metric == "accuracy"
                            else "Mean correct-answer probability (%)"
                        )
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{phase}, world {world}; initialization means within this world")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"{figure_name}-{phase}-w{world}.{suffix}", dpi=180)
            plt.close(fig)

        endpoints = [row for row in selected if row["is_fixed_endpoint"]]
        if endpoints:
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            for column, arm in enumerate(ARMS):
                for row_index, metric in enumerate(("accuracy", "answer_probability")):
                    matrix = np.full((3, 3), np.nan)
                    n_matrix = np.zeros((3, 3), dtype=int)
                    for i, left in enumerate(GROUPS):
                        for j, right in enumerate(GROUPS):
                            entries = [
                                row
                                for row in endpoints
                                if row["arm"] == arm
                                and row["split"] == f"test_{left}_{right}"
                                and row[metric] is not None
                            ]
                            if entries:
                                matrix[i, j] = np.mean([row[metric] for row in entries]) * 100
                                n_matrix[i, j] = entries[0]["n"]
                    ax = axes[row_index, column]
                    picture = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
                    if not np.isfinite(matrix).any():
                        ax.text(1, 1, "Endpoint unavailable", ha="center", va="center", fontsize=9)
                    for i in range(3):
                        for j in range(3):
                            if np.isfinite(matrix[i, j]):
                                ax.text(
                                    j,
                                    i,
                                    f"{matrix[i, j]:.1f}\n(n={n_matrix[i, j]})",
                                    ha="center",
                                    va="center",
                                    color="white" if matrix[i, j] < 50 else "black",
                                    fontsize=8,
                                )
                    ax.set_xticks(range(3), GROUPS)
                    ax.set_yticks(range(3), GROUPS)
                    ax.set_title(f"{arm}: {'accuracy' if row_index == 0 else 'probability'}")
                    ax.set_xlabel("Second fact cohort")
                    if column == 0:
                        ax.set_ylabel("First fact cohort")
            fig.suptitle(f"Fixed endpoints: {phase}, world {world}; percent")
            fig.subplots_adjust(top=0.9, bottom=0.09, left=0.07, right=0.88, hspace=0.35)
            color_axis = fig.add_axes([0.92, 0.22, 0.015, 0.56])
            fig.colorbar(picture, cax=color_axis)
            for suffix in ("png", "pdf"):
                fig.savefig(out / f"endpoint-nine-grid-{phase}-w{world}.{suffix}", dpi=180)
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        comparisons = (
            ("main_role_equal_cohorts", "Composition role (A/B swap)"),
            ("composition_minus_repetition_equal_cohorts", "Composition minus atomic repetition"),
            ("role_difference_in_differences", "Role difference minus repetition role difference"),
        )
        for ax, metric in zip(axes, ("accuracy", "answer_probability"), strict=True):
            for name, label in comparisons:
                points = [
                    row
                    for row in effects
                    if row["phase"] == phase
                    and row["world_seed"] == world
                    and row["comparison"] == name
                    and row["metric"] == metric
                ]
                ax.plot(
                    [row["step"] for row in points],
                    [row["effect"] * 100 for row in points],
                    label=label,
                )
            ax.axhline(0, color="black", lw=0.6)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Training updates")
            ax.set_ylabel("Paired effect (percentage points)")
            ax.set_title(metric.replace("_", " "))
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{phase}, world {world}; AA and BB cohorts weighted equally")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"paired-effects-{phase}-w{world}.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/grok-usage-v1")
    parser.add_argument("--out", default="docs/development-artifacts/grok-usage-v1/report")
    parser.add_argument("--phases", nargs="+", default=["development", "confirmation"])
    parser.add_argument("--configs", nargs="*", default=["configs/grok-usage-development-v1.json"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root, out = ROOT / args.input, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    expected_runs = {}
    for name in args.configs:
        config = json.loads((ROOT / name).read_text())
        for run_id, updates in config["runs"].items():
            spec = {**config["base"], **updates}
            if spec["phase"] in args.phases:
                expected_runs[spec["phase"], run_id] = spec
    rows, runs, errors, missing, exposure_records = [], [], [], [], []
    fact_endpoint_rows = []
    saved_exposure, inputs = {}, {}
    checks, max_metric_delta = 0, 0.0
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        spec = metadata["spec"]
        if spec["phase"] not in args.phases:
            continue
        run = metadata_path.parent
        run_id = run.name
        complete = (run / "complete.json").exists()
        runs.append({"run_id": run_id, "spec": spec, "complete": complete})
        if not (run / "learning.json").exists() or not (run / "world.npz").exists():
            missing.append(str(run.relative_to(ROOT)) + ":learning/world")
            continue
        learning = json.loads((run / "learning.json").read_text())
        with np.load(run / "world.npz") as source:
            world = {key: source[key] for key in source.files}
        evaluation = {key[5:]: value for key, value in world.items() if key.startswith("eval_")}
        planned = expected_exposures(world, spec, [row["step"] for row in learning])
        for path in (
            metadata_path,
            run / "world.npz",
            run / "world-metadata.json",
            run / "learning.json",
        ):
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for registered in learning:
            step = registered["step"]
            prediction_path = run / f"predictions-{step:07d}.npz"
            exposure_path = run / f"exposure-{step:07d}.npz"
            if not prediction_path.exists() or not exposure_path.exists():
                missing.append(f"{run_id}:{step}:predictions/exposure")
                continue
            with np.load(prediction_path) as predictions:
                common = {
                    "phase": spec["phase"],
                    "run_id": run_id,
                    "world_seed": spec["world_seed"],
                    "initialization": spec["initialization"],
                    "arm": spec["arm"],
                    "step": step,
                    "is_fixed_endpoint": step == spec["steps"],
                    "actual_examples": registered["actual_examples"],
                    "supervised_tokens": registered["supervised_tokens"],
                    "effective_input_tokens": registered["effective_input_tokens"],
                    "estimated_training_flops": registered["estimated_training_flops"],
                }
                for name, query_rows in evaluation.items():
                    scored = rescore(predictions, name, query_rows)
                    old = registered["metrics"][name]
                    for metric in ("n", "nll") + METRICS:
                        if scored.get(metric) is None:
                            continue
                        difference = abs(scored[metric] - old[metric])
                        max_metric_delta = max(max_metric_delta, difference)
                        checks += 1
                        if difference > 2e-7:
                            errors.append(
                                f"{run_id}:{step}:{name}:{metric}:score differs {difference}"
                            )
                    rows.append({**common, "split": name, **scored})
                supplements = supplementary_scores(
                    predictions, world["atomic"], evaluation["test_composite"]
                )
                for name, scored in supplements.items():
                    rows.append({**common, "split": name, **scored})
                    if name == "autonomous_calls":
                        for metric in (
                            "accuracy",
                            "answer_accuracy",
                            "all_intermediate_answers_and_eos_correct",
                        ):
                            if scored[metric] is None:
                                continue
                            checks += 1
                            if abs(scored[metric] - registered["metrics"][name][metric]) > 1e-12:
                                errors.append(f"{run_id}:{step}:autonomous:{metric}")
            with np.load(exposure_path) as source:
                counts, kinds = source["counts"], source["kinds"]
            expected_counts, expected_kinds = planned[step]
            checks += counts.size + kinds.size
            if not np.array_equal(counts, expected_counts) or not np.array_equal(
                kinds, expected_kinds
            ):
                errors.append(
                    f"{run_id}:{step}:exposure differs from independent sample-stream replay"
                )
            budget_values = {
                "actual_examples": int(expected_kinds.sum()),
                "supervised_tokens": int(expected_kinds.sum()) * 2,
                "effective_input_tokens": int((expected_kinds * np.array([3, 4, 4, 3])).sum()),
                "logical_slots": step * (spec["n_atomic"] + spec["n_background"] + spec["n_role"]),
            }
            for name, value in budget_values.items():
                checks += 1
                if registered[name] != value:
                    errors.append(f"{run_id}:{step}:{name}:budget count differs")
            key = (spec["phase"], spec["world_seed"], spec["initialization"], step, spec["arm"])
            saved_exposure[key] = counts
            if step == spec["steps"]:
                cohorts = {
                    tuple(fact): group for group in GROUPS for fact in evaluation["atomic_" + group]
                }
                for fact_index, fact in enumerate(world["atomic"]):
                    fact_endpoint_rows.append(
                        {
                            "phase": spec["phase"],
                            "run_id": run_id,
                            "world_seed": spec["world_seed"],
                            "initialization": spec["initialization"],
                            "arm": spec["arm"],
                            "step": step,
                            "fact_index": fact_index,
                            "head": int(fact[0]),
                            "relation": int(fact[1]),
                            "tail": int(fact[2]),
                            "cohort": cohorts[tuple(fact)],
                            **{
                                name: int(counts[kind, fact_index])
                                for kind, name in enumerate(
                                    (
                                        "base_atomic",
                                        "background_composite",
                                        "role_composite",
                                        "repetition_atomic",
                                    )
                                )
                            },
                            "total_raw_constituent_occurrences": int(counts[:, fact_index].sum()),
                            "independent_replay_exact": np.array_equal(
                                counts[:, fact_index], expected_counts[:, fact_index]
                            ),
                        }
                    )
            for kind in range(4):
                exposure_records.append(
                    {
                        "phase": spec["phase"],
                        "run_id": run_id,
                        "step": step,
                        "kind": kind,
                        "rows": int(kinds[kind]),
                        "constituent_occurrences": int(counts[kind].sum()),
                        "fact_count_min": int(counts[kind].min()),
                        "fact_count_max": int(counts[kind].max()),
                        "independent_replay_exact": np.array_equal(
                            counts[kind], expected_counts[kind]
                        ),
                    }
                )
            for path in (prediction_path, exposure_path):
                inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    pairing_checks = []
    prefixes = sorted({key[:-1] for key in saved_exposure})
    for prefix in prefixes:
        present = {
            arm: saved_exposure[prefix + (arm,)]
            for arm in ARMS
            if prefix + (arm,) in saved_exposure
        }
        for left, right in (("A", "B"), ("A", "repA"), ("B", "repB"), ("repA", "repB")):
            if left not in present or right not in present:
                continue
            for kind in (0, 1):
                exact = np.array_equal(present[left][kind], present[right][kind])
                checks += len(present[left][kind])
                pairing_checks.append(
                    {
                        "phase": prefix[0],
                        "world_seed": prefix[1],
                        "initialization": prefix[2],
                        "step": prefix[3],
                        "left": left,
                        "right": right,
                        "kind": kind,
                        "exact": exact,
                    }
                )
                if not exact:
                    errors.append(f"{prefix}:{left}/{right}:base/background exposure differs")
        for role in ("A", "B"):
            if role in present and "rep" + role in present:
                exact = np.array_equal(present[role][2], present["rep" + role][3])
                checks += len(present[role][2])
                pairing_checks.append(
                    {
                        "phase": prefix[0],
                        "world_seed": prefix[1],
                        "initialization": prefix[2],
                        "step": prefix[3],
                        "left": role,
                        "right": "rep" + role,
                        "kind": "role_vs_repetition",
                        "exact": exact,
                    }
                )
                if not exact:
                    errors.append(f"{prefix}:{role}:constituent repetition exposure differs")
    observed = {(run["spec"]["phase"], run["run_id"]) for run in runs}
    missing.extend(
        f"{phase}/{run_id}:missing run"
        for phase, run_id in expected_runs
        if (phase, run_id) not in observed
    )
    seed_effects = paired_effects(rows)
    world_effects = world_average_effects(seed_effects)
    endpoint_keys = {
        (run["spec"]["phase"], run["spec"]["world_seed"], run["spec"]["steps"]) for run in runs
    }
    endpoints = [
        row
        for row in world_effects
        if (row["phase"], row["world_seed"], row["step"]) in endpoint_keys
    ]
    expected_initializations = defaultdict(set)
    for spec in expected_runs.values():
        expected_initializations[spec["phase"], spec["world_seed"]].add(spec["initialization"])
    for row in endpoints:
        expected = len(expected_initializations[row["phase"], row["world_seed"]])
        row["expected_initializations"] = expected or row["initializations"]
        row["partial_initializations"] = row["initializations"] < row["expected_initializations"]
    phase_means = []
    for phase in args.phases:
        for metric in METRICS:
            for comparison in sorted({row["comparison"] for row in endpoints}):
                values = [
                    row
                    for row in endpoints
                    if row["phase"] == phase
                    and row["metric"] == metric
                    and row["comparison"] == comparison
                ]
                if values:
                    expected_worlds = len(
                        {
                            spec["world_seed"]
                            for spec in expected_runs.values()
                            if spec["phase"] == phase
                        }
                    )
                    phase_means.append(
                        {
                            "phase": phase,
                            "metric": metric,
                            "comparison": comparison,
                            "worlds": len(values),
                            "expected_worlds": expected_worlds or len(values),
                            "partial": (expected_worlds > len(values))
                            or any(row["partial_initializations"] for row in values),
                            "effect": float(np.mean([row["effect"] for row in values])),
                        }
                    )
    summary = {
        "status": "complete"
        if runs and not errors and not missing and all(run["complete"] for run in runs)
        else "partial_or_failed",
        "runs": runs,
        "expected_run_count": len(expected_runs),
        "missing": missing,
        "errors": errors,
        "checks": checks,
        "maximum_score_difference": max_metric_delta,
        "scored_split_nodes": len(rows),
        "world_endpoint_effects": endpoints,
        "phase_world_equal_means": phase_means,
        "interpretation": [
            "Main contrasts compare the same AA or BB held-out queries across A/B models; "
            "the two cohorts are averaged equally, not pooled by query count.",
            "Initializations are averaged within worlds; development and confirmation stay "
            "separate. Query counts are denominators, not independent worlds.",
            "Repetition matches raw constituent occurrences with half loss weight per atom. "
            "Information, direct supervision and actual compute differ.",
            "Differences identify the training treatment's overall effect; shared parameters "
            "and other composition training also change.",
            "Only fixed budget endpoints enter endpoint statistics; partial reports do not "
            "choose best checkpoints.",
            "The atomic-correct subset requires both separate atomic answers to be correct, "
            "with coverage reported; it never replaces the all-query score.",
        ],
        "input_sha256": inputs,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_csv(out / "nodes.csv", rows)
    write_csv(out / "seed-effects.csv", seed_effects)
    write_csv(out / "world-effects.csv", world_effects)
    write_csv(out / "exposure-summary.csv", exposure_records)
    write_csv(out / "fact-exposure-endpoints.csv", fact_endpoint_rows)
    write_csv(out / "paired-exposure-checks.csv", pairing_checks)
    aligned_worlds = align_learning_roles(rows)
    configured_worlds = {
        phase: {spec["world_seed"] for spec in expected_runs.values() if spec["phase"] == phase}
        or {row["world_seed"] for row in rows if row["phase"] == phase}
        for phase in args.phases
    }
    aligned_means = average_aligned_worlds(aligned_worlds, configured_worlds)
    write_csv(out / "role-aligned-world-nodes.csv", aligned_worlds)
    write_csv(out / "role-aligned-phase-nodes.csv", aligned_means)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if rows and not args.no_plots:
        plot_results(rows, world_effects, out)
        plot_aligned_learning(aligned_worlds, aligned_means, configured_worlds, out)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in ("status", "checks", "scored_split_nodes", "errors", "missing")
            },
            ensure_ascii=False,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
