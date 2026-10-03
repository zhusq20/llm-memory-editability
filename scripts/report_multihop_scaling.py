"""Audit raw multi-hop outputs and report world-balanced, paired comparisons.

All registered nodes and cells are retained. Initializations are paired within
worlds; worlds, rather than individual queries or runs, receive equal weight.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

TASKS = ("atomic",) + tuple(
    f"{pool}_{hop}" for hop in (2, 3, 4) for pool in ("train", "familiar", "strict")
)
COMMON_TASKS = tuple(task for task in TASKS if not task.startswith("train_"))
SCORES = (
    "accuracy",
    "answer_accuracy",
    "atomic_correct_coverage",
    "conditional_accuracy",
    "autonomous_two_calls",
    "autonomous_path_accuracy",
)
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_name(spec):
    return (
        f"w{spec['world_seed']}-i{spec['initialization']}-d{spec['width']}"
        f"-phi{spec['phi']:g}-l{spec['layers']}-r{spec['repeats']}-s{spec['steps']}"
    )


def identity(spec):
    return {
        "world": spec["world_seed"],
        "initialization": spec["initialization"],
        "width": spec["width"],
        "phi": spec["phi"],
        "architecture": "loop" if spec["repeats"] > 1 else "standard",
        "layers": spec["layers"],
        "repeats": spec["repeats"],
        "executed_depth": spec["layers"] * spec["repeats"],
    }


def mean(values):
    return float(np.mean(values)) if len(values) else None


def generated_score(rows, generated):
    generated = np.asarray(generated)
    if generated.shape != (len(rows), 2):
        raise ValueError("Expected greedy answer and EOS for every query")
    correct = (generated[:, 0] == rows[:, -1]) & (generated[:, 1] == 1)
    return mean(correct), correct


def truth_paths(world, rows):
    """Traverse the archived atomic graph without using the training evaluator."""
    lookup = {(int(h), int(r)): (i, int(t)) for i, (h, r, t) in enumerate(world["atomic"])}
    if len(lookup) != len(world["atomic"]):
        raise ValueError("Duplicate atomic keys")
    hops = rows.shape[1] - 2
    truth = np.empty((len(rows), hops), dtype=np.int64)
    indices = np.empty_like(truth)
    for i, row in enumerate(rows):
        current = int(row[0])
        for j, relation in enumerate(row[1:-1]):
            index, current = lookup[(current, int(relation))]
            indices[i, j], truth[i, j] = index, current
        if current != int(row[-1]):
            raise ValueError("Composition endpoint disagrees with archived atomic truth")
    return truth, indices


def _metric_equal(actual, expected, label):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(f"Missing metric differs: {label}")
    elif not np.isclose(actual, expected, atol=1e-7, rtol=1e-6):
        raise ValueError(f"Raw-token metric differs: {label}: {actual} != {expected}")


def audit_predictions(world, predictions, metrics):
    """Independently verify answer/EOS, constituent coverage and external calls."""
    _, atomic_correct = generated_score(world["atomic"], predictions["atomic_generated"])
    checked = 0
    for task in TASKS:
        rows = world[task]
        generated = predictions[f"{task}_generated"]
        accuracy, correct = generated_score(rows, generated)
        np.testing.assert_array_equal(correct, predictions[f"{task}_correct"])
        calculated = {
            "n": len(rows),
            "accuracy": accuracy,
            "answer_accuracy": mean(generated[:, 0] == rows[:, -1]),
            "answer_nll": mean(predictions[f"{task}_answer_nll"]),
        }
        if task != "atomic":
            truth, edges = truth_paths(world, rows)
            coverage = atomic_correct[edges].all(axis=1)
            calls = predictions[f"{task}_autonomous_generated"]
            if calls.shape != (len(rows), rows.shape[1] - 2, 2):
                raise ValueError("Expected answer and EOS for each autonomous call")
            formats = (calls[:, :, 1] == 1).all(axis=1)
            terminal = formats & (calls[:, -1, 0] == rows[:, -1])
            paths = formats & (calls[:, :, 0] == truth).all(axis=1)
            for key, value in (
                ("coverage", coverage),
                ("autonomous_correct", terminal),
                ("autonomous_path_correct", paths),
            ):
                np.testing.assert_array_equal(value, predictions[f"{task}_{key}"])
            calculated.update(
                {
                    "atomic_correct_coverage": mean(coverage),
                    "conditional_accuracy": mean(correct[coverage]),
                    "autonomous_two_calls": mean(terminal),
                    "autonomous_path_accuracy": mean(paths),
                    "hop_count": rows.shape[1] - 2,
                }
            )
        if set(calculated) != set(metrics[task]):
            raise ValueError("Metric fields differ: " + task)
        for metric, value in calculated.items():
            _metric_equal(value, metrics[task][metric], f"{task}/{metric}")
            checked += 1
    return {"passed": True, "tasks": len(TASKS), "metrics_independently_checked": checked}


def common_digest(world, id_mask):
    digest = hashlib.sha256()
    for task in COMMON_TASKS:
        rows = np.asarray(world[task], dtype="<i8")
        digest.update(task.encode())
        digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
        digest.update(rows.tobytes())
    digest.update(json.dumps(id_mask).encode())
    return digest.hexdigest()


def training_experience(world, exposures):
    """Describe fact-role coverage and realized exposure, without fitted predictors."""
    atoms = len(world["atomic"])
    roles = np.zeros((atoms, 4), dtype=np.int64)
    counts = np.zeros_like(roles)
    for hop in (2, 3, 4):
        task = f"train_{hop}"
        _, edges = truth_paths(world, world[task])
        weight = exposures[task]
        if weight.shape != (len(edges),) or np.any(weight < 0):
            raise ValueError("Invalid per-example training exposure counters")
        for position in range(hop):
            np.add.at(roles[:, position], edges[:, position], 1)
            np.add.at(counts[:, position], edges[:, position], weight)
    atomic = exposures["atomic"]
    if atomic.shape != (atoms,) or np.any(atomic < 0):
        raise ValueError("Invalid atomic training exposure counters")
    records = []
    for task in TASKS[1:]:
        _, edges = truth_paths(world, world[task])
        positions = np.arange(edges.shape[1])[None, :]
        same_role = roles[edges, positions] > 0
        ever_used = roles.sum(axis=1)[edges] > 0
        experienced = counts[edges, positions] > 0
        record = {
            "task": task,
            "n": len(edges),
            "unique_necessary_facts": len(np.unique(edges)),
            "all_facts_in_any_composition_training": mean(ever_used.all(axis=1)),
            "all_facts_in_same_position_training": mean(same_role.all(axis=1)),
            "all_facts_exposed_at_same_position": mean(experienced.all(axis=1)),
            "atomic_training_draws": int(atomic.sum()),
            "atomic_draws_per_fact_min": int(atomic.min()),
            "atomic_draws_per_fact_mean": mean(atomic),
            "atomic_draws_per_fact_max": int(atomic.max()),
            "composition_fact_occurrences": int(counts.sum()),
        }
        for position in range(edges.shape[1]):
            needed = edges[:, position]
            record.update(
                {
                    f"position_{position + 1}_role_coverage": mean(same_role[:, position]),
                    f"position_{position + 1}_distinct_facts": len(np.unique(needed)),
                    f"position_{position + 1}_atomic_exposure_mean": mean(atomic[needed]),
                    f"position_{position + 1}_composition_role_exposure_mean": mean(
                        counts[needed, position]
                    ),
                }
            )
        records.append(record)
    return records


def write_csv(path, records):
    columns = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def world_balanced(records, group_keys, value_keys):
    """Average initializations inside each world, then weight worlds equally."""
    by_world = defaultdict(list)
    seen = set()
    for row in records:
        group = tuple(row[key] for key in group_keys)
        unique = (group, row["world"], row["initialization"])
        if unique in seen:
            raise ValueError("Duplicate world/initialization measurement in aggregate")
        seen.add(unique)
        by_world[(group, row["world"])].append(row)
    world_rows = []
    for (group, world), members in sorted(by_world.items()):
        result = dict(zip(group_keys, group, strict=True))
        result.update(world=world, initializations=len(members))
        for key in value_keys:
            values = [row.get(key) for row in members if row.get(key) is not None]
            result[key] = mean(values)
            result[f"{key}_initializations"] = len(values)
        world_rows.append(result)
    aggregate = defaultdict(list)
    for row in world_rows:
        aggregate[tuple(row[key] for key in group_keys)].append(row)
    result_rows = []
    for group, members in sorted(aggregate.items()):
        result = dict(zip(group_keys, group, strict=True))
        result.update(worlds=len(members), runs=sum(row["initializations"] for row in members))
        for key in value_keys:
            values = [row[key] for row in members if row[key] is not None]
            result[key] = mean(values)
            result[f"{key}_worlds"] = len(values)
            result[f"{key}_min_world"] = min(values) if values else None
            result[f"{key}_max_world"] = max(values) if values else None
        result_rows.append(result)
    return world_rows, result_rows


def paired_contrasts(curves):
    """Registered finite differences, always paired by world and initialization."""
    strata = defaultdict(dict)
    for row in curves:
        key = (row["world"], row["initialization"], row["step"], row["task"])
        cell = (row["width"], row["phi"], row["architecture"], row["executed_depth"])
        if cell in strata[key]:
            raise ValueError("Duplicate paired cell")
        strata[key][cell] = row
    results = []
    for (world, initialization, step, task), cells in sorted(strata.items()):

        def append(
            name,
            terms,
            width=0,
            phi=0,
            architecture="both",
            depth=0,
            context=(world, initialization, step, task, cells),
        ):
            world, initialization, step, task, cells = context
            if not all(cell in cells for _, cell in terms):
                return
            for metric in SCORES:
                if any(cells[cell].get(metric) is None for _, cell in terms):
                    continue
                value = sum(coef * cells[cell][metric] for coef, cell in terms)
                results.append(
                    {
                        "world": world,
                        "initialization": initialization,
                        "step": step,
                        "task": task,
                        "metric": metric,
                        "contrast": name,
                        "width": width,
                        "phi": phi,
                        "architecture": architecture,
                        "executed_depth": depth,
                        "difference_pp": 100 * value,
                    }
                )

        for width in (128, 256):
            for architecture in ("standard", "loop"):
                for depth in (2, 4):
                    append(
                        "support_gain",
                        [
                            (1, (width, 4.0, architecture, depth)),
                            (-1, (width, 1.0, architecture, depth)),
                        ],
                        width=width,
                        architecture=architecture,
                        depth=depth,
                    )
                for phi in (1.0, 4.0):
                    append(
                        "depth_gain",
                        [
                            (1, (width, phi, architecture, 4)),
                            (-1, (width, phi, architecture, 2)),
                        ],
                        width=width,
                        phi=phi,
                        architecture=architecture,
                    )
                append(
                    "support_x_depth",
                    [
                        (1, (width, 4.0, architecture, 4)),
                        (-1, (width, 4.0, architecture, 2)),
                        (-1, (width, 1.0, architecture, 4)),
                        (1, (width, 1.0, architecture, 2)),
                    ],
                    width=width,
                    architecture=architecture,
                )
            for depth in (2, 4):
                for phi in (1.0, 4.0):
                    append(
                        "loop_minus_standard",
                        [
                            (1, (width, phi, "loop", depth)),
                            (-1, (width, phi, "standard", depth)),
                        ],
                        width=width,
                        phi=phi,
                        depth=depth,
                    )
                append(
                    "support_x_loop",
                    [
                        (1, (width, 4.0, "loop", depth)),
                        (-1, (width, 4.0, "standard", depth)),
                        (-1, (width, 1.0, "loop", depth)),
                        (1, (width, 1.0, "standard", depth)),
                    ],
                    width=width,
                    depth=depth,
                )
        for architecture in ("standard", "loop"):
            for depth in (2, 4):
                for phi in (1.0, 4.0):
                    append(
                        "width_gain",
                        [
                            (1, (256, phi, architecture, depth)),
                            (-1, (128, phi, architecture, depth)),
                        ],
                        phi=phi,
                        architecture=architecture,
                        depth=depth,
                    )
                append(
                    "width_x_support",
                    [
                        (1, (256, 4.0, architecture, depth)),
                        (-1, (128, 4.0, architecture, depth)),
                        (-1, (256, 1.0, architecture, depth)),
                        (1, (128, 1.0, architecture, depth)),
                    ],
                    architecture=architecture,
                    depth=depth,
                )
    return results


def _save_figure(figure, output, name):
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def plot(curves, contrasts, output):
    styles = (
        ("standard", 2, "#2467a2", "-"),
        ("standard", 4, "#2467a2", "--"),
        ("loop", 2, "#d56a22", "-"),
        ("loop", 4, "#d56a22", "--"),
    )
    for pool in ("atomic", "train", "familiar", "strict", "coverage", "external"):
        columns = (0,) if pool == "atomic" else (2, 3, 4)
        figure, axes = plt.subplots(4, len(columns), figsize=(4 * len(columns), 9), squeeze=False)
        for row_index, (width, phi) in enumerate(((128, 1), (128, 4), (256, 1), (256, 4))):
            for col_index, hop in enumerate(columns):
                axis = axes[row_index, col_index]
                task = "atomic" if pool == "atomic" else f"{pool}_{hop}"
                metric = "accuracy"
                if pool in ("coverage", "external"):
                    task = f"familiar_{hop}"
                    metric = (
                        "atomic_correct_coverage" if pool == "coverage" else "autonomous_two_calls"
                    )
                for architecture, depth, color, linestyle in styles:
                    line = sorted(
                        (
                            item
                            for item in curves
                            if item["width"] == width
                            and item["phi"] == phi
                            and item["architecture"] == architecture
                            and item["executed_depth"] == depth
                            and item["task"] == task
                        ),
                        key=lambda item: item["step"],
                    )
                    if line:
                        axis.plot(
                            [item["step"] for item in line],
                            [
                                np.nan if item[metric] is None else 100 * item[metric]
                                for item in line
                            ],
                            "o" + linestyle,
                            color=color,
                            markersize=3,
                            label=f"{architecture} D{depth}",
                        )
                axis.set_title(f"width {width} | support {phi} | {task}", fontsize=9)
                axis.set_ylim(-3, 103)
                axis.grid(alpha=0.2)
                if col_index == 0:
                    axis.set_ylabel("Accuracy / coverage (%)")
                if row_index == 3:
                    axis.set_xlabel("Training updates")
        axes[0, -1].legend(fontsize=7)
        figure.suptitle(f"{pool}: initialization mean within world, equal-weight world mean")
        _save_figure(figure, output, f"learning-{pool}")
    selected = [
        row
        for row in contrasts
        if row["metric"] == "accuracy"
        and row["task"].startswith("familiar_")
        and row["contrast"] in ("support_x_depth", "width_x_support", "support_x_loop")
    ]
    if selected:
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for axis, name in zip(
            axes, ("support_x_depth", "width_x_support", "support_x_loop"), strict=True
        ):
            for row in selected:
                if row["contrast"] != name:
                    continue
                label = (
                    f"w{row['width']} {row['architecture']} D{row['executed_depth']} {row['task']}"
                )
                x = [
                    item
                    for item in selected
                    if all(
                        item[key] == row[key]
                        for key in ("contrast", "width", "architecture", "executed_depth", "task")
                    )
                ]
                if row is not x[0]:
                    continue
                x.sort(key=lambda item: item["step"])
                axis.plot(
                    [item["step"] for item in x],
                    [item["difference_pp"] for item in x],
                    "o-",
                    label=label,
                )
            axis.axhline(0, color="gray", linewidth=0.8)
            axis.set_title(name)
            axis.set_xlabel("Training updates")
            axis.set_ylabel("Paired interaction (pp)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=5)
        _save_figure(figure, output, "paired-interactions")


def analyze(config_path, results_path, output, allow_incomplete=False):
    config_path, results_path, output = map(Path, (config_path, results_path, output))
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    records, curves, audits, pending, experience = [], [], [], [], []
    digests = {}
    paired_atomic_exposure = {}
    for spec in config["specs"]:
        folder = results_path / run_name(spec)
        if not (folder / "complete.json").exists() or not (folder / "audit.json").exists():
            if allow_incomplete:
                pending.append(folder.name)
                continue
            raise FileNotFoundError(f"Run has not completed training and reload audit: {folder}")
        complete = json.loads((folder / "complete.json").read_text())
        reload_audit = json.loads((folder / "audit.json").read_text())
        if complete["spec"] != spec or complete["source"] != config["source"]:
            raise ValueError("Run differs from the frozen specification/source: " + str(folder))
        if not reload_audit["passed"]:
            raise ValueError("Checkpoint reload audit failed: " + str(folder))
        history = json.loads((folder / "learning.json").read_text())
        if [row["step"] for row in history] != spec["nodes"]:
            raise ValueError("Missing or extra learning nodes: " + str(folder))
        if history[-1] != complete["endpoint"]:
            raise ValueError("Complete endpoint differs from full learning trajectory")
        metadata = json.loads((folder / "world-metadata.json").read_text())
        with np.load(folder / "world.npz") as world:
            digest = common_digest(world, metadata["id_mask"])
            if digest != spec["common_evaluation_sha256"]:
                raise ValueError(
                    "Archived common evaluation digest differs from frozen specification"
                )
            previous = digests.setdefault(spec["world_seed"], digest)
            if digest != previous:
                raise ValueError("Support arms/initializations do not share evaluation queries")
            for row in history:
                with np.load(folder / f"predictions-{row['step']:06d}.npz") as predictions:
                    audit = audit_predictions(world, predictions, row["metrics"])
                with np.load(folder / f"exposures-{row['step']:06d}.npz") as exposures:
                    pair = (spec["world_seed"], spec["initialization"], row["step"])
                    atomic_counts = exposures["atomic"].copy()
                    previous_counts = paired_atomic_exposure.setdefault(pair, atomic_counts)
                    np.testing.assert_array_equal(atomic_counts, previous_counts)
                    experience.extend(
                        {"run": folder.name, **identity(spec), "step": row["step"], **item}
                        for item in training_experience(world, exposures)
                    )
                audits.append({"run": folder.name, "step": row["step"], **audit})
                for task in TASKS:
                    curves.append(
                        {
                            "run": folder.name,
                            **identity(spec),
                            "step": row["step"],
                            "task": task,
                            **row["metrics"][task],
                            "parameters": complete["parameters"],
                            **{
                                key: row[key]
                                for key in (
                                    "supervised_tokens",
                                    "estimated_training_flops",
                                    "training_seconds",
                                )
                            },
                        }
                    )
        endpoint = {
            "run": folder.name,
            **identity(spec),
            "steps": spec["steps"],
            "parameters": complete["parameters"],
            "common_evaluation_sha256": digest,
            **{
                key: history[-1][key]
                for key in ("supervised_tokens", "estimated_training_flops", "training_seconds")
            },
        }
        for task in TASKS:
            endpoint.update(
                {f"{task}_{key}": value for key, value in history[-1]["metrics"][task].items()}
            )
        records.append(endpoint)
    groups = ("width", "phi", "architecture", "executed_depth", "step", "task")
    values = (
        "accuracy",
        "answer_accuracy",
        "answer_nll",
        "atomic_correct_coverage",
        "conditional_accuracy",
        "autonomous_two_calls",
        "autonomous_path_accuracy",
    )
    world_curves, aggregates = world_balanced(curves, groups, values)
    pairs = paired_contrasts(curves)
    contrast_keys = (
        "contrast",
        "width",
        "phi",
        "architecture",
        "executed_depth",
        "step",
        "task",
        "metric",
    )
    world_pairs, aggregate_pairs = world_balanced(pairs, contrast_keys, ("difference_pp",))
    for name, rows in (
        ("endpoints", records),
        ("learning", curves),
        ("world-learning", world_curves),
        ("aggregate-learning", aggregates),
        ("paired-differences", pairs),
        ("world-paired-differences", world_pairs),
        ("aggregate-paired-differences", aggregate_pairs),
        ("training-experience", experience),
    ):
        write_csv(output / f"{name}.csv", rows)
    summary = {
        "phase": config["phase"],
        "analysis_unit": "Equal-weight worlds; paired initializations within each world",
        "config_sha256": file_hash(config_path),
        "analysis_source_sha256": file_hash(Path(__file__)),
        "design_sha256": config.get("design_sha256"),
        "registered_runs": len(config["specs"]),
        "completed_audited_runs": len(records),
        "pending_runs": pending,
        "complete": not pending,
        "raw_prediction_checks": audits,
        "common_evaluation_sha256_by_world": digests,
        "paired_atomic_exposure_exact": True,
        "training_experience": experience,
        "supervised_tokens": sum(row["supervised_tokens"] for row in records),
        "summed_training_seconds": sum(row["training_seconds"] for row in records),
        "estimated_training_flops": sum(row["estimated_training_flops"] for row in records),
        "endpoints": records,
        "world_learning": world_curves,
        "aggregate_learning": aggregates,
        "world_paired_differences": world_pairs,
        "aggregate_paired_differences": aggregate_pairs,
        "limitations": [
            "Atomic and external-call success establish accessible facts, not an internal circuit",
            "Support changes unique chains, role coverage and per-chain repetitions together",
            "Joint k=2/3/4 training does not test unseen-hop-length transfer",
            "Complete familiar queries are held out; cross-length subpaths can occur in training",
            "Matched executed depth does not match parameters or initialization scales",
            "Test-time recurrence changes the trained execution distribution; no best-R selection",
        ],
    }
    write_json(output / "summary.json", summary)
    if curves:
        plot(aggregates, aggregate_pairs, output)
    print(
        json.dumps(
            {
                "audited_runs": len(records),
                "registered_runs": len(config["specs"]),
                "nodes": len(audits),
                "complete": not pending,
            }
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    analyze(args.config, args.results, args.out, args.allow_incomplete)


if __name__ == "__main__":
    main()
