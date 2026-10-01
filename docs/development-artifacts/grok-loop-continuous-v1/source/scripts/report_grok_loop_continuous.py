#!/usr/bin/env python3
"""Exploratory continuous scoring of every saved grok-loop learning node.

Read-only inputs; no torch, forward passes, training, fitted predictor, or
checkpoint selection. Initialize means nest within worlds. Gold-answer EOS
probability and EOS after the generated answer are deliberately separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    "configs/grok-loop-development-v1.json",
    "configs/grok-loop-sensitivity-v1.json",
    "configs/grok-loop-confirmation-v1.json",
)
SPLITS = (
    "atomic",
    "id_atomic",
    "ood_atomic",
    "train_composite",
    "test_composite",
    "ood_composite",
)
METRICS = (
    "p_answer",
    "p_eos_given_gold",
    "p_gold_sequence",
    "answer_nll",
    "eos_given_gold_nll",
    "two_target_nll",
    "answer_accuracy",
    "complete_accuracy",
    "generated_eos_accuracy",
    "generated_eos_given_correct_answer_accuracy",
    "answer_correct_eos_wrong_fraction",
)
DYNAMICS_METRICS = (
    "p_answer",
    "p_eos_given_gold",
    "p_gold_sequence",
    "answer_accuracy",
    "complete_accuracy",
    "generated_eos_accuracy",
    "answer_correct_eos_wrong_fraction",
)
SEED_KEYS = {"world_seed", "initialization", "stream_seed"}
COLORS = {"c1": "#999999", "c2": "#4477aa", "cd": "#228833", "l1": "#ee6677", "l2": "#aa3377"}
CHANGE_TOLERANCE = 1e-8


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance_record(path, kind):
    path = Path(path).resolve()
    return {"path": str(path), "kind": kind, "bytes": path.stat().st_size, "sha256": digest(path)}


def identity(spec, run, config):
    condition = {k: v for k, v in spec.items() if k not in SEED_KEYS}
    return {
        "config": str(config),
        "run": run,
        "condition_id": hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[
            :16
        ],
        "phase": spec["phase"],
        "world_seed": spec["world_seed"],
        "initialization": spec["initialization"],
        "architecture": spec["architecture"],
        "hops": spec["hops"],
        "unique_layers": spec["layers"],
        "training_repeats": spec["repeats"],
        "effective_depth": spec["layers"] * spec["repeats"],
        "init_scheme": spec["init_scheme"],
        "width": spec["width"],
        "phi": spec["phi"],
        "endpoint_step": spec["steps"],
    }


def atomic_masks(world):
    atoms = [tuple(map(int, row)) for row in world["atomic"]]
    ids = {tuple(map(int, row)) for row in world["id_atomic"]}
    oods = {tuple(map(int, row)) for row in world["ood_atomic"]}
    if (
        len(set(atoms)) != len(atoms)
        or len(ids) != len(world["id_atomic"])
        or len(oods) != len(world["ood_atomic"])
        or ids & oods
        or ids | oods != set(atoms)
    ):
        raise ValueError("ID/OOD atomic facts must be an exact disjoint partition")
    return {
        name: np.asarray([row in group for row in atoms])
        for name, group in (("id_atomic", ids), ("ood_atomic", oods))
    }


def dataset_digest(world):
    """Reproduce the frozen dataset digest over every symbolic data split."""
    h = hashlib.sha256()
    for name in (
        "atomic",
        "id_atomic",
        "ood_atomic",
        "train_composite",
        "test_composite",
        "test_full_composite",
        "ood_composite",
        "unused_composite",
    ):
        rows = world[name]
        h.update(name.encode())
        h.update(np.asarray(rows.shape, dtype="<i8").tobytes())
        h.update(rows.astype("<i8", copy=False).tobytes())
    return h.hexdigest()


def score_saved(archive, prefix, target, selected=None, *, missing_reason=""):
    """Average probabilities per item; do not exponentiate the average NLL."""
    target = np.asarray(target)
    selected = np.ones(len(target), dtype=bool) if selected is None else np.asarray(selected)
    if selected.dtype != bool or selected.shape != target.shape:
        raise ValueError("Selection mask must align with target")
    n = int(selected.sum())
    result = {
        "population_n": n,
        "pool_n": len(target),
        "partition_fraction": n / len(target) if len(target) else None,
        "available": False,
        "prediction_n": 0,
        "prediction_coverage": 0.0 if n else None,
        "correct_answer_n": None,
        "missing_reason": missing_reason,
        **{metric: None for metric in METRICS},
    }
    keys = [prefix + "_" + key for key in ("answer", "stop", "target", "nll")]
    missing = [key for key in keys if key not in archive]
    if not n:
        return {**result, "available": True, "missing_reason": "empty population"}
    if missing:
        return {**result, "missing_reason": missing_reason or "; ".join(missing)}
    answer, stop, saved_target, nll = (np.asarray(archive[key]) for key in keys)
    if (
        answer.shape != target.shape
        or stop.shape != target.shape
        or not np.array_equal(saved_target, target)
        or nll.shape != (len(target), 2)
    ):
        raise ValueError(f"Saved arrays are not aligned: {prefix}")
    if not np.isfinite(nll).all() or np.any(nll < -1e-6):
        raise ValueError(f"NLL must be finite and nonnegative: {prefix}")
    losses = nll[selected].astype(np.float64)
    answer_correct = answer[selected] == target[selected]
    eos_correct = stop[selected] == 1
    answer_n = int(answer_correct.sum())
    return {
        **result,
        "available": True,
        "prediction_n": n,
        "prediction_coverage": 1.0,
        "missing_reason": "",
        "correct_answer_n": answer_n,
        "p_answer": float(np.exp(-losses[:, 0]).mean()),
        "p_eos_given_gold": float(np.exp(-losses[:, 1]).mean()),
        "p_gold_sequence": float(np.exp(-losses.sum(axis=1)).mean()),
        "answer_nll": float(losses[:, 0].mean()),
        "eos_given_gold_nll": float(losses[:, 1].mean()),
        "two_target_nll": float(losses.mean()),
        "answer_accuracy": float(answer_correct.mean()),
        "complete_accuracy": float((answer_correct & eos_correct).mean()),
        "generated_eos_accuracy": float(eos_correct.mean()),
        "generated_eos_given_correct_answer_accuracy": (
            float(eos_correct[answer_correct].mean()) if answer_n else None
        ),
        "answer_correct_eos_wrong_fraction": float((answer_correct & ~eos_correct).mean()),
    }


def verify_atomic_duplicates(archive, world):
    """ID/OOD scoring uses full atomic arrays; separately saved arrays are audited."""
    lookup = {tuple(map(int, row)): i for i, row in enumerate(world["atomic"])}
    for split in ("id_atomic", "ood_atomic"):
        indices = [lookup[tuple(map(int, row))] for row in world[split]]
        for key in ("answer", "stop", "target", "nll"):
            dedicated, full = split + "_" + key, "atomic_" + key
            if dedicated not in archive or full not in archive:
                raise ValueError(f"Missing duplicate atomic array: {dedicated}")
            left, right = archive[dedicated], archive[full][indices]
            equal = (
                np.allclose(left, right, rtol=1e-5, atol=1e-5)
                if key == "nll"
                else np.array_equal(left, right)
            )
            if left.shape != right.shape or not equal:
                raise ValueError(f"Duplicate atomic predictions disagree: {dedicated}")


def verify_history_score(row, saved, split):
    original = saved.get(split)
    if not original or not row["available"]:
        return
    for old, new in (
        ("n", "population_n"),
        ("answer_accuracy", "answer_accuracy"),
        ("accuracy", "complete_accuracy"),
        ("nll", "two_target_nll"),
    ):
        left, right = original[old], row[new]
        if left is None and right is None:
            continue
        if left is None or right is None or not np.isclose(left, right, rtol=1e-5, atol=1e-6):
            raise ValueError(f"Recount differs from historical {split}.{old}")


def autonomous_hard(archive, target, hops):
    n = len(target)
    required = [
        f"autonomous_hop{hop}_{key}" for hop in range(1, hops + 1) for key in ("answer", "stop")
    ]
    if not n:
        return {"n": 0, "available": True, "answer_accuracy": None, "complete_accuracy": None}
    if any(key not in archive for key in required):
        return {"n": n, "available": False, "answer_accuracy": None, "complete_accuracy": None}
    if any(archive[key].shape != target.shape for key in required):
        raise ValueError("Autonomous arrays not aligned")
    correct = archive[f"autonomous_hop{hops}_answer"] == target
    eos = np.logical_and.reduce(
        [archive[f"autonomous_hop{hop}_stop"] == 1 for hop in range(1, hops + 1)]
    )
    return {
        "n": n,
        "available": True,
        "answer_accuracy": float(correct.mean()),
        "complete_accuracy": float((correct & eos).mean()),
        "continuous_available": False,
        "continuous_missing_reason": "NLL not saved",
    }


def collect(configs, root=ROOT):
    rows, provenance, matrix, errors, autonomous = [], [], [], [], []
    seen = set()
    for config in configs:
        config = Path(config).resolve()
        cfg = read_json(config)
        provenance.append(provenance_record(config, "configuration"))
        for run, override in cfg["runs"].items():
            spec = {**cfg["base"], **override}
            ident = identity(spec, run, config)
            if run in seen:
                raise ValueError("Duplicate registered run")
            seen.add(run)
            directory = root / cfg["output_root"] / spec["phase"] / run
            info = {
                **ident,
                "directory": str(directory.resolve()),
                "expected_nodes": spec["nodes"],
                "present_nodes": [],
                "missing_nodes": [],
                "extra_nodes": [],
                "errors": [],
            }
            matrix.append(info)
            try:
                metadata = read_json(directory / "metadata.json")
                if metadata["spec"] != spec:
                    raise ValueError("Source metadata specification differs from configuration")
                expected_config_hashes = [
                    value
                    for key, value in metadata["files"].items()
                    if Path(key).name == config.name
                ]
                if expected_config_hashes != [digest(config)]:
                    raise ValueError("Configuration hash differs from training metadata")
                history = read_json(directory / "learning.json")
                steps = [item["step"] for item in history]
                if steps != spec["nodes"]:
                    raise ValueError("Learning history nodes differ from registered nodes")
                history = {item["step"]: item for item in history}
                with np.load(directory / "world.npz", allow_pickle=False) as archive:
                    world = {key: archive[key].copy() for key in archive.files}
                masks = atomic_masks(world)
                meta = read_json(directory / "world-metadata.json")
                if dataset_digest(world) != meta["dataset_sha256"]:
                    raise ValueError("Actual symbolic dataset hash differs from metadata")
                indices = np.asarray(meta["probe_indices_in_full_test"], dtype=np.int64)
                if not np.array_equal(
                    world["test_composite"], world["test_full_composite"][indices]
                ):
                    raise ValueError("ID probe differs from its declared full-reserve subset")
                info["dataset_sha256"] = meta["dataset_sha256"]
                info["historical_source_sha256"] = metadata["files"]
                for name in ("metadata.json", "learning.json", "world.npz", "world-metadata.json"):
                    provenance.append(provenance_record(directory / name, "training_source"))
                found = {
                    int(path.stem.split("-")[1]) for path in directory.glob("predictions-*.npz")
                }
                info["extra_nodes"] = sorted(found - set(spec["nodes"]))
            except (OSError, ValueError, KeyError) as exc:
                raise ValueError(f"Cannot validate run {run}: {exc}") from exc
            for step in spec["nodes"]:
                path = directory / f"predictions-{step:07d}.npz"
                saved = history[step]
                common = {
                    **ident,
                    "step": step,
                    "training_flops": saved["estimated_training_flops"],
                    "examples": saved["examples"],
                    "supervised_tokens": saved["supervised_tokens"],
                    "source_predictions": str(path.resolve()),
                    "source_world": str((directory / "world.npz").resolve()),
                }
                archive = {}
                node_error = ""
                if path.exists():
                    provenance.append(provenance_record(path, "predictions"))
                    with np.load(path, allow_pickle=False) as raw:
                        archive = {key: raw[key].copy() for key in raw.files}
                    try:
                        verify_atomic_duplicates(archive, world)
                    except ValueError as exc:
                        node_error = str(exc)
                    info["present_nodes"].append(step)
                else:
                    info["missing_nodes"].append(step)
                    node_error = "prediction archive absent"
                for split in SPLITS + (("test_full_composite",) if step == spec["steps"] else ()):
                    pool = "atomic" if split in masks else split
                    target = world[pool][:, -1]
                    try:
                        score = score_saved(
                            archive, pool, target, masks.get(split), missing_reason=node_error
                        )
                        # Dedicated atomic predictions can use different batch shapes,
                        # but labels, outcomes and NLL are checked above in fact order.
                        verify_history_score(score, saved, split)
                    except ValueError as exc:
                        node_error = str(exc)
                        score = score_saved(
                            {}, pool, target, masks.get(split), missing_reason=node_error
                        )
                    rows.append({**common, "split": split, **score})
                    if not score["available"]:
                        errors.append(
                            {
                                "run": run,
                                "step": step,
                                "split": split,
                                "reason": score["missing_reason"],
                            }
                        )
                if node_error:
                    info["errors"].append({"step": step, "reason": node_error})
                hard = autonomous_hard(archive, world["test_composite"][:, -1], spec["hops"])
                for old, new in (
                    ("answer_accuracy", "answer_accuracy"),
                    ("accuracy", "complete_accuracy"),
                ):
                    value = saved["autonomous_calls"][old]
                    if hard["available"] and value is not None and not np.isclose(value, hard[new]):
                        raise ValueError("Autonomous recount disagrees with history")
                autonomous.append({**common, **hard})
            print(f"Read {run}: {len(spec['nodes'])} saved nodes", flush=True)
    return rows, provenance, matrix, errors, autonomous


def nested_summary(rows):
    """Missing observations never become zeros or independent worlds."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["condition_id"], row["step"], row["split"])].append(row)
    worlds, aggregate = [], []
    for (_, _, _, _), group in groups.items():
        first = group[0]
        base = {
            key: first[key]
            for key in (
                "config",
                "condition_id",
                "phase",
                "architecture",
                "hops",
                "unique_layers",
                "training_repeats",
                "effective_depth",
                "init_scheme",
                "width",
                "phi",
                "step",
                "endpoint_step",
                "split",
                "training_flops",
            )
        }
        by_world = defaultdict(list)
        for row in group:
            by_world[row["world_seed"]].append(row)
        wrows = []
        for world, repeated in sorted(by_world.items()):
            item = {
                **base,
                "world_seed": world,
                "expected_initializations": len(repeated),
                "available_initializations": sum(row["available"] for row in repeated),
            }
            for metric in METRICS:
                values = [row[metric] for row in repeated if row[metric] is not None]
                item[metric] = float(np.mean(values)) if values else None
                item[metric + "_initializations_with_denominator"] = len(values)
            worlds.append(item)
            wrows.append(item)
        item = {
            **base,
            "expected_worlds": len(wrows),
            "registered_runs": len(group),
            "available_runs": sum(row["available"] for row in group),
        }
        for metric in METRICS:
            values = [row[metric] for row in wrows if row[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
            item[metric + "_world_min"] = min(values) if values else None
            item[metric + "_world_max"] = max(values) if values else None
            item[metric + "_worlds_with_denominator"] = len(values)
        aggregate.append(item)
    return worlds, aggregate


def dynamics(rows, tolerance=CHANGE_TOLERANCE):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["run"], row["split"])].append(row)
    adjacent, reversals, summaries = [], [], []
    for (_, _), group in groups.items():
        group.sort(key=lambda row: row["step"])
        base = {
            key: group[0][key]
            for key in (
                "run",
                "phase",
                "world_seed",
                "initialization",
                "architecture",
                "hops",
                "unique_layers",
                "training_repeats",
                "init_scheme",
                "condition_id",
                "split",
            )
        }
        for left, right in zip(group, group[1:], strict=False):
            item = {
                **base,
                "from_step": left["step"],
                "to_step": right["step"],
                "step_gap": right["step"] - left["step"],
                "flop_gap": right["training_flops"] - left["training_flops"],
                "available": left["available"] and right["available"],
            }
            for metric in DYNAMICS_METRICS:
                a, b = left[metric], right[metric]
                item[metric + "_delta"] = b - a if a is not None and b is not None else None
            adjacent.append(item)
        for metric in DYNAMICS_METRICS:
            changes = []
            for left, right in zip(group, group[1:], strict=False):
                a, b = left[metric], right[metric]
                if a is not None and b is not None:
                    changes.append((b - a, left["step"], right["step"]))
            for left, middle, right in zip(group, group[1:], group[2:], strict=False):
                a, b, c = (row[metric] for row in (left, middle, right))
                if any(value is None for value in (a, b, c)):
                    continue
                d1, d2 = b - a, c - b
                if d1 < -tolerance and d2 > tolerance:
                    reversals.append(
                        {
                            **base,
                            "metric": metric,
                            "before_step": left["step"],
                            "trough_step": middle["step"],
                            "after_step": right["step"],
                            "before": a,
                            "trough": b,
                            "after": c,
                            "decline": -d1,
                            "rebound": d2,
                            "decline_step_gap": middle["step"] - left["step"],
                            "rebound_step_gap": right["step"] - middle["step"],
                        }
                    )
            declines = [change for change in changes if change[0] < -tolerance]
            increases = [change for change in changes if change[0] > tolerance]
            largest_drop = min(declines, default=None)
            largest_rise = max(increases, default=None)
            observed = [row for row in group if row[metric] is not None]
            local = [
                row
                for row in reversals
                if row["run"] == base["run"]
                and row["split"] == base["split"]
                and row["metric"] == metric
            ]
            summaries.append(
                {
                    **base,
                    "metric": metric,
                    "expected_nodes": len(group),
                    "observed_nodes": len(observed),
                    "first": observed[0][metric] if observed else None,
                    "endpoint": group[-1][metric],
                    "net_change": observed[-1][metric] - observed[0][metric] if observed else None,
                    "total_variation": sum(abs(change[0]) for change in changes),
                    "decline_count": len(declines),
                    "increase_count": len(increases),
                    "declines_ge_1pp": sum(change[0] <= -0.01 for change in changes),
                    "declines_ge_5pp": sum(change[0] <= -0.05 for change in changes),
                    "decline_rebound_count": len(local),
                    "decline_rebounds_both_ge_1pp": sum(
                        min(r["decline"], r["rebound"]) >= 0.01 for r in local
                    ),
                    "decline_rebounds_both_ge_5pp": sum(
                        min(r["decline"], r["rebound"]) >= 0.05 for r in local
                    ),
                    "largest_decline": -largest_drop[0] if largest_drop else 0.0,
                    "largest_decline_from": largest_drop[1] if largest_drop else None,
                    "largest_decline_to": largest_drop[2] if largest_drop else None,
                    "largest_rebound": largest_rise[0] if largest_rise else 0.0,
                    "largest_rebound_from": largest_rise[1] if largest_rise else None,
                    "largest_rebound_to": largest_rise[2] if largest_rise else None,
                }
            )
    return adjacent, reversals, summaries


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def create_figures(output, rows, world_rows, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    paths = []
    legend = [Line2D([], [], color=color, label=arch.upper()) for arch, color in COLORS.items()]
    legend += [
        Line2D([], [], color="black", label="Mean P(gold answer)"),
        Line2D([], [], color="black", ls="--", label="Answer accuracy"),
    ]

    def finish(fig, name, pdf):
        png, single_pdf = output / (name + ".png"), output / (name + ".pdf")
        fig.savefig(png, dpi=180, bbox_inches="tight")
        fig.savefig(single_pdf, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        paths.extend([str(png.resolve()), str(single_pdf.resolve())])

    def plot_group(ax, selected, metrics=("p_answer", "answer_accuracy"), scale=False, ranges=True):
        for arch in COLORS:
            data = sorted(
                [row for row in selected if row["architecture"] == arch],
                key=lambda row: row["step"],
            )
            if not data:
                continue
            x = [row["step"] / 1000 for row in data]
            for metric, style in zip(metrics, ("-", "--"), strict=True):
                y = [np.nan if row[metric] is None else row[metric] for row in data]
                ax.plot(x, y, color=COLORS[arch], ls=style, lw=1.4)
                if ranges and metric + "_world_min" in data[0]:
                    low = [
                        np.nan if row[metric + "_world_min"] is None else row[metric + "_world_min"]
                        for row in data
                    ]
                    high = [
                        np.nan if row[metric + "_world_max"] is None else row[metric + "_world_max"]
                        for row in data
                    ]
                    ax.fill_between(x, low, high, color=COLORS[arch], alpha=0.10)
        ax.set_xlim(0, 128)
        values = [row[metric] for row in selected for metric in metrics if row[metric] is not None]
        ax.set_ylim(0, min(1.02, max(0.10, max(values, default=1) * 1.08)) if scale else 1.02)
        ax.set_xlabel("Training updates (thousands)")
        ax.set_ylabel("Probability / fraction")
        ax.grid(alpha=0.2)

    with PdfPages(output / "continuous-trajectories.pdf") as pdf:
        confirmation = [row for row in aggregate if row["phase"] == "confirmation"]
        for name, title, splits in (
            (
                "confirmation-composition",
                "Confirmation: composition; bands span world means",
                ("test_composite", "ood_composite"),
            ),
            (
                "confirmation-atomic",
                "Confirmation: atomic facts; both classes trained as single hops",
                ("id_atomic", "ood_atomic"),
            ),
        ):
            fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
            for i, hop in enumerate((2, 3, 4)):
                for j, split in enumerate(splits):
                    selected = [
                        row for row in confirmation if row["hops"] == hop and row["split"] == split
                    ]
                    plot_group(axes[i, j], selected, scale=split == "ood_composite")
                    axes[i, j].set_title(f"{hop}-hop models: {split}")
            fig.suptitle(title)
            fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
            finish(fig, name, pdf)
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
        eos_legend = legend[:5] + [
            Line2D([], [], color="black", label="P(EOS | gold answer)"),
            Line2D([], [], color="black", ls="--", label="Generated EOS accuracy"),
        ]
        for i, hop in enumerate((2, 3, 4)):
            for j, split in enumerate(("test_composite", "ood_composite")):
                selected = [
                    row for row in confirmation if row["hops"] == hop and row["split"] == split
                ]
                plot_group(
                    axes[i, j], selected, metrics=("p_eos_given_gold", "generated_eos_accuracy")
                )
                axes[i, j].set_title(f"{hop}-hop models: {split}")
        fig.suptitle("EOS: gold-answer conditioning and generated-answer stopping are different")
        fig.legend(handles=eos_legend, loc="outside lower center", ncol=4, frameon=False)
        finish(fig, "confirmation-eos", pdf)
        fig, axes = plt.subplots(3, 5, figsize=(14, 9), constrained_layout=True)
        styles = {14501: "-", 14502: "--"}
        worlds = {
            world: color
            for world, color in zip(
                (145011, 145012, 145013), ("#4477aa", "#ee6677", "#228833"), strict=True
            )
        }
        for i, hop in enumerate((2, 3, 4)):
            for j, arch in enumerate(COLORS):
                selected = [
                    row
                    for row in rows
                    if row["phase"] == "confirmation"
                    and row["split"] == "ood_composite"
                    and row["hops"] == hop
                    and row["architecture"] == arch
                ]
                for world, color in worlds.items():
                    for seed, style in styles.items():
                        data = sorted(
                            [
                                row
                                for row in selected
                                if row["world_seed"] == world and row["initialization"] == seed
                            ],
                            key=lambda row: row["step"],
                        )
                        axes[i, j].plot(
                            [row["step"] / 1000 for row in data],
                            [row["p_answer"] for row in data],
                            color=color,
                            ls=style,
                            lw=1,
                        )
                axes[i, j].set_title(f"{hop}-hop {arch.upper()}")
                axes[i, j].set_xlabel("Updates (thousands)")
                axes[i, j].set_ylabel("Mean P(gold answer)")
                axes[i, j].set_xlim(0, 128)
                hop_values = [
                    row["p_answer"]
                    for row in rows
                    if row["phase"] == "confirmation"
                    and row["split"] == "ood_composite"
                    and row["hops"] == hop
                    and row["p_answer"] is not None
                ]
                axes[i, j].set_ylim(0, max(hop_values, default=0.1) * 1.08)
                axes[i, j].grid(alpha=0.2)
        fig.suptitle("Confirmation OOD: all 3 worlds × 2 initializations; no checkpoint selection")
        handles = [
            Line2D([], [], color=color, label=f"World {world}") for world, color in worlds.items()
        ]
        handles += [
            Line2D([], [], color="black", ls=style, label=f"Init {seed}")
            for seed, style in styles.items()
        ]
        fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
        finish(fig, "confirmation-all-ood", pdf)
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
        for i, hop in enumerate((2, 3, 4)):
            for j, split in enumerate(("atomic", "test_composite", "ood_composite")):
                selected = [
                    row
                    for row in aggregate
                    if row["phase"] == "development"
                    and row["hops"] == hop
                    and row["split"] == split
                ]
                plot_group(axes[i, j], selected, scale=split == "ood_composite")
                axes[i, j].set_title(f"{hop}-hop: {split}")
        fig.suptitle("Development: all 15 original runs; one world and initialization")
        fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
        finish(fig, "development-all", pdf)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        sensitivity = [row for row in rows if row["phase"] == "development-sensitivity"]
        recipes = sorted({(row["effective_depth"], row["init_scheme"]) for row in sensitivity})
        for j, (depth, init) in enumerate(recipes):
            for i, split in enumerate(("test_composite", "ood_composite")):
                selected = [
                    row
                    for row in sensitivity
                    if row["effective_depth"] == depth
                    and row["init_scheme"] == init
                    and row["split"] == split
                ]
                plot_group(axes[i, j], selected, scale=split == "ood_composite", ranges=False)
                axes[i, j].set_title(f"D={depth}; {init}; {split}")
        fig.suptitle("Sensitivity: all 6 added runs; independent from formal matrix")
        fig.legend(handles=legend, loc="outside lower center", ncol=4, frameon=False)
        finish(fig, "sensitivity-all", pdf)
    paths.append(str((output / "continuous-trajectories.pdf").resolve()))
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root; required when running the saved source snapshot",
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/development-artifacts/grok-loop-continuous-v1"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registration = {
        "analysis_kind": "exploratory reanalysis of existing predictions",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "configs": [str((root / path).resolve()) for path in args.configs],
        "definitions": {
            "p_answer": "mean_i exp(-saved NLL[i,0]); gold final-answer token, full vocabulary",
            "p_eos_given_gold": "mean_i exp(-saved NLL[i,1]); teacher-forced gold answer",
            "p_gold_sequence": "mean_i exp(-NLL[i,0]-NLL[i,1]); gold answer then EOS",
            "complete_accuracy": (
                "saved greedy answer equals target AND EOS after that generated answer"
            ),
            "generated_eos_accuracy": (
                "saved stop equals EOS id 1; answer correctness is separate"
            ),
            "answer_correct_eos_wrong_fraction": (
                "population fraction with correct answer but missing EOS"
            ),
            "atomic_partition": (
                "complete-fact membership in saved ID/OOD edge partition; both trained"
            ),
            "world_summary": (
                "average initializations within each world; then weight worlds equally"
            ),
            "bands": "range of world means; descriptive, not confidence intervals",
            "decline_rebound": (
                "three consecutive saved nodes; decline then rebound, tolerance 1e-8"
            ),
            "magnitude_counts": (
                "1pp/5pp are descriptive bins; not significance or selection criteria"
            ),
            "resolution": "observed changes span saved update intervals; no between-node inference",
            "missingness": (
                "null for absent or zero-denominator observations; never replaced by zero"
            ),
            "scope": (
                "no predictor, checkpoint selection, forward pass, training or historical rewriting"
            ),
            "id_full": (
                "test_full_composite measured at fixed endpoint only; never mixed with probe"
            ),
            "autonomous": "hard scores only; saved external-call arrays have no NLL",
            "unavailable_margin": (
                "wrong-answer probabilities and logits not saved; "
                "P(correct)-P(incorrect) cannot be reconstructed"
            ),
        },
        "source": provenance_record(__file__, "analysis_source"),
    }
    write_json(output / "analysis-definition.json", registration)
    snapshots = [
        (Path(__file__), "scripts/report_grok_loop_continuous.py"),
        (root / "tests/test_grok_loop_continuous.py", "tests/test_grok_loop_continuous.py"),
        (output / "analysis-definition.json", "analysis-definition.json"),
    ]
    snapshots.extend((root / path, "configs/" + Path(path).name) for path in args.configs)
    for source, relative in snapshots:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    rows, provenance, matrix, errors, autonomous = collect(
        [root / path for path in args.configs], root
    )
    worlds, aggregate = nested_summary(rows)
    adjacent, reversals, trajectory = dynamics(rows)
    endpoints = [row for row in rows if row["step"] == row["endpoint_step"]]
    formal_endpoints = [
        row
        for row in aggregate
        if row["phase"] == "confirmation" and row["step"] == row["endpoint_step"]
    ]
    outputs = {
        "nodes.csv": rows,
        "world-trajectories.csv": worlds,
        "nested-trajectories.csv": aggregate,
        "endpoints.csv": endpoints,
        "formal-nested-endpoints.csv": formal_endpoints,
        "adjacent-changes.csv": adjacent,
        "decline-rebounds.csv": reversals,
        "run-dynamics.csv": trajectory,
        "autonomous-hard.csv": autonomous,
    }
    for name, data in outputs.items():
        write_csv(output / name, data)
    phase_counts = {
        phase: sum(row["phase"] == phase for row in matrix)
        for phase in sorted({row["phase"] for row in matrix})
    }
    audit = {
        "state": "complete"
        if not errors and not any(row["errors"] or row["extra_nodes"] for row in matrix)
        else "incomplete",
        "expected_runs": len(matrix),
        "runs_by_phase": phase_counts,
        "expected_prediction_nodes": sum(len(row["expected_nodes"]) for row in matrix),
        "present_prediction_nodes": sum(len(row["present_nodes"]) for row in matrix),
        "missing_prediction_nodes": sum(len(row["missing_nodes"]) for row in matrix),
        "node_split_rows": len(rows),
        "available_node_split_rows": sum(row["available"] for row in rows),
        "errors": errors,
        "runs": matrix,
        "dedicated_atomic_crosscheck": (
            "outcomes exact; NLL rtol/atol=1e-5 after fact-based alignment"
        ),
        "historical_score_crosscheck": (
            "all populations: counts, answer, complete and two-target NLL"
        ),
        "nonendpoint_full_ID": "not scheduled; deliberately absent from expected split matrix",
    }
    paired_datasets = defaultdict(set)
    for run in matrix:
        paired_datasets[(run["phase"], run["world_seed"], run["initialization"], run["hops"])].add(
            run["dataset_sha256"]
        )
    audit["matched_world_initialization_hop_groups"] = len(paired_datasets)
    audit["paired_dataset_mismatches"] = [
        {"identity": key, "dataset_sha256": sorted(values)}
        for key, values in paired_datasets.items()
        if len(values) != 1
    ]
    if audit["paired_dataset_mismatches"]:
        audit["state"] = "incomplete"
    write_json(output / "matrix-audit.json", audit)
    sources = [
        root / "src/llm_memory_editability/grok_depth.py",
        root / "src/llm_memory_editability/grok_loop_train.py",
        root / "src/llm_memory_editability/grok_loop_data.py",
        root / "docs/development-artifacts/grok-loop-v1/preregistration.md",
        root / "docs/development-artifacts/grok-loop-v1/confirmation-plan.md",
        root / "docs/development-artifacts/grok-loop-v1/completion-manifest.json",
    ]
    sources.extend(
        root / "docs/development-artifacts/grok-loop-v1" / name
        for name in ("development-lock.json", "sensitivity-lock.json", "confirmation-lock.json")
    )
    provenance.extend(provenance_record(path, "scoring_contract_or_source") for path in sources)
    provenance.append(registration["source"])
    provenance.append(
        provenance_record(root / "tests/test_grok_loop_continuous.py", "analysis_contract_tests")
    )
    write_json(output / "input-provenance.json", provenance)
    figures = [] if args.no_plots else create_figures(output, rows, worlds, aggregate)
    summary = {
        "state": audit["state"],
        "analysis_kind": registration["analysis_kind"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            key: audit[key]
            for key in (
                "expected_runs",
                "runs_by_phase",
                "expected_prediction_nodes",
                "present_prediction_nodes",
                "missing_prediction_nodes",
                "node_split_rows",
                "available_node_split_rows",
            )
        },
        "definitions": registration["definitions"],
        "formal_endpoint_world_means": formal_endpoints,
        "phase_dynamics": {},
        "figures": figures,
    }
    for phase in phase_counts:
        summary["phase_dynamics"][phase] = {}
        for split in ("id_atomic", "ood_atomic", "test_composite", "ood_composite"):
            summary["phase_dynamics"][phase][split] = {}
            for metric in ("p_answer", "answer_accuracy", "complete_accuracy"):
                selected = [
                    row
                    for row in trajectory
                    if row["phase"] == phase and row["split"] == split and row["metric"] == metric
                ]
                summary["phase_dynamics"][phase][split][metric] = {
                    "runs": len(selected),
                    "runs_with_adjacent_decline_ge_1pp": sum(
                        row["declines_ge_1pp"] > 0 for row in selected
                    ),
                    "runs_with_decline_rebound_both_ge_1pp": sum(
                        row["decline_rebounds_both_ge_1pp"] > 0 for row in selected
                    ),
                    "runs_with_decline_rebound_both_ge_5pp": sum(
                        row["decline_rebounds_both_ge_5pp"] > 0 for row in selected
                    ),
                    "largest_observed_adjacent_decline": max(
                        (row["largest_decline"] for row in selected), default=None
                    ),
                    "caution": (
                        "counts describe saved trajectories, not independent worlds or modes"
                    ),
                }
    write_json(output / "summary.json", summary)
    output_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "output-manifest.json" and path.suffix != ".log"
    ]
    write_json(
        output / "output-manifest.json",
        [provenance_record(path, "analysis_output") for path in sorted(output_files)],
    )
    print(json.dumps({"state": audit["state"], "output": str(output), "counts": summary["counts"]}))
    return 0 if audit["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
