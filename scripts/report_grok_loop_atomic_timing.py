#!/usr/bin/env python3
"""Exploratory ID/OOD atomic reanalysis of saved recurrence predictions only.

No torch/model imports, forward passes, model selection, or fitted predictors.
Input worlds, scans and checkpoints are never modified. Every R=1..8 is kept,
including explicit missing observations. ID/OOD mean composition experience;
both fact classes were included in atomic training.
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
REPEATS = tuple(range(1, 9))
SPLITS = ("id_atomic", "ood_atomic", "test_composite", "ood_composite")
SEEDS = {"world_seed", "initialization", "stream_seed"}
METRICS = (
    "answer_accuracy",
    "complete_accuracy",
    "prediction_coverage",
    "population_n",
    "partition_fraction",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def identity(spec, run, step):
    condition = {k: v for k, v in spec.items() if k not in SEEDS}
    figure_recipe = {
        k: v
        for k, v in condition.items()
        if k
        not in {
            "architecture",
            "layers",
            "repeats",
            "hops",
            "phi",
        }
    }
    return {
        "run": run,
        "phase": spec["phase"],
        "world_seed": spec["world_seed"],
        "initialization": spec["initialization"],
        "architecture": spec["architecture"],
        "hops": spec["hops"],
        "unique_layers": spec["layers"],
        "training_repeats": spec["repeats"],
        "init_scheme": spec["init_scheme"],
        "width": spec["width"],
        "checkpoint_step": step,
        "condition_id": hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[
            :16
        ],
        "figure_recipe_id": hashlib.sha256(
            json.dumps(figure_recipe, sort_keys=True).encode()
        ).hexdigest()[:16],
    }


def atomic_masks(world):
    """Match complete facts, rather than assuming ID/OOD occupy contiguous rows."""
    atoms = [tuple(map(int, row)) for row in world["atomic"]]
    atom_set = set(atoms)
    ids = {tuple(map(int, row)) for row in world["id_atomic"]}
    oods = {tuple(map(int, row)) for row in world["ood_atomic"]}
    if len(atom_set) != len(atoms) or ids & oods or ids | oods != atom_set:
        raise ValueError("Saved ID/OOD atomic facts are not an exact disjoint partition")
    if len(ids) != len(world["id_atomic"]) or len(oods) != len(world["ood_atomic"]):
        raise ValueError("Duplicate facts in an atomic partition")
    return {
        "id_atomic": np.asarray([row in ids for row in atoms], dtype=bool),
        "ood_atomic": np.asarray([row in oods for row in atoms], dtype=bool),
    }


def recount(archive, prefix, target, selected, *, declared=True):
    """Preserve a missing observation; reject corrupted or misaligned arrays."""
    n = int(selected.sum())
    names = {key: prefix + "_" + key for key in ("answer", "stop", "target")}
    missing = [name for name in names.values() if name not in archive]
    base = {
        "population_n": n,
        "population_pool_n": len(target),
        "partition_fraction": n / len(target) if len(target) else None,
    }
    if not n:
        return {
            **base,
            "available": True,
            "prediction_n": 0,
            "prediction_coverage": None,
            "answer_accuracy": None,
            "complete_accuracy": None,
            "missing_reason": "empty population",
        }
    if missing or not declared:
        return {
            **base,
            "available": False,
            "prediction_n": 0,
            "prediction_coverage": 0.0,
            "answer_accuracy": None,
            "complete_accuracy": None,
            "missing_reason": "repeat absent from scan summary"
            if not declared
            else "; ".join(missing),
        }
    answer, stop, labels = (archive[names[key]] for key in ("answer", "stop", "target"))
    if (
        answer.shape != target.shape
        or stop.shape != target.shape
        or not np.array_equal(labels, target)
    ):
        raise ValueError(f"Saved predictions are not aligned with the world: {prefix}")
    return {
        **base,
        "available": True,
        "prediction_n": n,
        "prediction_coverage": 1.0,
        "answer_accuracy": float((answer[selected] == target[selected]).mean()),
        "complete_accuracy": float(((answer == target) & (stop == 1))[selected].mean()),
        "missing_reason": "",
    }


def expected_runs(configs, phases=None):
    result = []
    for path in configs:
        cfg = read_json(path)
        for run, override in cfg["runs"].items():
            spec = {**cfg["base"], **override}
            if spec["architecture"] not in ("l1", "l2") or (phases and spec["phase"] not in phases):
                continue
            result.append({**identity(spec, run, spec["steps"]), "config": str(path)})
    return result


def scan_paths(inputs):
    paths = set()
    for item in inputs:
        item = Path(item)
        if item.is_file():
            paths.add(item.resolve())
        elif item.is_dir():
            if (item / "summary.json").exists():
                paths.add((item / "summary.json").resolve())
            paths.update(path.resolve() for path in item.glob("**/recurrence-*/summary.json"))
    return sorted(paths)


def collect(inputs, phases=None, root=ROOT):
    rows, provenance, pending = [], [], []
    for path in scan_paths(inputs):
        summary = read_json(path)
        if "finished_utc" not in summary:
            pending.append(str(path))
            continue
        cfg = read_json(root / summary["config"])
        for run in summary["runs"]:
            spec = run["spec"]
            if spec["architecture"] not in ("l1", "l2") or (phases and spec["phase"] not in phases):
                continue
            registered = {**cfg["base"], **cfg["runs"][run["run"]]}
            if registered != spec:
                raise ValueError(
                    "Recurrence specification differs from supplied source configuration"
                )
            directory = root / cfg["output_root"] / spec["phase"] / run["run"]
            metadata = read_json(directory / "metadata.json")
            if metadata["spec"] != spec:
                raise ValueError("Recurrence specification differs from source training metadata")
            steps = {
                int(match.group(1))
                for name in run["checkpoint"]
                if (match := re.search(r"weights-(\d+)\.pt$", name))
            }
            if len(steps) != 1:
                raise ValueError("Scan must identify exactly one checkpoint step")
            step = steps.pop()
            for name, expected in run["checkpoint"].items():
                if digest(name) != expected:
                    raise ValueError("Saved checkpoint hash differs from scan evidence")
            with np.load(directory / "world.npz", allow_pickle=False) as saved:
                world = {key: saved[key].copy() for key in saved.files}
            masks = atomic_masks(world)
            raw_path = path.parent / (run["run"] + ".npz")
            with (
                np.load(raw_path, allow_pickle=False)
                if raw_path.exists()
                else _empty_archive() as saved
            ):
                archive = (
                    {key: saved[key] for key in saved.files} if hasattr(saved, "files") else saved
                )
                measurements = {item["repeats"]: item for item in run["measurements"]}
                if len(measurements) != len(run["measurements"]):
                    raise ValueError("Duplicate repeat count within one scan")
                for repeat in REPEATS:
                    measured = measurements.get(repeat)
                    for split in SPLITS:
                        atomic = split.endswith("atomic")
                        pool = "atomic" if atomic else split
                        target = world[pool][:, -1]
                        selected = masks[split] if atomic else np.ones(len(target), dtype=bool)
                        result = recount(
                            archive,
                            f"r{repeat}_{pool}",
                            target,
                            selected,
                            declared=measured is not None,
                        )
                        rows.append(
                            {
                                **identity(spec, run["run"], step),
                                "evaluated_repeats": repeat,
                                "split": split,
                                "kind": "atomic" if atomic else "composite",
                                **result,
                            }
                        )
                    if measured:
                        for pool in ("atomic", "test_composite", "ood_composite"):
                            target = world[pool][:, -1]
                            check = recount(
                                archive,
                                f"r{repeat}_{pool}",
                                target,
                                np.ones(len(target), dtype=bool),
                            )
                            if check["available"]:
                                original = measured[pool]
                                if original["n"] != len(target) or any(
                                    original[key] != check[new]
                                    for key, new in (
                                        ("answer_accuracy", "answer_accuracy"),
                                        ("accuracy", "complete_accuracy"),
                                    )
                                ):
                                    raise ValueError(
                                        "Raw predictions disagree with original scan scores"
                                    )
            hashes = {
                str(p): digest(p)
                for p in (
                    path,
                    root / summary["config"],
                    directory / "metadata.json",
                    directory / "world.npz",
                    directory / "world-metadata.json",
                )
            }
            if raw_path.exists():
                hashes[str(raw_path)] = digest(raw_path)
            world_meta = read_json(directory / "world-metadata.json")
            scan_provenance = run.get("provenance", {})
            for key, actual in (
                ("world_file_sha256", hashes[str(directory / "world.npz")]),
                ("dataset_sha256", world_meta["dataset_sha256"]),
            ):
                if key in scan_provenance and scan_provenance[key] != actual:
                    raise ValueError(f"World differs from recorded scan evidence: {key}")
            provenance.append(
                {
                    **identity(spec, run["run"], step),
                    "artifact_hashes": hashes,
                    "checkpoint_hashes": run["checkpoint"],
                    "scan_source_hashes": summary.get("source", {}),
                    "dataset_sha256": world_meta["dataset_sha256"],
                    "historical_scan_world_hash_verified": "world_file_sha256" in scan_provenance,
                    "source_scan_finished_utc": summary["finished_utc"],
                    "prediction_file_exists": raw_path.exists(),
                }
            )
    return rows, provenance, pending


class _empty_archive:
    def __enter__(self):
        return {}

    def __exit__(self, *args):
        return False


def summarize(rows, expected):
    """Average initializations within a world, then give worlds equal weight."""
    unique = {}
    for row in rows:
        key = tuple(
            row[k]
            for k in (
                "condition_id",
                "checkpoint_step",
                "world_seed",
                "initialization",
                "evaluated_repeats",
                "split",
            )
        )
        if key in unique and any(
            unique[key][k] != row[k] for k in (*METRICS, "available", "missing_reason")
        ):
            raise ValueError("Conflicting repeated scans for one checkpoint and repeat")
        unique.setdefault(key, row)
    rows = list(unique.values())
    worlds = defaultdict(list)
    for row in rows:
        worlds[
            row["condition_id"],
            row["checkpoint_step"],
            row["world_seed"],
            row["evaluated_repeats"],
            row["split"],
        ].append(row)
    world_rows = []
    for members in worlds.values():
        value = {
            k: v
            for k, v in members[0].items()
            if k
            not in {
                "run",
                "initialization",
                "available",
                "missing_reason",
                "prediction_n",
                *METRICS,
            }
        }
        value["initializations"] = sorted(row["initialization"] for row in members)
        value["available_initialization_ids"] = sorted(
            row["initialization"] for row in members if row["available"]
        )
        value["available_initializations"] = sum(row["available"] for row in members)
        for metric in METRICS:
            available = [row[metric] for row in members if row[metric] is not None]
            value[metric] = float(np.mean(available)) if available else None
        world_rows.append(value)
    groups = defaultdict(list)
    for row in world_rows:
        groups[
            row["condition_id"], row["checkpoint_step"], row["evaluated_repeats"], row["split"]
        ].append(row)
    wanted = defaultdict(set)
    for item in expected:
        wanted[item["condition_id"], item["checkpoint_step"]].add(
            (item["world_seed"], item["initialization"])
        )
    grouped = []
    for members in groups.values():
        first = members[0]
        value = {
            k: v
            for k, v in first.items()
            if k
            not in {
                "world_seed",
                "initializations",
                "available_initializations",
                "available_initialization_ids",
                *METRICS,
            }
        }
        value["worlds"] = sorted(row["world_seed"] for row in members)
        expected_pairs = wanted.get((first["condition_id"], first["checkpoint_step"]))
        observed = {
            (row["world_seed"], seed)
            for row in members
            for seed in row["available_initialization_ids"]
        }
        value["expected_worlds"] = len({w for w, _ in expected_pairs}) if expected_pairs else None
        value["available_worlds"] = len({w for w, _ in observed})
        value["available_model_count"] = len(observed)
        value["expected_model_count"] = len(expected_pairs) if expected_pairs else None
        value["registered_model_coverage"] = (
            len(expected_pairs & observed) / len(expected_pairs) if expected_pairs else None
        )
        value["missing_world_initializations"] = (
            sorted(expected_pairs - observed) if expected_pairs else None
        )
        value["metrics"] = {}
        for metric in METRICS:
            available = [row[metric] for row in members if row[metric] is not None]
            value["metrics"][metric] = {
                "mean": float(np.mean(available)) if available else None,
                "min": min(available) if available else None,
                "max": max(available) if available else None,
                "worlds_with_denominator": len(available),
            }
        grouped.append(value)
    missing = []
    observed_rows = {
        (
            row["condition_id"],
            row["checkpoint_step"],
            row["world_seed"],
            row["initialization"],
            row["evaluated_repeats"],
            row["split"],
        )
        for row in rows
        if row["available"]
    }
    for item in expected:
        for repeat in REPEATS:
            absent = [
                split
                for split in SPLITS
                if (
                    item["condition_id"],
                    item["checkpoint_step"],
                    item["world_seed"],
                    item["initialization"],
                    repeat,
                    split,
                )
                not in observed_rows
            ]
            if absent:
                missing.append({**item, "evaluated_repeats": repeat, "missing_splits": absent})
    return rows, world_rows, grouped, missing


def save_csv(path, rows):
    flattened = []
    for row in rows:
        result = {k: v for k, v in row.items() if k != "metrics"}
        for name, stats in row.get("metrics", {}).items():
            result.update({name + "_" + key: value for key, value in stats.items()})
        flattened.append(
            {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in result.items()}
        )
    fields = list(dict.fromkeys(k for row in flattened for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def plot(groups, expected, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    facets = defaultdict(list)
    for row in groups:
        if row["kind"] == "atomic":
            facets[row["phase"], row["figure_recipe_id"], row["checkpoint_step"]].append(row)
    paths = []
    for (phase, recipe, step), members in sorted(facets.items()):
        fig, axes = plt.subplots(3, 2, figsize=(11, 10), squeeze=False, constrained_layout=True)
        for i, hops in enumerate((2, 3, 4)):
            for j, arch in enumerate(("l1", "l2")):
                axis = axes[i, j]
                cell = [r for r in members if r["hops"] == hops and r["architecture"] == arch]
                conditions = sorted({r["condition_id"] for r in cell})
                for condition_index, condition in enumerate(conditions):
                    selected = [r for r in cell if r["condition_id"] == condition]
                    trained = selected[0]["training_repeats"]
                    for split, color, label in (
                        ("id_atomic", "#2878B5", "ID atoms"),
                        ("ood_atomic", "#D77A20", "OOD atoms"),
                    ):
                        points = sorted(
                            (r for r in selected if r["split"] == split),
                            key=lambda r: r["evaluated_repeats"],
                        )
                        x = [r["evaluated_repeats"] for r in points]
                        for metric, style, metric_label in (
                            ("answer_accuracy", "-", "answer"),
                            ("complete_accuracy", "--", "answer + EOS"),
                        ):
                            stats = [r["metrics"][metric] for r in points]
                            values = [
                                [100 * s[key] if s[key] is not None else np.nan for s in stats]
                                for key in ("mean", "min", "max")
                            ]
                            legend = f"{label}: {metric_label}" + (
                                f"; trained R={trained}" if len(conditions) > 1 else ""
                            )
                            axis.plot(
                                x,
                                values[0],
                                color=color,
                                linestyle=style,
                                marker=("o", "s", "^")[condition_index % 3],
                                markersize=3,
                                label=legend,
                            )
                            axis.fill_between(x, values[1], values[2], color=color, alpha=0.09)
                    axis.axvline(
                        trained,
                        color="#777777",
                        linestyle=":",
                        linewidth=1,
                        label=f"Training R={trained}",
                    )
                if cell:
                    nw = sorted({r["available_worlds"] for r in cell})
                    ne = cell[0]["expected_worlds"]
                    ni = sorted({r["available_model_count"] for r in cell})
                    nie = cell[0]["expected_model_count"]
                    axis.text(
                        0.02,
                        0.06,
                        f"Worlds {min(nw)}–{max(nw)}/{ne if ne is not None else '?'}; "
                        f"models {min(ni)}–{max(ni)}/{nie if nie is not None else '?'}",
                        transform=axis.transAxes,
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
                else:
                    axis.text(0.5, 0.5, "No completed scan", ha="center", transform=axis.transAxes)
                axis.set_title(f"{hops}-hop model | {arch.upper()} | {members[0]['init_scheme']}")
                axis.set_xticks(REPEATS)
                axis.set_ylim(-2, 102)
                axis.set_xlabel("Evaluation repeats at fixed weights")
                axis.set_ylabel("Atomic accuracy (%)")
                axis.grid(axis="y", alpha=0.16)
                axis.spines[["top", "right"]].set_visible(False)
        handles, labels = [], []
        for axis in axes.flat:
            hs, ls = axis.get_legend_handles_labels()
            for handle, label in zip(hs, ls, strict=True):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        if handles:
            fig.legend(
                handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9
            )
        fig.suptitle(
            f"Exploratory reanalysis | {phase} | checkpoint {step:,}\n"
            "Saved predictions only; seeds within world, then equal worlds; band = world range",
            fontsize=11,
        )
        for suffix in ("png", "pdf"):
            path = out / f"atomic-timing-{phase}-s{step}-{recipe[:8]}.{suffix}"
            fig.savefig(path, dpi=180)
            paths.append(str(path))
        plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[ROOT / "results/grok-loop-v1"])
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=[
            ROOT / "configs/grok-loop-development-v1.json",
            ROOT / "configs/grok-loop-confirmation-v1.json",
        ],
    )
    parser.add_argument("--phases", nargs="+")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/development-artifacts/grok-loop-v1/atomic-timing-report",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "docs/development-artifacts/grok-loop-v1/atomic-timing-registration.json",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    registration = read_json(args.registration)
    rows, provenance, pending = collect(args.inputs, args.phases)
    expected = expected_runs(args.configs, args.phases)
    rows, world_rows, groups, missing = summarize(rows, expected)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("model-results", rows),
        ("world-results", world_rows),
        ("world-summary", groups),
        ("missing", missing),
    ):
        save_csv(args.out / f"{name}.csv", values)
    figures = [] if args.no_plots else plot(groups, expected, args.out)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "exploratory reanalysis",
        "registration": registration,
        "registration_sha256": digest(args.registration),
        "source_sha256": digest(Path(__file__)),
        "source_file": str(Path(__file__).resolve()),
        "new_forward_passes": 0,
        "new_training_steps": 0,
        "repeat_counts": list(REPEATS),
        "selection": (
            "All available L1/L2 scans and every R=1..8; "
            "no best-R, threshold or successful-model filtering."
        ),
        "statistical_unit": (
            "Means over initializations within each world, then equal world means. "
            "Bands are world ranges, not confidence intervals."
        ),
        "interpretation": (
            "Output-level atomic extractability versus composite success; neither "
            "internal-encoding localization nor proof of MLP-exclusive knowledge storage."
        ),
        "models_observed": len(
            {
                (r["condition_id"], r["world_seed"], r["initialization"], r["checkpoint_step"])
                for r in rows
            }
        ),
        "models_expected": len(expected),
        "missing_model_repeat_groups": len(missing),
        "all_registered_model_repeat_groups_available": not missing if expected else None,
        "pending_scan_summaries": pending,
        "missing": missing,
        "groups": groups,
        "provenance": provenance,
        "figures": figures,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "models_observed",
                    "models_expected",
                    "missing_model_repeat_groups",
                    "new_forward_passes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
