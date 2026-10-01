#!/usr/bin/env python3
"""Annotate stored bridge donors by graph split without modifying frozen results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train_composite", "test_composite", "unused_composite")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def split_labels(world, arrays):
    """Derive split identities from saved donor prefixes and the original graph."""
    rows = arrays["original_rows"]
    require(np.array_equal(rows, world["test_composite"]), "original test rows differ")
    lookup = {tuple(row): name for name in SPLITS for row in world[name]}
    same = np.full(len(rows), "missing", dtype="U24")
    different = np.full(len(rows), "missing", dtype="U24")
    for i, (_, _, r2, tail) in enumerate(rows):
        if arrays["same_valid"][i]:
            h, r1, bridge = arrays["same_donor"][i]
            require(bridge == arrays["original_bridge"][i], "same donor changed the bridge")
            same[i] = lookup[(h, r1, r2, tail)]
        if arrays["different_valid"][i]:
            different[i] = lookup[tuple(arrays["counterfactual_rows"][i])]
            require(different[i] != "train_composite", "different donor counterfactual trained")
    return same, different


def stratum_scores(arrays, same_split):
    """Report original-answer retention in each prespecified donor identity stratum."""
    valid = arrays["same_valid"]
    relation = arrays["same_relation_control"]
    same_r1 = relation == "same_r1"
    other_r1 = relation == "different_r1"
    train = same_split == "train_composite"
    held_out = np.isin(same_split, ["test_composite", "unused_composite"])
    strata = {
        "all_same_donors": valid,
        "same_counterfactual_train": valid & train,
        "same_counterfactual_held_out": valid & held_out,
        "same_counterfactual_test": valid & (same_split == "test_composite"),
        "same_counterfactual_unused": valid & (same_split == "unused_composite"),
        "same_r1_train": valid & same_r1 & train,
        "same_r1_held_out": valid & same_r1 & held_out,
        "different_r1_train": valid & other_r1 & train,
        "different_r1_held_out": valid & other_r1 & held_out,
    }
    original = arrays["original_rows"][:, -1]
    result = {}
    for name, mask in strata.items():
        n = int(mask.sum())
        metrics = {"n": n, "total_test_n": len(mask), "coverage": float(mask.mean())}
        for condition in ("baseline", "same_bridge_r1"):
            answer_correct = arrays[condition + "_answer"] == original
            complete_correct = answer_correct & (arrays[condition + "_stop"] == 1)
            metrics[condition + "_original_answer_accuracy"] = (
                float(answer_correct[mask].mean()) if n else None
            )
            metrics[condition + "_original_complete_accuracy"] = (
                float(complete_correct[mask].mean()) if n else None
            )
        result[name] = metrics
    return result


def aggregate_worlds(runs):
    """Average initializations within each world, then give equal weight to worlds."""
    groups = defaultdict(list)
    for run in runs:
        for stratum, score in run["strata"].items():
            key = (run["layers"], run["width"], run["step"], stratum)
            groups[key].append((run["world_seed"], score))
    output = []
    metric_names = (
        "baseline_original_complete_accuracy",
        "same_bridge_r1_original_complete_accuracy",
    )
    for (layers, width, step, stratum), values in sorted(groups.items()):
        grouped = defaultdict(list)
        for world, score in values:
            grouped[world].append(score)
        world_scores = []
        for world, rows in sorted(grouped.items()):
            world_scores.append(
                {
                    "world_seed": world,
                    "initializations": len(rows),
                    "stratum_n": rows[0]["n"],
                    **{
                        key: float(np.mean([row[key] for row in rows]))
                        if all(row[key] is not None for row in rows)
                        else None
                        for key in metric_names
                    },
                }
            )
        output.append(
            {
                "layers": layers,
                "width": width,
                "step": step,
                "stratum": stratum,
                "worlds": world_scores,
                "metrics": {
                    key: {
                        "world_n": len(available),
                        "mean": float(np.mean(available)) if available else None,
                        "min": min(available) if available else None,
                        "max": max(available) if available else None,
                    }
                    for key in metric_names
                    for available in [[row[key] for row in world_scores if row[key] is not None]]
                },
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", default="configs/grok-depth-bridge-v1.json")
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    source_root = ROOT / config["source_root"]
    phase_dir = ROOT / config["output_root"] / args.phase
    batch = read_json(phase_dir / "batch-summary.json")
    require(batch["state"] == "complete", "annotation requires a complete batch")
    expected = {
        (item["run_id"], int(step))
        for item in config[args.phase + "_runs"]
        for step in item["steps"]
    }
    actual = {(row["run_id"], row["step"]) for row in batch["runs"]}
    require(actual == expected and len(actual) == len(batch["runs"]), "batch matrix differs")
    out = (
        ROOT / args.output_prefix
        if args.output_prefix
        else ROOT / f"docs/development-artifacts/grok-depth-bridge-v1/donor-splits-{args.phase}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = out.with_suffix(".csv"), out.with_suffix(".json")
    require(not csv_path.exists() and not json_path.exists(), "refusing to overwrite annotations")
    worlds, runs, csv_rows, provenance = {}, [], [], []
    for registered in batch["runs"]:
        run_id, step = registered["run_id"], registered["step"]
        summary_path = phase_dir / run_id / f"step-{step:07d}" / "summary.json"
        summary = read_json(summary_path)
        require(summary == registered, f"run summary differs from batch: {summary_path}")
        pred_path = summary_path.with_name("predictions.npz")
        require(digest(pred_path) == summary["predictions_sha256"], "prediction file hash mismatch")
        with np.load(pred_path, allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
        source_dir = source_root / args.phase / run_id
        world_path = source_dir / "world.npz"
        require(digest(world_path) == summary["source"]["world_file_sha256"], "world hash mismatch")
        with np.load(world_path, allow_pickle=False) as stored:
            world = {key: stored[key].copy() for key in stored.files}
        same, different = split_labels(world, arrays)
        world_seed = summary["spec"]["world_seed"]
        shared = {
            "original_rows": arrays["original_rows"],
            "same_donor": arrays["same_donor"],
            "different_donor": arrays["different_donor"],
            "same_split": same,
            "different_split": different,
        }
        if world_seed in worlds:
            for key, array in shared.items():
                require(
                    np.array_equal(array, worlds[world_seed][key]), "donors differ within world"
                )
        else:
            worlds[world_seed] = shared
            for i, row in enumerate(arrays["original_rows"]):
                csv_rows.append(
                    {
                        "world_seed": world_seed,
                        "test_index": i,
                        **dict(zip(("h", "r1", "r2", "t"), map(int, row), strict=True)),
                        "b": int(arrays["original_bridge"][i]),
                        **dict(
                            zip(
                                ("same_h", "same_r1", "same_b"),
                                map(int, arrays["same_donor"][i]),
                                strict=True,
                            )
                        ),
                        "same_relation_control": str(arrays["same_relation_control"][i]),
                        "same_counterfactual_split": str(same[i]),
                        "different_counterfactual_split": str(different[i]),
                    }
                )
        runs.append(
            {
                "run_id": run_id,
                "step": step,
                "world_seed": world_seed,
                "initialization": summary["spec"]["initialization"],
                "layers": summary["spec"]["layers"],
                "width": summary["spec"]["width"],
                "strata": stratum_scores(arrays, same),
            }
        )
        provenance.append(
            {"path": str(summary_path.relative_to(ROOT)), "sha256": digest(summary_path)}
        )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    result = {
        "purpose": (
            "Graph-only donor identity annotation and descriptive same-bridge stratification"
        ),
        "phase": args.phase,
        "config_sha256": digest(ROOT / args.config),
        "script_sha256": digest(Path(__file__)),
        "batch_summary_sha256": digest(phase_dir / "batch-summary.json"),
        "csv_sha256": digest(csv_path),
        "source_summaries": provenance,
        "worlds": {
            str(seed): {
                "total_test_n": len(data["original_rows"]),
                "same_counterfactual_split": dict(Counter(data["same_split"].tolist())),
                "different_counterfactual_split": dict(Counter(data["different_split"].tolist())),
            }
            for seed, data in worlds.items()
        },
        "runs": runs,
        "aggregates": aggregate_worlds(runs),
        "interpretation": (
            "This records an existing graph attribute; it changes no donor or evaluation rule. "
            "Within-stratum rates are descriptive and always retain stratum sizes. "
            "Initializations and checkpoints are repeated measurements, not independent worlds."
        ),
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"csv": str(csv_path), "summary": str(json_path)}))


if __name__ == "__main__":
    main()
