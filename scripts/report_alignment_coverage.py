#!/usr/bin/env python3
"""Report the complete, audited four-arm geometric-supervision comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = {
    "none": ("none", "batch_mean"),
    "composition_batch": ("composition_only", "batch_mean"),
    "composition_selected": ("composition_only", "selected_mean"),
    "all_batch": ("all_atomic_and_composition", "batch_mean"),
}
STRATA = ("common_atomic", "train_composite", "anchor_atomic")
SPLITS = (*STRATA, "familiar_test", "strict_test")
CONTRASTS = (
    ("all_batch", "composition_batch"),
    ("all_batch", "composition_selected"),
    ("all_batch", "none"),
    ("composition_batch", "none"),
    ("composition_selected", "none"),
    ("composition_selected", "composition_batch"),
)
ARM_FIELDS = {"arm", "alignment_coverage", "alignment_normalization", "name"}
IDENTITY = ("spec", "source", "implementation_sha256", "world_sha256", "initial_model_sha256")
COMMON_EXPOSURE = (
    "examples",
    "supervised_tokens",
    "auxiliary_target_presentations",
    "composition_epochs",
    "estimated_matmul_training_flops",
    "sample_stream_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_metrics(left, right):
    require(left.keys() == right.keys(), "Metric split schema changed")
    for split in left:
        require(left[split].keys() == right[split].keys(), f"Metric schema changed: {split}")
        for key, value in left[split].items():
            other = right[split][key]
            matches = (
                value is other
                if value is None or other is None
                else math.isclose(value, other, rel_tol=1e-5, abs_tol=1e-5)
            )
            require(matches, f"Endpoint/audit metric mismatch: {split}.{key}")


def planned_specs(config, phase):
    require(config["module"] == "alignment_coverage", "Not a coverage configuration")
    worlds, initializations = (
        ([810011], [811011])
        if phase == "development"
        else ([810101, 810102, 810103], [811101, 811102])
    )
    expected = {(w, i, a) for w in worlds for i in initializations for a in ARMS}
    specs = config["specs"]
    keys = [(s["world"], s["initialization"], s["arm"]) for s in specs]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete planned matrix")
    require(len({s["name"] for s in specs}) == len(specs), "Duplicate planned run names")
    for spec in specs:
        require(spec["phase"] == phase, "Configuration phase mismatch")
        require(
            (spec["alignment_coverage"], spec["alignment_normalization"]) == ARMS[spec["arm"]],
            "Arm coverage/normalization changed",
        )
        require(
            spec["bridge_weight"] == spec["alignment_weight"] == 0.3
            and spec["batch_size"] == 192
            and spec["steps"] == 16000,
            "The predeclared common objective or training budget changed",
        )
    for world in worlds:
        for initialization in initializations:
            pair = [
                s for s in specs if (s["world"], s["initialization"]) == (world, initialization)
            ]
            shared = [{k: v for k, v in s.items() if k not in ARM_FIELDS} for s in pair]
            require(all(s == shared[0] for s in shared), "Paired configurations differ beyond arm")
    return specs


def geometry_counts(spec, amounts):
    coverage = spec["alignment_coverage"]
    result = {
        name: amount
        if coverage == "all_atomic_and_composition"
        or (coverage == "composition_only" and name == "train_composite")
        else 0
        for name, amount in zip(STRATA, amounts, strict=True)
    }
    return {**result, "total": sum(result.values())}


def exposure_hash(path, sizes, presentations):
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {f"stratum{j}" for j in range(3)}, "Exposure schema mismatch")
        for j, size in enumerate(sizes):
            counts = arrays[f"stratum{j}"]
            require(
                counts.shape == (size,)
                and counts.dtype.kind in "iu"
                and (counts >= 0).all()
                and int(counts.sum()) == presentations,
                f"Wrong CE/bridge exposure: {path}, stratum {j}",
            )
            digest.update(np.asarray(counts, dtype="<i8").tobytes())
    return digest.hexdigest()


def load_run(path, spec, source=None):
    run, complete, audit, history, checkpoints = (
        read_json(path / name)
        for name in ("run.json", "complete.json", "audit.json", "learning.json", "checkpoints.json")
    )
    require(run["spec"] == complete["spec"] == spec, f"Run/config mismatch: {path}")
    require(all(run[k] == complete[k] for k in IDENTITY), f"Identity mismatch: {path}")
    if source is not None:
        require(run["source"] == source, f"Run/frozen source mismatch: {path}")
    require(audit.get("passed") is True, f"Missing successful audit: {path}")
    nodes = spec["nodes"]
    require(nodes[0] == 0 and nodes[-1] == spec["steps"], "Missing fixed endpoint")
    require(
        [r["step"] for r in history]
        == [r["step"] for r in audit["nodes"]]
        == [r["step"] for r in checkpoints]
        == complete["nodes_saved"]
        == nodes,
        f"Incomplete node artifacts: {path}",
    )
    require(all(r["passed"] is True for r in audit["nodes"]), f"Failed node audit: {path}")
    require(audit["nodes"][-1]["predictions_recomputed"] is True, "Endpoint was not reloaded")
    require(
        complete["model_sha256"] == checkpoints[-1]["sha256"] == sha256(path / "model.pt"),
        f"Endpoint checkpoint changed: {path}",
    )
    require(audit["world_sha256"] == run["world_sha256"], "Audit/world hash mismatch")
    require(run["bridge_ce_coverage"] == "all_atomic_and_composition", "Bridge CE coverage changed")
    require(run["geometry_normalization"] == spec["alignment_normalization"], "Reduction changed")
    sizes = [run["strata"][name] for name in STRATA]
    require(all(size > 0 for size in sizes), "Empty training stratum")
    require(
        run["geometry_unique_examples"] == geometry_counts(spec, sizes), "Geometry mask changed"
    )
    selected = geometry_counts(spec, [64] * 3)["total"]
    denominator = max(selected, 1) if spec["alignment_normalization"] == "selected_mean" else 192
    common = []
    for row in history:
        step = row["step"]
        require(
            row["examples"] == row["auxiliary_target_presentations"] == step * 192, "CE exposure"
        )
        require(row["supervised_tokens"] == step * 64 * 23, "Full-token CE exposure changed")
        require(row["composition_epochs"] == step * 64 / sizes[1], "Composition exposure changed")
        require(
            row["geometry_target_presentations"] == geometry_counts(spec, [step * 64] * 3),
            "Geometry target presentations changed",
        )
        require(
            row["alignment_normalization"] == spec["alignment_normalization"]
            and row["alignment_weight"] == spec["alignment_weight"],
            "Geometry coefficient changed",
        )
        if step:
            require(
                row["alignment_selected_count"] == selected
                and row["alignment_denominator"] == denominator,
                "Geometry mask/denominator changed",
            )
            reduced = row["alignment_selected_mean"] * selected / denominator
            require(
                math.isclose(row["alignment_loss"], reduced, rel_tol=1e-5, abs_tol=1e-6),
                "Geometry loss does not match declared reduction",
            )
            require(
                math.isclose(
                    row["alignment_batch_mean"],
                    row["alignment_selected_mean"] * selected / 192,
                    rel_tol=1e-5,
                    abs_tol=1e-6,
                ),
                "Geometry batch mean mismatch",
            )
        count_hash = exposure_hash(path / f"exposures-{step:06d}.npz", sizes, step * 64)
        common.append(
            {"step": step, **{k: row[k] for k in COMMON_EXPOSURE}, "exposure_sha256": count_hash}
        )
    require(
        exposure_hash(path / "exposures.npz", sizes, spec["steps"] * 64)
        == common[-1]["exposure_sha256"],
        "Final exposure alias changed",
    )
    endpoint = history[-1]
    for key in ("sample_stream_sha256", "geometry_target_presentations"):
        require(endpoint[key] == complete[key] == audit[key], f"Endpoint/audit mismatch: {key}")
    same_metrics(endpoint["metrics"], complete["metrics"])
    same_metrics(endpoint["metrics"], audit["metrics"])
    require(set(endpoint["metrics"]) == set(SPLITS), "Missing endpoint evaluation split")
    values = {
        f"{split}.{key}": value
        for split, group in endpoint["metrics"].items()
        for key, value in group.items()
    }
    require(
        all(v is None or math.isfinite(v) for v in values.values()), "Nonfinite endpoint metric"
    )
    return {
        "run": str(path),
        "world": spec["world"],
        "initialization": spec["initialization"],
        "arm": spec["arm"],
        "endpoint": spec["steps"],
        "values": values,
        "hashes": {k: run[k] for k in IDENTITY if k != "spec"},
        "strata": run["strata"],
        "common_exposure_by_node": common,
        "geometry": {
            "coverage": spec["alignment_coverage"],
            "normalization": spec["alignment_normalization"],
            "weight": spec["alignment_weight"],
            "selected_per_batch": selected,
            "denominator": denominator,
            "coefficient_per_selected_example": (
                spec["alignment_weight"] / denominator if selected else None
            ),
            "total_coefficient": spec["alignment_weight"] * selected / denominator,
            "unique_examples": run["geometry_unique_examples"],
            "target_presentations": endpoint["geometry_target_presentations"],
        },
        "last_batch_objectives": {
            k: endpoint[k]
            for k in (
                "text_ce",
                "bridge_ce",
                "alignment_loss",
                "alignment_all_mean",
                "alignment_selected_mean",
                "alignment_batch_mean",
            )
        },
        "training_seconds": complete["training_seconds"],
    }


def check_pairs(rows):
    pairs, worlds = defaultdict(list), defaultdict(set)
    for key in ("source", "implementation_sha256"):
        require(
            all(row["hashes"][key] == rows[0]["hashes"][key] for row in rows),
            f"Different {key} across the matrix",
        )
    for row in rows:
        pairs[(row["world"], row["initialization"])].append(row)
        worlds[row["world"]].add(row["hashes"]["world_sha256"])
    require(
        all(len(hashes) == 1 for hashes in worlds.values()), "World changed across initializations"
    )
    for key, group in pairs.items():
        require(
            {r["arm"] for r in group} == set(ARMS) and len(group) == 4, f"Incomplete pair: {key}"
        )
        for field in ("hashes", "strata", "common_exposure_by_node"):
            require(
                all(r[field] == group[0][field] for r in group), f"Paired {field} mismatch: {key}"
            )


def mean_or_missing(values):
    return None if any(v is None for v in values) else sum(values) / len(values)


def hierarchical_mean(rows):
    """Never silently remove models/worlds with undefined conditional scores."""
    groups = defaultdict(list)
    names = sorted(rows[0]["values"])
    for row in rows:
        require(sorted(row["values"]) == names, "Endpoint metric schema differs between runs")
        groups[row["world"]].append(row)
    per_world = [
        {
            "world": world,
            "initializations": sorted(r["initialization"] for r in group),
            "values": {k: mean_or_missing([r["values"][k] for r in group]) for k in names},
            "defined_initializations": {
                k: sum(r["values"][k] is not None for r in group) for k in names
            },
        }
        for world, group in sorted(groups.items())
    ]
    return {
        "independent_worlds": len(groups),
        "models": len(rows),
        "per_world": per_world,
        "values": {k: mean_or_missing([r["values"][k] for r in per_world]) for k in names},
        "defined_worlds": {k: sum(r["values"][k] is not None for r in per_world) for k in names},
    }


def report(root, phase, config_path=None):
    require(phase in ("development", "confirmation"), "Unknown phase")
    root = Path(root)
    repository = Path(__file__).resolve().parents[1]
    batch = f"alignment-coverage-{phase}-v1"
    if config_path is None:
        frozen = repository / "docs/development-artifacts" / batch / "frozen-config.json"
        config_path = frozen if frozen.exists() else repository / "configs" / f"{batch}.json"
    config = read_json(config_path)
    specs = planned_specs(config, phase)
    actual = {p.parent.name for p in (root / "runs").glob("*/run.json")}
    require(actual == {s["name"] for s in specs}, "Missing, duplicate or unexpected runs in matrix")
    rows = [load_run(root / "runs" / s["name"], s, config.get("source")) for s in specs]
    check_pairs(rows)
    lookup = {(r["world"], r["initialization"], r["arm"]): r for r in rows}
    contrasts = []
    for left, right in CONTRASTS:
        paired = []
        for row in (r for r in rows if r["arm"] == left):
            other = lookup[(row["world"], row["initialization"], right)]
            values = {
                k: None
                if value is None or other["values"][k] is None
                else value - other["values"][k]
                for k, value in row["values"].items()
            }
            paired.append(
                {"world": row["world"], "initialization": row["initialization"], "values": values}
            )
        contrasts.append(
            {
                "contrast": f"{left}_minus_{right}",
                "paired_endpoints": paired,
                **hierarchical_mean(paired),
            }
        )
    summary = {
        "phase": phase,
        "status": "complete_and_audited",
        "runs": len(rows),
        "independent_worlds": len({r["world"] for r in rows}),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "aggregation": "initializations averaged within world, then worlds equally weighted",
        "conditional_metrics": (
            "null if any required model/world has no denominator; defined counts reported"
        ),
        "primary_metric": "strict_test.accuracy",
        "conditional_comparability": (
            "Each arm uses its own necessary-atoms-correct subset; conditional contrasts are "
            "descriptive and must be read with coverage, not as a fixed-subset causal comparison"
        ),
        "endpoint_rule": "predeclared final step only; no best-checkpoint or arm selection",
        "pairing_verified": [
            "world",
            "initial_model",
            "source",
            "sample_stream",
            "CE_and_bridge_exposure",
        ],
        "endpoints": rows,
        "arms": {arm: hierarchical_mean([r for r in rows if r["arm"] == arm]) for arm in ARMS},
        "contrasts": contrasts,
    }
    out = root / f"report-{phase}"
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "summary.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("development", "confirmation"))
    args = parser.parse_args()
    summary = report(args.root, args.phase)
    print(f"{summary['runs']} audited runs; {summary['independent_worlds']} independent worlds")


if __name__ == "__main__":
    main()
