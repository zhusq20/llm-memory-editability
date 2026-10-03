#!/usr/bin/env python3
"""Report audited, fixed-300k architecture endpoints without choosing settings.

Only standard-library dependencies are required. Three paired initializations
are the replication units; questions and prerequisite-filtered subsets are
descriptive measurements on the same previously inspected graph.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

SEEDS = (811201, 811202, 811203)
NEW_ARMS = ("M8", "W8", "IHC8", "HC8")
GROUPS = ("atomic", "train_composition", "test_all", "test_oo")
PAIRS = (
    ("L4R2", "D4"),
    ("L4R2", "D8_hist"),
    ("M8", "D8_calib"),
    ("M8", "W8"),
    ("HC8", "IHC8"),
    ("IHC8", "D8_calib"),
)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_rows(rows, label):
    result = {row["id"]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate IDs in {label}")
    return result


def seed_summary(values):
    """No query-level standard error or independent-query inference."""
    selected = [float(value) for value in values if value is not None]
    require(all(math.isfinite(value) for value in selected), "Nonfinite report value")
    return {
        "n_seed_values": len(selected),
        "mean": statistics.mean(selected) if selected else None,
        "sample_std": statistics.stdev(selected) if len(selected) > 1 else None,
    }


def mean_accuracy(ids, predictions):
    return sum(predictions[i] for i in ids) / len(ids) if ids else None


def common_prerequisites(left, right, examples):
    """Report own and common subsets without replacing the full-pool endpoint."""
    pools = {"test_all": examples, "test_oo": [r for r in examples if r["role"] == "OO"]}
    result = []
    for pool, rows in pools.items():
        own_left = {r["id"] for r in rows if all(left["atoms"][a] == 1 for a in r["atom_ids"])}
        own_right = {r["id"] for r in rows if all(right["atoms"][a] == 1 for a in r["atom_ids"])}
        common = own_left & own_right
        n = len(rows)
        lscore = mean_accuracy(common, left["test"])
        rscore = mean_accuracy(common, right["test"])
        result.append(
            {
                "pool": pool,
                "full_pool_n": n,
                "treatment_prerequisite_n": len(own_left),
                "reference_prerequisite_n": len(own_right),
                "common_prerequisite_n": len(common),
                "treatment_prerequisite_coverage": len(own_left) / n if n else None,
                "reference_prerequisite_coverage": len(own_right) / n if n else None,
                "common_prerequisite_coverage": len(common) / n if n else None,
                "treatment_own_subset_em": mean_accuracy(own_left, left["test"]),
                "reference_own_subset_em": mean_accuracy(own_right, right["test"]),
                "treatment_common_subset_em": lscore,
                "reference_common_subset_em": rscore,
                "paired_common_subset_difference": lscore - rscore if common else None,
            }
        )
    return result


def completed_run(path, spec, arm, config, data):
    """Trust the independent weight audit, validate its identities and raw recount."""
    files = {
        name: path / name
        for name in (
            "complete.json",
            "audit.json",
            "run.json",
            "endpoint.json",
            "endpoint-predictions.json",
        )
    }
    require(all(p.is_file() for p in files.values()), f"Incomplete endpoint: {path}")
    require(not (path / "failure.json").exists(), f"Unresolved failure: {path}")
    complete, audit, manifest = (
        read(files[n]) for n in ("complete.json", "audit.json", "run.json")
    )
    require(
        complete.get("state") == "complete" and complete.get("step") == 300000,
        f"Not a completed 300k run: {path}",
    )
    require(
        complete.get("independently_reloaded") is True and audit.get("passed") is True,
        f"Independent audit missing: {path}",
    )
    require(audit.get("pid") != audit.get("source_training_pid"), f"Not independent: {path}")
    for key in ("model_sha256", "checkpoint_sha256"):
        require(bool(complete.get(key)) and complete[key] == audit.get(key), f"Audit {key}: {path}")
    require(manifest.get("world_sha256") == config["data_sha256"], f"Wrong data: {path}")
    require(manifest.get("spec") == spec and spec["steps"] == 300000, f"Recipe mismatch: {path}")
    seed = spec["initialization"]
    require(seed in SEEDS, f"Unexpected initialization: {seed}")
    require(
        spec["sampling_seed"] == seed + 1000 and spec["dropout_seed"] == seed + 2000,
        f"Pairing seeds changed: {path}",
    )
    metrics, raw = read(files["endpoint.json"]), read(files["endpoint-predictions.json"])
    mappings = {}
    for group, key in (
        ("atomic", "atoms"),
        ("train_composition", "train_compositions"),
        ("test_all", "evaluation_compositions"),
    ):
        mapping = index_rows(raw[group], f"{path}/{group}")
        require(set(mapping) == {r["id"] for r in data[key]}, f"Wrong full pool: {path}/{group}")
        require(all(p["alias_em"] in (0, 1) for p in mapping.values()), "EM must be binary")
        mappings[group] = mapping
    oo_ids = {r["id"] for r in data["evaluation_compositions"] if r["role"] == "OO"}
    mappings["test_oo"] = {i: mappings["test_all"][i] for i in oo_ids}
    for group in GROUPS:
        values = mappings[group]
        require(metrics[group]["n"] == len(values), f"Wrong denominator: {path}/{group}")
        require(len(values) > 0, f"Empty primary pool: {path}/{group}")
        recount = sum(p["alias_em"] for p in values.values()) / len(values)
        require(
            math.isclose(recount, metrics[group]["alias_em"], abs_tol=1e-12, rel_tol=0),
            f"EM recount mismatch: {path}/{group}",
        )
        require(math.isfinite(metrics[group]["nll"]), f"Nonfinite NLL: {path}/{group}")
    required_atoms = {a for row in data["evaluation_compositions"] for a in row["atom_ids"]}
    record = {
        "architecture": arm,
        "initialization": seed,
        "run": spec["name"],
        "path": str(path),
        "steps": 300000,
        "learning_rate": spec["learning_rate"],
        "parameters": manifest.get("parameters"),
        "alias_of": None,
        "metrics": {g: metrics[g] for g in GROUPS},
        "resources": {
            k: complete.get(k)
            for k in (
                "training_seconds",
                "wall_seconds",
                "examples",
                "supervised_tokens",
                "executed_input_tokens",
                "estimated_matmul_training_flops",
            )
        },
        "files_sha256": {name: digest(p) for name, p in files.items()},
        "atoms": {i: mappings["atomic"][i]["alias_em"] for i in required_atoms},
        "test": {i: p["alias_em"] for i, p in mappings["test_all"].items()},
    }
    return record


def csv_output(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_report(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = read(config_path)
    require(config.get("phase") == "main", "Report requires the main phase")
    lock = read(config["execution_lock"])
    require(digest(config_path) == lock["config_sha256"], "Main config freeze mismatch")
    require(digest(config["data_file"]) == config["data_sha256"], "Main data freeze mismatch")
    data = read(config["data_file"])
    require(
        all(len(row.get("atom_ids", [])) == 2 for row in data["evaluation_compositions"]),
        "Two-prerequisite structure unavailable; cannot silently omit requested analysis",
    )
    reporting = config.get("reporting", {})
    historical_config_path = Path(
        reporting.get(
            "historical_config",
            Path(config["repository"]) / "configs/realworld-composition-confirmation-v1.json",
        )
    )
    if reporting.get("historical_config_sha256"):
        require(
            digest(historical_config_path) == reporting["historical_config_sha256"],
            "Historical config freeze mismatch",
        )
    historic = read(historical_config_path)
    require(historic["data_sha256"] == config["data_sha256"], "Historical graph differs")
    observed = [(s["architecture"], s["initialization"]) for s in config["runs"]]
    expected = {(a, seed) for a in NEW_ARMS for seed in SEEDS}
    if any(a == "D8" for a, _ in observed):
        expected |= {("D8", seed) for seed in SEEDS}
    require(len(observed) == len(expected) and set(observed) == expected, "Incomplete main matrix")
    runs = {}
    # Check every main run before writing any report, including independently audited completion.
    for spec in config["runs"]:
        arm = "D8_calib" if spec["architecture"] == "D8" else spec["architecture"]
        path = Path(config["results_root"]) / "runs" / spec["name"]
        runs[(arm, spec["initialization"])] = completed_run(path, spec, arm, config, data)
    old_names = {"standard4": "D4", "standard8": "D8_hist", "loop4x2": "L4R2"}
    seen = set()
    for spec in historic["runs"]:
        arm = old_names[spec["architecture"]]
        key = (arm, spec["initialization"])
        require(key not in seen, "Duplicate historical seed")
        seen.add(key)
        path = Path(historic["results_root"]) / historic["phase"] / spec["name"]
        runs[key] = completed_run(path, spec, arm, config, data)
    require(
        seen == {(a, seed) for a in old_names.values() for seed in SEEDS},
        "Incomplete historical baseline matrix",
    )
    if ("D8_calib", SEEDS[0]) not in runs:
        require(
            reporting.get("historical_reuse_verified") is True,
            "D8_calib cannot alias historical D8 without verified reuse",
        )
        for seed in SEEDS:
            runs[("D8_calib", seed)] = {
                **runs[("D8_hist", seed)],
                "architecture": "D8_calib",
                "alias_of": "D8_hist",
            }
    pair_rows, common_rows = [], []
    for treatment, reference in PAIRS:
        for seed in SEEDS:
            left, right = runs[(treatment, seed)], runs[(reference, seed)]
            identity = {"treatment": treatment, "reference": reference, "initialization": seed}
            for group in GROUPS:
                for metric in ("alias_em", "nll"):
                    one, two = left["metrics"][group], right["metrics"][group]
                    pair_rows.append(
                        {
                            **identity,
                            "pool": group,
                            "metric": metric,
                            "n": one["n"],
                            "treatment_value": one[metric],
                            "reference_value": two[metric],
                            "difference": one[metric] - two[metric],
                        }
                    )
            common_rows.extend(
                {**identity, **r}
                for r in common_prerequisites(left, right, data["evaluation_compositions"])
            )
    run_rows = []
    for (arm, seed), run in runs.items():
        for group in GROUPS:
            run_rows.append(
                {
                    "architecture": arm,
                    "initialization": seed,
                    "run": run["run"],
                    "alias_of": run["alias_of"],
                    "learning_rate": run["learning_rate"],
                    "steps": 300000,
                    "pool": group,
                    "n": run["metrics"][group]["n"],
                    "alias_em": run["metrics"][group]["alias_em"],
                    "nll": run["metrics"][group]["nll"],
                    "parameters": run["parameters"],
                    **run["resources"],
                }
            )
    arm_summary = []
    for arm in dict.fromkeys(a for a, _ in runs):
        for pool in GROUPS:
            for metric in ("alias_em", "nll"):
                values = [runs[(arm, seed)]["metrics"][pool][metric] for seed in SEEDS]
                arm_summary.append(
                    {
                        "architecture": arm,
                        "pool": pool,
                        "metric": metric,
                        **seed_summary(values),
                        "seed_values": dict(zip(SEEDS, values, strict=True)),
                    }
                )
    paired_summary = []
    common_summary = []
    for treatment, reference in PAIRS:
        for pool in GROUPS:
            for metric in ("alias_em", "nll"):
                values = [
                    r["difference"]
                    for r in pair_rows
                    if r["treatment"] == treatment
                    and r["reference"] == reference
                    and r["pool"] == pool
                    and r["metric"] == metric
                ]
                paired_summary.append(
                    {
                        "treatment": treatment,
                        "reference": reference,
                        "pool": pool,
                        "metric": metric,
                        **seed_summary(values),
                    }
                )
        for pool in ("test_all", "test_oo"):
            selected = [
                r
                for r in common_rows
                if r["treatment"] == treatment and r["reference"] == reference and r["pool"] == pool
            ]
            common_summary.append(
                {
                    "treatment": treatment,
                    "reference": reference,
                    "pool": pool,
                    **seed_summary([r["paired_common_subset_difference"] for r in selected]),
                    "per_seed_denominators": [
                        {
                            k: r[k]
                            for k in ("initialization", "full_pool_n", "common_prerequisite_n")
                        }
                        for r in selected
                    ],
                }
            )
    summary = {
        "state": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": digest(config_path),
        "historical_config_sha256": digest(historical_config_path),
        "data_sha256": config["data_sha256"],
        "endpoint_steps": 300000,
        "independent_training_initializations": 3,
        "independent_data_worlds": 1,
        "accuracy_unit": "fraction; paired difference is treatment minus reference",
        "selection": "No selection or best-checkpoint search; every frozen 300k run is retained.",
        "interpretation": "Finite-budget architecture/training comparisons "
        "on a previously inspected graph. "
        "Question counts are denominators, not independent model replications. "
        "Common prerequisite subsets are post-training descriptions, not causal adjustment.",
        "runs": [
            {k: v for k, v in run.items() if k not in ("atoms", "test")} for run in runs.values()
        ],
        "arm_summary": arm_summary,
        "paired_summary": paired_summary,
        "paired_seed_values": pair_rows,
        "common_prerequisites": common_rows,
        "common_prerequisite_summary": common_summary,
    }
    # No report artifact is emitted until every run and raw endpoint passes.
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    csv_output(output / "runs.csv", run_rows)
    csv_output(output / "paired-effects.csv", pair_rows)
    csv_output(output / "common-prerequisites.csv", common_rows)
    temporary.replace(output / "summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.config, args.out)
    print(json.dumps({"state": result["state"], "output": str(args.out)}))
