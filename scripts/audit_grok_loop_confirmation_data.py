#!/usr/bin/env python3
"""Rebuild every registered confirmation world on CPU, without model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_sequence_multiple_targets(rows):
    """Descriptive within-pool subset; no fitted predictor or outcome selection."""
    if not len(rows):
        return {"n": 0, "total_n": 0, "coverage": None, "relation_sequences": 0}, np.zeros(0, bool)
    _, groups = np.unique(rows[:, 1:-1], axis=0, return_inverse=True)
    distinct = np.unique(np.c_[groups, rows[:, -1]], axis=0)
    counts = np.bincount(distinct[:, 0])
    mask = counts[groups] > 1
    return {
        "n": int(mask.sum()),
        "total_n": len(rows),
        "coverage": float(mask.mean()),
        "relation_sequences": int((counts > 1).sum()),
    }, mask


def donor_diagnostics(donors, multiple_target_mask):
    result = {}
    for family in ("different", "same"):
        valid = donors[family + "_valid"]
        counts = donors[family + "_candidate_count"]
        result[family] = {
            "eligible_n": int(valid.sum()),
            "total_n": len(valid),
            "coverage": float(valid.mean()) if len(valid) else None,
            "candidate_count_min_eligible": int(counts[valid].min()) if valid.any() else None,
            "candidate_count_max": int(counts.max()) if len(counts) else None,
            "candidate_count_mean": float(counts.mean()) if len(counts) else None,
            "reasons": dict(Counter(donors[family + "_reason"].tolist())),
            "counterfactual_split_counts": dict(
                Counter(donors[family + "_counterfactual_split"].tolist())
            ),
            "eligible_and_multiple_target_same_sequence_n": int(
                (valid & multiple_target_mask).sum()
            ),
        }
    result["different_candidate_rejection_counts"] = {
        key.removeprefix("different_rejected_"): int(value.sum())
        for key, value in donors.items()
        if key.startswith("different_rejected_")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-loop-confirmation-v1.json")
    parser.add_argument("--donor-seed", type=int, default=20260930)
    parser.add_argument(
        "--out",
        default="docs/development-artifacts/grok-loop-v1/confirmation-data-diagnostics.json",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    lock_path = ROOT / config["source_lock"]
    lock = json.loads(lock_path.read_text())
    snapshot = lock_path.parent / (lock_path.stem + "-source")
    source_checks = []
    for relative, digest in lock["files"].items():
        checks = {
            "path": relative,
            "expected_sha256": digest,
            "snapshot_matches": sha256(snapshot / relative) == digest,
            "current_matches": sha256(ROOT / relative) == digest,
        }
        source_checks.append(checks)
    if not all(check["snapshot_matches"] and check["current_matches"] for check in source_checks):
        raise ValueError("Source lock mismatch; refusing to reconstruct with unregistered code")
    # The isolated package path forces all relative imports to use the frozen snapshot.
    package = types.ModuleType("frozen_loop_confirmation_data")
    package.__path__ = [str(snapshot / "src/llm_memory_editability")]
    sys.modules[package.__name__] = package
    data = importlib.import_module(package.__name__ + ".grok_loop_data")
    mechanism = importlib.import_module(package.__name__ + ".grok_loop_mechanism")
    conditions = defaultdict(list)
    completed = []
    for run_id, override in config["runs"].items():
        spec = {**config["base"], **override}
        key = spec["world_seed"], spec["hops"]
        conditions[key].append((run_id, spec))
        directory = ROOT / config["output_root"] / spec["phase"] / run_id
        if (directory / "complete.json").is_file():
            completed.append((run_id, key, directory))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "config_sha256": sha256(ROOT / args.config),
        "source_lock": config["source_lock"],
        "source_lock_sha256": sha256(lock_path),
        "auditor_source_sha256": sha256(Path(__file__)),
        "source_checks": source_checks,
        "interpretation": {
            "selection": (
                "All registered world/hop conditions retained; "
                "no resampling or configuration changes"
            ),
            "coverage": (
                "Data availability and graph eligibility only; "
                "no model, accuracy or mastery evaluation"
            ),
            "multiple_targets": (
                "Within each named evaluation pool, identical complete relation "
                "sequences have multiple targets"
            ),
            "donor": (
                "Frozen pure-prefix selector; supplied donor input is only [head, r1]; "
                "donor seed controls choice, not eligibility"
            ),
            "unit": (
                "3 independent worlds, 3 hop conditions per world; "
                "architecture and initialization do not create new worlds"
            ),
        },
        "donor_seed": args.donor_seed,
        "conditions": [],
        "completed_data_checks": [],
    }
    for key, runs in sorted(conditions.items()):
        spec = runs[0][1]
        fields = {"world_seed": key[0], **{name: spec[name] for name in data.DATA_DEFAULTS}}
        for _, other in runs:
            assert all(other[name] == value for name, value in fields.items())
        world = data.build_world(fields)
        audit = data.audit_world(world)
        metadata = world["metadata"]
        diagnostics = metadata["loop_data_diagnostics"]
        entry = {
            "spec": fields,
            "registered_runs": [name for name, _ in runs],
            "audit": audit,
            "counts": metadata["counts"],
            "phi_actual": metadata["phi_actual"],
            "train_composite_capped": metadata["train_composite_capped"],
            "warnings": metadata["warnings"],
            "training_coverage": {
                name: diagnostics[name]
                for name in (
                    "atomic_facts",
                    "id_facts",
                    "ood_facts",
                    "id_facts_seen_in_composite_training",
                    "ood_facts_seen_in_composite_training",
                )
            },
            "training_coverage_by_hop": [
                {k: v for k, v in item.items() if k != "counts_per_atomic_edge"}
                for item in metadata["diagnostics"]["train_composite"]["edge_positions"]
            ],
            "evaluation": {},
            "pure_prefix_donor_eligibility": {},
        }
        for split in ("test_composite", "test_full_composite", "ood_composite"):
            multiple_targets, mask = same_sequence_multiple_targets(world[split])
            entry["evaluation"][split] = {
                **diagnostics["splits"][split],
                "multiple_target_same_relation_sequence": multiple_targets,
            }
            if split in ("test_composite", "ood_composite"):
                donors = mechanism.select_donors(world, args.donor_seed, split)
                entry["pure_prefix_donor_eligibility"][split] = donor_diagnostics(donors, mask)
        result["conditions"].append(entry)
        for run_id, completed_key, directory in completed:
            if completed_key != key:
                continue
            saved = dict(np.load(directory / "world.npz"))
            saved["metadata"] = json.loads((directory / "world-metadata.json").read_text())
            saved_audit = data.audit_world(saved)
            array_match = all(np.array_equal(saved[split], world[split]) for split in data.SPLITS)
            hash_match = saved_audit["dataset_sha256"] == audit["dataset_sha256"]
            result["completed_data_checks"].append(
                {
                    "run": run_id,
                    "world_seed": key[0],
                    "hops": key[1],
                    "dataset_sha256": saved_audit["dataset_sha256"],
                    "array_equality": array_match,
                    "hash_equality": hash_match,
                    "passed": array_match and hash_match,
                }
            )
        print(
            json.dumps({"world_seed": key[0], "hops": key[1], "counts": metadata["counts"]}),
            flush=True,
        )
    result["worlds"] = sorted({world for world, _ in conditions})
    result["unique_world_hop_conditions"] = len(conditions)
    result["passed"] = all(check["passed"] for check in result["completed_data_checks"])
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
