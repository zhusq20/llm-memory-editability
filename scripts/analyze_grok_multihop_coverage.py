#!/usr/bin/env python3
"""Recompute the registered CPU-only path-coverage calibration (no model results)."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.grok_multihop_data import build_world, path_details  # noqa: E402

OUTPUT = ROOT / "docs/development-artifacts/grok-multihop-v1/coverage-calibration.json"
WORLD_SEED = 144001
HOPS = (3, 4)
PHIS = (6, 12, 24, 48)
STEPS, BATCH_SIZE = 128000, 256
WORLD_ARGS = {
    "entities": 128,
    "relations": 16,
    "degree": 8,
    "id_fraction": 0.95,
    "id_test_fraction": 0.1,
    "evaluation_size": 1024,
}


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def coverage(flags):
    flags = np.asarray(flags, dtype=bool)
    return {
        "covered": int(flags.sum()),
        "total": len(flags),
        "fraction": float(flags.mean()) if len(flags) else None,
    }


def analyze():
    records, worlds = [], []
    for hops in HOPS:
        reference = None
        for phi in PHIS:
            world = build_world(WORLD_SEED, hops=hops, phi=phi, **WORLD_ARGS)
            train, test = world["train_composite"], world["test_full_composite"]
            meta, atomic = world["metadata"], world["atomic"]
            invariant = {
                key: world[key]
                for key in (
                    "atomic",
                    "id_atomic",
                    "ood_atomic",
                    "test_full_composite",
                    "test_composite",
                )
            }
            if reference is None:
                reference = {key: value.copy() for key, value in invariant.items()}
                worlds.append(
                    {
                        "hops": hops,
                        "array_hash_encoding": "little-endian int64 shape followed by array bytes",
                        "invariant_array_sha256": {
                            key: array_hash(value) for key, value in invariant.items()
                        },
                        "test_full_n": len(test),
                        "test_probe_n": len(world["test_composite"]),
                        "probe_indices_in_full_test": meta["probe_indices_in_full_test"],
                        "split_rng": meta["split_rng"],
                        "probe_rng": meta["probe_rng"],
                        "id_paths_total": meta["id_paths_total"],
                        "all_paths_total": meta["all_paths_total"],
                        "ood_paths_total": len(world["ood_composite"]),
                    }
                )
            for key, value in invariant.items():
                if not np.array_equal(reference[key], value):
                    raise AssertionError(f"Changing phi changed {key} for {hops} hops")
            train_nodes, train_edges = path_details(train, atomic, 128, 16)
            test_nodes, test_edges = path_details(test, atomic, 128, 16)
            by_position = np.zeros((hops, len(atomic)), dtype=bool)
            for position in range(hops):
                by_position[position, train_edges[:, position]] = True
            all_edges_seen = np.column_stack(
                [by_position[position, test_edges[:, position]] for position in range(hops)]
            ).all(axis=1)
            train_relations = set(map(tuple, train[:, 1:-1]))
            all_id = np.concatenate([train, test, world["unused_composite"]])
            all_relations = set(map(tuple, all_id[:, 1:-1]))
            subpaths = []
            for length in range(2, hops):
                for position in range(hops - length + 1):
                    seen = {
                        tuple(
                            [
                                int(train_nodes[index, position]),
                                *row[1 + position : 1 + position + length],
                            ]
                        )
                        for index, row in enumerate(train)
                    }
                    flags = [
                        tuple(
                            [
                                int(test_nodes[index, position]),
                                *row[1 + position : 1 + position + length],
                            ]
                        )
                        in seen
                        for index, row in enumerate(test)
                    ]
                    subpaths.append(
                        {
                            "subpath_hops": length,
                            "start_hop": position + 1,
                            "unique_training_subpaths_at_position": len(seen),
                            **coverage(flags),
                        }
                    )
            records_n = len(atomic) + len(train)
            exposure = STEPS * BATCH_SIZE / records_n
            records.append(
                {
                    "hops": hops,
                    "phi": phi,
                    "train_composite_n": len(train),
                    "training_records_n": records_n,
                    "train_composite_capped": meta["train_composite_capped"],
                    "training_array_sha256": array_hash(train),
                    "mean_exposures_per_record": exposure,
                    "atomic_training_examples_proportional_expectation": exposure * len(atomic),
                    "training_fraction_of_id_paths": meta["training_fraction_of_id_paths"],
                    "training_fraction_of_nonreserved_id_paths": meta[
                        "training_fraction_of_nonreserved_id_paths"
                    ],
                    "id_atomic_edges_total": len(world["id_atomic"]),
                    "id_atomic_edges_seen_by_position": by_position.sum(axis=1).tolist(),
                    "test_all_edges_seen_at_same_position": coverage(all_edges_seen),
                    "unique_relation_sequences_in_training": len(train_relations),
                    "unique_relation_sequences_in_all_id_paths": len(all_relations),
                    "test_relation_sequence_seen": coverage(
                        [tuple(row) in train_relations for row in test[:, 1:-1]]
                    ),
                    "adjacent_relation_pairs_seen_by_position": [
                        len(set(map(tuple, train[:, 1 + position : 3 + position])))
                        for position in range(hops - 1)
                    ],
                    "adjacent_relation_pair_possible_n": WORLD_ARGS["relations"] ** 2,
                    "test_subpath_seen_at_same_position": subpaths,
                }
            )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "CPU graph and registered split analysis only; "
            "no training results, checkpoints or predictions read"
        ),
        "inputs": {
            "world_seed": WORLD_SEED,
            "hops": list(HOPS),
            "phis": list(PHIS),
            **WORLD_ARGS,
            "training_steps": STEPS,
            "batch_size": BATCH_SIZE,
            "training_examples": STEPS * BATCH_SIZE,
            "model_initialization_in_development": 14401,
            "training_stream_seed_in_development": 144011,
            "model_and_stream_seeds_used_in_this_analysis": False,
        },
        "definitions": {
            "mean_exposures_per_record": (
                "steps * batch_size / (atomic_count + sampled_composite_count); "
                "uniform shuffled epochs, actual records differ by at most one presentation"
            ),
            "atomic_training_examples_proportional_expectation": (
                "mean_exposures_per_record * atomic_count; "
                "exact partial-epoch atomic count is not simulated"
            ),
            "test_denominator": "entire reserved ID test pool, not the fixed 1024-example probe",
            "same_position_atomic_coverage": (
                "Each constituent (head, relation, tail) occurred at that same hop in at least "
                "one training composition; this is data exposure, not learned correctness"
            ),
            "relation_sequence_coverage": (
                "Ordered relation tuple alone appeared in training, regardless of starting entity"
            ),
            "subpath_coverage": (
                "Actual intermediate starting entity plus ordered relation subsequence appeared "
                "at the same starting hop in a training composition; "
                "these subpaths are not separately supervised"
            ),
            "invariance": (
                "Exact equality of graph, atomic split, full test pool and test probe "
                "across phi is asserted"
            ),
            "training_sets": (
                "Uniform samples without replacement at each phi; samples need not be nested"
            ),
        },
        "source_sha256": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "scripts/analyze_grok_multihop_coverage.py",
                "src/llm_memory_editability/grok_multihop_data.py",
                "src/llm_memory_editability/grok_depth_data.py",
            )
        },
        "numpy_version": np.__version__,
        "worlds": worlds,
        "records": records,
        "recommendation": {
            "phi": 24,
            "retain": (
                "k3-d4 and k4-d6 architectures, width, learning rate, initialization, "
                "full test pool, probe, 128000 updates and batch256"
            ),
            "basis": (
                "phi6 already gives about 98% test coverage of atomic facts at their hop "
                "positions and complete adjacent-relation-pair coverage; phi24 raises "
                "same-position two-hop-subpath coverage to about 95-97% while preserving "
                "more per-record exposure than phi48"
            ),
            "prerequisite": (
                "Check atomic and training-composition learning in development; "
                "data coverage alone does not explain an optimization failure"
            ),
            "interpretation": (
                "One development data-amount calibration; no guarantee of success, "
                "no depth-law claim, and no trained-sample or model-result selection "
                "in this analysis"
            ),
        },
    }


if __name__ == "__main__":
    report = analyze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved {len(report['records'])} coverage rows to {OUTPUT}")
