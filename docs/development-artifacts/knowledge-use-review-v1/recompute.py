"""Post-hoc rescore of existing confirmation endpoints; no model execution."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "docs/development-artifacts/grok-usage-v1/report-confirmation/nodes.csv"
WORLDS = (146011, 146012, 146013)
ARMS = ("A", "B", "repA", "repB")
GROUPS = ("BG", "A", "B")
PAIRS = {
    "both_composition": (("A", "A_A"), ("B", "B_B")),
    "first_composition_only": (("A", "A_B"), ("B", "B_A")),
    "second_composition_only": (("A", "B_A"), ("B", "A_B")),
    "neither_composition": (("A", "B_B"), ("B", "A_A")),
    "matched_atomic_repetition": (("repA", "A_A"), ("repB", "B_B")),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with SOURCE.open() as handle:
        source_rows = list(csv.DictReader(handle))
    reference = {
        (int(r["world_seed"]), r["arm"], r["split"]): r
        for r in source_rows
        if r["step"] == "128000"
    }
    hashes = {str(SOURCE.relative_to(ROOT)): digest(SOURCE)}
    records, by_key = [], {}
    for world in WORLDS:
        same_queries = {}
        for arm in ARMS:
            run = ROOT / f"results/grok-usage-v1/confirmation/confirm-w{world}-{arm}"
            paths = [run / "world.npz", run / "predictions-0128000.npz"]
            for path in paths:
                hashes[str(path.relative_to(ROOT))] = digest(path)
            with np.load(paths[0]) as data, np.load(paths[1]) as predictions:
                for left in GROUPS:
                    for right in GROUPS:
                        split = f"test_{left}_{right}"
                        queries = data["eval_" + split]
                        if split in same_queries:
                            assert np.array_equal(queries, same_queries[split])
                        else:
                            same_queries[split] = queries.copy()
                        target = predictions[split + "_target"]
                        assert np.array_equal(target, queries[:, -1])
                        correct = (predictions[split + "_answer"] == target) & (
                            predictions[split + "_stop"] == 1
                        )
                        probability = np.exp(
                            -predictions[split + "_nll"][:, 0].astype(np.float64)
                        ).mean()
                        old = reference[world, arm, split]
                        assert int(old["n"]) == len(queries)
                        assert abs(float(old["accuracy"]) - correct.mean()) < 1e-12
                        assert abs(float(old["answer_probability"]) - probability) < 1e-12
                        row = {
                            "world": world,
                            "arm": arm,
                            "split": split,
                            "n": len(queries),
                            "correct": int(correct.sum()),
                            "accuracy": float(correct.mean()),
                            "answer_probability": float(probability),
                        }
                        records.append(row)
                        by_key[world, arm, left + "_" + right] = row
    with (OUT / "endpoint-cells.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    roles = []
    for role, pairs in PAIRS.items():
        per_world = []
        for world in WORLDS:
            rows = [by_key[world, arm, split] for arm, split in pairs]
            per_world.append(
                {
                    "world": world,
                    "accuracy": sum(r["accuracy"] for r in rows) / len(rows),
                    "cells": rows,
                }
            )
        roles.append(
            {
                "role": role,
                "accuracy": sum(r["accuracy"] for r in per_world) / len(per_world),
                "worlds": per_world,
            }
        )
    summary = {
        "status": "complete",
        "analysis_type": "post_hoc_existing_predictions_only",
        "new_training_runs": 0,
        "new_forward_passes": 0,
        "endpoint_runs": 12,
        "rescored_cells": len(records),
        "aggregation": "equal cohorts within world, then equal worlds; one initialization",
        "limits": [
            "Different role rows can contain different queries; not a factorial intervention.",
            "First/second indicates position at evaluation, not position-restricted training.",
            "No significance test or independent-world claim from query-level replication.",
        ],
        "roles": roles,
        "source_sha256": hashes,
        "analysis_sha256": digest(Path(__file__).resolve()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for role in roles:
        print(f"{role['role']}: {role['accuracy']:.6%}")
    print(f"Verified {len(records)} cells from 12 existing endpoints.")


if __name__ == "__main__":
    main()
