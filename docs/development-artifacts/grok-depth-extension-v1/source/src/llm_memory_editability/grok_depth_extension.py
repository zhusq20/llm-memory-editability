"""Frozen paired depth extension: validation and threshold accounting only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_pair(spec, baseline):
    """Only the layer count and bookkeeping phase may differ from the old baseline."""
    assert baseline["layers"] == 2
    assert spec["layers"] in (3, 4)
    ignored = {"layers", "phase"}
    assert {k: v for k, v in spec.items() if k not in ignored} == {
        k: v for k, v in baseline.items() if k not in ignored
    }, "Extension changes a non-depth training condition"


def threshold_time(rows, threshold, counts, budget):
    """First start of three registered consecutive nodes; never choose a best node."""
    assert 0 <= threshold <= 1
    steps = [row["step"] for row in rows]
    assert steps == sorted(set(steps)), "Unsorted or duplicate evaluation nodes"
    assert not steps or steps[-1] <= budget
    result = {
        "threshold": threshold,
        "reached": False,
        "step": None,
        "confirmed_at_step": None,
        "previous_evaluation_step": None,
        "examples": None,
        "atomic_exposures": None,
        "composite_exposures": None,
        "dataset_passes": None,
        "estimated_training_flops": None,
        "effective_input_tokens": None,
        "supervised_tokens": None,
        "last_observed_step": steps[-1] if steps else None,
        "budget_steps": budget,
    }
    for i in range(max(0, len(rows) - 2)):
        triple = rows[i : i + 3]
        if all(row["test_composite"]["accuracy"] >= threshold for row in triple):
            row = triple[0]
            result.update(
                reached=True,
                step=row["step"],
                confirmed_at_step=triple[-1]["step"],
                previous_evaluation_step=rows[i - 1]["step"] if i else None,
                examples=row["examples"],
                atomic_exposures=row["counts"]["atomic"] / counts["atomic"],
                composite_exposures=row["counts"]["composite"] / counts["train_composite"],
                dataset_passes=row["examples"] / (counts["atomic"] + counts["train_composite"]),
                estimated_training_flops=row["estimated_training_flops"],
                effective_input_tokens=row["effective_input_tokens"],
                supervised_tokens=row["supervised_tokens"],
            )
            break
    return result


def verify_files(root, expected):
    for relative, sha in expected.items():
        assert digest(root / relative) == sha, f"Frozen input changed: {relative}"
