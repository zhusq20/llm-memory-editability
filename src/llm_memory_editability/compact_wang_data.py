"""Lossless task-symbol remapping of the archived Wang graph and query pools.

Raw graph arrays stay byte-identical. The output vocabulary deliberately changes;
this is not numerical equivalence to the original GPT-2-vocabulary experiment.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import gate_extension_data as original
from .loop_hop_data import sha256

VOCABULARY = {
    "base_vocabulary": 3,
    "vocab_size": 2203,
    "entity_offset": 3,
    "relation_offset": 2003,
    "pad_token": 0,
    "end_marker": 2,
}
SOURCE_VOCABULARY = {
    "base_vocabulary": 50257,
    "vocab_size": 52463,
    "entity_offset": 50257,
    "relation_offset": 52257,
    "pad_token": 50256,
    "end_marker": 52460,
}


def remap(tokens, inverse=False):
    """Map used symbols exactly; preserve ignored targets, reject unused symbols."""
    values = np.asarray(tokens)
    source, target = (VOCABULARY, SOURCE_VOCABULARY) if inverse else (SOURCE_VOCABULARY, VOCABULARY)
    result = np.full(values.shape, -999, dtype=np.int64)
    result[values == -100] = -100
    for key in ("pad_token", "end_marker"):
        result[values == source[key]] = target[key]
    for key, count in (("entity_offset", 2000), ("relation_offset", 200)):
        selected = (values >= source[key]) & (values < source[key] + count)
        result[selected] = values[selected] - source[key] + target[key]
    if np.any(result == -999):
        raise ValueError("Token outside the used Wang task vocabulary")
    return result


def verify_metadata(metadata):
    if metadata.get("kind") != "wang" or metadata.get("vocabulary_variant") != "compact-v1":
        raise ValueError("Expected the compact Wang dataset")
    for key, value in {**VOCABULARY, "entities": 2000, "relations": 200}.items():
        if metadata.get(key) != value:
            raise ValueError(f"Compact Wang metadata mismatch: {key}")


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    world, panels, metadata = original.load(source)
    for key, value in SOURCE_VOCABULARY.items():
        if metadata.get(key) != value:
            raise ValueError(f"Original Wang vocabulary mismatch: {key}")
    original.audit_data(world, panels, metadata)
    compact = {
        **metadata,
        **VOCABULARY,
        "vocabulary_variant": "compact-v1",
        "source_metadata_sha256": sha256(source / "complete.json"),
        "source_vocabulary": SOURCE_VOCABULARY,
        "reserved_unused_token": 1,
        "task_status": (
            "Original Wang graph/support/pools; compact vocabulary; not old-recipe reproduction"
        ),
        "primary": "compact-full-vocabulary answer plus generated EOS; raw answer also reported",
    }
    verify_metadata(compact)
    checks = {}
    for name, rows in {
        **panels,
        "all_atoms": world["atoms"],
        "all_train_2": world["train_2"],
    }.items():
        old_x, old_pos, old_y = original.pack_rows(rows, metadata)
        new_x, new_pos, new_y = original.pack_rows(rows, compact)
        checks[name] = {
            "rows": len(rows),
            "inputs_exact_under_mapping": bool(np.array_equal(remap(old_x), new_x)),
            "targets_exact_under_mapping": bool(np.array_equal(remap(old_y), new_y)),
            "supervised_positions_unchanged": bool(np.array_equal(old_pos, new_pos)),
            "input_roundtrip": bool(np.array_equal(remap(new_x, inverse=True), old_x)),
            "target_roundtrip": bool(np.array_equal(remap(new_y, inverse=True), old_y)),
        }
    if not all(all(v for k, v in check.items() if k != "rows") for check in checks.values()):
        raise AssertionError(checks)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("world.npz", "panels.npz"):
        shutil.copyfile(source / name, destination / name)
        if sha256(destination / name) != sha256(source / name):
            raise ValueError(f"Raw data changed: {name}")
    shutil.copyfile(source / "complete.json", destination / "source-complete.json")
    (destination / "complete.json").write_text(json.dumps(compact, indent=2) + "\n")
    receipt = {
        "passed": True,
        "checks": checks,
        "world_sha256": compact["world_sha256"],
        "panels_sha256": compact["panels_sha256"],
        "vocabulary": VOCABULARY,
        "removed_output_competitors_change_loss_and_reencoding": True,
    }
    (destination / "mapping-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
