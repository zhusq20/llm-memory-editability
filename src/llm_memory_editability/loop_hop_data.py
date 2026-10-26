"""Balanced two-graph permutation task for separating layers from recurrence.

This is a registered extension, not the original DiscoLoop dataset: all facts
from two disjoint graphs are trained, compositions only from graph A. Each
relation is a permutation so a fixed relation sequence preserves head identity.
No intermediate entities are training targets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel_names():
    return ("atomic_id", "atomic_ood") + tuple(
        f"{split}_{k}" for k in (2, 3, 4) for split in ("train", "id", "ood")
    )


def traverse(heads, relations, permutations):
    heads = np.asarray(heads, dtype=np.int64)
    relations = np.asarray(relations, dtype=np.int64)
    nodes = np.empty((len(heads), relations.shape[1] + 1), dtype=np.int64)
    nodes[:, 0] = heads
    for i in range(relations.shape[1]):
        nodes[:, i + 1] = permutations[relations[:, i], nodes[:, i]]
    return nodes


def pack_rows(rows, metadata):
    """Preserve the framework's answer + EOS objective at variable path length."""
    rows = np.asarray(rows)
    x = np.full((len(rows), metadata["sequence_length"]), metadata["pad_token"], np.int64)
    x[:, : rows.shape[1]] = rows
    x[:, 0] += metadata["entity_offset"]
    x[:, rows.shape[1] - 1] += metadata["entity_offset"]
    x[:, 1 : rows.shape[1] - 1] += metadata["relation_offset"]
    positions = np.tile([rows.shape[1] - 2, rows.shape[1] - 1], (len(rows), 1))
    labels = np.column_stack(
        (rows[:, -1] + metadata["entity_offset"], np.full(len(rows), metadata["end_marker"]))
    )
    return x, positions, labels


def prepare(config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    spec = config["data"]
    n, r = spec["entities_per_graph"], spec["relations"]
    rng = np.random.default_rng(spec["world_seed"])
    permutations = np.stack([[rng.permutation(n) for _ in range(r)] for _ in range(2)])
    atoms = np.array(
        [
            (g * n + h, rel, g * n + permutations[g, rel, h])
            for g in range(2)
            for h in range(n)
            for rel in range(r)
        ],
        dtype=np.int64,
    )
    world = {"atoms": atoms, "permutations": permutations}
    panels = {"atomic_id": atoms[: n * r], "atomic_ood": atoms[n * r :]}
    for k in (2, 3, 4):
        count = spec["evaluation_templates"]
        if count >= r**k:
            raise ValueError("Evaluation templates would exhaust the relation sequences")
        # A fixed sequence is evaluated at EVERY head. This is a balanced task
        # diagnostic, not sampling based on a model's predictions.
        for graph, split in ((0, "id"), (1, "ood")):
            codes = rng.choice(r**k, count, replace=False)
            template = np.array([np.unravel_index(int(c), (r,) * k) for c in codes])
            relation_rows = np.repeat(template, n, axis=0)
            heads = np.tile(np.arange(n), count)
            nodes = traverse(heads, relation_rows, permutations[graph])
            panels[f"{split}_{k}"] = np.column_stack(
                (heads + graph * n, relation_rows, nodes[:, -1] + graph * n)
            )
        forbidden = {tuple(row[:-1]) for row in panels[f"id_{k}"]}
        seen, collected = set(), []
        target = spec["train_compositions_per_hop"]
        if target > n * (r**k - count):
            raise ValueError("Requested training support exceeds the unreserved query space")
        while len(collected) < target:
            head = int(rng.integers(n))
            sequence = tuple(map(int, rng.integers(r, size=k)))
            key = (head, *sequence)
            if key in seen or key in forbidden:
                continue
            seen.add(key)
            collected.append(key)
        queries = np.asarray(collected, dtype=np.int64)
        nodes = traverse(queries[:, 0], queries[:, 1:], permutations[0])
        world[f"train_{k}"] = np.column_stack((queries, nodes[:, -1]))
        take = rng.choice(target, min(target, spec["training_panel_size"]), replace=False)
        panels[f"train_{k}"] = world[f"train_{k}"][take]
    np.savez_compressed(destination / "world.npz", **world)
    np.savez_compressed(destination / "panels.npz", **panels)
    metadata = {
        **spec,
        "entity_offset": 3,
        "relation_offset": 3 + 2 * n,
        "pad_token": 0,
        "end_marker": 2,
        "vocab_size": 3 + 2 * n + r,
        "sequence_length": 6,
        "atomic_examples": len(atoms),
        "training_examples": len(atoms) + sum(len(world[f"train_{k}"]) for k in (2, 3, 4)),
        "world_sha256": sha256(destination / "world.npz"),
        "panels_sha256": sha256(destination / "panels.npz"),
        "panels": {name: len(rows) for name, rows in panels.items()},
        "primary": "full-vocabulary answer argmax; answer plus generated EOS also reported",
        "task_status": "controlled permutation extension; not a literal DiscoLoop reproduction",
    }
    (destination / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    findings = audit_data(world, panels, metadata)
    (destination / "audit.json").write_text(json.dumps(findings, indent=2) + "\n")
    return metadata


def load(destination):
    root = Path(destination)
    metadata = json.loads((root / "complete.json").read_text())
    for name in ("world", "panels"):
        if sha256(root / f"{name}.npz") != metadata[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
    return dict(np.load(root / "world.npz")), dict(np.load(root / "panels.npz")), metadata


def path_details(rows, metadata, permutations):
    n, r = metadata["entities_per_graph"], metadata["relations"]
    graphs = rows[:, 0] // n
    nodes = np.empty((len(rows), rows.shape[1] - 1), dtype=np.int64)
    nodes[:, 0] = rows[:, 0]
    for i in range(rows.shape[1] - 2):
        nodes[:, i + 1] = graphs * n + permutations[graphs, rows[:, i + 1], nodes[:, i] % n]
    if not np.array_equal(nodes[:, -1], rows[:, -1]):
        raise ValueError("Path target disagrees with its atomic facts")
    # Atomic table order: graph, head, relation.
    edges = nodes[:, :-1] * r + rows[:, 1:-1]
    return nodes, edges


def audit_data(world, panels, metadata):
    n, r = metadata["entities_per_graph"], metadata["relations"]
    permutations = world["permutations"]
    if permutations.shape != (2, r, n):
        raise ValueError("Incorrect permutation graph shape")
    if not np.all(np.sort(permutations, axis=-1) == np.arange(n)):
        raise ValueError("A relation is not a permutation")
    if set(panels) != set(panel_names()):
        raise ValueError("Incomplete registered panel matrix")
    all_rows = {"atoms": world["atoms"], **panels}
    for k in (2, 3, 4):
        all_rows[f"full_train_{k}"] = world[f"train_{k}"]
        train = world[f"train_{k}"]
        if np.any(train[:, [0, -1]] >= n):
            raise ValueError("Graph B composition leaked into training")
        if len({tuple(row[:-1]) for row in train}) != len(train):
            raise ValueError("Training queries are not unique")
        if {tuple(row[:-1]) for row in train} & {tuple(row[:-1]) for row in panels[f"id_{k}"]}:
            raise ValueError("ID test query leaked into training")
        for split, graph in (("id", 0), ("ood", 1)):
            rows = panels[f"{split}_{k}"]
            if not np.all(rows[:, 0] // n == graph):
                raise ValueError("Incorrect graph membership")
            _, groups = np.unique(rows[:, 1:-1], axis=0, return_inverse=True)
            for group in np.unique(groups):
                selected = rows[groups == group]
                if not np.array_equal(np.sort(selected[:, 0]), np.arange(n) + graph * n):
                    raise ValueError("Evaluation heads are not balanced per relation sequence")
                if not np.array_equal(np.sort(selected[:, -1]), np.arange(n) + graph * n):
                    raise ValueError("Suffix-only answer shortcut detected")
    diagnostics = {}
    for name, rows in all_rows.items():
        nodes, edges = path_details(rows, metadata, permutations)
        if not np.array_equal(world["atoms"][edges, -1], nodes[:, 1:]):
            raise ValueError("Edge indices disagree with trajectory")
        repeated = np.array([len(set(path)) != len(path) for path in nodes])
        diagnostics[name] = {"n": len(rows), "repeated_entity_paths": int(repeated.sum())}
    return {
        "passed": True,
        "panels": diagnostics,
        "balanced_suffix_only_best_accuracy": 1 / n,
        "graph_b_composition_training_examples": 0,
        "graph_b_subpath_training_examples": 0,
    }


def make_table(world, metadata, device):
    import torch

    rows = [world["atoms"], *(world[f"train_{k}"] for k in (2, 3, 4))]
    packed = [pack_rows(part, metadata) for part in rows]
    padding = (
        np.full((1, metadata["sequence_length"]), metadata["pad_token"], np.int64),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[-100, -100]], dtype=np.int64),
    )
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device)
        for parts in zip(*packed, padding, strict=True)
    )
    return table


def exposure_summary(counts, world, metadata, steps, batch_size):
    epochs, remaining = divmod(steps, math.ceil(len(counts) / batch_size))
    if counts.sum() != epochs * len(counts) + remaining * batch_size:
        raise AssertionError("Exposure does not match the merged shuffled sample stream")
    if counts.max() - counts.min() > 1:
        raise AssertionError("Merged epoch stream exposure became imbalanced")
    end, groups = 0, {}
    for name in ("atoms", "train_2", "train_3", "train_4"):
        begin, end = end, end + len(world[name])
        groups[name] = int(counts[begin:end].sum())
    actual = int(counts.sum())
    return {
        "examples": actual,
        "counts": groups,
        "atomic_epochs": groups["atoms"] / len(world["atoms"]),
        "composition_epochs": (actual - groups["atoms"]) / (len(counts) - len(world["atoms"])),
        "effective_input_tokens": groups["atoms"] * 3
        + sum(groups[f"train_{k}"] * (k + 2) for k in (2, 3, 4)),
        "supervised_tokens": actual * 2,
        "executed_input_tokens": steps * batch_size * metadata["sequence_length"],
        "masked_batch_slots": steps * batch_size - actual,
    }
