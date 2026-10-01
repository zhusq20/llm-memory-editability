"""Separate longer-path extension of the audited two-hop random-world task.

The graph and atomic split are reused exactly. A run trains all atomics and
one explicitly specified path length; no intermediate entities are targets.
Complete queries, not constituent facts or subpaths, define held-out examples.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .grok_depth_data import ENTITY_OFFSET
from .grok_depth_data import build_world as build_twohop_world

SPLITS = (
    "atomic",
    "id_atomic",
    "ood_atomic",
    "train_composite",
    "test_composite",
    "test_full_composite",
    "ood_composite",
    "unused_composite",
)


def path_details(rows, atomic, entities, relations):
    """Return the actual entity trajectory and atomic row indices at each hop."""
    offset = ENTITY_OFFSET + entities
    lookup = np.full((entities, relations), -1, dtype=np.int64)
    lookup[atomic[:, 0] - ENTITY_OFFSET, atomic[:, 1] - offset] = np.arange(len(atomic))
    nodes = np.empty((len(rows), rows.shape[1] - 1), dtype=np.int64)
    edges = np.empty((len(rows), rows.shape[1] - 2), dtype=np.int64)
    nodes[:, 0] = rows[:, 0]
    for j in range(edges.shape[1]):
        indices = lookup[nodes[:, j] - ENTITY_OFFSET, rows[:, j + 1] - offset]
        if np.any(indices < 0):
            raise ValueError("Path references a missing atomic fact")
        edges[:, j] = indices
        nodes[:, j + 1] = atomic[indices, -1]
    if np.any(nodes[:, -1] != rows[:, -1]):
        raise ValueError("Path target disagrees with atomic facts")
    return nodes, edges


def _diagnostics(rows, atomic, entities, relations, id_mask):
    nodes, edges = path_details(rows, atomic, entities, relations)
    repeated = np.zeros(len(rows), dtype=bool)
    for j in range(1, nodes.shape[1]):
        repeated |= (nodes[:, j, None] == nodes[:, :j]).any(axis=1)
    positional = []
    for j in range(edges.shape[1]):
        counts = np.bincount(edges[:, j], minlength=len(atomic))
        positional.append(
            {
                "hop": j + 1,
                "id_edges_covered": int((counts[id_mask] > 0).sum()),
                "id_edges_total": int(id_mask.sum()),
                "counts_per_atomic_edge": counts.tolist(),
                "id_count_min": int(counts[id_mask].min()) if id_mask.any() else None,
                "id_count_max": int(counts[id_mask].max()) if id_mask.any() else None,
            }
        )
    return {
        "n": len(rows),
        "any_repeated_entity_n": int(repeated.sum()),
        "any_repeated_entity_fraction": float(repeated.mean()) if len(rows) else None,
        "answer_equals_preceding_entity_n": int(
            (nodes[:, -1, None] == nodes[:, :-1]).any(axis=1).sum()
        ),
        "edge_positions": positional,
    }


def build_world(
    seed,
    hops=3,
    entities=128,
    relations=16,
    degree=8,
    phi=6.0,
    id_fraction=0.95,
    id_test_fraction=0.1,
    evaluation_size=1024,
):
    """Enumerate all paths, reserve test queries, then sample training queries.

    The fixed per-node evaluation subset is sampled from the entire reserved
    test pool without consulting training or predictions. Endpoints evaluate
    that full pool too. Mixed-ID/OOD paths remain excluded, as before.
    """
    if not isinstance(hops, int) or isinstance(hops, bool) or not 2 <= hops <= 4:
        raise ValueError("hops must be an integer from 2 through 4")
    if not isinstance(evaluation_size, int) or evaluation_size < 1:
        raise ValueError("evaluation_size must be a positive integer")
    if not math.isfinite(phi) or phi < 0:
        raise ValueError("phi must be finite and nonnegative")
    base = build_twohop_world(seed, entities, relations, degree, 0.0, id_fraction, id_test_fraction)
    atomic = base["atomic"]
    ood_set = {tuple(row) for row in base["ood_atomic"]}
    ood_mask = np.array([tuple(row) in ood_set for row in atomic])
    rows = atomic.copy()
    all_id, all_ood = ~ood_mask, ood_mask.copy()
    for _ in range(1, hops):
        following = (rows[:, -1, None] - ENTITY_OFFSET) * degree + np.arange(degree)
        following = following.reshape(-1)
        prefix = np.repeat(rows[:, :-1], degree, axis=0)
        rows = np.c_[prefix, atomic[following, 1:]]
        all_id = np.repeat(all_id, degree) & ~ood_mask[following]
        all_ood = np.repeat(all_ood, degree) & ood_mask[following]
    id_rows = rows[all_id]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(hops), 144001]))
    test_mask = rng.random(len(id_rows)) < id_test_fraction
    test_full = id_rows[test_mask]
    eligible = id_rows[~test_mask]
    probe_rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(hops), 144002]))
    probe_indices = probe_rng.choice(
        len(test_full), size=min(evaluation_size, len(test_full)), replace=False
    )
    requested = round(phi * len(base["id_atomic"]))
    ntrain = min(requested, len(eligible))
    indices = rng.choice(len(eligible), size=ntrain, replace=False)
    train_mask = np.zeros(len(eligible), dtype=bool)
    train_mask[indices] = True
    world = {
        "atomic": atomic,
        "id_atomic": base["id_atomic"],
        "ood_atomic": base["ood_atomic"],
        "train_composite": eligible[indices],
        "test_composite": test_full[probe_indices],
        "test_full_composite": test_full,
        "ood_composite": rows[all_ood],
        "unused_composite": eligible[~train_mask],
        "metadata": {
            **base["metadata"],
            "hops": hops,
            "task": "all atomic facts plus one target composition length",
            "phi_requested": phi,
            "phi_actual": ntrain / max(len(base["id_atomic"]), 1),
            "train_composite_requested": requested,
            "train_composite_capped": requested > len(eligible),
            "id_paths_total": len(id_rows),
            "all_paths_total": len(rows),
            "mixed_composite_excluded": int((~all_id & ~all_ood).sum()),
            "training_fraction_of_id_paths": ntrain / max(len(id_rows), 1),
            "training_fraction_of_nonreserved_id_paths": ntrain / max(len(eligible), 1),
            "evaluation_size_requested": evaluation_size,
            "probe_indices_in_full_test": probe_indices.tolist(),
            "split_rng": "default_rng(SeedSequence([world_seed,hops,144001]))",
            "probe_rng": "default_rng(SeedSequence([world_seed,hops,144002]))",
            "construction_differences": [
                "same graph and atomic split as historical two-hop constructor",
                "enumerated target length 2/3/4; only one length enters a run",
                "fresh deterministic complete-path reserve and phi downsampling",
                "fixed test probe at every node; entire reserve evaluated at endpoint",
            ],
            "warnings": [
                warning
                for condition, warning in (
                    (not len(test_full), "No reserved ID paths; never regenerate the world"),
                    (not int(all_ood.sum()), "No all-OOD paths; report zero denominator"),
                    (requested > len(eligible), "Training request capped at eligible pool"),
                )
                if condition
            ],
        },
    }
    audit = audit_world(world)
    world["metadata"]["counts"] = audit["counts"]
    world["metadata"]["dataset_sha256"] = audit["dataset_sha256"]
    world["metadata"]["diagnostics"] = {
        name: _diagnostics(world[name], atomic, entities, relations, ~ood_mask)
        for name in ("train_composite", "test_composite", "test_full_composite")
    }
    return world


def audit_world(world):
    """Independent graph truth, split completeness and full-query leakage checks."""
    meta = world["metadata"]
    entities, relations, hops = (int(meta[k]) for k in ("entities", "relations", "hops"))
    offset = ENTITY_OFFSET + entities
    digest = hashlib.sha256()
    counts = {}
    for name in SPLITS:
        rows = world[name]
        columns = 3 if name.endswith("atomic") else hops + 2
        if rows.dtype != np.int64 or rows.ndim != 2 or rows.shape[1] != columns:
            raise ValueError(f"Invalid shape or dtype: {name}")
        if np.any((rows[:, [0, -1]] < ENTITY_OFFSET) | (rows[:, [0, -1]] >= offset)):
            raise ValueError(f"Invalid entity tokens: {name}")
        if np.any((rows[:, 1:-1] < offset) | (rows[:, 1:-1] >= offset + relations)):
            raise ValueError(f"Invalid relation tokens: {name}")
        if len(np.unique(rows[:, :-1], axis=0)) != len(rows):
            raise ValueError(f"Duplicate complete queries: {name}")
        path_details(rows, world["atomic"], entities, relations)
        counts[name] = len(rows)
        digest.update(name.encode())
        digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
        digest.update(rows.astype("<i8", copy=False).tobytes())
    atoms = {tuple(row) for row in world["atomic"]}
    ids = {tuple(row) for row in world["id_atomic"]}
    oods = {tuple(row) for row in world["ood_atomic"]}
    if ids & oods or ids | oods != atoms:
        raise ValueError("Atomic ID/OOD split is not a partition")
    is_id = np.array([tuple(row) in ids for row in world["atomic"]])
    primary = ("train_composite", "test_full_composite", "unused_composite", "ood_composite")
    joined = np.concatenate([world[name] for name in primary])
    if len(np.unique(joined[:, :-1], axis=0)) != len(joined):
        raise ValueError("Complete queries overlap across train/reserve/unused/OOD")
    for name in primary:
        _, edges = path_details(world[name], world["atomic"], entities, relations)
        expected = ~is_id[edges] if name == "ood_composite" else is_id[edges]
        if not expected.all():
            raise ValueError(f"Incorrect atomic split along path: {name}")
    lookup = np.zeros((entities, relations), dtype=np.int64)
    id_lookup = lookup.copy()
    atomic = world["atomic"]
    for mask, table in ((is_id, id_lookup), (~is_id, lookup)):
        table.fill(-1)
        table[atomic[mask, 0] - ENTITY_OFFSET, atomic[mask, 1] - offset] = (
            atomic[mask, -1] - ENTITY_OFFSET
        )
    totals = []
    for table in (id_lookup, lookup):
        path_counts = np.ones(entities, dtype=np.int64)
        for _ in range(hops):
            path_counts = np.where(table >= 0, path_counts[np.maximum(table, 0)], 0).sum(1)
        totals.append(int(path_counts.sum()))
    if sum(counts[k] for k in primary[:3]) != totals[0] or counts[primary[3]] != totals[1]:
        raise ValueError("Path partitions are not exhaustive")
    probes = np.asarray(meta["probe_indices_in_full_test"], dtype=np.int64)
    if not np.array_equal(world["test_composite"], world["test_full_composite"][probes]):
        raise ValueError("Fixed test probe differs from its recorded full-pool indices")
    return {
        "counts": counts,
        "dataset_sha256": digest.hexdigest(),
        "id_paths_independently_counted": totals[0],
        "ood_paths_independently_counted": totals[1],
        "complete_query_overlap": 0,
        "atomic_facts_intentionally_shared_with_test": True,
    }
