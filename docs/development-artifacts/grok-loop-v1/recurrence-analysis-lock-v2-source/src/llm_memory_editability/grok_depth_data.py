"""Symbolic composition worlds adapted from GrokkedTransformer/composition.ipynb.

Source: https://github.com/OSU-NLP-Group/GrokkedTransformer
Revision: 734ca654ec7a71dd6737d640407fac14491d538c (MIT).

The graph, atomic ID/OOD split, reserved ID test chains, and phi downsampling
follow the authors' construction. Differences: integer tokens replace text;
randomness is local and seeded; atomic ordering is stable; graph size and the
reserved ID-test fraction are configurable. Unselected ID chains are returned
for audit only. No graph is regenerated to improve OOD counts or model results.

Adapted source license:
MIT License
Copyright (c) 2024 OSU Natural Language Processing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

PAD_TOKEN = 0
EOS_TOKEN = 1
ENTITY_OFFSET = 2
SOURCE_URL = "https://github.com/OSU-NLP-Group/GrokkedTransformer"
SOURCE_COMMIT = "734ca654ec7a71dd6737d640407fac14491d538c"
ATOMIC_SPLITS = ("atomic", "id_atomic", "ood_atomic")
COMPOSITE_SPLITS = (
    "train_composite",
    "test_composite",
    "ood_composite",
    "unused_composite",
)


def _array(rows: list, columns: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.int64).reshape(-1, columns)


def build_world(
    seed: int,
    entities: int = 128,
    relations: int = 16,
    degree: int = 8,
    phi: float = 8.0,
    id_fraction: float = 0.95,
    id_test_fraction: float = 0.005,
) -> dict:
    """Return atomic ``[h,r,t]`` and composite ``[h,r1,r2,t]`` int64 arrays.

    All atomics belong in training, including those labelled OOD. OOD refers
    to *composition experience*: neither constituent of an OOD composite is
    allowed in composite training. Mixed ID/OOD chains are excluded. The
    test set is reserved before phi downsampling; unused ID chains are never
    automatically reassigned to test. ``phi`` is relative to ID atomics, as in
    the source. Requests larger than the available training pool are capped
    and explicitly recorded. Empty test sets are retained and reported.

    Query tokens are every column except the last; the last is the entity
    answer. EOS supervision, padding, and loss masks belong to the trainer.
    """
    for name, value in (("entities", entities), ("relations", relations), ("degree", degree)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if degree > relations:
        raise ValueError("degree cannot exceed relations")
    if not math.isfinite(phi) or phi < 0:
        raise ValueError("phi must be finite and nonnegative")
    if not math.isfinite(id_fraction) or not 0 <= id_fraction <= 1:
        raise ValueError("id_fraction must be in [0, 1]")
    if not math.isfinite(id_test_fraction) or not 0 <= id_test_fraction <= 1:
        raise ValueError("id_test_fraction must be in [0, 1]")

    # RandomState keeps the source's NumPy draw family, without global RNG use.
    rng = np.random.RandomState(seed)
    relation_offset = ENTITY_OFFSET + entities
    rows = []
    outgoing = [[] for _ in range(entities)]
    for head_index in range(entities):
        selected = rng.choice(relations, size=degree, replace=False)
        for relation_index in selected:
            tail_index = int(rng.randint(entities))
            edge = (
                head_index + ENTITY_OFFSET,
                int(relation_index) + relation_offset,
                tail_index + ENTITY_OFFSET,
            )
            outgoing[head_index].append(len(rows))
            rows.append(edge)
    atomic = _array(rows, 3)
    ood_count = round(len(atomic) * (1 - id_fraction))
    ood_mask = np.zeros(len(atomic), dtype=bool)
    ood_mask[rng.choice(len(atomic), size=ood_count, replace=False)] = True

    eligible, test, ood = [], [], []
    mixed_count = 0
    for first_index, (head, relation1, bridge) in enumerate(atomic):
        for second_index in outgoing[int(bridge) - ENTITY_OFFSET]:
            _, relation2, tail = atomic[second_index]
            row = (head, relation1, relation2, tail)
            first_ood, second_ood = ood_mask[first_index], ood_mask[second_index]
            if first_ood and second_ood:
                ood.append(row)
            elif first_ood or second_ood:
                mixed_count += 1
            elif rng.uniform() > id_test_fraction:
                eligible.append(row)
            else:
                test.append(row)

    requested_count = round(phi * int((~ood_mask).sum()))
    train_count = min(requested_count, len(eligible))
    # Source choose() returns the full pool unchanged when it is saturated.
    indices = (
        np.arange(len(eligible))
        if train_count == len(eligible)
        else rng.choice(len(eligible), size=train_count, replace=False)
    )
    eligible_array = _array(eligible, 4)
    selected_mask = np.zeros(len(eligible), dtype=bool)
    selected_mask[indices] = True
    world = {
        "atomic": atomic,
        "id_atomic": atomic[~ood_mask],
        "ood_atomic": atomic[ood_mask],
        "train_composite": eligible_array[indices],
        "test_composite": _array(test, 4),
        "ood_composite": _array(ood, 4),
        "unused_composite": eligible_array[~selected_mask],
        "metadata": {
            "seed": int(seed),
            "entities": int(entities),
            "relations": int(relations),
            "degree": int(degree),
            "phi_requested": float(phi),
            "id_fraction": float(id_fraction),
            "id_test_fraction": float(id_test_fraction),
            "pad_token": PAD_TOKEN,
            "eos_token": EOS_TOKEN,
            "entity_offset": ENTITY_OFFSET,
            "relation_offset": int(relation_offset),
            "vocab_size": int(relation_offset + relations),
            "train_composite_requested": int(requested_count),
            "train_composite_capped": requested_count > len(eligible),
            "phi_actual": train_count / max(int((~ood_mask).sum()), 1),
            "mixed_composite_excluded": mixed_count,
            "source_url": SOURCE_URL,
            "source_commit": SOURCE_COMMIT,
            "source_file": "composition.ipynb",
            "source_license": "MIT",
            "construction_differences": [
                "integer tokens and local seeded RNG",
                "stable atomic ordering instead of Python set iteration",
                "configurable graph size and ID-test reserve fraction",
                "all evaluation rows retained instead of sampling at most 3000",
            ],
        },
    }
    audit = audit_world(world)
    world["metadata"]["counts"] = audit["counts"]
    world["metadata"]["dataset_sha256"] = audit["dataset_sha256"]
    world["metadata"]["warnings"] = audit["warnings"]
    return world


def audit_world(world: dict) -> dict:
    """Validate token ranges, graph truth, exhaustive partitions and no leakage.

    Raises ValueError on a broken contract; zero-sized test sets and capped
    training pools are valid and surfaced as warnings, never resampled.
    """
    metadata = world["metadata"]
    entities, relations, degree = (int(metadata[x]) for x in ("entities", "relations", "degree"))
    relation_offset = entities + ENTITY_OFFSET
    digest = hashlib.sha256()
    counts = {}
    tuples = {}
    for name in ATOMIC_SPLITS + COMPOSITE_SPLITS:
        array = world[name]
        columns = 3 if name in ATOMIC_SPLITS else 4
        if not isinstance(array, np.ndarray) or array.dtype != np.int64:
            raise ValueError(f"{name} must be an int64 numpy array")
        if array.ndim != 2 or array.shape[1] != columns:
            raise ValueError(f"{name} must have {columns} columns")
        for column in (0, columns - 1):
            if np.any(array[:, column] < ENTITY_OFFSET) or np.any(
                array[:, column] >= relation_offset
            ):
                raise ValueError(f"{name} has an invalid entity token")
        if np.any(array[:, 1:-1] < relation_offset) or np.any(
            array[:, 1:-1] >= relation_offset + relations
        ):
            raise ValueError(f"{name} has an invalid relation token")
        rows = {tuple(map(int, row)) for row in array}
        if len(rows) != len(array) or len({row[:-1] for row in rows}) != len(rows):
            raise ValueError(f"{name} has duplicate or conflicting queries")
        tuples[name] = rows
        counts[name] = len(array)
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.astype("<i8", copy=False).tobytes())

    atomics, id_atomics, ood_atomics = (tuples[name] for name in ATOMIC_SPLITS)
    if id_atomics & ood_atomics or id_atomics | ood_atomics != atomics:
        raise ValueError("ID/OOD atomics do not partition all atomic facts")
    if len(atomics) != entities * degree:
        raise ValueError("incorrect atomic graph size")
    outgoing = {}
    lookup = {}
    for head, relation, tail in atomics:
        outgoing.setdefault(head, []).append((relation, tail))
        lookup[head, relation] = tail
    if len(outgoing) != entities or any(len(edges) != degree for edges in outgoing.values()):
        raise ValueError("incorrect per-entity out-degree")

    id_chains, ood_chains, mixed_count = set(), set(), 0
    for head, relation1, bridge in atomics:
        for relation2, tail in outgoing[bridge]:
            row = (head, relation1, relation2, tail)
            first_id = (head, relation1, bridge) in id_atomics
            second_id = (bridge, relation2, tail) in id_atomics
            if first_id and second_id:
                id_chains.add(row)
            elif not first_id and not second_id:
                ood_chains.add(row)
            else:
                mixed_count += 1
    occupied = set()
    for name in COMPOSITE_SPLITS:
        rows = tuples[name]
        if rows & occupied:
            raise ValueError("composite splits overlap")
        occupied |= rows
        for head, relation1, relation2, tail in rows:
            bridge = lookup.get((head, relation1))
            if bridge is None or lookup.get((bridge, relation2)) != tail:
                raise ValueError(f"{name} contains a false chain")
        expected = ood_chains if name == "ood_composite" else id_chains
        if not rows <= expected:
            raise ValueError(f"{name} contains chains from the wrong atomic split")
    returned_id = set().union(
        *(tuples[name] for name in COMPOSITE_SPLITS if name != "ood_composite")
    )
    if returned_id != id_chains or tuples["ood_composite"] != ood_chains:
        raise ValueError("composite partitions omit eligible graph chains")
    if mixed_count != metadata["mixed_composite_excluded"]:
        raise ValueError("incorrect excluded mixed-chain count")
    warnings = [
        f"{name} is empty" for name in ("test_composite", "ood_composite") if not counts[name]
    ]
    if metadata["train_composite_capped"]:
        warnings.append("requested phi exceeds available composite training pool")
    return {
        "ok": True,
        "counts": counts,
        "id_composite_total": len(id_chains),
        "ood_composite_total": len(ood_chains),
        "mixed_composite_excluded": mixed_count,
        "dataset_sha256": digest.hexdigest(),
        "warnings": warnings,
    }
