"""Graph-only, exposure-matched same-bridge donors for two-hop OOD queries.

Every donor contains only [head, r1]; its bridge is saved as truth, never fed
to the model. ID donors must have actually occurred at the first position of
composite training. OOD donors have zero composite exposure at either position.
All eligible donors and ID x OOD pairs are retained. A pair is a repeated
measurement of its recipient, not an independent query or world.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from .grok_multihop_data import audit_world, path_details


def _nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _exposure_spec(world, run_metadata, steps):
    spec = run_metadata.get("spec", run_metadata)
    if not isinstance(spec, Mapping):
        raise ValueError("Run metadata must contain a specification mapping")
    count = _nonnegative_integer(spec["steps"] if steps is None else steps, "steps")
    if count > _nonnegative_integer(spec["steps"], "registered steps"):
        raise ValueError("Exposure cannot exceed the registered training endpoint")
    batch = _nonnegative_integer(spec["batch_size"], "batch_size")
    atomic_batch = _nonnegative_integer(spec["n_atomic_per_batch"], "n_atomic_per_batch")
    seed = _nonnegative_integer(spec["stream_seed"], "stream_seed")
    if batch < 1 or atomic_batch > batch:
        raise ValueError("Require batch_size > 0 and n_atomic_per_batch <= batch_size")
    for key in ("world_seed", "hops"):
        expected = world["metadata"]["seed" if key == "world_seed" else key]
        if key in spec and spec[key] != expected:
            raise ValueError(f"Run specification disagrees with the world: {key}")
    return {
        "dataset_sha256": world["metadata"]["dataset_sha256"],
        "stream_seed": seed,
        "steps": count,
        "batch_size": batch,
        "n_atomic_per_batch": atomic_batch,
        "atomic_size": len(world["atomic"]),
        "composite_size": len(world["train_composite"]),
        "stream": "StratifiedStream independent shuffled epochs v1",
    }


def _cache_key(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _record_exposures(size, draws, seed, tag):
    """Replay the final partial epoch without materializing the full sample stream."""
    if not size:
        if draws:
            raise ValueError("A sampled training stratum cannot be empty")
        return np.empty(0, dtype=np.int64)
    complete, remainder = divmod(draws, size)
    counts = np.full(size, complete, dtype=np.int64)
    if remainder:
        rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))
        for _ in range(complete):
            rng.permutation(size)
        counts[rng.permutation(size)[:remainder]] += 1
    return counts


def actual_exposure_counts(world, run_metadata, *, steps=None):
    """Count actual atomic and composite records sampled through a fixed step.

    The training sampler has independent atomic/composite RNG streams, so its
    interleaving draws do not change these counts. Full epochs expose every
    record once; only the final partial epoch needs its exact permutation.
    ``cache_key`` depends on data, sample-stream seed, step and batch mixture,
    but not architecture or initialization, which do not change this sampler.
    """
    checked = audit_world(world)
    if checked["dataset_sha256"] != world["metadata"].get("dataset_sha256"):
        raise ValueError("World metadata hash does not match the data")
    if world["metadata"]["hops"] != 2:
        raise ValueError("Same-bridge exposure comparison requires two-hop worlds")
    spec = _exposure_spec(world, run_metadata, steps)
    atom_count = _record_exposures(
        spec["atomic_size"],
        spec["steps"] * spec["n_atomic_per_batch"],
        spec["stream_seed"],
        145101,
    )
    composite_count = _record_exposures(
        spec["composite_size"],
        spec["steps"] * (spec["batch_size"] - spec["n_atomic_per_batch"]),
        spec["stream_seed"],
        145102,
    )
    _, edges = path_details(
        world["train_composite"],
        world["atomic"],
        world["metadata"]["entities"],
        world["metadata"]["relations"],
    )
    table = np.zeros((len(world["atomic"]), 2), dtype=np.int64)
    actual = np.zeros_like(table)
    for position in range(2):
        np.add.at(table[:, position], edges[:, position], 1)
        np.add.at(actual[:, position], edges[:, position], composite_count)
    return {
        "cache_key": _cache_key(spec),
        "cache_spec": spec,
        "atomic_record_counts": atom_count,
        "composite_record_counts": composite_count,
        "table_position_counts": table,
        "actual_position_counts": actual,
    }


def _rows(values, columns):
    return np.asarray(values, dtype=np.int64).reshape(-1, columns)


def select_same_bridge_donors(world, run_metadata, *, steps=None, exposure=None):
    """Enumerate same-r1/same-bridge ID and OOD donors without model selection.

    Inputs are a standard audited loop world and either run metadata with a
    ``spec`` field or the specification directly. All original OOD rows stay
    aligned with validity masks and explicit reasons. Flattened donor arrays
    include recipients with only one donor class. Main pairs require both.
    ``exposure`` may cache ``actual_exposure_counts`` for a shared sample stream.

    For pair-level scores, first average pairs within each recipient (using
    pair_weights), then average recipients. Pair weights sum to one for each
    common recipient; they do not sum to one across the whole world.
    """
    checked = audit_world(world)
    if checked["dataset_sha256"] != world["metadata"].get("dataset_sha256"):
        raise ValueError("World metadata hash does not match the data")
    if world["metadata"]["hops"] != 2:
        raise ValueError("Same-bridge selection requires two-hop worlds")
    spec = _exposure_spec(world, run_metadata, steps)
    exposure = (
        actual_exposure_counts(world, run_metadata, steps=steps) if exposure is None else exposure
    )
    if exposure.get("cache_key") != _cache_key(spec) or exposure.get("cache_spec") != spec:
        raise ValueError("Cached exposure belongs to another world or sample stream")
    atoms = world["atomic"]
    position_counts = np.asarray(exposure["actual_position_counts"])
    table_counts = np.asarray(exposure["table_position_counts"])
    if (
        position_counts.shape != (len(atoms), 2)
        or table_counts.shape != position_counts.shape
        or position_counts.dtype != np.int64
        or table_counts.dtype != np.int64
        or (position_counts < 0).any()
        or (table_counts < 0).any()
    ):
        raise ValueError("Cached position exposure counts have invalid shape or values")
    atom_index = {tuple(map(int, row)): i for i, row in enumerate(atoms)}
    is_id = np.zeros(len(atoms), dtype=bool)
    is_id[[atom_index[tuple(map(int, row))] for row in world["id_atomic"]]] = True
    if position_counts[~is_id].any() or table_counts[~is_id].any():
        raise ValueError("An OOD fact has composite training exposure")
    expected_draws = spec["steps"] * (spec["batch_size"] - spec["n_atomic_per_batch"])
    if not np.array_equal(position_counts.sum(axis=0), [expected_draws, expected_draws]):
        raise ValueError("Cached position counts disagree with the registered sample budget")
    incoming = defaultdict(list)
    for i, (head, relation, bridge) in enumerate(atoms):
        incoming[int(relation), int(bridge)].append((int(head), i))
    for candidates in incoming.values():
        candidates.sort()
    original = world["ood_composite"]
    nodes, edges = path_details(
        original, atoms, world["metadata"]["entities"], world["metadata"]["relations"]
    )
    if is_id[edges].any():
        raise ValueError("Recipients must contain only OOD facts")
    n = len(original)
    id_indices, ood_indices = [], []
    id_recipient, ood_recipient = [], []
    id_offsets, ood_offsets = [0], [0]
    second_only = np.zeros(n, dtype=np.int64)
    nominal_unexposed = np.zeros(n, dtype=np.int64)
    rejected_same_head = np.zeros(n, dtype=np.int64)
    rejected_answer_head = np.zeros(n, dtype=np.int64)
    for i, (head, r1, _r2, target) in enumerate(original):
        for donor_head, fact in incoming[int(r1), int(nodes[i, 1])]:
            if donor_head == head:
                rejected_same_head[i] += 1
                continue
            if donor_head == target:
                rejected_answer_head[i] += 1
                continue
            if is_id[fact]:
                if position_counts[fact, 0] > 0:
                    id_indices.append(fact)
                    id_recipient.append(i)
                elif position_counts[fact, 1] > 0:
                    second_only[i] += 1
                else:
                    nominal_unexposed[i] += 1
            else:
                ood_indices.append(fact)
                ood_recipient.append(i)
        id_offsets.append(len(id_indices))
        ood_offsets.append(len(ood_indices))
    id_offsets = np.asarray(id_offsets, dtype=np.int64)
    ood_offsets = np.asarray(ood_offsets, dtype=np.int64)
    id_count, ood_count = np.diff(id_offsets), np.diff(ood_offsets)
    valid = (id_count > 0) & (ood_count > 0)
    pair_recipient, pair_id, pair_ood, weights = [], [], [], []
    for i in np.flatnonzero(valid):
        weight = 1.0 / int(id_count[i] * ood_count[i])
        for j in range(id_offsets[i], id_offsets[i + 1]):
            for k in range(ood_offsets[i], ood_offsets[i + 1]):
                pair_recipient.append(i)
                pair_id.append(j)
                pair_ood.append(k)
                weights.append(weight)
    pair_recipient = np.asarray(pair_recipient, dtype=np.int64)
    pair_id = np.asarray(pair_id, dtype=np.int64)
    pair_ood = np.asarray(pair_ood, dtype=np.int64)
    id_indices = np.asarray(id_indices, dtype=np.int64)
    ood_indices = np.asarray(ood_indices, dtype=np.int64)
    reason = np.full(n, "eligible", dtype="U40")
    reason[(id_count == 0) & (ood_count > 0)] = "no_id_first_position_donor"
    reason[(id_count > 0) & (ood_count == 0)] = "no_other_ood_donor"
    reason[(id_count == 0) & (ood_count == 0)] = "no_id_or_other_ood_donor"
    result = {
        "original_rows": original.copy(),
        "original_bridge": nodes[:, 1].copy(),
        "original_atomic_indices": edges.copy(),
        "original_prefix_contains_answer": original[:, 0] == original[:, -1],
        "original_answer_equals_bridge": nodes[:, 1] == original[:, -1],
        "common_valid": valid,
        "reason": reason,
        "id_candidate_count": id_count,
        "ood_candidate_count": ood_count,
        "id_second_only_candidate_count": second_only,
        "id_unexposed_candidate_count": nominal_unexposed,
        "rejected_same_head_count": rejected_same_head,
        "rejected_answer_head_count": rejected_answer_head,
        "id_candidate_offsets": id_offsets,
        "ood_candidate_offsets": ood_offsets,
        "id_donor_atomic_indices": id_indices,
        "ood_donor_atomic_indices": ood_indices,
        "id_donor_rows": atoms[id_indices].copy(),
        "ood_donor_rows": atoms[ood_indices].copy(),
        "id_recipient_indices": np.asarray(id_recipient, dtype=np.int64),
        "ood_recipient_indices": np.asarray(ood_recipient, dtype=np.int64),
        "pair_recipient_indices": pair_recipient,
        "pair_rows": original[pair_recipient].copy(),
        "pair_id_indices": pair_id,
        "pair_ood_indices": pair_ood,
        "pair_weights": np.asarray(weights, dtype=np.float64),
        "pair_id_donor_rows": atoms[id_indices[pair_id]].copy(),
        "pair_ood_donor_rows": atoms[ood_indices[pair_ood]].copy(),
        "pair_self_donor_rows": atoms[edges[pair_recipient, 0]].copy(),
        "atomic_is_id": is_id,
        "atomic_single_training_exposure": np.asarray(exposure["atomic_record_counts"]).copy(),
        "atomic_composite_table_position_counts": table_counts.copy(),
        "atomic_composite_actual_position_exposure": position_counts.copy(),
        "composite_record_training_exposure": np.asarray(
            exposure["composite_record_counts"]
        ).copy(),
    }
    trained = {tuple(map(int, row[:-1])) for row in world["train_composite"]}
    for family in ("id", "ood"):
        cf = result["pair_rows"].copy()
        cf[:, 0] = result["pair_" + family + "_donor_rows"][:, 0]
        _, cf_edges = path_details(
            cf, atoms, world["metadata"]["entities"], world["metadata"]["relations"]
        )
        if any(tuple(map(int, row[:-1])) in trained for row in cf):
            raise ValueError("A same-bridge counterfactual query was used in training")
        if is_id[cf_edges[:, 1]].any():
            raise ValueError("A donor has changed the recipient's OOD successor")
        result["pair_" + family + "_counterfactual_rows"] = cf
        result["pair_" + family + "_counterfactual_atomic_indices"] = cf_edges
        result["pair_" + family + "_counterfactual_split"] = np.full(
            len(cf), "mixed_id_ood" if family == "id" else "pure_ood", dtype="U16"
        )
    result["audit"] = {
        "dataset_sha256": checked["dataset_sha256"],
        "exposure_cache_key": exposure["cache_key"],
        "exposure_cache_spec": spec,
        "recipient_split": "ood_composite",
        "selection": "enumerate all graph-qualified ID x OOD pairs; no model-based selection",
        "id_requirement": "same r1/bridge, different head, actual first-position exposure > 0",
        "ood_requirement": (
            "same r1/bridge, different head, no composite exposure at either position"
        ),
        "answer_head_excluded": True,
        "prefix_inputs": "donor_rows[:, :2]; bridge is audit truth, never model input",
        "successor": "unchanged OOD second fact for both donor classes",
        "all_recipient_n": n,
        "id_available_n": int((id_count > 0).sum()),
        "ood_available_n": int((ood_count > 0).sum()),
        "common_n": int(valid.sum()),
        "common_fraction": float(valid.mean()) if n else None,
        "pair_n": len(pair_recipient),
        "unique_original_first_facts_n": len(np.unique(edges[:, 0])),
        "unique_original_second_facts_n": len(np.unique(edges[:, 1])),
        "unique_common_first_facts_n": len(np.unique(edges[valid, 0])),
        "unique_common_second_facts_n": len(np.unique(edges[valid, 1])),
        "unique_common_id_donor_facts_n": len(np.unique(id_indices[pair_id])),
        "unique_common_ood_donor_facts_n": len(np.unique(ood_indices[pair_ood])),
        "unique_all_id_donor_facts_n": len(np.unique(id_indices)),
        "unique_all_ood_donor_facts_n": len(np.unique(ood_indices)),
        "unique_common_r1_bridge_groups_n": len(
            np.unique(np.c_[original[valid, 1], nodes[valid, 1]], axis=0)
        ),
        "original_prefix_contains_answer_n": int(result["original_prefix_contains_answer"].sum()),
        "common_prefix_contains_answer_n": int(
            result["original_prefix_contains_answer"][valid].sum()
        ),
        "original_answer_equals_bridge_n": int(result["original_answer_equals_bridge"].sum()),
        "common_answer_equals_bridge_n": int(result["original_answer_equals_bridge"][valid].sum()),
        "common_first_edge_self_loop_n": int((original[valid, 0] == nodes[valid, 1]).sum()),
        "common_id_donor_prefix_contains_answer_n": int(
            (result["pair_id_donor_rows"][:, 0] == result["pair_rows"][:, -1]).sum()
        ),
        "common_ood_donor_prefix_contains_answer_n": int(
            (result["pair_ood_donor_rows"][:, 0] == result["pair_rows"][:, -1]).sum()
        ),
        "id_second_only_candidate_n": int(second_only.sum()),
        "id_unexposed_candidate_n": int(nominal_unexposed.sum()),
        "missing_reasons": {
            str(value): int((reason == value).sum()) for value in np.unique(reason)
        },
        "aggregation": (
            "mean pairs per recipient, recipients per model, "
            "initializations per world, worlds equally"
        ),
    }
    return result
