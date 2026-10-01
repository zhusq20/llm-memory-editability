"""Audited data and fixed atomic/composite exposure for loop-depth comparisons.

The graph, truth, complete-query holdout and phi sampling are inherited from
``grok_multihop_data``. Changing the atomic ID fraction is an explicit design
choice; no graph is regenerated in response to diagnostics or model scores.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from . import grok_multihop_data as multihop

SPLITS = multihop.SPLITS
DATA_DEFAULTS = {
    "hops": 2,
    "entities": 128,
    "relations": 16,
    "degree": 8,
    "phi": 6.0,
    "id_fraction": 0.95,
    "id_test_fraction": 0.1,
    "evaluation_size": 1024,
}


def build_world(spec: Mapping) -> dict:
    """Adapt a flat run specification while retaining the historical world API.

    ``world_seed`` (or ``seed``) is required. Unrelated model/training keys are
    ignored. The default atomic ID/OOD fraction remains 95/5; use an explicit
    ``id_fraction=0.75`` for the new composition-experience comparison.

    ``relevant_atomic_indices[split][i, j]`` identifies query i's true hop-j
    fact in ``world['atomic']``. These truth indices are evaluation artifacts;
    training must only use the original atomic and train-composite rows.
    """
    if "world_seed" not in spec and "seed" not in spec:
        raise ValueError("spec must contain world_seed or seed")
    seed = spec.get("world_seed", spec.get("seed"))
    fields = {key: spec.get(key, default) for key, default in DATA_DEFAULTS.items()}
    world = multihop.build_world(seed, **fields)
    world["relevant_atomic_indices"] = {
        name: query_atomic_indices(world, world[name]) for name in SPLITS
    }
    world["metadata"]["loop_data_diagnostics"] = data_diagnostics(world)
    world["metadata"]["loop_data_contract"] = {
        "constructor": "grok_multihop_data.build_world",
        "world_selection": "fixed supplied seed; no diagnostic or model-based selection",
        "all_atomic_rows_are_training_facts": True,
        "ood_definition": "all constituent facts absent from composite training by split",
        "query_fact_indices": "evaluation truth only; no intermediate-entity supervision",
        "suffix_statistics": "within-split descriptive counts; not fitted or held-out predictors",
    }
    audit_world(world)
    return world


def query_atomic_indices(world: dict, rows: np.ndarray) -> np.ndarray:
    """Return true constituent fact indices, including [N, 1] for atomics."""
    meta = world["metadata"]
    _, edges = multihop.path_details(
        rows, world["atomic"], int(meta["entities"]), int(meta["relations"])
    )
    return edges


def audit_world(world: dict) -> dict:
    """Independently verify truth, ID/OOD partitions and recorded fact indices.

    The reused auditor checks truth by graph traversal and counts all-ID and
    all-OOD paths by dynamic programming, independently of path construction.
    Neither correct-looking diagnostics nor a saved metadata hash substitutes
    for those checks.
    """
    result = multihop.audit_world(world)
    recorded = world.get("relevant_atomic_indices")
    if recorded is not None:
        if set(recorded) != set(SPLITS):
            raise ValueError("Relevant atomic indices must cover every data split")
        for name in SPLITS:
            expected = query_atomic_indices(world, world[name])
            if (
                not isinstance(recorded[name], np.ndarray)
                or recorded[name].dtype != np.int64
                or not np.array_equal(recorded[name], expected)
            ):
                raise ValueError(f"Incorrect relevant atomic indices: {name}")
    if result["dataset_sha256"] != world["metadata"].get("dataset_sha256"):
        raise ValueError("Recorded dataset hash disagrees with actual data")
    return {**result, "relevant_atomic_indices_checked": recorded is not None}


def suffix_target_diagnostics(rows: np.ndarray) -> list[dict]:
    """Describe target concentration after omitting the head and prefix relations.

    The empirical modal fraction sums each suffix group's largest target count
    and divides by query count. It uses the same rows to count and summarize;
    singleton suffixes score one automatically. It is not a model-independent
    generalization bound, a learned baseline, or proof that heads are unnecessary.
    """
    results = []
    n = len(rows)
    for length in range(1, rows.shape[1] - 1):
        suffix = rows[:, -(length + 1) : -1]
        if not n:
            results.append(
                {
                    "suffix_relation_count": length,
                    "groups": 0,
                    "empirical_modal_target_fraction": None,
                    "single_target_group_query_fraction": None,
                    "singleton_group_query_fraction": None,
                    "multi_query_groups": 0,
                    "multi_query_coverage_fraction": None,
                    "multi_query_empirical_modal_target_fraction": None,
                    "multi_target_groups": 0,
                    "same_suffix_different_target_example": None,
                }
            )
            continue
        _, groups, totals = np.unique(suffix, axis=0, return_inverse=True, return_counts=True)
        pairs, pair_counts = np.unique(np.c_[groups, rows[:, -1]], axis=0, return_counts=True)
        maxima = np.zeros(len(totals), dtype=np.int64)
        np.maximum.at(maxima, pairs[:, 0], pair_counts)
        multi = totals > 1
        multi_n = int(totals[multi].sum())
        multi_target = maxima != totals
        example = None
        if multi_target.any():
            # Deterministic first lexicographic suffix with incompatible targets.
            group = int(np.flatnonzero(multi_target)[0])
            members = np.flatnonzero(groups == group)
            first = members[0]
            second = members[np.flatnonzero(rows[members, -1] != rows[first, -1])[0]]
            example = [rows[first].tolist(), rows[second].tolist()]
        results.append(
            {
                "suffix_relation_count": length,
                "groups": len(totals),
                "empirical_modal_target_fraction": float(maxima.sum() / n),
                "single_target_group_query_fraction": float(totals[maxima == totals].sum() / n),
                "singleton_group_query_fraction": float((totals == 1).sum() / n),
                "multi_query_groups": int(multi.sum()),
                "multi_query_coverage_fraction": float(multi_n / n),
                "multi_query_empirical_modal_target_fraction": (
                    float(maxima[multi].sum() / multi_n) if multi_n else None
                ),
                "multi_target_groups": int(multi_target.sum()),
                "same_suffix_different_target_example": example,
            }
        )
    return results


def majority_tail_diagnostics(rows: np.ndarray) -> dict:
    """Report the empirical within-split majority target, without fitting a model."""
    if not len(rows):
        return {"target_token": None, "count": 0, "fraction": None, "distinct_targets": 0}
    targets, counts = np.unique(rows[:, -1], return_counts=True)
    index = int(counts.argmax())
    return {
        "target_token": int(targets[index]),
        "count": int(counts[index]),
        "fraction": float(counts[index] / len(rows)),
        "distinct_targets": len(targets),
    }


def data_diagnostics(world: dict) -> dict:
    """Pure data coverage and target distributions; never evaluate a predictor."""
    atomic_n = len(world["atomic"])
    indices = world.get("relevant_atomic_indices")
    if indices is None:
        indices = {name: query_atomic_indices(world, world[name]) for name in SPLITS}
    training_counts = np.bincount(indices["train_composite"].ravel(), minlength=atomic_n)
    id_facts = np.zeros(atomic_n, dtype=bool)
    id_facts[indices["id_atomic"].ravel()] = True
    training_majority = majority_tail_diagnostics(world["train_composite"])
    per_split = {}
    for name in SPLITS:
        rows, edges = world[name], indices[name]
        distinct = np.unique(edges)
        trained = training_counts[edges] > 0
        all_id, all_ood = id_facts[edges].all(axis=1), (~id_facts[edges]).all(axis=1)
        per_split[name] = {
            "n": len(rows),
            "unique_relevant_atomic_facts": len(distinct),
            "relevant_atomic_fact_coverage_of_graph": len(distinct) / atomic_n,
            "all_relevant_facts_in_atomic_training_fraction": 1.0 if len(rows) else None,
            "all_relevant_facts_in_composite_training_n": int(trained.all(axis=1).sum()),
            "any_relevant_fact_in_composite_training_n": int(trained.any(axis=1).sum()),
            "relevant_fact_occurrences_seen_in_composite_training_fraction": (
                float(trained.mean()) if edges.size else None
            ),
            "all_id_queries": int(all_id.sum()),
            "all_ood_queries": int(all_ood.sum()),
            "mixed_queries": int((~all_id & ~all_ood).sum()),
            "unique_relevant_facts_by_hop": [
                len(np.unique(edges[:, j])) for j in range(edges.shape[1])
            ],
            "within_split_majority_tail": majority_tail_diagnostics(rows),
            "train_composite_majority_tail_fraction": (
                float((rows[:, -1] == training_majority["target_token"]).mean())
                if len(rows) and training_majority["target_token"] is not None
                else None
            ),
            "suffix_without_head": suffix_target_diagnostics(rows),
        }
    return {
        "interpretation": {
            "atomic_coverage": "data availability only; does not establish model mastery",
            "ood": "all facts are atomic training facts; none has composite-training experience",
            "suffix": (
                "empirical within-split target concentration; singleton suffixes score 1; "
                "not a predictor, generalization bound, or evidence that the model ignores head"
            ),
            "majority": "within-split descriptive ceiling plus fixed training-majority fraction",
        },
        "atomic_facts": atomic_n,
        "id_facts": int(id_facts.sum()),
        "ood_facts": int((~id_facts).sum()),
        "id_facts_seen_in_composite_training": int((training_counts[id_facts] > 0).sum()),
        "ood_facts_seen_in_composite_training": int((training_counts[~id_facts] > 0).sum()),
        "train_composite_fact_occurrence_counts": training_counts.tolist(),
        "splits": per_split,
    }


class _EpochStream:
    """NumPy-only shuffled epochs, with the existing EpochStream draw convention."""

    def __init__(self, size: int, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.remaining = np.empty(0, dtype=np.int64)

    def take(self, n: int) -> np.ndarray:
        if not n:
            return np.empty(0, dtype=np.int64)
        parts = []
        while n:
            if not len(self.remaining):
                self.remaining = self.rng.permutation(self.size)
            count = min(n, len(self.remaining))
            parts.append(self.remaining[:count])
            self.remaining = self.remaining[count:]
            n -= count
        return np.concatenate(parts)

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "remaining": self.remaining.copy(),
        }

    def load_state_dict(self, state: dict):
        if state["size"] != self.size:
            raise ValueError("Epoch size mismatch")
        remaining = np.asarray(state["remaining"])
        if (
            remaining.ndim != 1
            or remaining.dtype != np.int64
            or np.any((remaining < 0) | (remaining >= self.size))
            or len(np.unique(remaining)) != len(remaining)
        ):
            raise ValueError("Invalid remaining epoch indices")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.remaining = remaining.copy()


class StratifiedStream:
    """Fixed composition per batch over the table [atomics; train_composites].

    Atomic, composite and interleaving draws use independent RNGs. Thus changing
    phi or the composite pool cannot change the atomic sequence or its batch
    positions. Individual strata use shuffled epochs and may cross an epoch
    boundary within one batch. No evaluation data enters either stream.
    """

    def __init__(
        self, atomic_size: int, composite_size: int, batch_size: int, n_atomic: int, seed: int
    ):
        fields = {
            "atomic_size": atomic_size,
            "composite_size": composite_size,
            "batch_size": batch_size,
            "n_atomic": n_atomic,
        }
        for name, value in fields.items():
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if batch_size < 1 or n_atomic > batch_size:
            raise ValueError("Require batch_size > 0 and 0 <= n_atomic <= batch_size")
        if (n_atomic and not atomic_size) or (batch_size > n_atomic and not composite_size):
            raise ValueError("A sampled stratum cannot be empty")
        self.atomic_size, self.composite_size = int(atomic_size), int(composite_size)
        self.batch_size, self.n_atomic = int(batch_size), int(n_atomic)
        self.atomic = _EpochStream(atomic_size, np.random.SeedSequence([int(seed), 145101]))
        self.composite = _EpochStream(composite_size, np.random.SeedSequence([int(seed), 145102]))
        self.interleave = np.random.default_rng(np.random.SeedSequence([int(seed), 145103]))
        self.batches = 0

    def take(self, n: int | None = None) -> np.ndarray:
        if n is not None and n != self.batch_size:
            raise ValueError("StratifiedStream only emits its configured batch_size")
        indices = np.r_[
            self.atomic.take(self.n_atomic),
            self.composite.take(self.batch_size - self.n_atomic) + self.atomic_size,
        ]
        self.batches += 1
        return indices[self.interleave.permutation(self.batch_size)]

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "atomic_size": self.atomic_size,
            "composite_size": self.composite_size,
            "batch_size": self.batch_size,
            "n_atomic": self.n_atomic,
            "atomic": self.atomic.state_dict(),
            "composite": self.composite.state_dict(),
            "interleave_rng": copy.deepcopy(self.interleave.bit_generator.state),
            "batches": self.batches,
        }

    def load_state_dict(self, state: dict):
        if state.get("version") != 1:
            raise ValueError("Unsupported StratifiedStream state version")
        for name in ("atomic_size", "composite_size", "batch_size", "n_atomic"):
            if state[name] != getattr(self, name):
                raise ValueError(f"StratifiedStream {name} mismatch")
        atomic, composite = copy.deepcopy(self.atomic), copy.deepcopy(self.composite)
        interleave = copy.deepcopy(self.interleave)
        atomic.load_state_dict(state["atomic"])
        composite.load_state_dict(state["composite"])
        interleave.bit_generator.state = copy.deepcopy(state["interleave_rng"])
        batches = state["batches"]
        if not isinstance(batches, int) or batches < 0:
            raise ValueError("Invalid completed batch count")
        self.atomic, self.composite, self.interleave = atomic, composite, interleave
        self.batches = batches
