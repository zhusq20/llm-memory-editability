"""Paired fact-use assignments and constituent-occurrence repetition controls.

The same random graph and held-out queries are used in every arm.  A/B change
which random fact cohort may occur in composition training.  repA/repB replace
each selected role-composition slot with its two constituent atomic queries;
their half-weight losses preserve slot weight, not information or computation.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping

import numpy as np

from .grok_depth_data import build_world as build_graph_world
from .grok_loop_data import _EpochStream

GROUP_NAMES = ("BG", "A", "B")
ARMS = ("A", "B", "repA", "repB")
ROW_KINDS = ("base_atomic", "background_composite", "role_composite", "repetition_atomic")


def _digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fact_counts(indices: np.ndarray, n_atomic: int) -> np.ndarray:
    return np.bincount(indices[indices >= 0], minlength=n_atomic).astype(np.int64)


def build_experiment(spec: Mapping) -> dict:
    """Build one graph, one global reserve, and paired composition pools.

    Defaults are 128 entities, 16 relations, outdegree 8, and a 10% Bernoulli
    complete-query reserve.  A/B cohorts each contain floor(n_atomic/4) facts;
    all remaining facts form the shared background.  Cohort assignment and the
    reserve have independent RNGs and never depend on predictions.  Role pools
    are uniformly downsampled to their smaller available size.  Background
    paths are used exhaustively.  Empty sampled strata raise without resampling.
    """
    seed = int(spec.get("world_seed", spec.get("seed", 146001)))
    entities = int(spec.get("entities", 128))
    relations = int(spec.get("relations", 16))
    degree = int(spec.get("degree", 8))
    reserve_fraction = float(spec.get("test_fraction", 0.1))
    if spec.get("hops", 2) != 2:
        raise ValueError("This comparison is defined only for two hops")
    if not np.isfinite(reserve_fraction) or not 0 < reserve_fraction < 1:
        raise ValueError("test_fraction must lie strictly between zero and one")
    base = build_graph_world(
        seed, entities, relations, degree, phi=0, id_fraction=1, id_test_fraction=0
    )
    atomic = base["atomic"].copy()
    n_atomic = len(atomic)
    cohort_size = n_atomic // 4
    if not cohort_size:
        raise ValueError("At least four atomic facts are required")
    # All paths use the stable per-head block ordering of the inherited graph.
    second = ((atomic[:, -1, None] - 2) * degree + np.arange(degree)).reshape(-1)
    first = np.repeat(np.arange(n_atomic, dtype=np.int64), degree)
    all_edges = np.c_[first, second]
    all_rows = np.c_[atomic[first, :2], atomic[second, 1:]]
    reserve_rng = np.random.default_rng(np.random.SeedSequence([seed, 146101]))
    heldout = reserve_rng.random(len(all_rows)) < reserve_fraction
    assignment_rng = np.random.default_rng(np.random.SeedSequence([seed, 146102]))
    assignment = assignment_rng.permutation(n_atomic)
    groups = np.zeros(n_atomic, dtype=np.int64)
    groups[assignment[:cohort_size]] = 1
    groups[assignment[cohort_size : 2 * cohort_size]] = 2
    path_groups = groups[all_edges]
    background_indices = np.flatnonzero(~heldout & (path_groups == 0).all(axis=1))
    role_candidates = {}
    for name, group in (("A", 1), ("B", 2)):
        eligible = ((path_groups == 0) | (path_groups == group)).all(axis=1)
        role_candidates[name] = np.flatnonzero(
            ~heldout & eligible & (path_groups == group).any(axis=1)
        )
    role_size = min(map(len, role_candidates.values()))
    if not len(background_indices) or not role_size:
        raise ValueError("Empty training stratum; retain the supplied world, do not reroll")
    selected = {}
    for name, group in (("A", 1), ("B", 2)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 146103, group]))
        selected[name] = rng.choice(role_candidates[name], size=role_size, replace=False)
    exp = {
        "atomic": atomic,
        "fact_groups": groups,
        "all_composite": all_rows,
        "all_composite_fact_indices": all_edges,
        "heldout_mask": heldout,
        "test_composite": all_rows[heldout],
        "test_composite_fact_indices": all_edges[heldout],
        "background_composite": all_rows[background_indices],
        "background_fact_indices": all_edges[background_indices],
        "role_A_composite": all_rows[selected["A"]],
        "role_A_fact_indices": all_edges[selected["A"]],
        "role_B_composite": all_rows[selected["B"]],
        "role_B_fact_indices": all_edges[selected["B"]],
        "metadata": {
            "world_seed": seed,
            "seed": seed,
            "entities": entities,
            "relations": relations,
            "degree": degree,
            "hops": 2,
            "vocab_size": 2 + entities + relations,
            "test_fraction": reserve_fraction,
            "cohort_assignment_rng": "SeedSequence([world_seed,146102])",
            "complete_query_reserve_rng": "SeedSequence([world_seed,146101])",
            "role_pool_rng": "SeedSequence([world_seed,146103,group])",
            "group_names": list(GROUP_NAMES),
            "group_sizes": [int((groups == group).sum()) for group in range(3)],
            "role_candidate_sizes": {name: len(rows) for name, rows in role_candidates.items()},
            "role_pool_size": role_size,
            "background_pool_size": len(background_indices),
            "test_queries": int(heldout.sum()),
            "all_queries": len(all_rows),
            "source_graph_sha256": base["metadata"]["dataset_sha256"],
            "source_graph_constructor": "grok_depth_data.build_world (atomic graph only)",
            "role_meaning": "random eligibility; actual per-fact occurrence counts also reported",
            "selection": "one supplied graph; no data or model-score based seed selection",
        },
    }
    for name in ("background", "role_A", "role_B"):
        counts = _fact_counts(exp[name + "_fact_indices"], n_atomic)
        exp["metadata"][name + "_fact_occurrence_counts"] = counts.tolist()
    exp["metadata"]["test_group_counts"] = {
        f"{left}_{right}": int(
            (heldout & (path_groups[:, 0] == i) & (path_groups[:, 1] == j)).sum()
        )
        for i, left in enumerate(GROUP_NAMES)
        for j, right in enumerate(GROUP_NAMES)
    }
    audit = audit_experiment(exp)
    exp["metadata"]["dataset_sha256"] = audit["dataset_sha256"]
    return exp


def audit_experiment(exp: dict) -> dict:
    """Reconstruct truth and verify the global reserve and swapped eligibility."""
    atomic, groups, meta = exp["atomic"], exp["fact_groups"], exp["metadata"]
    if atomic.dtype != np.int64 or groups.dtype != np.int64 or len(groups) != len(atomic):
        raise ValueError("Invalid atomic or group arrays")
    if np.any((groups < 0) | (groups > 2)):
        raise ValueError("Invalid fact group")
    lookup = {(int(h), int(r)): (int(t), i) for i, (h, r, t) in enumerate(atomic)}
    if len(lookup) != len(atomic):
        raise ValueError("Duplicate atomic queries")
    heldout_queries = {tuple(row[:-1]) for row in exp["test_composite"]}
    if len(heldout_queries) != len(exp["test_composite"]):
        raise ValueError("Duplicate held-out query")
    arrays = {"atomic": atomic, "fact_groups": groups}
    for name in ("all", "test", "background", "role_A", "role_B"):
        rows = exp[name + "_composite"]
        index_key = (
            name + "_composite_fact_indices" if name in ("all", "test") else name + "_fact_indices"
        )
        recorded = exp[index_key]
        expected = np.empty((len(rows), 2), dtype=np.int64)
        seen = set()
        for i, (h, r1, r2, t) in enumerate(rows):
            query = (int(h), int(r1), int(r2))
            if query in seen:
                raise ValueError(f"Duplicate query in {name}")
            seen.add(query)
            try:
                bridge, first = lookup[int(h), int(r1)]
                target, second = lookup[bridge, int(r2)]
            except KeyError as error:
                raise ValueError(f"Missing path fact in {name}") from error
            if target != t:
                raise ValueError(f"Wrong path target in {name}")
            expected[i] = first, second
            if name in ("background", "role_A", "role_B") and query in heldout_queries:
                raise ValueError("Held-out query leaked into training")
        if not np.array_equal(recorded, expected):
            raise ValueError(f"Incorrect fact indices in {name}")
        if name == "background" and np.any(groups[expected] != 0):
            raise ValueError("Background pool uses a treatment fact")
        if name in ("role_A", "role_B"):
            role = 1 if name == "role_A" else 2
            actual_groups = groups[expected]
            if np.any((actual_groups != 0) & (actual_groups != role)):
                raise ValueError("Role pool uses the opposite cohort")
            if np.any(~(actual_groups == role).any(axis=1)):
                raise ValueError("Role pool contains a background-only path")
        arrays[name] = rows
        arrays[index_key] = recorded
    if len(exp["all_composite"]) != len(atomic) * int(meta["degree"]):
        raise ValueError("Incomplete full path enumeration")
    if not np.array_equal(exp["all_composite"][exp["heldout_mask"]], exp["test_composite"]):
        raise ValueError("Global held-out mask disagrees with test queries")
    if len(exp["role_A_composite"]) != len(exp["role_B_composite"]):
        raise ValueError("Paired role pools differ in size")
    digest = _digest(arrays)
    recorded_digest = meta.get("dataset_sha256")
    if recorded_digest is not None and recorded_digest != digest:
        raise ValueError("Recorded dataset hash differs from arrays")
    return {"passed": True, "dataset_sha256": digest, "test_queries": len(heldout_queries)}


def build_arm_world(exp: dict, arm: str) -> dict:
    """Return packed-table ingredients, evaluation sets and exposure provenance.

    Concatenate ``pack_rows(part['rows'], 4)`` in ``training_parts`` order.
    ``table_weights`` and ``table_fact_indices`` align with that concatenation.
    An atomic row has one fact id and -1; a composition row has two ids.  Thus
    raw fact occurrences can be accumulated without interpreting token counts
    or claiming equal supervision/information.  rep arms keep the same role
    path reference pool for paired evaluation, but do not train those queries.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    role_name = arm[-1]
    repeat = arm.startswith("rep")
    atomic = exp["atomic"]
    n_atomic = len(atomic)
    atom_indices = np.c_[np.arange(n_atomic), np.full(n_atomic, -1)].astype(np.int64)
    role_edges = exp[f"role_{role_name}_fact_indices"]
    role_rows = exp[f"role_{role_name}_composite"]
    if repeat:
        last_rows = atomic[role_edges.ravel()]
        last_edges = np.c_[role_edges.ravel(), np.full(role_edges.size, -1)].astype(np.int64)
    else:
        last_rows, last_edges = role_rows, role_edges
    parts = [
        {
            "name": "atomic",
            "rows": atomic,
            "weights": np.ones(n_atomic, dtype=np.float32),
            "fact_indices": atom_indices,
        },
        {
            "name": "background",
            "rows": exp["background_composite"],
            "weights": np.ones(len(exp["background_composite"]), dtype=np.float32),
            "fact_indices": exp["background_fact_indices"],
        },
        {
            "name": "repetition" if repeat else "role",
            "rows": last_rows,
            "weights": np.full(len(last_rows), 0.5 if repeat else 1, dtype=np.float32),
            "fact_indices": last_edges,
        },
    ]
    evaluation = {"atomic": atomic}
    for group, name in enumerate(GROUP_NAMES):
        evaluation[f"atomic_{name}"] = atomic[exp["fact_groups"] == group]
    evaluation["test_composite"] = exp["test_composite"]
    query_groups = exp["fact_groups"][exp["test_composite_fact_indices"]]
    for i, left in enumerate(GROUP_NAMES):
        for j, right in enumerate(GROUP_NAMES):
            mask = (query_groups[:, 0] == i) & (query_groups[:, 1] == j)
            evaluation[f"test_{left}_{right}"] = exp["test_composite"][mask]
    evaluation["train_background"] = exp["background_composite"]
    evaluation["role_A_reference"] = exp["role_A_composite"]
    evaluation["role_B_reference"] = exp["role_B_composite"]
    train_composite = (
        exp["background_composite"]
        if repeat
        else np.concatenate([exp["background_composite"], role_rows])
    )
    evaluation["train_composite"] = train_composite
    group_id = 1 if role_name == "A" else 2
    eligible = (exp["fact_groups"] == 0) | ((exp["fact_groups"] == group_id) & (not repeat))
    used_counts = _fact_counts(
        np.concatenate(
            [exp["background_fact_indices"], role_edges if not repeat else role_edges[:0]]
        ),
        n_atomic,
    )
    world = {
        "arm": arm,
        "atomic": atomic,
        "id_atomic": atomic[eligible],
        "ood_atomic": atomic[~eligible],
        "background_composite": exp["background_composite"],
        "role_composite": role_rows,
        "train_composite": train_composite,
        "test_composite": exp["test_composite"],
        "test_full_composite": exp["test_composite"],
        "training_parts": parts,
        "table_weights": np.concatenate([part["weights"] for part in parts]),
        "table_fact_indices": np.concatenate([part["fact_indices"] for part in parts]),
        "table_row_kinds": np.concatenate(
            [
                np.full(len(part["rows"]), kind, dtype=np.int64)
                for part, kind in zip(parts, [0, 1, 3 if repeat else 2], strict=True)
            ]
        ),
        "evaluation": evaluation,
        "metadata": {
            **copy.deepcopy(exp["metadata"]),
            "arm": arm,
            "repetition_control": repeat,
            "role_name": role_name,
            "row_kind_names": list(ROW_KINDS),
            "training_composite_fact_occurrence_counts": used_counts.tolist(),
            "eligible_facts_without_actual_composite_occurrence": int(
                (eligible & (used_counts == 0)).sum()
            ),
            "evaluation_note": "same global heldout and nine cohort-pair strata in every arm",
            "repetition_note": (
                "two constituent atomics per role slot, each loss weight 1/2; matches raw "
                "constituent occurrences, not information, direct supervision or compute"
            ),
        },
    }
    return world


class UsageStream:
    """Emit paired base/background/role slots, expanding rep slots into two rows."""

    def __init__(self, world, seed, n_atomic=32, n_background=112, n_role=112):
        for name, value in (
            ("n_atomic", n_atomic),
            ("n_background", n_background),
            ("n_role", n_role),
        ):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.arm = world["arm"]
        self.repeat = self.arm.startswith("rep")
        self.atomic_size = len(world["atomic"])
        self.background_size = len(world["background_composite"])
        self.role_size = len(world["role_composite"])
        self.n_atomic, self.n_background, self.n_role = (
            int(n_atomic),
            int(n_background),
            int(n_role),
        )
        if min(self.atomic_size, self.background_size, self.role_size) < 1:
            raise ValueError("Sampled strata cannot be empty")
        self.nominal_batch_size = self.n_atomic + self.n_background + self.n_role
        self.batch_size = self.nominal_batch_size + (self.n_role if self.repeat else 0)
        self.atomic = _EpochStream(self.atomic_size, np.random.SeedSequence([int(seed), 146201]))
        self.background = _EpochStream(
            self.background_size, np.random.SeedSequence([int(seed), 146202])
        )
        self.role = _EpochStream(self.role_size, np.random.SeedSequence([int(seed), 146203]))
        self.interleave = np.random.default_rng(np.random.SeedSequence([int(seed), 146204]))
        self.batches = 0

    def take(self, n=None):
        if n is not None and n != self.batch_size:
            raise ValueError("UsageStream emits its configured actual batch size")
        atoms = self.atomic.take(self.n_atomic)
        background = self.background.take(self.n_background) + self.atomic_size
        role = self.role.take(self.n_role)
        if self.repeat:
            role = (role[:, None] * 2 + np.arange(2)).ravel()
        role = role + self.atomic_size + self.background_size
        indices = np.r_[atoms, background, role]
        self.batches += 1
        return indices[self.interleave.permutation(self.batch_size)]

    def state_dict(self):
        return {
            "version": 1,
            "arm": self.arm,
            "n_atomic": self.n_atomic,
            "n_background": self.n_background,
            "n_role": self.n_role,
            "atomic": self.atomic.state_dict(),
            "background": self.background.state_dict(),
            "role": self.role.state_dict(),
            "interleave_rng": copy.deepcopy(self.interleave.bit_generator.state),
            "batches": self.batches,
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported UsageStream state version")
        for name in ("arm", "n_atomic", "n_background", "n_role"):
            if state[name] != getattr(self, name):
                raise ValueError(f"UsageStream {name} mismatch")
        if not isinstance(state["batches"], int) or state["batches"] < 0:
            raise ValueError("Invalid batch count")
        replacements = {}
        for name in ("atomic", "background", "role"):
            replacements[name] = copy.deepcopy(getattr(self, name))
            replacements[name].load_state_dict(state[name])
        interleave = copy.deepcopy(self.interleave)
        interleave.bit_generator.state = copy.deepcopy(state["interleave_rng"])
        for name, stream in replacements.items():
            setattr(self, name, stream)
        self.interleave = interleave
        self.batches = state["batches"]
