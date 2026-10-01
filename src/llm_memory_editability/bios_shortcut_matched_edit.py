"""Prepared E39 update sets shared by low/high shortcut-reliability worlds.

This file generates and audits data only. It neither trains nor edits a model.
The 96 proposed MLP edits are distinct from the archived E93 experiment.
"""

import numpy as np

from .bios_data import array_hash, rng_for
from .bios_shortcut_control import audit_high_exception

KINDS = ("coherent", "exception")
PHASES = ("low", "high")
BUDGET = {
    "prevalence_levels": 2,
    "worlds": 2,
    "initializations": 2,
    "organizations": 3,
    "models": 24,
    "chains": 2,
    "update_types": 2,
    "supports_per_chain": 1,
    "scopes": ["mlp"],
    "edit_cases": 96,
}
U_STRATA = {
    "0": "original_exception_actual_in_affected_groups",
    "1": "newly_exception_actual_in_affected_groups",
    "2": "other_unchanged_facts_of_affected_people",
    "3": "remaining_same_chain_facts",
    "4": "remaining_cross_chain_or_independent_facts",
}


def make_matched_edit_pair(low, high, chain, support=0):
    """Three roots + 36 actual facts; same IDs and support labels across phases.

    Each selected group contributes six QA-trained and six QA-heldout people
    who remain ordinary in the high-prevalence world. Coherent assigns all 12
    to the new default. Exception assigns three default/three alternative in
    each split. Histograms match ACROSS all three groups, not within a group.
    """
    if chain not in (0, 1) or support != 0:
        raise ValueError("The prepared contract fixes two chains and support zero")
    if low.seed != high.seed or low.n_base != high.n_base:
        raise ValueError("Low/high worlds must share their identity and query layout")
    audit_high_exception(low, high)
    rng = rng_for(low.seed, 931, chain)
    for _ in range(10000):
        groups = rng.choice(64, 3, replace=False)
        if len(np.unique(low.defaults[chain, groups])) == 3:
            break
    else:
        raise RuntimeError("Could not find three groups with distinct default cities")
    selected = np.empty((3, 12), dtype=np.int64)
    conflict = np.zeros((3, 12), dtype=bool)
    conflict[:, :3] = conflict[:, 6:9] = True
    targets = {
        f"{phase}_{kind}": world.answers.copy()
        for phase, world in (("low", low), ("high", high))
        for kind in KINDS
    }
    new_default = low.city_tokens[low.defaults[chain, np.roll(groups, -1)]]
    alternative = low.city_tokens[low.defaults[chain, np.roll(groups, -2)]]
    for index, group in enumerate(groups):
        ordinary = np.flatnonzero((high.memberships[chain] == group) & ~high.exceptions[chain])
        for split, pool in enumerate((low.train_ids[chain], low.heldout_ids[chain])):
            eligible = ordinary[np.isin(low.derived_ids[chain, ordinary], pool)]
            if len(eligible) != 8:
                raise ValueError("Expected eight still-ordinary people in each QA split")
            selected[index, split * 6 : (split + 1) * 6] = rng.permutation(eligible)[:6]
        actual = low.actual_ids[chain, selected[index]]
        derived = low.derived_ids[chain, low.memberships[chain] == group]
        for phase in PHASES:
            for kind in KINDS:
                target = targets[f"{phase}_{kind}"]
                target[low.root_ids[chain, group]] = new_default[index]
                target[derived] = new_default[index]
                target[actual] = new_default[index]
                if kind == "exception":
                    target[actual[conflict[index]]] = alternative[index]
    edited_people = selected.ravel()
    selected_people = np.isin(low.memberships[chain], groups)
    original = selected_people & low.exceptions[chain]
    newly = selected_people & high.exceptions[chain] & ~low.exceptions[chain]
    unedited_ordinary = (
        selected_people & ~high.exceptions[chain] & ~np.isin(np.arange(2048), edited_people)
    )
    e = np.sort(np.concatenate([low.root_ids[chain, groups], low.actual_ids[chain, edited_people]]))
    d = low.derived_ids[chain, selected_people]
    unchanged = np.ones(len(low.answers), dtype=bool)
    unchanged[np.concatenate([e, d])] = False
    affected_facts = (low.person >= 0) & selected_people[np.maximum(low.person, 0)]
    original_actual = np.isin(np.arange(len(low.answers)), low.actual_ids[chain, original])
    newly_actual = np.isin(np.arange(len(low.answers)), low.actual_ids[chain, newly])
    relevant = np.isin(low.relation, [0, 1, 2, 10] if chain == 0 else [7, 8, 9, 11])
    strata = np.full(len(low.answers), -1, dtype=np.int64)
    strata[unchanged & original_actual] = 0
    strata[unchanged & newly_actual] = 1
    strata[unchanged & affected_facts & ~original_actual & ~newly_actual] = 2
    strata[unchanged & ~affected_facts & relevant] = 3
    strata[unchanged & ~affected_facts & ~relevant] = 4
    heldout = []
    for stratum in range(5):
        ids = rng.permutation(np.flatnonzero((strata == stratum) & (low.relation < 10)))
        heldout.extend(ids[: max(1, int(np.ceil(0.2 * len(ids))))])
    heldout.extend(np.flatnonzero(unchanged & (low.relation >= 10)))
    heldout = np.array(sorted(heldout), dtype=np.int64)
    common_pool = np.flatnonzero(unchanged & (low.relation < 10) & (low.answers == high.answers))
    common_pool = np.setdiff1d(common_pool, heldout)
    if len(common_pool) < 4096:
        raise ValueError("Insufficient shared unchanged base facts for R4096")
    replay = rng.choice(common_pool, 4096, replace=False)
    paired_reference = np.sort(low.derived_ids[chain, selected[conflict]])
    pair = {
        **targets,
        "groups": groups,
        "selected_people": selected,
        "alternative_assignments": conflict,
        "new_default": new_default,
        "alternative": alternative,
        "E": e,
        "D": d,
        "E_roots": low.root_ids[chain, groups],
        "E_actual": np.sort(low.actual_ids[chain, edited_people]),
        "E_actual_trained": np.sort(low.actual_ids[chain, selected[:, :6].ravel()]),
        "E_actual_heldout": np.sort(low.actual_ids[chain, selected[:, 6:].ravel()]),
        "D_trained": np.intersect1d(d, low.train_ids[chain]),
        "D_heldout": np.intersect1d(d, low.heldout_ids[chain]),
        "D_edited_actual": np.sort(low.derived_ids[chain, edited_people]),
        "paired_reference_D": paired_reference,
        "paired_reference_D_heldout": np.intersect1d(paired_reference, low.heldout_ids[chain]),
        "exception_conflict_D": paired_reference.copy(),
        "exception_conflict_D_heldout": np.intersect1d(paired_reference, low.heldout_ids[chain]),
        "D_edited_aligned": np.sort(low.derived_ids[chain, selected[~conflict]]),
        "D_original_exception": low.derived_ids[chain, original],
        "D_newly_exception": low.derived_ids[chain, newly],
        "D_remaining_ordinary_unedited": low.derived_ids[chain, unedited_ordinary],
        "R": replay,
        "U_full": np.flatnonzero(unchanged),
        "U_heldout": heldout,
        "U_strata": strata,
    }
    for phase in PHASES:
        for kind in KINDS:
            target = targets[f"{phase}_{kind}"]
            actual_for_d = target[low.actual_ids[chain, selected_people]]
            pair[f"{phase}_{kind}_factual_conflict_D"] = d[target[d] != actual_for_d]
    pair["contract"] = {
        "protocol": "v2.8-p3-shortcut-matched-E39-prepared",
        "world": low.seed,
        "chain": chain,
        "support": support,
        "status": "data_only_not_executed",
        "budget": {**BUDGET, "scopes": BUDGET["scopes"].copy()},
        "model_checkpoint": (
            "width256, original 15360-step learning; not P2 30720-step continuation"
        ),
        "histogram_matching_scope": (
            "all three groups combined; also separately for trained/heldout QA"
        ),
        "coherent_actuals_per_group": "12 new-default (six in each QA split)",
        "exception_actuals_per_group": (
            "6 new-default + 6 alternative (three + three in each QA split)"
        ),
        "paired_reference_semantics": (
            "same 18 people; conflict only in exception, reference only in coherent"
        ),
        "R_semantics": (
            "4096 identical query IDs with identical original truths "
            "and unchanged under all four targets"
        ),
        "U_semantics": (
            "common query IDs; score each phase against its own truth and old-correct coverage"
        ),
        "U_strata": U_STRATA.copy(),
        "historical_E93_comparable": False,
        "low_truth_sha256": array_hash(low.answers),
        "high_truth_sha256": array_hash(high.answers),
        "array_sha256": {key: array_hash(value) for key, value in pair.items()},
    }
    audit_matched_edit_pair(low, high, chain, pair)
    return pair


def audit_matched_edit_pair(low, high, chain, pair):
    """Verify ID, target, split, marginal, primary-subset, and replay contracts."""
    audit_high_exception(low, high)
    if low.seed != high.seed or low.n_base != high.n_base:
        raise ValueError("Low/high world identities differ")
    e, d, selected, groups = (pair[key] for key in ("E", "D", "selected_people", "groups"))
    if len(e) != 39 or len(np.unique(e)) != 39 or len(d) != 96 or len(np.unique(d)) != 96:
        raise ValueError("E39/D96 support changed")
    if len(groups) != 3 or len(np.unique(low.defaults[chain, groups])) != 3:
        raise ValueError("Expected three groups with distinct original defaults")
    if selected.shape != (3, 12) or len(np.unique(selected)) != 36:
        raise ValueError("Expected twelve distinct people in each selected group")
    if high.exceptions[chain, selected].any():
        raise ValueError("Selected edit people must be ordinary in both worlds")
    np.testing.assert_array_equal(
        pair["new_default"], low.city_tokens[low.defaults[chain, np.roll(groups, -1)]]
    )
    np.testing.assert_array_equal(
        pair["alternative"], low.city_tokens[low.defaults[chain, np.roll(groups, -2)]]
    )
    np.testing.assert_array_equal(
        d, low.derived_ids[chain, np.isin(low.memberships[chain], groups)]
    )
    np.testing.assert_array_equal(
        e, np.sort(np.r_[low.root_ids[chain, groups], low.actual_ids[chain, selected.ravel()]])
    )
    conflict = pair["alternative_assignments"]
    if conflict.shape != (3, 12) or conflict.dtype != np.bool_:
        raise ValueError("Invalid alternative assignment mask")
    for index, group in enumerate(groups):
        if not np.all(low.memberships[chain, selected[index]] == group):
            raise ValueError("Selected person belongs to the wrong group")
        for split, pool in enumerate((low.train_ids[chain], low.heldout_ids[chain])):
            people = selected[index, split * 6 : (split + 1) * 6]
            if not np.isin(low.derived_ids[chain, people], pool).all():
                raise ValueError("Selected QA split changed")
            if conflict[index, split * 6 : (split + 1) * 6].sum() != 3:
                raise ValueError("Each group/split must have three default and three alternative")
        root = low.root_ids[chain, group]
        derived = low.derived_ids[chain, low.memberships[chain] == group]
        actual = low.actual_ids[chain, selected[index]]
        for phase in PHASES:
            for kind in KINDS:
                target = pair[f"{phase}_{kind}"]
                if target[root] != pair["new_default"][index] or not np.all(
                    target[derived] == target[root]
                ):
                    raise ValueError(
                        "The three root rotations and their full propagation targets must agree"
                    )
                expected = np.full(12, target[root], dtype=np.int64)
                if kind == "exception":
                    expected[conflict[index]] = pair["alternative"][index]
                np.testing.assert_array_equal(target[actual], expected)
    changed = np.union1d(e, d)
    for phase, world in (("low", low), ("high", high)):
        for kind in KINDS:
            target = pair[f"{phase}_{kind}"]
            np.testing.assert_array_equal(np.flatnonzero(target != world.answers), changed)
    for kind in KINDS:
        np.testing.assert_array_equal(pair[f"low_{kind}"][changed], pair[f"high_{kind}"][changed])
    for _split, pool in (
        ("all", low.derived_ids[chain]),
        ("trained", low.train_ids[chain]),
        ("heldout", low.heldout_ids[chain]),
    ):
        people = selected.ravel()[np.isin(low.derived_ids[chain, selected.ravel()], pool)]
        actual = low.actual_ids[chain, people]
        np.testing.assert_array_equal(
            np.sort(pair["low_coherent"][actual]), np.sort(pair["low_exception"][actual])
        )
    reference = np.sort(low.derived_ids[chain, selected[conflict]])
    np.testing.assert_array_equal(pair["paired_reference_D"], reference)
    np.testing.assert_array_equal(pair["exception_conflict_D"], reference)
    heldout_reference = np.intersect1d(reference, low.heldout_ids[chain])
    np.testing.assert_array_equal(pair["exception_conflict_D_heldout"], heldout_reference)
    np.testing.assert_array_equal(pair["paired_reference_D_heldout"], heldout_reference)
    if len(reference) != 18 or len(heldout_reference) != 9:
        raise ValueError("Expected eighteen primary exception conflicts, nine heldout")
    partition = [
        pair[key]
        for key in (
            "D_edited_actual",
            "D_original_exception",
            "D_newly_exception",
            "D_remaining_ordinary_unedited",
        )
    ]
    if [len(ids) for ids in partition] != [36, 6, 42, 12]:
        raise ValueError("Incorrect fixed propagation strata")
    np.testing.assert_array_equal(np.sort(np.concatenate(partition)), d)
    affected = np.isin(low.memberships[chain], groups)
    expected_subsets = {
        "E_roots": low.root_ids[chain, groups],
        "E_actual": np.sort(low.actual_ids[chain, selected.ravel()]),
        "E_actual_trained": np.sort(low.actual_ids[chain, selected[:, :6].ravel()]),
        "E_actual_heldout": np.sort(low.actual_ids[chain, selected[:, 6:].ravel()]),
        "D_trained": np.intersect1d(d, low.train_ids[chain]),
        "D_heldout": np.intersect1d(d, low.heldout_ids[chain]),
        "D_edited_actual": np.sort(low.derived_ids[chain, selected.ravel()]),
        "D_edited_aligned": np.sort(low.derived_ids[chain, selected[~conflict]]),
        "D_original_exception": low.derived_ids[chain, affected & low.exceptions[chain]],
        "D_newly_exception": low.derived_ids[
            chain, affected & high.exceptions[chain] & ~low.exceptions[chain]
        ],
    }
    for key, expected in expected_subsets.items():
        np.testing.assert_array_equal(pair[key], expected)
    for phase in PHASES:
        for kind in KINDS:
            target = pair[f"{phase}_{kind}"]
            conflicts = d[target[d] != target[low.actual_ids[chain, affected]]]
            np.testing.assert_array_equal(pair[f"{phase}_{kind}_factual_conflict_D"], conflicts)
    u = np.setdiff1d(np.arange(len(low.answers)), changed)
    np.testing.assert_array_equal(pair["U_full"], u)
    np.testing.assert_array_equal(pair["U_strata"] >= 0, np.isin(np.arange(len(low.answers)), u))
    replay = pair["R"]
    if len(replay) != 4096 or len(np.unique(replay)) != 4096 or np.any(replay >= low.n_base):
        raise ValueError("R must contain 4096 unique base-fact IDs")
    if not np.isin(replay, u).all() or np.intersect1d(replay, pair["U_heldout"]).size:
        raise ValueError("Replay leaks changed support/propagation or heldout retention")
    if not np.isin(pair["U_heldout"], u).all():
        raise ValueError("Heldout retention includes changed targets")
    np.testing.assert_array_equal(low.answers[replay], high.answers[replay])
    for phase in PHASES:
        for kind in KINDS:
            np.testing.assert_array_equal(pair[f"{phase}_{kind}"][replay], low.answers[replay])
    for key, value in pair.items():
        if isinstance(value, np.ndarray) and pair["contract"]["array_sha256"][key] != array_hash(
            value
        ):
            raise ValueError(f"Matched update array provenance changed: {key}")
    return {
        "passed": True,
        "E": 39,
        "D": 96,
        "exception_conflict_D": 18,
        "exception_conflict_D_heldout": 9,
        "R": 4096,
        "budget_edit_cases": 96,
        "histograms_match_across_three_groups_and_within_QA_split": True,
    }
