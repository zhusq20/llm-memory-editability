"""Two prospectively fixed E39 supports; preparation never launches a model.

Support zero exactly preserves development selection. Support one uses a
separate deterministic stream and excludes the first support's entire groups.
The original E39 audit validates both; all training and target budgets match.
"""

import numpy as np

from .bios_data import array_hash, rng_for
from .bios_shortcut_control import audit_high_exception
from .bios_shortcut_matched_edit import (
    KINDS,
    PHASES,
    U_STRATA,
    audit_matched_edit_pair,
)
from .bios_shortcut_matched_edit import (
    make_matched_edit_pair as development_pair,
)

BUDGET = {
    "prevalence_levels": 2,
    "worlds": 8,
    "initializations": 2,
    "organizations": 3,
    "models": 96,
    "chains": 2,
    "update_types": 2,
    "supports_per_chain": 2,
    "scopes": ["mlp"],
    "edit_cases": 768,
}


def make_confirmation_edit_pair(low, high, chain, support=0):
    """Three roots + 36 actual facts; same IDs and support labels across phases.

    Each selected group contributes six QA-trained and six QA-heldout people
    who remain ordinary in the high-prevalence world. Coherent assigns all 12
    to the new default. Exception assigns three default/three alternative in
    each split. Histograms match ACROSS all three groups, not within a group.
    """
    if chain not in (0, 1) or support not in (0, 1):
        raise ValueError("Confirmation fixes two chains and two disjoint supports")
    if low.seed != high.seed or low.n_base != high.n_base:
        raise ValueError("Low/high worlds must share their identity and query layout")
    audit_high_exception(low, high)
    first = development_pair(low, high, chain)
    if support == 0:
        first["contract"].update(
            protocol="v2.9-shortcut-behavior-confirmation-E39",
            status="prospectively_fixed_confirmation_support",
            budget=BUDGET.copy(),
            model_checkpoint="width256, independent-world 15360-step low/high learning",
        )
        return first
    eligible_groups = np.setdiff1d(np.arange(64), first["groups"])
    rng = rng_for(low.seed, 931, chain + 2 * support)
    for _ in range(10000):
        groups = rng.choice(eligible_groups, 3, replace=False)
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
        "protocol": "v2.9-shortcut-behavior-confirmation-E39",
        "world": low.seed,
        "chain": chain,
        "support": support,
        "status": "prospectively_fixed_confirmation_support",
        "budget": {**BUDGET, "scopes": BUDGET["scopes"].copy()},
        "model_checkpoint": ("width256, independent-world 15360-step low/high learning"),
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
