"""Predeclared splits and query-level capacity diagnostics."""

import numpy as np

from .bios_data import N_BASE, N_QUERIES, rng_for


def qa_split(world, mode):
    if mode == "all":
        return np.arange(N_BASE, N_QUERIES), np.array([], dtype=np.int64)
    if mode != "half":
        raise ValueError("Unknown QA split")
    rng = rng_for(world.seed, 910)
    train, heldout = [], []
    for company in np.unique(world.employers):
        for exception in (False, True):
            people = np.flatnonzero((world.employers == company) & (world.exceptions == exception))
            people = rng.permutation(people)
            train.extend((N_BASE + people[: len(people) // 2]).tolist())
            heldout.extend((N_BASE + people[len(people) // 2 :]).tolist())
    return np.sort(train), np.sort(heldout)


def split_metrics(world, arrays, train_ids, heldout_ids):
    correct, nll = arrays["correct"], arrays["value_nll"]
    # The derivation needs the person's employer and that employer's default city.
    # Build by semantic IDs rather than assuming a base-fact row layout.
    employer = {int(world.person[i]): i for i in np.flatnonzero(world.relation == 0)}
    defaults = {int(world.company[i]): i for i in np.flatnonzero(world.relation == 1)}
    result = {}
    for name, ids in (("trained", train_ids), ("heldout", heldout_ids)):
        result[f"derived_{name}_count"] = len(ids)
        result[f"derived_{name}_accuracy"] = float(correct[ids].mean()) if len(ids) else None
        result[f"derived_{name}_nll"] = float(nll[ids].mean()) if len(ids) else None
        eligible = np.array(
            [
                i
                for i in ids
                if correct[employer[int(world.person[i])]]
                and correct[defaults[int(world.company[i])]]
            ],
            dtype=np.int64,
        )
        result[f"derived_{name}_known_components_count"] = len(eligible)
        result[f"derived_{name}_known_components_accuracy"] = (
            float(correct[eligible].mean()) if len(eligible) else None
        )
    return result
