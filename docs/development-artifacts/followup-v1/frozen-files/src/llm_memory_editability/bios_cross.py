"""Two-chain organization crossover: shared facts, symmetric relational tasks.

This is a new development world, not an alteration of the archived A/B/C runs.
Company and project each have 64 groups of 32 people and two old exceptions.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bios_data import ANS, BOS, EOS, N_BASE, N_PEOPLE, array_hash, load_world, rng_for, write_json

CONDITIONS = ("company", "project", "neither")
CHAINS = ("company", "project")
SLOTS = 10


@dataclass
class CrossWorld:
    seed: int
    prompts: np.ndarray
    lengths: np.ndarray
    answers: np.ndarray
    relation: np.ndarray
    person: np.ndarray
    memberships: np.ndarray
    defaults: np.ndarray
    exceptions: np.ndarray
    membership_ids: np.ndarray
    root_ids: np.ndarray
    actual_ids: np.ndarray
    derived_ids: np.ndarray
    train_ids: np.ndarray
    heldout_ids: np.ndarray
    city_tokens: np.ndarray
    token_labels: list
    n_base: int

    @property
    def vocab_size(self):
        return len(self.token_labels)


def make_cross_world(seed, source_root="data/bios-organization-v1"):
    old = load_world(Path(source_root) / f"world-{seed}")
    labels = (
        old.token_labels
        + [f"project:{i}" for i in range(64)]
        + ["relation:project", "relation:project_default", "relation:project_actual"]
    )
    token = {name: i for i, name in enumerate(labels)}
    rng = rng_for(seed, 901)
    memberships = np.stack([old.employers, rng.permutation(np.repeat(np.arange(64), 32))])
    defaults = np.stack([old.defaults, rng.permutation(old.defaults)])
    exceptions = np.stack([old.exceptions, np.zeros(N_PEOPLE, dtype=bool)])
    actual = defaults[1, memberships[1]].copy()
    for group in range(64):
        people = rng.choice(np.flatnonzero(memberships[1] == group), 2, replace=False)
        exceptions[1, people] = True
        choices = np.unique(defaults[1][defaults[1] != defaults[1, group]])
        actual[people] = rng.choice(choices, 2)
    n_base = N_BASE + 2 * N_PEOPLE + 64
    size = n_base + 2 * N_PEOPLE
    prompts = np.zeros((size, 5), dtype=np.int64)
    answers = np.zeros(size, dtype=np.int64)
    lengths = np.full(size, 4, dtype=np.int64)
    relation = np.zeros(size, dtype=np.int64)
    person = np.full(size, -1, dtype=np.int64)
    prompts[:N_BASE], answers[:N_BASE] = old.prompts[:N_BASE], old.answers[:N_BASE]
    relation[:N_BASE], person[:N_BASE] = old.relation[:N_BASE], old.person[:N_BASE]
    membership_ids = np.stack([np.arange(N_PEOPLE), N_BASE + np.arange(N_PEOPLE)])
    root_ids = np.stack([N_PEOPLE + np.arange(64), N_BASE + N_PEOPLE + np.arange(64)])
    actual_ids = np.stack(
        [N_PEOPLE + 64 + np.arange(N_PEOPLE), N_BASE + N_PEOPLE + 64 + np.arange(N_PEOPLE)]
    )
    person_tokens = np.array([token[f"person:{i}"] for i in range(N_PEOPLE)])
    new_values = [
        np.array([token[f"project:{i}"] for i in memberships[1]]),
        old.city_tokens[defaults[1]],
        old.city_tokens[actual],
    ]
    for rel, ids, values in zip(
        (7, 8, 9), (membership_ids[1], root_ids[1], actual_ids[1]), new_values, strict=True
    ):
        subjects = (
            np.array([token[f"project:{i}"] for i in range(64)]) if rel == 8 else person_tokens
        )
        rtoken = token[
            ("relation:project", "relation:project_default", "relation:project_actual")[rel - 7]
        ]
        prompts[ids, :4] = np.column_stack(
            [np.full(len(ids), BOS), subjects, np.full(len(ids), rtoken), np.full(len(ids), ANS)]
        )
        answers[ids], relation[ids] = values, rel
        if rel != 8:
            person[ids] = np.arange(N_PEOPLE)
    derived_ids = n_base + np.arange(2 * N_PEOPLE).reshape(2, N_PEOPLE)
    trained, heldout = [], []
    for chain in range(2):
        ids = derived_ids[chain]
        prompts[ids] = np.column_stack(
            [
                np.full(N_PEOPLE, BOS),
                person_tokens,
                np.full(N_PEOPLE, token["relation:employer" if chain == 0 else "relation:project"]),
                np.full(
                    N_PEOPLE,
                    token["relation:default_city" if chain == 0 else "relation:project_default"],
                ),
                np.full(N_PEOPLE, ANS),
            ]
        )
        answers[ids] = old.city_tokens[defaults[chain, memberships[chain]]]
        lengths[ids], relation[ids], person[ids] = 5, 10 + chain, np.arange(N_PEOPLE)
        split_rng = rng_for(seed, 902, chain)
        included = []
        for group in range(64):
            for exception in (False, True):
                candidates = np.flatnonzero(
                    (memberships[chain] == group) & (exceptions[chain] == exception)
                )
                included.extend(split_rng.permutation(candidates)[: len(candidates) // 2])
        trained.append(np.sort(ids[included]))
        heldout.append(np.setdiff1d(ids, trained[-1]))
    # Randomize the complete vocabulary, so appended token numbers carry no task order.
    permutation = np.arange(len(labels))
    permutation[4:] = rng_for(seed, 903).permutation(permutation[4:])
    new_labels = [""] * len(labels)
    for index, label in enumerate(labels):
        new_labels[permutation[index]] = label
    return CrossWorld(
        seed,
        permutation[prompts],
        lengths,
        permutation[answers],
        relation,
        person,
        memberships,
        defaults,
        exceptions,
        membership_ids,
        root_ids,
        actual_ids,
        derived_ids,
        np.stack(trained),
        np.stack(heldout),
        permutation[old.city_tokens],
        new_labels,
        n_base,
    )


def documents(world, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown crossover condition")
    docs = np.empty((N_PEOPLE, SLOTS), dtype=np.int64)
    for rel in range(SLOTS):
        if rel in (1, 8):
            continue
        ids = np.flatnonzero(world.relation[: world.n_base] == rel)
        docs[world.person[ids], rel] = ids
    for chain, slot in enumerate((1, 8)):
        membership = world.memberships[chain]
        group = membership.copy()
        if condition != CHAINS[chain]:
            rng = rng_for(world.seed, 904, chain)
            members = np.stack(
                [rng.permutation(np.flatnonzero(membership == g)) for g in range(64)]
            )
            for lane in range(32):
                cycle = rng.permutation(64)
                group[members[cycle, lane]] = np.roll(cycle, -1)
        docs[:, slot] = world.root_ids[chain, group]
    return docs


def epoch_documents(docs, seed, epoch):
    rows = rng_for(seed, 905, epoch).permutation(N_PEOPLE)
    return np.roll(docs[rows], epoch % SLOTS, axis=1)


def render(world, ids):
    tokens = np.empty((*ids.shape, 6), dtype=np.int64)
    tokens[:, :, :4] = world.prompts[ids, :4]
    tokens[:, :, 4], tokens[:, :, 5] = world.answers[ids], EOS
    positions = (np.arange(SLOTS)[:, None] * 6 + np.array([3, 4])).ravel()
    labels = np.stack([world.answers[ids], np.full_like(ids, EOS)], axis=-1)
    return {
        "tokens": tokens.reshape(len(ids), SLOTS * 6),
        "positions": np.broadcast_to(positions, (len(ids), SLOTS * 2)).copy(),
        "labels": labels.reshape(len(ids), SLOTS * 2),
    }


def qa_schedule(world, steps):
    streams = []
    for chain in range(2):
        rng = rng_for(world.seed, 906, chain)
        pool = world.train_ids[chain]
        streams.append(
            np.concatenate(
                [rng.permutation(pool) for _ in range((steps * 20 + len(pool) - 1) // len(pool))]
            )[: steps * 20].reshape(steps, 20)
        )
    return np.concatenate(streams, axis=1)


def audit(world):
    expected = np.ones(world.n_base, dtype=np.int64)
    expected[world.root_ids.ravel()] = 32
    report = {}
    for condition in CONDITIONS:
        docs = documents(world, condition)
        if not np.array_equal(np.bincount(docs.ravel(), minlength=world.n_base), expected):
            raise ValueError("Unequal fact exposure")
        personal = world.person[docs[:, [0, 2, 3, 4, 5, 6, 7, 9]]]
        if not np.all(personal == np.arange(N_PEOPLE)[:, None]):
            raise ValueError("Person grouping changed")
        for chain, slot in enumerate((1, 8)):
            linked = docs[:, slot] == world.root_ids[chain, world.memberships[chain]]
            if not np.all(linked == (condition == CHAINS[chain])):
                raise ValueError("Incorrect linked/unlinked manipulation")
        slots = np.zeros((world.n_base, SLOTS), dtype=np.int64)
        for epoch in range(SLOTS):
            arranged = epoch_documents(docs, world.seed, epoch)
            np.add.at(slots, (arranged, np.arange(SLOTS)[None, :]), 1)
        if not np.array_equal(slots, np.repeat(expected[:, None], SLOTS, axis=1)):
            raise ValueError("Unequal position exposure")
        report[condition] = {
            "documents_sha256": array_hash(docs),
            "exposure_sha256": array_hash(expected),
        }
    for chain in range(2):
        if not np.array_equal(np.bincount(world.memberships[chain]), np.full(64, 32)):
            raise ValueError("Unequal group sizes")
        for group in range(64):
            members = world.memberships[chain] == group
            if world.exceptions[chain, members].sum() != 2:
                raise ValueError("Unequal exceptions")
        if len(np.intersect1d(world.train_ids[chain], world.heldout_ids[chain])):
            raise ValueError("QA leakage")
        truth = world.city_tokens[world.defaults[chain, world.memberships[chain]]]
        np.testing.assert_array_equal(world.answers[world.derived_ids[chain]], truth)
    np.testing.assert_array_equal(np.sort(world.defaults[0]), np.sort(world.defaults[1]))
    return {
        "passed": True,
        "world": world.seed,
        "conditions": report,
        "base_facts": world.n_base,
        "queries": len(world.answers),
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "QA_train_per_chain": 1024,
        "QA_heldout_per_chain": 1024,
    }


def save_world(world, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "world.npz",
        **{key: value for key, value in vars(world).items() if isinstance(value, np.ndarray)},
    )
    write_json(
        directory / "metadata.json",
        {"seed": world.seed, "n_base": world.n_base, "token_labels": world.token_labels},
    )
    write_json(directory / "audit.json", audit(world))


def edit_pair(world, chain, support=0):
    rng = rng_for(world.seed, 907, chain * 100 + support)
    while True:
        groups = rng.choice(64, 3, replace=False)
        if len(set(world.defaults[chain, groups])) == 3:
            break
    coherent, exception = world.answers.copy(), world.answers.copy()
    conflict_people = []
    for j, group in enumerate(groups):
        new, alt = world.city_tokens[world.defaults[chain, groups[[(j + 1) % 3, (j + 2) % 3]]]]
        root = world.root_ids[chain, group]
        coherent[root] = exception[root] = new
        members = rng.permutation(
            np.flatnonzero((world.memberships[chain] == group) & ~world.exceptions[chain])
        )
        ids = world.actual_ids[chain, members]
        coherent[ids] = new
        exception[ids[:15]], exception[ids[15:]] = new, alt
        conflict_people.extend(members[15:])
        derived = world.derived_ids[chain, world.memberships[chain] == group]
        coherent[derived] = exception[derived] = new
    changed = coherent != world.answers
    np.testing.assert_array_equal(changed, exception != world.answers)
    e, d = np.flatnonzero(changed[: world.n_base]), np.flatnonzero(changed & (world.relation >= 10))
    assert len(e) == 93 and len(d) == 96
    np.testing.assert_array_equal(np.sort(coherent[e]), np.sort(exception[e]))
    # Disjoint retention strata: old exceptions, other facts of affected people,
    # remaining task-chain facts, and remaining independent/cross-chain facts.
    selected_people = np.isin(world.memberships[chain], groups)
    selected_facts = (world.person >= 0) & selected_people[np.maximum(world.person, 0)]
    old_exceptions = np.zeros(len(world.answers), dtype=bool)
    old_exceptions[world.actual_ids[chain, selected_people & world.exceptions[chain]]] = True
    relevant = np.isin(world.relation, [0, 1, 2, 10] if chain == 0 else [7, 8, 9, 11])
    strata = np.full(len(world.answers), -1, dtype=np.int64)
    strata[~changed & old_exceptions] = 0
    strata[~changed & selected_facts & ~old_exceptions] = 1
    strata[~changed & ~selected_facts & relevant] = 2
    strata[~changed & ~selected_facts & ~relevant] = 3
    assert np.array_equal(strata >= 0, ~changed)
    heldout, replay_pool = [], []
    for g in range(4):
        ids = rng.permutation(np.flatnonzero((strata == g) & (world.relation < 10)))
        n = max(1, int(np.ceil(0.2 * len(ids))))
        heldout.extend(ids[:n])
        replay_pool.extend(ids[n:])
    heldout.extend(np.flatnonzero((strata >= 0) & (world.relation >= 10)))
    replay = rng.choice(replay_pool, min(4096, len(replay_pool)), replace=False)
    return {
        "coherent": coherent,
        "exception": exception,
        "E": e,
        "D": d,
        "conflict_D": world.derived_ids[chain, conflict_people],
        "strata": strata,
        "heldout": np.array(sorted(heldout)),
        "replay": replay,
        "groups": groups,
    }
