"""Equal-presentation bioS corpora differing only in document organization.

A joins a person's six facts to the true employer default. B keeps the person's
facts together but supplies an unrelated company's default. C keeps B's exact
actual/default pair and moves the other five personal facts to distinct company
origins. No fact is falsified; every condition uses the same presentation pool.
"""

import numpy as np

from .bios_data import EOS, N_BASE, N_COMPANIES, N_PEOPLE, RELATIONS, array_hash, rng_for

CONDITIONS = ("A", "B", "C")
CONDITION_NAMES = {"A": "relation_linked", "B": "person_grouped", "C": "shuffled"}
PERSONAL_RELATIONS = (0, 2, 3, 4, 5, 6)
N_PRESENTATIONS = N_PEOPLE * len(RELATIONS)


def _fact_lookup(world):
    """Look up fact IDs without depending on the current contiguous numbering."""
    lookup = np.full((7, N_PEOPLE), -1, dtype=np.int64)
    for relation in range(7):
        ids = np.flatnonzero(world.relation[:N_BASE] == relation)
        subjects = world.company[ids] if relation == 1 else world.person[ids]
        lookup[relation, subjects] = ids
    return lookup


def make_documents(world, condition, organization_seed=0):
    """Return (2048, 7) fact IDs in canonical relation order.

    Organization is fixed across epochs. Random group ordering and within-group
    assignment use a separate seed, shared across all three conditions. C's
    seven company origins are distinct, including its unlinked default.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown organization condition: {condition}")
    if organization_seed < 0:
        raise ValueError("organization_seed must be nonnegative")
    lookup = _fact_lookup(world)
    documents = np.empty((N_PEOPLE, 7), dtype=np.int64)
    people = np.arange(N_PEOPLE)
    for relation in PERSONAL_RELATIONS:
        documents[:, relation] = lookup[relation, people]
    documents[:, 1] = lookup[1, world.employers]
    if condition == "A":
        return documents

    rng = rng_for(world.seed, 101, organization_seed)
    members = np.stack(
        [rng.permutation(np.flatnonzero(world.employers == c)) for c in range(N_COMPANIES)]
    )
    if members.shape != (N_COMPANIES, 32):
        raise ValueError("Organization protocol requires 64 companies with 32 people each")
    # Independent company cycles per employee-rank lane avoid a stable unrelated
    # company partner. Every cycle is a derangement and visits every company.
    cycles = np.stack([rng.permutation(N_COMPANIES) for _ in range(32)])
    lane = np.arange(32)[:, None]
    anchors = members[cycles, lane]
    documents[anchors, 1] = lookup[1, np.roll(cycles, -1, axis=1)]
    if condition == "C":
        for shift, relation in enumerate((0, 3, 4, 5, 6), start=2):
            donor_members = members.copy()
            # Each donor person contributes exactly one fact of this relation.
            for row in donor_members:
                rng.shuffle(row)
            donors = donor_members[np.roll(cycles, -shift, axis=1), lane]
            documents[anchors, relation] = lookup[relation, donors]
    return documents


def apply_epoch(documents, world_seed, epoch):
    """Apply a shared row permutation and cyclic slot rotation (zero-based epoch).

    Every seven complete epochs, every presentation visits all seven positions.
    The permutation depends on the world and epoch, never on the condition.
    """
    documents = np.asarray(documents)
    if documents.shape != (N_PEOPLE, 7) or epoch < 0:
        raise ValueError("Expected canonical (2048, 7) documents and nonnegative epoch")
    rows = rng_for(world_seed, 102, epoch).permutation(N_PEOPLE)
    return np.roll(documents[rows], epoch % 7, axis=1)


def document_epoch(world, condition, epoch, organization_seed=0):
    return apply_epoch(make_documents(world, condition, organization_seed), world.seed, epoch)


def render_documents(world, ids):
    """Render facts with cross-fact causal attention, answer/EOS-only supervision.

    Each row is one document, independently batched; no inter-document attention
    is required. Positions select logits that predict the next answer/EOS token.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[1] != 7 or np.any((ids < 0) | (ids >= N_BASE)):
        raise ValueError("Expected seven base fact IDs per document")
    tokens = np.empty((*ids.shape, 6), dtype=np.int64)
    tokens[:, :, :4] = world.prompts[ids, :4]
    tokens[:, :, 4] = world.answers[ids]
    tokens[:, :, 5] = EOS
    positions = (np.arange(7)[:, None] * 6 + np.array([3, 4])).ravel()
    labels = np.stack([world.answers[ids], np.full_like(ids, EOS)], axis=-1)
    return {
        "tokens": tokens.reshape(len(ids), 42),
        "positions": np.broadcast_to(positions, (len(ids), 14)).copy(),
        "labels": labels.reshape(len(ids), 14),
        "fact_ids": ids.copy(),
    }


def presentation_ids(world, documents):
    """Assign each shared presentation to exactly one canonical document slot.

    Personal presentation IDs are 7*person+relation. The 32 default copies use
    7*employee+1, assigned in sorted row order. This assignment is made before
    epoch rotation, and permits a fixed text/template for every presentation.
    """
    documents = np.asarray(documents, dtype=np.int64)
    if documents.shape != (N_PEOPLE, 7):
        raise ValueError("Expected canonical (2048, 7) documents")
    if not np.all(world.relation[documents] == np.arange(7)):
        raise ValueError("Presentation assignment requires canonical relation order")
    ids = world.person[documents] * 7 + world.relation[documents]
    for company in range(N_COMPANIES):
        rows = np.flatnonzero(world.company[documents[:, 1]] == company)
        people = np.flatnonzero(world.employers == company)
        if len(rows) != len(people):
            raise ValueError("Each company default must have 32 presentations")
        ids[rows, 1] = people * 7 + 1
    if not np.array_equal(np.sort(ids.ravel()), np.arange(N_PRESENTATIONS)):
        raise ValueError("Presentations were lost or duplicated")
    return ids


def audit_organizations(world, documents):
    """Fail on confounds and report the measured organization intervention."""
    if set(documents) != set(CONDITIONS):
        raise ValueError("Audit requires A, B and C")
    expected_counts = np.where(world.relation[:N_BASE] == 1, 32, 1)
    expected_answers = np.bincount(
        world.answers[:N_BASE], weights=expected_counts, minlength=world.vocab_size
    ).astype(np.int64)
    conditions = {}
    for condition, ids in documents.items():
        if ids.shape != (N_PEOPLE, 7) or not np.all(world.relation[ids] == np.arange(7)):
            raise ValueError("Invalid document shape or relation slots")
        counts = np.bincount(ids.ravel(), minlength=N_BASE)
        if not np.array_equal(counts, expected_counts):
            raise ValueError("Per-fact presentation counts differ")
        answers = np.bincount(world.answers[ids].ravel(), minlength=world.vocab_size)
        if not np.array_equal(answers, expected_answers):
            raise ValueError("Answer-token presentation counts differ")
        companies, people = world.company[ids], world.person[ids]
        same_company = companies[:, :, None] == companies[:, None, :]
        same_person = (people[:, :, None] == people[:, None, :]) & (people[:, :, None] >= 0)
        personal = people[:, PERSONAL_RELATIONS]
        all_personal_same = np.all(personal == personal[:, :1], axis=1)
        if condition in ("A", "B") and not all_personal_same.all():
            raise ValueError("Person aggregation was broken")
        if condition == "A" and not np.all(companies == companies[:, :1]):
            raise ValueError("A contains a broken employer/default link")
        if condition in ("B", "C") and np.any(companies[:, 1] == companies[:, 2]):
            raise ValueError("Unrelated default accidentally matches actual person's employer")
        if condition == "C" and np.any(
            np.sort(companies, axis=1)[:, 1:] == np.sort(companies, axis=1)[:, :-1]
        ):
            raise ValueError("C contains two facts with the same company origin")
        pids = presentation_ids(world, ids)
        slots = np.zeros((N_BASE, 7), dtype=np.int64)
        for epoch in range(7):
            arranged = apply_epoch(ids, world.seed, epoch)
            for slot in range(7):
                np.add.at(slots[:, slot], arranged[:, slot], 1)
        if not np.array_equal(slots, np.repeat(expected_counts[:, None], 7, axis=1)):
            raise ValueError("Per-fact positions are not balanced over seven epochs")
        default_actual_same = world.answers[ids[:, 1]] == world.answers[ids[:, 2]]
        conditions[condition] = {
            "name": CONDITION_NAMES[condition],
            "documents": len(ids),
            "presentations": int(ids.size),
            "unique_base_facts": int(np.count_nonzero(counts)),
            "document_tokens": 42,
            "supervised_tokens_per_document": 14,
            "total_input_tokens_per_epoch": int(ids.size * 6),
            "total_supervised_tokens_per_epoch": int(ids.size * 2),
            "facts_sha256": array_hash(ids),
            "presentation_ids_sha256": array_hash(pids),
            "exposure_sha256": array_hash(counts),
            "answer_histogram_sha256": array_hash(answers),
            "seven_epoch_fact_slot_sha256": array_hash(slots),
            "company_origin_cooccurrence": same_company.mean(axis=0).tolist(),
            "person_origin_cooccurrence": same_person.mean(axis=0).tolist(),
            "all_personal_same_fraction": float(all_personal_same.mean()),
            "default_actual_same_answer_count": int(default_actual_same.sum()),
            "default_actual_same_answer_rate": float(default_actual_same.mean()),
            "relation_counts": {
                relation: int((world.relation[ids] == i).sum())
                for i, relation in enumerate(RELATIONS)
            },
        }
    if not np.array_equal(documents["B"][:, 1:3], documents["C"][:, 1:3]):
        raise ValueError("B and C must share exactly the same default/actual fact pairs")
    return {
        "schema": "bios-organization-v1",
        "world_seed": world.seed,
        "truth_sha256": array_hash(world.answers),
        "relation_slot_order": list(RELATIONS),
        "same_truth_and_per_fact_presentations": True,
        "position_balance_period_epochs": 7,
        "same_default_actual_pairs_B_C": True,
        "templates_and_aliases": "symbolic tokens fixed; English view uses common presentations",
        "conditions": conditions,
    }


def english_fact(world, fact_id, variant=0):
    """Readable supplemental view; this text is not used in symbolic training."""
    if not 0 <= fact_id < N_BASE:
        raise ValueError("Expected a base fact")
    relation = int(world.relation[fact_id])
    subject = (
        world.names["companies"][int(world.company[fact_id])]
        if relation == 1
        else (world.names["people"][int(world.person[fact_id])])
    )
    kind, raw = world.token_labels[int(world.answers[fact_id])].split(":", 1)
    fields = {
        "company": "companies",
        "city": "cities",
        "university": "universities",
        "major": "majors",
    }
    value = raw if kind == "date" else world.names[fields[kind]][int(raw)]
    templates = (
        ("{s} works for {v}.", "{s}'s employer is {v}.", "{v} employs {s}."),
        (
            "{s}'s default work city is {v}.",
            "{s} uses {v} as the default work city.",
            "The default work city for {s} is {v}.",
        ),
        (
            "{s}'s actual work city is {v}.",
            "{s} actually works in {v}.",
            "The actual work city of {s} is {v}.",
        ),
        ("{s} was born on {v}.", "{s}'s birth date is {v}.", "The birth date of {s} is {v}."),
        ("{s} was born in {v}.", "{s}'s birth city is {v}.", "The birth city of {s} is {v}."),
        ("{s} attended {v}.", "{s}'s university is {v}.", "{s} studied at {v}."),
        (
            "{s} majored in {v}.",
            "{s}'s academic major is {v}.",
            "The academic major of {s} is {v}.",
        ),
    )
    return templates[relation][variant % 3].format(s=subject, v=value)
