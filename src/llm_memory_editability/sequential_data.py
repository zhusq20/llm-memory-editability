"""Deterministic, source-grounded splits for sequential knowledge acquisition.

The source examples are existing audited 2Wiki adaptations.  We change only their
training availability, never their facts, names, answers, or tokenization.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .realworld_composition_data import normalize_answer, order


@dataclass(frozen=True)
class SplitSettings:
    split: str
    bb_count: int = 64
    train_count: int = 512
    evaluation_count: int = 64
    max_shared_b_fact: int = 4


def chain_key(row):
    return tuple(tuple(edge) for edge in row["edges"])


def operation(row):
    return tuple(edge[1] for edge in row["edges"])


def entities(row):
    return {value for edge in row["edges"] for value in (edge[0], edge[2])}


def balanced_order(rows, salt):
    """Round-robin relation pairs rather than favor the largest answer families."""
    groups = defaultdict(list)
    for row in rows:
        groups[operation(row)].append(row)
    for group in groups.values():
        group.sort(key=lambda row: order(row["id"], salt))
    pairs = sorted(groups, key=lambda pair: order(pair, salt))
    for position in range(max(map(len, groups.values()), default=0)):
        for pair in pairs:
            if position < len(groups[pair]):
                yield groups[pair][position]


def safe_atom_ids(atoms, b_ids):
    """Exclude exact answer-equivalent addresses and explicit inverse facts.

    This guards known label leakage, not every possible real-world inference.
    Entity names may appear in unrelated facts, as required for name familiarity.
    """
    direct = {(atoms[key]["edge"][0], atoms[key]["edge"][2]) for key in b_ids}
    inverse = set()
    for key in b_ids:
        head, relation, tail = atoms[key]["edge"]
        if relation == "spouse":
            inverse.add((tail, "spouse", head))
        elif relation in {"father", "mother"}:
            inverse.add((tail, "child", head))
        elif relation == "child":
            inverse.update({(tail, "father", head), (tail, "mother", head)})
    return {
        key
        for key, row in atoms.items()
        if key not in b_ids
        and (row["edge"][0], row["edge"][2]) not in direct
        and tuple(row["edge"]) not in inverse
    }


def build_split(source, settings, labels=None):
    """Select BB first, reserve real A witnesses, then add A-only practice."""
    atoms = {row["id"]: row for row in source["atoms"]}
    incident = defaultdict(set)
    for key, row in atoms.items():
        for entity in (row["edge"][0], row["edge"][2]):
            incident[entity].add(key)
    label_owners = defaultdict(set)
    for entity, label in (labels or {}).items():
        label_owners[normalize_answer(label)].add(entity)

    def unambiguous(row):
        return all(
            len(label_owners[normalize_answer(labels[entity])]) == 1
            for entity in entities(row)
            if labels and entity in labels
        )

    unique = {}
    for row in source["train_compositions"] + source["evaluation_compositions"]:
        if row.get("structural_errors") or row.get("known_direct_shortcut"):
            continue
        if not unambiguous(row):
            continue
        key = chain_key(row)
        if key not in unique or row["id"] < unique[key]["id"]:
            unique[key] = row
    rows = list(unique.values())
    counts = Counter(operation(row) for row in rows)
    candidates = [
        row
        for row in rows
        if counts[operation(row)] >= 12
        and all(incident[entity] - set(row["atom_ids"]) for entity in entities(row))
    ]
    b_ids, witness_ids, selected_bb = set(), set(), []
    b_uses = Counter()
    for row in balanced_order(candidates, settings.split + ":bb:"):
        proposed = b_ids | set(row["atom_ids"])
        if proposed & witness_ids:
            continue
        if any(b_uses[key] >= settings.max_shared_b_fact for key in row["atom_ids"]):
            continue
        safe = safe_atom_ids(atoms, proposed)
        if not witness_ids <= safe:
            continue
        witnesses = set()
        for entity in entities(row):
            options = incident[entity] & safe
            if not options:
                break
            # Reuse an existing witness when possible; otherwise use a fixed hash.
            options = (options & witness_ids) or options
            witnesses.add(min(options, key=lambda key: order(key, settings.split + ":witness:")))
        else:
            b_ids = proposed
            witness_ids |= witnesses
            selected_bb.append(row)
            b_uses.update(row["atom_ids"])
            if len(selected_bb) == settings.bb_count:
                break
    if len(selected_bb) < settings.bb_count:
        raise ValueError(f"Only {len(selected_bb)} valid BB chains for {settings.split}")

    safe = safe_atom_ids(atoms, b_ids)
    b_head_answers = {(atoms[key]["edge"][0], atoms[key]["edge"][2]) for key in b_ids}

    def admissible_a_chain(row):
        return (
            set(row["atom_ids"]) <= safe
            and (row["edges"][0][0], row["edges"][1][2]) not in b_head_answers
        )

    aa = [row for row in rows if admissible_a_chain(row)]
    available_operations = Counter(operation(row) for row in aa)
    required_operations = {operation(row) for row in selected_bb}
    if any(available_operations[pair] < 2 for pair in required_operations):
        raise ValueError("A target operation lacks A-only training and evaluation examples")
    # Reserve held-out AA chains and real training witnesses for both old roles.
    # BA/AB likewise require the old constituent to have prior use in that role.
    by_first, by_second = defaultdict(list), defaultdict(list)
    for row in aa:
        by_first[row["atom_ids"][0]].append(row)
        by_second[row["atom_ids"][1]].append(row)
    chosen_train, train_keys, heldout_keys = [], set(), set()
    pools = {"AA": [], "BA": [], "AB": [], "BB": selected_bb}

    def choose_witness(options, forbidden):
        options = [row for row in options if chain_key(row) not in forbidden]
        if not options:
            return None
        existing = [row for row in options if chain_key(row) in train_keys]
        return min(existing or options, key=lambda row: order(row["id"], settings.split + ":role:"))

    def add_training(row):
        key = chain_key(row)
        if key not in train_keys:
            chosen_train.append(row)
            train_keys.add(key)

    heldout_operations = Counter()
    for row in balanced_order(aa, settings.split + ":eval:AA"):
        key = chain_key(row)
        if (
            key in train_keys
            or heldout_operations[operation(row)] >= available_operations[operation(row)] - 2
        ):
            continue
        forbidden = heldout_keys | {key}
        first = choose_witness(by_first[row["atom_ids"][0]], forbidden)
        second = choose_witness(by_second[row["atom_ids"][1]], forbidden)
        if first is None or second is None:
            continue
        pools["AA"].append(row)
        heldout_keys.add(key)
        heldout_operations[operation(row)] += 1
        add_training(first)
        add_training(second)
        if len(pools["AA"]) == settings.evaluation_count:
            break
    available_ids = safe | b_ids
    for role in ["BA", "AB"]:
        candidates = [
            row
            for row in rows
            if set(row["atom_ids"]) <= available_ids
            and "".join("B" if key in b_ids else "A" for key in row["atom_ids"]) == role
            and operation(row) in available_operations
        ]
        for row in balanced_order(candidates, settings.split + ":eval:" + role):
            old_index = 1 if role == "BA" else 0
            index = by_second if old_index else by_first
            witness = choose_witness(index[row["atom_ids"][old_index]], heldout_keys)
            if witness is None:
                continue
            pools[role].append(row)
            add_training(witness)
            if len(pools[role]) == settings.evaluation_count:
                break
    for role in ["AA", "BA", "AB"]:
        if len(pools[role]) < min(8, settings.evaluation_count):
            raise ValueError(f"Insufficient role-experienced {role}: {len(pools[role])}")
    required_operations |= {operation(row) for pool in pools.values() for row in pool}
    operation_support = Counter(operation(row) for row in chosen_train)
    for row in balanced_order(aa, settings.split + ":operation-support:"):
        pair = operation(row)
        target = min(8, max(1, available_operations[pair] // 2))
        if (
            pair in required_operations
            and operation_support[pair] < target
            and chain_key(row) not in heldout_keys
        ):
            before = len(chosen_train)
            add_training(row)
            operation_support[pair] += len(chosen_train) - before
    if len(chosen_train) > settings.train_count:
        raise ValueError(
            f"Training size below required role/operation support: {len(chosen_train)}"
        )
    target_entities = {entity for row in selected_bb for entity in entities(row)}
    relevant_aa = [row for row in aa if entities(row) & target_entities]
    for pool in [relevant_aa, aa]:
        if len(chosen_train) == settings.train_count:
            break
        for row in balanced_order(pool, settings.split + ":train:"):
            if chain_key(row) in train_keys | heldout_keys:
                continue
            add_training(row)
            if len(chosen_train) == settings.train_count:
                break
    trained_operations = {operation(row) for row in chosen_train}
    if not required_operations <= trained_operations:
        raise ValueError(f"A training lacks operations: {required_operations - trained_operations}")

    a_ids = witness_ids | {
        key
        for row in chosen_train + [row for pool in pools.values() for row in pool]
        for key in row["atom_ids"]
        if key not in b_ids
    }
    final_atoms = []
    for key in sorted(a_ids | b_ids):
        row = copy.deepcopy(atoms[key])
        row["subset"] = "A" if key in a_ids else "B"
        final_atoms.append(row)
    training = copy.deepcopy(chosen_train)
    for row in training:
        row["sequential_role"] = row["role"] = "AA"
    evaluation = []
    for role, pool in pools.items():
        for original in pool:
            row = copy.deepcopy(original)
            row["sequential_role"] = row["role"] = role
            row["all_entities_seen_in_a"] = entities(row) <= {
                entity for key in a_ids for entity in (atoms[key]["edge"][0], atoms[key]["edge"][2])
            }
            row["operation_seen"] = operation(row) in trained_operations
            evaluation.append(row)
    data = {
        "schema": "sequential-transfer-v1",
        "split": settings.split,
        "settings": asdict(settings),
        "atoms": final_atoms,
        "train_compositions": training,
        "evaluation_compositions": evaluation,
        "stage_a_atom_ids": sorted(a_ids),
        "stage_b_atom_ids": sorted(b_ids),
        "entity_witness_atom_ids": sorted(witness_ids),
        "panels": {
            "atomic_A": sorted(a_ids),
            "atomic_B": sorted(b_ids),
            "train_composition": [row["id"] for row in training],
            **{role: [row["id"] for row in pool] for role, pool in pools.items()},
        },
        "selection_audit": {
            "source_unique_eligible_chains": len(rows),
            "source_known_entity_bb_candidates": len(candidates),
            "selection_uses_model_outputs": False,
            "selection_uses_original_official_split": False,
            "known_entity_definition": "Each entity occurs in at least one real A atomic fact.",
            "facts_are_original_source_edges": True,
            "training_and_evaluation_use_original_questions": True,
            "semantic_scope": (
                "Reuse prior structural/semantic exclusions; not an exhaustive new human review. "
                "Exclude exact B head-answer supervision and explicit spouse/parent inverses. "
                "Other possible real-world inferences are not ruled out."
            ),
        },
    }
    data["audit"] = audit_split(data)
    return data


def audit_split(data):
    atoms = {row["id"]: row for row in data["atoms"]}
    a_ids, b_ids = set(data["stage_a_atom_ids"]), set(data["stage_b_atom_ids"])
    if a_ids & b_ids or a_ids | b_ids != set(atoms):
        raise ValueError("Atomic A/B partition is not disjoint and exhaustive")
    a_entities = {
        entity for key in a_ids for entity in (atoms[key]["edge"][0], atoms[key]["edge"][2])
    }
    trains = data["train_compositions"]
    evaluations = data["evaluation_compositions"]
    train_keys = {chain_key(row) for row in trains}
    if train_keys & {chain_key(row) for row in evaluations}:
        raise ValueError("A held-out semantic chain appears in composition training")
    if not all(set(row["atom_ids"]) <= a_ids for row in trains):
        raise ValueError("B fact leaked through A composition training")
    if not a_ids <= safe_atom_ids(atoms, b_ids):
        raise ValueError("An equivalent or explicit inverse B fact leaked into A")
    b_answers = {(atoms[key]["edge"][0], atoms[key]["edge"][2]) for key in b_ids}
    if any((row["edges"][0][0], row["edges"][1][2]) in b_answers for row in trains):
        raise ValueError("A composition label directly teaches a B head-answer pair")
    train_operations = {operation(row) for row in trains}
    trained_first = {row["atom_ids"][0] for row in trains}
    trained_second = {row["atom_ids"][1] for row in trains}
    for row in trains + evaluations:
        if not set(row["atom_ids"]) <= set(atoms):
            raise ValueError("Missing required atomic knowledge")
        if [atoms[key]["edge"] for key in row["atom_ids"]] != row["edges"]:
            raise ValueError("Atom IDs disagree with the real evidence chain")
    for row in evaluations:
        role = "".join("A" if key in a_ids else "B" for key in row["atom_ids"])
        if role != row["sequential_role"]:
            raise ValueError("Incorrect sequential role")
        if operation(row) not in train_operations:
            raise ValueError("An evaluation operation was not trained on A")
        if role[0] == "A" and row["atom_ids"][0] not in trained_first:
            raise ValueError("An old first-hop fact lacks prior composition use")
        if role[1] == "A" and row["atom_ids"][1] not in trained_second:
            raise ValueError("An old second-hop fact lacks prior composition use")
        if role == "BB" and not entities(row) <= a_entities:
            raise ValueError("BB contains an entity absent from A")
    counts = Counter(row["sequential_role"] for row in evaluations)
    return {
        "passed": True,
        "atomic_A": len(a_ids),
        "atomic_B": len(b_ids),
        "train_compositions": len(trains),
        "evaluation_counts": dict(counts),
        "bb_all_entities_seen_in_a": all(
            entities(row) <= a_entities for row in evaluations if row["sequential_role"] == "BB"
        ),
        "b_fact_composition_training_overlap": 0,
        "heldout_semantic_chain_training_overlap": 0,
        "all_evaluation_a_facts_have_prior_role_experience": True,
        "relation_pairs": {
            role: dict(
                Counter(
                    " -> ".join(operation(row))
                    for row in evaluations
                    if row["sequential_role"] == role
                )
            )
            for role in counts
        },
        "bb_unique_atomic_facts": len(
            {
                key
                for row in evaluations
                if row["sequential_role"] == "BB"
                for key in row["atom_ids"]
            }
        ),
    }
