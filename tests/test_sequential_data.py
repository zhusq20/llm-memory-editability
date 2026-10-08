import copy

import pytest

from llm_memory_editability.sequential_data import (
    SplitSettings,
    audit_split,
    build_split,
    safe_atom_ids,
)


def source_graph(size=40):
    atoms = {}
    for head in range(size):
        for relation, distance in [("r1", 1), ("r2", 3)]:
            key = f"{head}:{relation}"
            atoms[key] = {
                "id": key,
                "edge": [f"e{head}", relation, f"e{(head + distance) % size}"],
                "question": key,
                "answer": f"e{(head + distance) % size}",
                "aliases": [],
                "encoded": {"prefix": [1], "target": [2, 0], "input": [1, 2]},
            }
    chains = []
    for first in atoms.values():
        bridge = int(first["edge"][2][1:])
        for relation in ["r1", "r2"]:
            second = atoms[f"{bridge}:{relation}"]
            chains.append(
                {
                    "id": first["id"] + "/" + second["id"],
                    "atom_ids": [first["id"], second["id"]],
                    "edges": [first["edge"], second["edge"]],
                    "question": first["id"] + "/" + relation,
                    "answer": second["answer"],
                    "aliases": [],
                    "encoded": {"prefix": [1], "target": [2, 0], "input": [1, 2]},
                }
            )
    return {
        "atoms": list(atoms.values()),
        "train_compositions": chains,
        "evaluation_compositions": [],
    }


@pytest.fixture
def dataset():
    return build_split(source_graph(), SplitSettings("test", 4, 32, 2))


def test_split_is_reproducible_source_preserving_and_name_familiar(dataset):
    source = source_graph()
    assert dataset == build_split(source, SplitSettings("test", 4, 32, 2))
    assert dataset["audit"]["passed"]
    assert dataset["audit"]["bb_all_entities_seen_in_a"]
    assert dataset["audit"]["evaluation_counts"] == {"AA": 2, "BA": 2, "AB": 2, "BB": 4}
    source_atoms = {row["id"]: row for row in source["atoms"]}
    for row in dataset["atoms"]:
        assert {key: value for key, value in row.items() if key != "subset"} == source_atoms[
            row["id"]
        ]


def test_alias_variant_of_heldout_chain_cannot_enter_training(dataset):
    changed = copy.deepcopy(dataset)
    row = copy.deepcopy(changed["evaluation_compositions"][0])
    row["id"] = "alternate-question-id"
    row["question"] = "An alternative wording for the same evidence chain"
    changed["train_compositions"].append(row)
    with pytest.raises(ValueError, match="held-out semantic chain"):
        audit_split(changed)


def test_a_composition_cannot_expose_new_atom(dataset):
    changed = copy.deepcopy(dataset)
    original = next(
        row
        for row in source_graph()["train_compositions"]
        if set(row["atom_ids"]) & set(dataset["stage_b_atom_ids"])
        and row["id"] not in {item["id"] for item in dataset["evaluation_compositions"]}
    )
    changed["train_compositions"].append(original)
    with pytest.raises(ValueError, match="B fact leaked"):
        audit_split(changed)


def test_explicit_inverse_and_equivalent_answers_are_excluded():
    atoms = {
        "b": {"edge": ["person", "father", "dad"]},
        "inverse": {"edge": ["dad", "child", "person"]},
        "same_answer": {"edge": ["person", "parent", "dad"]},
        "safe": {"edge": ["person", "country", "place"]},
    }
    assert safe_atom_ids(atoms, {"b"}) == {"safe"}


def test_evaluation_role_and_truth_are_checked(dataset):
    changed = copy.deepcopy(dataset)
    changed["evaluation_compositions"][-1]["sequential_role"] = "AA"
    with pytest.raises(ValueError, match="Incorrect sequential role"):
        audit_split(changed)
    changed = copy.deepcopy(dataset)
    changed["evaluation_compositions"][0]["edges"][0][2] = "false-bridge"
    with pytest.raises(ValueError, match="disagree"):
        audit_split(changed)
