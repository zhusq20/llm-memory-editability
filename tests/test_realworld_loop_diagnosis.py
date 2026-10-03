"""Check that new support cannot train on held-out knowledge compositions."""

import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_realworld_loop.py"
SPEC = importlib.util.spec_from_file_location("diagnosis", PATH)
diagnosis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnosis)


def fixture():
    edges = {
        "a": ["A", "father", "B"],
        "b": ["B", "father", "C"],
        "c": ["C", "father", "D"],
        "d": ["D", "father", "E"],
        "ood": ["E", "father", "F"],
    }
    atoms = {k: {"id": k, "edge": e} for k, e in edges.items()}
    training = [
        {"atom_ids": ["a", "b"], "edges": [edges["a"], edges["b"]]},
        {"atom_ids": ["c", "d"], "edges": [edges["c"], edges["d"]]},
    ]
    return atoms, training


def test_new_support_uses_only_id_and_excludes_original():
    atoms, training = fixture()
    assert diagnosis.augmentation_pairs(atoms, training, []) == [("b", "c")]


def test_heldout_chain_and_same_answer_are_excluded():
    atoms, training = fixture()
    heldout = [{"atom_ids": ["b", "c"], "edges": [atoms["b"]["edge"], atoms["c"]["edge"]]}]
    assert diagnosis.augmentation_pairs(atoms, training, heldout) == []
    heldout[0]["atom_ids"] = ["alternative-first", "alternative-second"]
    assert diagnosis.augmentation_pairs(atoms, training, heldout) == []


def test_canonical_input_has_head_relations_but_no_bridge_or_answer():
    atom = {"edge": ["Q1", "father", "Q2"], "question": "Who is the father of Alice?"}
    subject = diagnosis.subject_of(atom)
    assert subject == "Alice"
    question = diagnosis.canonical_question(subject, ["father", "place of birth"])
    assert question == "Starting entity: Alice. Relation sequence: father -> place of birth."
    assert "Q2" not in question
