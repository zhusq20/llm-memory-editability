"""Scientific contracts for the original-paper reproduction."""

import copy
from pathlib import Path

import pytest

from llm_memory_editability.paper_reproduction import (
    decode_unmapped_tokens,
    prepare,
    read_json,
    score_predictions,
    validate_data,
)

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "docs/development-artifacts/implicit-reasoning-paper-reproduction-v1"
    / "upstream/notebooks/data_preparation.ipynb"
)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    destination = tmp_path / "data"
    audit = prepare(NOTEBOOK, destination, entities=30, relations=40, degree=10)
    return destination, audit


def test_paper_count_and_paired_compositions(prepared):
    root, audit = prepared
    only = read_json(root / "only_ii/train.json")
    plus = read_json(root / "ii_plus_id/train.json")
    # An accidentally float-valued count would select every candidate in the notebook.
    assert len(only) == round(7.2 * 285) == 2052
    assert audit["candidate_train_ii"] > len(only)
    assert plus[285:] == only
    assert read_json(root / "only_ii/test.json") == read_json(root / "ii_plus_id/test.json")
    assert audit["evaluation_counts"]["OOD Triples"] == 15
    assert audit["same_ii_and_probes"]


def test_all_queries_keep_truth_and_held_out_roles(prepared):
    _, audit = prepared
    assert audit["passed"]
    assert audit["atomic_truths"] == 300
    assert audit["composition_truths"] > 2052


def test_existing_dataset_cannot_be_overwritten(prepared):
    root, _ = prepared
    with pytest.raises(FileExistsError):
        prepare(NOTEBOOK, root)


def test_wrong_hop_role_is_rejected():
    atomic = {"input_text": "<e_0><r_0>", "target_text": "<e_0><r_0><e_0></a>"}
    composition = {"input_text": "<e_0><r_0><r_0>", "target_text": "<e_0><r_0><r_0><e_0></a>"}
    groups = {"ID Triples": [atomic], "OOD Triples": [], "Train-II": [composition]}
    train = {"only_ii": [composition], "ii_plus_id": [atomic, composition]}
    assert validate_data(groups, train, expected_id=1, expected_ood=0, expected_ii=1)["passed"]
    bad = copy.deepcopy(groups)
    bad["Test-OI"] = [composition]
    with pytest.raises(AssertionError):
        validate_data(bad, train, expected_id=1, expected_ood=0, expected_ii=1)


def test_paper_canonicalization_does_not_hide_missing_eos():
    rows = [
        {
            "type": "Test-II",
            "target_text": "answer</a>",
            "model_output": "answer</a>",
            "strict_answer_eos": False,
        }
    ]
    metrics = score_predictions(rows)["Test-II"]
    assert metrics["paper_accuracy"] == 1
    assert metrics["strict_answer_eos_accuracy"] == 0


def test_padding_class_is_an_incorrect_printable_token():
    from transformers import GPT2Tokenizer

    reference = NOTEBOOK.parents[5] / "data/implicit-reasoning-paper-reference-v1/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(reference, local_files_only=True)
    vocabulary = (
        NOTEBOOK.parents[5] / "data/implicit-reasoning-paper-reproduction-v1/only_ii/vocab.json"
    )
    tokenizer.add_tokens(read_json(vocabulary))
    text = "normal text"
    ids = tokenizer.encode(text)
    unmapped = len(tokenizer)
    assert unmapped == 52463
    assert tokenizer._convert_id_to_token(unmapped) is None
    with decode_unmapped_tokens(tokenizer):
        assert tokenizer.decode(ids) == text
        assert tokenizer.decode([unmapped]) == f"<unmapped_{unmapped}>"
    assert tokenizer._convert_id_to_token(unmapped) is None
