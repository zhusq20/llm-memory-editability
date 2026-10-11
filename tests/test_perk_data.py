"""Contracts that affect coverage, supervision and split isolation in PERK."""

from __future__ import annotations

import copy

import pytest

from llm_memory_editability.perk_data import (
    audit_rows,
    canonical_hash,
    fticr_example,
    partition_training,
    qa_response,
    validate_row,
)


def row(index=0):
    fact = f"John Event {index}: John went to the kitchen."
    return {
        "context": [f"Background. {fact} More background."],
        "facts": [fact],
        "support": [fact],
        "question": "Where is John?",
        "answer": " kitchen",
        "source_id": f"train-{index}",
    }


def test_support_supervision_and_answer_whitespace_are_preserved():
    value = row()
    value["support"].append("John Event 1: John travelled to the hallway.")
    target = qa_response(value)
    assert target == (
        "<support>John Event 0: John went to the kitchen.\n"
        "John Event 1: John travelled to the hallway.</support>\n<output> kitchen</output>"
    )
    example = fticr_example(value)
    assert example["completion"] == target
    assert value["context"][0] in example["prompt"]
    assert "You have memorized 1 documents.\nWhere is John?\nAnswer:" in example["prompt"]
    assert "<support>" not in example["prompt"]


def test_validation_uses_only_training_and_keeps_duplicates_together():
    rows = [row(i) for i in range(20)]
    duplicate = copy.deepcopy(rows[0])
    duplicate["source_id"] = "another-index"
    rows.append(duplicate)
    result = partition_training(rows, validation_count=5, seed=42)
    assert result == partition_training(rows, validation_count=5, seed=42)
    assert len(result["train"]) + len(result["validation"]) == len(rows)
    assert {canonical_hash(x) for x in result["train"]}.isdisjoint(
        {canonical_hash(x) for x in result["validation"]}
    )
    assert all(x["source_id"].startswith(("train-", "another-")) for x in result["validation"])


def test_missing_support_is_retained_in_audit_coverage():
    value = row()
    value["support"].append("Missing fact.")
    report = audit_rows([value])
    assert report["rows"] == 1
    assert report["coverage"]["support_absent_from_context"] == 1
    assert report["coverage"]["support_absent_from_facts"] == 1


def test_invalid_data_rejected_and_too_small_split_rejected():
    value = row()
    value["context"] = []
    with pytest.raises(ValueError, match="context"):
        validate_row(value)
    with pytest.raises(ValueError, match="validation_count"):
        partition_training([row()], validation_count=1, seed=42)
