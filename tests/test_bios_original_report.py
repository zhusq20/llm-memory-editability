"""Checks for score preservation, denominators and repeated-measurement pairing."""

import pytest

from llm_memory_editability.bios_original_report import (
    COUNTS,
    check_identity,
    extract,
    finalized,
    paired_differences,
    review_flags,
    rollup,
)


def test_explanation_is_not_implicitly_stripped_from_direct_score():
    row = dict(task="year", cot=False, generated="born in 2000; Answer: 2000", answer="2000")
    flags = review_flags(row, {}, {})
    assert not flags["correct"]
    assert flags["terminal_answer_match"]
    assert flags["target_text_present"]
    row["cot"] = True
    assert extract(row) == "2000"
    assert review_flags(row, {}, {})["correct"]


def test_literal_presence_and_city_confusion_are_not_semantic_rescoring():
    row = dict(task="birthcity", cot=False, answer="Boston, MA", generated="Chicago, IL")
    p = dict(birthcity="Boston, MA", workcity="Chicago, IL")
    flags = review_flags(row, p, {"birthcity": {"boston, ma", "chicago, il"}})
    assert flags["city_swap"] and flags["wrong_recognized"] and not flags["correct"]
    row.update(task="year", answer="2000", generated="20001")
    assert not review_flags(row, {}, {})["target_text_present"]


def test_comparison_wrong_person_and_outside_pair_are_distinct():
    row = dict(task="comparison", cot=False, answer="A Person", generated="B Person")
    vocab = {"name": {"a person", "b person", "c person"}}
    p, other = dict(name="A Person"), dict(name="B Person")
    flags = review_flags(row, p, vocab, other)
    assert not flags["correct"] and not flags["comparison_outside_pair"]
    row["generated"] = "C Person"
    assert review_flags(row, p, vocab, other)["comparison_outside_pair"]


def test_duplicate_identity_is_rejected_and_cot_is_distinct():
    seen = bytearray([255]) * 132
    index = check_identity(seen, 0, "date", 0, False)
    seen[index] = 0
    with pytest.raises(ValueError, match="Duplicate"):
        check_identity(seen, 0, "date", 0, False)
    assert check_identity(seen, 0, "date", 0, True) != index


def test_heldout_denominator_is_all_five_views_and_pair_is_within_world():
    rows = []
    for world in (1, 2):
        for condition in ("S", "M"):
            for view in range(6):
                row = dict(
                    world=world,
                    initialization=1,
                    condition=condition,
                    stage="adapt",
                    split="test",
                    task="date",
                    view=view,
                    cot=False,
                )
                row.update(dict.fromkeys(COUNTS, 0))
                row.update(n=10, correct=5 if condition == "M" else 1)
                rows.append(finalized(row))
    groups = rollup(rows, views=True)
    assert all(r["n"] == (10 if r["view_group"] == "canonical" else 50) for r in groups)
    pairs = paired_differences(groups)
    assert len(pairs) == 4
    assert all(r["difference_pp"] == 40 for r in pairs)
    groups[1]["n"] = 1
    with pytest.raises(ValueError, match="Unequal paired"):
        paired_differences(groups)
