import pytest

from llm_memory_editability.knowledge_use_coverage_budget import (
    TokenStream,
    microbatch_weight,
    pressure_example,
    select_development_node,
)


def samples():
    return [
        {"id": "a", "student": [1], "teacher": [2, 3], "answer": [4, 5, 6, 9]},
        {"id": "b", "student": [7], "teacher": [8], "answer": [10, 11, 9]},
    ]


def test_budget_carries_prefix_without_fake_eos():
    stream = TokenStream(samples()[:1], 1)
    first = stream.take(2)[0]
    second = stream.take(2)[0]
    assert first["answer"] == [4, 5]
    assert second["answer"] == [6, 9]
    assert second["student"] == [1, 4, 5]
    assert second["teacher"] == [2, 3, 4, 5]
    assert stream.examples_started == stream.examples_completed == 1


def test_exact_restore_across_shuffles_and_split_answers():
    stream = TokenStream(samples(), 23)
    stream.take(9)
    state = stream.state_dict()
    expected = stream.take(31)
    restored = TokenStream(samples(), 999)
    restored.load_state_dict(state)
    assert restored.take(31) == expected
    assert restored.tokens == 40


def test_normalization_preserves_distinct_objective_contracts():
    rows = samples()
    assert microbatch_weight("pd", rows[:1], rows) == 0.5
    assert microbatch_weight("sft", rows[:1], rows) == 4 / 7


def test_development_selection_requires_all_arms_and_keeps_failures():
    assert select_development_node(
        [100, 200],
        {
            "100": {"A": 0.9, "B": 0.7, "C": 0.9},
            "200": {"A": 0.9, "B": 0.79, "C": 0.9},
        },
    ) == {"tokens": 200, "target_met": False}
    with pytest.raises(ValueError, match="Incomplete"):
        select_development_node([100], {"100": {"A": 1.0}})


def test_pressure_combines_three_distinct_real_maxima():
    rows = [
        {"id": "student", "student": [1] * 20, "teacher": [2], "answer": [3]},
        {"id": "teacher", "student": [1], "teacher": [2] * 30, "answer": [3, 4]},
        {"id": "answer", "student": [1], "teacher": [2], "answer": [3] * 10},
    ]
    row, provenance = pressure_example(rows)
    assert len(row["answer"]) == 10
    assert len(row["student"]) + len(row["answer"]) == 21
    assert len(row["teacher"]) + len(row["answer"]) == 32
    assert provenance["student_source"] == "student"
    assert provenance["teacher_source"] == "teacher"
    assert provenance["answer_source"] == "answer"
