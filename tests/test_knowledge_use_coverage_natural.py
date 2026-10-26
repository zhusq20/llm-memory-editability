import hashlib
import json
from collections import Counter

import pytest

from llm_memory_editability import knowledge_use_coverage_natural as natural


@pytest.fixture
def sources(tmp_path):
    source = tmp_path / "author.csv"
    training = tmp_path / "train.jsonl"
    csv_rows, training_rows = [], []
    for index in range(96):
        paragraphs = [
            f'Entity {index} belongs to group L; its "label" is Amber.',
            f"Group L of case {index} has score 12.",
        ]
        csv_rows.append(f"SECRET QUESTION {index};SECRET GOLD {index};" + "---".join(paragraphs))
        for paragraph_index, paragraph in enumerate(paragraphs):
            pid = f"author-csv-row-{index:04d}:p{paragraph_index:02d}"
            for question_index in range(5):
                training_rows.append(
                    {
                        "id": f"{pid}:q{question_index:03d}",
                        "case_id": f"author-csv-row-{index:04d}",
                        "paragraph_id": pid,
                        "context": paragraph,
                        "question": f"Question {index}/{paragraph_index}/{question_index}?",
                        "sft_answer": f"  Exact SFT {question_index}\n",
                        "pd_answer": f"  Exact PD {question_index}\n",
                        "sft_truncated": False,
                        "pd_truncated": True,
                        "sft_generated_tokens": 3,
                        "pd_generated_tokens": 8,
                        "sft_finish_reason": "eos",
                        "pd_finish_reason": "length",
                    }
                )
    source.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    # Original completion order is not a selection criterion.
    training.write_text(
        "\n".join(json.dumps(row) for row in reversed(training_rows)) + "\n", encoding="utf-8"
    )
    return source, training


def valid_question(request):
    return json.dumps(
        {
            "question": f"What follows from the statements in request {request['id']}?",
            "evidence": [
                {"paragraph_id": pid, "quote": paragraph}
                for pid, paragraph in request["paragraphs"].items()
            ],
        }
    )


def generation(text, *, truncated=False):
    return {
        "text": text,
        "truncated": truncated,
        "generated_tokens": 8,
        "finish_reason": "length" if truncated else "eos",
    }


def test_fixed_development_and_application_ranges_are_not_new_confirmation(sources):
    source, _ = sources
    development = natural.load_training_cases(source, split="development")
    application = natural.load_training_cases(source, split="application")
    assert [case["source_row_index"] for case in development] == list(range(32))
    assert [case["source_row_index"] for case in application] == list(range(32, 96))
    evaluation = natural.load_evaluation_cases(source, split="application")
    assert evaluation[0]["question"] == "SECRET QUESTION 32"
    assert evaluation[-1]["answer"] == "SECRET GOLD 95"
    assert all(row["previously_evaluated_in_factorial"] for row in evaluation)
    assert application[0]["context"].count("---") == 1
    assert '; its "label" is Amber.' in application[0]["paragraphs"][0]


def test_test_columns_cannot_change_training_selection_or_generation_requests(sources, monkeypatch):
    source, training = sources
    cases_before = natural.load_training_cases(source, split="application")
    rows_before = natural.select_existing_rows(cases_before, training)
    requests_before = natural.build_use_requests(cases_before)
    source.write_text(
        source.read_text().replace("SECRET QUESTION", "CHANGED TEST").replace("SECRET GOLD", "XYZ")
    )
    monkeypatch.setattr(
        natural, "read_source", lambda *args: pytest.fail("training called evaluation parser")
    )
    cases_after = natural.load_training_cases(source, split="application")
    assert cases_after == cases_before
    assert natural.select_existing_rows(cases_after, training) == rows_before
    assert natural.build_use_requests(cases_after) == requests_before
    serialized = json.dumps([cases_after, rows_before, requests_before])
    assert "SECRET QUESTION" not in serialized
    assert "SECRET GOLD" not in serialized
    assert "CHANGED TEST" not in serialized
    assert "XYZ" not in serialized
    for case in cases_after:
        assert set(case) == natural.TRAINING_CASE_FIELDS


def test_generation_rejects_test_bearing_case_objects(sources):
    source, training = sources
    evaluation = natural.load_evaluation_cases(source, split="application")
    with pytest.raises(ValueError, match="context-only schema"):
        natural.build_use_requests(evaluation)
    with pytest.raises(ValueError, match="context-only schema"):
        natural.select_existing_rows(evaluation, training)


def test_original_selection_uses_fixed_question_indices_and_preserves_answers(sources):
    source, training = sources
    cases = natural.load_training_cases(source, split="application")
    variants = natural.select_existing_rows(cases, training)
    assert {key: len(rows) for key, rows in variants.items()} == {"A": 128, "B": 512, "C": 256}
    contexts = {case["case_id"]: case["context"] for case in cases}
    raw_lines = training.read_bytes().splitlines(keepends=True)
    file_hash = hashlib.sha256(training.read_bytes()).hexdigest()
    for variant, rows in variants.items():
        expected = {"A": {0}, "B": {0, 1, 2, 3}, "C": {0, 1}}[variant]
        assert {row["provenance"]["source_question_index"] for row in rows} == expected
        for row in rows:
            assert row["context"] == contexts[row["case_id"]]
            assert row["category"] == natural.ORIGINAL_CATEGORY
            assert row["fact_id_kind"] == natural.FACT_ID_KIND
            assert row["atomicity_verified"] is False
            trace = row["provenance"]
            raw = raw_lines[trace["source_line_number"] - 1]
            assert trace["source_file_sha256"] == file_hash
            assert trace["source_line_sha256"] == hashlib.sha256(raw).hexdigest()
            original = json.loads(raw)
            for kind in ("pd", "sft"):
                for field in ("answer", "truncated", "generated_tokens", "finish_reason"):
                    assert row[f"{kind}_{field}"] == original[f"{kind}_{field}"]


def test_missing_selected_original_question_fails_instead_of_substituting(sources):
    source, training = sources
    lines = training.read_text().splitlines()
    training.write_text(
        "\n".join(
            line for line in lines if json.loads(line)["id"] != "author-csv-row-0032:p00:q002"
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Missing fixed original"):
        natural.select_existing_rows(
            natural.load_training_cases(source, split="application"), training
        )


def test_requests_are_bounded_and_balanced_with_full_background(sources):
    source, _ = sources
    cases = natural.load_training_cases(source, split="application")
    requests = natural.build_use_requests(cases)
    assert len(requests) == 256
    assert set(Counter(row["case_id"] for row in requests).values()) == {4}
    counts = Counter(row["category"] for row in requests)
    assert max(counts.values()) - min(counts.values()) <= 1
    by_case = {case["case_id"]: case for case in cases}
    for request in requests:
        assert request["context"] == by_case[request["case_id"]]["context"]
        assert all(paragraph in request["user"] for paragraph in request["paragraphs"].values())
        assert request["variant"] == "C"


def test_new_questions_require_real_quotes_but_do_not_claim_semantic_verification(sources):
    source, _ = sources
    request = natural.build_use_requests(natural.load_training_cases(source, split="application"))[
        0
    ]
    row = natural.parse_use_question(request, valid_question(request))
    assert row["semantic_grounding_status"] == "requires_review"
    assert row["fact_ids"] == list(request["paragraphs"])
    short_ids = json.loads(valid_question(request))
    for evidence in short_ids["evidence"]:
        evidence["paragraph_id"] = evidence["paragraph_id"].rsplit(":", 1)[-1]
    assert natural.parse_use_question(request, json.dumps(short_ids))["fact_ids"] == list(
        request["paragraphs"]
    )
    short_ids["evidence"][0]["paragraph_id"] = "unrelated-case:p00"
    with pytest.raises(ValueError, match="does not occur"):
        natural.parse_use_question(request, json.dumps(short_ids))
    ungrounded = json.loads(valid_question(request))
    ungrounded["evidence"][0]["quote"] = "This external fact is not in the document."
    with pytest.raises(ValueError, match="does not occur"):
        natural.parse_use_question(request, json.dumps(ungrounded))
    missing_relation = json.loads(valid_question(request))
    missing_relation["evidence"] = missing_relation["evidence"][:1]
    with pytest.raises(ValueError, match="more distinct"):
        natural.parse_use_question(request, json.dumps(missing_relation))
    with pytest.raises(ValueError, match="unavailable"):
        natural.parse_use_question(request, '{"unavailable": true}')


def test_two_facts_can_share_a_paragraph_but_duplicate_quotes_do_not_count(sources):
    source, _ = sources
    request = natural.build_use_requests(natural.load_training_cases(source, split="application"))[
        0
    ]
    pid = next(iter(request["paragraphs"]))
    value = {
        "question": "What label belongs to the group membership recorded for this entity?",
        "evidence": [
            {"paragraph_id": pid, "quote": "belongs to group L"},
            {"paragraph_id": pid, "quote": 'its "label" is Amber'},
        ],
    }
    row = natural.parse_use_question(request, json.dumps(value))
    assert row["fact_ids"] == [pid]
    assert row["distinct_evidence_quotes"] == 2
    assert row["semantic_grounding_status"] == "requires_review"
    value["evidence"][1] = dict(value["evidence"][0])
    with pytest.raises(ValueError, match="distinct source evidence quotes"):
        natural.parse_use_question(request, json.dumps(value))


def test_answer_requests_use_common_context_and_keep_off_policy_temperatures(sources):
    source, _ = sources
    request = natural.build_use_requests(natural.load_training_cases(source, split="application"))[
        0
    ]
    question = natural.parse_use_question(request, valid_question(request))
    answers = natural.answer_requests(question)
    assert [(item["kind"], item["temperature"]) for item in answers] == [("pd", 1.5), ("sft", 0.25)]
    assert answers[0]["user"] == answers[1]["user"]
    assert question["context"] in answers[0]["user"]
    row = natural.attach_answers(
        question,
        pd_generation=generation("<think>discard</think>Amber", truncated=True),
        sft_generation=generation("Amber"),
    )
    assert row["pd_answer"] == row["sft_answer"] == "Amber"
    assert row["pd_truncated"] is True
    assert row["sft_truncated"] is False
    assert row["pd_raw_answer_sha256"] != row["sft_raw_answer_sha256"]
    broken = generation("Amber")
    broken["finish_reason"] = "length"
    with pytest.raises(ValueError, match="Inconsistent"):
        natural.attach_answers(question, pd_generation=broken, sft_generation=generation("Amber"))


def test_coverage_and_post_generation_overlap_audit_do_not_filter_rows(sources):
    source, training = sources
    cases = natural.load_training_cases(source, split="application")
    variants = natural.select_existing_rows(cases, training)
    requests = natural.build_use_requests(cases)
    variants["C"].extend(
        natural.attach_answers(
            natural.parse_use_question(request, valid_question(request)),
            pd_generation=generation("Amber"),
            sft_generation=generation("Amber"),
        )
        for request in requests
    )
    stats = natural.coverage_statistics(variants)
    assert stats["new_confirmation_set"] is False
    assert stats["original_question_atomicity_verified"] is False
    assert stats["variants"]["C"]["rows"] == stats["variants"]["B"]["rows"] == 512
    assert stats["variants"]["C"]["semantic_review_pending"] == 256
    evaluation = natural.load_evaluation_cases(source, split="application")
    assert natural.audit_evaluation_overlap(variants, evaluation) == []
    variants["C"][-1]["question"] = "  secret    QUESTION  32 "
    overlaps = natural.audit_evaluation_overlap(variants, evaluation)
    assert len(overlaps) == 1
    assert overlaps[0]["evaluation_case_ids"] == ["author-csv-row-0032"]
    assert len(variants["C"]) == 512


def test_changed_context_is_rejected(sources):
    source, training = sources
    cases = natural.load_training_cases(source, split="application")
    cases[0]["context"] += " external fact"
    with pytest.raises(ValueError, match="hash changed"):
        natural.build_use_requests(cases)
    with pytest.raises(ValueError, match="hash changed"):
        natural.select_existing_rows(cases, training)


def test_short_source_never_silently_changes_fixed_application_size(tmp_path):
    source = tmp_path / "short.csv"
    source.write_text("Question;Answer;Context\n" * 95)
    with pytest.raises(ValueError, match="fixed application range"):
        natural.load_training_cases(source, split="application")
