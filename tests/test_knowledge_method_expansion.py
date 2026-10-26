import json

import pytest

from llm_memory_editability import knowledge_method_data as D
from llm_memory_editability import knowledge_method_expansion as E


def fixture_parts(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("Held out zero?;A;Paragraph zero\nHeld out one?;B;Paragraph one\n")
    config = {
        "source_csv": str(source),
        "n_cases": 2,
        "reuse_cases": 1,
        "expected_training_rows": 2,
        "questions_per_paragraph": 1,
        "model": "test",
        "model_revision": "fixed",
        "seed": 1234,
        "question_temperature": 1.5,
        "pd_temperature": 1.5,
        "sft_temperature": 0.25,
        "max_new_tokens": 512,
        "prepare_shards": {"shard-0": {"start": 1, "stop": 2, "paragraphs": 1}},
    }
    batch = tmp_path / "expanded"
    old = tmp_path / "old-shared"
    cases = D.read_source(source, 2)
    for i, folder in enumerate([old, batch / "preparation/shard-0/shared"]):
        folder.mkdir(parents=True)
        case = cases[i]
        pid = case["case_id"] + ":p00"
        row = {
            "id": pid + ":q000",
            "paragraph_id": pid,
            "case_id": case["case_id"],
            "context": case["paragraphs"][0],
            "question": f"Training {i}?",
            "pd_answer": "Answer",
            "sft_answer": "Answer",
            "pd_truncated": False,
            "sft_truncated": False,
        }
        (folder / "train.jsonl").write_text(json.dumps(row) + "\n")
        D._atomic_json(folder / "eval.json", [case])
        D._atomic_json(
            folder / "provenance.json", {"config": config, "source_csv_sha256": D._sha(source)}
        )
        D._atomic_json(
            folder / "prepare-complete.json",
            {
                "status": "complete",
                "files_sha256": {
                    f: D._sha(folder / f) for f in ["train.jsonl", "eval.json", "provenance.json"]
                },
            },
        )
    return config, batch, old


def test_expansion_copies_frozen_rows_and_preserves_all_coverage(tmp_path):
    config, batch, old = fixture_parts(tmp_path)
    before = (old / "train.jsonl").read_bytes()
    result = E.merge(config, batch, old)
    assert result["questions"] == result["cases"] == 2
    assert (batch / "shared/train.jsonl").read_bytes().startswith(before)
    assert (old / "train.jsonl").read_bytes() == before
    assert E.merge(config, batch, old) == result


def test_changed_input_and_generation_contract_fail_before_training(tmp_path):
    config, batch, old = fixture_parts(tmp_path)
    with pytest.raises(ValueError, match="Generation setting"):
        E.merge({**config, "pd_temperature": 0.25}, batch, old)
    (old / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="Committed input changed"):
        E.merge(config, batch, old)


def test_final_answer_normalization_preserves_original_and_records_change(tmp_path):
    config, batch, old = fixture_parts(tmp_path)
    config["answer_format"] = "qwen_final_answer_only"
    row = json.loads((old / "train.jsonl").read_text())
    row["pd_answer"] = "Unwanted draft</think>\n\nFinal answer"
    (old / "train.jsonl").write_text(json.dumps(row) + "\n")
    marker = json.loads((old / "prepare-complete.json").read_text())
    marker["files_sha256"]["train.jsonl"] = D._sha(old / "train.jsonl")
    D._atomic_json(old / "prepare-complete.json", marker)
    original = (old / "train.jsonl").read_bytes()
    E.merge(config, batch, old)
    result = json.loads((batch / "shared/train.jsonl").read_text().splitlines()[0])
    audit = json.loads((batch / "shared/answer-format-audit.json").read_text())
    assert result["pd_answer"] == "Final answer"
    assert audit["normalized_counts"] == {"pd": 1, "sft": 0}
    assert audit["reused_answer_changes"][0]["before"] == row["pd_answer"]
    assert (old / "train.jsonl").read_bytes() == original


@pytest.mark.parametrize("answer", ["<think>unfinished", "draft</think>\n", ""])
def test_incomplete_thinking_is_not_accepted_as_a_training_answer(answer):
    with pytest.raises(ValueError):
        D.final_answer_text(answer)


def test_non_thinking_preparation_retains_failed_raw_and_trains_final_answer(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("Held out?;A;Paragraph\n")
    cfg = {
        "source_csv": str(source),
        "n_cases": 1,
        "questions_per_paragraph": 1,
        "generation_batch_size": 1,
        "model_path": str(tmp_path),
        "answer_format": "qwen_final_answer_only",
    }
    calls = 0

    def generate(users, temperature, seed):
        nonlocal calls
        if users[0].startswith("Here is"):
            text = "<question>Training?</question>"
        else:
            calls += 1
            text = "<think>unfinished" if calls == 1 else "draft</think>\n\nFinal answer"
        return [{"text": text, "truncated": False, "generated_tokens": 9, "finish_reason": "eos"}]

    monkeypatch.setattr(D, "_load_generator", lambda c: generate)
    D.prepare(cfg, tmp_path / "prepared")
    shared = tmp_path / "prepared/shared"
    raw = [json.loads(line) for line in (shared / "answers-raw.jsonl").read_text().splitlines()]
    row = json.loads((shared / "train.jsonl").read_text())
    assert calls == 3 and raw[0]["failed"] and raw[0]["format_error"]
    assert raw[0]["generation"]["text"] == "<think>unfinished"
    assert row["pd_answer"] == row["sft_answer"] == "Final answer"
    assert row["pd_format_normalized"] and row["sft_format_normalized"]


def test_case_range_excludes_test_questions_outside_the_shard(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    source.write_text("Outside test?;A;Outside paragraph\nInside test?;B;Inside paragraph\n")
    cfg = {
        "source_csv": str(source),
        "n_cases": 2,
        "case_start": 1,
        "case_stop": 2,
        "expected_paragraphs": 1,
        "expected_training_rows": 1,
        "questions_per_paragraph": 1,
        "model_path": str(tmp_path),
        "generation_batch_size": 1,
    }

    def generate(users, temperature, seed):
        return [
            {
                "text": "<question>Outside test?</question><question>Training?</question>"
                if u.startswith("Here is")
                else "Answer",
                "truncated": False,
                "generated_tokens": 5,
                "finish_reason": "eos",
            }
            for u in users
        ]

    monkeypatch.setattr(D, "_load_generator", lambda c: generate)
    result = D.prepare(cfg, tmp_path / "shard")
    assert result["excluded_eval_questions"] == 1
    rows = [json.loads(x) for x in (tmp_path / "shard/shared/train.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["case_id"] == "author-csv-row-0001"
    assert rows[0]["question"] == "Training?"
