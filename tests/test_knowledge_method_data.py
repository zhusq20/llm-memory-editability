import json
from collections import Counter

import pytest

from llm_memory_editability import knowledge_method_data as data


def config_for(tmp_path, *, paragraphs=2, questions=5, batch_size=2):
    source = tmp_path / "author.csv"
    source.write_text(
        "A held-out multi-hop question?;A hidden target;"
        + "---".join(f'Paragraph {i} has "quotes"; and facts.' for i in range(paragraphs))
        + "\n",
        encoding="utf-8",
    )
    return {
        "source_csv": str(source),
        "model_path": str(tmp_path / "unused-model"),
        "n_cases": 1,
        "questions_per_paragraph": questions,
        "generation_batch_size": batch_size,
        "expected_paragraphs": paragraphs,
        "expected_training_rows": paragraphs * questions,
    }


def output(text, *, truncated=False):
    return {
        "text": text,
        "raw_text": text,
        "token_ids": [1] * (512 if truncated else 3),
        "generated_tokens": 512 if truncated else 3,
        "truncated": truncated,
        "finish_reason": "length" if truncated else "eos",
    }


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.stop_after = None
        self.truncated = False

    def __call__(self, users, temperature, seed):
        if self.stop_after is not None and len(self.calls) >= self.stop_after:
            raise KeyboardInterrupt("simulated process interruption")
        self.calls.append((list(users), temperature, seed))
        rows = []
        for user in users:
            assert "A hidden target" not in user
            if user.startswith("Here is a paragraph"):
                assert "A held-out multi-hop question?" not in user
                paragraph = user.splitlines()[1]
                rows.append(
                    output(
                        "\n".join(
                            f"<question>Fact {i} about {paragraph}?</question>" for i in range(5)
                        )
                    )
                )
            else:
                rows.append(output(f"Teacher answer at {temperature}", truncated=self.truncated))
        return rows


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_source_preserves_all_paragraphs_literal_quotes_and_semicolons(tmp_path):
    cfg = config_for(tmp_path)
    parsed = data.read_source(data.Path(cfg["source_csv"]), 1)
    assert len(parsed) == 1
    assert len(parsed[0]["paragraphs"]) == 2
    assert '"quotes"; and facts.' in parsed[0]["paragraphs"][0]
    assert parsed[0]["source_row_index"] == 0
    assert parsed[0]["case_id"] == "author-csv-row-0000"
    assert parsed[0]["context"].count("---") == 1


def test_complete_balanced_dataset_and_completed_resume_never_loads_model(tmp_path, monkeypatch):
    cfg = config_for(tmp_path)
    fake = FakeGenerator()
    monkeypatch.setattr(data, "_load_generator", lambda _: fake)
    batch = tmp_path / "batch"
    result = data.prepare(cfg, batch)
    shared = batch / "shared"
    train = rows(shared / "train.jsonl")
    assert result["status"] == "complete"
    assert len(train) == len({row["id"] for row in train}) == 10
    assert set(Counter(row["paragraph_id"] for row in train).values()) == {5}
    assert all(row["pd_answer"] != row["sft_answer"] for row in train)
    assert all(not row["pd_truncated"] for row in train)
    assert len(rows(shared / "answers-raw.jsonl")) == 20
    assert max(len(users) for users, _, _ in fake.calls) == 2
    provenance = json.loads((shared / "provenance.json").read_text())
    assert provenance["paragraph_count"] == 2
    assert provenance["source_id_kind"] == "zero_based_author_csv_row_not_hf_id"
    assert "FP32" in provenance["precision"]
    monkeypatch.setattr(data, "_load_generator", lambda _: pytest.fail("completed reload"))
    assert data.prepare(cfg, batch) == result


def test_resume_reuses_each_completed_answer_and_questions(tmp_path, monkeypatch):
    cfg = config_for(tmp_path)
    first = FakeGenerator()
    first.stop_after = 2  # Questions, then the first two PD answers are durable.
    monkeypatch.setattr(data, "_load_generator", lambda _: first)
    batch = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        data.prepare(cfg, batch)
    assert not (batch / "shared/prepare-complete.json").exists()
    old_answers = rows(batch / "shared/answers-raw.jsonl")
    assert len(old_answers) == 2
    second = FakeGenerator()
    monkeypatch.setattr(data, "_load_generator", lambda _: second)
    data.prepare(cfg, batch)
    assert all(
        not user.startswith("Here is a paragraph") for us, _, _ in second.calls for user in us
    )
    old_users = set(first.calls[1][0])
    assert all(not old_users.intersection(us) for us, temp, _ in second.calls if temp == 1.5)
    assert rows(batch / "shared/answers-raw.jsonl")[:2] == old_answers


def test_reconstruct_samples_after_raw_record_was_committed(tmp_path, monkeypatch):
    cfg = config_for(tmp_path, paragraphs=1)
    fake = FakeGenerator()
    original_append = data._append

    def interrupted_append(path, row):
        if path.name == "questions.jsonl":
            raise KeyboardInterrupt("after raw, before derived row")
        original_append(path, row)

    monkeypatch.setattr(data, "_load_generator", lambda _: fake)
    monkeypatch.setattr(data, "_append", interrupted_append)
    batch = tmp_path / "batch"
    with pytest.raises(KeyboardInterrupt):
        data.prepare(cfg, batch)
    assert len(rows(batch / "shared/questions-raw.jsonl")) == 1
    monkeypatch.setattr(data, "_append", original_append)
    second = FakeGenerator()
    monkeypatch.setattr(data, "_load_generator", lambda _: second)
    data.prepare(cfg, batch)
    assert all(not u.startswith("Here is a paragraph") for us, _, _ in second.calls for u in us)
    assert len(rows(batch / "shared/train.jsonl")) == 5


def test_exact_eval_question_is_excluded_and_short_generation_is_topped_up(tmp_path, monkeypatch):
    cfg = config_for(tmp_path, paragraphs=1)
    normal = FakeGenerator()
    question_calls = 0

    def generate(users, temperature, seed):
        nonlocal question_calls
        if users[0].startswith("Here is a paragraph"):
            question_calls += 1
            if question_calls == 1:
                return [
                    output(
                        "<question>A held-out multi-hop question?</question>"
                        + "".join(f"<question>Novel question {i}?</question>" for i in range(4))
                    )
                ]
        return normal(users, temperature, seed)

    monkeypatch.setattr(data, "_load_generator", lambda _: generate)
    batch = tmp_path / "batch"
    result = data.prepare(cfg, batch)
    assert result["excluded_eval_questions"] == 1
    assert result["questions"] == 5
    assert question_calls == 2
    assert sum(result["failures_by_paragraph"].values()) == 1
    assert all(
        row["question"] != "A held-out multi-hop question?"
        for row in rows(batch / "shared/train.jsonl")
    )


def test_parse_failure_budget_is_durable_and_never_creates_complete(tmp_path, monkeypatch):
    cfg = {**config_for(tmp_path, paragraphs=1), "max_failures_per_paragraph": 2}
    calls = []

    def bad(users, *_):
        calls.append(users)
        return [output("No parseable questions") for _ in users]

    monkeypatch.setattr(data, "_load_generator", lambda _: bad)
    batch = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="Failure limit"):
        data.prepare(cfg, batch)
    assert len(calls) == 2
    assert not (batch / "shared/prepare-complete.json").exists()
    monkeypatch.setattr(data, "_load_generator", lambda _: pytest.fail("exhausted budget reloaded"))
    with pytest.raises(RuntimeError, match="Failure limit"):
        data.prepare(cfg, batch)
    status = json.loads((batch / "shared/prepare-status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["questions"] == 0


def test_nonempty_length_limited_answers_are_kept_with_explicit_labels(tmp_path, monkeypatch):
    cfg = config_for(tmp_path, paragraphs=1)
    fake = FakeGenerator()
    fake.truncated = True
    monkeypatch.setattr(data, "_load_generator", lambda _: fake)
    batch = tmp_path / "batch"
    result = data.prepare(cfg, batch)
    assert result["truncated_answers"] == {"pd": 5, "sft": 5}
    train = rows(batch / "shared/train.jsonl")
    assert all(row["pd_truncated"] and row["sft_truncated"] for row in train)
    assert all(row["sft_generated_tokens"] == 512 for row in train)


def test_empty_answers_obey_per_paragraph_budget_even_in_a_large_batch(tmp_path, monkeypatch):
    cfg = {**config_for(tmp_path, paragraphs=1, batch_size=16), "max_failures_per_paragraph": 2}
    normal = FakeGenerator()

    def empty(users, temperature, seed):
        if users[0].startswith("Here is a paragraph"):
            return normal(users, temperature, seed)
        return [output("") for _ in users]

    monkeypatch.setattr(data, "_load_generator", lambda _: empty)
    batch = tmp_path / "batch"
    with pytest.raises(RuntimeError, match="Failure limit"):
        data.prepare(cfg, batch)
    assert len(rows(batch / "shared/answers-raw.jsonl")) == 2
    assert not (batch / "shared/prepare-complete.json").exists()


def test_torn_final_record_is_preserved_and_committed_corruption_is_rejected(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_bytes(b'{"id": 1}\n{"partial":')
    assert data._read_jsonl(path) == [{"id": 1}]
    assert path.read_bytes() == b'{"id": 1}\n'
    assert rows(tmp_path / "recovery.jsonl")[0]["torn_bytes_b64"]
    path.write_bytes(b'{"partial":\n{"id": 1}\n')
    with pytest.raises(ValueError, match="Corrupt committed"):
        data._read_jsonl(path)


def test_source_or_completed_dataset_change_is_rejected(tmp_path, monkeypatch):
    cfg = config_for(tmp_path, paragraphs=1)
    monkeypatch.setattr(data, "_load_generator", lambda _: FakeGenerator())
    batch = tmp_path / "batch"
    data.prepare(cfg, batch)
    with (batch / "shared/train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="artifact changed"):
        data.prepare(cfg, batch)
    source = data.Path(cfg["source_csv"])
    source.write_text(source.read_text().replace("and facts", "changed facts"))
    with pytest.raises(ValueError, match="Frozen input changed"):
        data.prepare(cfg, batch)


def test_wrong_paragraph_contract_fails_before_loading_model(tmp_path, monkeypatch):
    cfg = {**config_for(tmp_path), "expected_paragraphs": 10}
    monkeypatch.setattr(data, "_load_generator", lambda _: pytest.fail("should not load"))
    with pytest.raises(ValueError, match="paragraph count"):
        data.prepare(cfg, tmp_path / "batch")


def test_shared_chat_helpers_disable_thinking_and_preserve_author_separator():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            assert messages[0] == {"role": "system", "content": data.SYSTEM}
            return messages[1]["content"]

    assert data.render_chat(Tokenizer(), data.chat_user(" Question? ")) == "Question?"
    assert data.context_user("Fact text.", "Question?") == "Fact text.\n\n---\n\nQuestion?"
