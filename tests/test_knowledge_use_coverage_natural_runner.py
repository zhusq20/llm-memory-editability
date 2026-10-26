import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_memory_editability import knowledge_use_coverage_natural as N
from llm_memory_editability.knowledge_method_data import read_source

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_knowledge_use_coverage_natural.py"
SPEC = importlib.util.spec_from_file_location("natural_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[-1]["content"]

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


@pytest.fixture
def setup(tmp_path):
    batch, factorial = tmp_path / "new", tmp_path / "factorial"
    source = factorial / "upstream/datasets/hotpotqa.csv"
    source.parent.mkdir(parents=True)
    lines, training = [], []
    for index in range(1000):
        paragraphs = [f"Background {index} says Amber.", f"Another statement {index} says Blue."]
        lines.append(f"SECRET QUESTION {index};SECRET ANSWER {index};" + "---".join(paragraphs))
        if index >= 96:
            continue
        cid = f"author-csv-row-{index:04d}"
        for p, paragraph in enumerate(paragraphs):
            for q in range(4):
                training.append(
                    {
                        "id": f"{cid}:p{p:02d}:q{q:03d}",
                        "case_id": cid,
                        "paragraph_id": f"{cid}:p{p:02d}",
                        "context": paragraph,
                        "question": f"Original question {index}/{p}/{q}?",
                        "pd_answer": f" Original PD {q}\n",
                        "sft_answer": f" Original SFT {q}\n",
                        "pd_truncated": True,
                        "sft_truncated": False,
                        "pd_generated_tokens": 5,
                        "sft_generated_tokens": 4,
                    }
                )
    source.write_text("\n".join(lines) + "\n")
    runner.write_rows(factorial / "shared/train.jsonl", training)
    runner.write_json(factorial / "shared/eval.json", read_source(source, 1000))
    runner.write_json(
        factorial / "shared/prepare-complete.json",
        {
            "files_sha256": {"train.jsonl": runner.digest(factorial / "shared/train.jsonl")},
        },
    )
    replay = [
        {
            "id": "replay/0",
            "question": "Add 5 and 3",
            "context": "",
            "pd_answer": "8",
            "sft_answer": "8",
            "pd_truncated": False,
            "sft_truncated": False,
        }
    ]
    runner.write_rows(batch / "data/formal/prepared-replay.jsonl", replay)
    runner.write_json(
        batch / "data/formal/prepare-complete.json",
        {
            "sha256": {
                "prepared-replay.jsonl": runner.digest(batch / "data/formal/prepared-replay.jsonl")
            },
        },
    )
    runner.write_json(
        factorial / "upstream/retention-base.json",
        {
            "natural": [
                {
                    "qid": f"old-{i}",
                    "prompt": f"Old question {i}?",
                    "gold": "Poland",
                    "type": "country",
                }
                for i in range(317)
            ],
        },
    )
    config = runner.natural_config(
        {
            "seed": 1234,
            "model_revision": "fixed-base",
            "natural_token_nodes": [512],
            "scope": "down",
        },
        factorial,
    )
    return batch, factorial, source, config


class FakeRuntime:
    def __init__(self, source, *, invalid_question=False, recover_on_attempt_two=False):
        cases = N.load_training_cases(source, split="application") + N.load_training_cases(
            source, split="development"
        )
        self.requests = {r["id"]: r for r in N.build_use_requests(cases)}
        self.calls = []
        self.invalid_question = invalid_question
        self.recover_on_attempt_two = recover_on_attempt_two
        self.question_attempts = Counter()
        self.K = SimpleNamespace(
            seed_all=lambda _: None, load_model=lambda _: (object(), FakeTokenizer())
        )

    def generate(self, model, tok, inputs, config, context, temperature):
        self.calls.append((inputs, context, temperature))
        assert config["max_sequence_tokens"] == 8192
        assert config["generation_max_new_tokens"] == 512
        assert "SECRET QUESTION" not in json.dumps(inputs)
        assert "SECRET ANSWER" not in json.dumps(inputs)
        assert all("gold" not in row and "answer" not in row for row in inputs)
        result = []
        for row in inputs:
            if not context:
                self.question_attempts[row["id"]] += 1
                request = self.requests[row["id"]]
                text = json.dumps(
                    {
                        "question": f"New use {row['id']}?",
                        "evidence": [
                            {"paragraph_id": pid, "quote": paragraph}
                            for pid, paragraph in request["paragraphs"].items()
                        ],
                    }
                )
                if self.invalid_question and (
                    not self.recover_on_attempt_two or self.question_attempts[row["id"]] == 1
                ):
                    text = '{"unavailable": true}'
            else:
                text = f"Sampled answer at {temperature}"
            result.append(
                {
                    "id": row["id"],
                    "raw": text,
                    "answer": text,
                    "token_ids": [1, 2, 3],
                    "truncated": False,
                    "final_answer_valid": True,
                }
            )
        return result

    def encoded_rows(self, tok, records, objective, config):
        return [
            {"answer": tok.encode(r[f"{objective}_answer"]), "teacher": tok.encode(r["context"])}
            for r in records
        ]


def approve(batch, factorial, source, config, attempts=1):
    runner.calibrate(config, batch, factorial, runtime=FakeRuntime(source))
    out = batch / "data/natural-development"
    runner.write_json(
        out / "calibration-approved.json",
        {
            "status": "approved",
            "semantic_review_completed": True,
            "question_attempts": attempts,
            "generation_protocol_sha256": runner.generation_protocol(config)[1],
            "quality_summary_sha256": runner.digest(out / "quality-summary.json"),
        },
    )


def test_prepare_never_opens_evaluation_and_reuses_original_answers(setup, monkeypatch):
    batch, factorial, source, config = setup
    approve(batch, factorial, source, config)
    runtime = FakeRuntime(source)
    original_read = runner.read

    def protected_read(path):
        if Path(path).name == "eval.json":
            pytest.fail("Preparation opened evaluation JSON")
        return original_read(path)

    monkeypatch.setattr(runner, "read", protected_read)
    runner.prepare(config, batch, factorial, runtime=runtime)
    out = batch / "data/natural"
    complete = original_read(out / "prepare-complete.json")
    assert complete["rows"] == {"A": 128, "B": 512, "C": 512}
    assert original_read(out / "manifest.json")["evaluation_files_read_for_generation"] is False
    originals = {r["id"]: r for r in runner.rows(factorial / "shared/train.jsonl")}
    for arm in "ABC":
        for row in runner.rows(out / f"prepared-{arm}.jsonl"):
            if row["category"] == N.ORIGINAL_CATEGORY:
                original = originals[row["provenance"]["source_id"]]
                for field in ("pd_answer", "sft_answer", "pd_truncated", "sft_truncated"):
                    assert row[field] == original[field]
            assert "---" in row["context"]
    assert (out / "prepared-replay.jsonl").read_bytes() == (
        batch / "data/formal/prepared-replay.jsonl"
    ).read_bytes()
    assert set(temperature for _, _, temperature in runtime.calls) == {0.7, 1.5, 0.25}
    assert not (out / "prepared-rewrite.jsonl").exists()
    with pytest.raises(FileExistsError):
        runner.prepare(config, batch, factorial, runtime=runtime)


def test_question_failure_is_durable_and_never_retried(setup):
    batch, factorial, source, config = setup
    approve(batch, factorial, source, config, attempts=3)
    runtime = FakeRuntime(source, invalid_question=True)
    with pytest.raises(ValueError, match="unresolved requests"):
        runner.prepare(config, batch, factorial, runtime=runtime)
    out = batch / "data/natural"
    assert len(runtime.calls) == 192
    assert len(runner.rows(out / "raw-questions.jsonl")) == 768
    assert set(runtime.question_attempts.values()) == {3}
    assert runner.read(out / "failure.json")["automatic_retry"] is False
    assert not (out / "prepare-complete.json").exists()
    with pytest.raises(FileExistsError):
        runner.prepare(config, batch, factorial, runtime=runtime)
    assert len(runtime.calls) == 192


def test_calibration_preserves_first_pass_and_bounded_corrections_without_approval(
    setup, monkeypatch
):
    batch, factorial, source, config = setup
    runtime = FakeRuntime(source, invalid_question=True, recover_on_attempt_two=True)
    old_read = runner.read

    def guarded(path):
        assert Path(path).name != "eval.json"
        return old_read(path)

    monkeypatch.setattr(runner, "read", guarded)
    runner.calibrate(config, batch, factorial, runtime=runtime)
    out = batch / "data/natural-development"
    quality = old_read(out / "quality-summary.json")
    assert quality["request_count"] == 96
    assert quality["first_attempt"] == {"unavailable": 96}
    assert quality["final_attempt"] == {"valid": 96}
    assert quality["actual_generation_requests"] == 192
    assert quality["status"] == "awaiting_manual_semantic_review"
    assert not (out / "calibration-approved.json").exists()
    assert set(runtime.question_attempts.values()) == {2}
    raw = runner.rows(out / "raw-questions.jsonl")
    assert len(raw) == 192
    assert {r["category"] for r in raw} == set(N.USE_CATEGORIES)
    assert {r["case_id"] for r in raw} == {f"author-csv-row-{i:04d}" for i in range(32)}
    assert set(Counter(r["case_id"] for r in raw).values()) == {6}
    first = {r["request_id"]: r for r in raw if r["attempt"] == 1}
    for repaired in (r for r in raw if r["attempt"] == 2):
        original = first[repaired["request_id"]]
        assert (repaired["case_id"], repaired["category"]) == (
            original["case_id"],
            original["category"],
        )
        assert original["generation"]["raw"] in repaired["user"]


def test_formal_preparation_requires_review_bound_to_current_protocol(setup):
    batch, factorial, source, config = setup
    runtime = FakeRuntime(source)
    with pytest.raises(FileNotFoundError):
        runner.prepare(config, batch, factorial, runtime=runtime)
    assert runtime.calls == []
    assert not (batch / "data/natural/started.json").exists()
    approve(batch, factorial, source, config)
    assert runner.approved_calibration(config, batch)["question_attempts"] == 1
    changed = {**config, "generation_top_k": 0}
    with pytest.raises(ValueError, match="matching reviewed calibration"):
        runner.approved_calibration(changed, batch)
    path = batch / "data/natural-development/raw-questions.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after review"):
        runner.approved_calibration(config, batch)


def test_fenced_unavailable_is_retained_as_unavailable(setup):
    _, _, source, _ = setup
    request = runner.development_requests(source)[0]
    result = {
        "raw": '```json\n{"unavailable": true}\n```',
        "answer": '```json\n{"unavailable": true}\n```',
        "truncated": False,
        "final_answer_valid": True,
    }
    status, parsed, error = runner._question_status(request, result)
    assert status == "unavailable"
    assert parsed is None
    assert error


def test_prepared_hashes_and_post_generation_overlap_are_checked(setup):
    batch, factorial, source, config = setup
    approve(batch, factorial, source, config)
    runner.prepare(config, batch, factorial, runtime=FakeRuntime(source))
    runner.verify_preparation(batch, factorial)
    audit = runner.read(batch / "data/natural/evaluation-overlap-audit.json")
    assert audit == {"passed": True, "overlaps": [], "filtering_performed": False}
    path = batch / "data/natural/prepared-A.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify_preparation(batch, factorial)


def test_evaluation_copy_mismatch_is_not_silently_accepted(setup):
    _, factorial, source, _ = setup
    source.write_text(source.read_text().replace("SECRET QUESTION 32;", "CHANGED QUESTION;"))
    with pytest.raises(ValueError, match="evaluation columns differ"):
        runner.application_cases(factorial)


def test_paper_prompts_are_question_only_or_original_context(setup):
    _, factorial, _, _ = setup
    cases = runner.application_cases(factorial)
    records = runner.paper_rows(cases, FakeTokenizer())
    assert Counter(r["group"] for r in records) == {"closed": 64, "oracle_context": 64}
    assert records[0]["prompt"] == cases[0]["question"]
    assert records[1]["prompt"].endswith("Question: " + cases[0]["question"])
    assert "SECRET ANSWER" not in records[1]["prompt"]


def test_wrong_checkpoint_identity_is_rejected():
    config = {"scope": "down"}
    saved = {
        "config": config,
        "tokens": 512,
        "identity": {"cohort": "natural", "objective": "pd", "arm": "A", "stage": "addition"},
    }
    runner.validate_checkpoint(saved, config, "pd", "A", 512)
    for objective, arm, node in (("sft", "A", 512), ("pd", "B", 512), ("pd", "A", 1024)):
        with pytest.raises(ValueError, match="checkpoint identity"):
            runner.validate_checkpoint(saved, config, objective, arm, node)


def test_retention_tracks_old_correct_separately_from_net_accuracy():
    before = [{"id": str(i), "strict": i % 2 == 0} for i in range(317)]
    after = [dict(row) for row in before]
    after[0]["strict"], after[1]["strict"] = False, True
    summary = runner.retention_summary(before, after)
    assert summary["before_correct_n"] == summary["after_correct_n"] == 159
    assert summary["initially_correct_n"] == 159
    assert summary["retained_n"] == 158
    with pytest.raises(ValueError, match="coverage differ"):
        runner.retention_summary(before, after[:-1])


def test_baseline_calls_paper_judge_without_documents_and_preserves_alias_scores(
    setup, monkeypatch
):
    batch, factorial, source, config = setup
    approve(batch, factorial, source, config)
    runtime = FakeRuntime(source)
    runner.prepare(config, batch, factorial, runtime=runtime)
    runtime.K.predict = lambda model, tok, records, cfg: [
        {**record, "strict": i % 2 == 0, "answer": "Poland"} for i, record in enumerate(records)
    ]
    runtime.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

    def predict(model, tok, records, cfg, output, origin):
        assert cfg["paper_eval_batch_size"] == 4
        return [{**record, "answer": "Candidate"} for record in records]

    def grade(records, cfg, output):
        assert cfg["judge_model"] == "Qwen/Qwen3-4B"
        assert cfg["judge_enable_thinking"] is False
        assert all("prompt" not in record and "context" not in record for record in records)
        assert len(records) == 128
        return {
            "summary": {
                "groups": {
                    f"base/before/{group}": {
                        "n": 64,
                        "correct_n": 32,
                        "valid_n": 63,
                        "invalid_n": 1,
                        "coverage": 63 / 64,
                        "paper_accuracy_valid_denominator": 32 / 63,
                    }
                    for group in ("closed", "oracle_context")
                }
            }
        }

    monkeypatch.setattr(runner.P, "predict_records", predict)
    monkeypatch.setattr(runner.P, "grade_records", grade)
    runner.evaluate(config, batch, factorial, runtime=runtime)
    summary = runner.read(batch / "baseline/natural/summary.json")
    assert summary["new_confirmation_set"] is False
    assert summary["groups"]["closed"]["accuracy"] == 32 / 63
    assert summary["retention"]["retained_n"] == 159
    assert len(runner.rows(batch / "baseline/natural/retention.jsonl")) == 317
    synced = []
    runtime.training_name = (
        lambda cohort, objective, arm, stage: f"{cohort}-{objective}-{arm}-{stage}"
    )
    runtime.sync_node_tracking = lambda *args: synced.append(args)
    node_dir = batch / "runs/natural-pd-A-addition/node-512"
    node_dir.mkdir(parents=True)
    (node_dir / "resume.pt").write_bytes(b"CPU test checkpoint stub")
    monkeypatch.setattr(runner, "reload_model", lambda *args: (object(), FakeTokenizer()))
    runner.evaluate(config, batch, factorial, runtime=runtime, objective="pd", arm="A", node=512)
    assert synced == [(config, batch, node_dir.parent, node_dir)]
    assert runner.read(node_dir / "evaluation/summary.json")["independent_reload_passed"] is True
