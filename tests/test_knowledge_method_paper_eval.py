import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_memory_editability import knowledge_method_paper_eval as P


def test_upstream_template_fingerprint():
    assert hashlib.sha256(P.HOTPOT_JUDGE_TEMPLATE.encode()).hexdigest() == (
        "d756fc8588dc77ffcfa647d92f9449b0e8b58387d48d4fe2a1aa150a98c5af7e"
    )
    prompt = P.create_judge_prompt("QUESTION", "GOLD", "ANSWER")
    assert "Question: QUESTION\nGround-truth answer: GOLD\nProposed answer: ANSWER" in prompt
    assert prompt.startswith("<|start_header_id|>system")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert "<grade>true</grad>" in prompt  # Preserve author's example typo.


def test_author_ast_parity_when_available():
    source = Path("/tmp/lm-prompt-distillation-20261021/evaluation/grade_answers_llm.py")
    if not source.exists():
        pytest.skip("Official source not mounted; fixed template hash remains checked")
    tree = ast.parse(source.read_text())
    selected = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("create_prompt", "extract_grade")
    ]
    ns = {
        "Dict": dict,
        "Any": object,
        "List": list,
        "LLM": object,
        "re": P.re,
        "Message": lambda role, text: SimpleNamespace(content=text),
        "Role": SimpleNamespace(USER="user"),
    }
    exec(compile(ast.Module(selected, type_ignores=[]), str(source), "exec"), ns)
    llm = SimpleNamespace(messages_to_prompt=lambda rows: rows[0].content)
    for q, g, a in [("Who?", "A", "A and B"), ("{q}\n", "'yes'", "<tag>false</tag>")]:
        assert ns["create_prompt"](q, g, a, "hotpotqa", llm) == (
            P.HOTPOT_JUDGE_TEMPLATE.format(q=q, gt=g, a=a)
        )
    for answer in [
        "<grade>false</grade>",
        "true and false",
        "<grade>\nfalse\n</grade>",
        "<GRADE>TRUE</GRADE>",
        "Invalid",
        "<grade>untrue</grade>",
    ]:
        assert P.extract_grade(answer) == ns["extract_grade"](answer)


class FakeJudge:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompts):
        self.calls.append(prompts)
        return [{"text": next(self.outputs), "truncated": False} for _ in prompts]


def records():
    return [
        dict(
            id=str(i),
            question=f"Q{i}",
            gold="yes",
            answer="Yes, indeed.",
            arm="base",
            phase="before",
            group="closed",
        )
        for i in range(3)
    ]


def test_grade_preserves_invalid_and_has_both_denominators(tmp_path):
    judge = FakeJudge(["<grade>true</grade>", "<grade>false</grade>", "unclear"])
    result = P.grade_records(records(), {"judge_batch_size": 2}, tmp_path, generator=judge)
    s = result["summary"]["all"]
    assert s["n"] == 3 and s["valid_n"] == 2 and s["invalid_n"] == 1
    assert s["paper_accuracy_valid_denominator"] == 0.5
    assert s["correct_fraction_all_records"] == 1 / 3
    assert s["coverage"] == 2 / 3
    assert [len(x) for x in judge.calls] == [2, 1]
    repeat = FakeJudge([])
    P.grade_records(records(), {"judge_batch_size": 2}, tmp_path, generator=repeat)
    assert repeat.calls == []
    changed = records()
    changed[0]["answer"] = "No"
    with pytest.raises(ValueError, match="changed"):
        P.grade_records(changed, {"judge_batch_size": 2}, tmp_path, generator=repeat)


def test_partial_resume_and_torn_tail(tmp_path):
    judge = FakeJudge(["true", "false", "invalid"])
    P.grade_records(records(), {"judge_batch_size": 2}, tmp_path, generator=judge)
    path = tmp_path / "grading.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b'{"id":')
    repeat = FakeJudge(["false", "invalid"])
    P.grade_records(records(), {"judge_batch_size": 2}, tmp_path, generator=repeat)
    assert len(repeat.calls[0]) == 2
    assert list(tmp_path.glob("grading.jsonl.torn-*"))
    assert len(path.read_text().splitlines()) == 3


def test_parser_and_substring_are_paper_rules_not_new_semantics():
    assert P.extract_grade("<reasoning>true</reasoning><grade>false</grade>") == (
        {"valid_grade": True, "grade": False}
    )
    assert P.extract_grade("false and true")["grade"] is True
    assert P.paper_substring_match("Annie is older than Terry Richardson", "Terry Richardson")
    assert P.paper_substring_match("yesterday", "yes")  # Original has no word boundaries.


def test_old_baseline_deduplication_requires_exact_equality(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared/eval.json").write_text(
        json.dumps([{"case_id": "c", "question": "Question?", "answer": "Gold"}])
    )
    rows = [
        {
            "id": f"c/{g}",
            "group": g,
            "gold": "Gold",
            "answer": "Gold",
            "prompt": "old short instruction",
            "em": 1,
            "f1": 1,
        }
        for g in ("closed", "context")
    ]
    obj = {"before": rows, "after": rows, "summary": {"retention": {"n": 317}}}
    for arm in ("a", "b"):
        dest = tmp_path / "runs" / arm / "eval/evaluate.json"
        dest.parent.mkdir(parents=True)
        dest.write_text(json.dumps(obj))
    result, provenance = P.old_records(tmp_path, ["a", "b"])
    assert len(result) == 6 and provenance["base_records_identical"]
    assert sum(r["arm"] == "base" for r in result) == 2
    assert {r["group"] for r in result} == {"closed", "oracle_context"}
    obj["before"][0]["answer"] = "Changed"
    dest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="baselines differ"):
        P.old_records(tmp_path, ["a", "b"])


def test_new_prompts_and_revision_alias(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared/eval.json").write_text(
        json.dumps(
            [{"case_id": "c", "question": "Who?", "answer": "A", "paragraphs": ["Doc1", "Doc2"]}]
        )
    )
    template_calls = []

    def render(messages, **kwargs):
        template_calls.append(kwargs)
        return messages[-1]["content"]

    tok = SimpleNamespace(apply_chat_template=render)
    rows = P.evaluation_rows({"n_cases": 1}, tmp_path, tok)
    assert rows[0]["prompt"] == "Who?"
    assert rows[1]["prompt"] == "Doc1\n\nDoc2\n\nQuestion: Who?"
    assert rows[1]["group"] == "oracle_context"
    assert all(call["enable_thinking"] is False for call in template_calls)
    result = P.grade_records(
        records(),
        {"judge_revision": "official-rev"},
        tmp_path / "graded",
        generator=FakeJudge(["true"] * 3),
    )
    assert result["source"]["judge_model_revision"] == "official-rev"


def test_generation_budget_and_resume_are_per_question(tmp_path, monkeypatch):
    import contextlib

    import torch

    from llm_memory_editability import knowledge_method_diagnosis as D

    monkeypatch.setattr(D, "autocast", contextlib.nullcontext)
    monkeypatch.setattr(D, "seed_all", lambda value: None)

    class Batch(dict):
        def __getattr__(self, key):
            return self[key]

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 9
        pad_token_id = 0

        def encode(self, prompt, **kwargs):
            return [1] * int(prompt)

        def __call__(self, prompts, **kwargs):
            width = max(map(int, prompts))
            return Batch(
                input_ids=torch.ones(len(prompts), width, dtype=torch.long),
                attention_mask=torch.ones(len(prompts), width, dtype=torch.long),
            )

        def decode(self, ids, **kwargs):
            return "Answer"

    class Model:
        config = SimpleNamespace(max_position_embeddings=8192)

        def __init__(self):
            self.calls = []

        def eval(self):
            return self

        def parameters(self):
            yield torch.tensor(0)

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            inputs = kwargs["input_ids"]
            return torch.cat((inputs, torch.tensor([[42, 9]] * inputs.shape[0])), dim=1)

    rows = [
        {
            "id": str(i),
            "case_id": str(i),
            "group": "closed",
            "question": "Q",
            "prompt": str(length),
            "gold": "Answer",
        }
        for i, length in enumerate([6100, 100])
    ]
    model = Model()
    config = {"seed": 1234, "eval_batch_size": 8}
    first = P.predict_records(model, Tokenizer(), rows, config, tmp_path, {"model": "stub"})
    assert sorted(c["max_new_tokens"] for c in model.calls) == [28, 500]
    assert all(c["temperature"] == 0.25 for c in model.calls)
    assert all(not r["truncated"] for r in first)
    model.calls.clear()
    assert P.predict_records(model, Tokenizer(), rows, config, tmp_path, {"model": "stub"}) == first
    assert not model.calls
    path = tmp_path / "answers.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    assert P.predict_records(model, Tokenizer(), rows, config, tmp_path, {"model": "stub"}) == first


def test_tracking_closes_on_failure(monkeypatch, tmp_path):
    calls = []
    run = SimpleNamespace(finish=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(P, "_tracking_run", lambda *args: run)

    def fail(*args):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(P, "_evaluate_impl", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        P.evaluate({}, tmp_path, "pd_full")
    assert calls == [{"exit_code": 1}]


def test_full_evaluation_unloads_qwen_before_judge(monkeypatch, tmp_path):
    import weakref

    import torch

    from llm_memory_editability import knowledge_method_diagnosis as D
    from llm_memory_editability import knowledge_method_factorial as F

    (tmp_path / "shared").mkdir()
    (tmp_path / "shared/eval.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "c",
                    "question": "Who?",
                    "answer": "A",
                    "paragraphs": ["FACT_ONLY_IN_ORACLE"],
                }
            ]
        )
    )
    (tmp_path / "shared/train.jsonl").write_text("{}\n")
    output = tmp_path / "runs/pd_full"
    output.mkdir(parents=True)
    (output / "weights.pt").write_bytes(b"weights")
    retention = tmp_path / "retention.json"
    retention.write_text("{}")
    alive = []

    class Model:
        after = False

    def load(config):
        model = Model()
        alive.append(weakref.ref(model))
        tok = SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: messages[-1]["content"]
        )
        return model, tok

    monkeypatch.setattr(D, "load_model", load)
    monkeypatch.setattr(D, "seed_all", lambda value: None)
    monkeypatch.setattr(D, "restore", lambda model, weights: setattr(model, "after", True))
    monkeypatch.setattr(D, "evaluation_rows", lambda *args: [{"group": "retention"}])
    monkeypatch.setattr(
        D, "predict", lambda model, *args: [{"id": "old", "strict": not model.after}]
    )
    monkeypatch.setattr(F, "configure_arm", lambda model, *args: model)
    monkeypatch.setattr(F, "fit_diagnostic", lambda model, *args: {"after": model.after})
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        P, "predict_records", lambda model, tok, rows, *args: [{**r, "answer": "A"} for r in rows]
    )

    class Judge(FakeJudge):
        def __init__(self, config):
            assert alive[-1]() is None
            super().__init__(["<grade>true</grade>"] * 4)

        def generate(self, prompts):
            assert all("FACT_ONLY_IN_ORACLE" not in p for p in prompts)
            return super().generate(prompts)

    monkeypatch.setattr(P, "_HFJudge", Judge)
    config = {
        "arms": ["pd_full"],
        "seed": 1234,
        "n_cases": 1,
        "model_path": "unused",
        "retention_source": str(retention),
        "paper_eval_wandb": False,
    }
    result = P.evaluate(config, tmp_path, "pd_full")
    assert result["complete"]
    assert result["summary"]["closed"]["before"]["correct_n"] == 1
    assert result["summary"]["oracle_context"]["after"]["correct_n"] == 1
    assert result["summary"]["retention"]["initially_correct_n"] == 1
    assert result["summary"]["retention"]["retained_n"] == 0
    assert result["teacher_target_fit"] == {"before": {"after": False}, "after": {"after": True}}


class QwenTemplate:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        assert kwargs["enable_thinking"] is False
        return (
            "<|im_start|>system\n" + messages[0]["content"] + "<|im_end|>\n"
            "<|im_start|>user\n" + messages[1]["content"] + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def test_qwen_judge_uses_own_template_and_disables_thinking():
    tok = QwenTemplate()
    config = {"judge_model": P.QWEN_JUDGE_MODEL, "judge_enable_thinking": False}
    prompt = P.create_judge_prompt("Q", "G", "A", config=config, tokenizer=tok)
    messages, kwargs = tok.calls[0]
    assert messages == [
        {"role": "system", "content": P.JUDGE_SYSTEM},
        {"role": "user", "content": P.HOTPOT_JUDGE_TEMPLATE.format(q="Q", gt="G", a="A")},
    ]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert "<|start_header_id|>" not in prompt and "<|eot_id|>" not in prompt
    assert prompt.endswith("<think>\n\n</think>\n\n")
    assert "<reasoning>" in prompt  # Preserve author's requested visible explanation.
    with pytest.raises(ValueError, match="requires its tokenizer"):
        P.create_judge_prompt("Q", "G", "A", config=config)
    with pytest.raises(ValueError, match="thinking=False"):
        P.create_judge_prompt(
            "Q", "G", "A", config={**config, "judge_enable_thinking": True}, tokenizer=tok
        )


def test_judge_model_revision_and_template_changes_cannot_reuse_cache(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    template = model_dir / "chat_template.jinja"
    template.write_text("template-v1")
    config = {
        "judge_model": P.QWEN_JUDGE_MODEL,
        "judge_model_path": str(model_dir),
        "judge_revision": "r1",
    }
    judge = FakeJudge(["<reasoning>Matches.</reasoning><grade>true</grade>"] * 3)
    judge.tok = QwenTemplate()
    output = tmp_path / "grading"
    result = P.grade_records(records(), config, output, generator=judge)
    assert result["source"]["judge_substitution"] is True
    assert result["source"]["judge_enable_thinking"] is False
    assert result["summary"]["all"]["thinking_marker_outputs_n"] == 0
    assert "not the original paper's judge" in result["source"]["evaluation_protocol"]
    for changed in [{**config, "judge_model": P.JUDGE_MODEL}, {**config, "judge_revision": "r2"}]:
        with pytest.raises(ValueError, match="changed"):
            P.grade_records(records(), changed, output, generator=judge)
    template.write_text("template-v2")
    with pytest.raises(ValueError, match="changed"):
        P.grade_records(records(), config, output, generator=judge)


def test_qwen_judge_generation_uses_native_eos_and_no_extra_special_tokens():
    import torch

    calls = {}

    class Batch(dict):
        def __getattr__(self, key):
            return self[key]

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 151645
        pad_token_id = 151643

        def __call__(self, prompts, **kwargs):
            calls["tokenizer"] = kwargs
            return Batch(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, ids, **kwargs):
            assert ids == [42, self.eos_token_id]
            return "<grade>true</grade><|im_end|>"

    def generate(**kwargs):
        calls["generate"] = kwargs
        return torch.tensor([[1, 2, 42, 151645, 151643]])

    judge = P._HFJudge.__new__(P._HFJudge)
    judge.torch = torch
    judge.device = "cpu"
    judge.identity = P._judge_identity({"judge_model": P.QWEN_JUDGE_MODEL})
    judge.tok = Tokenizer()
    judge.stop_ids = [judge.tok.eos_token_id]
    judge.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=8192), generate=generate
    )
    output = judge.generate(["Qwen chat prompt"])
    assert calls["tokenizer"]["add_special_tokens"] is False
    assert calls["generate"]["eos_token_id"] == [151645]
    assert calls["generate"]["pad_token_id"] == 151643
    assert calls["generate"]["do_sample"] is False
    assert calls["generate"]["max_new_tokens"] == 512
    assert output[0]["truncated"] is False
