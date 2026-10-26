import contextlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from llm_memory_editability import knowledge_method_diagnosis as diagnosis


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9
    answers = {"PD-A": [3, 4], "PD-B": [3], "SFT-A": [5, 6], "SFT-B": [5, 6, 7]}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text in self.answers:
            return list(self.answers[text])
        assert "DO-NOT-SELECT" not in text
        return [19, 20] if "CONTEXT_SECRET" in text else [11, 12]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(message["content"] for message in messages)


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = []

    def forward(self, input_ids, attention_mask, use_cache):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert use_cache is False
        self.seen.append(input_ids.clone())
        logits = torch.zeros(*input_ids.shape, 32)
        logits[..., 3] = 4
        return SimpleNamespace(logits=logits)


def write_data(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    # Case B occurs first; the later row for it must never replace its first row.
    rows = [
        {
            "id": "b0",
            "case_id": "B",
            "question": "Question B?",
            "context": "CONTEXT_SECRET",
            "pd_answer": "PD-B",
            "sft_answer": "SFT-B",
            "pd_truncated": True,
            "sft_truncated": False,
        },
        {
            "id": "a0",
            "case_id": "A",
            "question": "Question A?",
            "context": "CONTEXT_SECRET",
            "pd_answer": "PD-A",
            "sft_answer": "SFT-A",
            "pd_truncated": False,
            "sft_truncated": False,
        },
        {
            "id": "b1",
            "case_id": "B",
            "question": "DO-NOT-SELECT",
            "context": "CONTEXT_SECRET",
            "pd_answer": "DO-NOT-SELECT",
            "sft_answer": "DO-NOT-SELECT",
        },
    ]
    (shared / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    # No eval file exists: the diagnostic must not consult held-out answers.
    return rows


@pytest.mark.parametrize("arm", ["pd_lora", "sft_lora", "sft_full", "sft_down"])
def test_fit_selects_first_training_row_per_case_and_only_scores_answer_tokens(
    tmp_path, monkeypatch, arm
):
    rows = write_data(tmp_path)
    monkeypatch.setattr(diagnosis, "autocast", contextlib.nullcontext)
    model, tok = ConstantModel(), Tokenizer()
    config = {"n_cases": 2, "max_sequence_tokens": 128, "eval_batch_size": 2}
    result = diagnosis.fit_diagnostic(model, tok, config, tmp_path, arm)
    assert result["label"] == "teacher-target-fit"
    assert result["used_for_selection_or_stopping"] is False
    assert [row["id"] for row in result["rows"]] == ["b0", "a0"]
    assert [row["train_line"] for row in result["rows"]] == [1, 2]
    assert result["summary"]["n"] == 2
    assert result["train_data_sha256"] == diagnosis.digest(tmp_path / "shared/train.jsonl")
    assert len(model.seen) == 1
    assert model.seen[0][:, :2].tolist() == [[11, 12], [11, 12]]

    kind = "pd" if arm == "pd_lora" else "sft"
    expected_targets = []
    for source, record in zip(rows[:2], result["rows"], strict=True):
        target = tok.encode(source[f"{kind}_answer"])
        if not source[f"{kind}_truncated"]:
            target.append(tok.eos_token_id)
        expected_targets.append(target)
        assert record["target_token_ids"] == target
        assert record["target_key"] == f"{kind}_answer"
        assert record["valid_target_tokens"] == len(target)
        assert record["eos_tokens"] == target.count(tok.eos_token_id)
        assert record["non_eos_tokens"] == len(target) - target.count(tok.eos_token_id)
        logits = torch.zeros(len(target), 32)
        logits[:, 3] = 4
        expected_nll = F.cross_entropy(logits, torch.tensor(target))
        assert record["nll"] == pytest.approx(float(expected_nll))
        assert record["next_token_greedy_fraction"] == pytest.approx(target.count(3) / len(target))

    total = sum(map(len, expected_targets))
    assert result["summary"]["valid_target_tokens"] == total
    assert result["summary"]["eos_tokens"] == (1 if kind == "pd" else 2)
    assert result["summary"]["token_next_token_greedy_fraction"] == pytest.approx(
        sum(target.count(3) for target in expected_targets) / total
    )
    repeat = diagnosis.fit_diagnostic(model, tok, config, tmp_path, arm)
    assert repeat == result


def test_fit_rejects_missing_case_without_replacing_or_filtering(tmp_path, monkeypatch):
    write_data(tmp_path)
    monkeypatch.setattr(diagnosis, "autocast", contextlib.nullcontext)
    model = ConstantModel()
    with pytest.raises(ValueError, match="coverage"):
        diagnosis.fit_diagnostic(
            model,
            Tokenizer(),
            {"n_cases": 3, "max_sequence_tokens": 128, "eval_batch_size": 2},
            tmp_path,
            "pd_lora",
        )
    assert not model.seen
