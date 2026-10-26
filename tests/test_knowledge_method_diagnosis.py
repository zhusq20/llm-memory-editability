import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from llm_memory_editability.knowledge_method_diagnosis import (
    collate,
    configure_arm,
    encode_sample,
    forward_kl,
    normalize_answer,
    restore,
    score_answer,
    state,
    summarize,
)


@pytest.mark.parametrize("truncated_kind", ["pd", "sft"])
def test_prepared_answer_truncation_controls_eos_for_every_arm(
    tmp_path, monkeypatch, truncated_kind
):
    from llm_memory_editability import knowledge_method_data as data

    source = tmp_path / "author.csv"
    source.write_text("Held-out composite question?;Held-out answer;A source paragraph.\n")

    def generate(users, temperature, seed):
        del seed
        outputs = []
        for user in users:
            assert "Held-out" not in user
            is_question = user.startswith("Here is a paragraph")
            kind = "pd" if temperature == 1.5 else "sft"
            cut = not is_question and kind == truncated_kind
            outputs.append(
                {
                    "text": (
                        "<question>A single-hop training question?</question>"
                        if is_question
                        else "A partial answer"
                        if cut
                        else "A complete answer"
                    ),
                    "generated_tokens": 512 if cut else 3,
                    "truncated": cut,
                    "finish_reason": "length" if cut else "eos",
                }
            )
        return outputs

    monkeypatch.setattr(data, "_load_generator", lambda _: generate)
    batch = tmp_path / "batch"
    data.prepare(
        {
            "source_csv": str(source),
            "model_path": str(tmp_path / "unused-model"),
            "n_cases": 1,
            "questions_per_paragraph": 1,
            "generation_batch_size": 1,
            "expected_paragraphs": 1,
            "expected_training_rows": 1,
        },
        batch,
    )
    row = json.loads((batch / "shared/train.jsonl").read_text())
    assert row[f"{truncated_kind}_truncated"]

    class FakeTokenizer:
        eos_token_id = 999

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return [ord(char) for char in text]

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "\n".join(message["content"] for message in messages) + "\nAssistant:"

    tok = FakeTokenizer()
    for arm in ("pd_lora", "sft_lora", "sft_full", "sft_down"):
        kind = "pd" if arm == "pd_lora" else "sft"
        expected = tok.encode(row[f"{kind}_answer"])
        if kind != truncated_kind:
            expected.append(tok.eos_token_id)
        sample = encode_sample(tok, row, arm, max_length=4096)
        assert sample["answer"] == expected
        encoded = collate([sample], pad_id=0, device="cpu")
        for side in ("teacher", "student"):
            value = encoded[side]
            assert value["labels"][value["answer_mask"]].tolist() == expected


def test_shifted_masks_cover_only_answer_and_eos():
    samples = [
        {"student": [1, 2], "teacher": [8, 9, 1, 2], "answer": [3, 4]},
        {"student": [5], "teacher": [9, 5], "answer": [7]},
    ]
    result = collate(samples, 0, device="cpu")
    for side in ("student", "teacher"):
        value = result[side]
        assert value["labels"][value["answer_mask"]].tolist() == [3, 4, 7]
        assert value["answer_mask"].sum(1).tolist() == [2, 1]
    assert result["student"]["input_ids"][0].tolist() == [1, 2, 3]
    assert not result["student"]["answer_mask"][0, 0]


def test_forward_kl_matches_author_reduction_and_gradient():
    generator = torch.Generator().manual_seed(77)
    s = torch.randn(6, 17, generator=generator, requires_grad=True)
    t = torch.randn(6, 17, generator=generator)
    counts = torch.tensor([1, 5])
    actual = forward_kl(s, t, counts, 2.0)
    # train_student.py compute_logit_loss: token sum -> per sample token mean -> batch mean.
    tokens = F.kl_div(
        F.log_softmax(s / 2, -1), F.log_softmax(t / 2, -1), log_target=True, reduction="none"
    ).sum(-1)
    expected = (tokens[0] + tokens[1:].mean()) / 2
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual, s, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, s)[0]
    torch.testing.assert_close(ga, ge)
    assert not torch.isclose(actual, expected * 4)
    assert not torch.isclose(actual, tokens.mean())


def test_kl_refuses_alignment_loss():
    with pytest.raises(ValueError, match="Unaligned"):
        forward_kl(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([1]), 2)
    with pytest.raises(ValueError, match="Every example"):
        forward_kl(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([0, 2]), 2)


@pytest.mark.parametrize("arm", ["sft_down", "sft_full", "sft_lora", "pd_lora"])
def test_parameter_contract_and_reload(arm):
    transformers = pytest.importorskip("transformers")
    if arm.endswith("lora"):
        pytest.importorskip("peft")
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=True,
    )
    settings = json.loads(
        (
            Path(__file__).parents[1] / "configs/knowledge-method-diagnosis-development-v1.json"
        ).read_text()
    )
    settings.update(lora_r=2, lora_alpha=4, gradient_checkpointing=False)
    model = configure_arm(transformers.Qwen3ForCausalLM(config), settings, arm)
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trained
    if arm == "sft_down":
        assert all(".mlp.down_proj." in n for n in trained)
        assert len(trained) == 2
    elif arm.endswith("lora"):
        assert all("lora_" in n for n in trained)
        assert any("lm_head" in n for n in trained)
    else:
        assert len(trained) == len(list(model.named_parameters()))
    saved = state(model)
    first = next(p for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        first.add_(1)
    restore(model, saved)
    for name, p in model.named_parameters():
        if name in saved:
            torch.testing.assert_close(p, saved[name])
    with pytest.raises(ValueError):
        restore(model, {})


def test_hotpot_scoring_and_retention_do_not_compensate_for_lost_answers():
    assert normalize_answer("The U.S.!") == "us"
    assert score_answer("New York", "New York City")["f1"] == 0.8
    assert score_answer("yes because...", "yes")["f1"] == 0
    before = []
    after = []
    for group in ("closed", "context", "retention"):
        for i in range(2):
            before.append(
                {
                    "id": f"{group}{i}",
                    "group": group,
                    "em": float(i == 0),
                    "f1": float(i == 0),
                    "strict": i == 0,
                }
            )
            after.append(
                {
                    "id": f"{group}{i}",
                    "group": group,
                    "em": float(i == 1),
                    "f1": float(i == 1),
                    "strict": i == 1,
                }
            )
    result = summarize(before, after)
    assert result["retention"]["before"] == result["retention"]["after"] == 0.5
    assert result["retention"]["retained_n"] == 0
    assert result["retention"]["newly_correct_n"] == 1
