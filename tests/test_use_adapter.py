"""Contracts of use-adapter-development-v1: arm data and frozen E3 writes under the update."""

import copy
from pathlib import Path

import pytest
import torch

from llm_memory_editability import read_interface_data as d
from llm_memory_editability import read_interface_e3 as e3
from llm_memory_editability import read_interface_model as m
from llm_memory_editability import read_interface_v2_data as v
from llm_memory_editability import use_adapter as u


def written():
    known = {q["qid"]: True for q in d.screening_queries()}
    facts, _, _ = d.build_facts(known, seed=7)
    return [f for f in facts if f.split != "calibration"][:50]


def test_arms_differ_only_in_practiced_extra_data():
    facts = written()
    pu, pp = u.arm_examples(facts, "pu"), u.arm_examples(facts, "pp")
    replay = u.u0_replay(facts)
    assert len(replay) == 50 and pu[:50] == pp[:50] == replay
    assert len(pu) == len(pp) == 50 + 8 * len(u.practiced(facts))
    assert u.arm_examples(facts, "p0") == replay
    for f, e in zip(facts, replay, strict=True):
        assert e == {"user": d.fill(d.U0, f) + d.DIRECT, "assistant": f.v}


def test_extra_data_never_answers_held_out_operations_or_touches_test_facts():
    facts = written()
    g = v.graders()
    test = [f for f in facts if f.split == "test"]
    assert test and u.practiced(facts)
    for arm in ("pu", "pp"):
        for e in u.extra_examples(facts, arm):
            reply = e["assistant"]
            for continent in v.CONTINENT_ALIASES:
                reply = reply.replace(continent, "")
            owner = [f for f in u.practiced(facts) if f.h in e["user"]
                     and d.RELATIONS[f.relation] in e["user"]]  # fmt: skip
            assert len(owner) == 1, e["user"]
            f = owner[0]
            assert f.country_v not in g["country"].mentions(reply)
            currency = d.currency_of(f.country_v)
            assert not currency or currency not in g["currency"].mentions(e["assistant"])
            for t in test:
                assert not (t.h in e["user"] and d.RELATIONS[t.relation] in e["user"])
    # usage practice and paraphrases are E4's exact samples
    for f in u.practiced(facts):
        assert [(x["user"], x["assistant"]) for x in u.extra_examples([f], "pu")] == (
            v.practice_examples(f)
        )
        assert [(x["user"], x["assistant"]) for x in u.extra_examples([f], "pp")] == (
            v.paraphrase_examples(f)
        )
    # no evaluated prompt except U0 is trained
    trained = {e["user"] for arm in ("pu", "pp") for e in u.arm_examples(facts, arm)}
    for f in facts:
        for r in d.use_queries(f) + v.op_queries(f):
            if r["prompt"].startswith(d.fill(d.U0, f)):
                continue
            assert r["prompt"] not in trained


TOKENIZER = Path(m.MODELS["qwen3-1.7b"]["path"])


@pytest.mark.skipif(not TOKENIZER.exists(), reason="needs the Qwen3 tokenizer files")
def test_e3_writes_stay_frozen_while_released_weights_update():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.padding_side = "left"
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, tie_word_embeddings=False)  # fmt: skip
    model = Qwen3ForCausalLM(config).eval()
    reference = copy.deepcopy(model)
    for p in reference.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(1)
    frozen = [
        [{k: torch.randn(16, generator=gen) for k in ("g", "u", "w")} for _ in range(3)]
        for _ in range(2)
    ]
    wrappers = u.attach(model, frozen, device="cpu")
    saved = [[{k: t.clone() for k, t in f.items()} for f in w.frozen] for w in wrappers]
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        with_writes = model(input_ids=ids).logits
        for w in wrappers:
            w.mask = torch.zeros(1, 3)
        without = model(input_ids=ids).logits
        for w in wrappers:
            w.mask = None
    assert not torch.allclose(with_writes, without)
    params = u.trainable(model)
    assert len(params) == len(list(reference.parameters()))
    facts = written()[:4]
    examples = u.arm_examples(facts, "pu")[:12]
    replay = [{"user": "What is the capital of France?" + d.DIRECT, "assistant": "Paris",
               "replay": True, "mode": "direct"} for _ in range(8)]  # fmt: skip
    summary = u.train(model, reference, tok, examples, replay, steps=3, lr=1e-2, device="cpu")
    assert summary["steps"] == 3 and summary["kl_tokens"] > 0 and summary["ce_tokens"] > 0
    assert summary["trainable_parameters"] == sum(p.numel() for p in reference.parameters())
    for w, before in zip(wrappers, saved, strict=True):
        assert w.current is None and len(w.frozen) == 3
        for f, g in zip(w.frozen, before, strict=True):
            for k in f:
                assert torch.equal(f[k], g[k]) and not f[k].requires_grad
    ref = dict(reference.named_parameters())
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, ref[u.base_name(n)])]
    assert len(changed) == len(params)
    u.reset_base(model, reference)
    assert all(torch.equal(p, ref[u.base_name(n)]) for n, p in model.named_parameters())
    assert all(isinstance(layer.mlp, e3.Adapted) for layer in m.layers(model))
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, with_writes)
