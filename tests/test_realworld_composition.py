"""Scientific contracts for variable-answer, source-grounded 2Wiki training."""

import ast
import copy
import re
import string
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability import realworld_composition as rw
from llm_memory_editability.realworld_composition_data import (
    Components,
    answer_scores,
    extract_chain,
    normalize_answer,
)


def small_config(dropout=0.0):
    return dict(vocab_size=31, positions=32, hidden_size=8, attention_heads=2, dropout=dropout)


def spec(layers=4, repeats=1):
    return dict(initialization=42, unique_layers=layers, repeats=repeats, microbatch_size=2)


def records():
    return [
        dict(
            id=str(i),
            encoded=dict(
                prefix=[1, 2, 3],
                target=[4] * (i % 3 + 1) + [30],
                input=[1, 2, 3] + [4] * (i % 3 + 1),
            ),
        )
        for i in range(5)
    ]


def test_author_scoring_parity():
    path = Path(
        "docs/development-artifacts/realworld-composition-v1/reference/2wikimultihop_evaluate_v1.1.py"
    )
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"normalize_answer", "f1_score", "exact_match_score"}
    ]
    scope = dict(re=re, string=string, Counter=Counter)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    for prediction in [
        "The United States",
        "United States.",
        "12 June 1516",
        "",
        "yes",
        "no",
        "New-York",
        "NY",
    ]:
        golds = ["United States", "U.S.", "12 June 1516", "New York", "yes"]
        assert normalize_answer(prediction) == scope["normalize_answer"](prediction)
        expected = (
            max(float(scope["exact_match_score"](prediction, g)) for g in golds),
            max(scope["f1_score"](prediction, g)[0] for g in golds),
        )
        assert answer_scores(prediction, golds) == expected


def test_chain_uses_ids_and_expands_literal_placeholders():
    row = dict(
        type="compositional",
        evidences_id=[["Q2", "date of death", "date_information"], ["Q1", "father", "Q2"]],
        evidences=[["Bridge alias", "date of death", "12 June 1516"], ["Head", "father", "Bridge"]],
    )
    edges, _ = extract_chain(row)
    assert edges == (("Q1", "father", "Q2"), ("Q2", "date of death", "literal:12 june 1516"))
    row["evidences_id"][0][0] = "Q3"
    assert extract_chain(row) is None


def test_grouping_joins_head_bridge_but_not_shared_tail_answers():
    groups = Components()
    groups.union("film1", "director1")
    groups.union("director1", "parent1")
    groups.union("film2", "director2")
    assert groups.find("film1") == groups.find("parent1")
    assert groups.find("film1") != groups.find("film2")


def test_common_initialization_and_digest_do_not_change_layers():
    models = [
        rw.construct(small_config(), spec(layers, repeat), "cpu")
        for layers, repeat in [(8, 1), (4, 2), (4, 1)]
    ]
    before = [rw.model_digest(m) for m in models]
    assert len({rw.shared_prefix_digest(m) for m in models}) == 1
    assert [len(m.transformer.h) for m in models] == [8, 4, 4]
    assert [rw.model_digest(m) for m in models] == before
    for m in models:
        assert (
            m.transformer.wte.weight.data_ptr()
            == dict(m.named_parameters())["transformer.wte.weight"].data_ptr()
        )


def test_answer_eos_supervision_and_equal_example_weight():
    x, positions, labels = rw.pack(records()[:2], 30, "cpu")
    assert positions[0, :2].tolist() == [2, 3]
    assert labels[0].tolist() == [4, 30, -100]
    assert x[0, :4].tolist() == [1, 2, 3, 4]
    logits = torch.randn(2, 3, 31, requires_grad=True)
    loss = rw.example_losses(logits, labels)
    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(
                logits[i, labels[i] != -100], labels[i, labels[i] != -100]
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert torch.equal(logits.grad[0, 2], torch.zeros(31))


@pytest.mark.parametrize("layers,repeats", [(8, 1), (4, 2), (4, 1)])
def test_causal_prefix_is_unchanged_by_future_padding(layers, repeats):
    model = rw.construct(small_config(), spec(layers, repeats), "cpu").eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    baseline = model(tokens, torch.tensor([[2]]))
    tokens[:, 3:] = 29
    torch.testing.assert_close(model(tokens, torch.tensor([[2]])), baseline, rtol=0, atol=0)


def test_partial_epoch_and_stream_restore_preserve_order():
    stream = rw.BatchStream(7, 91)
    a, b = stream.batch(5), stream.batch(5)
    assert len(a) == 5 and len(b) == 2
    assert sorted(np.concatenate([a, b]).tolist()) == list(range(7))
    state = stream.state_dict()
    next_ids = stream.batch(5)
    restored = rw.BatchStream(7, 999)
    restored.load_state_dict(state)
    assert np.array_equal(next_ids, restored.batch(5))


def test_accumulation_matches_full_batch_with_unequal_answers():
    a = rw.construct(small_config(), spec(), "cpu")
    b = rw.construct(small_config(), spec(), "cpu")
    opt_a = torch.optim.SGD(a.parameters(), lr=0.001)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.001)
    full = {**spec(), "microbatch_size": 5}
    result_a = rw.update(a, opt_a, records(), full, "cpu", 30)
    result_b = rw.update(b, opt_b, records(), spec(), "cpu", 30)
    assert abs(result_a["loss"] - result_b["loss"]) < 1e-6
    for left, right in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=1e-7)


def test_dropout_adam_restore_reproduces_next_update():
    model = rw.construct(small_config(0.1), spec(), "cpu")
    optimizer = rw.optimizer_for(model, 0.001, 0.1)
    rw.update(model, optimizer, records(), spec(), "cpu", 30)
    saved = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        torch.get_rng_state(),
    )
    rw.update(model, optimizer, records(), spec(), "cpu", 30)
    expected = copy.deepcopy(model.state_dict())
    resumed = rw.construct(small_config(0.1), spec(), "cpu")
    resumed_opt = rw.optimizer_for(resumed, 0.001, 0.1)
    resumed.load_state_dict(saved[0])
    resumed_opt.load_state_dict(saved[1])
    torch.set_rng_state(saved[2])
    rw.update(resumed, resumed_opt, records(), spec(), "cpu", 30)
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_generation_receives_question_prefix_without_gold_answer():
    class Model(torch.nn.Module):
        def forward(self, tokens, positions):
            assert 27 not in tokens.tolist()[0]
            logits = torch.zeros((len(tokens), 1, 31))
            logits[:, :, 30] = 1
            return logits

    tokenizer = SimpleNamespace(eos_token_id=30, decode=lambda ids, **kw: str(ids))
    result = rw.generate(Model(), [[1, 2, 3]], tokenizer, "cpu")
    assert result[0]["generated_tokens"] == [30]


def test_generation_limits_each_prompt_without_shortening_other_answers():
    class Model(torch.nn.Module):
        def forward(self, tokens, positions):
            assert tokens.shape[1] <= 8
            assert int(positions.max()) < tokens.shape[1]
            logits = torch.zeros((len(tokens), 1, 31))
            logits[:, :, 5] = 1
            return logits

    tokenizer = SimpleNamespace(eos_token_id=30, decode=lambda ids, **kw: str(ids))
    results = rw.generate(Model(), [[1] * 7, [2, 3]], tokenizer, "cpu", limit=4, sequence_limit=8)
    assert [len(r["generated_tokens"]) for r in results] == [1, 4]
    assert results[0]["context_limit_reached"] and not results[0]["eos"]
    assert results[1]["generation_limit_reached"] and not results[1]["context_limit_reached"]


def test_invalid_bridge_failure_does_not_lower_mean_answer_nll():
    predictions = [
        dict(
            canonical_em=1.0,
            alias_em=1.0,
            unambiguous_alias_em=1.0,
            f1=1.0,
            nll=2.0,
            eos=True,
            truncated=False,
        ),
        dict(
            canonical_em=0.0,
            alias_em=0.0,
            unambiguous_alias_em=0.0,
            f1=0.0,
            nll=None,
            eos=False,
            truncated=True,
        ),
    ]
    result = rw.aggregate(predictions)
    assert result["n"] == 2 and result["nll_n"] == 1
    assert result["alias_em"] == 0.5 and result["nll"] == 2.0


def test_autonomous_second_call_uses_emitted_bridge(monkeypatch):
    seen = []

    def fake_evaluate(model, items, tokenizer, device):
        seen.extend(r["question"] for r in items)
        predictions = [
            dict(
                id=r["id"],
                prediction="Wrong Person",
                canonical_em=0.0,
                alias_em=0.0,
                unambiguous_alias_em=0.0,
                f1=0.0,
                nll=1.0,
                eos=True,
                truncated=False,
                role=r.get("role"),
                required_role=r.get("required_role"),
            )
            for r in items
        ]
        return rw.aggregate(predictions), predictions

    monkeypatch.setattr(rw, "evaluate", fake_evaluate)
    monkeypatch.setattr(rw, "encode_example", lambda *a: {})
    atoms = [dict(id=i, question="Atomic", answer="Gold Bridge") for i in ["a", "b"]]
    chain = dict(
        id="c",
        question="Original question",
        answer="Final",
        atom_ids=["a", "b"],
        edges=[["Q1", "father", "Q2"], ["Q2", "place of birth", "Q3"]],
        role="OO",
        required_role="OO",
    )
    data = dict(
        atoms=atoms,
        train_compositions=[],
        evaluation_compositions=[chain],
        panels=dict(atomic=["a", "b"], train_composition=[], test_oo=["c"]),
    )
    metrics, _ = rw.evaluate_dataset(None, data, None, "cpu", autonomous=True)
    assert "Where was Wrong Person born?" in seen
    assert "Where was Gold Bridge born?" not in seen
    assert metrics["test_prerequisites_correct"]["coverage"] == 0.0
