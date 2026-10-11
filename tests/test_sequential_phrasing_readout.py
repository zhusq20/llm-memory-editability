"""Contracts for truth-preserving phrasing and zero-update replay evidence."""

import copy
import json

import pytest
import torch

from llm_memory_editability import sequential_phrasing_readout as readout
from llm_memory_editability.realworld_composition_data import encode_example


class TinyTokenizer:
    eos_token_id = 255

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def fixture_data():
    tokenizer = TinyTokenizer()
    atoms = []
    for key, subject, subset in (("a0", "Alice", "A"), ("a1", "Bob", "A"), ("b0", "Carol", "B")):
        question = f"Who is the father of {subject}?"
        atoms.append(
            {
                "id": key,
                "edge": [subject, "father", "Dave"],
                "subset": subset,
                "question": question,
                "answer": "Dave",
                "aliases": ["D"],
                "unambiguous_answers": ["Dave"],
                "encoded": encode_example(tokenizer, question, "Dave"),
            }
        )
    return {
        "atoms": atoms,
        "train_compositions": [{"id": "train", "atom_ids": ["a0", "a1"], "question": "Chain?"}],
        "evaluation_compositions": [
            {"id": "aa", "atom_ids": ["a1", "a0"], "sequential_role": "AA"},
            {"id": "ab", "atom_ids": ["a0", "b0"], "sequential_role": "AB"},
            {"id": "ba", "atom_ids": ["b0", "a0"], "sequential_role": "BA"},
            {"id": "bb", "atom_ids": ["b0", "b0"], "sequential_role": "BB"},
        ],
    }


def prediction(key, answer="Dave", correct=1.0):
    return {
        "id": key,
        "prediction": answer,
        "generated_tokens": [1, 2, 255],
        "canonical_em": correct,
        "alias_em": correct,
        "unambiguous_alias_em": correct,
        "f1": correct,
        "eos": True,
        "truncated": False,
        "context_limit_reached": False,
        "generation_limit_reached": False,
        "gold": "Dave",
        "nll": 0.1,
        "has_unambiguous_answer": True,
    }


def test_templates_change_only_query_and_preserve_every_fact():
    data = fixture_data()
    original = copy.deepcopy(data)
    pools = readout.phrasing_records(
        data, TinyTokenizer(), {"father": "Name the father of {subject}."}
    )
    assert data == original
    assert [r["id"] for r in pools["canonical"]] == [r["id"] for r in data["atoms"]]
    assert len(pools["heldout"]) == 3
    for old, new in zip(pools["canonical"], pools["heldout"], strict=True):
        for key in ("id", "edge", "answer", "aliases", "unambiguous_answers", "subset"):
            assert old[key] == new[key]
        assert old["encoded"]["target"] == new["encoded"]["target"]
        assert old["encoded"]["prefix"] != new["encoded"]["prefix"]


@pytest.mark.parametrize(
    "templates", ({}, {"father": "No entity"}, {"father": "{subject}{subject}"})
)
def test_rejects_missing_relation_or_malformed_template(templates):
    with pytest.raises(ValueError):
        readout.phrasing_records(fixture_data(), TinyTokenizer(), templates)


def test_rejects_previously_trained_query_and_changed_canonical_encoding():
    with pytest.raises(ValueError, match="appeared"):
        readout.phrasing_records(
            fixture_data(), TinyTokenizer(), {"father": "Who is the father of {subject}?"}
        )
    data = fixture_data()
    data["atoms"][0]["encoded"]["target"][0] += 1
    with pytest.raises(ValueError, match="re-encoding"):
        readout.phrasing_records(data, TinyTokenizer(), {"father": "Name the father of {subject}."})


def test_subject_is_inverted_from_exact_original_template():
    row = fixture_data()["atoms"][0]
    assert readout.original_subject(row) == "Alice"
    row["question"] = "Who is Alice's father?"
    with pytest.raises(ValueError, match="Canonical"):
        readout.original_subject(row)


def test_smoke_is_fixed_first_facts_without_changing_template_validation():
    pool = readout.phrasing_records(
        fixture_data(), TinyTokenizer(), {"father": "Name the father of {subject}."}, limit=2
    )
    assert [r["id"] for r in pool["heldout"]] == ["a0", "a1"]
    with pytest.raises(ValueError):
        readout.phrasing_records(fixture_data(), TinyTokenizer(), {"father": "{subject}"}, limit=0)


def test_native_gate_uses_actual_tokens_and_scores_not_nll(tmp_path):
    native = {"atomic_A": [prediction("a0")], "atomic_B": [prediction("b0")]}
    path = tmp_path / "native.json"
    path.write_text(json.dumps(native))
    regenerated = [prediction("a0"), prediction("b0")]
    regenerated[0]["nll"] = 0.9
    assert readout.native_gate(regenerated, path)["passed"]
    regenerated[0]["generated_tokens"] = [1, 3, 255]
    result = readout.native_gate(regenerated, path)
    assert not result["passed"]
    assert result["exact_output_match_fraction"] == 0.5
    assert result["differences"][0]["fields"] == ["generated_tokens"]


def test_native_text_only_fallback_never_claims_verifiable_token_ids(tmp_path):
    old = prediction("a0")
    del old["generated_tokens"]
    path = tmp_path / "native.json"
    path.write_text(json.dumps({"atomic_A": [old], "atomic_B": []}))
    result = readout.native_gate([prediction("a0")], path)
    assert result["passed"] and result["token_ids_available_n"] == 0
    changed = prediction("a0", answer="Other", correct=0)
    assert not readout.native_gate([changed], path)["passed"]


def test_missing_native_coverage_is_rejected(tmp_path):
    path = tmp_path / "native.json"
    path.write_text(json.dumps({"atomic_A": [prediction("a0")], "atomic_B": []}))
    with pytest.raises(ValueError, match="lack native"):
        readout.native_gate([prediction("b0")], path)


def test_paired_metrics_preserve_both_directions_and_empty_groups():
    original = [prediction("a0"), prediction("b0", correct=0)]
    changed = [prediction("a0", correct=0), prediction("b0")]
    result = readout.paired_metrics(original, changed)
    assert result["canonical_correct_heldout_wrong_n"] == 1
    assert result["canonical_wrong_heldout_correct_n"] == 1
    assert result["heldout_minus_canonical_alias_em"] == 0
    assert readout.paired_metrics([], [])["n"] == 0
    with pytest.raises(ValueError):
        readout.paired_metrics(original, list(reversed(changed)))


def test_independent_replay_checks_every_token_both_snapshots_forms():
    raw = {
        snapshot: {form: [prediction("a0")] for form in ("canonical", "heldout")}
        for snapshot in readout.SNAPSHOTS
    }
    assert readout.compare_replay(raw, copy.deepcopy(raw), 1e-5)["passed"]
    replay = copy.deepcopy(raw)
    replay["final"]["heldout"][0]["generated_tokens"][-1] = 1
    assert not readout.compare_replay(raw, replay, 1e-5)["passed"]
    replay = copy.deepcopy(raw)
    replay["stage_a"]["canonical"][0]["nll"] += 1e-3
    assert not readout.compare_replay(raw, replay, 1e-5)["passed"]


def test_reader_never_loads_optimizer_and_freezes_parameters(monkeypatch):
    model = torch.nn.Linear(2, 3)
    checkpoint = {
        "step": 8,
        "model": model.state_dict(),
        "counts": [2, 2, 0, 4],
        "counters": {"examples": 8},
        "optimizer": "not a restorable optimizer",
    }
    monkeypatch.setattr(readout, "construct", lambda *args: torch.nn.Linear(2, 3))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    loaded, history = readout.load_reader(
        {"model": {}}, {"path": "unused", "step": 8}, fixture_data(), "cpu"
    )
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert all(parameter.grad is None for parameter in loaded.parameters())
    assert history["atomic_exposures"] == {"a0": 2, "a1": 2, "b0": 0}


def test_pool_restores_eval_mode_after_native_evaluator_returns_train(monkeypatch):
    model = torch.nn.Linear(2, 3).eval()

    def native_evaluate(model, records, *args):
        model.train()
        return {}, [prediction(row["id"]) for row in records]

    monkeypatch.setattr(readout, "evaluate", native_evaluate)
    rows = fixture_data()["atoms"]
    result = readout.evaluate_pool(model, rows, TinyTokenizer(), "cpu", 2)
    assert not model.training
    assert [row["id"] for row in result] == [row["id"] for row in rows]
