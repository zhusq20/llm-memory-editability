"""Contracts that affect robustness and depth conclusions."""

import copy
import json

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.loop_hop_data import load, prepare
from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT
from llm_memory_editability.loop_robustness import (
    audit_function_world,
    nested_indices,
    rewrite_function_world,
    stream_predictions,
    summarize_predictions,
)


@pytest.fixture
def function_data(tmp_path):
    config = {
        "data": {
            "world_seed": 71,
            "entities_per_graph": 20,
            "relations": 4,
            "evaluation_templates": 2,
            "train_compositions_per_hop": 30,
            "training_panel_size": 10,
        }
    }
    prepare(config, tmp_path / "permutation")
    rewrite_function_world(tmp_path / "permutation", tmp_path / "function", 73)
    return load(tmp_path / "permutation"), load(tmp_path / "function")


def test_data_change_preserves_queries_splits_and_trained_premises(function_data):
    (a, pa, _), (b, pb, metadata) = function_data
    for name in pa:
        np.testing.assert_array_equal(pa[name][:, :-1], pb[name][:, :-1])
    for name in ("train_2", "train_3", "train_4", "atoms"):
        np.testing.assert_array_equal(a[name][:, :-1], b[name][:, :-1])
    assert not np.all(np.sort(b["permutations"], axis=-1) == np.arange(20))
    audit = audit_function_world(b, pb, metadata)
    assert audit["passed"]
    assert audit["graph_b_composition_training_examples"] == 0
    assert audit["panels"]["ood_4"]["relation_sequence_only_oracle_accuracy"] > 1 / 20


def test_bad_path_and_query_leakage_are_rejected(function_data):
    _, (world, panels, metadata) = function_data
    broken = copy.deepcopy(panels)
    broken["ood_3"][0, -1] = (broken["ood_3"][0, -1] - 20 + 1) % 20 + 20
    with pytest.raises(ValueError, match="Path target"):
        audit_function_world(world, broken, metadata)
    broken = copy.deepcopy(world)
    broken["train_3"][0] = panels["id_3"][0]
    with pytest.raises(ValueError, match="leaked"):
        audit_function_world(broken, panels, metadata)


def test_subsets_are_nested_repeatable_and_model_independent():
    small = nested_indices("ood_4", 5000, 256, 71)
    large = nested_indices("ood_4", 5000, 1024, 71)
    np.testing.assert_array_equal(small, large[:256])
    np.testing.assert_array_equal(small, nested_indices("ood_4", 5000, 256, 71))
    assert len(np.unique(large)) == 1024


def test_unchanged_accuracy_can_hide_correct_answers_lost_and_gained():
    prediction = np.array([[3, 0, 0], [0, 4, 0], [3, 4, 0], [3, 0, 5]])
    stat = summarize_predictions(prediction, np.array([3, 4, 5]), reference_round=1)
    assert stat["answer_accuracy"][0] == stat["answer_accuracy"][1]
    assert stat["reference_correct_now_wrong"][1] == 1
    assert stat["reference_wrong_now_correct"][1] == 1
    assert stat["reference_correct_ever_lost_after_reference"][-1] == 1
    assert stat["reference_correct_now_wrong"][-1] == 0
    assert stat["first_correct_round"] == [1, 2, 4]


@pytest.mark.parametrize("stage", ("smoke", "smoke-grid", "train", "grid"))
def test_native_stage_initializes_threads_and_cuda_only_once(monkeypatch, tmp_path, stage):
    import run_loop_robustness as runner

    plan = {
        "batch": "test",
        "jobs": {"pair": {"kind": "new", "config": "config.json", "arm": "invariant"}},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run",
            stage,
            "--plan",
            str(tmp_path / "plan.json"),
            "--job",
            "pair",
            "--base",
            str(tmp_path / "out"),
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Wrapper must leave runtime initialization to the original runner")

    monkeypatch.setattr(runner.torch, "set_num_threads", forbidden)
    monkeypatch.setattr(runner.torch, "set_num_interop_threads", forbidden)
    monkeypatch.setattr(runner.torch.cuda, "set_device", forbidden)
    called = []
    monkeypatch.setattr(runner.native, "main", lambda: called.append(list(runner.sys.argv)))
    runner.main()
    assert called[0][1] == stage


@pytest.mark.parametrize("handoff", ("ema", "invariant", "additive", "content", "plain"))
def test_streaming_argmax_matches_original_trace_and_fresh_runs(handoff):
    torch.manual_seed(71)
    config = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    model = LoopNativeHandoffGPT(config, handoff=handoff).double().eval()
    tokens = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5]])
    positions = torch.tensor([[3], [3]])
    streamed = torch.stack(list(stream_predictions(model, tokens, positions, 8)))
    logits, _, _ = model.trace(tokens, positions, 8)
    assert torch.equal(streamed, logits[:, :, 0].argmax(-1))
    for r in (1, 4, 8):
        fresh, _ = model.run(tokens, positions, rounds=r, carry=False)
        assert torch.equal(streamed[r - 1], fresh[:, 0].argmax(-1))
