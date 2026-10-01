"""Query weighting and probability contracts for the same-bridge follow-up."""

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_loop_data import build_world
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.grok_loop_same_bridge import select_same_bridge_donors
from llm_memory_editability.grok_loop_same_bridge_eval import (
    METRICS,
    evaluate_same_bridge,
    prediction_metrics,
    query_mean,
    summarize,
)


def test_query_average_does_not_overweight_queries_with_many_donors():
    index = np.array([0, 0, 0, 1])
    value = np.array([1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(query_mean(value, index, 3)[:2], [1.0, 0.0])
    score = summarize({metric: value for metric in METRICS}, index, np.ones(3, bool), 3)
    assert score["complete_accuracy"] == 0.5
    assert score["n_recipients"] == 2
    assert score["n_donor_evaluations"] == 4
    assert score["coverage"] == 2 / 3


def test_empty_donor_pool_stays_undefined():
    score = summarize(
        {metric: np.empty(0) for metric in METRICS},
        np.empty(0, dtype=np.int64),
        np.ones(2, bool),
        2,
    )
    assert score["n_recipients"] == 0
    assert all(score[metric] is None for metric in METRICS)
    assert np.isnan(query_mean(np.empty(0), np.empty(0, dtype=np.int64), 2)).all()


def test_generated_eos_and_answer_probability_have_distinct_definitions():
    prediction = {
        "answer": np.array([1, 2]),
        "stop": np.array([0, 1]),
        "answer_logits": np.array([[0.0, 2.0, 1.0], [0.0, 1.0, 2.0]]),
    }
    measured = prediction_metrics(prediction, np.array([1, 1]))
    np.testing.assert_array_equal(measured["answer_accuracy"], [1, 0])
    np.testing.assert_array_equal(measured["complete_accuracy"], [0, 0])
    np.testing.assert_array_equal(measured["eos_accuracy"], [0, 1])
    np.testing.assert_allclose(
        measured["target_probability"],
        np.array([np.exp(2), np.exp(1)]) / (1 + np.exp(1) + np.exp(2)),
    )
    np.testing.assert_array_equal(measured["target_margin"], [1, -1])


def test_probability_is_invariant_to_large_common_logit_offset():
    prediction = {
        "answer": np.array([1]),
        "stop": np.array([1]),
        "answer_logits": np.array([[0.0, 1.0, 2.0]]),
    }
    before = prediction_metrics(prediction, np.array([1]))
    prediction["answer_logits"] += 10000
    after = prediction_metrics(prediction, np.array([1]))
    np.testing.assert_allclose(
        before["target_probability"], after["target_probability"], atol=1e-12
    )


def test_reject_invalid_recipient_indices_and_nonfinite_logits():
    with pytest.raises(ValueError, match="outside"):
        query_mean(np.array([1.0]), np.array([-1]), 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        prediction_metrics(
            {
                "answer": np.array([1]),
                "stop": np.array([1]),
                "answer_logits": np.array([[0.0, np.nan]]),
            },
            np.array([1]),
        )


@pytest.mark.parametrize("repeats", [1, 3])
def test_real_component_evaluator_keeps_prefix_controls_and_all_donor_pairs(repeats):
    torch.set_num_threads(1)
    torch.manual_seed(722)
    spec = {
        "world_seed": 3,
        "hops": 2,
        "entities": 8,
        "relations": 3,
        "degree": 3,
        "phi": 0.4,
        "id_fraction": 0.5,
        "id_test_fraction": 0.1,
        "evaluation_size": 24,
        "steps": 64,
        "batch_size": 8,
        "n_atomic_per_batch": 2,
        "stream_seed": 813,
    }
    world = build_world(spec)
    donors = select_same_bridge_donors(world, spec)
    assert donors["common_valid"].sum() == 2
    model = (
        LoopGPT(ModelConfig(vocab_size=13, width=16, layers=1, heads=2, context=8), repeats=repeats)
        .eval()
        .requires_grad_(False)
    )
    arrays, report = evaluate_same_bridge(model, world, donors, device="cpu", batch_size=1024)
    assert report["n_common_queries"] == 2
    assert len(report["conditions"]) == 23
    assert report["engineering"]["parameters_unchanged"]
    for family in ("id", "ood"):
        np.testing.assert_array_equal(
            arrays[family + "_both_answer_logits"], arrays[family + "_full_answer_logits"]
        )
    if repeats == 1:
        assert report["engineering"]["c1_last_block_inert"]
        assert report["contrasts"]["paired:full"]["id_minus_ood"]["complete_accuracy"] == 0
