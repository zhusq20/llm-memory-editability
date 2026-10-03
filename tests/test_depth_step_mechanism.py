"""Causal, data-only selection and frozen parameter contracts for mechanisms."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.depth_step import prompt_rows, truth_path_details
from llm_memory_editability.depth_step_mechanism import (
    edit_cases,
    edit_one,
    entity_readouts,
    generate,
    select_donors,
    trace_analysis,
    traced_forward,
)
from llm_memory_editability.grok_loop_model import LoopGPT


@pytest.fixture
def world():
    entities, relations = 8, 2
    atoms = np.asarray(
        [
            [h + 3, r + entities + 3, (h + r + 1) % entities + 3]
            for h in range(entities)
            for r in range(relations)
        ],
        dtype=np.int64,
    )
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    rows = np.asarray(
        [
            [h, r1, r2, lookup[(lookup[(h, r1)], r2)]]
            for h in range(3, entities + 3)
            for r1 in range(entities + 3, entities + relations + 3)
            for r2 in range(entities + 3, entities + relations + 3)
        ],
        dtype=np.int64,
    )
    return {
        "atomic": atoms,
        "familiar_2": rows[::2],
        "strict_2": rows[1::2],
        "metadata": {
            "entities": entities,
            "relations": relations,
            "separator_token": entities + relations + 3,
        },
    }


@pytest.fixture
def model(world):
    torch.manual_seed(714)
    return (
        LoopGPT(
            ModelConfig(vocab_size=14, width=8, layers=2, heads=2, context=9),
            repeats=2,
            dropout=0.0,
        )
        .double()
        .eval()
    )


def test_trace_matches_real_model_logits_and_records_every_execution(model, world):
    tokens = torch.as_tensor(prompt_rows(world["familiar_2"][:3], 13))
    with torch.no_grad():
        reference = model(tokens)
        actual, cache = traced_forward(model, tokens)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert cache["postresidual"].shape == (4, 3, 5, 8)
    assert cache["attention_map"].shape == (4, 3, 2, 5, 5)
    maps = cache["attention_map"]
    torch.testing.assert_close(maps.sum(-1), torch.ones_like(maps.sum(-1)))
    assert not torch.count_nonzero(maps.triu(1))


def test_first_relation_states_cannot_see_second_relation_or_separator(model, world):
    full = torch.as_tensor(prompt_rows(world["familiar_2"][:3], 13))
    altered = full.clone()
    altered[:, 3:] = 0
    donor = torch.cat((full[:, :3], torch.full((3, 1), 13)), 1)
    with torch.no_grad():
        _, original_cache = traced_forward(model, full)
        _, altered_cache = traced_forward(model, altered)
        _, donor_cache = traced_forward(model, donor)
    for name in ("mlp_delta", "postresidual"):
        torch.testing.assert_close(
            original_cache[name][:, :, 2], altered_cache[name][:, :, 2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            original_cache[name][:, :, 2], donor_cache[name][:, :, 2], rtol=0, atol=0
        )


@pytest.mark.parametrize("component", ["mlp_delta", "postresidual"])
def test_self_patch_preserves_logits_and_position_scope(model, world, component):
    tokens = torch.as_tensor(prompt_rows(world["familiar_2"][:3], 13))
    with torch.no_grad():
        before, cache = traced_forward(model, tokens)
        self_patch = {(1, component, 2): cache[component][1, :, 2]}
        same, _ = traced_forward(model, tokens, self_patch)
        changed, _ = traced_forward(
            model, tokens, {(1, component, 2): cache[component][1, :, 2] + 1}
        )
    torch.testing.assert_close(same, before, rtol=0, atol=0)
    torch.testing.assert_close(changed[:, :2], before[:, :2], rtol=0, atol=0)


def test_trace_refuses_training_and_unknown_patch(model, world):
    tokens = torch.as_tensor(prompt_rows(world["familiar_2"][:1], 13))
    with pytest.raises(ValueError, match="evaluation"):
        traced_forward(copy.deepcopy(model).train(), tokens)
    with pytest.raises(ValueError, match="Unknown"):
        traced_forward(model, tokens, {(8, "postresidual", 2): torch.zeros(1, 8)})


def test_donors_are_graph_valid_and_answer_blind(world):
    rows = world["familiar_2"]
    result = select_donors(world, rows)
    nodes, _ = truth_path_details(world, rows)
    atoms = world["atomic"]
    same = atoms[result["same_bridge_atomic_indices"]]
    changed = atoms[result["different_bridge_atomic_indices"]]
    np.testing.assert_array_equal(same[:, 2], nodes[:, 1])
    assert np.all(changed[:, 2] != nodes[:, 1])
    assert np.all(result["different_bridge_tail"] != rows[:, -1])
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    expected = [
        lookup[(int(donor[2]), int(query[2]))] for donor, query in zip(changed, rows, strict=True)
    ]
    np.testing.assert_array_equal(expected, result["different_bridge_tail"])
    donor_prompts = prompt_rows(changed, 13)
    assert donor_prompts.shape[1] == 4
    np.testing.assert_array_equal(donor_prompts[:, 1:3], changed[:, :2])


def test_entity_readouts_are_exact_unfitted_readouts(model, world):
    rows = world["familiar_2"][:3]
    tokens = torch.as_tensor(prompt_rows(rows, 13))
    nodes, _ = truth_path_details(world, rows)
    with torch.no_grad():
        logits, cache = traced_forward(model, tokens)
        arrays = entity_readouts(model, cache, {"bridge": nodes[:, 1]}, 8)
    expected = logits.softmax(-1)[torch.arange(3), 2, torch.as_tensor(nodes[:, 1])]
    np.testing.assert_allclose(
        arrays["bridge_residual_vocab_probability"][-1, :, 2],
        expected.numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert arrays["bridge_mlp_embedding_cosine_rank"].shape == (4, 3, 5)
    assert set(arrays) == {
        "bridge_residual_vocab_probability",
        "bridge_residual_entity_rank",
        "bridge_residual_entity_probability",
        "bridge_mlp_embedding_cosine",
        "bridge_mlp_embedding_cosine_rank",
    }


def test_edit_data_partition_recomputes_counterfactual_truth_without_behavior(world):
    cases = edit_cases(world, n_facts=2, replay_n=4)
    for case in cases:
        target_index = case["atomic_index"]
        assert target_index not in case["replay_indices"]
        assert not (
            set(map(tuple, case["tasks"]["R_atomic"])) & set(map(tuple, case["tasks"]["U_atomic"]))
        )
        changed_world = copy.deepcopy(world)
        changed_world["atomic"][target_index] = case["new_fact"]
        for name, rows in case["tasks"].items():
            if name.startswith("D_") and len(rows):
                truth_path_details(changed_world, rows)
            if name.startswith("U_") and rows.shape[1] == 4 and len(rows):
                _, path = truth_path_details(world, rows)
                assert target_index not in path
                truth_path_details(changed_world, rows)
        assert len(case["tasks"]["D_first_familiar_2"]) > 0
        assert case["propagation_scope"]["D_first_familiar_2"]["changed_answer_n"] == len(
            case["tasks"]["D_first_familiar_2"]
        )


def test_edit_freezes_all_but_one_shared_down_matrix_and_preserves_parent(model, world):
    case = edit_cases(world, n_facts=1, replay_n=4)[0]
    before = copy.deepcopy(model.state_dict())
    original_flags = [parameter.requires_grad for parameter in model.parameters()]
    result, raw = edit_one(model, case, world, "cpu", "edit", nodes=(0, 2), lr=0.01)
    assert result["changed_state_tensors"] == ["blocks.0.mlp.down.weight"]
    assert result["weight_delta_l2"] > 0
    assert raw["mlp_down_weight_delta"].shape == (8, 32)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert original_flags == [parameter.requires_grad for parameter in model.parameters()]
    assert set(result["history"][-1]["metrics"]) == set(case["tasks"])
    assert raw["D_first_familiar_2_changed_answer_mask"].all()
    assert result["history"][-1]["metrics"]["D_first_familiar_2"]["changed_answer_coverage"] == 1


def test_generation_retains_free_eos_and_full_vocabulary(model, world):
    rows = world["atomic"][:2]
    metrics, raw = generate(model, rows, world, "cpu")
    assert raw["predictions"].shape == (2, 2)
    assert raw["answer_probabilities"].shape == (2, 14)
    assert metrics["accuracy"] <= metrics["answer_accuracy"]


def test_trace_analysis_integration_covers_all_prespecified_layers(model, world, tmp_path):
    report = trace_analysis(model, world, tmp_path / "trace", "cpu", max_queries=4)
    assert report["selected_queries"] == 4
    assert report["forward_all_argmax_identical"]
    assert len(report["interventions"]) == 3 * 4 * 2 * 2
    raw = np.load(tmp_path / "trace" / "trace-raw.npz")
    assert raw["cache_attention_map"].shape == (4, 4, 2, 5, 5)
    assert raw["different_bridge_atomic_indices"].shape == (4,)
    assert raw["baseline_familiar_2_predictions"].shape == (16, 2)
