"""Counterfactual matching, intervention norm, and causal-time controls."""

import numpy as np
import pytest
from test_bios_cross import cross_world as source_cross_world

from llm_memory_editability.bios_causal_paths import (
    cache_clean_states,
    generate_with_state_patch,
    intervention_state,
    make_donor_plan,
)
from llm_memory_editability.bios_path_diagnostics import free_generate_values


@pytest.fixture(scope="module")
def cross_world():
    return source_cross_world.__wrapped__()


@pytest.mark.parametrize("chain", [0, 1])
def test_truth_only_donor_plan_has_disjoint_qa_splits_and_correct_relations(cross_world, chain):
    world = cross_world
    plan = make_donor_plan(world, chain)
    repeated = make_donor_plan(world, chain)
    for key in plan:
        np.testing.assert_array_equal(plan[key], repeated[key])
    assert len(np.unique(plan["recipient"])) == 128
    assert np.isin(world.derived_ids[chain, plan["recipient"]], world.heldout_ids[chain]).all()
    group = world.memberships[chain]
    actual = world.answers[world.actual_ids[chain]]
    default = world.answers[world.derived_ids[chain]]
    for i, donor in enumerate(plan["donor"]):
        if donor == -1:
            assert plan["candidate_count"][i] == 0
            continue
        recipient = plan["recipient"][i]
        assert world.derived_ids[chain, donor] in world.train_ids[chain]
        same_group, same_actual = (
            group[recipient] == group[donor],
            actual[recipient] == actual[donor],
        )
        same_default = default[recipient] == default[donor]
        pair_type = plan["pair_type"][i]
        assert (
            (same_group and not same_actual),
            (not same_group and same_actual and not same_default),
            (same_group and same_actual and recipient != donor),
            (not same_group and not same_actual and not same_default),
        )[pair_type]


def test_random_control_matches_delta_not_absolute_activation_norm():
    recipient = np.random.default_rng(12).normal(size=(5, 16)).astype(np.float32) + 10
    donor = recipient + np.random.default_rng(13).normal(size=(5, 16)).astype(np.float32)
    donor[0] = recipient[0]
    patched, expected, actual = intervention_state(recipient, donor, "matched_random", [1, 2])
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(patched[0], recipient[0])
    sham, _, norm = intervention_state(recipient, donor, "sham", [1, 2])
    np.testing.assert_array_equal(sham, recipient)
    assert not norm.any()


def test_sham_and_final_layer_past_positions_are_causal_negative_controls(cross_world):
    import torch

    from llm_memory_editability.bios_model import CausalLM, ModelConfig

    torch.manual_seed(91)
    world = cross_world
    model = CausalLM(ModelConfig(world.vocab_size, width=16, layers=2, heads=1))
    ids = world.derived_ids[0, :4]
    prompts, lengths = world.prompts[ids], world.lengths[ids]
    device = torch.device("cpu")
    baseline = free_generate_values(model, prompts, lengths, device, 4)
    states = cache_clean_states(model, prompts, lengths, device, 4)
    assert states.shape == (2, 4, 3, 16)
    for layer in (0, 1):
        sham = generate_with_state_patch(
            model, prompts, lengths, states[layer, :, 2], layer, 4, device, 4
        )
        for key in baseline:
            np.testing.assert_array_equal(sham[key], baseline[key])
    for position_index, position in enumerate((2, 3)):
        patched = generate_with_state_patch(
            model, prompts, lengths, states[1, :, position_index] + 100, 1, position, device, 4
        )
        for key in baseline:
            np.testing.assert_array_equal(patched[key], baseline[key])
