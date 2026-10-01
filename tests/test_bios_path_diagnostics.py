"""Behavioral attribution and no-oracle-bridge contracts for P0 diagnostics."""

from unittest.mock import patch

import numpy as np
import pytest
from test_bios_cross import cross_world as source_cross_world

from llm_memory_editability.bios_path_diagnostics import (
    INDISTINGUISHABLE,
    OTHER,
    ROLE_CODE_START,
    TERMINATION_ERROR,
    annotate_predictions,
    autonomous_two_step,
    candidate_answers,
    classify_answers,
    editing_subsets,
    free_generate_values,
    score_saved_arrays,
)


@pytest.fixture(scope="module")
def cross_world():
    return source_cross_world.__wrapped__()


def test_overlapping_answers_remain_ambiguous_and_eos_is_required():
    candidates = np.array([[11, 11, 12], [11, 12, 13], [11, 12, 13], [11, 12, 13]])
    category, bits = classify_answers([11, 12, 14, 11], [True, True, True, False], candidates)
    np.testing.assert_array_equal(
        category, [INDISTINGUISHABLE, ROLE_CODE_START + 1, OTHER, TERMINATION_ERROR]
    )
    np.testing.assert_array_equal(bits, [3, 2, 0, 1])


def test_saved_score_rejects_value_only_success():
    with pytest.raises(ValueError, match="EOS"):
        score_saved_arrays(
            {"prediction": np.array([8]), "ended": np.array([False]), "correct": np.array([True])},
            np.array([8]),
        )


def test_edit_candidates_preserve_old_and_new_roles(cross_world):
    from llm_memory_editability.bios_cross import edit_pair

    world = cross_world
    pair = edit_pair(world, 0)
    target = pair["exception"]
    roles, candidates = candidate_answers(world, 0, target)
    query = pair["conflict_D"][0]
    person = world.person[query]
    assert candidates[person, roles.index("new_default")] == target[query]
    assert candidates[person, roles.index("new_actual")] != target[query]
    arrays = {
        "prediction": target,
        "ended": np.ones(len(target), dtype=bool),
        "correct": np.ones(len(target), dtype=bool),
    }
    annotated = annotate_predictions(world, arrays, target)
    assert annotated["correct"].all()
    assert (annotated["class_code"][: world.n_base] == -1).all()
    assert annotated["candidate_tokens"].shape == (2, 2048, 6)


def test_cohort_is_not_mislabeled_as_factual_conflict_in_coherent_control(cross_world):
    from llm_memory_editability.bios_cross import edit_pair

    pair = edit_pair(cross_world, 0)
    subsets = dict(editing_subsets(cross_world, pair, 0, pair["coherent"]))
    assert len(subsets["D/all/manipulated_conflict_cohort"]) == 45
    assert not np.intersect1d(subsets["D/all/factual_conflict"], pair["conflict_D"]).size


def test_two_step_uses_generated_wrong_bridge_and_rejects_invalid_or_unterminated(cross_world):
    world = cross_world
    allowed = world.prompts[world.root_ids[0], 1]
    first = np.full(2048, allowed[0])
    first[1] = 0  # Invalid organization token.
    ended = np.ones(2048, dtype=bool)
    ended[2] = False
    calls = []

    def generate(model, prompts, lengths, device, batch_size):
        calls.append(prompts.copy())
        if len(calls) == 1:
            return {"prediction": first, "ended": ended}
        assert np.all(prompts[:, 1] == allowed[0])
        return {
            "prediction": np.full(len(prompts), world.answers[world.root_ids[0, 0]]),
            "ended": np.ones(len(prompts), dtype=bool),
        }

    with patch("llm_memory_editability.bios_path_diagnostics.free_generate_values", generate):
        result = autonomous_two_step(None, world, 0, "cpu")
    assert len(calls[1]) == 2046
    assert not result["correct"][[1, 2]].any()
    assert (result["prediction"][[1, 2]] == -1).all()
    assert not result["bridge_correct"].all()
    # Valid wrong bridges can coincidentally produce the correct default; do not
    # replace first-hop outputs with ground truth or condition scoring on hop 1.
    expected = result["bridge_valid"] & (
        result["prediction"] == world.answers[world.derived_ids[0]]
    )
    np.testing.assert_array_equal(result["correct"], expected)


def test_two_step_model_inputs_are_independent_of_answer_truth(cross_world):
    world = cross_world
    allowed = world.prompts[world.root_ids[1], 1]
    captured = []

    def generate(model, prompts, lengths, device, batch_size):
        captured.append(prompts.copy())
        return {
            "prediction": np.full(len(prompts), allowed[3]),
            "ended": np.ones(len(prompts), dtype=bool),
        }

    with patch("llm_memory_editability.bios_path_diagnostics.free_generate_values", generate):
        autonomous_two_step(None, world, 1, "cpu")
        autonomous_two_step(None, world, 1, "cpu", answers=np.zeros_like(world.answers))
    np.testing.assert_array_equal(captured[0], captured[2])
    np.testing.assert_array_equal(captured[1], captured[3])


def test_free_generation_matches_frozen_interface(cross_world):
    import torch

    from llm_memory_editability.bios_cross_train import evaluate, tensor_queries
    from llm_memory_editability.bios_model import CausalLM, ModelConfig

    torch.manual_seed(77)
    world = cross_world
    model = CausalLM(ModelConfig(world.vocab_size, width=16, layers=1, heads=1))
    ids = np.array([0, world.n_base, world.n_base + 1])
    data = tensor_queries(world, torch.device("cpu"))
    subset = {key: value[ids] for key, value in data.items()}
    expected = evaluate(model, subset, world.answers[ids], batch_size=2)
    actual = free_generate_values(
        model, world.prompts[ids], world.lengths[ids], torch.device("cpu"), batch_size=2
    )
    for key in ("prediction", "ended"):
        np.testing.assert_array_equal(actual[key], expected[key])
