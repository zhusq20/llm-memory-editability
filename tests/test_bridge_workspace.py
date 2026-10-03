"""Contracts that distinguish semantic intervention evidence from answer leakage."""

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.bridge_workspace import (
    ModelView,
    average_prefix_jacobian,
    control_observation,
    generate,
    load_task,
    match_norm,
    projection,
    query_training_masks,
    select_pairs,
    summarize,
    swap_coordinates,
)
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.grokking_reproduction import ReproductionGPT
from llm_memory_editability.latent_scaling import model_digest


def make_view(large=False):
    torch.set_num_threads(1)
    torch.manual_seed(804101)
    if large:
        config = GPT2Config(
            vocab_size=97,
            n_positions=16,
            n_embd=16,
            n_layer=2,
            n_head=4,
            resid_pdrop=0,
            attn_pdrop=0,
            embd_pdrop=0,
        )
        model = ReproductionGPT(config, repeats=2)
    else:
        model = LoopGPT(
            ModelConfig(vocab_size=97, width=16, layers=2, heads=4, context=16),
            repeats=2,
            dropout=0,
        )
    return ModelView(model.eval().requires_grad_(False))


@pytest.mark.parametrize("large", [False, True])
def test_native_logits_shared_execution_and_parameters_are_preserved(large):
    view = make_view(large)
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4], [2, 22, 3, 14, 3, 18, 4]])
    before = model_digest(view.model)
    torch.testing.assert_close(view.logits(tokens), view.model(tokens)[:, -1], atol=1e-6, rtol=1e-5)
    assert len(view.blocks) == 4
    assert view.blocks[0] is view.blocks[2]
    assert model_digest(view.model) == before
    assert all(p.grad is None and not p.requires_grad for p in view.model.parameters())


@pytest.mark.parametrize("large", [False, True])
def test_prefix_sender_has_no_access_to_future_relation_or_answer(large):
    view = make_view(large)
    tokens = torch.tensor([[21, 13, 17, 80], [21, 13, 19, 91]])
    _, full = view.hidden(tokens, sender=1, capture=True)
    _, prefix = view.hidden(tokens[:, :2], sender=1, capture=True)
    for a, b in zip(full, prefix, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(a[:, 0], a[:, 1], atol=0, rtol=0)


def test_nonorthogonal_coordinate_swap_preserves_complement_and_is_involutive():
    torch.manual_seed(805101)
    state = torch.randn(5, 16)
    basis = torch.randn(5, 16, 2)
    basis[:, :, 1] += basis[:, :, 0] * 0.8
    swapped = swap_coordinates(state, basis)
    inverse = swap_coordinates(swapped, basis)
    pinv = torch.linalg.pinv(basis)
    torch.testing.assert_close(
        (pinv @ swapped[..., None]).squeeze(-1),
        (pinv @ state[..., None]).squeeze(-1).flip(-1),
        atol=2e-6,
        rtol=2e-5,
    )
    complement = torch.eye(16)[None] - basis @ pinv
    torch.testing.assert_close(
        complement @ swapped[..., None], complement @ state[..., None], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(inverse, state, atol=1e-6, rtol=1e-5)


def test_average_jacobian_matches_independent_finite_difference_and_final_identity():
    view = make_view()
    tokens = torch.tensor([[2, 21, 3, 13], [2, 22, 3, 14], [2, 23, 3, 15]])
    jacobian = average_prefix_jacobian(view, tokens, 3, chunk=7)
    torch.testing.assert_close(jacobian[-1], torch.eye(16), atol=1e-6, rtol=1e-6)
    direction = torch.linspace(-0.5, 0.5, 16).repeat(3, 1)
    plus = view.hidden(tokens, layer=0, sender=3, delta=0.001 * direction, math_attention=True)[
        :, 3
    ].mean(0)
    minus = view.hidden(tokens, layer=0, sender=3, delta=-0.001 * direction, math_attention=True)[
        :, 3
    ].mean(0)
    torch.testing.assert_close(
        (plus - minus) / 0.002, jacobian[0] @ direction[0], atol=4e-5, rtol=4e-4
    )
    assert all(p.grad is None for p in view.model.parameters())


def test_jacobian_rejects_trainable_parameters():
    view = make_view()
    view.model.requires_grad_(True)
    with pytest.raises(ValueError, match="Freeze"):
        average_prefix_jacobian(view, torch.tensor([[2, 21, 3, 13]]), 3)


def test_graph_selection_retains_all_shared_relations_and_true_counterfactuals():
    atoms = np.array(
        [
            [1, 10, 3],
            [2, 10, 4],
            [3, 11, 5],
            [3, 12, 6],
            [3, 13, 7],
            [4, 11, 6],
            [4, 12, 7],
            [4, 13, 7],
        ]
    )
    selected = select_pairs(atoms, {"heldout": np.array([0, 1])}, 7, 2)
    groups, queries = selected["groups"], selected["queries"]
    lookup = {(h, r): t for h, r, t in atoms}
    assert len(groups) == 2 and len(queries) == 4
    for group, r2, tail, ct, second, cf_second in queries:
        assert r2 in (11, 12)
        assert lookup[groups[group, 2], r2] == tail
        assert lookup[groups[group, 4], r2] == ct
        assert atoms[second, 2] == tail and atoms[cf_second, 2] == ct
    again = select_pairs(atoms, {"heldout": np.array([1, 0])}, 7, 2)
    np.testing.assert_array_equal(again["groups"], groups)
    np.testing.assert_array_equal(again["queries"], queries)


def test_local_component_deletion_and_restoration_preserve_other_prefix_positions():
    view = make_view()
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4]])
    original, trace = view.hidden(tokens, sender=3, capture=True)
    direction = torch.linspace(-1, 1, 16)[None]
    removed = projection(trace[2][0], direction)
    deleted = view.hidden(tokens, layer=0, sender=3, delta=-removed)
    restored = view.hidden(tokens, layer=0, sender=3, delta=-removed + removed)
    torch.testing.assert_close(deleted[:, :3], original[:, :3], atol=0, rtol=0)
    torch.testing.assert_close(restored, original, atol=0, rtol=0)
    assert not torch.allclose(deleted[:, 3], original[:, 3])


def test_random_controls_match_per_example_norm_only():
    torch.manual_seed(7)
    reference, random = torch.randn(5, 16), torch.randn(5, 16)
    matched = match_norm(random, reference)
    torch.testing.assert_close(matched.norm(dim=-1), reference.norm(dim=-1))
    assert not torch.allclose(matched, reference)


def test_generator_feeds_back_answer_and_separates_answer_from_termination():
    class ToyView:
        device = torch.device("cpu")
        large = False

        def logits(self, tokens, **_):
            logits = torch.full((len(tokens), 32), -20.0)
            answer = 21 if tokens.shape[1] == 5 else 5 if tokens.shape[1] == 6 else 1
            if tokens.shape[1] > 5:
                assert torch.all(tokens[:, 5] == 21)
            logits[:, answer] = 20
            return logits

    result = generate(
        ToyView(),
        np.array([[2, 22, 3, 13, 4]]),
        np.array([21]),
        np.array([23]),
        {"entity_offset": 0},
        8,
    )
    assert result["generated"].tolist() == [[21, 5, 1]]
    assert result["original_logp"][0] > result["cf_logp"][0]
    metrics = summarize(
        result, np.array([21]), np.array([23]), {"entity_offset": 0}, np.array(["atomic"])
    )
    assert metrics["all"]["accuracy"] == 1 and metrics["all"]["cf_accuracy"] == 0
    result["generated"][0, 2] = 0
    metrics = summarize(
        result, np.array([21]), np.array([23]), {"entity_offset": 0}, np.array(["atomic"])
    )
    assert metrics["all"]["answer_accuracy"] == 1 and metrics["all"]["accuracy"] == 0


def test_group_denominators_do_not_count_relations_as_independent_facts():
    result = {
        "generated": np.array([[23, 5, 1], [23, 5, 1], [21, 5, 1]]),
        "original_logp": np.zeros(3),
        "cf_logp": np.zeros(3),
    }
    metrics = summarize(
        result,
        np.full(3, 21),
        np.full(3, 23),
        {"entity_offset": 0},
        np.full(3, "heldout"),
        np.array([0, 0, 1]),
    )["all"]
    assert metrics["n_queries"] == 3 and metrics["n_first_facts"] == 2
    assert metrics["cf_answer_accuracy"] == pytest.approx(2 / 3)
    assert metrics["mean_first_fact_cf_accuracy"] == 0.5
    assert metrics["all_relations_cf_accuracy"] == 0.5


def test_training_masks_keep_native_and_counterfactual_exposure_separate():
    groups = np.array([[1, 10, 3, 2, 4, 0, 1]])
    queries = np.array([[0, 11, 5, 6, 2, 3], [0, 12, 7, 8, 4, 5], [0, 13, 8, 9, 6, 7]])
    world = {"train_composite": np.array([[1, 10, 3, 11, 5], [2, 10, 4, 12, 8]])}
    masks = query_training_masks(world, {}, groups, queries, False)
    assert masks["recipient_query_trained"].tolist() == [True, False, False]
    assert masks["counterfactual_query_trained"].tolist() == [False, True, False]
    assert masks["both_queries_untrained"].tolist() == [False, False, True]


def test_final_sender_control_applies_to_composition_but_atomic_r1_is_readout():
    view = make_view(large=True)
    delta = torch.linspace(-1, 1, view.width)[None]
    composition = torch.tensor([[21, 13, 17]])
    atomic = composition[:, :2]
    patch = {"layer": len(view.blocks) - 1, "sender": 1, "delta": delta}
    torch.testing.assert_close(
        view.logits(composition, **patch), view.logits(composition), atol=0, rtol=0
    )
    assert not torch.allclose(view.logits(atomic, **patch), view.logits(atomic))


def test_alignment_checkpoint_reload_uses_saved_spec_steps(tmp_path):
    from llm_memory_editability.latent_scaling import build_world
    from llm_memory_editability.representation_alignment import new_model

    spec = dict(
        world=811101,
        initialization=812101,
        stream_seed=813101,
        heads_n=32,
        bridges_n=32,
        tails_n=16,
        familiar_n=8,
        strict_n=4,
        anchor_n=4,
        holdout_fraction=0.25,
        low_extra="anchors",
        composition_count=16,
        width=16,
        heads=4,
        layers=1,
        repeats=2,
        dropout=0,
        steps=16000,
    )
    model = new_model(spec, "cpu")
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "spec": spec}, checkpoint)
    np.savez_compressed(tmp_path / "world.npz", **build_world(spec))
    task = {"family": "alignment", "checkpoint": str(checkpoint), "checkpoint_step": 16000}
    loaded, payload, *_ = load_task(task, "cpu")
    assert "step" not in payload
    assert model_digest(loaded.model) == model_digest(model)
    task["checkpoint_step"] = 8000
    with pytest.raises(AssertionError, match="Wrong checkpoint"):
        load_task(task, "cpu")


def test_numerically_sensitive_controls_are_retained_with_separate_denominators():
    original = np.array([[21, 5, 1], [22, 5, 1], [23, 5, 1]])
    changed = original.copy()
    changed[0, 0] = 24
    changed[1, 2] = 0
    result = control_observation(changed, original)
    assert not result["exact"]
    assert result["generated_rows_changed"] == 2 and result["answer_rows_changed"] == 1
    assert result["changed_query_indices"] == [0, 1]
    assert result["changed_fraction"] == pytest.approx(2 / 3)
    assert control_observation(original, original)["exact"]
