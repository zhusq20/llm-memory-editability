"""Scientific contracts for auxiliary targets and causal representation alignment."""

import numpy as np
import torch

from llm_memory_editability.latent_scaling import build_world, construct, model_digest
from llm_memory_editability.representation_alignment import new_model, objective, pack_training


def small_spec():
    return dict(
        world=771001,
        initialization=772001,
        stream_seed=773001,
        heads_n=32,
        bridges_n=32,
        tails_n=16,
        familiar_n=8,
        strict_n=4,
        anchor_n=4,
        holdout_fraction=0.25,
        low_extra="anchors",
        composition_count=32,
        width=32,
        heads=4,
        layers=1,
        repeats=2,
        dropout=0.0,
    )


def test_exposed_state_preserves_complete_gpt_logits_and_initialization():
    spec = small_spec()
    reference, model = construct(spec, "cpu"), new_model(spec, "cpu")
    assert model_digest(reference) == model_digest(model)
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4]])
    torch.testing.assert_close(reference(tokens), model(tokens), atol=0, rtol=0)
    logits, state = model(tokens, return_bridge=True)
    torch.testing.assert_close(logits, reference(tokens), atol=0, rtol=0)
    assert state.shape == (1, 32)
    assert all(p.requires_grad for p in model.parameters())


def test_early_state_cannot_read_later_relation_or_final_answer():
    model = new_model(small_spec(), "cpu")
    a = torch.tensor([[2, 21, 3, 13, 3, 17, 4, 69]])
    b = torch.tensor([[2, 21, 3, 13, 4, 20, 5, 70]])
    _, sa = model(a, return_bridge=True)
    _, sb = model(b, return_bridge=True)
    torch.testing.assert_close(sa, sb, atol=0, rtol=0)


def test_auxiliary_bridge_is_truth_target_and_never_composed_input():
    world = build_world(small_spec())
    (tokens, labels, targets), sizes = pack_training(world)
    start, stop = sizes[0], sizes[0] + sizes[1]
    np.testing.assert_array_equal(targets[start:stop], world["train_composite"][:, 2])
    assert not np.any(tokens[start:stop] == targets[start:stop, None])
    rows = world["train_composite"].copy()
    world["train_composite"] = rows.copy()
    world["train_composite"][:, 2] += 1
    (other_tokens, other_labels, other_targets), _ = pack_training(world)
    np.testing.assert_array_equal(tokens, other_tokens)
    np.testing.assert_array_equal(labels, other_labels)
    assert np.all(other_targets[start:stop] == targets[start:stop] + 1)


def test_zero_auxiliary_weights_recover_base_loss_and_every_parameter_gradient():
    model = new_model(small_spec(), "cpu")
    reference = construct(small_spec(), "cpu")
    arrays, _ = pack_training(build_world(small_spec()))
    tokens, labels, targets = (torch.as_tensor(a[:8]) for a in arrays)
    loss, _ = objective(model, tokens, labels, targets, 0.0, 0.0)
    logits = reference(tokens)
    base = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=-100
    )
    loss.backward()
    base.backward()
    torch.testing.assert_close(loss, base, atol=0, rtol=0)
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5)
