"""Contracts that distinguish changing gradients from changing input exposure."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.text_loss_source import loss_sources, weighted_loss, weights_for
from llm_memory_editability.text_pretrain import build_world, training_table


def setup():
    spec = json.loads(Path("configs/text-pretrain-development-v1.json").read_text())["base"]
    world = build_world(spec)
    table, bounds = training_table(world, "P1")
    return table, bounds, loss_sources(table[3], bounds)


def test_sources_cover_exactly_all_valid_targets_and_do_not_use_answers_to_select():
    table, bounds, source = setup()
    assert (source[: bounds[1]] == 0).sum() == 7 * bounds[1]
    assert (source[bounds[1] : bounds[2]] == 1).sum() == 9 * (bounds[2] - bounds[1])
    assert (source[bounds[2] :] == 0).sum() == 14 * (bounds[3] - bounds[2])
    assert (source[bounds[2] :] == 2).sum() == 9 * (bounds[3] - bounds[2])
    np.testing.assert_array_equal(source >= 0, table[3] != -100)
    assert np.all(source[: bounds[1], 7] == -1)
    assert np.all(source[bounds[2] :, 7] == -1)
    assert np.all(source[bounds[2] :, 15] == -1)


def test_factorial_weights_share_inputs_and_keep_neutral_loss():
    _, _, source = setup()
    weights = {a: weights_for(source, a) for a in ("AB", "A", "B", "N")}
    np.testing.assert_array_equal(weights["AB"] + weights["N"], weights["A"] + weights["B"])
    for weight in weights.values():
        assert np.all(weight[source == 2] == 1)
        assert np.all(weight[source == -1] == 0)
    assert np.all(weights["A"][source == 1] == 0)
    assert np.all(weights["B"][source == 0] == 0)


def test_full_loss_matches_original_and_removed_losses_are_not_renormalized():
    table, bounds, source = setup()
    indices = np.r_[np.arange(32), bounds[1] + np.arange(32), bounds[2] + np.arange(64)]
    labels = torch.tensor(table[3][indices])
    torch.manual_seed(91)
    logits = torch.randn(128, 25, 213, dtype=torch.float64, requires_grad=True)
    losses = {
        a: weighted_loss(logits, labels, torch.tensor(weights_for(source, a)[indices]))
        for a in ("AB", "A", "B", "N")
    }
    torch.testing.assert_close(
        losses["AB"],
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()),
        rtol=1e-12,
        atol=1e-12,
    )
    grads = {
        a: torch.autograd.grad(loss, logits, retain_graph=True)[0] for a, loss in losses.items()
    }
    torch.testing.assert_close(
        grads["AB"] + grads["N"], grads["A"] + grads["B"], rtol=1e-12, atol=1e-12
    )
    for a, count in [("AB", 1984), ("A", 1696), ("B", 864), ("N", 576)]:
        assert weights_for(source, a)[indices].sum() == count
