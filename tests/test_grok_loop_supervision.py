"""Scientific contracts for distributed answer supervision, not new task labels."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_loop_supervision import SupervisionGPT, answer_losses, evaluate
from llm_memory_editability.grok_multihop import evaluate_rows


def model(dropout=0):
    torch.manual_seed(914)
    return SupervisionGPT(
        ModelConfig(vocab_size=19, width=16, layers=2, heads=2, context=8),
        repeats=2,
        dropout=dropout,
    ).double()


@pytest.mark.parametrize("dropout", [0, 0.1])
def test_branch_heads_preserve_native_trajectory_rng_and_causality(dropout):
    m = model(dropout)
    x = torch.tensor([[2, 12, 13, 7], [3, 14, 15, 8]])
    pos = torch.tensor([[2, 3], [2, 3]])
    rng = torch.get_rng_state()
    outputs = m.at_loops(x, pos)
    after = torch.get_rng_state()
    for i, r in enumerate((2, 3, 4)):
        torch.set_rng_state(rng)
        torch.testing.assert_close(outputs[i], m(x, pos, repeats=r), rtol=0, atol=0)
    assert torch.equal(after, torch.get_rng_state())
    m.eval()
    base = m.at_loops(x, pos)
    x[:, 3] = 17
    altered = m.at_loops(x, pos)
    for a, b in zip(base, altered, strict=True):
        torch.testing.assert_close(a[:, 0], b[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize("weights", [(0, 0, 1), (1 / 3, 1 / 3, 1 / 3)])
def test_all_loop_gradients_equal_weighted_native_gradients(weights):
    m = model()
    native = copy.deepcopy(m)
    x = torch.tensor([[2, 12, 13, 7], [3, 14, 15, 8]])
    pos, labels = torch.tensor([[2, 3], [2, 3]]), torch.tensor([[7, 1], [8, 1]])
    loss = (answer_losses(m, x, pos, labels) * torch.tensor(weights)).sum()
    loss.backward()
    expected = sum(
        w * F.cross_entropy(native(x, pos, repeats=r).flatten(0, 1), labels.flatten())
        for w, r in zip(weights, (2, 3, 4), strict=True)
    )
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=1e-7, atol=1e-12)
    for a, b in zip(m.parameters(), native.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-6, atol=1e-9)
    assert m.token.weight.grad.abs().sum() > 0
    assert m.blocks[0].mlp.up.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("r", [2, 4, 8])
def test_scoring_matches_historical_generated_eos(r):
    m = model()
    m.repeats = r
    rows = np.array([[2, 12, 13, 7], [3, 14, 15, 8]])
    actual, pred = evaluate(m, rows, "cpu", r)
    expected, old = evaluate_rows(m, rows, "cpu", 4)
    for key in old:
        np.testing.assert_array_equal(pred[key], old[key])
    assert actual["accuracy"] == expected["accuracy"]
    assert actual["nll"] == expected["nll"]
