"""Contracts that determine the interpretation of the gradient experiment."""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_context_gradient import (
    TARGET,
    allowed_mask,
    attention_mode,
    ln_vjp,
    module_differences,
    statistic_predictors,
    vector_metrics,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    return CausalLM(ModelConfig(31, width=8, layers=2, heads=2, context=24)).double()


def records():
    return torch.tensor(
        [
            [1, 4, 8, 2, 11, 3, 1, 5, 9, 2, 12, 3],
            [1, 6, 8, 2, 13, 3, 1, 7, 9, 2, 14, 3],
        ]
    )


def test_layernorm_vjp_matches_autograd_with_nontrivial_affine(tiny_model):
    layer = tiny_model.ln_final
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(0.2, 1.7, 8))
        layer.bias.fill_(0.4)
    x = torch.randn(3, 4, 8, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn_like(x)
    actual = torch.autograd.grad((layer(x) * upstream).sum(), x)[0]
    torch.testing.assert_close(ln_vjp(x, upstream, layer), actual, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("position_aware", [False, True])
def test_statistic_contraction_equals_independent_zero_branch_derivative(
    tiny_model, isolated, position_aware
):
    """The predictor must equal the specified surrogate, not the full model."""
    model = tiny_model
    tokens = records()
    positions = torch.tensor([3, 4, 9, 10])
    labels = tokens[:, positions + 1]
    predicted = statistic_predictors(model, tokens, positions, labels, labels.flip(0), isolated)
    x = model.token(tokens)
    if position_aware:
        x = x + model.position(torch.arange(tokens.shape[1]))
    width = model.config.width
    value = model.blocks[0].attention.qkv(model.blocks[0].ln1(x))[:, :, 2 * width :]
    # Explicit per-query prefix averaging, independent from the matrix mask implementation.
    context = []
    for p in positions:
        begin = int(p) // 6 * 6 if isolated else 0
        context.append(value[:, begin : p + 1].mean(1))
    context = torch.stack(context, 1)
    branch = torch.zeros(width, width, dtype=torch.float64, requires_grad=True)
    logits = F.linear(
        model.ln_final(x[:, positions] + F.linear(context, branch)), model.token.weight
    )
    for offset, kind in enumerate(("answer", "eos")):
        loss = F.cross_entropy(logits[:, offset::2].flatten(0, 1), labels[:, offset::2].flatten())
        actual = torch.autograd.grad(loss, branch, retain_graph=True)[0]
        name = "position" if position_aware else "token"
        torch.testing.assert_close(predicted[f"{name}/{kind}"], actual, atol=1e-12, rtol=1e-11)


def test_future_and_other_records_are_excluded():
    mask = allowed_mask(12, True)
    assert mask[9, 6:10].all()
    assert not mask[9, :6].any()
    assert not mask[9, 10:].any()
    assert torch.equal(mask.sum(1), torch.tensor([1, 2, 3, 4, 5, 6] * 2))


def test_fact_regrouping_cancels_isolated_full_model_gradient(tiny_model):
    model = tiny_model
    first = records()
    regrouped = first.clone()
    regrouped[:, 6:] = first.flip(0)[:, 6:]
    positions = torch.tensor([3, 4, 9, 10])
    parameters = dict(model.named_parameters())
    gradients = []
    with attention_mode(model, True):
        for tokens in (first, regrouped):
            logits = model(tokens, positions[None].expand(len(tokens), -1))
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, positions + 1].flatten())
            gradients.append(torch.autograd.grad(loss, tuple(parameters.values())))
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    # Restoring the model must actually restore cross-record dependence.
    a = model(first, positions[None].expand(len(first), -1))
    b = model(regrouped, positions[None].expand(len(first), -1))
    assert not torch.allclose(a.sum(0), b.sum(0), atol=1e-10, rtol=1e-10)


def test_mask_scope_restores_original_forward_after_error(tiny_model):
    original = tiny_model.blocks[0].attention.forward
    with pytest.raises(RuntimeError), attention_mode(tiny_model, True):
        raise RuntimeError("synthetic failure")
    assert tiny_model.blocks[0].attention.forward == original


def test_common_qa_cancels_and_answer_eos_weight_is_half():
    first, second, qa = np.array([1.0, 3.0]), np.array([2.0, -1.0]), np.array([7.0, 5.0])
    full_a, full_b = 0.8 * first + 0.2 * qa, 0.8 * second + 0.2 * qa
    np.testing.assert_allclose(full_a - full_b, 0.8 * (first - second))
    assert 0.8 * (first.mean()) == (0.4 * first).sum()


def test_module_accounting_splits_qkv_without_double_counting():
    first = {"blocks.0.attention.qkv.weight": np.ones((6, 2)), TARGET: np.ones((2, 2))}
    second = {key: np.zeros_like(value) for key, value in first.items()}
    groups = module_differences(first, second)
    assert len(groups) == 4
    assert sum(group["difference_norm"] ** 2 for group in groups.values()) == 16


def test_direction_and_scale_errors_are_distinguished():
    metrics = vector_metrics(np.array([1.0, 2.0]), np.array([3.0, 6.0]))
    assert metrics["cosine"] == pytest.approx(1)
    assert metrics["relative_error"] == pytest.approx(2)
    assert metrics["norm_ratio"] == pytest.approx(3)
    assert vector_metrics(np.zeros(2), np.zeros(2))["cosine"] is None


def test_supervised_bigram_has_rank_at_most_two():
    tokens = records()
    positions = torch.tensor([3, 4, 9, 10])
    bigram = np.zeros((31, 31))
    current, targets = tokens[:, positions].numpy(), tokens[:, positions + 1].numpy()
    np.add.at(bigram, (current.ravel(), targets.ravel()), 1)
    bigram -= bigram.sum(1, keepdims=True) / 31
    assert np.linalg.matrix_rank(bigram) == 2
