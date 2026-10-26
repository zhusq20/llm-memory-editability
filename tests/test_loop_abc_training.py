"""Forward equivalence, gradient truncation, and generic target weighting."""

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.loop_abc_training import abc_loss, auxiliary_depth
from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT


def build(dropout=0):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    cfg = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
    )
    return (
        LoopNativeHandoffGPT(cfg, 8, handoff="invariant", first_round_input="embedding")
        .double()
        .train()
    )


X = torch.tensor([[2, 4, 7, 8], [3, 2, 9, 1]])
P = torch.tensor([[0, 1, 2], [0, 1, 2]])
Y = torch.tensor([[4, -100, -100], [2, 9, 1]])


def compare_gradients(a, b):
    for name, value in a.named_parameters():
        torch.testing.assert_close(
            value.grad, b.get_parameter(name).grad, rtol=1e-7, atol=1e-9, msg=name
        )


def test_a_matches_original_full_loss_and_gradients():
    a = build()
    b = copy.deepcopy(a)
    actual = abc_loss(a, X, P, Y, "A")[0]
    logits = b.run(X, P, 8, carry=False)[0]
    expected = F.cross_entropy(logits.flatten(0, 1), Y.flatten())
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    compare_gradients(a, b)


def test_b_matches_explicit_detached_continuation():
    a = build()
    b = copy.deepcopy(a)
    actual = abc_loss(a, X, P, Y, "B", audit=True)
    _, carry = b.run(X, P, 4)
    carry = tuple(v.detach() for v in carry[:4]) + (carry[4],)
    logits = b.run(X, P, 4, carry, carry=False)[0]
    expected = F.cross_entropy(logits.flatten(0, 1), Y.flatten())
    actual[0].backward()
    expected.backward()
    torch.testing.assert_close(actual[0], expected)
    compare_gradients(a, b)
    assert not actual[4]["first_state"].requires_grad
    assert actual[4]["embedding"].grad.norm() > 0


@pytest.mark.parametrize("depth", [1, 4, 7])
def test_c_is_exact_weighted_losses(depth):
    model = build()
    loss, end, auxiliary, _, _ = abc_loss(model, X, P, Y, "C", depth)
    expected_end = F.cross_entropy(model.run(X, P, 8, carry=False)[0].flatten(0, 1), Y.flatten())
    expected_aux = F.cross_entropy(
        model.run(X, P, depth, carry=False)[0].flatten(0, 1), Y.flatten()
    )
    torch.testing.assert_close(end, expected_end)
    torch.testing.assert_close(auxiliary, expected_aux)
    torch.testing.assert_close(loss, 0.9 * expected_end + 0.1 * expected_aux)


def test_all_methods_have_same_forward_and_dropout_stream():
    model = build(0.2)
    rng = torch.get_rng_state()
    reference = None
    after = None
    for arm in ("A", "B", "C"):
        torch.set_rng_state(rng)
        result = abc_loss(model, X, P, Y, arm, 3)
        if reference is None:
            reference = result[3].detach()
            after = torch.get_rng_state()
        torch.testing.assert_close(result[3], reference, rtol=0, atol=0)
        assert torch.equal(after, torch.get_rng_state())


@pytest.mark.parametrize("arm", ["A", "C"])
def test_full_backprop_reaches_first_state(arm):
    model = build()
    result = abc_loss(model, X, P, Y, arm, 3, audit=True)
    result[0].backward()
    assert result[4]["first_state"].grad.norm() > 0


def test_auxiliary_sampler_preserves_torch_rng_and_is_replayable():
    before = torch.get_rng_state()
    values = [auxiliary_depth(19, n) for n in range(1, 151)]
    assert set(values) == set(range(1, 8))
    assert values == [auxiliary_depth(19, n) for n in range(1, 151)]
    assert torch.equal(before, torch.get_rng_state())
