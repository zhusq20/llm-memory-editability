"""Contracts for revised labels, diagonal cached attention and soft channels."""

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import ReproductionGPT
from llm_memory_editability.supervised_alignment import (
    SupervisedAlignmentGPT,
    supervised_alignment_loss,
)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return GPT2Config(
        vocab_size=23,
        n_positions=8,
        n_embd=16,
        n_head=2,
        n_layer=4,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        use_cache=False,
    )


@pytest.fixture
def packed():
    return (
        torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8], [0, 0, 0, 0]]),
        torch.tensor([[1, 2], [2, 3], [0, 0]]),
        torch.tensor([[4, 1], [8, 1], [-100, -100]]),
    )


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("selected_logits", [False, True])
@pytest.mark.parametrize("alpha_zero", [False, True])
def test_disabled_adaptations_match_original_weights_logits_rng_and_gradients(
    config, packed, training, selected_logits, alpha_zero
):
    torch.manual_seed(123)
    original = ReproductionGPT(config, repeats=2).double().train(training)
    torch.manual_seed(123)
    model = (
        SupervisedAlignmentGPT(config, discoloop=alpha_zero, discoloop_alpha=0 if alpha_zero else 1)
        .double()
        .train(training)
    )
    assert original.state_dict().keys() == model.state_dict().keys()
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in original.state_dict().items()
    )
    tokens, positions, _ = packed
    positions = positions if selected_logits else None
    rng = torch.get_rng_state()
    expected = original(tokens, positions)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual, first, final = model.forward_with_states(tokens, positions)
    assert first.shape == final.shape == (*tokens.shape, config.n_embd)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_rng, torch.get_rng_state())
    coefficients = torch.linspace(-1, 1, expected.numel()).view_as(expected)
    (expected * coefficients).sum().backward()
    (actual * coefficients).sum().backward()
    for left, right in zip(original.parameters(), model.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)


@pytest.mark.parametrize("first_round_kv,discoloop", [(True, False), (False, True), (True, True)])
def test_adaptations_add_no_trainable_parameters_or_state(config, first_round_kv, discoloop):
    torch.manual_seed(22)
    original = ReproductionGPT(config, repeats=2)
    torch.manual_seed(22)
    model = SupervisedAlignmentGPT(config, first_round_kv=first_round_kv, discoloop=discoloop)
    assert original.state_dict().keys() == model.state_dict().keys()
    assert sum(p.numel() for p in original.parameters()) == sum(
        p.numel() for p in model.parameters()
    )
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in original.state_dict().items()
    )


@pytest.mark.parametrize("kind", ["alignment", "label_ce"])
def test_only_original_supervision_is_selected_and_target_identity_matters(config, packed, kind):
    torch.manual_seed(10)
    model = SupervisedAlignmentGPT(config).double().eval()
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    final = torch.randn_like(first, requires_grad=True)
    loss, info = supervised_alignment_loss(model, first, final, positions, labels, kind=kind)
    changed = torch.where(labels == -100, labels, (labels + 5) % config.vocab_size)
    changed_loss, changed_info = supervised_alignment_loss(
        model, first, final, positions, changed, kind=kind
    )
    assert not torch.equal(loss, changed_loss)
    assert torch.equal(info["targets"], labels)
    assert torch.equal(changed_info["targets"], changed)
    assert info["valid_positions"] == info["selected_positions"] == 4
    assert torch.equal(info["valid_mask"], labels != -100)
    assert all(not value.requires_grad for value in info.values())
    loss.backward()
    token_mask = torch.tensor([[False, True, True, False], [False, False, True, True], [False] * 4])
    assert torch.count_nonzero(first.grad[~token_mask]) == 0
    assert torch.all(first.grad[token_mask].norm(dim=-1) > 0)
    assert final.grad is None
    altered_first = first.detach().clone()
    altered_first[~token_mask] *= -100
    unaffected, _ = supervised_alignment_loss(
        model, altered_first, -100 * final.detach(), positions, labels, kind=kind
    )
    assert torch.equal(loss, unaffected)


def test_alignment_matches_one_minus_cosine_and_analytic_stop_gradient(config, packed):
    torch.manual_seed(45)
    model = SupervisedAlignmentGPT(config).double().eval()
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    final = torch.randn_like(first, requires_grad=True)
    loss, info = supervised_alignment_loss(model, first, final, positions, labels)
    expected_loss = first.new_zeros(())
    expected_gradient = torch.zeros_like(first)
    for batch, slot in (labels != -100).nonzero().tolist():
        index, target = positions[batch, slot], labels[batch, slot]
        h = first.detach()[batch, index]
        e = model.transformer.wte.weight.detach()[target]
        u, v = F.normalize(h, dim=-1), F.normalize(e, dim=-1)
        logits = F.linear(model.transformer.ln_f(h), model.transformer.wte.weight)
        confidence = logits.softmax(-1)[target].detach()
        cosine = u @ v
        expected_loss += confidence * (1 - cosine) / 4
        expected_gradient[batch, index] = -confidence * (v - u * cosine) / h.norm() / 4
    torch.testing.assert_close(loss, expected_loss, rtol=1e-12, atol=1e-14)
    loss.backward()
    torch.testing.assert_close(first.grad, expected_gradient, rtol=1e-12, atol=1e-14)
    assert all(p.grad is None for p in model.parameters())
    assert final.grad is None
    assert info["confidence"][labels == -100].sum() == 0


def test_label_ce_uses_label_probability_and_one_native_supervision_readout(config, packed):
    torch.manual_seed(90)
    model = SupervisedAlignmentGPT(config).double().eval()
    reference = copy.deepcopy(model)
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    first_reference = first.detach().clone().requires_grad_()
    seen = []
    hook = model.transformer.ln_f.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].shape)
    )
    loss, info = supervised_alignment_loss(model, first, first, positions, labels, kind="label_ce")
    hook.remove()
    assert seen == [(3, 2, config.n_embd)]
    row, slot = (labels != -100).nonzero(as_tuple=True)
    h = first_reference[row, positions[row, slot]]
    logits = F.linear(reference.transformer.ln_f(h), reference.transformer.wte.weight)
    target = labels[row, slot]
    confidence = logits.detach().softmax(-1).gather(-1, target[:, None]).squeeze(-1)
    expected = (confidence * F.cross_entropy(logits, target, reduction="none")).mean()
    torch.testing.assert_close(info["confidence"][row, slot], confidence)
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-14)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(first.grad, first_reference.grad, rtol=1e-12, atol=1e-14)
    for name in ("transformer.ln_f.weight", "transformer.ln_f.bias", "transformer.wte.weight"):
        actual = dict(model.named_parameters())[name].grad
        expected_grad = dict(reference.named_parameters())[name].grad
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected_grad, rtol=1e-12, atol=1e-14)


def test_probability_is_for_original_label_even_when_self_argmax_is_wrong(config):
    model = SupervisedAlignmentGPT(config).double().eval()
    first = torch.randn(1, 3, config.n_embd, dtype=torch.double, requires_grad=True)
    positions = torch.tensor([[1]])
    logits = F.linear(model.transformer.ln_f(first[:, 1]), model.transformer.wte.weight)
    labels = ((logits.argmax(-1) + 1) % config.vocab_size)[:, None]
    _, info = supervised_alignment_loss(model, first, first, positions, labels)
    assert info["targets"].item() != logits.argmax(-1).item()
    torch.testing.assert_close(info["confidence"], logits.detach().softmax(-1).gather(-1, labels))


@pytest.mark.parametrize("kind", ["alignment", "label_ce"])
def test_ignored_slots_can_have_out_of_range_metadata_and_empty_loss_is_differentiable(
    config, kind
):
    model = SupervisedAlignmentGPT(config).double().eval()
    first = torch.randn(2, 4, config.n_embd, dtype=torch.double, requires_grad=True)
    labels = torch.full((2, 2), -100)
    positions = torch.tensor([[-99, 100], [500, -1]])
    loss, info = supervised_alignment_loss(model, first, first, positions, labels, kind=kind)
    assert loss == 0 and torch.isfinite(loss)
    assert info["valid_positions"] == info["selected_positions"] == info["confidence_sum"] == 0
    loss.backward()
    assert torch.count_nonzero(first.grad) == 0
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_small_probability_is_not_renormalized_to_one(config):
    model = SupervisedAlignmentGPT(config).double().eval()
    with torch.no_grad():
        model.transformer.wte.weight.zero_()
    state = torch.ones(1, 4, config.n_embd, dtype=torch.double, requires_grad=True)
    loss, info = supervised_alignment_loss(
        model, state, state, torch.tensor([[1, 2]]), torch.tensor([[4, 1]])
    )
    assert info["valid_positions"] == 2
    torch.testing.assert_close(loss, torch.tensor(1 / config.vocab_size, dtype=torch.double))


@pytest.mark.parametrize("repeats", [2, 3])
def test_diagonal_cache_matches_independent_joint_softmax_and_parameter_gradients(
    config, packed, repeats
):
    torch.manual_seed(11)
    model = SupervisedAlignmentGPT(config, repeats=repeats, first_round_kv=True).double().eval()
    manual = copy.deepcopy(model)
    tokens, _, _ = packed
    logits, first, final = model.forward_with_states(tokens)
    batch, length = tokens.shape
    x = manual.transformer.wte(tokens) + manual.transformer.wpe(torch.arange(length))
    causal = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    diagonal = torch.eye(length, dtype=torch.bool)
    cache = []
    for round_index in range(repeats):
        for layer_index, block in enumerate(manual.transformer.h):
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(config.n_embd, dim=2)
            shape = (batch, length, config.n_head, config.n_embd // config.n_head)
            q, k, v = (item.view(shape).transpose(1, 2) for item in (q, k, v))
            mask = causal
            if round_index == 0:
                cache.append((k, v))
            else:
                k = torch.cat((cache[layer_index][0], k), dim=2)
                v = torch.cat((cache[layer_index][1], v), dim=2)
                mask = torch.cat((diagonal, causal), dim=1)
            weights = (
                (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5)
                .masked_fill(~mask, -torch.inf)
                .softmax(-1)
            )
            attended = weights @ v
            x = x + block.attn.c_proj(
                attended.transpose(1, 2).reshape(batch, length, config.n_embd)
            )
            x = x + block.mlp(block.ln_2(x))
        if round_index == 0:
            torch.testing.assert_close(first, x, rtol=1e-12, atol=1e-14)
    expected = F.linear(manual.transformer.ln_f(x), manual.transformer.wte.weight)
    torch.testing.assert_close(final, x, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(logits, expected, rtol=1e-12, atol=1e-14)
    coefficients = torch.linspace(-1, 1, logits.numel()).view_as(logits)
    (logits * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    for left, right in zip(model.parameters(), manual.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-10, atol=1e-12)


def test_cached_bank_has_only_same_position_gradient_current_bank_keeps_full_prefix(config):
    config.n_layer = 1
    torch.manual_seed(13)
    model = SupervisedAlignmentGPT(config, first_round_kv=True).double().eval()
    shape = (1, config.n_head, 4, config.n_embd // config.n_head)
    cached_k, cached_v = (
        torch.randn(shape, dtype=torch.double, requires_grad=True) for _ in range(2)
    )
    x = torch.randn(1, 4, config.n_embd, dtype=torch.double, requires_grad=True)
    output = model._round(x, first_round_cache=[(cached_k, cached_v)])
    coefficients = torch.linspace(-1, 1, config.n_embd, dtype=torch.double)
    (output[:, 2] * coefficients).sum().backward()
    for tensor in (cached_k, cached_v):
        assert torch.count_nonzero(tensor.grad[:, :, [0, 1, 3]]) == 0
        assert torch.count_nonzero(tensor.grad[:, :, 2]) > 0
    assert torch.all(x.grad[:, :3].norm(dim=-1) > 0)
    assert torch.count_nonzero(x.grad[:, 3]) == 0
    changed = [(cached_k.detach().clone(), cached_v.detach().clone())]
    for tensor in changed[0]:
        tensor[:, :, [0, 1, 3]] *= -100
    altered = model._round(x.detach(), first_round_cache=changed)
    assert torch.equal(output[:, 2], altered[:, 2])


@pytest.mark.parametrize("first_round_kv", [False, True])
@pytest.mark.parametrize("discoloop", [False, True])
def test_no_future_token_or_cross_forward_cache_leakage(config, first_round_kv, discoloop):
    torch.manual_seed(15)
    model = (
        SupervisedAlignmentGPT(config, first_round_kv=first_round_kv, discoloop=discoloop)
        .double()
        .eval()
    )
    tokens = torch.tensor([[2, 3, 4, 5]])
    before = copy.deepcopy(model.state_dict())
    original = model(tokens)
    changed = model(torch.tensor([[2, 3, 17, 18]]))
    assert torch.equal(original[:, :2], changed[:, :2])
    prefix = model(tokens[:, :2])
    torch.testing.assert_close(original[:, :2], prefix, rtol=1e-11, atol=1e-13)
    model(torch.tensor([[9, 10, 11]]))
    assert torch.equal(model(tokens), original)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_cached_first_round_kv_stays_gradient_connected(config):
    model = SupervisedAlignmentGPT(config, first_round_kv=True).double().eval()
    cache = []
    first = model._round(model._embedded(torch.tensor([[2, 3, 4, 5]])), capture_cache=cache)
    for k, v in cache:
        k.retain_grad()
        v.retain_grad()
    # Sever the ordinary residual path so any first-round gradient is cache-only.
    final = model._round(first.detach(), first_round_cache=cache)
    final.square().sum().backward()
    for k, v in cache:
        assert k.grad is not None and torch.count_nonzero(k.grad) > 0
        assert v.grad is not None and torch.count_nonzero(v.grad) > 0


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_discoloop_uses_native_decoder_normalized_addition_only_between_rounds(
    config, packed, repeats
):
    torch.manual_seed(18)
    model = SupervisedAlignmentGPT(config, repeats=repeats, discoloop=True).double().eval()
    with torch.no_grad():
        model.transformer.ln_f.weight.copy_(torch.linspace(0.3, 2.0, config.n_embd))
        model.transformer.ln_f.bias.copy_(torch.linspace(-0.5, 0.5, config.n_embd))
    tokens, _, _ = packed
    calls = []
    hook = model.transformer.ln_f.register_forward_pre_hook(
        lambda _, args: calls.append(args[0].shape)
    )
    logits, first, final = model.forward_with_states(tokens)
    hook.remove()
    assert len(calls) == repeats  # one per boundary and one final readout
    x = model._embedded(tokens)
    for index in range(repeats):
        x = model._round(x)
        if index == 0:
            assert torch.equal(first, x)
        if index < repeats - 1:
            p = F.linear(model.transformer.ln_f(x), model.transformer.wte.weight).softmax(-1)
            soft_embedding = p @ model.transformer.wte.weight
            normalized = (
                soft_embedding / (soft_embedding.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            )
            x = x + normalized
    torch.testing.assert_close(final, x, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(
        logits,
        F.linear(model.transformer.ln_f(x), model.transformer.wte.weight),
        rtol=1e-12,
        atol=1e-14,
    )


def test_soft_channel_has_gradients_through_probabilities_embedding_and_ln_f(config):
    config.n_embd, config.n_head, config.n_layer, config.vocab_size = 4, 1, 1, 5
    torch.manual_seed(82)
    model = (
        SupervisedAlignmentGPT(config, discoloop=True, discoloop_temperature=0.7).double().eval()
    )
    state = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(model._discoloop_boundary, (state,), eps=1e-6, atol=1e-5)
    channel = model._discoloop_boundary(state) - state
    coefficients = torch.linspace(-1, 1, channel.numel()).reshape_as(channel)
    (channel * coefficients).sum().backward()
    assert torch.count_nonzero(state.grad) > 0
    assert torch.count_nonzero(model.transformer.wte.weight.grad) > 0
    assert torch.count_nonzero(model.transformer.ln_f.weight.grad) > 0
    assert torch.count_nonzero(model.transformer.ln_f.bias.grad) > 0


@pytest.mark.parametrize(
    "kind,scope,eps",
    [("self_ce", "supervised", 1e-8), ("alignment", "all", 1e-8), ("alignment", "supervised", 0)],
)
def test_invalid_loss_contract_is_rejected(config, packed, kind, scope, eps):
    model = SupervisedAlignmentGPT(config)
    tokens, positions, labels = packed
    _, first, final = model.forward_with_states(tokens)
    with pytest.raises(ValueError):
        supervised_alignment_loss(
            model, first, final, positions, labels, kind=kind, scope=scope, eps=eps
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_round_kv": 1},
        {"discoloop": 1},
        {"discoloop_alpha": float("nan")},
        {"discoloop_temperature": 0},
        {"discoloop_eps": -1},
    ],
)
def test_invalid_architecture_contract_is_rejected(config, kwargs):
    with pytest.raises(ValueError):
        SupervisedAlignmentGPT(config, **kwargs)
