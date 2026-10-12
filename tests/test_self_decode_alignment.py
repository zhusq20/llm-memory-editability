"""Forward fidelity, information flow, auxiliary gradients and causal diagnostics."""

import copy

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import ReproductionGPT
from llm_memory_editability.loop_kv_learning import FirstRoundKVGPT
from llm_memory_editability.self_decode_alignment import (
    SelfDecodeGPT,
    detached_self_targets,
    norm_matched_replacement,
    self_decode_loss,
    valid_token_mask,
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
@pytest.mark.parametrize("repeats", [1, 2])
def test_original_forward_state_dict_rng_and_gradients_exact(
    config, packed, training, selected_logits, repeats
):
    torch.manual_seed(127)
    original = ReproductionGPT(config, repeats).double().train(training)
    torch.manual_seed(127)
    model = SelfDecodeGPT(config, repeats).double().train(training)
    assert original.state_dict().keys() == model.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
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
    # Independent and unequal logit weights exercise more than a sum invariant.
    coefficients = torch.linspace(-1, 1, expected.numel()).view_as(expected)
    (expected * coefficients).sum().backward()
    (actual * coefficients).sum().backward()
    for (_, left), (_, right) in zip(
        original.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.equal(left.grad, right.grad)


def test_first_state_is_after_all_four_complete_blocks_and_before_ln_f(config, packed):
    torch.manual_seed(7)
    model = SelfDecodeGPT(config, repeats=2).double().eval()
    once = ReproductionGPT(config, repeats=1).double().eval()
    once.load_state_dict(model.state_dict(), strict=True)
    tokens, _, _ = packed
    logits, first, final = model.forward_with_states(tokens)
    decoded_first = F.linear(model.transformer.ln_f(first), model.transformer.wte.weight)
    assert torch.equal(decoded_first, once(tokens))
    assert torch.equal(
        logits, F.linear(model.transformer.ln_f(final), model.transformer.wte.weight)
    )
    assert not torch.equal(first, model.transformer.ln_f(first))
    assert not torch.equal(first, final)
    clone = ReproductionGPT(config, repeats=2).double().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    assert torch.equal(logits, clone(tokens))


def test_mask_uses_only_last_nonignored_supervised_index(packed):
    _, positions, labels = packed
    expected = torch.tensor(
        [[True, True, True, False], [True, True, True, True], [False, False, False, False]]
    )
    assert torch.equal(valid_token_mask(positions, labels, 4), expected)
    # Ignored target metadata must not extend a row or resurrect a dummy row.
    positions = torch.tensor([[1, 99], [2, 3], [100, 100]])
    labels = torch.tensor([[7, -100], [-100, 2], [-100, -100]])
    assert valid_token_mask(positions, labels, 4).tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [False, False, False, False],
    ]


@pytest.mark.parametrize("kind", ["alignment", "self_ce"])
@pytest.mark.parametrize("scope", ["all", "consistent"])
def test_labels_supply_no_target_identity_and_dummy_padding_have_zero_gradients(
    config, packed, kind, scope
):
    torch.manual_seed(12)
    model = SelfDecodeGPT(config, repeats=2).double().eval()
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    final = first.detach().clone().requires_grad_()
    loss, info = self_decode_loss(model, first, final, positions, labels, kind=kind, scope=scope)
    changed_labels = torch.where(labels == -100, labels, (labels + 9) % config.vocab_size)
    changed_loss, changed_info = self_decode_loss(
        model, first, final, positions, changed_labels, kind=kind, scope=scope
    )
    assert torch.equal(loss, changed_loss)
    for key in info:
        assert torch.equal(info[key], changed_info[key])
        assert not info[key].requires_grad
    loss.backward()
    assert torch.count_nonzero(first.grad[~info["valid_mask"]]) == 0
    assert torch.count_nonzero(first.grad[info["valid_mask"]]) > 0
    assert final.grad is None
    changed_first = first.detach().clone()
    changed_first[~info["valid_mask"]] = torch.randn_like(changed_first[~info["valid_mask"]])
    changed_final = final.detach().clone()
    changed_final[~info["valid_mask"]] *= -2
    unaffected, _ = self_decode_loss(
        model, changed_first, changed_final, positions, labels, kind=kind, scope=scope
    )
    torch.testing.assert_close(loss, unaffected, rtol=0, atol=0)


def test_alignment_stops_readout_embedding_target_confidence_and_final_gradients(config, packed):
    torch.manual_seed(19)
    model = SelfDecodeGPT(config, repeats=2).double().eval()
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    final = first.detach().clone().requires_grad_()
    loss, info = self_decode_loss(model, first, final, positions, labels)
    direction = F.normalize(first.detach(), dim=-1)
    target = F.normalize(model.transformer.wte.weight.detach()[info["pseudo_targets"]], dim=-1)
    cosine = (direction * target).sum(-1, keepdim=True)
    manual_gradient = -2 * (target - direction * cosine) / first.detach().norm(dim=-1, keepdim=True)
    manual_gradient *= (info["confidence"] * info["valid_mask"])[..., None]
    manual_gradient /= info["valid_positions"]
    loss.backward()
    torch.testing.assert_close(first.grad, manual_gradient, rtol=1e-12, atol=1e-14)
    assert final.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not info["confidence"].requires_grad
    assert not info["pseudo_targets"].requires_grad


def test_self_ce_uses_same_rule_and_differentiates_through_original_ln_and_embedding(
    config, packed
):
    torch.manual_seed(55)
    model = SelfDecodeGPT(config, repeats=2).double().eval()
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    final = first.detach().clone().requires_grad_()
    _, alignment = self_decode_loss(model, first, final, positions, labels)
    loss, info = self_decode_loss(model, first, final, positions, labels, kind="self_ce")
    assert torch.equal(info["pseudo_targets"], alignment["pseudo_targets"])
    assert torch.equal(info["confidence"], alignment["confidence"])
    original_logits = F.linear(model.transformer.ln_f(first), model.transformer.wte.weight)
    expected = F.cross_entropy(
        original_logits.flatten(0, 1), info["pseudo_targets"].flatten(), reduction="none"
    ).view_as(info["confidence"])
    expected = (expected * info["confidence"] * info["valid_mask"]).sum() / 7
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert torch.count_nonzero(first.grad) > 0
    assert torch.count_nonzero(model.transformer.ln_f.weight.grad) > 0
    assert torch.count_nonzero(model.transformer.wte.weight.grad) > 0
    assert final.grad is None


def test_low_confidence_is_not_renormalized(config):
    model = SelfDecodeGPT(config).double().eval()
    with torch.no_grad():
        model.transformer.wte.weight.zero_()
    first = torch.ones(1, 4, config.n_embd, dtype=torch.double, requires_grad=True)
    positions, labels = torch.tensor([[1, 2]]), torch.tensor([[2, 1]])
    loss, info = self_decode_loss(model, first, first, positions, labels)
    assert info["valid_positions"] == 3
    # All logits are zero, so pmax=1/V. A zero target has squared distance one.
    torch.testing.assert_close(loss, torch.tensor(1 / config.vocab_size, dtype=torch.double))
    assert loss < 0.1


@pytest.mark.parametrize("kind", ["alignment", "self_ce"])
def test_consistency_uses_original_same_position_argmax_and_all_valid_denominator(
    config, packed, kind
):
    torch.manual_seed(28)
    model = SelfDecodeGPT(config, repeats=2).double().eval()
    tokens, positions, labels = packed
    first = torch.randn(*tokens.shape, config.n_embd, dtype=torch.double, requires_grad=True)
    final = first.detach().clone()
    final[:, 1] *= -1
    final.requires_grad_()
    _, all_info = self_decode_loss(model, first, final, positions, labels, kind=kind)
    loss, info = self_decode_loss(
        model, first, final, positions, labels, kind=kind, scope="consistent"
    )
    last_targets, _ = detached_self_targets(model, final)
    expected_mask = info["valid_mask"] & (info["pseudo_targets"] == last_targets)
    assert torch.equal(info["selected_mask"], expected_mask)
    assert 0 < info["selected_positions"] < info["valid_positions"]
    expected = (all_info["per_token_loss"] * info["confidence"] * expected_mask).sum() / 7
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert final.grad is None
    assert torch.count_nonzero(first.grad[~expected_mask]) == 0


@pytest.mark.parametrize("kind", ["alignment", "self_ce"])
def test_entire_dummy_batch_is_differentiable_zero(config, packed, kind):
    model = SelfDecodeGPT(config, repeats=2).eval()
    tokens, positions, labels = packed
    _, first, final = model.forward_with_states(tokens)
    loss, info = self_decode_loss(
        model, first, final, positions, torch.full_like(labels, -100), kind=kind
    )
    assert info["valid_positions"] == 0
    assert torch.isfinite(loss) and loss == 0
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


@pytest.mark.parametrize("scope", ["none", "r1", "all", "consistent"])
@pytest.mark.parametrize("first_round_kv", [False, True])
def test_replacement_and_consistency_do_not_access_future_tokens(config, scope, first_round_kv):
    torch.manual_seed(61)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=first_round_kv).double().eval()
    tokens = torch.tensor([[2, 3, 4, 5]])
    changed = torch.tensor([[2, 3, 17, 18]])
    original, original_info = model.forward_with_replacement(tokens, scope=scope, return_info=True)
    altered, altered_info = model.forward_with_replacement(changed, scope=scope, return_info=True)
    torch.testing.assert_close(original[:, :2], altered[:, :2], rtol=0, atol=0)
    for key in ("first_state", "pseudo_targets", "confidence", "selected_mask"):
        assert torch.equal(original_info[key][:, :2], altered_info[key][:, :2])
    prefix, prefix_info = model.forward_with_replacement(
        tokens[:, :2], scope=scope, return_info=True
    )
    torch.testing.assert_close(original[:, :2], prefix, rtol=1e-12, atol=1e-14)
    assert torch.equal(original_info["selected_mask"][:, :2], prefix_info["selected_mask"])
    if scope == "consistent":
        assert torch.equal(
            original_info["original_final_targets"][:, :2],
            altered_info["original_final_targets"][:, :2],
        )


@pytest.mark.parametrize("first_round_kv", [False, True])
def test_replacement_gate_comes_from_unintervened_final_pass(config, packed, first_round_kv):
    torch.manual_seed(81)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=first_round_kv).double().eval()
    tokens, positions, labels = packed
    _, first, final = model.forward_with_states(tokens)
    first_targets, _ = detached_self_targets(model, first)
    final_targets, _ = detached_self_targets(model, final)
    valid = valid_token_mask(positions, labels, tokens.shape[1])
    _, info = model.forward_with_replacement(
        tokens, positions, scope="consistent", valid_mask=valid, return_info=True
    )
    assert torch.equal(info["original_final_targets"], final_targets)
    assert torch.equal(info["selected_mask"], valid & (first_targets == final_targets))
    assert torch.equal(info["first_state"], first)
    assert torch.equal(info["replacement_state"][~valid], first[~valid])


def test_replacement_alpha_and_norm_contract():
    state = torch.tensor([[[3.0, 4.0], [1.0, 2.0]]])
    embedding = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]], requires_grad=True)
    selected = torch.tensor([[True, False]])
    assert norm_matched_replacement(state, embedding, selected, alpha=0) is state
    replaced = norm_matched_replacement(state, embedding, selected, alpha=1)
    assert torch.equal(replaced, torch.tensor([[[0.0, 5.0], [1.0, 2.0]]]))
    torch.testing.assert_close(replaced.norm(dim=-1), state.norm(dim=-1), rtol=0, atol=0)
    mixed = norm_matched_replacement(state, embedding, selected, alpha=0.5)
    assert torch.equal(mixed, (state + replaced) / 2)
    assert mixed[0, 0].norm() < state[0, 0].norm()
    assert not replaced.requires_grad


@pytest.mark.parametrize("scope", ["none", "r1", "all", "consistent"])
def test_disabled_intervention_preserves_original_forward_and_weights(config, packed, scope):
    model = SelfDecodeGPT(config, repeats=2).eval()
    before = copy.deepcopy(model.state_dict())
    tokens, positions, _ = packed
    expected = model(tokens, positions)
    actual = model.forward_with_replacement(tokens, positions, scope=scope, alpha=0)
    assert torch.equal(expected, actual)
    if scope == "none":
        assert torch.equal(expected, model.forward_with_replacement(tokens, positions, scope=scope))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_r1_scope_and_explicit_valid_mask(config, packed):
    model = SelfDecodeGPT(config, repeats=2).eval()
    tokens, positions, labels = packed
    valid = valid_token_mask(positions, labels, 4)
    _, info = model.forward_with_replacement(tokens, scope="r1", valid_mask=valid, return_info=True)
    assert info["selected_mask"].tolist() == [
        [False, True, False, False],
        [False, True, False, False],
        [False, False, False, False],
    ]
    selected = info["selected_mask"]
    torch.testing.assert_close(
        info["replacement_state"][selected].norm(dim=-1),
        info["first_state"][selected].norm(dim=-1),
    )
    assert torch.equal(info["replacement_state"][~selected], info["first_state"][~selected])


def test_invalid_diagnostic_usage_is_rejected(config, packed):
    tokens, _, _ = packed
    model = SelfDecodeGPT(config, repeats=2)
    with pytest.raises(ValueError, match="inference diagnostic"):
        model.forward_with_replacement(tokens)
    model.eval()
    with pytest.raises(ValueError, match="scope"):
        model.forward_with_replacement(tokens, scope="oracle")
    with pytest.raises(ValueError, match="alpha"):
        model.forward_with_replacement(tokens, alpha=float("nan"))
    with pytest.raises(ValueError, match="r1_position"):
        model.forward_with_replacement(tokens[:, :1], scope="r1")
    with pytest.raises(ValueError, match="valid_mask"):
        model.forward_with_replacement(tokens, valid_mask=tokens)
    model.repeats = 1
    with pytest.raises(ValueError, match="downstream round"):
        model.forward_with_replacement(tokens, scope="all")


@pytest.mark.parametrize("training", [False, True])
def test_kv_on_matches_existing_implementation_logits_rng_and_gradients(config, packed, training):
    torch.manual_seed(144)
    original = FirstRoundKVGPT(config, repeats=2).double().train(training)
    torch.manual_seed(144)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=True).double().train(training)
    assert model.state_dict().keys() == original.state_dict().keys()
    for name, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    tokens, positions, _ = packed
    rng = torch.get_rng_state()
    expected = original(tokens, positions)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = model(tokens, positions)
    assert torch.equal(expected_rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coefficients = torch.linspace(-1, 1, expected.numel()).view_as(expected)
    (actual * coefficients).sum().backward()
    (expected * coefficients).sum().backward()
    for left, right in zip(original.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("repeats", [2, 3])
def test_shared_kv_matches_explicit_same_layer_double_causal_softmax(config, packed, repeats):
    torch.manual_seed(101)
    model = SelfDecodeGPT(config, repeats=repeats, first_round_kv=True).double().eval()
    tokens, _, _ = packed
    expected, expected_first, expected_final = model.forward_with_states(tokens)
    # Independent attention formula tests scale, branch order, causal masks and
    # same-layer cache identity; it deliberately does not call the model's round.
    batch, length = tokens.shape
    x = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(length))
    causal = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    cache = []
    for round_index in range(repeats):
        for layer_index, block in enumerate(model.transformer.h):
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(config.n_embd, dim=2)
            shape = (batch, length, config.n_head, config.n_embd // config.n_head)
            q, k, v = (item.view(shape).transpose(1, 2) for item in (q, k, v))
            mask = causal
            if round_index == 0:
                cache.append((k, v))
            else:
                k = torch.cat((cache[layer_index][0], k), dim=2)
                v = torch.cat((cache[layer_index][1], v), dim=2)
                mask = torch.cat((causal, causal), dim=1)
            scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
            probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            attended = probabilities @ v
            y = block.attn.c_proj(attended.transpose(1, 2).reshape(batch, length, config.n_embd))
            x = x + y
            x = x + block.mlp(block.ln_2(x))
        if round_index == 0:
            torch.testing.assert_close(x, expected_first, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(x, expected_final, rtol=1e-12, atol=1e-14)
    manual = F.linear(model.transformer.ln_f(x), model.transformer.wte.weight)
    torch.testing.assert_close(manual, expected, rtol=1e-12, atol=1e-14)


def test_kv_first_round_unchanged_and_disabled_after_use_restores_original(config, packed):
    torch.manual_seed(304)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=True).double().eval()
    original = ReproductionGPT(config, repeats=2).double().eval()
    original.load_state_dict(model.state_dict(), strict=True)
    tokens, positions, _ = packed
    with_kv, kv_first, _ = model.forward_with_states(tokens, positions)
    model.first_round_kv = False
    without_kv, original_first, _ = model.forward_with_states(tokens, positions)
    assert torch.equal(kv_first, original_first)
    assert torch.equal(without_kv, original(tokens, positions))
    assert not torch.equal(with_kv, without_kv)
    without_kv.square().sum().backward()
    original(tokens, positions).square().sum().backward()
    for left, right in zip(original.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_cached_keys_and_values_have_gradients_without_residual_handoff(config, packed):
    torch.manual_seed(422)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=True).double().eval()
    tokens, _, _ = packed
    cache = []
    first = model._round(model._embedded(tokens), capture_cache=cache)
    assert len(cache) == config.n_layer
    for keys, values in cache:
        assert keys.requires_grad and values.requires_grad
        keys.retain_grad()
        values.retain_grad()
    # Remove the residual-state path entirely. Any first-round gradient must
    # now arrive through the extra first-round K/V access.
    final = model._round(first.detach(), first_round_cache=cache)
    logits = F.linear(model.transformer.ln_f(final), model.transformer.wte.weight)
    coefficients = torch.linspace(-1, 1, logits.numel()).view_as(logits)
    (logits * coefficients).sum().backward()
    for keys, values in cache:
        assert torch.count_nonzero(keys.grad) > 0
        assert torch.count_nonzero(values.grad) > 0
    assert torch.count_nonzero(model.transformer.wpe.weight.grad) > 0


def test_kv_cache_resets_between_calls_and_generation_rebuilds_from_current_tokens(config):
    torch.manual_seed(117)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=True).double().eval()
    tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    before = model(tokens)
    model(torch.tensor([[10, 11]]))
    assert torch.equal(before, model(tokens))
    generated = tokens.clone()
    generated[:, 3] = before[:, 2].argmax(-1)
    after = model(generated)
    # Changing a teacher-forced answer can never alter the earlier answer
    # readout, including through the first-round branch of later attention.
    assert torch.equal(before[:, :3], after[:, :3])
    fresh = SelfDecodeGPT(config, repeats=2, first_round_kv=True).double().eval()
    fresh.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(after, fresh(generated))
    assert not any("cache" in name for name in model.__dict__)


@pytest.mark.parametrize("kind", ["alignment", "self_ce"])
def test_kv_combines_with_auxiliary_losses_and_backward(config, packed, kind):
    torch.manual_seed(127)
    model = SelfDecodeGPT(config, repeats=2, first_round_kv=True).double().eval()
    tokens, positions, labels = packed
    logits, first, final = model.forward_with_states(tokens, positions)
    auxiliary, info = self_decode_loss(model, first, final, positions, labels, kind=kind)
    supervised = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    total = supervised + 0.3 * auxiliary
    total.backward()
    assert torch.isfinite(total)
    assert info["valid_positions"] == 7
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
