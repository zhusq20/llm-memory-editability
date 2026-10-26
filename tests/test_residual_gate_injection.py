"""Contracts isolating the cross-round connector from the optional fixed q channel."""

import io
import math

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.residual_gate_injection import ResidualGateInjectionGPT
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

INJECTIONS = ("none", "fixed1")
MODES = ("full", "fixed", "token")
TOKENS = torch.tensor([[2, 4, 7, 8], [3, 2, 9, 1]])


def config(dropout=0.0):
    return GPT2Config(
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


def model(injection, mode, dtype=torch.float64, repeats=3, dropout=0.0):
    torch.manual_seed(71)
    return ResidualGateInjectionGPT(
        config(dropout),
        repeats=repeats,
        injection=injection,
        update_mode=mode,
        initial_update=0.95,
    ).to(dtype)


def reference_model(injection, dtype=torch.float64, repeats=3, dropout=0.0):
    torch.manual_seed(71)
    return SupervisedAlignmentGPT(
        config(dropout), repeats=repeats, discoloop=injection == "fixed1"
    ).to(dtype)


def assert_common_gradients(reference, actual):
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.equal(parameter.grad, actual.get_parameter(name).grad), name


def independent_boundary(current, x, h):
    """Mathematical reference, independent of both production boundary methods."""
    normalized = F.layer_norm(h, (h.shape[-1],), eps=current.config.layer_norm_epsilon)
    u = torch.sigmoid(
        (normalized * current.gate_update_weight).sum(-1, keepdim=True) + current.gate_update_bias
    )
    expected = x + u * (h - x)
    if current.injection == "fixed1":
        probability = (current.transformer.ln_f(h) @ current.transformer.wte.weight.T).softmax(-1)
        v = probability @ current.transformer.wte.weight
        q = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
        expected = expected + q
    return expected


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_full_preserves_reference_state_rng_dropout_outputs_and_all_gradients(injection, dtype):
    current = model(injection, "full", dtype, dropout=0.15)
    actual_rng = torch.get_rng_state().clone()
    reference = reference_model(injection, dtype, dropout=0.15)
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert current.state_dict().keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, current.state_dict()[name]), name
    assert not any("gate_" in name for name in current.state_dict())

    torch.manual_seed(83)
    a = reference.forward_with_states(TOKENS)
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(83)
    b = current.forward_with_states(TOKENS)
    assert torch.equal(reference_rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    a[0].square().mean().backward()
    b[0].square().mean().backward()
    assert_common_gradients(reference, current)


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_fixed_and_token_share_initial_state_rng_forward_and_backbone_gradients(injection, dtype):
    fixed = model(injection, "fixed", dtype)
    actual_rng = torch.get_rng_state().clone()
    learned = model(injection, "token", dtype)
    assert torch.equal(actual_rng, torch.get_rng_state())
    baseline = reference_model(injection, dtype)
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert fixed.state_dict().keys() == learned.state_dict().keys()
    for name, value in fixed.state_dict().items():
        assert torch.equal(value, learned.state_dict()[name]), name
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, learned.state_dict()[name]), name
    assert not any("gate_paper" in name for name in learned.state_dict())
    gate_names = {"gate_update_weight", "gate_update_bias"}
    assert set(fixed.state_dict()) - set(baseline.state_dict()) == gate_names
    assert gate_names <= set(dict(fixed.named_buffers()))
    assert gate_names.isdisjoint(dict(fixed.named_parameters()))
    assert gate_names <= set(dict(learned.named_parameters()))
    assert sum(p.numel() for p in fixed.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )
    assert (
        sum(p.numel() for p in learned.parameters()) - sum(p.numel() for p in fixed.parameters())
        == fixed.config.n_embd + 1
    )
    assert torch.count_nonzero(learned.gate_update_weight) == 0
    assert learned.gate_update_bias.item() == pytest.approx(math.log(0.95 / 0.05), rel=1e-7)

    a, b = fixed(TOKENS), learned(TOKENS)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    assert_common_gradients(fixed, learned)
    for name in gate_names:
        gradient = learned.get_parameter(name).grad
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient), name


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("pair", ("reference_full", "fixed_token"))
def test_cpu_bfloat16_autocast_preserves_outputs_and_common_gradients(injection, pair):
    if pair == "reference_full":
        reference = reference_model(injection, torch.float32)
        current = model(injection, "full", torch.float32)
    else:
        reference = model(injection, "fixed", torch.float32)
        current = model(injection, "token", torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a = reference.forward_with_states(TOKENS)
        b = current.forward_with_states(TOKENS)
        assert a[0].dtype == b[0].dtype == torch.bfloat16
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        loss_a, loss_b = a[0].float().square().mean(), b[0].float().square().mean()
    loss_a.backward()
    loss_b.backward()
    assert_common_gradients(reference, current)
    if pair == "fixed_token":
        for name in ("gate_update_weight", "gate_update_bias"):
            gradient = current.get_parameter(name).grad
            assert torch.isfinite(gradient).all(), name
            assert torch.count_nonzero(gradient), name


@pytest.mark.parametrize("injection", INJECTIONS)
def test_nonconstant_gate_matches_independent_formula_without_scaling_q(injection):
    current = model(injection, "token")
    with torch.no_grad():
        current.gate_update_weight.copy_(torch.linspace(-0.7, 0.4, 12))
        current.gate_update_bias.fill_(-0.2)
    x, h = (torch.randn(2, 4, 12, dtype=torch.float64) for _ in range(2))
    u = current.update_gate(h)
    assert u.shape == (2, 4, 1)
    assert u.std() > 0
    actual = current._update_boundary(x, h)
    assert torch.allclose(actual, independent_boundary(current, x, h), atol=1e-14, rtol=1e-14)
    if injection == "fixed1":
        joint_gated = x + u * (current._discoloop_boundary(h) - x)
        assert not torch.allclose(actual, joint_gated)


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("mode", MODES)
def test_only_nonfinal_handoffs_gate_and_decode_and_final_state_is_raw_h(injection, mode):
    current = model(injection, mode).eval()
    rounds, feedback_calls, gate_calls, decoder_calls = [], [], [], []
    original_round = current._round
    original_feedback = current._discoloop_boundary
    original_gate = current.update_gate

    def record_round(x, *args, **kwargs):
        h = original_round(x, *args, **kwargs)
        rounds.append((x, h))
        return h

    def record_feedback(h):
        feedback_calls.append(h)
        return original_feedback(h)

    def record_gate(h):
        gate_calls.append(h)
        return original_gate(h)

    current._round = record_round
    current._discoloop_boundary = record_feedback
    current.update_gate = record_gate
    hook = current.transformer.ln_f.register_forward_hook(
        lambda _module, _args, output: decoder_calls.append(output)
    )
    try:
        logits, first, last = current.forward_with_states(TOKENS)
    finally:
        hook.remove()
    assert len(rounds) == 3
    assert len(gate_calls) == (0 if mode == "full" else 2)
    assert len(feedback_calls) == (2 if injection == "fixed1" else 0)
    assert len(decoder_calls) == (3 if injection == "fixed1" else 1)
    assert torch.equal(first, rounds[0][1])
    assert torch.equal(last, rounds[-1][1])
    assert torch.equal(logits, current.transformer.ln_f(last) @ current.transformer.wte.weight.T)
    if mode != "full":
        for (x, h), (next_input, _) in zip(rounds[:-1], rounds[1:], strict=True):
            assert torch.allclose(
                next_input, independent_boundary(current, x, h), atol=1e-14, rtol=1e-14
            )


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("mode", MODES)
def test_one_round_matches_plain_gpt_and_unused_gate_has_no_gradient(injection, mode):
    current = model(injection, mode, repeats=1)
    reference = reference_model("none", repeats=1)
    a, b = current.forward_with_states(TOKENS), reference.forward_with_states(TOKENS)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    a[0].square().mean().backward()
    b[0].square().mean().backward()
    assert_common_gradients(reference, current)
    if mode == "token":
        assert current.gate_update_weight.grad is None
        assert current.gate_update_bias.grad is None


@pytest.mark.parametrize("injection", INJECTIONS)
def test_learned_connector_is_differentiable_token_local_and_causal(injection):
    current = model(injection, "token").eval()
    with torch.no_grad():
        current.gate_update_weight.copy_(torch.linspace(-0.4, 0.2, 12))
    x, h = (torch.randn(1, 1, 12, dtype=torch.float64, requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(current._update_boundary, (x, h), atol=1e-5, fast_mode=True)
    states = torch.randn(1, 4, 12, dtype=torch.float64)
    changed = states.clone()
    changed[:, 2:] += torch.randn_like(changed[:, 2:])
    assert torch.equal(current.update_gate(states)[:, :2], current.update_gate(changed)[:, :2])
    a, b = torch.tensor([[2, 3, 4, 5]]), torch.tensor([[2, 3, 9, 8]])
    assert torch.equal(current(a)[:, :2], current(b)[:, :2])


@pytest.mark.parametrize("injection", INJECTIONS)
@pytest.mark.parametrize("mode", ("fixed", "token"))
def test_checkpoint_roundtrip_preserves_gate_storage_and_position_readout(injection, mode):
    current = model(injection, mode)
    positions = torch.tensor([[1, 3], [0, 2]])
    stream = io.BytesIO()
    torch.save(current.state_dict(), stream)
    stream.seek(0)
    restored = model(injection, mode)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(current(TOKENS, positions), restored(TOKENS, positions))
    expected = current(TOKENS)[torch.arange(len(TOKENS))[:, None], positions]
    assert torch.equal(current(TOKENS, positions), expected)
    assert len(list(restored.parameters())) == len({id(p) for p in restored.parameters()})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"injection": "paper"},
        {"update_mode": "dual"},
        {"initial_update": 0},
        {"initial_update": 1},
        {"initial_update": float("nan")},
        {"initial_update": float("inf")},
        {"first_round_kv": True},
        {"injection": "none", "discoloop": True},
        {"injection": "fixed1", "discoloop": False},
        {"discoloop_alpha": 0},
        {"discoloop_temperature": 2},
        {"discoloop_eps": 1e-5},
    ],
)
def test_conflicting_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        ResidualGateInjectionGPT(config(), **kwargs)


def test_paper_gate_configuration_cannot_silently_reintroduce_learned_injection():
    with pytest.raises((TypeError, ValueError)):
        ResidualGateInjectionGPT(config(), gate_variant="paper")
