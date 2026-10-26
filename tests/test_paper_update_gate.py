"""Contracts for changing only the continuous update of the existing P model."""

import copy
import io

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.dual_gate import DualGateGPT
from llm_memory_editability.paper_update_gate import UPDATE_MODES, PaperUpdateGateGPT


def config():
    return GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )


def model(mode, dtype=torch.float64, repeats=3):
    torch.manual_seed(71)
    return PaperUpdateGateGPT(config(), repeats=repeats, update_mode=mode).to(dtype)


TOKENS = torch.tensor([[2, 4, 7, 8], [3, 2, 9, 1]])


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_full_preserves_old_p_initialization_rng_outputs_all_gradients_and_optimizer(dtype):
    current = model("full", dtype)
    actual_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    reference = DualGateGPT(config(), repeats=3, discoloop=True, gate_variant="paper").to(dtype)
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert current.state_dict().keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        assert torch.equal(value, current.state_dict()[name]), name
    optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001) for m in (reference, current)]
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        a = reference.forward_with_states(TOKENS)
        b = current.forward_with_states(TOKENS)
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        a[0].square().mean().backward()
        b[0].square().mean().backward()
        for name, parameter in reference.named_parameters():
            assert torch.equal(parameter.grad, current.get_parameter(name).grad), name
        for optimizer in optimizers:
            optimizer.step()
        for name, value in reference.state_dict().items():
            assert torch.equal(value, current.state_dict()[name]), name


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_fixed_and_token_pair_exact_initial_state_rng_logits_and_backbone_p_gradients(dtype):
    fixed = model("fixed", dtype)
    actual_rng = torch.get_rng_state().clone()
    learned = model("token", dtype)
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert fixed.state_dict().keys() == learned.state_dict().keys()
    for name, value in fixed.state_dict().items():
        assert torch.equal(value, learned.state_dict()[name]), name
    assert "gate_update_weight" in dict(fixed.named_buffers())
    assert "gate_update_weight" not in dict(fixed.named_parameters())
    assert "gate_update_weight" in dict(learned.named_parameters())
    a, b = fixed(TOKENS), learned(TOKENS)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, parameter in fixed.named_parameters():
        assert torch.equal(parameter.grad, learned.get_parameter(name).grad), name
    for name in ("gate_update_weight", "gate_update_bias"):
        gradient = learned.get_parameter(name).grad
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    assert (
        sum(p.numel() for p in learned.parameters()) - sum(p.numel() for p in fixed.parameters())
        == fixed.config.n_embd + 1
    )


@pytest.mark.parametrize("pair", ("old_p_full", "fixed_token"))
def test_cpu_bfloat16_autocast_preserves_pair_outputs_and_common_gradients(pair):
    if pair == "old_p_full":
        current = model("full", torch.float32)
        torch.manual_seed(71)
        reference = DualGateGPT(config(), repeats=3, discoloop=True, gate_variant="paper")
    else:
        reference = model("fixed", torch.float32)
        current = model("token", torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a = reference.forward_with_states(TOKENS)
        b = current.forward_with_states(TOKENS)
        assert a[0].dtype == b[0].dtype == torch.bfloat16
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        loss_a, loss_b = a[0].float().square().mean(), b[0].float().square().mean()
    loss_a.backward()
    loss_b.backward()
    for name, parameter in reference.named_parameters():
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.equal(parameter.grad, current.get_parameter(name).grad), name
    if pair == "fixed_token":
        for name in ("gate_update_weight", "gate_update_bias"):
            gradient = current.get_parameter(name).grad
            assert torch.isfinite(gradient).all()
            assert torch.count_nonzero(gradient)


def test_nonconstant_gate_matches_independent_formula_without_gating_p_channel():
    current = model("token")
    with torch.no_grad():
        current.gate_update_weight.copy_(torch.linspace(-1, 1, 12))
        current.gate_update_bias.fill_(0.2)
        current.gate_paper_weight.copy_(torch.linspace(0.4, -0.4, 12))
        current.gate_paper_bias.fill_(-0.3)
    x, h = (torch.randn(2, 4, 12, dtype=torch.float64) for _ in range(2))
    normalized = F.layer_norm(h, (12,), eps=current.config.layer_norm_epsilon)
    u = torch.sigmoid(
        (normalized * current.gate_update_weight).sum(-1, keepdim=True) + current.gate_update_bias
    )
    v = (current.transformer.ln_f(h) @ current.transformer.wte.weight.T).softmax(-1)
    v = v @ current.transformer.wte.weight
    q = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
    g = torch.sigmoid(
        (v * current.gate_paper_weight).sum(-1, keepdim=True) + current.gate_paper_bias
    )
    expected = x + u * (h - x) + g * q
    actual = current._update_boundary(x, h)
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert not torch.allclose(actual, x + u * (h + g * q - x))


@pytest.mark.parametrize("mode", UPDATE_MODES)
@pytest.mark.parametrize("repeats", (1, 3))
def test_final_round_is_raw_h_and_decoded_channel_runs_only_at_handoffs(mode, repeats):
    current = model(mode, repeats=repeats)
    calls = []
    original = current._discoloop_boundary
    current._discoloop_boundary = lambda h: (calls.append(h), original(h))[1]
    logits, first, last = current.forward_with_states(TOKENS)
    assert len(calls) == repeats - 1
    expected_logits = current.transformer.ln_f(last) @ current.transformer.wte.weight.T
    assert torch.equal(logits, expected_logits)
    # Independent recurrence uses raw final H, with no correction after it.
    x = current._embedded(TOKENS)
    expected_first = None
    for index in range(repeats):
        h = current._round(x)
        if index == 0:
            expected_first = h
        x = current._update_boundary(x, h) if index < repeats - 1 else h
    assert torch.equal(last, h)
    assert torch.equal(first, expected_first)
    if repeats == 1:
        logits.square().mean().backward()
        assert current.gate_paper_weight.grad is None
        if mode == "token":
            assert current.gate_update_weight.grad is None


def test_update_boundary_is_differentiable_and_token_local():
    current = model("token").eval()
    with torch.no_grad():
        current.gate_update_weight.copy_(torch.linspace(-0.4, 0.2, 12))
    x, h = (torch.randn(1, 1, 12, dtype=torch.float64, requires_grad=True) for _ in range(2))
    assert torch.autograd.gradcheck(current._update_boundary, (x, h), atol=1e-5)
    states = torch.randn(1, 4, 12, dtype=torch.float64)
    changed = states.clone()
    changed[:, 2:] += torch.randn_like(changed[:, 2:])
    assert torch.equal(current.update_gate(states)[:, :2], current.update_gate(changed)[:, :2])
    a = torch.tensor([[2, 3, 4, 5]])
    b = torch.tensor([[2, 3, 9, 8]])
    assert torch.equal(current(a)[:, :2], current(b)[:, :2])


@pytest.mark.parametrize("mode", UPDATE_MODES)
def test_checkpoint_roundtrip_retains_fixed_buffers_and_positions(mode):
    current = model(mode)
    positions = torch.tensor([[1, 3], [0, 2]])
    stream = io.BytesIO()
    torch.save(current.state_dict(), stream)
    stream.seek(0)
    restored = model(mode)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.equal(current(TOKENS, positions), restored(TOKENS, positions))
    expected = current(TOKENS)[torch.arange(len(TOKENS))[:, None], positions]
    assert torch.equal(current(TOKENS, positions), expected)
    assert len(list(restored.parameters())) == len({id(p) for p in restored.parameters()})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"update_mode": "dual"},
        {"initial_update": 0},
        {"initial_update": 1},
        {"initial_update": float("nan")},
        {"initial_update": float("inf")},
        {"gate_variant": "scalar"},
        {"first_round_kv": True},
        {"discoloop": False},
        {"discoloop_alpha": 0},
        {"discoloop_temperature": 2},
        {"discoloop_eps": 1e-5},
    ],
)
def test_conflicting_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        PaperUpdateGateGPT(config(), **kwargs)


def test_full_accepts_old_p_checkpoint_without_any_key_remapping():
    torch.manual_seed(9)
    old = DualGateGPT(config(), repeats=3, discoloop=True, gate_variant="paper").double()
    current = model("full")
    current.load_state_dict(copy.deepcopy(old.state_dict()), strict=True)
    assert torch.equal(old(TOKENS), current(TOKENS))
