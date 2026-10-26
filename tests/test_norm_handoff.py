"""Model contracts run by the user on the target machine before any science."""

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
GPT2Config = pytest.importorskip("transformers").GPT2Config

from torch.nn import functional as F  # noqa: E402

from llm_memory_editability.norm_handoff import (  # noqa: E402
    VARIANTS,
    NormHandoffGPT,
    gate_parameter_count,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_norm_handoff import optimizer_for  # noqa: E402

torch.set_num_threads(1)


def model(handoff="fixed", scale=1, dropout=0):
    config = GPT2Config(
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
    torch.manual_seed(71)
    return NormHandoffGPT(
        config, repeats=4, discoloop=True, handoff=handoff, residual_scale=scale
    ).double()


def channel(current, h):
    logits = current.transformer.ln_f(h) @ current.transformer.wte.weight.T
    v = logits.softmax(-1) @ current.transformer.wte.weight
    return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)


@pytest.mark.parametrize("handoff", VARIANTS)
def test_same_backbone_rng_parameters_and_tied_embedding(handoff):
    current = model(handoff)
    actual_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    reference = SupervisedAlignmentGPT(current.config, repeats=4, discoloop=True).double()
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert all(
        torch.equal(p, reference.state_dict()[n])
        for n, p in current.state_dict().items()
        if not n.startswith("gate_")
    )
    assert sum(
        p.numel() for n, p in current.named_parameters() if n.startswith("gate_")
    ) == gate_parameter_count(handoff, 12)
    assert len(current.transformer.h) == 1


def test_native_path_forward_and_every_gradient_with_dropout():
    current = model(dropout=0.1)
    reference = SupervisedAlignmentGPT(current.config, repeats=4, discoloop=True).double()
    reference.load_state_dict(current.state_dict(), strict=True)
    x = torch.tensor([[2, 4, 7, 8], [3, 2, 9, 1]])
    torch.manual_seed(19)
    a = reference(x)
    torch.manual_seed(19)
    b = current(x)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for n, p in reference.named_parameters():
        assert torch.equal(p.grad, current.get_parameter(n).grad), n


def test_additive_complementary_same_initial_forward_backbone_grads_gate_grads_differ():
    a, c = model("additive", dropout=0.1), model("complementary", dropout=0.1)
    x = torch.tensor([[2, 4, 7, 8], [3, 2, 9, 1]])
    torch.manual_seed(19)
    ya = a(x)
    torch.manual_seed(19)
    yc = c(x)
    assert torch.equal(ya, yc)
    ya.square().mean().backward()
    yc.square().mean().backward()
    different = []
    for name, p in a.named_parameters():
        q = c.get_parameter(name)
        assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
        if name.startswith("gate_"):
            different.append(not torch.equal(p.grad, q.grad))
        else:
            assert torch.equal(p.grad, q.grad), name
    assert any(different)


@pytest.mark.parametrize("handoff", VARIANTS)
def test_exact_declared_boundary_with_nonconstant_gate(handoff):
    current = model(handoff)
    for name, p in current.named_parameters():
        if name.startswith("gate_"):
            with torch.no_grad():
                p.copy_(torch.linspace(-0.3, 0.4, p.numel()).reshape(p.shape))
    h = torch.randn(2, 3, 12, dtype=torch.float64)
    q = channel(current, h)
    if handoff == "fixed":
        expected = h + q
    elif handoff == "normalized":
        mixed = h + q
        expected = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
    else:
        u = F.layer_norm(h, (12,), eps=current.config.layer_norm_epsilon)
        alpha = torch.sigmoid(
            (u * current.gate_u + q * current.gate_q).sum(-1, keepdim=True) + current.gate_bias
        )
        carry = 0.5 if handoff == "additive" else 1 - alpha
        expected = carry * h + alpha * q
        assert bool(((alpha > 0) & (alpha < 1)).all())
    assert torch.allclose(current._discoloop_boundary(h), expected, atol=1e-14, rtol=1e-14)
    if handoff == "normalized":
        assert bool((expected.square().mean(-1) <= 1).all())


def test_residual_scale_applies_to_both_increments_all_rounds_and_not_channel():
    current = model(scale=0.25).eval()
    h = torch.randn(2, 4, 12, dtype=torch.float64)
    block = current.transformer.h[0]
    q, k, v = block.attn.c_attn(block.ln_1(h)).split(12, dim=2)
    q, k, v = (x.view(2, 4, 3, 4).transpose(1, 2) for x in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
    z = h + 0.25 * block.attn.c_proj(y.transpose(1, 2).reshape(2, 4, 12))
    expected = z + 0.25 * block.mlp(block.ln_2(z))
    assert torch.equal(current._round(h), expected)
    assert torch.allclose(current._discoloop_boundary(h), h + channel(current, h), atol=1e-14)
    x = torch.tensor([[2, 3, 4]])
    state = current._embedded(x)
    for index in range(4):
        state = current._round(state)
        if index < 3:
            state = current._discoloop_boundary(state)
    assert torch.equal(current.forward_with_states(x)[2], state)


@pytest.mark.parametrize("handoff", VARIANTS)
def test_causality_boundary_count_differentiability_reload_and_optimizer(handoff):
    current = model(handoff).eval()
    h = torch.randn(1, 1, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(current._discoloop_boundary, (h,), atol=1e-5)
    x, changed = torch.tensor([[2, 3, 4, 5]]), torch.tensor([[2, 3, 9, 8]])
    assert torch.equal(current(x)[:, :2], current(changed)[:, :2])
    calls = []
    original = current._discoloop_boundary
    current._discoloop_boundary = lambda h: (calls.append(1), original(h))[1]
    current(x)
    assert len(calls) == 3
    current._discoloop_boundary = original
    clone = model(handoff).eval()
    clone.load_state_dict(copy.deepcopy(current.state_dict()), strict=True)
    assert torch.equal(clone(x), current(x))
    optimizer = optimizer_for(current, 1e-4, 0.1)
    gates = {id(p) for n, p in current.named_parameters() if n.startswith("gate_")}
    identifiers = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(identifiers) == len(set(identifiers)) == len(list(current.parameters()))
    assert all(
        g["weight_decay"] == 0
        for g in optimizer.param_groups
        if any(id(p) in gates for p in g["params"])
    )


@pytest.mark.parametrize("scale", (0, -1, 1.1, float("nan"), float("inf")))
def test_invalid_scales_rejected(scale):
    with pytest.raises(ValueError):
        model(scale=scale)
