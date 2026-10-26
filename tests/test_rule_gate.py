"""Contracts for the parameter-free decode-rule gates of the fixed-q-plus-u handoff."""

import math

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.residual_gate_injection import ResidualGateInjectionGPT
from llm_memory_editability.rule_gate import RULES, RuleGateGPT, readiness
from llm_memory_editability.self_decode_alignment import _readout
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

TOKENS = torch.tensor([[2, 4, 7, 8, 5, 1], [3, 2, 9, 1, 6, 4]])


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


def build(rule, repeats=4, dtype=torch.float64, dropout=0.0, tau=0.1):
    torch.manual_seed(71)
    return RuleGateGPT(config(dropout), repeats=repeats, rule=rule, tau=tau).to(dtype)


def naive_readiness(stability, attention, previous, tau):
    batch, length = stability.shape
    out = torch.empty_like(stability)
    for b in range(batch):
        for t in range(length):
            sources = [j for j in range(t) if attention[b, t, j] >= tau]
            if not sources:
                out[b, t] = 1.0
            else:
                out[b, t] = stability[b, t] * min(previous[b, j] for j in sources)
    return out


def manual(model, tokens, repeats):
    """Independent replay with explicit one-hot agreement and naive readiness."""
    x = model._embedded(tokens)
    previous = F.one_hot(tokens, model.config.vocab_size).to(x.dtype)
    ready = torch.ones(tokens.shape, dtype=x.dtype)
    gates = []
    for index in range(repeats):
        if index < repeats - 1 and model.rule == "readiness":
            cfg, block = model.config, model.transformer.h[0]
            q, k, _ = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
            heads, dim = cfg.n_head, cfg.n_embd // cfg.n_head
            q = q.view(*tokens.shape, heads, dim).transpose(1, 2)
            k = k.view(*tokens.shape, heads, dim).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
            mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool).tril()
            attention = scores.masked_fill(~mask, float("-inf")).softmax(-1).mean(1)
        h = model._round(x)
        if index == repeats - 1:
            return _readout(model, h), gates
        p = _readout(model, h).softmax(-1)
        stability = (p * previous).sum(-1)
        if model.rule == "readiness":
            ready = naive_readiness(stability, attention, ready, model.tau)
            u = ready
        else:
            u = stability
        previous = p
        gates.append(u)
        x = model._discoloop_boundary(h) + (1 - u).unsqueeze(-1) * (x - h)
    raise AssertionError("unreachable")


def test_no_parameters_buffers_or_rng_added():
    for rule in RULES:
        torch.manual_seed(5)
        reference = SupervisedAlignmentGPT(config(), 4, discoloop=True)
        state = torch.get_rng_state()
        torch.manual_seed(5)
        model = RuleGateGPT(config(), 4, rule=rule)
        assert torch.equal(state, torch.get_rng_state())
        assert model.state_dict().keys() == reference.state_dict().keys()
        pairs = zip(model.state_dict().values(), reference.state_dict().values(), strict=True)
        assert all(torch.equal(a, b) for a, b in pairs)
        assert sum(p.numel() for p in model.parameters()) == sum(
            p.numel() for p in reference.parameters()
        )


def test_rule_values_match_independent_replay():
    for rule in RULES:
        model = build(rule).eval()
        logits, _, _, gates = model.forward_with_states(TOKENS, return_gates=True)
        expected, expected_gates = manual(model, TOKENS, 4)
        assert torch.allclose(
            _readout(model, model.forward_with_states(TOKENS)[2]), expected, atol=1e-12
        )
        assert len(gates) == 3
        for got, want in zip(gates, expected_gates, strict=False):
            assert torch.allclose(got, want, atol=1e-12)
            assert ((got >= 0) & (got <= 1 + 1e-12)).all()
        assert logits.shape == (2, 6, 23)


def test_constant_gate_reduces_to_fixed_q_plus_u_handoff():
    model = build("stability").eval()
    reference = (
        ResidualGateInjectionGPT(
            config(), 4, injection="fixed1", update_mode="fixed", initial_update=0.95
        )
        .to(torch.float64)
        .eval()
    )
    missing = reference.load_state_dict(model.state_dict(), strict=False)
    assert set(missing.missing_keys) == {"gate_update_weight", "gate_update_bias"}
    constant = reference.update_gate(torch.zeros(1, 1, 12, dtype=torch.float64))[0, 0, 0]
    model.rule_gates = lambda tokens, p, *_: (
        torch.full(tokens.shape, float(constant), dtype=p.dtype),
        None,
    )
    got = model.forward_with_states(TOKENS)[0]
    want = reference.forward_with_states(TOKENS)[0]
    assert torch.allclose(got, want, atol=1e-12)


def test_causal_prefix_and_no_total_round_dependence():
    for rule in RULES:
        model = build(rule).eval()
        changed = TOKENS.clone()
        changed[:, 4:] = torch.tensor([[11, 12], [13, 14]])
        a = model.forward_with_states(TOKENS, return_gates=True)
        b = model.forward_with_states(changed, return_gates=True)
        assert torch.allclose(a[0][:, :4], b[0][:, :4], atol=1e-12)
        for ga, gb in zip(a[3], b[3], strict=False):
            assert torch.allclose(ga[:, :4], gb[:, :4], atol=1e-12)
        short = build(rule, repeats=2).eval()
        long = build(rule, repeats=5).eval()
        long.load_state_dict(short.state_dict())
        gates_short = short.forward_with_states(TOKENS, return_gates=True)[3]
        gates_long = long.forward_with_states(TOKENS, return_gates=True)[3]
        assert torch.allclose(gates_short[0], gates_long[0], atol=1e-12)
        expected, _ = manual(long, TOKENS, 2)
        assert torch.allclose(short.forward_with_states(TOKENS)[0], expected, atol=1e-12)


def test_gate_signal_carries_no_gradient_and_backbone_gradients_are_finite():
    for rule in RULES:
        model = build(rule, dtype=torch.float32, dropout=0.1).train()
        torch.manual_seed(3)
        logits, _, _, gates = model.forward_with_states(TOKENS, return_gates=True)
        assert all(not gate.requires_grad for gate in gates)
        loss = F.cross_entropy(logits.flatten(0, 1), TOKENS.flatten())
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)


def test_readiness_edge_cases():
    stability = torch.tensor([[0.9, 0.2, 1.0]])
    previous = torch.tensor([[1.0, 0.0, 1.0]])
    attention = torch.tensor([[[1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.05, 0.9, 0.05]]])
    got = readiness(stability, attention, previous, 0.1)
    # t=0 has no earlier source; t=1 reads ready position 0; t=2 reads unready position 1.
    assert torch.allclose(got, torch.tensor([[1.0, 0.2, 0.0]]))
    with pytest.raises(ValueError):
        RuleGateGPT(config(), 4, rule="learned")
    with pytest.raises(ValueError):
        RuleGateGPT(config(), 4, rule="readiness", tau=1.0)
    with pytest.raises(ValueError):
        RuleGateGPT(config(), 4, rule="stability", discoloop=False)
