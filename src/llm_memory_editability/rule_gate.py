"""Parameter-free continuous-update gates computed from the model's own decodes.

For a nonfinal round with input X and complete Transformer output H, the handoff is
the fixed-q-plus-u formula X + u(H - X) + q(H), with q the unchanged DiscoLoop decoded
feedback at strength one. Here u has no parameters:

* stability: u_t(r) = sum_v p_r(t, v) p_{r-1}(t, v), where p_r is the full-vocabulary
  decode softmax(E LN_f(H_r)) at token t and p_0 is the one-hot input token;
* readiness: c_t(r) = 1 when token t reads no earlier position with round-r attention
  of at least tau, otherwise s_t(r) * min over those sources of c_j(r - 1), with
  c(0) = 1 and s the stability value above; u = c.

The gate signal is computed without gradient: it is a fixed rule over the model's own
decodes and attention, not a learned controller, and it never sees graph identity,
hop counts, answer positions or labels. Attention for the rule is recomputed from the
single shared block's Q/K on the round input, head-averaged, in float32. The final
round reads raw H. No module, parameter or buffer is added to the GPT backbone.
"""

from __future__ import annotations

import math

import torch

from .self_decode_alignment import _loss_dtype, _readout
from .supervised_alignment import SupervisedAlignmentGPT

RULES = ("stability", "readiness")


def first_agreement(probability, tokens):
    """Agreement with a one-hot input token, without materializing the one-hot."""
    return probability.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)


def decode_agreement(probability, previous):
    return (probability * previous).sum(-1)


def readiness(stability, attention, previous, tau):
    """Dependency-closed stability over earlier attended positions; [B, T]."""
    length = attention.shape[-1]
    earlier = torch.ones(length, length, dtype=torch.bool, device=attention.device).tril(-1)
    sources = (attention >= tau) & earlier
    candidate = torch.where(
        sources,
        previous[:, None, :].expand(-1, length, -1),
        torch.full_like(attention, float("inf")),
    )
    upstream = candidate.min(-1).values
    return torch.where(sources.any(-1), stability * upstream, torch.ones_like(stability))


class RuleGateGPT(SupervisedAlignmentGPT):
    """fixed-q-plus-u handoff whose token-scalar u is a parameter-free decode rule."""

    def __init__(self, config, repeats=4, *, rule="stability", tau=0.1, **kwargs):
        if rule not in RULES:
            raise ValueError(f"Unknown gate rule: {rule}")
        if not math.isfinite(tau) or not 0 < tau < 1:
            raise ValueError("tau must be strictly between zero and one")
        required = {
            "first_round_kv": False,
            "discoloop": True,
            "discoloop_alpha": 1.0,
            "discoloop_temperature": 1.0,
            "discoloop_eps": 1e-6,
        }
        for name, value in required.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"This comparison requires {name}={value!r}")
            kwargs[name] = value
        super().__init__(config, repeats, **kwargs)
        if rule == "readiness" and len(self.transformer.h) != 1:
            raise ValueError("The readiness rule reads the single shared block's attention")
        self.rule, self.tau = rule, tau
        self.injection = "fixed1"
        self.update_mode = rule

    def round_attention(self, x):
        """Head-averaged causal attention probabilities of the round, [B, T(query), T(key)]."""
        cfg, block = self.config, self.transformer.h[0]
        batch, length, _ = x.shape
        heads, dim = cfg.n_head, cfg.n_embd // cfg.n_head
        q, k, _ = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float().view(batch, length, heads, dim).transpose(1, 2)
            k = k.float().view(batch, length, heads, dim).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
            causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            return scores.masked_fill(~causal, float("-inf")).softmax(-1).mean(1)

    def handoff(self, previous, state, update):
        """Same operation order as ResidualGateInjectionGPT with injection fixed1."""
        candidate = self._discoloop_boundary(state)
        return candidate + (1 - update).to(state.dtype) * (previous - state)

    def rule_gates(self, tokens, probability, previous_probability, attention, previous_ready):
        stability = (
            first_agreement(probability, tokens)
            if previous_probability is None
            else decode_agreement(probability, previous_probability)
        )
        if self.rule == "stability":
            return stability, None
        ready_before = torch.ones_like(stability) if previous_ready is None else previous_ready
        ready = readiness(stability, attention, ready_before, self.tau)
        return ready, ready

    def forward_with_states(self, tokens, positions=None, return_gates=False):
        if self.repeats < 1 or not len(self.transformer.h):
            raise ValueError("at least one round and one block are required")
        x = self._embedded(tokens)
        first_state = None
        previous_probability = previous_ready = None
        gates = []
        for index in range(self.repeats):
            final = index == self.repeats - 1
            attention = None
            if self.rule == "readiness" and not final:
                with torch.no_grad():
                    attention = self.round_attention(x)
            h = self._round(x)
            if index == 0:
                first_state = h
            if final:
                x = h
                break
            with torch.no_grad():
                probability = _loss_dtype(_readout(self, h)).softmax(-1)
                update, previous_ready = self.rule_gates(
                    tokens, probability, previous_probability, attention, previous_ready
                )
                previous_probability = probability
            gates.append(update)
            x = self.handoff(x, h, update.unsqueeze(-1))
        logits = _readout(self, x, positions)
        if return_gates:
            return logits, first_state, x, gates
        return logits, first_state, x
