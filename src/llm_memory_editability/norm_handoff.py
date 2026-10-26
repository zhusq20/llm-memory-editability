"""Residual scaling and normalized/complementary DiscoLoop handoffs.

All variants retain the native differentiable decode/re-encode channel. Only
nonfinal boundaries change. ``fixed`` exists for equivalence checks, never as a
new scientific training arm in this batch.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .self_decode_alignment import _loss_dtype, _readout
from .supervised_alignment import SupervisedAlignmentGPT

VARIANTS = ("fixed", "normalized", "additive", "complementary")


class NormHandoffGPT(SupervisedAlignmentGPT):
    def __init__(self, config, repeats=4, *, handoff="fixed", residual_scale=1.0, **kwargs):
        if handoff not in VARIANTS:
            raise ValueError(f"Unknown handoff: {handoff}")
        if not math.isfinite(residual_scale) or not 0 < residual_scale <= 1:
            raise ValueError("residual_scale must be finite and in (0, 1]")
        if kwargs.get("first_round_kv", False):
            raise ValueError("This comparison excludes cross-round KV")
        super().__init__(config, repeats, **kwargs)
        self.handoff = handoff
        self.residual_scale = residual_scale
        if handoff in ("additive", "complementary"):
            # These constants consume no initialization/dropout RNG.
            self.gate_u = nn.Parameter(torch.zeros(config.n_embd))
            self.gate_q = nn.Parameter(torch.zeros(config.n_embd))
            self.gate_bias = nn.Parameter(torch.zeros(()))

    def _round(self, x, first_round_cache=None, capture_cache=None):
        if self.residual_scale == 1:
            return super()._round(x, first_round_cache, capture_cache)
        if first_round_cache is not None or capture_cache is not None:
            raise ValueError("Residual scaling is defined here only without KV caching")
        cfg = self.config
        batch, length, _ = x.shape
        for block in self.transformer.h:
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
            shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
            )
            y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, cfg.n_embd))
            x = x + self.residual_scale * F.dropout(y, cfg.resid_pdrop, self.training)
            x = x + self.residual_scale * block.mlp(block.ln_2(x))
        return x

    def _discoloop_boundary(self, state):
        if self.discoloop_alpha == 0 or self.handoff == "fixed":
            # Preserve the native cast-before-add order, including BF16 AMP.
            return super()._discoloop_boundary(state)
        logits = _readout(self, state)
        probability = (_loss_dtype(logits) / self.discoloop_temperature).softmax(-1)
        value = _loss_dtype(F.linear(probability, self.transformer.wte.weight.T))
        q = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.discoloop_eps)
        if self.handoff == "normalized":
            # First form precisely the native h+q, then reduce in FP32 (or FP64).
            mixed = _loss_dtype(state + self.discoloop_alpha * q.to(state.dtype))
            return (
                mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + self.discoloop_eps)
            ).to(state.dtype)
        h = _loss_dtype(state)
        u = F.layer_norm(h, (self.config.n_embd,), eps=self.config.layer_norm_epsilon)
        alpha = torch.sigmoid(
            (u * self.gate_u + q * self.gate_q).sum(-1, keepdim=True) + self.gate_bias
        )
        carry = 0.5 if self.handoff == "additive" else 1 - alpha
        return (carry * h + self.discoloop_alpha * alpha * q).to(state.dtype)


def gate_parameter_count(handoff, width):
    if handoff not in VARIANTS:
        raise ValueError(handoff)
    return 2 * width + 1 if handoff in ("additive", "complementary") else 0
