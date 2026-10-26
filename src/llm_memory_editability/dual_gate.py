"""Token-local loop-boundary gates; the complete GPT blocks remain unchanged."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .self_decode_alignment import _loss_dtype, _readout
from .supervised_alignment import SupervisedAlignmentGPT

VARIANTS = ("paper", "scalar", "dual", "vector_injection")


class DualGateGPT(SupervisedAlignmentGPT):
    def __init__(self, config, repeats=4, *, gate_variant="dual", **kwargs):
        if gate_variant not in VARIANTS:
            raise ValueError(f"Unknown gate variant: {gate_variant}")
        if kwargs.get("first_round_kv", False):
            raise ValueError("This comparison excludes cross-round KV")
        super().__init__(config, repeats, **kwargs)
        self.gate_variant = gate_variant
        d = config.n_embd
        # torch.zeros/constant construction consumes no initialization RNG.
        if gate_variant == "paper":
            self.gate_paper_weight = nn.Parameter(torch.zeros(d))
            self.gate_paper_bias = nn.Parameter(torch.tensor(math.log(99.0)))
        else:
            self.gate_u = nn.Parameter(torch.zeros(d))
            self.gate_q = nn.Parameter(torch.zeros(d))
            self.gate_bias = nn.Parameter(torch.zeros(()))
            if gate_variant in ("dual", "vector_injection"):
                self.gate_r_u = nn.Parameter(torch.zeros(d))
                self.gate_r_q = nn.Parameter(torch.zeros(d))
                self.gate_r_bias = nn.Parameter(torch.zeros(d))

    def _discoloop_boundary(self, state):
        if self.discoloop_alpha == 0:
            return state
        logits = _readout(self, state)
        probability = (_loss_dtype(logits) / self.discoloop_temperature).softmax(-1)
        v = _loss_dtype(F.linear(probability, self.transformer.wte.weight.T))
        q = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + self.discoloop_eps)
        if self.gate_variant == "paper":
            g = torch.sigmoid(
                (v * self.gate_paper_weight).sum(-1, keepdim=True) + self.gate_paper_bias
            )
            return state + (self.discoloop_alpha * g * q).to(state.dtype)
        u = F.layer_norm(
            _loss_dtype(state), (self.config.n_embd,), eps=self.config.layer_norm_epsilon
        )
        g = 2 * torch.sigmoid(
            (u * self.gate_u + q * self.gate_q).sum(-1, keepdim=True) + self.gate_bias
        )
        if self.gate_variant == "scalar":
            return state + (self.discoloop_alpha * g * q).to(state.dtype)
        r = 2 * torch.sigmoid(self.gate_r_u * u + self.gate_r_q * q + self.gate_r_bias)
        if self.gate_variant == "dual":
            return r.to(state.dtype) * state + (self.discoloop_alpha * g * q).to(state.dtype)
        return state + (self.discoloop_alpha * g * (r * q)).to(state.dtype)


def gate_parameter_count(variant, width):
    return (
        width + 1
        if variant == "paper"
        else (2 * width + 1 if variant == "scalar" else 5 * width + 1)
    )
