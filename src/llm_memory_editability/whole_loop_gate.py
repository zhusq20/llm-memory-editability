"""Depth-consistent whole-loop interpolation of the complete native GPT stack."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .self_decode_alignment import SelfDecodeGPT, _loss_dtype, _readout


class WholeLoopGateGPT(SelfDecodeGPT):
    def __init__(self, config, repeats=5, *, feedback_beta=0, gate_mode="full", alpha=0.9):
        if feedback_beta not in (0, 1) or gate_mode not in ("full", "fixed", "token"):
            raise ValueError("Invalid whole-loop configuration")
        if not 0 < alpha < 1:
            raise ValueError("Initial alpha must be strictly between zero and one")
        super().__init__(config, repeats, first_round_kv=False)
        self.feedback_beta, self.gate_mode = feedback_beta, gate_mode
        # Both paths use exactly the same sigmoid/logit rounding, including AMP.
        self.register_buffer("fixed_logit", torch.tensor(math.log(alpha / (1 - alpha))))
        if gate_mode == "token":
            self.gate_weight = nn.Parameter(torch.zeros(config.n_embd))
            self.gate_bias = nn.Parameter(self.fixed_logit.clone())

    def feedback(self, h):
        probability = _loss_dtype(_readout(self, h)).softmax(-1)
        embedded = _loss_dtype(F.linear(probability, self.transformer.wte.weight.T))
        return embedded * torch.rsqrt(embedded.square().mean(-1, keepdim=True) + 1e-6)

    def step(self, x, *, gate_override=None):
        h = self._round(x)
        candidate = h + self.feedback(h).to(h.dtype) if self.feedback_beta else h
        if self.gate_mode == "token":
            normalized = F.layer_norm(
                _loss_dtype(h), (self.config.n_embd,), eps=self.config.layer_norm_epsilon
            )
            alpha = torch.sigmoid(
                (normalized * self.gate_weight).sum(-1, keepdim=True) + self.gate_bias
            )
        else:
            value = self.fixed_logit.sigmoid() if self.gate_mode == "fixed" else x.new_tensor(1)
            alpha = value.expand(*x.shape[:-1], 1)
        if gate_override is not None:
            alpha = gate_override(alpha)
        # Exact endpoint branches preserve the native forward (no cancellation).
        if gate_override is None and self.gate_mode == "full":
            updated = candidate
        else:
            mixed = x + alpha.to(x.dtype) * (candidate - x)
            updated = torch.where(alpha == 0, x, torch.where(alpha == 1, candidate, mixed))
        return updated, {"alpha": alpha, "candidate": candidate, "previous": x}

    def trajectory(self, tokens, num_loops=None, *, gate_override=None):
        loops = self.repeats if num_loops is None else num_loops
        if not isinstance(loops, int) or loops < 1:
            raise ValueError("num_loops must be a positive integer")
        x = self._embedded(tokens)
        for r in range(1, loops + 1):
            override = None if gate_override is None else lambda a, r=r: gate_override(r, a)
            x, info = self.step(x, gate_override=override)
            yield r, x, info

    def readout(self, state, positions=None):
        if positions is None:
            positions = torch.full((len(state), 1), state.shape[1] - 1, device=state.device)
        return _readout(self, state, positions).squeeze(1)

    def forward(self, tokens, positions=None, *, num_loops=None):
        for _, _state, _ in self.trajectory(tokens, num_loops):
            pass
        return self.readout(_state, positions)


def matrix_flops(config, vocab_size, sequence_length, beta, loops=5, batch_size=1):
    """Dense matmul estimate; excludes norms/softmax/activations and optimizer."""
    d, layers = config["n_embd"], config["n_layer"]
    inner = config.get("n_inner", 4 * d)
    t = sequence_length
    blocks = layers * (8 * t * d * d + 4 * t * d * inner + 4 * t * t * d)
    feedback = beta * 4 * t * d * vocab_size
    return batch_size * (loops * (blocks + feedback) + 2 * d * vocab_size)
