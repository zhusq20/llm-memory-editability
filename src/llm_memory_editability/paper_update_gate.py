"""Add one continuous-update gate to the unchanged DiscoLoop paper gate.

For a nonfinal round, X is the input and H is the complete GPT output. The
handoff is X + u(H - X) + g_P(H) q(H). The native paper channel, final readout,
and all Transformer blocks are inherited. This is not a gate on H itself or
on the combined continuous/discrete update. There is no auxiliary objective.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .dual_gate import DualGateGPT
from .self_decode_alignment import _loss_dtype, _readout

UPDATE_MODES = ("full", "fixed", "token")


class PaperUpdateGateGPT(DualGateGPT):
    """Paper injection gate plus full, fixed, or learned continuous updates.

    ``full`` delegates the old paper implementation exactly, with no extra
    tensors in its state_dict. ``fixed`` and ``token`` have identical forward
    operations and initial values; fixed gate tensors are buffers and learned
    tensors are parameters. Both use one scalar per token, sharing across rounds,
    and affine-free LayerNorm of H with the unchanged GPT normalization eps.

    The final round produces H and reads it directly: neither gate is called
    after that round. Thus a one-round forward is the original GPT round.
    """

    def __init__(self, config, repeats=4, *, update_mode="full", initial_update=0.99, **kwargs):
        if update_mode not in UPDATE_MODES:
            raise ValueError(f"Unknown update mode: {update_mode}")
        if not math.isfinite(initial_update) or not 0 < initial_update < 1:
            raise ValueError("initial_update must be finite and strictly between zero and one")
        # This comparison changes only the continuous-update gate. Do not let
        # a caller silently change P's channel, gate variant, or cache access.
        required = {
            "gate_variant": "paper",
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
        self.update_mode = update_mode
        self.initial_update = initial_update
        if update_mode != "full":
            # Constant construction consumes no RNG: the backbone and P gate
            # retain their original initialization and subsequent sample RNG.
            weight = torch.zeros(config.n_embd)
            bias = torch.tensor(math.log(initial_update / (1 - initial_update)))
            if update_mode == "token":
                self.gate_update_weight = nn.Parameter(weight)
                self.gate_update_bias = nn.Parameter(bias)
            else:
                self.register_buffer("gate_update_weight", weight)
                self.register_buffer("gate_update_bias", bias)

    def update_gate(self, state):
        """Return u with shape [batch, tokens, 1], using only the current token."""
        if self.update_mode == "full":
            return state.new_ones((*state.shape[:-1], 1))
        normalized = F.layer_norm(
            _loss_dtype(state), (self.config.n_embd,), eps=self.config.layer_norm_epsilon
        )
        return torch.sigmoid(
            (normalized * self.gate_update_weight).sum(-1, keepdim=True) + self.gate_update_bias
        )

    def _update_boundary(self, previous, state):
        # Reuse the exact P channel once. This operation order evaluates
        # [H + g_P q] + (1-u)(X-H), algebraically X + u(H-X) + g_P q.
        # It avoids recomputing any decoder or changing its finite-precision
        # arithmetic. The full path uses the original expression exactly.
        paper_state = self._discoloop_boundary(state)
        if self.update_mode == "full":
            return paper_state
        update = self.update_gate(state)
        return paper_state + (1 - update).to(state.dtype) * (previous - state)

    def forward_with_states(self, tokens, positions=None):
        if self.update_mode == "full":
            return super().forward_with_states(tokens, positions)
        if self.repeats < 1 or not len(self.transformer.h):
            raise ValueError("at least one round and one block are required")
        x = self._embedded(tokens)
        first_state = None
        for index in range(self.repeats):
            h = self._round(x)
            if index == 0:
                first_state = h
            x = self._update_boundary(x, h) if index < self.repeats - 1 else h
        return _readout(self, x, positions), first_state, x
