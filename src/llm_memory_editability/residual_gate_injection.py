"""A continuous cross-round update gate with absent or fixed decoded feedback.

For a nonfinal round, X is its input and H is its complete Transformer output.
The two handoffs are X + u(H)(H-X) and X + u(H)(H-X) + q(H). The final
round always reads raw H. No attention/MLP, objective, or stopping rule changes.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .self_decode_alignment import _loss_dtype, _readout
from .supervised_alignment import SupervisedAlignmentGPT

INJECTIONS = ("none", "fixed1")
UPDATE_MODES = ("full", "fixed", "token")


class ResidualGateInjectionGPT(SupervisedAlignmentGPT):
    """Token-scalar continuous gate, shared across all nonfinal boundaries.

    ``none`` never computes decoded feedback. ``fixed1`` uses the existing raw
    DiscoLoop channel at strength one, outside u. There is no learned injection
    gate. ``full`` and ``fixed`` are engineering references, not scientific arms.
    Constant gate construction consumes no RNG and preserves backbone weights.
    """

    def __init__(
        self,
        config,
        repeats=4,
        *,
        injection="none",
        update_mode="token",
        initial_update=0.95,
        **kwargs,
    ):
        if injection not in INJECTIONS:
            raise ValueError(f"Unknown injection: {injection}")
        if update_mode not in UPDATE_MODES:
            raise ValueError(f"Unknown update mode: {update_mode}")
        if not math.isfinite(initial_update) or not 0 < initial_update < 1:
            raise ValueError("initial_update must be finite and strictly between zero and one")
        required = {
            "first_round_kv": False,
            "discoloop": injection == "fixed1",
            "discoloop_alpha": 1.0,
            "discoloop_temperature": 1.0,
            "discoloop_eps": 1e-6,
        }
        for name, value in required.items():
            if name in kwargs and kwargs[name] != value:
                raise ValueError(f"This comparison requires {name}={value!r}")
            kwargs[name] = value
        super().__init__(config, repeats, **kwargs)
        self.injection = injection
        self.update_mode = update_mode
        self.initial_update = initial_update
        if update_mode != "full":
            weight = torch.zeros(config.n_embd)
            bias = torch.tensor(math.log(initial_update / (1 - initial_update)))
            if update_mode == "token":
                self.gate_update_weight = nn.Parameter(weight)
                self.gate_update_bias = nn.Parameter(bias)
            else:
                self.register_buffer("gate_update_weight", weight)
                self.register_buffer("gate_update_bias", bias)

    def update_gate(self, state):
        if self.update_mode == "full":
            return state.new_ones((*state.shape[:-1], 1))
        normalized = F.layer_norm(
            _loss_dtype(state), (self.config.n_embd,), eps=self.config.layer_norm_epsilon
        )
        return torch.sigmoid(
            (normalized * self.gate_update_weight).sum(-1, keepdim=True) + self.gate_update_bias
        )

    def _update_boundary(self, previous, state):
        candidate = self._discoloop_boundary(state) if self.injection == "fixed1" else state
        if self.update_mode == "full":
            return candidate
        # Same residual-correction order as PaperUpdateGateGPT. q is outside u.
        update = self.update_gate(state)
        return candidate + (1 - update).to(state.dtype) * (previous - state)

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
