"""Depth-shared GPT blocks for controlled composition-learning comparisons.

This keeps the existing SmallGPT tokenization, causal attention, dropout and tied
readout. ``config.layers`` counts independently parameterized blocks; each loop
executes this whole block group. Positions and token embeddings are added once.
"""

from __future__ import annotations

import math
from dataclasses import replace
from numbers import Integral

import torch
from torch.nn import functional as F

from .bios_model import matmul_flops
from .grok_depth import SmallGPT

INITIALIZATIONS = frozenset({"legacy_unique", "scaled_effective", "zero_residual"})


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


class LoopGPT(SmallGPT):
    """Vanilla fixed-loop Transformer with shared attention, MLP and block norms.

    ``legacy_unique`` reproduces SmallGPT's residual initialization based on the
    number of unique blocks. ``scaled_effective`` scales residual output weights
    by the total executed depth; ``zero_residual`` instead zeros those weights.
    Embeddings, other projections and all biases retain SmallGPT initialization.

    Changing ``repeats`` in forward does not change parameter initialization. It
    is a computation-depth intervention, not an independently initialized model.
    Save repeats, dropout and initialization alongside ``config_dict()`` and the
    state dict when constructing checkpoints; they are architectural metadata.
    """

    def __init__(self, config, repeats=1, dropout=0.1, initialization="scaled_effective"):
        repeats = _positive_integer(repeats, "repeats")
        _positive_integer(config.layers, "config.layers")
        if initialization not in INITIALIZATIONS:
            raise ValueError(f"initialization must be one of {sorted(INITIALIZATIONS)}")
        super().__init__(config, dropout=dropout)
        self.repeats = repeats
        self.initialization = initialization
        with torch.no_grad():
            for block in self.blocks:
                for module in (block.attention.proj, block.mlp.down):
                    if initialization == "scaled_effective":
                        module.weight.div_(math.sqrt(repeats))
                    elif initialization == "zero_residual":
                        module.weight.zero_()

    @property
    def effective_depth(self):
        return self.config.layers * self.repeats

    def iter_blocks(self, repeats=None):
        """Yield executed blocks; repeated occurrences are the same module object."""
        count = self.repeats if repeats is None else _positive_integer(repeats, "repeats")
        for _ in range(count):
            yield from self.blocks

    def forward(self, tokens, positions=None, repeats=None):
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
        for block in self.iter_blocks(repeats):
            z = block.ln1(x)
            batch, length, width = z.shape
            a = block.attention
            q, k, v = a.qkv(z).view(batch, length, 3, a.heads, width // a.heads).unbind(2)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
                dropout_p=p,
            )
            y = a.proj(y.transpose(1, 2).reshape(batch, length, width))
            x = x + F.dropout(y, p=p, training=self.training)
            x = x + F.dropout(block.mlp(block.ln2(x)), p=p, training=self.training)
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, self.token.weight)


def flops(config, repeats, batch, sequence, output_positions=2, backward=True):
    """Estimate matmul FLOPs using executed blocks, counting the readout once.

    Uses the same executed-shape estimate as bios_model.matmul_flops. Excludes
    elementwise operations and uses three forward equivalents for training.
    """
    count = _positive_integer(repeats, "repeats")
    layers = _positive_integer(config.layers, "config.layers")
    return matmul_flops(
        replace(config, layers=layers * count),
        batch,
        sequence,
        output_positions=output_positions,
        backward=backward,
    )
