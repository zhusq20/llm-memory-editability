"""Loop handoffs with rotary (relative) positions instead of learned absolute positions.

Hypothesis under test: in a loop, hop k is computed at token position k. With learned
absolute position embeddings a fact learned only as an atomic query (entity at 0,
relation at 1) is looked up in a position-bound way and does not transfer to later
hop positions, and positions beyond the trained length are untrained. Rotary position
embeddings make attention scores depend only on relative offsets, so the lookup
"entity one token before the relation" is the same at every position.

``position="absolute"`` reproduces LoopNativeHandoffGPT exactly. ``position="rope"``
drops the learned position embedding from the input (the parameter stays in the state
dict, untouched by the forward pass, so the step-0 backbone still matches the reused
runs) and rotates every query and key with NeoX-style rotary embeddings over the full
head dimension. Nothing else changes: same blocks, residuals, dropout, normalization,
tied readout, decoded feedback and handoffs.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .loop_native_handoff import LoopNativeHandoffGPT

POSITIONS = ("absolute", "rope")


def rotate(x, positions, base):
    """Rotary embedding of [B, H, T, D] by integer positions [T], computed in float32."""
    width = x.shape[-1]
    frequency = 1.0 / (
        base ** (torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    )
    angles = positions.to(torch.float32)[:, None] * frequency[None]
    cos, sin = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    value = x.float()
    first, second = value.chunk(2, dim=-1)
    turned = torch.cat((-second, first), dim=-1)
    return (value * cos + turned * sin).to(x.dtype)


class LoopPositionGPT(LoopNativeHandoffGPT):
    """LoopNativeHandoffGPT whose attention uses rotary or learned absolute positions."""

    def __init__(self, config, repeats=4, *, position="rope", rope_base=10000.0, **kwargs):
        if position not in POSITIONS:
            raise ValueError(f"Unknown position scheme: {position}")
        if (config.n_embd // config.n_head) % 2:
            raise ValueError("Rotary embeddings need an even head dimension")
        super().__init__(config, repeats, **kwargs)
        self.position, self.rope_base = position, rope_base
        self.position_offset = 0  # A global shift; rotary outputs must not depend on it.

    def _embedded(self, tokens):
        if self.position == "absolute":
            return super()._embedded(tokens)
        return F.dropout(self.transformer.wte(tokens), self.config.embd_pdrop, self.training)

    def _round(self, x, first_round_cache=None, capture_cache=None):
        if self.position == "absolute":
            return super()._round(x, first_round_cache, capture_cache)
        if first_round_cache is not None or capture_cache is not None:
            raise ValueError("First-round KV is not part of this comparison")
        cfg, t = self.config, self.transformer
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device) + self.position_offset
        shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
        for block in t.h:
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            q, k = rotate(q, positions, self.rope_base), rotate(k, positions, self.rope_base)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
            )
            y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, cfg.n_embd))
            x = x + F.dropout(y, cfg.resid_pdrop, self.training)
            x = x + block.mlp(block.ln_2(x))
        return x
