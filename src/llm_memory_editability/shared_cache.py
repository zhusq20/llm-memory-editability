"""First-recursion KV access in the historical closed-book LoopGPT backbone.

Only attention access changes. Each layer retains its own first-pass K/V;
embeddings, normalization, residuals, MLPs and supervision are unchanged.
"""

from __future__ import annotations

from numbers import Integral

import torch
from torch.nn import functional as F

from .grok_loop_model import LoopGPT, _positive_integer
from .grok_loop_model import flops as ordinary_flops
from .latent_scaling import construct as ordinary_construct

ARMS = ("local", "shared_full", "shared_window", "shared_detached")


class SharedCacheGPT(LoopGPT):
    def __init__(self, config, repeats, dropout=0.0, arm="shared_full", window=2):
        if arm not in ARMS:
            raise ValueError(f"Unknown memory arm: {arm}")
        if isinstance(window, bool) or not isinstance(window, Integral) or window < 0:
            raise ValueError("window counts nonnegative preceding local positions")
        super().__init__(config, repeats, dropout, "legacy_unique")
        self.memory_arm, self.window = arm, int(window)
        self.disable_shared = False
        self.duplicate_current = False  # engineering identity control only
        positions = torch.arange(config.context)
        causal = positions[None, :] <= positions[:, None]
        local = causal & (positions[None, :] >= positions[:, None] - self.window)
        self.register_buffer("shared_causal_mask", causal, persistent=False)
        self.register_buffer("local_window_mask", local, persistent=False)

    def forward(self, tokens, positions=None, repeats=None):
        count = self.repeats if repeats is None else _positive_integer(repeats, "repeats")
        if self.memory_arm == "local" or count == 1:
            return super().forward(tokens, positions, count)
        length = tokens.shape[1]
        if length > self.config.context:
            raise ValueError("Input exceeds mask context")
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
        causal = self.shared_causal_mask[:length, :length]
        local_mask = (
            self.local_window_mask[:length, :length]
            if self.memory_arm == "shared_window"
            else causal
        )
        joined_mask = torch.cat((causal, local_mask), dim=-1)
        shared = []
        for recursion in range(count):
            for layer, block in enumerate(self.blocks):
                z = block.ln1(x)
                batch, size, width = z.shape
                attention = block.attention
                q, k, v = (
                    attention.qkv(z)
                    .view(batch, size, 3, attention.heads, width // attention.heads)
                    .unbind(2)
                )
                q, k, v = (t.transpose(1, 2) for t in (q, k, v))
                if recursion == 0:
                    shared.append((k, v))
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
                elif self.disable_shared:
                    y = F.scaled_dot_product_attention(q, k, v, attn_mask=local_mask, dropout_p=p)
                else:
                    first_k, first_v = (k, v) if self.duplicate_current else shared[layer]
                    if self.memory_arm == "shared_detached":
                        first_k, first_v = first_k.detach(), first_v.detach()
                    y = F.scaled_dot_product_attention(
                        q,
                        torch.cat((first_k, k), dim=-2),
                        torch.cat((first_v, v), dim=-2),
                        attn_mask=joined_mask,
                        dropout_p=p,
                    )
                y = attention.proj(y.transpose(1, 2).reshape(batch, size, width))
                x = x + F.dropout(y, p=p, training=self.training)
                x = x + F.dropout(block.mlp(block.ln2(x)), p=p, training=self.training)
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, self.token.weight)


def construct(spec, device):
    if "memory_arm" not in spec:
        return ordinary_construct(spec, device)
    reference = ordinary_construct(spec, "cpu")
    torch.manual_seed(spec["initialization"])
    return SharedCacheGPT(
        reference.config, spec["repeats"], spec["dropout"], spec["memory_arm"], spec["window"]
    ).to(device)


def executed_flops(config, repeats, batch, sequence, arm, output_positions=9, backward=True):
    """Executed dense matmul shapes, including masked work; excludes elementwise ops.

    Concatenated attention has twice the key dimension, even for a small window.
    The ordinary estimate counts QK and AV over a square sequence, not only the
    visible causal entries. Backward follows the historical 3-forward convention.
    Detachment changes true backward work; this common estimate is approximate.
    """
    base = ordinary_flops(config, repeats, batch, sequence, output_positions, backward)
    if arm == "local":
        return base
    extra = 4 * batch * config.layers * (repeats - 1) * sequence**2 * config.width
    return base + extra * (3 if backward else 1)


def cached_positions(layers, repeats, length, arm, window):
    """Analytic decoding history slots, not measured training allocation."""
    if arm in {"local", "shared_full", "shared_detached"}:
        return layers * repeats * length
    return layers * (length + (repeats - 1) * min(window, length))


def install_training_adapter(spec):
    """Reuse the unchanged historical sampler, optimizer, scorer and audit."""
    from . import latent_scaling as latent

    latent.construct = construct
    latent.flops = lambda config, repeats, batch, sequence, output_positions=9, backward=True: (
        executed_flops(
            config, repeats, batch, sequence, spec["memory_arm"], output_positions, backward
        )
    )
    return latent
