"""Apply a saved MLP write only at chosen token positions (zero training).

The native reader forward is repeated here with one change: each MLP evaluates both the
current parameters and a second parameter set, and keeps the second only where a boolean
position mask is true. With an all-false mask the output equals the reader; with an all-true
mask it equals the reader after loading the second set. Loop rounds share the mask, so a
position uses the new write in every round or in none.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

MLP_KEYS = ("c_fc.weight", "c_fc.bias", "c_proj.weight", "c_proj.bias")


def layer_values(values, layers):
    """Group a saved {name: tensor} MLP state into per-layer tuples in MLP_KEYS order."""
    out = []
    for index in range(layers):
        prefix = f"transformer.h.{index}.mlp."
        out.append(tuple(values[prefix + key] for key in MLP_KEYS))
    if len(values) != 4 * layers:
        raise ValueError("Saved MLP state must contain exactly four tensors per physical layer")
    return out


def _mlp(block, x, params):
    if params is None:
        return block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(x)))
    fc_w, fc_b, proj_w, proj_b = params
    return torch.addmm(proj_b, block.mlp.act(torch.addmm(fc_b, x.flatten(0, 1), fc_w)), proj_w)


@torch.no_grad()
def masked_logits(reader, tokens, positions, new, mask):
    """Scores at `positions`; MLP layer l uses `new[l]` where `mask` is true."""
    model, cfg = reader.backbone, reader.backbone.config
    if reader.training or model.training:
        raise ValueError("Evaluation only")
    batch, length = tokens.shape
    if mask.shape != (batch, length) or mask.dtype != torch.bool:
        raise ValueError("One boolean mask entry per token is required")
    t = model.transformer
    x = t.wte(tokens) + t.wpe(torch.arange(length, device=tokens.device))
    keep = mask[..., None]
    for round_index in range(model.repeats):
        for layer, block in enumerate(t.h):
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
            shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            u = x + block.attn.c_proj(out.transpose(1, 2).reshape(batch, length, cfg.n_embd))
            h = block.ln_2(u)
            old = _mlp(block, h, None)
            fresh = _mlp(block, h, new[layer]).view_as(old)
            x = u + torch.where(keep, fresh, old)
        if getattr(model, "discoloop", False) and round_index < model.repeats - 1:
            x = model._discoloop_boundary(x)
    x = t.ln_f(x)
    x = x[torch.arange(batch, device=tokens.device)[:, None], positions]
    return F.linear(x, t.wte.weight)
