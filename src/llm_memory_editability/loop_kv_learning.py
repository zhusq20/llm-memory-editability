"""Ordinary independent layers and first-round KV access in the audited GPT.

The KV arm keeps every current-round causal key/value and additionally attends
to the same layer's first-round causal keys/values in one softmax. No cache is
carried between examples, batches, or calls. Both branches remain differentiable.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config

from .grokking_reproduction import ReproductionGPT
from .loop_learning import architecture_manifest as base_manifest

ARMS = ("standard", "loop_shared_kv", "loop_local")


class FirstRoundKVGPT(ReproductionGPT):
    def __init__(self, config, repeats):
        super().__init__(config, repeats)
        self.disable_shared = False
        self.duplicate_current = False  # Engineering identity check, never a trained arm.
        positions = torch.arange(config.n_positions)
        causal = positions[None, :] <= positions[:, None]
        self.register_buffer("shared_causal_mask", causal, persistent=False)

    def forward(self, tokens, positions=None):
        if self.disable_shared:
            return super().forward(tokens, positions)
        batch, length = tokens.shape
        config, transformer = self.config, self.transformer
        x = transformer.wte(tokens) + transformer.wpe(torch.arange(length, device=tokens.device))
        x = F.dropout(x, config.embd_pdrop, self.training)
        first_round = []
        for round_index in range(self.repeats):
            for layer_index, block in enumerate(transformer.h):
                q, k, v = block.attn.c_attn(block.ln_1(x)).split(config.n_embd, dim=2)
                shape = (batch, length, config.n_head, config.n_embd // config.n_head)
                q, k, v = (item.view(shape).transpose(1, 2) for item in (q, k, v))
                if round_index == 0:
                    first_round.append((k, v))
                    y = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=True,
                        dropout_p=config.attn_pdrop if self.training else 0.0,
                    )
                else:
                    cached_k, cached_v = (
                        (k, v) if self.duplicate_current else first_round[layer_index]
                    )
                    mask = self.shared_causal_mask[:length, :length]
                    y = F.scaled_dot_product_attention(
                        q,
                        torch.cat((cached_k, k), dim=2),
                        torch.cat((cached_v, v), dim=2),
                        attn_mask=torch.cat((mask, mask), dim=1),
                        dropout_p=config.attn_pdrop if self.training else 0.0,
                    )
                y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, config.n_embd))
                x = x + F.dropout(y, config.resid_pdrop, self.training)
                x = x + block.mlp(block.ln_2(x))
        x = transformer.ln_f(x)
        if positions is not None:
            x = x[torch.arange(batch, device=tokens.device)[:, None], positions]
        return F.linear(x, transformer.wte.weight)


def construct(spec, device):
    arm = spec["arm"]
    if arm not in ARMS:
        raise ValueError(f"Unknown architecture: {arm}")
    base, repeats = int(spec["unique_layers"]), int(spec["repeats"])
    if base != int(spec["base_unique_blocks"]):
        raise ValueError("Stored and base layer counts must agree")
    if arm == "standard" and repeats != 1:
        raise ValueError("The ordinary baseline executes independent layers once")
    if arm != "standard" and repeats < 2:
        raise ValueError("Loop comparison requires at least two rounds")
    reference = int(spec["initialization_reference_layers"])
    if base < 1 or reference < base:
        raise ValueError("Reference initialization must include every selected layer")
    options = spec["model"]
    if options["hidden_size"] % options["attention_heads"]:
        raise ValueError("Hidden width must be divisible by the attention head count")
    config = GPT2Config(
        vocab_size=options["vocab_size"],
        n_positions=options["positions"],
        n_embd=options["hidden_size"],
        n_head=options["attention_heads"],
        n_layer=reference,
        resid_pdrop=options["dropout"],
        embd_pdrop=options["dropout"],
        attn_pdrop=options["dropout"],
        activation_function="gelu_new",
        use_cache=False,
    )
    torch.manual_seed(spec["initialization"])
    model = (
        FirstRoundKVGPT(config, repeats)
        if arm == "loop_shared_kv"
        else ReproductionGPT(config, repeats)
    )
    # These are separately initialized layers of the reference, never copies.
    model.transformer.h = nn.ModuleList(list(model.transformer.h)[:base])
    model.config.n_layer = base
    model.loop_learning_info = {
        "arm": arm,
        "base_unique_blocks": base,
        "logical_repeats": repeats,
        "initialization_reference_layers": reference,
        "initialization": int(spec["initialization"]),
        "initialization_policy": "independent random reference layers; selected prefix",
        "kv_access": "same-layer first round plus full current causal prefix"
        if arm == "loop_shared_kv"
        else "full current causal prefix",
        "kv_gradient": "connected",
        "kv_persistence": "one forward call only",
        "kv_window": None,
        "softmax_branches": "joint",
        "initial_forward_equality_required": False,
    }
    return model.to(device)


def architecture_manifest(model):
    result = base_manifest(model)
    extra = (
        len(model.transformer.h) * (model.repeats - 1) if isinstance(model, FirstRoundKVGPT) else 0
    )
    return {
        **result,
        "attention_pair_multiples_per_sequence_square": result["executed_blocks"] + extra,
        "cost_contract": "actual padded batches; shared attention pairs included; matmul estimate",
    }


def extra_attention_flops(model, records, microbatch):
    """Count additional dense QK and AV pairs, including padding and backward."""
    if not isinstance(model, FirstRoundKVGPT) or model.disable_shared:
        return 0
    pairs = 0
    for start in range(0, len(records), microbatch):
        chunk = records[start : start + microbatch]
        length = math.ceil(max(len(r["encoded"]["input"]) for r in chunk) / 8) * 8
        pairs += len(chunk) * length * length
    return 12 * model.config.n_embd * len(model.transformer.h) * (model.repeats - 1) * pairs
