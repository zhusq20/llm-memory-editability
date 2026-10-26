"""Ordinary causal GPT-2-style language model used by protocol v2.3."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    width: int = 768
    layers: int = 8
    heads: int = 12
    context: int = 128


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).unbind(2)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        return self.proj(output.transpose(1, 2).reshape(batch, length, width))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up = nn.Linear(width, 4 * width)
        self.activation = nn.GELU(approximate="tanh")
        self.down = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.down(self.activation(self.up(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(config.width), nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.mlp = MLP(config.width)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.ln_final = nn.LayerNorm(config.width)
        self.apply(self._initialize)
        for block in self.blocks:
            for module in (block.attention.proj, block.mlp.down):
                nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, tokens, positions=None):
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, self.token.weight)

    def config_dict(self):
        return asdict(self.config)


def select_parameters(model, scope, start=3):
    if scope not in {"all", "mlp", "down"}:
        raise ValueError("Scope must be all, mlp or down")
    selected = list(model.parameters()) if scope == "all" else []
    if scope == "mlp":
        selected = [p for block in model.blocks[start : start + 3] for p in block.mlp.parameters()]
    elif scope == "down":
        selected = [model.blocks[start + 1].mlp.down.weight]
    ids = {id(p) for p in selected}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in ids)
        parameter.grad = None
    return selected


def matmul_flops(config, batch, sequence=6, output_positions=2, backward=True):
    """Executed-shape dense/attention FLOPs estimate; excludes elementwise operations.

    Multiplication and addition each count as one. Backward uses a factor of two
    forwards, a conventional upper approximation also used for frozen parameters.
    This is not a hardware counter or a cross-editor exact FLOP measurement.
    """
    d, layers, vocab = config.width, config.layers, config.vocab_size
    forward = layers * (24 * batch * sequence * d * d + 4 * batch * sequence * sequence * d)
    forward += 2 * batch * output_positions * d * vocab
    return forward * (3 if backward else 1)
