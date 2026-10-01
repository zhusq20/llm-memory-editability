"""GPT-2 blocks, quarter-head NeoX RoPE, tied vocabulary, explicit Q/V/embedding LoRA."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["width"]
        self.heads = cfg["heads"]
        self.dim = width // self.heads
        self.rotary_dim = int(self.dim * cfg["rotary_fraction"])
        assert self.rotary_dim % 2 == 0 and self.rotary_dim > 0
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        freq = 1.0 / (
            cfg["rotary_base"] ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
        )
        angle = torch.outer(torch.arange(cfg["context"]).float(), freq)
        angle = torch.cat((angle, angle), dim=-1)[None, None]
        self.register_buffer("cos", angle.cos(), persistent=False)
        self.register_buffer("sin", angle.sin(), persistent=False)
        self.dropout = cfg["dropout"]
        self.rank = 0

    def add_lora(self, rank):
        self.rank = rank
        width = self.heads * self.dim
        self.q_a = nn.Parameter(torch.randn(rank, width) * 0.02)
        self.q_b = nn.Parameter(torch.zeros(width, rank))
        self.v_a = nn.Parameter(torch.randn(rank, width) * 0.02)
        self.v_b = nn.Parameter(torch.zeros(width, rank))

    def rotate(self, x):
        z, tail = x[..., : self.rotary_dim], x[..., self.rotary_dim :]
        half = self.rotary_dim // 2
        swapped = torch.cat((-z[..., half:], z[..., :half]), dim=-1)
        size = x.shape[-2]
        z = z * self.cos[:, :, :size].to(x.dtype) + swapped * self.sin[:, :, :size].to(x.dtype)
        return torch.cat((z, tail), dim=-1)

    def forward(self, x):
        b, t, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if self.rank:
            q = q + F.linear(F.linear(x, self.q_a), self.q_b)
            v = v + F.linear(F.linear(x, self.v_a), self.v_b)
        q, k, v = [z.view(b, t, self.heads, self.dim).transpose(1, 2) for z in (q, k, v)]
        y = F.scaled_dot_product_attention(
            self.rotate(q),
            self.rotate(k),
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0,
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, width))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["width"]
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = Attention(cfg)
        self.up, self.down = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        return x + self.dropout(self.down(F.gelu(self.up(self.ln2(x)), approximate="tanh")))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["width"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(cfg["width"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.embedding_rank = 0
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=0.02 / math.sqrt(2 * cfg["layers"]))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * cfg["layers"]))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.bias)

    def add_lora(self, qv_rank, embedding_rank):
        self.requires_grad_(False)
        for b in self.blocks:
            b.attn.add_lora(qv_rank)
        self.embedding_rank = embedding_rank
        self.emb_a = nn.Parameter(torch.randn(self.cfg["vocab_size"], embedding_rank) * 0.02)
        self.emb_b = nn.Parameter(torch.zeros(embedding_rank, self.cfg["width"]))

    def hidden(self, ids):
        x = self.embedding(ids)
        if self.embedding_rank:
            x = x + F.embedding(ids, self.emb_a) @ self.emb_b
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def logits(self, hidden):
        # Preserve input/output tying also for the embedding update.
        logits = F.linear(hidden, self.embedding.weight)
        if self.embedding_rank:
            logits = logits + F.linear(F.linear(hidden, self.emb_b), self.emb_a)
        return logits

    def forward(self, ids, targets=None):
        h = self.hidden(ids)
        if targets is None:
            return self.logits(h)
        # Avoid materializing logits at padded/unsupervised positions.
        valid = targets != -100
        return F.cross_entropy(self.logits(h[valid]).float(), targets[valid], reduction="mean")
