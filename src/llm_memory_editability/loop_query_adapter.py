"""A shared, stateless nonlinear bottleneck added only to attention queries.

Every physical block recomputes its adapter from its own pre-attention LN output.
The two matrices are shared across blocks and rounds, matching the controller's
parameter-sharing scope. There is no independent recurrent state or input skip.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .grokking_reproduction import ReproductionGPT


class QueryAdapterGPT(ReproductionGPT):
    def __init__(self, config, repeats=2, adapter_dim=48, adapter_seed=20261016):
        if type(adapter_dim) is not int or not 0 < adapter_dim <= config.n_embd:
            raise ValueError("adapter_dim must be a positive integer no larger than content width")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        super().__init__(config, repeats)
        self.adapter_dim = adapter_dim
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(adapter_seed)
            self.adapter_down = nn.Linear(config.n_embd, adapter_dim, bias=False)
            self.adapter_up = nn.Linear(adapter_dim, config.n_embd, bias=False)
            nn.init.normal_(self.adapter_down.weight, std=config.initializer_range)
            nn.init.zeros_(self.adapter_up.weight)

    def _round(self, x):
        cfg, t = self.config, self.transformer
        batch, length, width = x.shape
        for block in t.h:
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(width, dim=2)
            q = q + self.adapter_up(torch.tanh(self.adapter_down(z)))
            shape = (batch, length, cfg.n_head, width // cfg.n_head)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
            )
            y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, width))
            x = x + F.dropout(y, cfg.resid_pdrop, self.training)
            x = x + block.mlp(block.ln_2(x))
        return x

    def forward_with_states(self, tokens, positions=None):
        t = self.transformer
        x = t.wte(tokens) + t.wpe(torch.arange(tokens.shape[1], device=tokens.device))
        x = F.dropout(x, self.config.embd_pdrop, self.training)
        first = None
        for index in range(self.repeats):
            x = self._round(x)
            if index == 0:
                first = x
        final = t.ln_f(x)
        if positions is not None:
            final = final[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(final, t.wte.weight), first, x

    def forward(self, tokens, positions=None):
        return self.forward_with_states(tokens, positions)[0]


def adapter_accounting(spec, width, adapter_dim, sequence=4):
    """Dense matrix FLOPs, training=3*forward; recomputed at EVERY block execution."""
    if min(spec["repeats"], spec["unique_layers"], width, adapter_dim, sequence) < 1:
        raise ValueError("Invalid accounting dimensions")
    forward = 4 * sequence * width * adapter_dim * spec["unique_layers"] * spec["repeats"]
    return {
        "additional_parameters": 2 * width * adapter_dim,
        "independent_control_state_elements": 0,
        "adapter_intermediate_elements_per_layer_sequence": sequence * adapter_dim,
        "additional_inference_matmul_flops_per_sequence": forward,
        "additional_training_matmul_flops_per_step": 3 * spec["batch_size"] * forward,
    }
