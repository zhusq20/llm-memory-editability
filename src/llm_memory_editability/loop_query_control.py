"""A token-local recurrent controller that only adds to attention queries.

The original GPT-2 pre-LN blocks, residuals, MLPs, dropout and tied readout are
retained. One small controller and one query map are shared across ALL physical
layers and rounds. The controller is held fixed within a round and updated only
between rounds. It has no persistent state, labels, teacher trajectory or pool.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .grokking_reproduction import ReproductionGPT


class QueryControlGPT(ReproductionGPT):
    def __init__(self, config, repeats=2, control_dim=32, control_seed=20261016):
        if type(control_dim) is not int or not 0 < control_dim <= config.n_embd:
            raise ValueError("control_dim must be a positive integer no larger than content width")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        super().__init__(config, repeats)
        self.control_dim = control_dim
        # Preserve the caller's RNG after constructing the original backbone.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(control_seed)
            self.control_input = nn.Linear(config.n_embd, control_dim, bias=False)
            self.control_content = nn.Linear(config.n_embd, control_dim, bias=False)
            self.control_recurrent = nn.Linear(control_dim, control_dim, bias=False)
            self.control_query = nn.Linear(control_dim, config.n_embd, bias=False)
            for module in (self.control_input, self.control_content, self.control_recurrent):
                nn.init.normal_(module.weight, std=config.initializer_range)
            nn.init.zeros_(self.control_query.weight)

    @staticmethod
    def _rms(x):
        # Parameter-free, token-local normalization ONLY on controller inputs.
        value = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)

    def update_control(self, control, content, input_projection):
        return torch.tanh(
            self.control_recurrent(control)
            + self.control_content(self._rms(content))
            + input_projection
        )

    def _round(self, x, query_delta):
        cfg, t = self.config, self.transformer
        batch, length, width = x.shape
        for block in t.h:
            q, k, v = block.attn.c_attn(block.ln_1(x)).split(width, dim=2)
            q = q + query_delta
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
        input_projection = self.control_input(self._rms(x))
        control = torch.tanh(input_projection)
        first = None
        for index in range(self.repeats):
            # The same map and delta are used in every physical layer this round.
            x = self._round(x, self.control_query(control))
            if index == 0:
                first = x
            if index + 1 < self.repeats:
                control = self.update_control(control, x, input_projection)
        final = t.ln_f(x)
        if positions is not None:
            final = final[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(final, t.wte.weight), first, x

    def forward(self, tokens, positions=None):
        return self.forward_with_states(tokens, positions)[0]


def control_accounting(spec, width, control_dim, sequence=4):
    """Dense matrix FLOPs (training=3*forward); excludes tanh/norm/additions."""
    rounds = spec["repeats"]
    if rounds < 1 or not math.isfinite(width) or control_dim < 1:
        raise ValueError("Invalid accounting dimensions")
    elements = 3 * width * control_dim + control_dim**2
    forward_per_token = 2 * (2 * rounds * width * control_dim + (rounds - 1) * control_dim**2)
    return {
        "additional_parameters": elements,
        "control_elements_per_sequence": sequence * control_dim,
        "additional_inference_matmul_flops_per_sequence": sequence * forward_per_token,
        "additional_training_matmul_flops_per_step": (
            3 * spec["batch_size"] * sequence * forward_per_token
        ),
    }
