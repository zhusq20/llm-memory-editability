"""Shared LoopGPT backbone with independently controlled residual and KV access.

The single/gamma=1 numerical path delegates to the historical shared-cache
implementation. Eight unique controllers are reused across every loop; streams
and the first-loop per-layer KV are carried through the complete unroll.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .grok_depth import make_optimizer as ordinary_optimizer
from .grok_loop_model import _positive_integer
from .latent_scaling import construct as ordinary_construct
from .parametric_architecture import HyperConnection, residual_mix, sinkhorn
from .shared_cache import SharedCacheGPT
from .shared_cache import executed_flops as cache_flops

KINDS = ("single", "identity_mhc", "mhc")
_compiled_sinkhorn = None


def training_sinkhorn(logits):
    global _compiled_sinkhorn
    if _compiled_sinkhorn is None:
        _compiled_sinkhorn = torch.compile(
            sinkhorn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
    return _compiled_sinkhorn(logits)


class LoopHyperConnection(HyperConnection):
    def coefficients(self, x):
        n = self.streams
        with torch.autocast(x.device.type, enabled=False):
            dynamic = self.projection(self.norm(x.float().flatten(-2)))
            pre = torch.sigmoid(dynamic[..., :n] * self.alpha_pre + self.pre_bias) + 1e-6
            post = 2 * torch.sigmoid(dynamic[..., n : 2 * n] * self.alpha_post + self.post_bias)
            matrix = None
            if not self.identity:
                logits = dynamic[..., 2 * n :].reshape(*x.shape[:-2], n, n)
                logits = logits * self.alpha_res + self.res_bias
                projection = training_sinkhorn if self.training and x.is_cuda else sinkhorn
                matrix = projection(logits)
        return pre, post, matrix


class ResidualCacheGPT(SharedCacheGPT):
    def __init__(
        self,
        config,
        repeats,
        dropout=0.0,
        arm="local",
        window=2,
        residual_kind="single",
        gamma=1.0,
        controller_seed=107260751,
    ):
        super().__init__(config, repeats, dropout, arm, window)
        if residual_kind not in KINDS or gamma <= 0:
            raise ValueError("Invalid residual kind or branch scale")
        self.residual_kind, self.gamma = residual_kind, float(gamma)
        self.diagnostic_records = None
        if residual_kind != "single":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(controller_seed)
                self.connections = nn.ModuleList(
                    LoopHyperConnection(config.width, i, identity=residual_kind == "identity_mhc")
                    for i in range(2 * config.layers)
                )

    def _record(self, x, update, recursion, layer, branch, matrix=None):
        if self.diagnostic_records is None:
            return
        xf, uf = x.detach().float(), update.detach().float()
        if self.residual_kind == "single":
            mean, contrast = xf, xf.new_zeros(())
        else:
            mean = xf.mean(-2)
            contrast = (xf - mean.unsqueeze(-2)).square().mean()
        record = dict(
            loop=recursion + 1,
            layer=layer,
            branch=branch,
            mean_square=float(mean.square().mean()),
            coordinate_variance=float(mean.var(-1, unbiased=False).mean()),
            stream_contrast_square=float(contrast),
            update_square=float(uf.square().mean()),
        )
        if matrix is not None:
            a = matrix.detach().float()
            previous = getattr(self, "diagnostic_product", None)
            product = a if previous is None else previous @ a
            self.diagnostic_product = product
            record.update(
                max_row_error=float((a.sum(-1) - 1).abs().max()),
                max_column_error=float((a.sum(-2) - 1).abs().max()),
                max_spectral_norm=float(torch.linalg.matrix_norm(a, ord=2).max()),
                product_max_row_error=float((product.sum(-1) - 1).abs().max()),
                product_max_column_error=float((product.sum(-2) - 1).abs().max()),
                product_max_spectral_norm=float(torch.linalg.matrix_norm(product, ord=2).max()),
            )
        self.diagnostic_records.append(record)

    def _connection(self, x, sublayer, connection, recursion, layer, branch):
        pre, post, matrix = connection.coefficients(x)
        with torch.autocast(x.device.type, enabled=False):
            contracted = (pre[..., None] * x.float()).sum(-2)
        y = sublayer(contracted.to(x.dtype))
        with torch.autocast(x.device.type, enabled=False):
            residual = x.float() if matrix is None else residual_mix(x.float(), matrix)
            update = post[..., None] * (self.gamma * y.float()).unsqueeze(-2)
            output = residual + update
        self._record(output, update, recursion, layer, branch, matrix)
        return output.to(x.dtype)

    def forward(self, tokens, positions=None, repeats=None):
        count = self.repeats if repeats is None else _positive_integer(repeats, "repeats")
        if self.residual_kind == "single" and self.gamma == 1 and self.diagnostic_records is None:
            return super().forward(tokens, positions, count)
        length = tokens.shape[1]
        if length > self.config.context:
            raise ValueError("Input exceeds mask context")
        p = self.dropout if self.training else 0.0
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        x = F.dropout(x, p=p, training=self.training)
        if self.residual_kind != "single":
            x = x.unsqueeze(-2).expand(*x.shape[:-1], 4, x.shape[-1])
        causal = self.shared_causal_mask[:length, :length]
        joined_mask = torch.cat((causal, causal), dim=-1)
        shared = []
        for recursion in range(count):
            for layer, block in enumerate(self.blocks):

                def attention_branch(state, block=block, recursion=recursion, layer=layer):
                    z = block.ln1(state)
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
                    if recursion == 0 or self.memory_arm == "local" or self.disable_shared:
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
                    else:
                        first_k, first_v = (k, v) if self.duplicate_current else shared[layer]
                        y = F.scaled_dot_product_attention(
                            q,
                            torch.cat((first_k, k), dim=-2),
                            torch.cat((first_v, v), dim=-2),
                            attn_mask=joined_mask,
                            dropout_p=p,
                        )
                    y = attention.proj(y.transpose(1, 2).reshape(batch, size, width))
                    return F.dropout(y, p=p, training=self.training)

                def mlp_branch(state, block=block):
                    return F.dropout(block.mlp(block.ln2(state)), p=p, training=self.training)

                if self.residual_kind == "single":
                    for branch, sublayer in [("attention", attention_branch), ("mlp", mlp_branch)]:
                        update = self.gamma * sublayer(x)
                        x = x + update
                        self._record(x, update, recursion, layer, branch)
                else:
                    x = self._connection(
                        x,
                        attention_branch,
                        self.connections[2 * layer],
                        recursion,
                        layer,
                        "attention",
                    )
                    x = self._connection(
                        x, mlp_branch, self.connections[2 * layer + 1], recursion, layer, "mlp"
                    )
        if self.residual_kind != "single":
            x = x.mean(-2)
        x = self.ln_final(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, self.token.weight)


def construct(spec, device):
    reference = ordinary_construct(spec, "cpu")
    torch.manual_seed(spec["initialization"])
    return ResidualCacheGPT(
        reference.config,
        spec["repeats"],
        spec["dropout"],
        spec.get("memory_arm", "local"),
        spec.get("window", 2),
        spec.get("residual_kind", "single"),
        spec.get("gamma", 1),
        spec.get("controller_initialization_seed", 107260751),
    ).to(device)


def make_optimizer(model, lr, weight_decay):
    if model.residual_kind == "single":
        return ordinary_optimizer(model, lr, weight_decay)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        # res_bias is a matrix but, like the other controller biases, has no decay.
        group = no_decay if parameter.ndim < 2 or name.endswith(".res_bias") else decay
        group.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0}],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
        capturable=True,
    )


def executed_flops(config, repeats, batch, sequence, arm, kind, output_positions=9, backward=True):
    """Historical matmul ledger plus controller projections and stream mixing.

    Sinkhorn and elementwise operations are excluded and measured in wall time.
    The inherited three-forward backward convention is an approximation.
    """
    base = cache_flops(config, repeats, batch, sequence, arm, output_positions, backward)
    if kind == "single":
        return base
    n, d = 4, config.width
    count = 2 * n + (n * n if kind == "mhc" else 0)
    controllers = 2 * config.layers * repeats
    per_token = 2 * n * d * count + (2 * n * n * d if kind == "mhc" else 0)
    return base + batch * sequence * controllers * per_token * (3 if backward else 1)


def install_training_adapter(spec):
    from . import latent_scaling as latent

    latent.construct = construct
    latent.make_optimizer = make_optimizer
    latent.flops = lambda config, repeats, batch, sequence, output_positions=9, backward=True: (
        executed_flops(
            config,
            repeats,
            batch,
            sequence,
            spec.get("memory_arm", "local"),
            spec.get("residual_kind", "single"),
            output_positions,
            backward,
        )
    )
    return latent
