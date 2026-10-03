"""Paired GPT-2 architecture extensions for the closed-book memory experiment.

Dense execution delegates to the historical ReproductionGPT without changing
its numerical path. New modules use an independent, checkpointed initialization
seed; the eight-block reference is always constructed before taking a prefix.
The mHC residual convention is ``output[n] = sum_m H_res[m,n] * input[m]``.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from .grokking_reproduction import ReproductionGPT

ARMS = {
    "D4": (4, 1, "dense"),
    "D8": (8, 1, "dense"),
    "L4R2": (4, 2, "dense"),
    "M8": (8, 1, "moe"),
    "W8": (8, 1, "wide"),
    "IHC8": (8, 1, "identity_mhc"),
    "HC8": (8, 1, "mhc"),
    "M4": (4, 1, "moe"),
    "LM4R2": (4, 2, "moe"),
}


def sinkhorn(logits: torch.Tensor, iterations: int = 20, eps: float = 1e-6):
    """Accepted mHC A.6: initial row softmax, then 20 column normalizations.

    There are 19 intervening row normalizations. This is a finite-iteration
    approximation, not an exact doubly stochastic projection.
    """
    if iterations < 1:
        raise ValueError("Sinkhorn requires at least one iteration")
    with torch.autocast(logits.device.type, enabled=False):
        matrix = logits.float().softmax(dim=-1) + eps
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
        for _ in range(iterations - 1):
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def residual_mix(streams, matrix):
    """Rows are source streams and columns are destination streams."""
    return torch.einsum("...mn,...mc->...nc", matrix, streams)


class ControllerRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-20) * self.weight


class HyperConnection(nn.Module):
    """Four-stream pre/post controller with optional constrained residual map."""

    def __init__(self, width, sublayer_id, streams=4, identity=False):
        super().__init__()
        self.streams, self.identity = streams, identity
        count = 2 * streams + (0 if identity else streams * streams)
        self.norm = ControllerRMSNorm(streams * width)
        self.projection = nn.Linear(streams * width, count, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.pre_bias = nn.Parameter(torch.full((streams,), -3.0))
        with torch.no_grad():
            self.pre_bias[sublayer_id % streams] = 3.0
        self.post_bias = nn.Parameter(torch.zeros(streams))
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        if not identity:
            self.res_bias = nn.Parameter(6 * torch.eye(streams) - 3)
            self.alpha_res = nn.Parameter(torch.tensor(0.01))

    def coefficients(self, x):
        n = self.streams
        with torch.autocast(x.device.type, enabled=False):
            dynamic = self.projection(self.norm(x.float().flatten(-2)))
            pre = torch.sigmoid(dynamic[..., :n] * self.alpha_pre + self.pre_bias) + 1e-6
            post = 2 * torch.sigmoid(dynamic[..., n : 2 * n] * self.alpha_post + self.post_bias)
            res = None
            if not self.identity:
                logits = dynamic[..., 2 * n :].reshape(*x.shape[:-2], n, n)
                res = sinkhorn(logits * self.alpha_res + self.res_bias)
        return pre, post, res

    def forward(self, x, sublayer):
        pre, post, res = self.coefficients(x)
        with torch.autocast(x.device.type, enabled=False):
            contracted = (pre[..., None] * x.float()).sum(dim=-2)
        y = sublayer(contracted.to(x.dtype))
        with torch.autocast(x.device.type, enabled=False):
            residual = x.float() if res is None else residual_mix(x.float(), res)
            output = residual + post[..., None] * y.float().unsqueeze(-2)
        return output.to(x.dtype)


def _initialize_mlp(mlp, initializer_range):
    # Eight-block GPT-2 reference residual projection scale, also for loops.
    nn.init.normal_(mlp.c_fc.weight, std=initializer_range)
    nn.init.normal_(mlp.c_proj.weight, std=initializer_range / math.sqrt(2 * 8))
    nn.init.zeros_(mlp.c_fc.bias)
    nn.init.zeros_(mlp.c_proj.bias)


class DroplessMoE(nn.Module):
    """Top-2 normalized routing, one output dropout, no capacity/token drops."""

    def __init__(self, config, experts=8, top_k=2):
        super().__init__()
        self.expert_count, self.top_k = experts, top_k
        self.router = nn.Linear(config.n_embd, experts, bias=False)
        nn.init.normal_(self.router.weight, std=config.initializer_range)
        self.experts = nn.ModuleList()
        for _ in range(experts):
            expert = GPT2MLP(2 * config.n_embd, config)
            expert.dropout = nn.Identity()
            _initialize_mlp(expert, config.initializer_range)
            self.experts.append(expert)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, valid_mask=None):
        flat = x.reshape(-1, x.shape[-1])
        if valid_mask is None:
            selected_tokens = torch.arange(len(flat), device=x.device)
        else:
            selected_tokens = valid_mask.flatten().nonzero(as_tuple=True)[0]
        valid = flat.index_select(0, selected_tokens)
        with torch.autocast(x.device.type, enabled=False):
            probabilities = F.linear(valid.float(), self.router.weight.float()).softmax(-1)
            top_weights, expert_ids = probabilities.topk(self.top_k, dim=-1)
            weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        # Accumulate selected expert outputs in FP32 for consistent BF16 routing.
        combined = torch.zeros_like(valid, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = (expert_ids == expert_id).nonzero(as_tuple=True)
            if token_ids.numel():
                values = expert(valid.index_select(0, token_ids))
                weighted = values.float() * weights[token_ids, slots, None]
                combined = combined.index_add(0, token_ids, weighted)
        output = torch.zeros_like(flat).index_copy(0, selected_tokens, combined.to(flat.dtype))
        counts = torch.bincount(expert_ids.flatten(), minlength=self.expert_count).detach()
        n_valid = probabilities.new_tensor(len(valid))
        stats = {
            "assignment_counts": counts,
            "probability_sum": probabilities.sum(dim=0),
            "valid_tokens": n_valid,
            "top_k": self.top_k,
            "processed_assignments": counts.sum(),
            "dropped_assignments": counts.new_zeros(()),
            "entropy_sum": -(probabilities * probabilities.clamp_min(1e-30).log()).sum().detach(),
            "selected_mass_sum": top_weights.sum().detach(),
        }
        return self.dropout(output.view_as(x)), stats


def combine_router_fractions(stats_by_microbatch: Sequence[Sequence[dict]]):
    """Combine detached counts before a replay that computes exact batch aux.

    Counts are per executed block (thus distinguish Loop passes). Replay must
    restore dropout RNG state, so its hard routes equal the counting pass.
    """
    if not stats_by_microbatch or not stats_by_microbatch[0]:
        return [], 0
    layers = len(stats_by_microbatch[0])
    if any(len(stats) != layers for stats in stats_by_microbatch):
        raise ValueError("All microbatches must have the same executed routing layers")
    total = sum(stats[0]["valid_tokens"] for stats in stats_by_microbatch)
    fractions = []
    for layer in range(layers):
        counts = sum(stats[layer]["assignment_counts"] for stats in stats_by_microbatch)
        fractions.append(
            counts.float() / (total * stats_by_microbatch[0][layer]["top_k"]).clamp_min(1)
        )
    return fractions, total


def router_balance_loss(stats, assignment_fractions=None, total_valid_tokens=None):
    """N * sum(f * p), averaged over executed layers.

    With externally supplied whole-batch fractions and denominator, this returns
    the current microbatch's additive contribution to the exact batch objective.
    Assignment fractions are discrete and intentionally detached.
    """
    if not stats:
        return torch.tensor(0.0)
    if assignment_fractions is not None and len(assignment_fractions) != len(stats):
        raise ValueError("One assignment-fraction vector is required per executed layer")
    losses = []
    for index, record in enumerate(stats):
        count = record["valid_tokens"]
        denominator = count if total_valid_tokens is None else total_valid_tokens
        denominator = torch.as_tensor(denominator, device=count.device).clamp_min(1)
        fractions = (
            record["assignment_counts"].float() / (count * record["top_k"]).clamp_min(1)
            if assignment_fractions is None
            else assignment_fractions[index].detach()
        )
        probabilities = record["probability_sum"] / denominator
        losses.append(probabilities.numel() * (fractions * probabilities).sum())
    return torch.stack(losses).mean()


class ParametricGPT(ReproductionGPT):
    def __init__(self, config, architecture="D8", module_seed=0):
        blocks, repeats, kind = ARMS[architecture]
        # Always initialize the same independent eight blocks before slicing.
        config.n_layer = 8
        super().__init__(config, repeats=repeats)
        self.transformer.h = nn.ModuleList(list(self.transformer.h)[:blocks])
        self.config.n_layer = blocks
        self.architecture, self.kind, self.module_seed = architecture, kind, module_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(module_seed)
            if kind in {"wide", "moe"}:
                for block in self.transformer.h:
                    if kind == "moe":
                        block.mlp = DroplessMoE(config)
                    else:
                        block.mlp = GPT2MLP(16 * config.n_embd, config)
                        _initialize_mlp(block.mlp, config.initializer_range)
            elif kind in {"mhc", "identity_mhc"}:
                self.connections = nn.ModuleList(
                    HyperConnection(config.n_embd, index, identity=kind == "identity_mhc")
                    for index in range(2 * blocks)
                )

    def _attention(self, block, x):
        cfg = self.config
        batch, length = x.shape[:2]
        q, k, v = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
        shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
        q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
        )
        y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, cfg.n_embd))
        return F.dropout(y, cfg.resid_pdrop, self.training)

    def forward(self, tokens, positions=None, *, valid_mask=None, return_aux=False):
        if valid_mask is not None and (
            valid_mask.shape != tokens.shape or valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be boolean and have the tokens' shape")
        if self.kind in {"dense", "wide"}:
            logits = super().forward(tokens, positions)
            if return_aux:
                return logits, {"balance_loss": logits.new_zeros(()), "router_stats": []}
            return logits
        cfg, t = self.config, self.transformer
        length = tokens.shape[1]
        x = t.wte(tokens) + t.wpe(torch.arange(length, device=tokens.device))
        x = F.dropout(x, cfg.embd_pdrop, self.training)
        stats = []
        if self.kind in {"mhc", "identity_mhc"}:
            x = x.unsqueeze(-2).expand(-1, -1, 4, -1)
        for _ in range(self.repeats):
            for index, block in enumerate(t.h):
                if self.kind == "moe":
                    x = x + self._attention(block, x)
                    y, layer_stats = block.mlp(block.ln_2(x), valid_mask)
                    x = x + y
                    stats.append(layer_stats)
                else:
                    x = self.connections[2 * index](x, lambda z, b=block: self._attention(b, z))
                    x = self.connections[2 * index + 1](x, lambda z, b=block: b.mlp(b.ln_2(z)))
        if self.kind in {"mhc", "identity_mhc"}:
            x = x.mean(dim=-2)
        x = t.ln_f(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        logits = F.linear(x, t.wte.weight)
        if return_aux:
            loss = router_balance_loss(stats) if stats else logits.new_zeros(())
            return logits, {"balance_loss": loss, "router_stats": stats}
        return logits


def construct(config, spec, device):
    """Historical config/spec convention, extended by ``spec['architecture']``."""
    architecture = spec["architecture"]
    if architecture not in ARMS:
        raise ValueError(f"Unknown architecture: {architecture}")
    torch.manual_seed(spec["initialization"])
    cfg = GPT2Config(
        vocab_size=config["vocab_size"],
        n_positions=config.get("positions", 1024),
        n_embd=config["hidden_size"],
        n_head=config.get("attention_heads", config.get("heads", 12)),
        n_layer=8,
        resid_pdrop=config["dropout"],
        embd_pdrop=config["dropout"],
        attn_pdrop=config["dropout"],
        activation_function="gelu_new",
        use_cache=False,
    )
    # Separate reproducible new-shape RNG, explicitly recorded in the manifest.
    module_seed = spec.get("module_initialization", spec["initialization"] + 1_000_003)
    model = ParametricGPT(cfg, architecture, module_seed)
    model.initialization_seed = spec["initialization"]
    return model.to(device)


def optimizer_for(model, lr, decay):
    groups = [{"params": [], "weight_decay": decay}, {"params": [], "weight_decay": 0.0}]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = any(part in name for part in ("bias", "ln"))
        exempt |= name.startswith("connections.") and (".norm." in name or ".alpha_" in name)
        groups[int(exempt)]["params"].append(parameter)
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=next(model.parameters()).is_cuda
    )


def parameter_ledger(model):
    d, layers = model.config.n_embd, len(model.transformer.h)
    total = sum(parameter.numel() for parameter in model.parameters())
    expert = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)
    controller = sum(p.numel() for n, p in model.named_parameters() if n.startswith("connections."))
    router = sum(p.numel() for n, p in model.named_parameters() if ".router." in n)
    fixed = (model.config.vocab_size + model.config.n_positions) * d + 2 * d
    attention_and_norms = 4 * d * d + 8 * d
    if model.kind == "moe":
        per_block = attention_and_norms + 8 * (4 * d * d + 3 * d) + 8 * d
    else:
        width = (16 if model.kind == "wide" else 4) * d
        per_block = attention_and_norms + 2 * d * width + width + d
    expected_controller = 0
    if model.kind in {"mhc", "identity_mhc"}:
        n = 4
        count = 2 * n + (n * n if model.kind == "mhc" else 0)
        alphas = 3 if model.kind == "mhc" else 2
        expected_controller = 2 * layers * (n * d * count + n * d + count + alphas)
    if controller != expected_controller:
        raise AssertionError(
            f"Controller parameter mismatch: {controller} != {expected_controller}"
        )
    expected = fixed + layers * per_block + expected_controller
    if total != expected:
        raise AssertionError(f"Parameter ledger mismatch: {total} != {expected}")
    return {
        "architecture": model.architecture,
        "total_parameters": total,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "expected_parameters": expected,
        "expert_parameters": expert,
        "router_parameters": router,
        "controller_parameters": controller,
        "nominal_parameters_selected_per_token": total - expert + expert // 4,
        "active_parameter_convention": (
            "All nonexpert parameters (including whole embedding table) plus two of eight "
            "experts; not FLOPs or touched embedding rows."
        ),
        "unique_blocks": layers,
        "repeats": model.repeats,
        "executed_blocks": layers * model.repeats,
        "residual_streams": 4 if controller else 1,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "residual_dtype_under_bf16_autocast": "torch.float32 with FP32 parameters",
        "controller_dtype": "torch.float32",
    }


def flop_ledger(model, *, executed_tokens, projected_positions, attention_pairs, valid_tokens=None):
    """Leading forward matmuls with actual padding/dispatch/readout counts.

    ``attention_pairs`` is sum(B*T*T) over microbatches, before block count.
    Training 3x is explicitly an approximation; Sinkhorn autograd, scalar ops,
    and a counting/replay pass are not silently folded into this estimate.
    """
    d, blocks = model.config.n_embd, len(model.transformer.h) * model.repeats
    valid_tokens = executed_tokens if valid_tokens is None else valid_tokens
    attention = blocks * (8 * executed_tokens * d * d + 4 * attention_pairs * d)
    readout = 2 * projected_positions * d * model.config.vocab_size
    router = controller = 0
    if model.kind == "moe":
        ffn = blocks * 4 * valid_tokens * d * (2 * 2 * d)
        router = blocks * 2 * valid_tokens * d * 8
    else:
        inner = (16 if model.kind == "wide" else 4) * d
        ffn = blocks * 4 * executed_tokens * d * inner
    if model.kind in {"mhc", "identity_mhc"}:
        n = 4
        count = 2 * n + (n * n if model.kind == "mhc" else 0)
        controller = (
            2 * blocks * (2 * executed_tokens * n * d * count + 2 * executed_tokens * d * count)
        )
    forward = attention + readout + ffn + router + controller
    return {
        "forward_attention_matmul_flops": attention,
        "forward_ffn_matmul_flops": ffn,
        "forward_readout_matmul_flops": readout,
        "forward_router_matmul_flops": router,
        "forward_controller_matmul_flops": controller,
        "forward_leading_matmul_flops": forward,
        "estimated_matmul_training_flops": 3 * forward,
        "sinkhorn_normalization_passes": 39 if model.kind == "mhc" else 0,
        "sinkhorn_matrix_elements_per_forward": 2 * blocks * executed_tokens * 16
        if model.kind == "mhc"
        else 0,
        "selected_expert_assignments": blocks * valid_tokens * 2 if model.kind == "moe" else 0,
        "executed_input_tokens": executed_tokens,
        "effective_input_tokens": valid_tokens,
        "projected_positions": projected_positions,
        "excluded_ops": (
            "Nonlinearities, normalization, dropout, dispatch/index-add, softmax, Sinkhorn "
            "scalar arithmetic/autograd, final mean, counting/replay and optimizer; "
            "3x is approximate."
        ),
    }


def initialization_manifest(model):
    """CPU-byte hashes of each actual initialized tensor; no sampled hash."""
    shared, introduced = {}, {}
    for name, tensor in model.state_dict().items():
        is_new = name.startswith("connections.") or (
            model.kind in {"moe", "wide"} and ".mlp." in name
        )
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        (introduced if is_new else shared)[name] = hashlib.sha256(raw).hexdigest()
    return {
        "architecture": model.architecture,
        "reference_initialization": getattr(model, "initialization_seed", None),
        "module_initialization": model.module_seed,
        "reference_blocks_before_slicing": 8,
        "shared_tensor_sha256": shared,
        "introduced_tensor_sha256": introduced,
    }
