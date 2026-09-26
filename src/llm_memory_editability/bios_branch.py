"""Equal-parameter residual branches from old-world activation directions."""

import time

import numpy as np
import torch
from torch import nn

from .bios_data import ANS
from .bios_train import precision


class ResidualBranch(nn.Module):
    def __init__(self, q, mean, scale, width=32, seed=73):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        # Explicit local RNG: same A in all arms without changing training RNG state.
        generator = torch.Generator(device=q.device).manual_seed(seed)
        self.a = nn.Parameter(
            torch.randn(width, q.shape[1], generator=generator, device=q.device) / q.shape[1] ** 0.5
        )
        self.b = nn.Parameter(torch.zeros(q.shape[0], width, device=q.device))

    def forward(self, v):
        s = (v.float() @ self.q - self.mean) / self.scale
        return (torch.nn.functional.gelu(s @ self.a.T) @ self.b.T).to(v.dtype)


@torch.no_grad()
def build_bases(model, world, data, layer=4, rank=32):
    device = data["tokens"].device
    values = []

    def capture(_module, _inputs, output):
        values.append(output[:, 3].float())

    handle = model.blocks[layer].register_forward_hook(capture)
    started = time.perf_counter()
    try:
        with precision(device):
            for begin in range(2112, 4160, 256):
                model(
                    data["prompts"][begin : begin + 256],
                    data["lengths"][begin : begin + 256, None] - 1,
                )
    finally:
        handle.remove()
    v = torch.cat(values)
    group_means = torch.stack(
        [v[torch.as_tensor(world.employers == c, device=device)].mean(0) for c in range(64)]
    )
    centered_means = group_means - v.mean(0)
    _, shared_s, shared_vh = torch.linalg.svd(centered_means, full_matrices=False)
    q_shared = shared_vh[:rank].T.contiguous()
    residual = v - group_means[torch.as_tensor(world.employers, device=device)]
    residual = residual - (residual @ q_shared) @ q_shared.T
    _, independent_s, independent_vh = torch.linalg.svd(residual, full_matrices=False)
    q_independent = independent_vh[:rank].T.contiguous()
    valid = bool(
        shared_s[rank - 1] > shared_s[0] * 1e-6
        and independent_s[rank - 1] > independent_s[0] * 1e-6
    )
    if not valid:
        return None, {"available": False, "reason": "fewer than 32 nondegenerate directions"}
    generator = torch.Generator(device=device).manual_seed(74)
    q_random = torch.linalg.qr(torch.randn(v.shape[1], rank, generator=generator, device=device)).Q
    bases, checks = {}, {}
    for kind, q in (("shared", q_shared), ("independent", q_independent), ("random", q_random)):
        projected = v @ q
        mean, std = projected.mean(0), projected.std(0, correction=0)
        std = std.clamp_min(0.01 * std.square().mean().sqrt())
        normalized = (projected - mean) / std
        # Equal total variance across all branches (one, not dependent on their rank).
        scale = std * normalized.square().sum(-1).mean().sqrt()
        bases[kind] = (q, mean, scale)
        normalized = (projected - mean) / scale
        within = []
        for c in range(64):
            mask = (world.employers == c) & ~world.exceptions
            z = normalized[torch.as_tensor(mask, device=device)]
            within.append((z - z.mean(0)).square().sum(-1).mean().item())
        checks[kind] = {
            "same_company_relation_old_answer_identity_variance": float(np.mean(within)),
            "total_variance": normalized.square().sum(-1).mean().item(),
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return bases, {
        "available": True,
        "construction_seconds": time.perf_counter() - started,
        "rank": rank,
        "shared_32nd_relative_singular_value": (shared_s[rank - 1] / shared_s[0]).item(),
        "independent_32nd_relative_singular_value": (
            independent_s[rank - 1] / independent_s[0]
        ).item(),
        "checks": checks,
    }


def attach_branch(model, basis, layer=4, width=32):
    branch = ResidualBranch(*basis, width=width)
    model.add_module("edit_branch", branch)

    def save_position(_module, args):
        branch.positions = args[0].eq(ANS).long().argmax(-1)

    def add(_module, _inputs, output):
        rows = torch.arange(len(output), device=output.device)
        result = output.clone()
        result[rows, branch.positions] = result[rows, branch.positions] + branch(
            output[rows, branch.positions]
        )
        return result

    handles = [
        model.register_forward_pre_hook(save_position),
        model.blocks[layer].register_forward_hook(add),
    ]
    return branch, handles


def remove_branch(model, handles):
    for handle in handles:
        handle.remove()
    delattr(model, "edit_branch")
