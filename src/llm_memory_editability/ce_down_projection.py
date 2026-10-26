"""Atomic CE writes to shared MLP down matrices, with optional GPM projection.

The backbone is frozen. Explicit additive matrices are exactly equivalent to
changing its down weights, and are reused in every loop execution. No context
distillation, outer training, composite supervision or old-data replay is used
during a write. Projection protects a fixed, sampled old-feature subspace; it
does not guarantee preservation of the complete nonlinear network.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .grokking_reproduction import encode_rows


def project_gradient(gradient, basis):
    """GPT Conv1D layout [intermediate, hidden]; basis columns are old keys."""
    if basis.ndim != 2 or gradient.ndim != 2 or basis.shape[0] != gradient.shape[0]:
        raise ValueError("Projection dimensions do not match the down matrix")
    q = basis.to(device=gradient.device, dtype=gradient.dtype)
    return gradient - q @ (q.T @ gradient)


def basis_from_covariance(covariance, energy=0.95):
    if not 0 < energy <= 1 or covariance.ndim != 2:
        raise ValueError("Invalid feature energy or covariance")
    values, vectors = torch.linalg.eigh(covariance.double().cpu())
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    total = values.sum()
    if total <= 0:
        raise ValueError("Old-feature covariance has zero energy")
    rank = min(int(torch.searchsorted(values.cumsum(0), total * energy)) + 1, len(values))
    basis = vectors[:, :rank].contiguous()
    return basis, {
        "rank": rank,
        "dimension": len(values),
        "energy_threshold": energy,
        "retained_energy": float(values[:rank].sum() / total),
        "orthogonality_error": float((basis.T @ basis - torch.eye(rank)).abs().max()),
    }


@torch.no_grad()
def collect_covariances(reader, row_groups, metadata, batch_size=32):
    """Observe every token and every repeat at each physical down projection."""
    device = next(reader.parameters()).device
    covariances = {
        i: torch.zeros(w.shape[0], w.shape[0], dtype=torch.float64, device=device)
        for i, w in reader.empty_fast().items()
    }
    counts = {i: {"calls": 0, "positions": 0} for i in covariances}
    valid_length = [0]
    handles = []
    for layer, block in enumerate(reader.backbone.transformer.h):

        def observe(module, args, layer=layer):
            features = args[0][:, : valid_length[0]].detach().reshape(-1, args[0].shape[-1]).double()
            covariances[layer].add_(features.T @ features)
            counts[layer]["calls"] += 1
            counts[layer]["positions"] += len(features)

        handles.append(block.mlp.c_proj.register_forward_pre_hook(observe))
    try:
        for rows in row_groups:
            valid_length[0] = np.asarray(rows).shape[1]
            for begin in range(0, len(rows), batch_size):
                arrays = encode_rows(np.asarray(rows[begin : begin + batch_size]), metadata)
                tokens, positions, _ = (torch.as_tensor(a, device=device) for a in arrays)
                reader(tokens, positions)
    finally:
        for handle in handles:
            handle.remove()
    return covariances, counts


def atomic_ce_write(reader, fact, metadata, learning_rate, *, bases=None, steps=1,
                    stop_loss=None, max_relative_update=None):
    """Fixed SGD budget, answer+EOS only; theta is never trainable."""
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("Freeze every backbone parameter before injecting a fact")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("A finite positive write learning rate is required")
    if type(steps) is not int or steps < 1:
        raise ValueError("A fixed positive integer write budget is required")
    if stop_loss is not None and (not math.isfinite(stop_loss) or stop_loss < 0):
        raise ValueError("Invalid atomic stopping loss")
    if max_relative_update is not None and (
        not math.isfinite(max_relative_update) or max_relative_update <= 0
    ):
        raise ValueError("Invalid down-weight trust radius")
    if bases is not None and set(bases) != set(reader.empty_fast()):
        raise ValueError("Protect every shared physical down matrix")
    device = next(reader.parameters()).device
    arrays = encode_rows(np.asarray([fact], dtype=np.int64), metadata)
    tokens, positions, labels = (torch.as_tensor(a, device=device) for a in arrays)
    fast, history = reader.empty_fast(), []
    for _ in range(steps):
        with torch.enable_grad():
            fast = {k: v.detach().requires_grad_() for k, v in fast.items()}
            loss = F.cross_entropy(reader(tokens, positions, fast=fast).flatten(0, 1), labels.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite new-fact CE")
            if stop_loss is not None and float(loss.detach()) <= stop_loss:
                fast = {k: v.detach() for k, v in fast.items()}
                break
            gradients = torch.autograd.grad(loss, tuple(fast.values()))
        updated, details = {}, {}
        for (layer, state), gradient in zip(fast.items(), gradients, strict=True):
            direction = gradient if bases is None else project_gradient(gradient, bases[layer])
            if not torch.isfinite(direction).all():
                raise FloatingPointError("Nonfinite CE write direction")
            updated[layer] = (state - learning_rate * direction).detach()
            if max_relative_update is not None:
                maximum = max_relative_update * reader.backbone.transformer.h[layer].mlp.c_proj.weight.norm()
                scale = (maximum / updated[layer].norm().clamp_min(1e-30)).clamp(max=1)
                updated[layer] = updated[layer] * scale
            norm = gradient.double().norm().clamp_min(1e-30)
            protected = None
            if bases is not None:
                q = bases[layer].to(device=device, dtype=direction.dtype)
                protected = float((q.T @ direction).norm() / direction.norm().clamp_min(1e-30))
            details[layer] = {
                "gradient_norm": float(norm),
                "retained_gradient_norm_fraction": float(direction.double().norm() / norm),
                "protected_subspace_relative_residual": protected,
                "delta_norm": float(updated[layer].double().norm()),
            }
        reader.validate_fast(updated)
        history.append({"loss_before": float(loss.detach()), "layers": details})
        fast = updated
    return fast, {"steps": len(history), "maximum_steps": steps, "history": history}


def choose_learning_rate(records, minimum_retention=0.98):
    """No new-composition score may participate in development selection."""
    if not records or any(set(r) != {"learning_rate", "atomic", "old_retention"} for r in records):
        raise ValueError("Selection accepts only atomic accuracy and old retention")
    eligible = [r for r in records if r["old_retention"] >= minimum_retention]
    pool = eligible or records
    best = max(pool, key=lambda r: (r["atomic"], r["old_retention"], -r["learning_rate"]))
    return {**best, "retention_threshold_met": bool(eligible)}
