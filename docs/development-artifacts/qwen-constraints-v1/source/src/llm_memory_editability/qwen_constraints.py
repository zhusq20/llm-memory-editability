"""Constrained response primitives; empirical scopes are recorded by the runner."""

from __future__ import annotations

import hashlib

import torch


def select_records(records, count, seed, role):
    """Select before diagnostics, independently of accuracy or measured geometry."""
    if len(records) < count:
        raise ValueError(f"Insufficient records for {role}")
    return sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{role}:{row['case_id']}".encode()).hexdigest(),
    )[:count]


def psd_solve(gram, rhs, rtol):
    values, vectors = torch.linalg.eigh(gram.double())
    eligible = values > values.max().clamp_min(1e-30) * rtol
    projected = vectors[:, eligible].T @ rhs.double()
    result = vectors[:, eligible] @ (projected / values[eligible])
    return result, int(eligible.sum())


def functional_projection(target, keep, rtol):
    """Orthogonal projection onto the numerical null space of keep gradients."""
    g, r = target.double(), keep.double()
    coefficients, rank = psd_solve(r @ r.T, r @ g, rtol)
    result = g - coefficients @ r
    denominator = g.square().sum().clamp_min(1e-30)
    return result.to(target.dtype), {
        "rank": rank,
        "retained_energy": float(result.square().sum() / denominator),
        "relative_keep_residual": float(
            (r @ result).norm() / (r.norm() * result.norm()).clamp_min(1e-30)
        ),
    }


def feature_basis(features, rtol):
    """Rows are real token features; right singular vectors span protected keys."""
    _, singular, vh = torch.linalg.svd(features.double(), full_matrices=False)
    selected = singular > singular[0] * rtol
    return vh[selected], singular


def module_projection(gradient, basis):
    g = gradient.double()
    return (g - (g @ basis.T) @ basis).to(gradient.dtype)


def local_gradients(x, up, gate, down_weight, output_derivative):
    """All-position exact SwiGLU chain rule; no replacement of the forward model."""
    sigmoid = torch.sigmoid(gate)
    activated = gate * sigmoid
    derivative = sigmoid + gate * sigmoid * (1 - sigmoid)
    value_derivative = output_derivative @ down_weight
    return {
        "down": output_derivative.T @ (up * activated),
        "up": (value_derivative * activated).T @ x,
        "gate": (value_derivative * up * derivative).T @ x,
    }


def route_output(current, original, attention, route):
    """Forward activation intervention; baseline function is identical at zero update."""
    if route == "all":
        return current
    mask = torch.zeros_like(attention, dtype=torch.bool)
    mask[torch.arange(len(mask), device=mask.device), attention.sum(1) - 1] = True
    if route == "earlier":
        mask = attention.bool() & ~mask
    elif route != "last":
        raise ValueError(route)
    return torch.where(mask.unsqueeze(-1), current, original)


def update_scale(direction, target_gradient, step, parameter_norm, cap, metric):
    d_norm = direction.double().norm()
    g_norm = target_gradient.double().norm()
    response = target_gradient.double() @ direction.double()
    if d_norm == 0 or response <= 0:
        return 0.0, True
    if metric == "equal_norm":
        scale = step / (g_norm * d_norm)
    elif metric == "equal_target":
        scale = step / response
    else:
        raise ValueError(metric)
    maximum = cap * parameter_norm / d_norm
    return float(torch.minimum(scale, maximum)), bool(scale > maximum)
