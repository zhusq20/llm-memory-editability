"""AlphaEdit's latent target and null-space solve on a tied, recurrent GPT.

Adapted from jianghoucheng/AlphaEdit (MIT, copyright 2025 jianghc). The
single-edit solve is algebraically the author's solve with cache_c=0. Synthetic
facts use their causal relation position, one native context, and the final
execution of each physical layer. An actual down update applies in every round.
No composite labels or backbone optimizer are accepted by the writer.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .grokking_reproduction import encode_rows


def null_basis(covariance, threshold):
    """Author get_project: retain covariance singular directions below threshold."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("A square mean second-moment matrix is required")
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Invalid numerical null-space threshold")
    # Symmetric PSD covariance: eigh and the author's SVD have the same subspaces.
    values, vectors = torch.linalg.eigh(covariance.double())
    values = values.clamp_min(0)
    selected = values < threshold
    basis = vectors[:, selected].contiguous()
    dimension, rank = len(values), int(selected.sum())
    error = (
        float((basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max()) if rank else 0.0
    )
    return basis, {
        "dimension": dimension,
        "null_dimension": rank,
        "protected_dimension": dimension - rank,
        "threshold": threshold,
        "minimum_eigenvalue": float(values[0]),
        "maximum_eigenvalue": float(values[-1]),
        "null_maximum_eigenvalue": float(values[selected].max()) if rank else None,
        "null_energy_fraction": float(values[selected].sum() / values.sum())
        if float(values.sum()) > 0
        else 0.0,
        "orthogonality_error": error,
        "selected_eigenvalues": values[selected].cpu(),
    }


def alphaedit_update(keys, residual, basis, l2):
    """Author single-edit equation in GPT [input, output] storage, no edit history.

    K=[input, requests], R=[output, requests], P=NN^T. This is exactly
    solve(P K K^T + l2 I, P K R^T), evaluated with a request-sized solve.
    """
    if keys.ndim != 2 or residual.ndim != 2 or keys.shape[1] != residual.shape[1]:
        raise ValueError("Keys and residual must have aligned request columns")
    if basis.ndim != 2 or basis.shape[0] != keys.shape[0]:
        raise ValueError("Null-space basis has the wrong orientation")
    if not math.isfinite(l2) or l2 <= 0:
        raise ValueError("AlphaEdit requires a positive L2 coefficient")
    dtype = keys.dtype
    k, r = keys.double(), residual.double()
    n = basis.to(device=k.device, dtype=torch.float64)
    projected = n.T @ k
    gram = projected.T @ projected
    gram += l2 * torch.eye(k.shape[1], device=k.device, dtype=torch.float64)
    update = n @ projected @ torch.linalg.solve(gram, r.T)
    if not torch.isfinite(update).all():
        raise FloatingPointError("Nonfinite AlphaEdit solution")
    return update.to(dtype)


class ExecutionTrace:
    """Observe real LN/MLP calls; optionally add a latent vector at one execution."""

    def __init__(self, reader, fast=None, *, delta=None, target=None, position=1):
        self.reader, self.fast = reader, fast
        self.delta, self.target, self.position = delta, target, position
        self.handles, self.incoming, self.keys, self.states = [], {}, {}, {}
        self.calls = dict.fromkeys(range(len(reader.backbone.transformer.h)), 0)

    def __enter__(self):
        for layer, block in enumerate(self.reader.backbone.transformer.h):

            def incoming(module, args, layer=layer):
                self.incoming[layer] = args[0]

            def down(module, args, output, layer=layer):
                repeat = self.calls[layer]
                self.calls[layer] += 1
                z, p = args[0], self.position
                execution = (repeat, layer)
                replacement = None
                if self.delta is not None and execution == self.target:
                    mask = output.new_zeros((1, output.shape[1], 1))
                    mask[:, p] = 1
                    replacement = output + mask * self.delta
                    output = replacement
                down_output = output[:, p]
                if self.fast is not None:
                    down_output = down_output + (
                        z[:, p].to(self.fast[layer].dtype) @ self.fast[layer]
                    ).to(output.dtype)
                state = self.incoming[layer][:, p] + down_output
                self.keys[execution] = z[:, p].detach().clone()
                self.states[execution] = state.detach().clone()
                return replacement

            self.handles.append(block.ln_2.register_forward_pre_hook(incoming))
            self.handles.append(block.mlp.c_proj.register_forward_hook(down))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        if exc_type is None and any(
            count != self.reader.backbone.repeats for count in self.calls.values()
        ):
            raise AssertionError("Every physical layer must execute once in each round")


def atomic_arrays(rows, metadata, device):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("Writer accepts atomic triples only")
    return tuple(torch.as_tensor(a, device=device) for a in encode_rows(rows, metadata))


def fit_target(reader, fact, controls, metadata, settings):
    """Original CE + KL(new||old) + latent norm objective and Adam/clamp recipe."""
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("Freeze all backbone parameters")
    device = next(reader.parameters()).device
    tokens, positions, labels = atomic_arrays([fact, *controls], metadata, device)
    target_execution = (reader.backbone.repeats - 1, len(reader.backbone.transformer.h) - 1)
    with torch.no_grad(), ExecutionTrace(reader) as clean:
        logits = reader(tokens, positions)
    initial = clean.states[target_execution][0]
    initial_kl = logits[1:, 0].log_softmax(-1).detach()
    delta = torch.zeros_like(initial, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=settings["v_lr"])
    history, updates = [], 0
    for iteration in range(settings["v_num_grad_steps"]):
        optimizer.zero_grad()
        with torch.enable_grad(), ExecutionTrace(reader, delta=delta, target=target_execution):
            logits = reader(tokens, positions)
            nll = F.cross_entropy(logits[:1, 0], labels[:1, 0])
            kl = (
                settings["kl_factor"]
                * F.kl_div(
                    initial_kl,
                    logits[1:, 0].log_softmax(-1),
                    log_target=True,
                    reduction="batchmean",
                )
                if len(controls)
                else nll.new_zeros(())
            )
            decay = settings["v_weight_decay"] * delta.norm() / initial.norm().square()
            loss = nll + kl + decay
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite latent-target objective")
            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach()),
                    "nll": float(nll.detach()),
                    "kl": float(kl.detach()),
                    "norm_penalty": float(decay.detach()),
                    "target_probability": float(torch.exp(-nll.detach())),
                    "argmax": int(logits[0, 0].argmax()),
                    "delta_norm": float(delta.detach().norm()),
                }
            )
            # Match compute_z.py: final iteration evaluates without another step.
            if (
                float(loss.detach()) < settings["stop_loss"]
                or iteration == settings["v_num_grad_steps"] - 1
            ):
                break
            loss.backward()
            reader.compute["estimated_backward_matmul_flops_proxy"] += (
                2 * len(tokens) * reader.backbone.config.n_embd * reader.backbone.config.vocab_size
            )
            optimizer.step()
            updates += 1
            with torch.no_grad():
                maximum = settings["clamp_norm_factor"] * initial.norm()
                if delta.norm() > maximum:
                    delta.mul_(maximum / delta.norm())
    if any(p.grad is not None for p in reader.parameters()):
        raise AssertionError("Backbone accumulated gradients")
    return (initial + delta.detach()).detach(), {
        "latent_updates": updates,
        "history": history,
        "target_execution": list(target_execution),
        "lookup_position": 1,
        "new_atomic_labels": 1,
        "new_composition_labels": 0,
        "controls": controls,
        "initial_norm": float(initial.norm()),
        "delta_norm": float(delta.detach().norm()),
    }


@torch.no_grad()
def write_target(reader, fact, metadata, target, bases, l2):
    """Recompute keys and distribute the final residual across physical layers."""
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("Freeze all backbone parameters")
    device = next(reader.parameters()).device
    tokens, positions, _ = atomic_arrays([fact], metadata, device)
    fast, details = reader.empty_fast(), []
    last = (reader.backbone.repeats - 1, len(fast) - 1)
    target = target.to(device)
    for layer in fast:
        with ExecutionTrace(reader, fast) as trace:
            reader(tokens, positions, fast=fast)
        keys = trace.keys[(last[0], layer)].T
        residual = (target - trace.states[last][0])[:, None] / (len(fast) - layer)
        update = alphaedit_update(keys, residual, bases[layer], l2)
        fast[layer].add_(update)
        n = bases[layer].to(device=device, dtype=torch.float64)
        reconstructed = n @ (n.T @ update.double())
        details.append(
            {
                "layer": layer,
                "key_execution": [last[0], layer],
                "residual_norm": float(residual.norm()),
                "key_norm": float(keys.norm()),
                "projected_key_norm": float((n.T @ keys.double()).norm()),
                "update_norm": float(update.norm()),
                "relative_update_norm": float(
                    update.norm() / reader.backbone.transformer.h[layer].mlp.c_proj.weight.norm()
                ),
                "outside_null_space_relative_norm": float(
                    (update.double() - reconstructed).norm()
                    / update.double().norm().clamp_min(1e-30)
                ),
            }
        )
    reader.validate_fast(fast)
    return fast, {
        "down_matrix_updates": len(fast),
        "layers": details,
        "l2": l2,
        "previous_edit_cache": "zero; isolated single-fact states",
        "new_composition_labels": 0,
    }


@torch.no_grad()
def collect_old_covariances(reader, row_groups, metadata, batch_size=64):
    """Mean old-key second moments, all valid positions and all loop executions."""
    device = next(reader.parameters()).device
    cov = {
        i: torch.zeros(w.shape[0], w.shape[0], device=device, dtype=torch.float64)
        for i, w in reader.empty_fast().items()
    }
    counts = {i: {"calls": 0, "positions": 0} for i in cov}
    valid_length, handles = [0], []
    for layer, block in enumerate(reader.backbone.transformer.h):

        def observe(module, args, layer=layer):
            features = args[0][:, : valid_length[0]].reshape(-1, args[0].shape[-1]).double()
            cov[layer].add_(features.T @ features)
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
    return {i: value / counts[i]["positions"] for i, value in cov.items()}, counts


def select_l2(records, minimum_retention):
    """Select using development atomic writes and the worst old control pool."""
    eligible = [r for r in records if r["minimum_old_retention"] >= minimum_retention]
    if eligible:
        chosen = max(
            eligible, key=lambda r: (r["atomic_accuracy"], r["minimum_old_retention"], r["l2"])
        )
    else:
        chosen = max(
            records, key=lambda r: (r["minimum_old_retention"], r["atomic_accuracy"], r["l2"])
        )
    return {
        **chosen,
        "retention_threshold_met": bool(eligible),
        "selection_new_composition_queries": 0,
    }
