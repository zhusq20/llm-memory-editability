"""Matched atomic CE edits of down weights, entire MLPs, or the full DiscoLoop.

All scopes use the same old-output KL, elementwise trust region, Adam budget
and teacher batches. Only the new atomic tail is a supervised new target.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from .grokking_reproduction import encode_rows

SCOPES = ("down", "mlp", "all")


def parameter_names(model, scope):
    if scope not in SCOPES:
        raise ValueError("Unknown CE edit scope")
    names = [
        name
        for name, _ in model.named_parameters()
        if scope == "all"
        or (scope == "mlp" and ".mlp." in name)
        or (scope == "down" and name.endswith(".mlp.c_proj.weight"))
    ]
    if not names:
        raise ValueError("An edit scope cannot be empty")
    return names


def set_scope(reader, scope):
    names = parameter_names(reader.backbone, scope)
    for name, parameter in reader.backbone.named_parameters():
        parameter.requires_grad_(name in names)
        parameter.grad = None
    reader.trainable_backbone = True
    return names


def freeze(reader):
    reader.requires_grad_(False)
    reader.trainable_backbone = False
    for parameter in reader.parameters():
        parameter.grad = None


@torch.no_grad()
def restore(reader, base):
    freeze(reader)
    reader.backbone.load_state_dict(base, strict=True)


@torch.no_grad()
def load_edit(reader, values, scope):
    freeze(reader)
    if set(values) != set(parameter_names(reader.backbone, scope)):
        raise ValueError("Saved edit does not have exactly its declared parameter scope")
    for name, value in values.items():
        parameter = reader.backbone.get_parameter(name)
        if parameter.shape != value.shape or parameter.dtype != value.dtype:
            raise ValueError("Saved parameter shape/dtype differs")
        parameter.copy_(value)


def arrays(rows, metadata, device):
    return tuple(torch.as_tensor(a, device=device) for a in encode_rows(np.asarray(rows), metadata))


def choose_recipe(records, minimum_retention):
    required = {
        "learning_rate",
        "atomic_accuracy",
        "minimum_old_retention",
        "retention_by_pool",
    }
    if not records or any(set(r) != required for r in records):
        raise ValueError("Recipe selection accepts only atomic and old-retention outcomes")
    eligible = [r for r in records if r["minimum_old_retention"] >= minimum_retention]
    if eligible:
        best = max(
            eligible,
            key=lambda r: (
                r["atomic_accuracy"],
                r["minimum_old_retention"],
                -r["learning_rate"],
            ),
        )
    else:
        best = max(
            records,
            key=lambda r: (
                r["minimum_old_retention"],
                r["atomic_accuracy"],
                -r["learning_rate"],
            ),
        )
    return {**best, "retention_threshold_met": bool(eligible)}


@torch.no_grad()
def teacher_batches(reader, world, split, settings):
    """Use exactly the same protected input sequence and old distributions in all arms."""
    result = []
    rng = np.random.RandomState(settings["batch_seed"])
    count = settings["old_per_kind_per_step"]
    steps = settings["steps"]
    selected = {}
    for kind in ("atoms", "chains"):
        candidates = np.asarray(split[f"protection_{kind}"])
        if steps * count > len(candidates):
            raise ValueError("Protective batches must sample without replacement")
        selected[kind] = rng.permutation(candidates)[: steps * count].reshape(steps, count)
    device = next(reader.parameters()).device
    for step in range(steps):
        batches = []
        for kind in ("atoms", "chains"):
            rows = getattr(world, kind)[selected[kind][step]]
            tokens, positions, labels = arrays(rows, world.metadata, device)
            logits = reader(tokens, positions)
            batches.append(
                {
                    "kind": kind,
                    "indices": selected[kind][step].tolist(),
                    "rows": rows.tolist(),
                    "tokens": tokens.cpu(),
                    "positions": positions.cpu(),
                    "log_probability": logits.log_softmax(-1).cpu(),
                    "target_slots": int(labels.numel()),
                }
            )
        result.append(batches)
    return result


def atomic_ce_edit(reader, fact, metadata, scope, settings, teachers):
    """Exactly N Adam steps, independent optimizer; no composite new labels accepted."""
    if len(fact) != 3 or len(teachers) != settings["steps"]:
        raise ValueError("Only one atomic fact and the fixed protective schedule are accepted")
    for key in ("learning_rate", "kl_factor", "norm_constraint", "gradient_clip"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid positive edit setting: {key}")
    names = set_scope(reader, scope)
    parameters = [reader.backbone.get_parameter(name) for name in names]
    original = {name: p.detach().clone() for name, p in zip(names, parameters, strict=True)}
    device = next(reader.parameters()).device
    tokens, positions, labels = arrays([fact], metadata, device)
    # Match AlphaEdit: supervise only the tail, not EOS or intermediate states.
    tokens, positions, labels = tokens[:, :2], positions[:, :1], labels[:, :1]
    optimizer = torch.optim.Adam(
        parameters,
        lr=settings["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    )
    history = []
    started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    ended = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if started is not None:
        started.record()
    for step in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(reader(tokens, positions)[:, 0], labels[:, 0])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite new atomic CE")
        nll = float(loss.detach())
        loss.backward()
        kls = []
        for batch in teachers[step]:
            x, p, old = (batch[k].to(device) for k in ("tokens", "positions", "log_probability"))
            logits = reader(x, p)
            kl = F.kl_div(logits.log_softmax(-1), old, log_target=True, reduction="sum")
            kl = kl / (logits.shape[0] * logits.shape[1])
            if not torch.isfinite(kl):
                raise FloatingPointError("Nonfinite old-output KL")
            # Equal weight for old atoms and old training-II combinations.
            (settings["kl_factor"] * kl / len(teachers[step])).backward()
            kls.append(float(kl.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite CE gradient")
        optimizer.step()
        with torch.no_grad():
            for name, parameter in zip(names, parameters, strict=True):
                base = original[name]
                eps = settings["norm_constraint"]
                parameter.copy_(torch.clamp(parameter, min=base - eps, max=base + eps))
        history.append(
            {
                "step": step + 1,
                "new_atomic_ce_before": nll,
                "old_KL_before": kls,
                "gradient_norm_before_clip": float(norm),
            }
        )
    if ended is not None:
        ended.record()
        torch.cuda.synchronize(device)
    with torch.no_grad():
        values = {
            name: parameter.detach().cpu().clone()
            for name, parameter in zip(names, parameters, strict=True)
        }
        squared, maximum = 0.0, 0.0
        for name, parameter in zip(names, parameters, strict=True):
            difference = parameter.detach() - original[name]
            squared += float(difference.double().square().sum())
            maximum = max(maximum, float(difference.abs().max()))
        final_ce = float(F.cross_entropy(reader(tokens, positions)[:, 0], labels[:, 0]))
    freeze(reader)
    return values, {
        "scope": scope,
        "steps": len(history),
        "maximum_steps": settings["steps"],
        "new_atomic_exposures": len(history),
        "new_supervised_tokens": len(history),
        "old_atomic_exposures": len(history) * settings["old_per_kind_per_step"],
        "old_composite_exposures": len(history) * settings["old_per_kind_per_step"],
        "old_distillation_slots": sum(b["target_slots"] for bs in teachers for b in bs),
        "updated_composite_supervision": 0,
        "early_entity_supervision": 0,
        "parameter_names": names,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "delta_frobenius_norm": math.sqrt(squared),
        "maximum_absolute_parameter_change": maximum,
        "final_new_atomic_ce": final_ce,
        "gpu_optimizer_milliseconds": started.elapsed_time(ended) if started is not None else None,
        "settings": settings,
        "history": history,
    }
