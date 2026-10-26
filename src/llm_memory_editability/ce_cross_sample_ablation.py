"""Single-factor continuous-write interventions, preserving the native CE update order."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from .ce_cross_sample import settings_for
from .ce_scope_editing import arrays, freeze, parameter_names, set_scope


def arm_settings(plan, base_plan, arm, learning_rate=None):
    item = plan["arms"][arm]
    settings = settings_for(base_plan, item["architecture"])
    candidates = item["learning_rate_candidates"]
    lr = candidates[0] if learning_rate is None else learning_rate
    if lr not in candidates or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be a frozen candidate")
    original = settings["learning_rate"]
    variant = item["variant"]
    if variant not in ("lower-lr", "carry-adam", "no-bound"):
        raise ValueError("Unknown single-factor ablation")
    carry, bound = variant == "carry-adam", variant != "no-bound"
    if (
        item["optimizer_mode"] != ("carry_per_stream" if carry else "fresh_per_fact")
        or item["constraint_enabled"] != bound
        or (variant != "lower-lr" and candidates != [original])
        or (
            variant == "lower-lr"
            and (len(candidates) != 2 or not all(x < original for x in candidates))
        )
    ):
        raise ValueError("Each arm must alter exactly one authorized factor")
    settings["learning_rate"] = lr
    return settings, item["optimizer_mode"], bound


def optimizer_counts(optimizer):
    if optimizer is None:
        return []
    return [
        int(optimizer.state.get(parameter, {}).get("step", 0))
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def tree_digest(value):
    """Stable nested optimizer identity including tensor dtype/shape and scalar settings."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict:")
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b";")

    visit(value)
    return digest.hexdigest()


def optimizer_snapshot(optimizer):
    def cpu(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().clone()
        if isinstance(item, dict):
            return {key: cpu(value) for key, value in item.items()}
        if isinstance(item, list):
            return [cpu(value) for value in item]
        if isinstance(item, tuple):
            return tuple(cpu(value) for value in item)
        return copy.deepcopy(item)

    return cpu(optimizer.state_dict())


def make_optimizer(reader, settings):
    parameters = [reader.backbone.get_parameter(n) for n in parameter_names(reader.backbone, "mlp")]
    return torch.optim.Adam(
        parameters,
        lr=settings["learning_rate"],
        betas=settings["betas"],
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
    )


@torch.no_grad()
def update_geometry(reader, anchor, previous, radius):
    squared, offset_squared, projection, denominator = 0.0, 0.0, 0.0, 0.0
    maximum, offset_max = 0.0, 0.0
    saturated, outside, flipped, elements = 0, 0, 0, 0
    for name, origin in anchor.items():
        current = reader.backbone.get_parameter(name)
        before = previous[name]
        delta, offset, old_offset = current - before, current - origin, before - origin
        squared += float(delta.double().square().sum())
        offset_squared += float(offset.double().square().sum())
        projection += float((offset.double() * old_offset.double()).sum())
        denominator += float(old_offset.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        offset_max = max(offset_max, float(offset.abs().max()))
        lo, hi = origin - radius, origin + radius
        saturated += int(((current == lo) | (current == hi)).sum())
        outside += int(((current < lo) | (current > hi)).sum())
        flipped += int(
            (((before == lo) & (current == hi)) | ((before == hi) & (current == lo))).sum()
        )
        elements += current.numel()
    return {
        "per_fact_update_rms": math.sqrt(squared / elements),
        "per_fact_update_maximum": maximum,
        "theta0_offset_rms": math.sqrt(offset_squared / elements),
        "theta0_offset_maximum": offset_max,
        "original_region_saturation_fraction": saturated / elements,
        "original_region_outside_fraction": outside / elements,
        "opposite_boundary_flip_fraction": flipped / elements,
        "previous_offset_projection": projection / denominator if denominator else None,
    }


def write_fact(
    reader,
    fact,
    metadata,
    settings,
    teachers,
    anchor,
    *,
    optimizer=None,
    optimizer_mode="fresh_per_fact",
    constraint_enabled=True,
    progress=None,
):
    if len(fact) != 3 or len(teachers) != settings["steps"]:
        raise ValueError("One fact and the exact fixed teacher batches are required")
    if optimizer_mode not in ("fresh_per_fact", "carry_per_stream"):
        raise ValueError("Unknown optimizer lifecycle")
    if optimizer_mode == "fresh_per_fact" and optimizer is not None:
        raise ValueError("Fresh Adam must not receive persistent state")
    for key in ("learning_rate", "kl_factor", "norm_constraint", "gradient_clip"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError("Invalid positive write setting")
    names = parameter_names(reader.backbone, "mlp")
    if set(anchor) != set(names):
        raise ValueError("The immutable anchor must contain exactly the unique physical MLP")
    set_scope(reader, "mlp")
    parameters = [reader.backbone.get_parameter(name) for name in names]
    previous = {name: p.detach().clone() for name, p in zip(names, parameters, strict=True)}
    device = next(reader.parameters()).device
    tokens, positions, labels = arrays([fact], metadata, device)
    tokens, positions, labels = tokens[:, :2], positions[:, :1], labels[:, :1]
    if optimizer is None:
        optimizer = make_optimizer(reader, settings)
    if [id(p) for g in optimizer.param_groups for p in g["params"]] != [id(p) for p in parameters]:
        raise ValueError("Persistent Adam must reference these exact native physical parameters")
    group = optimizer.param_groups[0]
    if (
        any(group[key] != settings[key] for key in ("betas", "eps", "weight_decay"))
        or group["lr"] != settings["learning_rate"]
    ):
        raise ValueError("Persistent Adam settings changed across facts")
    counts_before = optimizer_counts(optimizer)
    if len(set(counts_before)) != 1:
        raise ValueError("Adam counters diverged across editable parameters")
    history = []
    for step, batches in enumerate(teachers, 1):
        optimizer.zero_grad(set_to_none=True)
        ce = F.cross_entropy(reader(tokens, positions)[:, 0], labels[:, 0])
        if not torch.isfinite(ce):
            raise FloatingPointError("Nonfinite new-fact CE")
        ce.backward()
        kls = []
        for batch in batches:
            x, p, old = (batch[k].to(device) for k in ("tokens", "positions", "log_probability"))
            logits = reader(x, p)
            kl = F.kl_div(logits.log_softmax(-1), old, log_target=True, reduction="sum")
            kl = kl / (logits.shape[0] * logits.shape[1])
            if not torch.isfinite(kl):
                raise FloatingPointError("Nonfinite protected old-output KL")
            (settings["kl_factor"] * kl / len(batches)).backward()
            kls.append(float(kl.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite write gradient")
        optimizer.step()
        radius = settings["norm_constraint"]
        if constraint_enabled:
            with torch.no_grad():
                for name, parameter in zip(names, parameters, strict=True):
                    origin = anchor[name]
                    parameter.copy_(
                        torch.clamp(parameter, min=origin - radius, max=origin + radius)
                    )
        entry = {
            "step": step,
            "new_ce_before": float(ce.detach()),
            "old_KL_before": kls,
            "gradient_norm_before_clip": float(norm),
        }
        history.append(entry)
        if progress:
            progress(step, entry)
    freeze(reader)
    counts_after = optimizer_counts(optimizer)
    if counts_after != [x + settings["steps"] for x in counts_before]:
        raise AssertionError("Adam did not execute exactly the intended updates")
    geometry = update_geometry(reader, anchor, previous, settings["norm_constraint"])
    return {
        "steps": settings["steps"],
        "settings": settings,
        "optimizer_reset_before_fact": optimizer_mode == "fresh_per_fact",
        "optimizer_mode": optimizer_mode,
        "optimizer_counts_before": counts_before,
        "optimizer_counts_after": counts_after,
        "optimizer_state_digest": tree_digest(optimizer.state_dict()),
        "constraint_enabled": constraint_enabled,
        "global_theta0_anchor": True,
        "maximum_absolute_parameter_change": geometry["theta0_offset_maximum"],
        "trainable_parameters": sum(p.numel() for p in parameters),
        "new_supervised_tokens": settings["steps"],
        "old_examples": sum(int(b["tokens"].shape[0]) for bs in teachers for b in bs),
        "new_composition_supervision": 0,
        "geometry": geometry,
        "history": history,
    }, optimizer


def gate_summary(records, gate):
    expected = gate["total_current_readouts_per_candidate"]
    if len(records) != expected or len({r["key"] for r in records}) != expected:
        raise ValueError("Development gate must contain all current-fact readouts")
    correct = sum(bool(r["readout"]["final_correct"][0]) for r in records)
    drops = [
        r["write_info"]["history"][0]["new_ce_before"]
        - r["write_info"]["history"][-1]["new_ce_before"]
        for r in records
    ]
    finite = all(math.isfinite(x) for x in drops)
    median_drop = float(np.median(drops)) if finite else None
    eligible = finite and correct >= gate["minimum_correct"] and median_drop > 0
    return {
        "eligible": bool(eligible),
        "gate_valid": bool(eligible),
        "current_correct": correct,
        "current_total": expected,
        "current_accuracy": correct / expected,
        "median_ce_decrease": median_drop,
        "finite": finite,
    }
