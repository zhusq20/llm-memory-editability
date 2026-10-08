"""Local gradient geometry at the current weights of a paired Loop model.

The caller supplies a fixed batch drawn only from training records. These
measurements describe gradients at one parameter point; they are not predictors
of held-out behavior or evidence that an observed geometry causes generalization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time

import torch
from torch import nn

from .realworld_composition import example_losses, pack


def _unfold(model):
    """Untie every occurrence while retaining its exact current parameters."""
    expanded = copy.deepcopy(model)
    expanded.transformer.h = nn.ModuleList(
        [copy.deepcopy(block) for _ in range(model.repeats) for block in model.transformer.h]
    )
    expanded.repeats = 1
    expanded.config.n_layer = len(expanded.transformer.h)
    return expanded


def _gradient_vector(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None for gradient in gradients):
        raise ValueError("Diagnostic modules must have nonempty, fully connected gradients")
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients]).double()


def _gradient_geometry(vectors):
    """Return Gram geometry, with undefined zero-gradient cosines as null."""
    stacked = torch.stack(vectors).double()
    gram = stacked @ stacked.T
    norms = gram.diag().clamp_min(0).sqrt()
    denominator = norms[:, None] * norms[None, :]
    cosines = gram / denominator.clamp_min(torch.finfo(gram.dtype).tiny)
    norm_values = norms.cpu().tolist()
    dots = gram.cpu().tolist()
    cosine_values = cosines.clamp(-1, 1).cpu().tolist()
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if not norm_values[i] or not norm_values[j]:
                cosine_values[i][j] = None
    summed = stacked.sum(dim=0)
    sum_squared = float(summed.square().sum())
    individual_squared = float(gram.diag().sum())
    return {
        "occurrence_gradient_norms": norm_values,
        "gradient_dot_products": dots,
        "gradient_cosines": cosine_values,
        "summed_gradient_norm": sum_squared**0.5,
        "summed_gradient_squared_norm": sum_squared,
        "sum_individual_squared_norms": individual_squared,
        "twice_pairwise_inner_product_sum": float(gram.sum() - gram.diag().sum()),
        # One means zero net cross term, zero means complete cancellation;
        # identical nonzero gradients give the number of occurrences.
        "squared_norm_ratio": sum_squared / individual_squared if individual_squared else None,
        "gradient_coordinates": stacked.shape[1],
    }


def _gradient_reconstruction(reference, expanded):
    """Compare each independent parameter with the sum over its actual aliases."""
    groups = {}
    for repeat in range(reference.repeats):
        for index, block in enumerate(reference.transformer.h):
            occurrence = repeat * len(reference.transformer.h) + index
            independent = dict(expanded.transformer.h[occurrence].named_parameters())
            for name, parameter in block.named_parameters():
                group = groups.setdefault(id(parameter), (parameter, []))
                group[1].append(independent[name])
    expanded_parameters = dict(expanded.named_parameters())
    for name, parameter in reference.named_parameters():
        if not name.startswith("transformer.h."):
            groups[id(parameter)] = (parameter, [expanded_parameters[name]])
    absolute, relative, tied_count, coordinate_count = 0.0, 0.0, 0, 0
    for parameter, occurrences in groups.values():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or any(other.grad is None for other in occurrences):
            raise ValueError("A trainable parameter has no diagnostic gradient")
        expected = sum(other.grad.detach().double() for other in occurrences)
        difference = parameter.grad.detach().double() - expected
        absolute = max(absolute, float(difference.abs().max()))
        denominator = max(float(expected.norm()), torch.finfo(torch.float64).tiny)
        relative = max(relative, float(difference.norm()) / denominator)
        tied_count += len(occurrences) > 1
        coordinate_count += parameter.numel()
    return {
        "maximum_absolute_error": absolute,
        "maximum_relative_l2_error": relative,
        "unique_parameter_tensors": len(groups),
        "actually_tied_parameter_tensors": tied_count,
        "trainable_parameter_coordinates": coordinate_count,
        "interpretation": (
            "Each reference gradient equals the sum over its actual execution aliases. "
            "For an untied parameter this is a one-occurrence equality."
        ),
    }


def diagnose(model, records, spec, device, pad):
    """Measure current occurrence gradients without changing the source model.

    Two private model copies retain and fully unfold the source's current
    weights. Both run in evaluation mode with autograd enabled, so dropout is
    disabled and the loss is deterministic. No optimizer step is taken. The
    original parameters, gradients, mode, and Torch RNG states are preserved.
    Autocast is disabled: the diagnostic uses the model's own parameter dtype.
    ``records`` must be a fixed, caller-selected training-only batch.
    """
    if not records:
        raise ValueError("Gradient diagnostics require a nonempty training batch")
    info = model.loop_learning_info
    for key in ("arm", "base_unique_blocks"):
        if key in spec and spec[key] != info[key]:
            raise ValueError(f"Diagnostic spec {key} differs from the actual model")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Loop learning diagnostics expect the fully trainable model")
    target = torch.device(device)
    devices = []
    if target.type == "cuda":
        devices = [target.index if target.index is not None else torch.cuda.current_device()]
    start = time.monotonic()
    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        reference = copy.deepcopy(model).to(target).eval()
        expanded = _unfold(model).to(target).eval()
        reference.zero_grad(set_to_none=True)
        expanded.zero_grad(set_to_none=True)
        tokens, positions, labels = pack(records, pad, target)
        with torch.autocast(target.type, enabled=False):
            reference_logits = reference(tokens, positions)
            reference_loss = example_losses(reference_logits, labels).mean()
            reference_loss.backward()
            expanded_logits = expanded(tokens, positions)
            expanded_loss = example_losses(expanded_logits, labels).mean()
            expanded_loss.backward()
        base = info["base_unique_blocks"]
        actual_blocks = [
            block for _ in range(reference.repeats) for block in reference.transformer.h
        ]
        groups = []
        for index in range(base):
            occurrences = list(range(index, len(expanded.transformer.h), base))
            for component, attribute in (("attention", "attn"), ("mlp", "mlp")):
                modules = [getattr(actual_blocks[i], attribute) for i in occurrences]
                vectors = [
                    _gradient_vector(getattr(expanded.transformer.h[i], attribute))
                    for i in occurrences
                ]
                tied = len({id(module) for module in modules}) == 1 and len(modules) > 1
                shared_error = (
                    float(
                        (_gradient_vector(modules[0]) - torch.stack(vectors).sum(dim=0)).abs().max()
                    )
                    if tied
                    else None
                )
                groups.append(
                    {
                        "base_block": index,
                        "component": component,
                        "executed_block_indices": occurrences,
                        "loop_iterations": [i // base for i in occurrences],
                        "actual_unique_modules": len({id(module) for module in modules}),
                        "actually_shared_across_occurrences": tied,
                        "shared_gradient_maximum_absolute_error": shared_error,
                        **_gradient_geometry(vectors),
                    }
                )
        result = {
            "kind": "current_weights_occurrence_gradient_geometry",
            "arm": info["arm"],
            "base_unique_blocks": base,
            "executed_blocks": len(expanded.transformer.h),
            "record_count": len(records),
            "record_ids": [record.get("id") for record in records],
            "encoded_records_sha256": hashlib.sha256(
                json.dumps(
                    [record["encoded"] for record in records], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "training_batch_selection": "fixed training-only batch supplied by the caller",
            "loss_definition": "mean of per-example mean answer-and-EOS cross entropy",
            "dropout_enabled": False,
            "parameter_dtype": str(next(reference.parameters()).dtype),
            "autocast_enabled": False,
            "reference_loss": float(reference_loss.detach()),
            "expanded_loss": float(expanded_loss.detach()),
            "forward_maximum_absolute_error": float(
                (reference_logits.detach() - expanded_logits.detach()).abs().max()
            ),
            "loss_absolute_error": float((reference_loss.detach() - expanded_loss.detach()).abs()),
            "gradient_reconstruction": _gradient_reconstruction(reference, expanded),
            "groups": groups,
            "interpretation": (
                "Local derivatives at the same current weights. Attention and MLP reports "
                "exclude their LayerNorm parameters; the complete gradient reconstruction "
                "includes LayerNorm, embeddings and output readout. Sums for independent "
                "modules describe a coordinated perturbation, not an actual shared parameter. "
                "No optimizer update, held-out prediction, or causal claim about learning "
                "efficiency is made by this diagnostic."
            ),
        }
        # Reject nonfinite output rather than silently writing nonstandard JSON.
        json.dumps(result, allow_nan=False)
    result["diagnostic_wall_seconds"] = time.monotonic() - start
    return result
