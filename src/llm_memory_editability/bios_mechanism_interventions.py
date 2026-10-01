"""Uninstantiated P4 projection diagnostics and equal-parameter branches.

These operators do not choose a real layer, fit a real model's basis, read world
membership truth, or run an experiment. The caller must freeze a candidate and
an information budget first. Group means are not, by themselves, evidence of
an organizational representation.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Site:
    layer: int
    position: int


@dataclass(frozen=True)
class Basis:
    q: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @property
    def rank(self):
        return self.q.shape[1]

    def tensors(self, device):
        return tuple(
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in (self.q, self.mean, self.scale)
        )


def calibration_plan(actual_query_ids, person_by_query, membership_query_ids, supervised, replay):
    """Only actual queries in S union R; bridge IDs specify additional self-queries.

    All arguments are query/identity metadata. No group truth or answers are
    accepted. In particular this never constructs a derived query to fit Q.
    """
    allowed = np.union1d(supervised, replay)
    actual = np.intersect1d(actual_query_ids, allowed)
    people = np.asarray(person_by_query)[actual]
    if (people < 0).any() or len(np.unique(people)) != len(people):
        raise ValueError("Calibration must contain one actual query per valid person")
    membership = np.asarray(membership_query_ids)[people]
    return {
        "actual_query_ids": actual,
        "person_ids": people,
        "membership_query_ids": membership,
        "logical_actual_queries": len(actual),
        "logical_additional_membership_queries": len(membership),
        "derived_calibration_queries": 0,
    }


def valid_predicted_groups(prediction, ended, allowed_group_tokens):
    prediction, ended = np.asarray(prediction), np.asarray(ended)
    if prediction.ndim != 1 or ended.shape != prediction.shape or ended.dtype != bool:
        raise ValueError("Group predictions require one value and a boolean termination mask")
    valid = ended & np.isin(prediction, allowed_group_tokens)
    # Invalid predictions are recorded, never replaced with the true group.
    return np.where(valid, prediction, -1), valid


def _row_span(values, relative_tolerance=1e-6, absolute_tolerance=1e-8):
    _, singular, vh = np.linalg.svd(values, full_matrices=False)
    threshold = (
        max(absolute_tolerance, relative_tolerance * singular[0])
        if len(singular)
        else absolute_tolerance
    )
    numerical_rank = int((singular > threshold).sum())
    return vh[:numerical_rank].T, singular, numerical_rank


def _orthogonal_residual(values, q):
    return values - (values @ q) @ q.T if q.shape[1] else values.copy()


def _normalized_basis(q, calibration):
    projected = calibration @ q
    mean, std = projected.mean(0), projected.std(0)
    floor = max(float(np.sqrt(np.mean(std * std))) * 0.01, 1e-8)
    std = np.maximum(std, floor)
    unit = (projected - mean) / std
    total_std = float(np.sqrt(np.mean(np.sum(unit * unit, axis=-1))))
    if not np.isfinite(total_std) or total_std <= 1e-8:
        raise ValueError("A nonzero calibration variance is required")
    basis = Basis(
        q.astype(np.float32), mean.astype(np.float32), (std * total_std).astype(np.float32)
    )
    if not np.allclose(basis.q.T @ basis.q, np.eye(q.shape[1]), atol=1e-5):
        raise ValueError("The constructed basis is not orthonormal")
    return basis


def fit_predicted_group_bases(
    activations,
    predicted_groups,
    valid,
    *,
    rank=32,
    seed=74,
    variant="group_means",
    predicted_answers=None,
    answer_valid=None,
):
    """Construct a target, orthogonal within-group complement, and random basis.

    Main method input: permitted actual-query states and model-predicted groups.
    The optional answer-centered variant is a diagnostic until separately
    preregistered. It uses model predictions from those same actual queries;
    it does not request default/derived answer truth or extra derived states.
    Rank failure returns unavailable; random directions never fill missing rank.
    """
    x = np.asarray(activations, dtype=np.float64)
    groups, used = np.asarray(predicted_groups), np.asarray(valid).copy()
    if x.ndim != 2 or groups.shape != (len(x),) or used.shape != groups.shape or used.dtype != bool:
        raise ValueError("Mismatched calibration arrays")
    if not np.isfinite(x).all() or rank < 1 or rank > x.shape[1]:
        raise ValueError("Finite activations and a feasible positive rank are required")
    if (groups[used] < 0).any():
        raise ValueError("Invalid group predictions cannot be marked valid")
    if variant not in ("group_means", "answer_centered_diagnostic"):
        raise ValueError("Unknown basis variant")
    answers = None
    if variant == "answer_centered_diagnostic":
        if predicted_answers is None or answer_valid is None:
            raise ValueError("Answer centering requires model answers and validity flags")
        answers, av = np.asarray(predicted_answers), np.asarray(answer_valid)
        if answers.shape != groups.shape or av.shape != groups.shape or av.dtype != bool:
            raise ValueError("Mismatched predicted answer arrays")
        used &= av
    report = {
        "available": False,
        "variant": variant,
        "rank_requested": rank,
        "calibration_rows": len(x),
        "used_rows": int(used.sum()),
        "invalid_rows": int((~used).sum()),
        "uses_group_truth": False,
        "uses_derived_calibration": False,
        "random_seed": seed,
    }
    if not used.any():
        return None, {**report, "reason": "No valid model-predicted calibration groups"}
    values, labels = x[used], groups[used]
    unique, inverse = np.unique(labels, return_inverse=True)
    group_means = np.stack([values[inverse == index].mean(0) for index in range(len(unique))])
    target_matrix = group_means - group_means.mean(0)
    answer_span = np.empty((x.shape[1], 0))
    if answers is not None:
        answer_labels = answers[used]
        answer_means = np.stack(
            [values[answer_labels == label].mean(0) for label in np.unique(answer_labels)]
        )
        answer_span, _, _ = _row_span(answer_means - answer_means.mean(0))
        target_matrix = _orthogonal_residual(target_matrix, answer_span)
    target_span, singular, target_rank = _row_span(target_matrix)
    report.update(
        predicted_groups=len(unique),
        answer_span_rank=answer_span.shape[1],
        target_numerical_rank=target_rank,
        target_singular_values=singular.tolist(),
    )
    if target_rank < rank:
        return None, {
            **report,
            "reason": "Target has fewer than the required independent directions",
        }
    target = target_span[:, :rank]
    residual = values - group_means[inverse]
    residual = _orthogonal_residual(_orthogonal_residual(residual, answer_span), target)
    complement_span, complement_singular, complement_rank = _row_span(residual)
    report.update(
        complement_numerical_rank=complement_rank,
        complement_singular_values=complement_singular.tolist(),
    )
    if complement_rank < rank:
        return None, {
            **report,
            "reason": "Complement has fewer than the required independent directions",
        }
    if x.shape[1] - answer_span.shape[1] < rank:
        return None, {**report, "reason": "Insufficient common admissible dimension"}
    random_values = np.random.default_rng(seed).standard_normal((x.shape[1], rank))
    random_values = _orthogonal_residual(random_values.T, answer_span).T
    random, triangular = np.linalg.qr(random_values, mode="reduced")
    if (np.abs(np.diag(triangular)) < 1e-8).any():
        return None, {**report, "reason": "Random control is numerically rank deficient"}
    bases = {
        name: _normalized_basis(q, values)
        for name, q in (
            ("target", target),
            ("complement", complement_span[:, :rank]),
            ("random", random),
        )
    }
    report["normalization"] = {
        name: {
            "rank": basis.rank,
            "total_variance": float(
                np.mean(np.sum(((values @ basis.q - basis.mean) / basis.scale) ** 2, axis=1))
            ),
            "answer_span_overlap": float(np.linalg.norm(answer_span.T @ basis.q)),
        }
        for name, basis in bases.items()
    }
    return bases, {**report, "available": True, "reason": None}


def projection_component(values, q, mean):
    # Explicit FP32 is necessary: .float() alone does not disable an outer
    # autocast context for matrix products in a forward hook.
    with torch.autocast(device_type=values.device.type, enabled=False):
        return (values.float() @ q.float() - mean.float()) @ q.float().T


def match_delta_norm(delta, reference_norms, tolerance=1e-12):
    """Zero-length controls with a positive target norm are explicitly unusable."""
    norms = torch.linalg.vector_norm(delta.float(), dim=-1)
    reference_norms = torch.as_tensor(reference_norms, dtype=norms.dtype, device=norms.device)
    if reference_norms.shape != norms.shape or bool((reference_norms < 0).any()):
        raise ValueError("One nonnegative reference norm is required per case")
    valid = (norms > tolerance) | (reference_norms <= tolerance)
    scale = torch.where(
        valid, reference_norms / norms.clamp_min(tolerance), torch.zeros_like(norms)
    )
    return delta * scale[:, None], valid


def remove_projection(values, q, mean, reference_norms=None):
    delta = -projection_component(values, q, mean)
    valid = torch.ones(len(values), device=values.device, dtype=torch.bool)
    if reference_norms is not None:
        delta, valid = match_delta_norm(delta, reference_norms)
    return values + delta.to(values.dtype), {"delta": delta, "valid": valid}


def restore_projection(values, clean_source, q, reference_norms=None):
    """Restore only a fixed later-site projection, not an entire clean activation.

    clean_source must be from the same declared checkpoint and clean query, or
    an explicitly labeled wrong-source control. A pre-edit source in an edited
    model is a separate diagnostic and may undo the edit; it is never implicit.
    """
    if values.shape != clean_source.shape:
        raise ValueError("Restoration source and receiving states must have the same shape")
    with torch.autocast(device_type=values.device.type, enabled=False):
        delta = ((clean_source.float() - values.float()) @ q.float()) @ q.float().T
    valid = torch.ones(len(values), device=values.device, dtype=torch.bool)
    if reference_norms is not None:
        delta, valid = match_delta_norm(delta, reference_norms)
    return values + delta.to(values.dtype), {"delta": delta, "valid": valid}


def validate_restoration_sites(source, receiver, layers, answer_positions, allow_readout=False):
    """A later relational post-block must still have a path to the value readout."""
    if not (0 <= source.layer < receiver.layer < layers):
        raise ValueError("Restoration must be at a strictly later block, never the ablation point")
    if source.position < 0 or receiver.position < 0:
        raise ValueError("Token positions must be nonnegative")
    if receiver.position < source.position:
        raise ValueError("The receiving token is causally before the source token")
    answer_positions = np.asarray(answer_positions)
    if (answer_positions < receiver.position).any():
        raise ValueError("The receiving state cannot include an answer token")
    if receiver.layer == layers - 1:
        if (receiver.position < answer_positions).any():
            raise ValueError("Final-block earlier-token patches cannot affect value readout")
        if not allow_readout:
            raise ValueError("Final answer-marker restoration is a readout control, not routing")
    return {"readout_control": receiver.layer == layers - 1}


def downstream_routing_site(source, layers, answer_positions):
    """Prespecified L+1/same-token rescue; late candidates have no such rescue."""
    if source.layer > layers - 3:
        raise ValueError("This late candidate has no nonterminal same-token routing rescue")
    receiver = Site(source.layer + 1, source.position)
    validate_restoration_sites(source, receiver, layers, answer_positions)
    return receiver


def query_site_roles(lengths, position):
    lengths = np.asarray(lengths)
    if not np.isin(lengths, [4, 5]).all() or position not in (2, 3, 4):
        raise ValueError("Only the declared basic/derived symbolic query forms are supported")
    if (position >= lengths).any():
        raise ValueError("An intervention cannot use a future answer-token position")
    return np.where(
        position == lengths - 1,
        "answer_marker",
        "first_relation" if position == 2 else "second_relation",
    )


def attach_state_operation(model, site, operation):
    """Attach a batch-local operation; caller supplies fixed source/control states.

    Operations return (new_state, diagnostics). Prompt and EOS passes may use
    the same pre-answer operation; future answer tokens must never be sources.
    """
    if site.layer < 0 or site.layer >= len(model.blocks) or site.position < 0:
        raise ValueError("Invalid intervention site")
    traces = []

    def intervene(_module, _inputs, output):
        if site.position >= output.shape[1]:
            raise ValueError("Intervention site is outside this query")
        changed, trace = operation(output[:, site.position])
        if changed.shape != output[:, site.position].shape:
            raise ValueError("An intervention changed the residual dimension")
        result = output.clone()
        result[:, site.position] = changed
        traces.append(trace)
        return result

    return model.blocks[site.layer].register_forward_hook(intervene), traces


def attach_equal_branch(model, basis, site, bottleneck=32, seed=73):
    """The main experiment adds this branch to a common trainable MLP scope."""
    from .bios_branch import ResidualBranch

    if hasattr(model, "edit_branch"):
        raise ValueError("An edit branch is already attached")
    if site.layer < 0 or site.layer >= len(model.blocks) or site.position < 0:
        raise ValueError("Invalid branch site")
    if site.position not in (2, 3):
        raise ValueError("Method branches are restricted to locked relation candidates 2/3")
    device = next(model.parameters()).device
    branch = ResidualBranch(*basis.tensors(device), width=bottleneck, seed=seed)
    model.add_module("edit_branch", branch)

    def add(_module, _inputs, output):
        if site.position >= output.shape[1]:
            raise ValueError("Branch site is outside this query")
        changed = output.clone()
        changed[:, site.position] += branch(output[:, site.position])
        return changed

    handle = model.blocks[site.layer].register_forward_hook(add)
    expected = bottleneck * (basis.rank + model.config.width)
    if sum(parameter.numel() for parameter in branch.parameters()) != expected:
        raise ValueError("Unequal branch parameter count")
    return branch, handle


def select_branch_parameters(model, branch, *, train_common_mlp=True, start=3):
    from .bios_model import select_parameters

    if model.edit_branch is not branch:
        raise ValueError("The branch is not attached to this model")
    if train_common_mlp:
        selected = select_parameters(model, "mlp", start)
    else:
        selected = []
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    for parameter in branch.parameters():
        parameter.requires_grad_(True)
        parameter.grad = None
        selected.append(parameter)
    return selected


def remove_equal_branch(model, handle):
    handle.remove()
    delattr(model, "edit_branch")
