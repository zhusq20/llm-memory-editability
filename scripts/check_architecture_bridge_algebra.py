#!/usr/bin/env python3
"""CPU float64 algebra checks, not evidence about pretrained model mechanisms.

Scope and assumptions, fixed before running:
* MoE: inputs, expert features, router weights and shared-expert gates are fixed.
  Concatenate their weighted features; only the unconstrained down matrices vary.
  P1 uses the Euclidean/Frobenius parameter metric and finite keep/edit columns.
  Numerical rank uses an explicit absolute threshold; it is not exact arithmetic.
* Hard top-k: expert independence is conditional on unchanged routing. Crossing a
  selection boundary need not be continuous, so a smooth Taylor claim is invalid.
* Gated DeltaNet: S_t = S_(t-1) A_t + W_t, with fixed keys, values and gates,
  A_t = alpha_t (I - beta_t k_t k_t^T), W_t = beta_t v_t k_t^T. Perturbations
  are injected into the state, not silently equated with parameter updates.
* Two-hop: the complete interface includes a nonlinear query and a bypass state.
  Smooth finite-dimensional examples check the chain rule and local O(step^2)
  remainders; omitting a live bypass can produce an O(step) error.
* SwiGLU: B [phi(x+delta)-phi(x)] is exact for fixed B, up and gate. Applying a
  nonlinear downstream readout still needs a finite bound or Taylor remainder.

All cases are constructed examples. They neither prove universality nor certify
GLM, Kimi, Qwen, natural-language reasoning, learned routing or training dynamics.
No model loading, fitting, training, GPU work or access to previous artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/development-artifacts/architecture-bridge-v1/algebra-checks.json"
DEFAULT_SEED = 20260929
RANK_RTOL = 1e-12


def _array(value):
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("All inputs must be finite CPU float64 arrays.")
    return result


def fixed_feature_edit(keep, edit, target, *, rank_rtol=RANK_RTOL):
    """Minimum Frobenius down update; reject an incompatible numerical target.

    Feature columns are samples/positions. A common absolute cutoff based on
    the *original* feature scale avoids promoting projection roundoff to a free
    direction when edit features are already in the keep span.
    """
    keep, edit, target = map(_array, (keep, edit, target))
    if any(matrix.ndim != 2 for matrix in (keep, edit, target)):
        raise ValueError("keep, edit and target must be matrices.")
    if keep.shape[0] != edit.shape[0] or target.shape[1] != edit.shape[1]:
        raise ValueError("Feature dimensions and target columns must agree.")
    if rank_rtol <= 0 or not np.isfinite(rank_rtol):
        raise ValueError("rank_rtol must be finite and positive.")
    scale = max(float(np.linalg.norm(keep)), float(np.linalg.norm(edit)), 1.0)
    cutoff = rank_rtol * scale
    _, singular, vh = np.linalg.svd(keep.T, full_matrices=True)
    basis = vh[int(np.count_nonzero(singular > cutoff)) :].T
    reduced = basis.T @ edit
    u, singular, vh = np.linalg.svd(reduced, full_matrices=False)
    active = singular > cutoff
    inverse = (vh[active].T / singular[active]) @ u[:, active].T
    incompatibility = float(np.linalg.norm(target - target @ inverse @ reduced))
    compatibility_tolerance = 20 * rank_rtol * max(float(np.linalg.norm(target)), 1.0)
    if incompatibility > compatibility_tolerance:
        raise ValueError("Target is incompatible with the numerical keep nullspace.")
    delta = target @ inverse @ basis.T
    return delta, {
        "projection": basis @ basis.T,
        "residual": basis @ reduced,
        "keep_rank": keep.shape[0] - basis.shape[1],
        "residual_rank": int(active.sum()),
        "absolute_singular_cutoff": cutoff,
        "compatibility_error": incompatibility,
    }


def pack_moe_features(expert_features, router, shared_features, shared_gate):
    """Return weighted concatenation with examples in columns."""
    expert_features, router, shared_features, shared_gate = map(
        _array, (expert_features, router, shared_features, shared_gate)
    )
    if expert_features.ndim != 3 or shared_features.ndim != 2:
        raise ValueError("Expert features must be [expert, feature, sample].")
    experts, width, samples = expert_features.shape
    if router.shape != (experts, samples) or shared_gate.shape != (samples,):
        raise ValueError("Router/shared gate dimensions must match samples.")
    if shared_features.shape[1] != samples:
        raise ValueError("Shared feature columns must match samples.")
    routed = (expert_features * router[:, None, :]).reshape(experts * width, samples)
    return np.concatenate([routed, shared_features * shared_gate], axis=0)


def topk_router(logits, k):
    """Hard top-k with selected-logit softmax; ties use index order."""
    logits = _array(logits)
    if logits.ndim != 2 or not isinstance(k, int) or not 1 <= k <= logits.shape[0]:
        raise ValueError("Use [expert, sample] logits and an integer valid k.")
    selected = np.argsort(-logits, axis=0, kind="stable")[:k]
    weights = np.zeros_like(logits)
    columns = np.arange(logits.shape[1])
    values = logits[selected, columns]
    exponentials = np.exp(values - values.max(axis=0, keepdims=True))
    weights[selected, columns] = exponentials / exponentials.sum(axis=0, keepdims=True)
    return weights


def gdn_transition(key, value, alpha, beta):
    key, value = map(_array, (key, value))
    if key.ndim != 1 or value.ndim != 1:
        raise ValueError("Keys and values must be vectors.")
    if not np.isfinite(alpha) or not np.isfinite(beta):
        raise ValueError("Gates must be finite.")
    return alpha * (np.eye(key.size) - beta * np.outer(key, key)), beta * np.outer(value, key)


def right_product(transitions):
    if not transitions:
        raise ValueError("Use an explicit identity for an empty transition product.")
    product = np.eye(transitions[0].shape[0])
    for transition in transitions:
        product = product @ transition
    return product


def gdn_rollout(initial, transitions, writes, injections=None):
    if len(transitions) != len(writes):
        raise ValueError("Each transition needs one write.")
    injections = {} if injections is None else injections
    if any(not isinstance(t, int) or t < 0 or t >= len(writes) for t in injections):
        raise ValueError("Injection indexes must be inside the rollout.")
    state = _array(initial).copy()
    states = []
    for t, (transition, write) in enumerate(zip(transitions, writes, strict=True)):
        state = state @ transition + write
        if t in injections:
            state = state + injections[t]
        states.append(state.copy())
    return states


def gdn_expansion(initial, transitions, writes):
    """Closed expansion in chronological right-product order."""
    if len(transitions) != len(writes):
        raise ValueError("Each transition needs one write.")
    if not transitions:
        return _array(initial).copy()
    final = initial @ right_product(transitions)
    for t, write in enumerate(writes):
        suffix = transitions[t + 1 :]
        final = final + (write @ right_product(suffix) if suffix else write)
    return final


def interface_state(theta):
    t0, t1, t2 = theta
    return np.array([np.tanh(t0 + 0.3 * t1), t1**2 + 0.4 * t0, t2 + np.sin(t0)])


def interface_jacobian(theta):
    t0, t1, _ = theta
    sech2 = 1 - np.tanh(t0 + 0.3 * t1) ** 2
    return np.array([[sech2, 0.3 * sech2, 0], [0.4, 2 * t1, 0], [np.cos(t0), 0, 1]])


def nonlinear_head(state):
    u, bypass = state[:2], state[2]
    transform = np.array([[0.8, -0.4], [0.2, 0.7]])
    mixed = transform @ u + np.array([0.5, -0.3]) * bypass
    query = np.tanh(mixed) + 0.1 * u**2
    inner = query[0] + 0.4 * query[1] + 0.5 * bypass
    return np.tanh(inner) + 0.3 * query[0] * query[1] + 0.2 * bypass**2


def nonlinear_head_jacobian(state):
    u, bypass = state[:2], state[2]
    transform = np.array([[0.8, -0.4], [0.2, 0.7]])
    bypass_map = np.array([0.5, -0.3])
    nonlinear = np.tanh(transform @ u + bypass_map * bypass)
    query = nonlinear + 0.1 * u**2
    sech2 = 1 - np.tanh(query[0] + 0.4 * query[1] + 0.5 * bypass) ** 2
    grad_query = sech2 * np.array([1, 0.4]) + 0.3 * query[::-1]
    query_u = np.diag(1 - nonlinear**2) @ transform + np.diag(0.2 * u)
    query_bypass = (1 - nonlinear**2) * bypass_map
    return np.append(grad_query @ query_u, grad_query @ query_bypass + 0.5 * sech2 + 0.4 * bypass)


def torch_head(state):
    u, bypass = state[:2], state[2]
    transform = torch.tensor([[0.8, -0.4], [0.2, 0.7]], dtype=torch.float64, device="cpu")
    bypass_map = torch.tensor([0.5, -0.3], dtype=torch.float64, device="cpu")
    query = torch.tanh(transform @ u + bypass_map * bypass) + 0.1 * u.square()
    return (
        torch.tanh(query[0] + 0.4 * query[1] + 0.5 * bypass)
        + 0.3 * query[0] * query[1]
        + 0.2 * bypass.square()
    )


def torch_full(theta):
    t0, t1, t2 = theta.unbind()
    state = torch.stack([torch.tanh(t0 + 0.3 * t1), t1.square() + 0.4 * t0, t2 + torch.sin(t0)])
    return torch_head(state)


def autodiff_gradient(function, point):
    """Independent CPU float64 automatic differentiation of the smooth example."""
    point = torch.tensor(point, dtype=torch.float64, device="cpu", requires_grad=True)
    return torch.autograd.grad(function(point), point)[0].detach().numpy()


def silu(value):
    return value / (1 + np.exp(-value))


class Recorder:
    def __init__(self):
        self.checks = []

    def close(self, name, actual, expected=0.0, tolerance=1e-10):
        error = float(np.linalg.norm(np.asarray(actual) - np.asarray(expected)))
        self.checks.append(
            dict(
                name=name,
                kind="identity",
                error=error,
                tolerance=tolerance,
                passed=error <= tolerance,
            )
        )

    def interval(self, name, value, lower, upper):
        value = float(value)
        self.checks.append(
            dict(
                name=name,
                kind="inequality_or_counterexample",
                value=value,
                lower=lower,
                upper=upper,
                passed=bool(np.isfinite(value) and lower <= value <= upper),
            )
        )


def check_moe(recorder, rng):
    inputs = rng.normal(size=(5, 9))
    ups = rng.normal(size=(3, 4, 5))
    gates = rng.normal(size=(3, 4, 5))
    features = (ups @ inputs) * silu(gates @ inputs)
    router = topk_router(rng.normal(size=(3, 9)), 2)
    shared = (rng.normal(size=(3, 5)) @ inputs) * silu(rng.normal(size=(3, 5)) @ inputs)
    shared_gate = 0.6 + 0.3 / (1 + np.exp(-inputs[0]))
    packed = pack_moe_features(features, router, shared, shared_gate)
    down = rng.normal(size=(2, packed.shape[0]))
    explicit = sum(down[:, 4 * e : 4 * (e + 1)] @ (features[e] * router[e]) for e in range(3))
    explicit += down[:, 12:] @ (shared * shared_gate)
    recorder.close("moe.weighted_concat_including_shared", down @ packed, explicit)
    keep, edit = packed[:, :4], packed[:, 4:7]
    target = rng.normal(size=(2, 3))
    delta, info = fixed_feature_edit(keep, edit, target)
    recorder.close("moe.P1_keep", delta @ keep)
    recorder.close("moe.P1_target", delta @ edit, target)
    recorder.close("moe.P1_compatibility", info["compatibility_error"])
    recorder.close("moe.P1_projected_update", delta @ info["projection"], delta)
    constraint = np.concatenate([keep, edit], axis=1)
    rhs = np.concatenate([np.zeros((2, 4)), target], axis=1)
    independent = np.linalg.lstsq(constraint.T, rhs.T, rcond=RANK_RTOL)[0].T
    recorder.close("moe.P1_independent_minimum_norm_solver", delta, independent)
    _, singular, vh = np.linalg.svd(constraint.T, full_matrices=True)
    rank = np.count_nonzero(singular > RANK_RTOL * max(np.linalg.norm(constraint), 1.0))
    free = rng.normal(size=(2, constraint.shape[0] - rank)) @ vh[rank:]
    recorder.close("moe.minimum_norm_free_direction_feasible", free @ constraint)
    recorder.close("moe.minimum_norm_orthogonality", np.sum(delta * free))
    recorder.close(
        "moe.minimum_norm_pythagoras",
        np.linalg.norm(delta + free) ** 2,
        np.linalg.norm(delta) ** 2 + np.linalg.norm(free) ** 2,
    )
    recorder.interval("moe.alternative_solution_has_larger_norm", np.linalg.norm(free), 0.1, 100)
    rejected = False
    try:
        fixed_feature_edit(keep, keep[:, :1].copy(), np.ones((2, 1)))
    except ValueError:
        rejected = True
    recorder.close("moe.keep_span_incompatible_target_rejected", float(rejected), 1.0)
    return {key: value for key, value in info.items() if not isinstance(value, np.ndarray)}


def check_router(recorder):
    logits = np.array([[2.0, -2.0], [-2.0, 2.0]])
    weights = topk_router(logits, 1)
    feature = np.ones((2, 1, 2), dtype=np.float64)
    packed = pack_moe_features(feature, weights, np.ones((1, 2)), np.ones(2))
    delta = np.array([[3.0, 0.0, 0.0]])
    recorder.close("router.disjoint_expert_keep", (delta @ packed)[0, 1])
    recorder.close("router.selected_expert_change", (delta @ packed)[0, 0], 3.0)
    shared_delta = np.array([[0.0, 0.0, 1.0]])
    recorder.close("router.shared_expert_change_leaks_to_keep", (shared_delta @ packed)[0, 1], 1.0)
    scales = [1e-1, 1e-3, 1e-6]
    jumps = []
    for epsilon in scales:
        left = topk_router(np.array([[-epsilon], [epsilon]]), 1)
        right = topk_router(np.array([[epsilon], [-epsilon]]), 1)
        outputs = np.array([[1.0, -1.0]])
        jump = float((outputs @ (right - left)).item())
        jumps.append(dict(input_distance=2 * epsilon, output_jump=jump))
        recorder.close(f"router.boundary_jump_at_{epsilon:g}", jump, 2.0)
    recorder.interval("router.boundary_secant_unbounded_example", 2 / (2 * scales[-1]), 1e5, 1e7)
    return {"boundary_crossings": jumps, "branch_derivative": 0.0, "smooth_at_boundary": False}


def check_gdn(recorder, rng):
    keys = rng.normal(size=(5, 3))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    values = rng.normal(size=(5, 2))
    alphas = np.array([0.9, 0.8, 0.7, 0.95, 0.85])
    betas = np.array([0.3, 0.7, 0.4, 0.8, 0.6])
    pairs = [
        gdn_transition(k, v, a, b) for k, v, a, b in zip(keys, values, alphas, betas, strict=True)
    ]
    transitions, writes = map(list, zip(*pairs, strict=True))
    initial = rng.normal(size=(2, 3))
    states = gdn_rollout(initial, transitions, writes)
    for t, state in enumerate(states):
        recorder.close(
            f"gdn.full_expansion_t{t}",
            gdn_expansion(initial, transitions[: t + 1], writes[: t + 1]),
            state,
        )
    injections = {1: rng.normal(size=(2, 3)), 3: rng.normal(size=(2, 3))}
    altered = gdn_rollout(initial, transitions, writes, injections)
    propagation = np.zeros_like(initial)
    for t in range(len(states)):
        propagation = propagation @ transitions[t] + injections.get(t, 0.0)
        recorder.close(f"gdn.state_injection_propagation_t{t}", altered[t] - states[t], propagation)
    suffix = right_product(transitions[2:])
    single = gdn_rollout(initial, transitions, writes, {1: injections[1]})[-1] - states[-1]
    recorder.close("gdn.single_injection_chronological_product", single, injections[1] @ suffix)
    changed_writes = list(writes)
    value_delta = np.array([0.3, -0.2])
    _, changed_writes[1] = gdn_transition(keys[1], values[1] + value_delta, alphas[1], betas[1])
    actual_write_response = gdn_rollout(initial, transitions, changed_writes)[-1] - states[-1]
    write_injection = betas[1] * np.outer(value_delta, keys[1])
    recorder.close(
        "gdn.changed_value_write_propagation", actual_write_response, write_injection @ suffix
    )
    reverse = right_product(list(reversed(transitions[2:])))
    recorder.interval(
        "gdn.reversing_noncommuting_product_is_wrong",
        np.linalg.norm(single - injections[1] @ reverse),
        1e-5,
        100,
    )
    bound = float(np.linalg.norm(injections[1]) * np.prod(alphas[2:]))
    recorder.interval(
        "gdn.unit_key_gate_contraction_bound", np.linalg.norm(single), 0, bound + 1e-12
    )
    modified, key_changed_writes = list(transitions), list(writes)
    modified[-1], key_changed_writes[-1] = gdn_transition(
        keys[0], values[-1], alphas[-1], betas[-1]
    )
    with_changed_key = gdn_rollout(initial, modified, key_changed_writes, {1: injections[1]})[-1]
    recorder.interval(
        "gdn.changed_future_key_invalidates_fixed_transition_formula",
        np.linalg.norm(with_changed_key - states[-1] - single),
        1e-5,
        100,
    )
    return {"sequence_length": 5, "key_dim": 3, "value_dim": 2, "state_injection_bound": bound}


def check_twohop(recorder):
    theta = np.array([0.3, -0.4, 0.2])
    direction = np.array([0.6, 0.3, -0.7])
    state = interface_state(theta)
    head_jac = nonlinear_head_jacobian(state)
    total_jac = head_jac @ interface_jacobian(theta)
    function = lambda point: nonlinear_head(interface_state(point))  # noqa: E731
    recorder.close(
        "twohop.complete_state_chain_rule", total_jac, autodiff_gradient(torch_full, theta)
    )
    recorder.close("twohop.head_derivative", head_jac, autodiff_gradient(torch_head, state))
    reduced_jac = head_jac[:2] @ interface_jacobian(theta)[:2]
    rows = []
    for step in [0.04, 0.02, 0.01, 0.005]:
        change = function(theta + step * direction) - function(theta)
        rows.append(
            dict(
                step=step,
                complete_error=abs(float(change - step * total_jac @ direction)),
                omitted_bypass_error=abs(float(change - step * reduced_jac @ direction)),
            )
        )
    for index in range(len(rows) - 1):
        full_order = np.log2(rows[index]["complete_error"] / rows[index + 1]["complete_error"])
        missing_order = np.log2(
            rows[index]["omitted_bypass_error"] / rows[index + 1]["omitted_bypass_error"]
        )
        recorder.interval(f"twohop.complete_state_second_order_{index}", full_order, 1.9, 2.1)
        recorder.interval(f"twohop.omitted_bypass_first_order_{index}", missing_order, 0.85, 1.15)
    recorder.interval(
        "twohop.omitted_bypass_derivative_is_nonzero",
        abs(float((total_jac - reduced_jac) @ direction)),
        1e-3,
        100,
    )
    return {
        "step_halving": rows,
        "complete_directional_derivative": float(total_jac @ direction),
        "omitted_bypass_directional_derivative": float(reduced_jac @ direction),
    }


def check_finite_features(recorder):
    up = np.array([[0.8, -0.2], [0.3, 0.9], [-0.6, 0.5]])
    gate = np.array([[0.2, 0.7], [-0.4, 0.6], [0.8, -0.3]])
    down = np.array([[0.5, -0.3, 0.8], [0.2, 0.7, -0.1]])
    readout = np.array([0.7, -0.4])
    x, direction = np.array([0.4, -0.5]), np.array([0.3, 0.8])
    feature = lambda point: (up @ point) * silu(gate @ point)  # noqa: E731
    h0 = down @ feature(x)
    maximum_hessian = 4 / (3 * np.sqrt(3)) * np.linalg.norm(readout) ** 2
    rows = []
    for step in [1.0, 0.5, 0.25, 0.125]:
        h1 = down @ feature(x + step * direction)
        finite_feature_response = down @ (feature(x + step * direction) - feature(x))
        recorder.close(
            f"features.exact_finite_module_response_{step}", h1 - h0, finite_feature_response
        )
        recorder.close(
            f"features.exact_linear_readout_{step}",
            readout @ h1 - readout @ h0,
            readout @ finite_feature_response,
        )
        actual = float(np.tanh(readout @ h1) - np.tanh(readout @ h0))
        local = float((1 - np.tanh(readout @ h0) ** 2) * readout @ finite_feature_response)
        remainder = abs(actual - local)
        finite_bound = 0.5 * maximum_hessian * np.linalg.norm(finite_feature_response) ** 2
        recorder.interval(
            f"features.certified_downstream_taylor_bound_{step}", remainder, 0, finite_bound + 1e-15
        )
        recorder.interval(
            f"features.global_downstream_lipschitz_bound_{step}",
            abs(actual),
            0,
            np.linalg.norm(readout) * np.linalg.norm(finite_feature_response) + 1e-15,
        )
        rows.append(
            dict(
                step=step,
                actual_change=actual,
                feature_plus_local_downstream=local,
                remainder=remainder,
                analytic_remainder_bound=float(finite_bound),
            )
        )
    recorder.interval(
        "features.nonlinear_downstream_is_not_an_exact_finite_formula",
        rows[0]["remainder"],
        1e-8,
        1,
    )
    h_before = float(2 * silu(np.array(2.0)))
    h_after = float(silu(np.array(1.0)))
    local_bound = (1 - np.tanh(h_before) ** 2) * abs(h_after - h_before)
    actual_change = abs(np.tanh(h_after) - np.tanh(h_before))
    recorder.interval(
        "features.point_derivative_is_not_a_finite_lipschitz_certificate",
        actual_change - local_bound,
        0.01,
        2,
    )
    return {
        "perturbations": rows,
        "global_tanh_hessian_bound": float(maximum_hessian),
        "point_derivative_counterexample": {
            "actual_change": float(actual_change),
            "incorrect_local_bound": float(local_bound),
        },
    }


def run_checks(seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    recorder = Recorder()
    details = {
        "fixed_router_moe": check_moe(recorder, rng),
        "hard_router_boundary": check_router(recorder),
        "gdn_frozen_state_transition": check_gdn(recorder, rng),
        "twohop_complete_interface": check_twohop(recorder),
        "finite_features_and_downstream": check_finite_features(recorder),
    }
    identities = [item["error"] for item in recorder.checks if item["kind"] == "identity"]
    return {
        "batch": "architecture-bridge-v1",
        "scope": __doc__,
        "seed": seed,
        "precision": "float64",
        "device": "CPU",
        "rank_rtol": RANK_RTOL,
        "count": len(recorder.checks),
        "all_passed": all(item["passed"] for item in recorder.checks),
        "max_identity_error": max(identities),
        "checks": recorder.checks,
        "details": details,
        "conclusion": {
            "numerical_consistency": (
                "Constructed fixed-router, fixed-transition and smooth-interface identities "
                "pass only if all_passed is true."
            ),
            "counterexamples": (
                "Hard route switches, changed future GDN keys, omitted bypasses and pointwise "
                "derivative bounds invalidate the corresponding stronger claims."
            ),
            "mechanism_evidence": False,
            "not_established": [
                "pretrained model mechanisms",
                "cross-family empirical generalization",
                "natural-language two-hop failure causes",
                "long-term learning dynamics",
            ],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_checks(args.seed)
    paths = [Path(__file__), ROOT / "tests/test_architecture_bridge_algebra.py"]
    result["provenance"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "wall_seconds": time.perf_counter() - started,
        "model_or_data_downloaded": False,
        "model_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("count", "all_passed", "max_identity_error")}))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
