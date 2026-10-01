#!/usr/bin/env python3
"""CPU consistency checks for plan §14; these are not empirical mechanism evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CHECKS = []
RNG = np.random.default_rng(20260928)


def small(name, error, tolerance=1e-10):
    error = float(error)
    passed = bool(np.isfinite(error) and error <= tolerance)
    CHECKS.append(dict(name=name, error=error, tolerance=tolerance, passed=passed))
    if not passed:
        raise AssertionError(CHECKS[-1])


def positive(name, value, lower=0.0):
    value = float(value)
    passed = bool(np.isfinite(value) and value > lower)
    CHECKS.append(dict(name=name, value=value, strict_lower_bound=lower, passed=passed))
    if not passed:
        raise AssertionError(CHECKS[-1])


def null_basis(matrix):
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    threshold = max(matrix.shape) * np.finfo(float).eps * singular.max(initial=0)
    rank = int(np.count_nonzero(singular > threshold))
    return vh[rank:].T


def check_down():
    keep = RNG.normal(size=(8, 3))
    edit = RNG.normal(size=(8, 2))
    target = RNG.normal(size=(4, 2))
    projection = np.eye(8) - keep @ np.linalg.pinv(keep)
    residual = projection @ edit
    delta = target @ np.linalg.pinv(residual)
    small("P1.keep_exact", np.linalg.norm(delta @ keep))
    small("P1.edit_exact", np.linalg.norm(delta @ edit - target))
    small("P1.compatibility", np.linalg.norm(target - target @ np.linalg.pinv(residual) @ residual))
    basis = null_basis(np.concatenate([keep, edit], axis=1).T)
    addition = RNG.normal(size=(4, basis.shape[1])) @ basis.T
    small("P1.minimum_norm_orthogonality", abs(np.sum(delta * addition)))
    positive("P1.other_solution_larger", np.linalg.norm(delta + addition) - np.linalg.norm(delta))
    _, singular, vh = np.linalg.svd(residual, full_matrices=False)
    predicted_squared_norm = np.sum(np.sum((target @ vh.T) ** 2, axis=0) / singular**2)
    small("P1.multitarget_spectral_cost", abs(np.linalg.norm(delta) ** 2 - predicted_squared_norm))
    sweep = []
    for epsilon in [1.0, 0.1, 0.01, 0.001]:
        phi = np.array([1.0, epsilon])
        z = np.diag([0.0, 1.0]) @ phi
        change = np.outer(np.array([0.0, 1.0]), z) / (z @ z)
        small(f"P1.inverse_distance_{epsilon}", abs(np.linalg.norm(change) * epsilon - 1.0))
        sweep.append(dict(epsilon=epsilon, minimum_norm=float(np.linalg.norm(change))))
    # With two keep keys spanning R², pairwise similarities do not provide a free direction.
    keys = np.eye(2)
    phi = np.ones(2) / np.sqrt(2)
    small("P1.span_obstruction", np.linalg.norm((np.eye(2) - keys @ np.linalg.pinv(keys)) @ phi))
    return sweep


def silu_numpy(x):
    return x / (1 + np.exp(-x))


def check_swiglu_counterexamples():
    x = np.eye(2)
    gate = np.ones((1, 2))
    down = np.array([[1.0], [0.0]])
    c = float(silu_numpy(np.array(1.0)))

    def output(up, value=down):
        return value @ ((up @ x) * silu_numpy(gate @ x))

    up = np.ones((1, 2))
    features = (up @ x) * silu_numpy(gate @ x)
    small("P2.example_A_equal_features", np.linalg.norm(features[:, 0] - features[:, 1]))
    target = np.array([[0.0, 0.4], [0.0, 0.0]])
    changed = up + np.array([[0.0, 0.4 / c]])
    small("P2.example_A_up_finite_success", np.linalg.norm(output(changed) - output(up) - target))
    best_down = target @ np.linalg.pinv(features)
    positive("P2.example_A_down_incompatible", np.linalg.norm(best_down @ features - target), 0.1)

    up = np.array([[0.0, 1.0]])
    target = np.array([[0.0, 0.0], [0.0, 1.0]])
    change_down = np.array([[0.0], [1.0 / c]])
    small(
        "P2.example_B_down_finite_success",
        np.linalg.norm(output(up, down + change_down) - output(up) - target),
    )
    obstruction = (np.eye(2) - down @ np.linalg.pinv(down)) @ target[:, 1]
    small("P2.example_B_up_unreachable_residual", abs(np.linalg.norm(obstruction) - 1.0))

    up = np.array([[0.0, 1.0]])
    scale = 7.0
    small(
        "metric.same_function_after_rescaling",
        np.linalg.norm(output(scale * up, down / scale) - output(up)),
    )
    original_cost = np.linalg.norm(change_down)
    rescaled_cost = np.linalg.norm(change_down / scale)
    small("metric.down_cost_changes_with_coordinates", abs(rescaled_cost * scale - original_cost))

    origin = torch.zeros(2, dtype=torch.float64)

    def function(v):
        return v[0] * F.silu(v[1])

    jacobian = torch.autograd.functional.jacobian(function, origin)
    small("P2.double_zero_first_derivative", float(jacobian.norm()))
    finite = float(function(torch.full((2,), 1e-3, dtype=torch.float64)))
    positive("P2.double_zero_finite_response", finite)


def check_full_network():
    d, m, p = 3, 4, 3
    up = RNG.normal(size=(m, d))
    gate = RNG.normal(size=(m, d))
    down = RNG.normal(size=(p, m))
    inputs = RNG.normal(size=(2, d))
    dtype = torch.float64
    x = torch.tensor(inputs, dtype=dtype)
    packed = np.concatenate(
        [up.flatten(order="F"), gate.flatten(order="F"), down.flatten(order="F")]
    )
    theta = torch.tensor(packed, dtype=dtype)

    def unpack(vector):
        a = vector[: m * d].reshape(d, m).T
        g = vector[m * d : 2 * m * d].reshape(d, m).T
        b = vector[2 * m * d :].reshape(m, p).T
        return a, g, b

    def module(vector):
        a, g, b = unpack(vector)
        return ((x @ a.T) * F.silu(x @ g.T)) @ b.T

    blocks = []
    for xi in inputs:
        ai, gi = up @ xi, gate @ xi
        sigmoid = 1 / (1 + np.exp(-gi))
        derivative = sigmoid + gi * sigmoid * (1 - sigmoid)
        du = np.diag(silu_numpy(gi))
        dg = np.diag(ai * derivative)
        phi = ai * silu_numpy(gi)
        blocks.append(
            np.concatenate(
                [
                    np.kron(xi.reshape(1, -1), down @ du),
                    np.kron(xi.reshape(1, -1), down @ dg),
                    np.kron(phi.reshape(1, -1), np.eye(p)),
                ],
                axis=1,
            )
        )
    explicit_module = np.concatenate(blocks, axis=0)
    autodiff_module = torch.autograd.functional.jacobian(
        lambda v: module(v).flatten(), theta
    ).numpy()
    small("P2.factored_jacobian_vs_autodiff", np.linalg.norm(explicit_module - autodiff_module))

    query, key, value = [torch.tensor(RNG.normal(size=(p, p)), dtype=dtype) for _ in range(3)]
    readout = torch.tensor(RNG.normal(size=(p, 2)), dtype=dtype)
    mask = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)

    def downstream(flat):
        residual = x + flat.reshape(2, p)
        normalized = residual / torch.sqrt(residual.square().mean(dim=-1, keepdim=True) + 0.4)
        scores = (normalized @ query) @ (normalized @ key).T / np.sqrt(p)
        attention = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
        return (residual + attention @ (normalized @ value))[-1] @ readout

    def full(vector):
        return downstream(module(vector).flatten())

    downstream_jac = torch.autograd.functional.jacobian(downstream, module(theta).flatten()).numpy()
    jac = torch.autograd.functional.jacobian(full, theta).numpy()
    small("P3.all_position_chain_rule", np.linalg.norm(jac - downstream_jac @ explicit_module))
    keep, edit = jac[:1], jac[1:]
    basis = null_basis(keep)
    response = edit @ basis
    target = np.array([0.2])
    direction = basis @ np.linalg.pinv(response) @ target
    small("P3.functional_keep_linear", np.linalg.norm(keep @ direction))
    small("P3.functional_edit_linear", np.linalg.norm(edit @ direction - target))
    small(
        "P3.minimum_norm_cost",
        abs(np.linalg.norm(direction) - np.linalg.norm(np.linalg.pinv(response) @ target)),
    )
    direction = direction / np.linalg.norm(direction)
    errors = []
    for step in [0.01, 0.005, 0.0025]:
        actual = (
            (full(theta + torch.tensor(step * direction, dtype=dtype)) - full(theta))
            .detach()
            .numpy()
        )
        error = float(np.linalg.norm(actual - jac @ (step * direction)))
        errors.append(dict(step=step, error=error))
    order = np.log(errors[0]["error"] / errors[-1]["error"]) / np.log(4)
    small("P3.second_order_remainder_order", abs(order - 2), 0.1)
    small(
        "P3.function_can_ignore_module_change",
        np.linalg.norm(np.array([[0.0, 1.0]]) @ np.array([1.0, 0.0])),
    )
    return errors


def check_learning():
    response = np.diag([2.0, 0.25, 0.0])
    target = np.ones(3)
    z = np.zeros(3)
    eta, steps = 0.1, 1000
    for _ in range(steps):
        z -= eta * response.T @ (response @ z - target)
    residual = response @ z - target
    predicted = (1 - eta * np.diag(response) ** 2) ** steps * (-target)
    small("P4.discrete_learning_modes", np.linalg.norm(residual - predicted))
    small("P4.unreachable_component_persists", abs(residual[-1] + 1))
    return dict(singular_values=[2.0, 0.25, 0.0], eta=eta, steps=steps, residual=residual.tolist())


def main():
    torch.set_num_threads(1)
    down_sweep = check_down()
    check_swiglu_counterexamples()
    taylor = check_full_network()
    learning = check_learning()
    result = {
        "time": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "plan section 14: numerical consistency of P1-P4 and counterexamples; "
            "not evidence for H1"
        ),
        "seed": 20260928,
        "device": "cpu",
        "dtype": "float64",
        "numpy": np.__version__,
        "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": CHECKS,
        "inverse_distance_sweep": down_sweep,
        "full_network_taylor_errors": taylor,
        "learning_modes": learning,
        "pass": all(item["passed"] for item in CHECKS),
    }
    output = ROOT / "docs/development-artifacts/architecture-mechanism-v1/algebra-checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"pass": result["pass"], "checks": len(CHECKS), "output": str(output)}))


if __name__ == "__main__":
    main()
