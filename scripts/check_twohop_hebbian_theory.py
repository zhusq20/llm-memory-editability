#!/usr/bin/env python3
"""Numerical checks of plan §14.29; algebra checks, not evidence about trained LLMs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(142901)
CHECKS = []


def close(name, actual, expected, tolerance=1e-10):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    CHECKS.append(dict(name=name, error=error, tolerance=tolerance, passed=error <= tolerance))
    if not CHECKS[-1]["passed"]:
        raise AssertionError(CHECKS[-1])


def holds(name, condition, **values):
    CHECKS.append(dict(name=name, passed=bool(condition), **values))
    if not condition:
        raise AssertionError(CHECKS[-1])


def check_whitening():
    features = RNG.normal(size=(7, 19))
    readout = RNG.normal(size=(4, 7))
    query = RNG.normal(size=7)
    covariance = features @ features.T / features.shape[1]
    outputs = readout @ features
    kernel = features.T @ np.linalg.solve(covariance, query)
    close("full_rank_hebbian_identity", outputs @ kernel / 19, readout @ query)
    features = RNG.normal(size=(7, 3))
    covariance = features @ features.T / 3
    projector = covariance @ np.linalg.pinv(covariance)
    outputs = readout @ features
    reconstructed = outputs @ features.T @ np.linalg.pinv(covariance) @ query / 3
    close("singular_covariance_is_projection", reconstructed, readout @ projector @ query)
    close(
        "singular_residual_exact",
        readout @ query - reconstructed,
        readout @ (np.eye(7) - projector) @ query,
    )
    holds(
        "singular_extension_not_generally_exact",
        np.linalg.norm(readout @ query - reconstructed) > 1e-4,
    )
    regularization = 0.1
    inverse = np.linalg.inv(covariance + regularization * np.eye(7))
    ridge_reconstructed = outputs @ features.T @ inverse @ query / 3
    close(
        "ridge_residual_exact",
        readout @ query - ridge_reconstructed,
        regularization * readout @ inverse @ query,
    )


def check_counterexamples():
    # Both clean atomic calls correct; a misaligned interface swaps the entity codes.
    entity_codes = np.eye(2)
    interface = np.array([[0.0, 1.0], [1.0, 0.0]])
    clean = entity_codes[0]
    holds(
        "perfect_atomic_calls_do_not_imply_composition",
        np.argmax(clean) == 0 and np.argmax(interface @ clean) == 1,
    )
    # First-hop decoder ignores y; its positive margin gives no bound on y.
    first_decoder = np.array([[1.0, 0.0], [-1.0, 0.0]])
    second_difference = np.array([1.0, -1.0])
    emitted = np.array([1.0, 2.0])
    close(
        "first_hop_clean_and_noisy_margin_equal",
        np.diff(first_decoder @ emitted),
        np.diff(first_decoder @ clean),
    )
    holds(
        "correct_first_hop_wrong_second_hop",
        (first_decoder @ emitted).argmax() == 0 and second_difference @ emitted < 0,
    )
    harmful = np.array([0.0, 2.0])
    helpful = -harmful
    close("same_norm_direction_control", np.linalg.norm(harmful), np.linalg.norm(helpful))
    holds(
        "direction_changes_success_at_equal_norm",
        second_difference @ (clean + harmful) < 0 < second_difference @ (clean + helpful),
    )
    # A negative sufficient-bound value is inconclusive, not a necessary failure.
    safe = np.array([2.0, 2.0])
    margin = float(second_difference @ clean)
    lower = margin - np.linalg.norm(second_difference) * np.linalg.norm(safe)
    holds("certificate_failure_is_not_task_failure", lower < 0 < second_difference @ (clean + safe))
    # Reducing Euclidean error can move it into a more damaging direction.
    smaller_bad = np.array([-1.1, 0.0])
    larger_safe = np.array([0.0, -2.0])
    holds(
        "smaller_error_need_not_improve_answer",
        np.linalg.norm(smaller_bad) < np.linalg.norm(larger_safe)
        and second_difference @ (clean + smaller_bad) < 0
        and second_difference @ (clean + larger_safe) > 0,
    )


def check_bilinear():
    d, hidden, values = 6, 17, 5
    up, gate = RNG.normal(size=(2, hidden, d))
    down = RNG.normal(size=(d, hidden))
    codes = RNG.normal(size=(values, d))
    q, delta = RNG.normal(size=(2, d))

    def memory(x):
        return down @ ((up @ x) * (gate @ x))

    target = int(np.argmax(codes @ memory(q)))
    clean_scores = codes @ memory(q)
    rows = []
    for competitor in range(values):
        if competitor == target:
            continue
        difference = codes[target] - codes[competitor]
        weight = down.T @ difference
        matrix = up.T @ (weight[:, None] * gate)
        symmetric = (matrix + matrix.T) / 2
        gamma = float(clean_scores[target] - clean_scores[competitor])
        close(f"quadratic_form_{competitor}", q @ symmetric @ q, gamma)
        exact = gamma + 2 * q @ symmetric @ delta + delta @ symmetric @ delta
        close(f"exact_directional_expansion_{competitor}", difference @ memory(q + delta), exact)
        linear = 2 * np.linalg.norm(symmetric @ q)
        quadratic = np.linalg.norm(symmetric, 2)
        radius = 2 * gamma / (linear + np.sqrt(linear**2 + 4 * quadratic * gamma))
        close(f"radius_root_{competitor}", gamma - linear * radius - quadratic * radius**2, 0)
        rows.append((competitor, radius, symmetric, gamma))
        # A non-diagonal ellipsoid tests a genuinely anisotropic bound.
        root = RNG.normal(size=(d, d)) / 10
        ellipsoid_bound = 2 * np.linalg.norm(root.T @ symmetric @ q)
        ellipsoid_bound += np.linalg.norm(root.T @ symmetric @ root, 2)
        worst_violation = -np.inf
        for _ in range(100):
            vector = RNG.normal(size=d)
            vector /= np.linalg.norm(vector)
            perturbation = root @ vector * RNG.uniform(0, 1)
            loss = gamma - float(difference @ memory(q + perturbation))
            worst_violation = max(worst_violation, loss - ellipsoid_bound)
        holds(
            f"ellipsoidal_bound_{competitor}",
            worst_violation < 1e-10,
            maximum_violation=float(worst_violation),
        )
    certified_radius = min(row[1] for row in rows)
    correct = 0
    for _ in range(1000):
        direction = RNG.normal(size=d)
        direction /= np.linalg.norm(direction)
        perturbed = q + direction * certified_radius * RNG.uniform(0, 0.999)
        correct += int(np.argmax(codes @ memory(perturbed)) == target)
    holds("all_1000_certified_perturbations_correct", correct == 1000, correct=correct)
    # Positive rescaling of every score cannot change the certified input radius.
    _, radius, symmetric, gamma = rows[0]
    scale = 37.0
    linear = 2 * np.linalg.norm(scale * symmetric @ q)
    quadratic = np.linalg.norm(scale * symmetric, 2)
    scaled_radius = (
        2 * scale * gamma / (linear + np.sqrt(linear**2 + 4 * quadratic * scale * gamma))
    )
    close("radius_invariant_to_logit_scaling", radius, scaled_radius)
    # Coordinate changes leave exact predictions invariant, but not naive L2 distances.
    transform = np.diag(np.arange(1, d + 1))
    inv = np.linalg.inv(transform)
    close(
        "quadratic_coordinate_invariance",
        (transform @ q) @ (inv.T @ symmetric @ inv) @ (transform @ q),
        q @ symmetric @ q,
    )
    return dict(certified_radius=float(certified_radius), tested_perturbations=1000)


def check_composition_and_edit():
    d = 8
    bridge = RNG.normal(size=d)
    noise = RNG.normal(size=d) / 20
    transport = RNG.normal(size=(d, d)) / 3
    key = RNG.normal(size=d)
    routing = RNG.normal(size=d) / 20
    mismatch = transport @ bridge - key
    observed = transport @ (bridge + noise) + routing
    delta = transport @ noise + mismatch + routing
    close("composition_error_decomposition", observed - key, delta)
    ceiling = np.linalg.norm(transport, 2) * np.linalg.norm(noise)
    ceiling += np.linalg.norm(mismatch) + np.linalg.norm(routing)
    holds("composition_error_upper_bound", np.linalg.norm(delta) <= ceiling + 1e-12)
    # Updating a value has an exact effect only with fixed kernels, queries and decoders.
    keys = RNG.normal(size=(11, d))
    values = RNG.normal(size=(11, d))
    query = RNG.normal(size=d)
    kernel = (keys @ query) ** 2
    selected = 3
    change = RNG.normal(size=d)
    updated = values.copy()
    updated[selected] += change
    close(
        "fixed_kernel_value_edit_exact",
        updated.T @ kernel - values.T @ kernel,
        change * kernel[selected],
    )
    # First-hop value edit changes second-hop query; holding it fixed would be incorrect.
    up, gate = RNG.normal(size=(2, 12, d))
    down = RNG.normal(size=(d, 12))

    def second(x):
        return down @ ((up @ x) * (gate @ x))

    change_query = transport @ change * kernel[selected]
    actual = second(observed + change_query) - second(observed)
    exact = down @ (
        (up @ observed) * (gate @ change_query)
        + (up @ change_query) * (gate @ observed)
        + (up @ change_query) * (gate @ change_query)
    )
    close("first_hop_edit_second_hop_quadratic_effect", actual, exact, tolerance=1e-8)
    holds("frozen_second_query_misses_propagation", np.linalg.norm(actual) > 1)


def main():
    check_whitening()
    check_counterexamples()
    experiment = check_bilinear()
    check_composition_and_edit()
    result = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Float64 consistency checks of conditional theorems and explicit counterexamples; "
        "not a proof by experiment or a learned-model benchmark.",
        seed=142901,
        checks=CHECKS,
        summary=dict(total=len(CHECKS), passed=len(CHECKS)),
        perturbation_experiment=experiment,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__,
    )
    output = ROOT / "docs/development-artifacts/twohop-hebbian-v1/math-checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
