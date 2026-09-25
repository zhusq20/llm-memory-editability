"""Fully observed matrix representability; no training or editing algorithm.

The best approximation minimizes unweighted Frobenius error over *all* entries.
It need not preserve unchanged facts exactly or prioritize requested edits.
Numerical rank uses an absolute singular-value threshold, not an exact-rank proof.
"""

from numbers import Integral, Real
from typing import Any

import numpy as np


def _matrix(value: Any, name: str = "matrix") -> np.ndarray:
    """Convert a nonempty, finite, real numeric matrix to float64."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric matrix") from exc
    if raw.ndim != 2 or 0 in raw.shape:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float64 values")
    return result


def _rank(value: Any, shape: tuple[int, int]) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("rank must be a nonnegative integer")
    if not 0 <= value <= min(shape):
        raise ValueError(f"rank must be between 0 and {min(shape)}")
    return int(value)


def _tolerance(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("atol must be a finite positive real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("atol must be a finite positive real number")
    return value


def numerical_rank(matrix: Any, atol: float = 1e-10) -> int:
    """Count singular values strictly greater than the absolute tolerance."""
    array = _matrix(matrix)
    tolerance = _tolerance(atol)
    return int(np.count_nonzero(np.linalg.svd(array, compute_uv=False) > tolerance))


def best_rank_approximation(matrix: Any, rank: int) -> np.ndarray:
    """Return a best rank-at-most-``rank`` Frobenius approximation using SVD.

    ``rank`` may be zero and may not exceed the smaller matrix dimension.
    The returned matrix is independent of the input. Tied singular values can
    make the optimum nonunique; NumPy's SVD selects one such optimum.
    """
    array = _matrix(matrix)
    budget = _rank(rank, array.shape)
    left, values, right = np.linalg.svd(array, full_matrices=False)
    return (left[:, :budget] * values[:budget]) @ right[:budget, :]


def representation_gap(matrix: Any, rank: int, atol: float = 1e-10) -> int:
    """Return max(0, numerical target rank minus the available rank budget)."""
    array = _matrix(matrix)
    budget = _rank(rank, array.shape)
    return max(0, numerical_rank(array, atol=atol) - budget)


def evaluate_update(original: Any, target: Any, rank: int, atol: float = 1e-10) -> dict[str, Any]:
    """Report an update's analytical optimum under a fixed total rank budget.

    ``original`` and ``target`` must have identical shapes. The target is fitted
    directly, with no optimization path or restriction on the update rank.
    ``squared_error_lower_bound`` is the sum of discarded squared singular
    values (Eckart–Young–Mirsky); floating-point residuals may differ slightly.

    Edited entries are exactly those where original and target differ. ``atol``
    affects rank estimates only. RMSE is measured against the target separately
    on edited and retained entries; an empty group is represented by ``None``.
    The retained-entry metric measures collateral error at this global optimum,
    not the side effects of a learned editing algorithm.
    """
    before = _matrix(original, "original")
    after = _matrix(target, "target")
    if before.shape != after.shape:
        raise ValueError("original and target must have identical shapes")
    budget = _rank(rank, after.shape)
    tolerance = _tolerance(atol)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = _matrix(after - before, "target - original")
    left, values, right = np.linalg.svd(after, full_matrices=False)
    reconstruction = (left[:, :budget] * values[:budget]) @ right[:budget, :]
    residual = reconstruction - after
    edited = before != after
    target_rank = int(np.count_nonzero(values > tolerance))
    gap = max(0, target_rank - budget)
    error = float(np.linalg.norm(residual, ord="fro"))
    lower_bound = float(np.sum(values[budget:] ** 2))

    def group_rmse(mask: np.ndarray) -> float | None:
        return float(np.linalg.norm(residual[mask]) / np.sqrt(mask.sum())) if mask.any() else None

    report = {
        "rank_budget": budget,
        "atol": tolerance,
        "original_rank": numerical_rank(before, tolerance),
        "target_rank": target_rank,
        "delta_rank": numerical_rank(delta, tolerance),
        "representation_gap": gap,
        "representable_within_rank_tolerance": gap == 0,
        "edit_count": int(edited.sum()),
        "original": before.tolist(),
        "target": after.tolist(),
        "delta": delta.tolist(),
        "reconstruction": reconstruction.tolist(),
        "singular_values": values.tolist(),
        "frobenius_error": error,
        "squared_frobenius_error": error**2,
        "squared_error_lower_bound": lower_bound,
        "edited_entry_rmse": group_rmse(edited),
        "retained_entry_rmse": group_rmse(~edited),
    }
    metrics = [
        error,
        error**2,
        lower_bound,
        report["edited_entry_rmse"],
        report["retained_entry_rmse"],
    ]
    if not all(value is None or np.isfinite(value) for value in metrics):
        raise ValueError("matrix scale exceeds finite float64 error metrics; rescale the inputs")
    return report
