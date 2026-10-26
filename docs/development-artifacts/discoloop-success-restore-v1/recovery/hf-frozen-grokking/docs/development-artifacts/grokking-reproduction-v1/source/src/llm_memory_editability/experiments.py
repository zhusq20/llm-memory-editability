"""Deterministic analytical illustration from the supplied research memo."""

import platform
from typing import Any

import numpy as np

from .low_rank import evaluate_update


def run_minimal_example(rank: int = 1, atol: float = 1e-10) -> dict[str, Any]:
    """Compare a shared-rule update with a one-entry exception.

    At rank budget one, both changes have rank-one deltas but only the rule's
    target is representable. At budget two, both targets are representable.
    This is a classical matrix fact, not evidence of LLM editing difficulty.
    """
    original = np.ones((2, 2), dtype=float)
    cases = {
        "original": evaluate_update(original, original, rank, atol),
        "rule": evaluate_update(original, 2 * original, rank, atol),
        "exception": evaluate_update(original, [[2, 1], [1, 1]], rank, atol),
    }
    return {
        "experiment": "minimal_low_rank",
        "scope": (
            "Analytical SVD illustration for fully observed real matrices with a "
            "fixed total rank budget and unweighted Frobenius error. No training, "
            "optimization path, or Transformer editing is evaluated. Numerical "
            "rank counts singular values greater than an absolute tolerance; "
            "this example does not validate the broader research hypothesis."
        ),
        "parameters": {
            "rank_budget": cases["original"]["rank_budget"],
            "atol": cases["original"]["atol"],
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "cases": cases,
    }
