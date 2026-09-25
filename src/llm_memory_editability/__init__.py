"""Analytical examples for research on shared structure and editability."""

from .experiments import run_minimal_example
from .low_rank import (
    best_rank_approximation,
    evaluate_update,
    numerical_rank,
    representation_gap,
)

__version__ = "0.1.0"

__all__ = [
    "best_rank_approximation",
    "evaluate_update",
    "numerical_rank",
    "representation_gap",
    "run_minimal_example",
]
