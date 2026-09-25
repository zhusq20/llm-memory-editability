"""Checks for the minimal low-rank example and its mathematical claims."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from llm_memory_editability.low_rank import (
    best_rank_approximation,
    evaluate_update,
    numerical_rank,
    representation_gap,
)

ORIGINAL = np.ones((2, 2))
RULE = 2 * ORIGINAL
EXCEPTION = np.array([[2.0, 1.0], [1.0, 1.0]])
EXCEPTION_ERROR = (3 - math.sqrt(5)) / 2


def test_equal_delta_ranks_do_not_imply_equal_target_ranks():
    assert numerical_rank(RULE - ORIGINAL) == 1
    assert numerical_rank(EXCEPTION - ORIGINAL) == 1
    assert numerical_rank(RULE) == 1
    assert numerical_rank(EXCEPTION) == 2
    assert representation_gap(RULE, 1) == 0
    assert representation_gap(EXCEPTION, 1) == 1


def test_rule_update_is_exact_at_rank_one():
    report = evaluate_update(ORIGINAL, RULE, rank=1)

    assert report["representable_within_rank_tolerance"] is True
    assert report["frobenius_error"] == pytest.approx(0.0, abs=1e-14)
    assert report["edited_entry_rmse"] == pytest.approx(0.0, abs=1e-14)
    assert report["edit_count"] == 4
    assert report["retained_entry_rmse"] is None
    np.testing.assert_allclose(report["reconstruction"], RULE, atol=1e-14)
    json.dumps(report, allow_nan=False)


def test_exception_has_unavoidable_error_at_rank_one():
    report = evaluate_update(ORIGINAL, EXCEPTION, rank=1)

    assert report["target_rank"] == 2
    assert report["delta_rank"] == 1
    assert report["representation_gap"] == 1
    assert report["representable_within_rank_tolerance"] is False
    assert report["edit_count"] == 1
    assert report["frobenius_error"] == pytest.approx(EXCEPTION_ERROR)
    assert report["squared_frobenius_error"] == pytest.approx(EXCEPTION_ERROR**2)
    assert report["squared_error_lower_bound"] == pytest.approx(EXCEPTION_ERROR**2)
    assert report["edited_entry_rmse"] > 0
    assert report["retained_entry_rmse"] > 0
    json.dumps(report, allow_nan=False)


def test_extra_capacity_eliminates_exception_error():
    report = evaluate_update(ORIGINAL, EXCEPTION, rank=2)

    assert report["representation_gap"] == 0
    assert report["representable_within_rank_tolerance"] is True
    assert report["frobenius_error"] == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(report["reconstruction"], EXCEPTION, atol=1e-14)


def test_projection_error_matches_known_singular_value_tail():
    target = np.diag([5.0, 3.0, 1.0])
    approximation = best_rank_approximation(target, rank=1)
    report = evaluate_update(np.zeros_like(target), target, rank=1)

    np.testing.assert_allclose(approximation, np.diag([5.0, 0.0, 0.0]))
    assert report["frobenius_error"] == pytest.approx(math.sqrt(10))
    assert report["squared_error_lower_bound"] == pytest.approx(10.0)


def test_zero_rank_is_valid_for_rectangular_matrices():
    target = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 0.0]])

    np.testing.assert_array_equal(best_rank_approximation(target, rank=0), np.zeros((2, 3)))
    assert representation_gap(target, rank=0) == 2
    report = evaluate_update(np.zeros_like(target), target, rank=0)
    assert report["frobenius_error"] == pytest.approx(math.sqrt(30))


def test_numerical_rank_respects_explicit_tolerance():
    matrix = np.diag([1.0, 1e-12])

    assert numerical_rank(matrix, atol=1e-10) == 1
    assert numerical_rank(matrix, atol=1e-14) == 2
    assert numerical_rank(np.zeros((2, 3))) == 0


@pytest.mark.parametrize(
    "matrix",
    [
        [1.0, 2.0],
        [[1.0], [2.0, 3.0]],
        np.empty((0, 2)),
        [[float("nan")]],
        [[float("inf")]],
        [[1.0 + 2.0j]],
        [["not a number"]],
    ],
)
def test_invalid_matrices_are_rejected(matrix):
    with pytest.raises(ValueError):
        numerical_rank(matrix)


@pytest.mark.parametrize("rank", [-1, 3, 0.5, True])
def test_invalid_rank_budgets_are_rejected(rank):
    with pytest.raises(ValueError):
        best_rank_approximation(ORIGINAL, rank=rank)


@pytest.mark.parametrize("atol", [0, -1e-10, float("nan"), float("inf"), True])
def test_invalid_tolerances_are_rejected(atol):
    with pytest.raises(ValueError):
        numerical_rank(ORIGINAL, atol=atol)


def test_updates_require_matching_shapes():
    with pytest.raises(ValueError):
        evaluate_update(ORIGINAL, np.ones((2, 3)), rank=1)


def test_module_cli_writes_same_json_as_stdout(tmp_path):
    output_path = tmp_path / "minimal-example.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_memory_editability",
            "--rank",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    assert report["parameters"]["rank_budget"] == 1
    assert report["cases"]["rule"]["frobenius_error"] == pytest.approx(0.0, abs=1e-14)
    assert report["cases"]["exception"]["frobenius_error"] == pytest.approx(EXCEPTION_ERROR)
    assert report["cases"]["exception"]["representation_gap"] == 1
