"""The report must score raw entity/EOS tokens, including empty pools."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "depth_step_report", Path(__file__).parents[1] / "scripts/report_depth_step.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_eos_errors_are_not_reported_as_success():
    rows = np.array([[3, 67, 68, 9], [4, 67, 68, 10], [5, 67, 68, 11]])
    prediction = np.array([[9, 1], [10, 2], [12, 1]])
    accuracy, correct = MODULE.generated_score(rows, prediction)
    assert accuracy == pytest.approx(1 / 3)
    np.testing.assert_array_equal(correct, [True, False, False])


def test_empty_pool_is_missing_score():
    accuracy, correct = MODULE.generated_score(np.empty((0, 4)), np.empty((0, 2)))
    assert accuracy is None
    assert correct.shape == (0,)


@pytest.mark.parametrize("shape", [(2, 1), (2, 3), (1, 2)])
def test_incomplete_generation_fails(shape):
    with pytest.raises(ValueError, match="greedy answer and EOS"):
        MODULE.generated_score(np.array([[3, 67, 68, 9], [4, 67, 68, 10]]), np.zeros(shape))
