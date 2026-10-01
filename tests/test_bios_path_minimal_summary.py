"""Check that a sampled negative onset is not mislabeled as a sign transition."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_path_minimal.py"
SPEC = importlib.util.spec_from_file_location("minimal_summary", PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def test_initial_negative_is_not_a_later_transition():
    timeline = [{"step": 8 * i, "kernel_derived": value} for i, value in enumerate([-1, 2, -3, -4])]
    assert summary.first_competition(timeline) == 16
    assert summary.first_competition(timeline[:2]) is None
    assert (
        summary.first_competition(
            [{"step": 0, "kernel_derived": -1}, {"step": 8, "kernel_derived": -2}]
        )
        == 0
    )


def test_minimal_saved_factors_use_norm_matched_source(tmp_path):
    z = np.ones((2, 3, 2, 2))
    delta = np.ones((2, 3, 2, 2))
    # Raw intervened source is twice the true source; norm matching cancels it.
    path = tmp_path / "factors.npz"
    np.savez(path, z=z, delta=delta, source_z=z[:, :1], source_delta=2 * delta[:, :1])
    assert summary.factor_check(path, [{"kernel_derived": 16}, {"kernel_derived": 16}]) == 0
    with pytest.raises(ValueError, match="transfer kernel"):
        summary.factor_check(path, [{"kernel_derived": 32}, {"kernel_derived": 32}])
