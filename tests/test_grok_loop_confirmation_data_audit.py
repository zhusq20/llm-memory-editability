"""Descriptive subset coverage must use the full evaluation denominator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def audit():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_grok_loop_confirmation_data.py"
    spec = importlib.util.spec_from_file_location("loop_confirmation_data_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiple_targets_require_same_complete_relation_sequence(audit):
    rows = np.array(
        [
            [1, 10, 11, 4],
            [2, 10, 11, 5],
            [3, 10, 11, 5],
            [1, 12, 11, 4],  # Matching final relation alone does not join the group.
            [2, 12, 11, 4],  # Two heads with one target do not establish multiple targets.
            [3, 12, 10, 6],
        ]
    )
    summary, mask = audit.same_sequence_multiple_targets(rows)
    assert summary == {"n": 3, "total_n": 6, "coverage": 0.5, "relation_sequences": 1}
    np.testing.assert_array_equal(mask, [True, True, True, False, False, False])
    empty, empty_mask = audit.same_sequence_multiple_targets(np.empty((0, 4), int))
    assert empty["coverage"] is None and empty["total_n"] == 0
    assert empty_mask.shape == (0,)


def test_donor_coverage_preserves_ineligible_rows_and_intersection(audit):
    donors = {}
    for family in ("different", "same"):
        donors.update(
            {
                family + "_valid": np.array([True, False, True, False]),
                family + "_candidate_count": np.array([2, 0, 1, 0]),
                family + "_reason": np.array(["eligible", "missing", "eligible", "missing"]),
                family + "_counterfactual_split": np.array(["ood", "none", "ood", "none"]),
            }
        )
    donors["different_rejected_same_target"] = np.array([1, 2, 0, 0])
    result = audit.donor_diagnostics(donors, np.array([False, True, True, False]))
    for family in ("different", "same"):
        assert result[family]["total_n"] == 4
        assert result[family]["eligible_n"] == 2
        assert result[family]["coverage"] == 0.5
        assert result[family]["eligible_and_multiple_target_same_sequence_n"] == 1
        assert result[family]["candidate_count_mean"] == 0.75
        assert result[family]["reasons"] == {"eligible": 2, "missing": 2}
    assert result["different_candidate_rejection_counts"]["same_target"] == 3
