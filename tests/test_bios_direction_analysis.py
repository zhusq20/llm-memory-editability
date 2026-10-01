"""Inference and prospective-input contracts for the independent worlds."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_direction_refresh import refresh_mask, update_failure_counts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_bios_direction_confirmation import holm_two, world_test
from predict_bios_direction import features
from report_bios_direction import solver_terminated_early


def test_world_sign_flip_uses_eight_independent_units():
    result = world_test(np.ones(8))
    assert result["effect"] == 1
    assert result["p_two_sided"] == 2 / 256
    assert result["interval95"] == [1, 1]
    assert world_test(np.zeros(8))["p_two_sided"] == 1
    with pytest.raises(AssertionError):
        world_test(np.ones(96))


def test_holm_is_monotone_in_order_and_returns_input_order():
    np.testing.assert_allclose(holm_two([0.03125, 0.015625]), [0.03125, 0.03125])
    np.testing.assert_allclose(holm_two([0.8, 0.02]), [0.8, 0.04])


def test_prediction_ignores_retention_labels_and_global_accuracy():
    factor = dict(
        new_target_nll=2.0,
        margin_ab=[0.4, -0.3],
        kernel=[[2, 1], [1, 2]],
        ratio=0.3,
        mixing=0.1,
        common=3,
        difference=1,
        feature_cos=0.9,
        downstream_cos=0.8,
        gate_cos=0.7,
        relation_input_difference=0.2,
        gate_difference=0.1,
    )
    learning = [dict(step=s, factors=[copy.deepcopy(factor)]) for s in (128, 1024)]
    altered = copy.deepcopy(learning)
    for row in altered:
        row.update(accuracy=-100, U_labels=[999], D_labels=[999], final_edit_success=True)
    for mechanism in (False, True):
        before = features(learning, 0, "coherent", "func-soft-adam", mechanism)
        after = features(altered, 0, "coherent", "func-soft-adam", mechanism)
        np.testing.assert_array_equal(before, after)
        assert np.isclose(before[4], np.log(1.8))


def test_completed_member_is_not_an_early_solver_failure():
    assert not solver_terminated_early(dict(failed=True, accepted=1024))
    assert solver_terminated_early(dict(failed=True, accepted=14))
    assert not solver_terminated_early(dict(failed=False, accepted=1024))


def test_rejection_refresh_does_not_change_other_trajectories():
    active = torch.tensor([True, True, False])
    failed = torch.tensor([1, 0, 10])
    assert refresh_mask(active, failed, 5, 16, False).tolist() == [True, False, False]
    assert refresh_mask(active, failed, 16, 16, False).tolist() == [True, True, False]


def test_idle_rejection_counter_does_not_advance():
    active = torch.tensor([True, True, False])
    accepted = torch.tensor([True, False, False])
    failed = torch.tensor([2, 3, 0])
    assert update_failure_counts(active, accepted, failed).tolist() == [0, 4, 0]
