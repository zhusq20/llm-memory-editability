"""Failures that would change the scope of the new architecture identities."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "architecture_bridge", ROOT / "scripts/check_architecture_bridge_algebra.py"
)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def test_projected_roundoff_cannot_make_a_keep_feature_editable():
    rng = np.random.default_rng(19)
    keep = rng.normal(size=(7, 3))
    with pytest.raises(ValueError, match="incompatible"):
        bridge.fixed_feature_edit(keep, keep[:, :1].copy(), np.ones((2, 1)))


def test_conflicting_targets_on_identical_features_are_rejected():
    with pytest.raises(ValueError, match="incompatible"):
        bridge.fixed_feature_edit(np.zeros((3, 0)), np.ones((3, 2)), np.array([[1.0, -1.0]]))


def test_empty_keep_and_zero_feature_cases():
    delta, _ = bridge.fixed_feature_edit(np.zeros((2, 0)), np.eye(2), np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(delta, [[1, 2]])
    delta, _ = bridge.fixed_feature_edit(np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((1, 1)))
    np.testing.assert_array_equal(delta, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="incompatible"):
        bridge.fixed_feature_edit(np.eye(2), np.ones((2, 1)), np.ones((1, 1)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_features_are_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        bridge.fixed_feature_edit(np.array([[bad]]), np.ones((1, 1)), np.ones((1, 1)))


def test_minimum_norm_matches_independent_constraint_solver():
    rng = np.random.default_rng(51)
    keep = rng.normal(size=(9, 3))
    edit = rng.normal(size=(9, 2))
    target = rng.normal(size=(4, 2))
    actual, _ = bridge.fixed_feature_edit(keep, edit, target)
    constraint = np.concatenate([keep, edit], axis=1)
    rhs = np.concatenate([np.zeros((4, 3)), target], axis=1)
    expected = np.linalg.lstsq(constraint.T, rhs.T, rcond=1e-12)[0].T
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_shared_expert_is_not_protected_by_disjoint_routing():
    packed = bridge.pack_moe_features(np.ones((2, 1, 2)), np.eye(2), np.ones((1, 2)), np.ones(2))
    np.testing.assert_array_equal(np.array([[1.0, 0.0, 0.0]]) @ packed, [[1, 0]])
    np.testing.assert_array_equal(np.array([[0.0, 0.0, 1.0]]) @ packed, [[1, 1]])


@pytest.mark.parametrize("k", [0, 3, 1.5])
def test_invalid_topk_is_rejected(k):
    with pytest.raises(ValueError, match="valid k"):
        bridge.topk_router(np.ones((2, 1)), k)


def test_hard_route_crossing_changes_finite_update_support():
    before = bridge.topk_router(np.array([[1e-9], [-1e-9]]), 1)
    after = bridge.topk_router(np.array([[-1e-9], [1e-9]]), 1)
    edited_expert = np.array([[0.0, 1.0]])
    assert (edited_expert @ before).item() == 0
    assert (edited_expert @ after).item() == 1


def test_gdn_product_order_is_chronological_and_not_commutative():
    transitions = [np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 1.0]])]
    initial = np.array([[1.0, 0.0]])
    writes = [np.zeros((1, 2)), np.zeros((1, 2))]
    expected = bridge.gdn_rollout(initial, transitions, writes)[-1]
    np.testing.assert_array_equal(expected, [[2, 1]])
    np.testing.assert_array_equal(bridge.gdn_expansion(initial, transitions, writes), expected)
    assert not np.allclose(initial @ bridge.right_product(transitions[::-1]), expected)


def test_gdn_future_writes_cancel_only_when_transitions_remain_fixed():
    key, value = np.array([1.0, 0.0]), np.array([1.0])
    transition, write = bridge.gdn_transition(key, value, 0.8, 0.7)
    initial = np.array([[0.2, 0.4]])
    injection = np.array([[0.5, -0.3]])
    before = bridge.gdn_rollout(initial, [transition] * 2, [write] * 2)[-1]
    after = bridge.gdn_rollout(initial, [transition] * 2, [write] * 2, {0: injection})[-1]
    np.testing.assert_allclose(after - before, injection @ transition, atol=1e-15)
    other, other_write = bridge.gdn_transition(key[::-1], value, 0.8, 0.7)
    changed = bridge.gdn_rollout(
        initial, [transition, other], [write, other_write], {0: injection}
    )[-1]
    assert not np.allclose(changed - before, injection @ transition)


def test_injection_outside_sequence_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="inside"):
        bridge.gdn_rollout(np.zeros((1, 2)), [np.eye(2)], [np.zeros((1, 2))], {1: np.ones((1, 2))})


def test_query_only_interface_misses_a_pure_bypass_change():
    theta = np.array([0.3, -0.4, 0.2])
    direction = np.array([0.0, 0.0, 1.0])
    state_jac = bridge.interface_jacobian(theta)
    head_jac = bridge.nonlinear_head_jacobian(bridge.interface_state(theta))
    assert (state_jac[:2] @ direction == 0).all()
    assert abs(head_jac @ state_jac @ direction) > 0.01


def test_point_derivative_cannot_certify_large_nonlinear_displacement():
    before, after = 3.5, 0.7
    actual = abs(np.tanh(after) - np.tanh(before))
    invalid_bound = (1 - np.tanh(before) ** 2) * abs(after - before)
    valid_bound = abs(after - before)
    assert invalid_bound < actual < valid_bound


def test_all_constructed_checks_pass_and_do_not_claim_model_evidence():
    report = bridge.run_checks()
    failures = [item for item in report["checks"] if not item["passed"]]
    assert not failures, failures
    assert report["all_passed"]
    assert report["max_identity_error"] < 1e-10
    assert report["conclusion"]["mechanism_evidence"] is False
