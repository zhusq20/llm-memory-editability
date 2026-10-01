"""Independent tests for P4 query-level recomputation and scientific denominators."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_mechanism_causal.py"
SPEC = importlib.util.spec_from_file_location("p4_causal_summary", MODULE)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def predictions(values, valid=None):
    result = {"prediction": np.array(values), "ended": np.ones(len(values), dtype=bool)}
    if valid is not None:
        result["source_valid"] = np.repeat(np.asarray(valid, dtype=bool)[:, None], 2, axis=1)
    return result


def test_query_pairing_uses_common_availability_not_unmatched_means():
    baseline = predictions([1, 1, 1])
    left, right = predictions([1, 0, 0], [True, False, True]), predictions([0, 1, 0])
    plan = {
        "answer": np.ones(3),
        "family": np.zeros(3),
        "recipient_old_exception": np.array([0, 1, 0]),
    }
    row = summary.contrast_rows(left, right, baseline, plan, "left", "right")[0]
    assert row["common_available"] == 2
    assert row["left_unavailable"] == 1
    assert row["accuracy_difference"] == 0.5
    assert row["original_known_right_wrong"] == 2
    assert row["conditional_recovery_auxiliary"] == 0.5


def test_selectivity_pairs_people_and_keeps_root_cohort_distinct():
    # Default and actual use the same two people; one actual control is unavailable.
    plan = {
        "answer": np.ones(6),
        "family": np.array([0, 0, 1, 1, 9, 9]),
        "person": np.array([10, 11, 10, 11, -1, -1]),
        "recipient_old_exception": np.array([0, 1, 0, 1, -1, -1]),
    }
    target = predictions([0, 1, 1, 0, 1, 1])
    control = predictions([1, 1, 1, 1, 1, 1], [True, True, True, False, True, True])
    rows = summary.selectivity_rows(target, control, plan, "random")
    actual = next(
        row for row in rows if row["control_family"] == "actual" and row["population"] == "all"
    )
    root = next(row for row in rows if row["control_family"] == "group_root")
    assert actual["default_queries"] == actual["control_queries"] == 1
    assert actual["paired_people"] and actual["default_minus_other_contrast"] == -1
    assert root["default_queries"] == 2 and root["control_queries"] == 2
    assert not root["paired_people"] and root["default_minus_other_contrast"] == -0.5


def test_projection_capture_uses_original_distinct_answer_rule_and_four_way_mask():
    plan = {
        "pair_type": np.zeros(3),
        "recipient_default": np.array([1, 1, 1]),
        "recipient_actual": np.array([2, 3, 2]),
        "donor_default": np.array([3, 3, 3]),
        "recipient_old_exception": np.array([0, 1, 0]),
    }
    arms = {
        "full_donor": predictions([3, 3, 3]),
        "full_norm_random": predictions([1, 1, 1]),
        "projected_target": predictions([3, 1, 1]),
        "projected_random": predictions([1, 1, 1], [True, True, False]),
    }
    for arm in arms:
        arms[arm]["source_delta_norm"] = np.ones((3, 2))
    rows = summary.capture_rows(arms, plan)
    all_row = rows[0]
    assert all_row["eligible_matched"] == 2  # Row 1 is an ambiguous recipient-actual hit.
    assert all_row["common_available"] == 1  # Row 2 fails one of the four controls.
    assert all_row["full_minus_full_random"] == 1
    assert all_row["projected_minus_projected_random"] == 1
    assert all_row["projected_contrast_minus_full_contrast"] == 0


def test_donor_prediction_strata_are_descriptive_and_preserve_primary_population():
    plan = {
        "pair_type": np.zeros(2),
        "recipient_default": np.array([1, 1]),
        "recipient_actual": np.array([2, 2]),
        "donor_default": np.array([3, 3]),
        "donor_actual": np.array([2, 2]),
        "recipient_old_exception": np.array([0, 1]),
        "donor": np.array([10, 11]),
    }
    baseline, donor = predictions([1, 1]), predictions([3, 2])
    arms = {"full_donor": predictions([3, 2]), "full_norm_random": predictions([1, 1])}
    rows = summary.transfer_contrasts(arms, baseline, donor, plan, plan)
    primary = next(
        row
        for row in rows
        if row["scope"] == "locked_distinct"
        and row["population"] == "all"
        and row["donor_class"] == "all"
    )
    assert primary["common_available"] == 2
    assert primary["donor_default_hit_difference"] == 0.5
    default_only = next(
        row
        for row in rows
        if row["scope"] == "locked_distinct"
        and row["population"] == "all"
        and row["donor_class"] == "default_only"
    )
    assert default_only["common_available"] == 1


def test_norm_check_excludes_unavailable_controls_but_rejects_real_mismatch():
    target, control = predictions([1, 1]), predictions([1, 1], [True, False])
    target["source_delta_norm"] = np.ones((2, 2))
    control["source_delta_norm"] = np.array([[1.0, 1.0], [0.0, 0.0]])
    summary.check_norms({"target_remove": target, "random_remove": control})
    control["source_delta_norm"][0] = 2
    with pytest.raises(ValueError, match="Norm-matched"):
        summary.check_norms({"target_remove": target, "random_remove": control})


def test_raw_reader_rejects_query_reordering_and_invalid_eos_type(tmp_path):
    path = tmp_path / "raw.npz"
    values = {
        "query_id": np.array([8, 9]),
        "prediction": np.array([1, 2]),
        "ended": np.array([True, False]),
        "forward_calls": np.array(2),
        "forward_examples": np.array(4),
        "padded_token_positions": np.array(22),
        "logical_token_positions": np.array(20),
    }
    np.savez(path, **values)
    summary.read_predictions(path, np.array([8, 9]), 5, 3, 256, np.array([4, 5]))
    with pytest.raises(ValueError, match="ordering"):
        summary.read_predictions(path, np.array([9, 8]), 5, 3, 256, np.array([4, 5]))
    np.savez(path, **{**values, "ended": np.array([1, 0])})
    with pytest.raises(ValueError, match="Malformed"):
        summary.read_predictions(path, np.array([8, 9]), 5, 3, 256, np.array([4, 5]))


def test_saved_csv_is_checked_against_independent_raw_recomputation(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("arm,family,population,correct,accuracy\na,default,all,2,1.0\n")
    expected = [
        {"arm": "a", "family": "default", "population": "all", "correct": 2, "accuracy": 1.0}
    ]
    summary.check_csv(path, expected, ("arm", "family", "population"))
    expected[0]["correct"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        summary.check_csv(path, expected, ("arm", "family", "population"))


def test_v2_archive_difference_is_preserved_and_checked_without_selecting_rows(tmp_path):
    ids = np.array([8, 9])
    arrays = predictions([1, 2])
    archive = predictions([1] * 10)
    path = tmp_path / "archive.npz"
    np.savez(
        path,
        query_id=ids,
        local_prediction=arrays["prediction"],
        local_ended=arrays["ended"],
        archive_prediction=archive["prediction"][ids],
        archive_ended=archive["ended"][ids],
        prediction_changed=np.array([False, True]),
        ended_changed=np.array([False, False]),
    )
    counts, changed = summary.audit_archive_difference(path, arrays, archive, ids)
    assert counts["queries"] == 2 and counts["case_exclusions"] == 0
    assert counts["prediction_differences"] == 1
    assert changed[0]["query_id"] == 9
    assert changed[0]["local_prediction"] == 2 and changed[0]["archive_prediction"] == 1
    with pytest.raises(ValueError, match="Plan changed"):
        summary.audit_archive_difference(path, predictions([1, 1]), archive, ids)
