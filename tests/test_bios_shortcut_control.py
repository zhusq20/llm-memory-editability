"""Manipulation tests for exact marginal and exposure controls."""

import numpy as np
import pytest

from llm_memory_editability.bios_shortcut_control import (
    audit_high_exception,
    high_exception_world,
)


@pytest.fixture(scope="module")
def pair():
    from test_bios_cross import cross_world

    old = cross_world.__wrapped__()
    new, report, selected, donors = high_exception_world(old)
    return old, new, report, selected, donors


def test_manipulation_exactly_preserves_histograms_and_split(pair):
    old, new, report, selected, donors = pair
    assert report["passed"] and report["changed_total"] == 2 * 64 * 14
    assert selected.shape == (2, 64, 14) and donors.shape == (2, 14, 64)
    assert int(old.exceptions.sum()) == 256
    assert int(new.exceptions.sum()) == 2048
    assert audit_high_exception(old, new)["passed"]


def test_generation_is_deterministic_and_does_not_mutate_source(pair):
    old, new, report, selected, donors = pair
    again, repeated, s2, d2 = high_exception_world(old)
    np.testing.assert_array_equal(again.answers, new.answers)
    np.testing.assert_array_equal(selected, s2)
    np.testing.assert_array_equal(donors, d2)
    assert report == repeated
    assert not np.shares_memory(old.answers, new.answers)
    assert not np.shares_memory(old.exceptions, new.exceptions)


def test_audit_rejects_nonlocal_truth_and_histogram_changes(pair):
    from dataclasses import replace

    old, new, _, _, _ = pair
    corrupt = new.answers.copy()
    corrupt[new.root_ids[0, 0]] = new.answers[new.root_ids[0, 1]] + 1
    with pytest.raises(AssertionError):
        audit_high_exception(old, replace(new, answers=corrupt))
    corrupt = new.answers.copy()
    corrupt[new.actual_ids[0, 0]] = -1
    with pytest.raises(AssertionError):
        audit_high_exception(old, replace(new, answers=corrupt))
