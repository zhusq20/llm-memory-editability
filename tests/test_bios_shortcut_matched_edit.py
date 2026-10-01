"""E39 is a new common-support update, never a relabeling of historical E93."""

import copy

import numpy as np
import pytest

from llm_memory_editability.bios_shortcut_control import high_exception_world
from llm_memory_editability.bios_shortcut_matched_edit import (
    BUDGET,
    audit_matched_edit_pair,
    make_matched_edit_pair,
)


@pytest.fixture(scope="module")
def worlds():
    from test_bios_cross import cross_world

    low = cross_world.__wrapped__()
    high, _, _, _ = high_exception_world(low)
    return low, high


@pytest.mark.parametrize("chain", (0, 1))
def test_same_E39_D96_with_only_exception_primary_conflict(worlds, chain):
    low, high = worlds
    pair = make_matched_edit_pair(low, high, chain)
    assert audit_matched_edit_pair(low, high, chain, pair)["passed"]
    assert len(pair["E"]) == 39 and len(pair["D"]) == 96
    assert len(pair["E_roots"]) == 3 and len(pair["E_actual"]) == 36
    assert len(pair["exception_conflict_D"]) == 18
    assert len(pair["exception_conflict_D_heldout"]) == 9
    reference = pair["paired_reference_D"]
    people = np.searchsorted(low.derived_ids[chain], reference)
    actual = low.actual_ids[chain, people]
    for phase in ("low", "high"):
        coherent, exception = pair[f"{phase}_coherent"], pair[f"{phase}_exception"]
        assert np.all(coherent[reference] == coherent[actual])
        assert np.all(exception[reference] != exception[actual])
    for world, phase in ((low, "low"), (high, "high")):
        for kind in ("coherent", "exception"):
            np.testing.assert_array_equal(
                np.flatnonzero(pair[f"{phase}_{kind}"] != world.answers),
                np.union1d(pair["E"], pair["D"]),
            )


@pytest.mark.parametrize("chain", (0, 1))
def test_histograms_match_across_groups_and_splits_not_within_group(worlds, chain):
    low, high = worlds
    pair = make_matched_edit_pair(low, high, chain)
    selected = pair["selected_people"]
    for pool in (low.derived_ids[chain], low.train_ids[chain], low.heldout_ids[chain]):
        people = selected.ravel()[np.isin(low.derived_ids[chain, selected.ravel()], pool)]
        ids = low.actual_ids[chain, people]
        for phase in ("low", "high"):
            np.testing.assert_array_equal(
                np.sort(pair[f"{phase}_coherent"][ids]), np.sort(pair[f"{phase}_exception"][ids])
            )
    for index, group in enumerate(pair["groups"]):
        assert not high.exceptions[chain, selected[index]].any()
        for split in (slice(0, 6), slice(6, 12)):
            ids = low.actual_ids[chain, selected[index, split]]
            assert np.count_nonzero(pair["low_exception"][ids] == pair["new_default"][index]) == 3
            assert np.count_nonzero(pair["low_exception"][ids] == pair["alternative"][index]) == 3
            assert np.count_nonzero(pair["low_coherent"][ids] == pair["new_default"][index]) == 6
        assert np.all(low.memberships[chain, selected[index]] == group)


@pytest.mark.parametrize("chain", (0, 1))
def test_common_replay_truths_and_phase_specific_retention(worlds, chain):
    low, high = worlds
    pair = make_matched_edit_pair(low, high, chain)
    replay = pair["R"]
    assert len(np.unique(replay)) == 4096
    np.testing.assert_array_equal(low.answers[replay], high.answers[replay])
    for phase in ("low", "high"):
        for kind in ("coherent", "exception"):
            np.testing.assert_array_equal(pair[f"{phase}_{kind}"][replay], low.answers[replay])
    assert not np.intersect1d(replay, np.union1d(pair["E"], pair["D"])).size
    assert not np.intersect1d(replay, pair["U_heldout"]).size
    assert np.all(replay < low.n_base)
    # U is not silently restricted to common-truth facts: different original
    # actual answers remain scored against each model's own old knowledge.
    assert np.count_nonzero(low.answers[pair["U_full"]] != high.answers[pair["U_full"]]) == 1792
    assert len(pair["D_original_exception"]) == 6
    assert len(pair["D_newly_exception"]) == 42
    assert len(pair["D_remaining_ordinary_unedited"]) == 12


def test_deterministic_nonmutating_generation_and_explicit_new_budget(worlds):
    low, high = worlds
    originals = low.answers.copy(), high.answers.copy()
    first, second = make_matched_edit_pair(low, high, 0), make_matched_edit_pair(low, high, 0)
    for key, value in first.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, second[key])
        else:
            assert value == second[key]
    np.testing.assert_array_equal(low.answers, originals[0])
    np.testing.assert_array_equal(high.answers, originals[1])
    assert BUDGET["models"] == 2 * 2 * 2 * 3 == 24
    assert BUDGET["edit_cases"] == 24 * 2 * 2 == 96
    assert BUDGET["scopes"] == ["mlp"]
    assert not first["contract"]["historical_E93_comparable"]
    with pytest.raises(ValueError, match="support zero"):
        make_matched_edit_pair(low, high, 0, support=1)


def test_audit_rejects_different_replay_truths_or_propagation_leakage(worlds):
    low, high = worlds
    pair = make_matched_edit_pair(low, high, 0)
    corrupt = copy.deepcopy(pair)
    changed_old_truth = np.flatnonzero(low.answers != high.answers)[0]
    corrupt["R"][0] = changed_old_truth
    with pytest.raises((AssertionError, ValueError)):
        audit_matched_edit_pair(low, high, 0, corrupt)
    corrupt = copy.deepcopy(pair)
    corrupt["R"][0] = pair["D"][0]
    with pytest.raises(ValueError):
        audit_matched_edit_pair(low, high, 0, corrupt)
    corrupt = copy.deepcopy(pair)
    person = corrupt["selected_people"][0, 0]
    corrupt["high_exception"][low.actual_ids[0, person]] = corrupt["new_default"][0]
    with pytest.raises(AssertionError):
        audit_matched_edit_pair(low, high, 0, corrupt)
