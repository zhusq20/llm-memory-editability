"""Use development worlds only, before any independent-world confirmation."""

import numpy as np
import pytest

from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_shortcut_confirmation_sets import make_confirmation_edit_pair
from llm_memory_editability.bios_shortcut_control import high_exception_world
from llm_memory_editability.bios_shortcut_matched_edit import (
    audit_matched_edit_pair,
    make_matched_edit_pair,
)


@pytest.mark.parametrize("world_seed", [0, 1])
def test_two_supports_are_disjoint_and_preserve_development_contract(world_seed):
    low = make_cross_world(world_seed)
    high, _, _, _ = high_exception_world(low)
    for chain in (0, 1):
        original = make_matched_edit_pair(low, high, chain)
        first = make_confirmation_edit_pair(low, high, chain, 0)
        second = make_confirmation_edit_pair(low, high, chain, 1)
        for key, value in original.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(first[key], value)
        for key in ("groups", "E", "D", "selected_people"):
            assert not np.intersect1d(first[key], second[key]).size
        for pair in (first, second):
            audit_matched_edit_pair(low, high, chain, pair)
            assert len(pair["exception_conflict_D_heldout"]) == 9
            assert pair["contract"]["budget"]["edit_cases"] == 768
        repeat = make_confirmation_edit_pair(low, high, chain, 1)
        for key, value in second.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(repeat[key], value)
        # Corrupting a support to overlap cannot pass its declared truth/layout audit.
        bad = {**second, "groups": first["groups"]}
        with pytest.raises((ValueError, AssertionError)):
            audit_matched_edit_pair(low, high, chain, bad)


def test_confirmation_rejects_unregistered_support_without_generating_data():
    with pytest.raises(ValueError, match="two disjoint supports"):
        make_confirmation_edit_pair(None, None, 0, 2)
