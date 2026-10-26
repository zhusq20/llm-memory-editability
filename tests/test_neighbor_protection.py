"""Contracts for address-neighbour protection sets."""

from types import SimpleNamespace

import numpy as np

from llm_memory_editability.neighbor_protection import (
    eligibility,
    neighbor_pools,
    neighbor_schedule,
)


def toy_world():
    atoms = np.array(
        [[0, 0, 5], [0, 1, 6], [1, 0, 7], [2, 2, 0], [3, 1, 4], [5, 2, 8], [4, 2, 9]],
        dtype=np.int64,
    )
    # chains (h, r1, r2, t) with first-hop atom index
    first = np.array([0, 3, 4, 2])
    chains = np.array([[0, 0, 2, 8], [2, 2, 0, 5], [3, 1, 2, 9], [1, 0, 1, 0]], dtype=np.int64)
    return SimpleNamespace(atoms=atoms, chains=chains, first=first)


def test_pools_follow_subject_and_relation_and_skip_evaluation_rows():
    world = toy_world()
    excluded = {(1, 0, 7)}
    support = np.ones(len(world.chains), dtype=bool)
    masks = eligibility(world, excluded, support)
    pools = neighbor_pools(world, (0, 1, 9), *masks)
    assert pools["atoms"]["subject"].tolist() == [0, 1]
    assert pools["atoms"]["relation"].tolist() == [4]
    # chain 1 has bridge 0 (atom 3 is (2, 2, 0)); chain 0 starts at entity 0
    assert pools["chains"]["subject"].tolist() == [0, 1]
    assert pools["chains"]["relation"].tolist() == [2, 3]
    pools = neighbor_pools(world, (1, 0, 9), *masks)
    assert 2 not in pools["atoms"]["subject"].tolist()


def test_schedule_balances_and_cycles():
    pools = {
        "atoms": {"subject": np.array([1, 2]), "relation": np.array([3, 4, 5, 6])},
        "chains": {"subject": np.array([], dtype=int), "relation": np.array([7, 8, 9])},
    }
    schedule = neighbor_schedule(pools, 5, 4, np.random.RandomState(0))
    assert len(schedule) == 5
    for step in schedule:
        assert len(step["atoms"]) == 4 and len(step["chains"]) == 4
        assert set(step["atoms"][:2]) <= {1, 2} and set(step["atoms"][2:]) <= {3, 4, 5, 6}
        assert set(step["chains"]) <= {7, 8, 9}
    assert set(schedule[0]["atoms"][:2]) == {1, 2}
