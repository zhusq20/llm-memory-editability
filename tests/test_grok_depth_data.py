import copy

import numpy as np
import pytest

from llm_memory_editability.grok_depth_data import (
    ATOMIC_SPLITS,
    COMPOSITE_SPLITS,
    audit_world,
    build_world,
)


def test_world_is_deterministic_and_does_not_consume_global_randomness():
    np.random.seed(983)
    before = np.random.get_state()
    first = build_world(41, entities=32, relations=8, degree=4, phi=2, id_test_fraction=0.1)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.random.uniform(size=57)
    second = build_world(41, entities=32, relations=8, degree=4, phi=2, id_test_fraction=0.1)
    for name in ATOMIC_SPLITS + COMPOSITE_SPLITS:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["metadata"] == second["metadata"]
    other = build_world(42, entities=32, relations=8, degree=4, phi=2, id_test_fraction=0.1)
    assert first["metadata"]["dataset_sha256"] != other["metadata"]["dataset_sha256"]


def test_train_and_tests_use_correct_atomic_knowledge_without_query_leakage():
    world = build_world(7, entities=48, relations=12, degree=8, phi=3, id_test_fraction=0.1)
    facts = {(h, r): t for h, r, t in world["atomic"]}
    id_keys = {(h, r) for h, r, _ in world["id_atomic"]}
    ood_keys = {(h, r) for h, r, _ in world["ood_atomic"]}
    assert id_keys | ood_keys == set(facts)
    assert not id_keys & ood_keys
    query_sets = []
    for name in COMPOSITE_SPLITS:
        keys = ood_keys if name == "ood_composite" else id_keys
        for head, r1, r2, tail in world[name]:
            bridge = facts[head, r1]
            assert facts[bridge, r2] == tail
            assert (head, r1) in keys
            assert (bridge, r2) in keys
        query_sets.append({tuple(row[:-1]) for row in world[name]})
    assert len(set().union(*query_sets)) == sum(map(len, query_sets))
    assert len(world["train_composite"]) == round(3 * len(world["id_atomic"]))
    assert len(world["test_composite"]) > 0
    assert len(world["ood_composite"]) > 0
    assert world["metadata"]["counts"]["atomic"] == 48 * 8
    assert audit_world(world)["ok"]


def test_all_facts_can_be_used_in_either_chain_position_and_unused_is_not_test():
    world = build_world(
        19, entities=16, relations=4, degree=4, phi=1, id_fraction=1, id_test_fraction=0
    )
    assert len(world["test_composite"]) == len(world["ood_composite"]) == 0
    all_chains = np.concatenate([world["train_composite"], world["unused_composite"]])
    facts = {(h, r): t for h, r, t in world["atomic"]}
    first = {(h, r1) for h, r1, _, _ in all_chains}
    second = {(facts[h, r1], r2) for h, r1, r2, _ in all_chains}
    assert first == set(facts)
    assert first & second  # No role-specific disjoint entity/fact groups.
    assert len(all_chains) == 16 * 4 * 4
    assert len(world["unused_composite"]) == 16 * 4 * 3
    assert len(world["metadata"]["warnings"]) == 2


def test_saturated_phi_and_empty_splits_are_reported_without_resampling():
    world = build_world(0, entities=1, relations=1, degree=1, phi=99, id_fraction=1)
    assert world["metadata"]["seed"] == 0
    assert world["metadata"]["train_composite_capped"]
    assert world["metadata"]["phi_actual"] == 1
    np.testing.assert_array_equal(world["atomic"], [[2, 3, 2]])
    np.testing.assert_array_equal(world["train_composite"], [[2, 3, 3, 2]])
    assert world["ood_composite"].shape == (0, 4)
    assert world["ood_atomic"].shape == (0, 3)
    assert world["metadata"]["vocab_size"] == 4


def test_zero_phi_and_all_ood_still_include_every_atomic_fact_for_training():
    world = build_world(3, entities=8, relations=4, degree=2, phi=0, id_fraction=0)
    assert len(world["atomic"]) == len(world["ood_atomic"]) == 16
    assert world["train_composite"].shape == (0, 4)
    assert world["test_composite"].shape == (0, 4)
    assert len(world["ood_composite"]) == 32
    assert audit_world(world)["ok"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entities": 0},
        {"entities": 1.5},
        {"degree": 0},
        {"relations": 2, "degree": 3},
        {"phi": -1},
        {"phi": float("nan")},
        {"id_fraction": 1.1},
        {"id_test_fraction": -0.1},
    ],
)
def test_invalid_world_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_world(0, **kwargs)


def test_audit_rejects_corrupted_truth_and_evaluation_leakage():
    world = build_world(4, entities=32, relations=8, degree=4, phi=2, id_test_fraction=0.1)
    corrupted = copy.deepcopy(world)
    original = corrupted["test_composite"][0, -1]
    corrupted["test_composite"][0, -1] = 2 + (original - 2 + 1) % 32
    with pytest.raises(ValueError, match="false chain"):
        audit_world(corrupted)
    corrupted = copy.deepcopy(world)
    corrupted["test_composite"] = np.vstack(
        [corrupted["test_composite"], corrupted["train_composite"][:1]]
    )
    with pytest.raises(ValueError, match="overlap"):
        audit_world(corrupted)
