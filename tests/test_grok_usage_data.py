"""Complete-query holdout, randomized role exchange and exposure contracts."""

import copy

import numpy as np
import pytest

from llm_memory_editability.grok_usage_data import (
    ARMS,
    UsageStream,
    audit_experiment,
    build_arm_world,
    build_experiment,
)


def experiment():
    return build_experiment(
        {"world_seed": 146001, "entities": 16, "relations": 8, "degree": 4, "test_fraction": 0.2}
    )


def fact_counts(world, indices, kind):
    selected = indices[world["table_row_kinds"][indices] == kind]
    edges = world["table_fact_indices"][selected]
    return np.bincount(edges[edges >= 0], minlength=len(world["atomic"]))


def test_one_graph_one_reserve_and_swapped_role_eligibility():
    exp = experiment()
    assert audit_experiment(exp)["passed"]
    other = experiment()
    assert exp["metadata"]["dataset_sha256"] == other["metadata"]["dataset_sha256"]
    assert exp["metadata"]["group_sizes"] == [32, 16, 16]
    assert len(exp["role_A_composite"]) == len(exp["role_B_composite"]) > 0
    assert sum(exp["metadata"]["test_group_counts"].values()) == len(exp["test_composite"])
    test_queries = {tuple(row[:-1]) for row in exp["test_composite"]}
    reference = build_arm_world(exp, "A")
    for arm in ARMS:
        world = build_arm_world(exp, arm)
        np.testing.assert_array_equal(world["atomic"], reference["atomic"])
        for name, rows in reference["evaluation"].items():
            if name.startswith("test_"):
                np.testing.assert_array_equal(world["evaluation"][name], rows)
        assert not test_queries.intersection(tuple(row[:-1]) for row in world["train_composite"])
        used = np.array(world["metadata"]["training_composite_fact_occurrence_counts"])
        if arm == "A":
            assert not used[exp["fact_groups"] == 2].any()
            assert used[exp["fact_groups"] == 1].any()
        elif arm == "B":
            assert not used[exp["fact_groups"] == 1].any()
            assert used[exp["fact_groups"] == 2].any()
        else:
            assert not used[exp["fact_groups"] != 0].any()
        assert len(world["table_weights"]) == len(world["table_fact_indices"])


def test_every_repetition_batch_exactly_matches_constituent_occurrences():
    exp = experiment()
    worlds = {arm: build_arm_world(exp, arm) for arm in ARMS}
    streams = {arm: UsageStream(world, 14601, 3, 5, 7) for arm, world in worlds.items()}
    for _ in range(37):
        batches = {arm: stream.take() for arm, stream in streams.items()}
        for arm, indices in batches.items():
            assert len(indices) == (22 if arm.startswith("rep") else 15)
            assert worlds[arm]["table_weights"][indices].sum() == 15
            np.testing.assert_array_equal(
                fact_counts(worlds[arm], indices, 0), fact_counts(worlds["A"], batches["A"], 0)
            )
            np.testing.assert_array_equal(
                fact_counts(worlds[arm], indices, 1), fact_counts(worlds["A"], batches["A"], 1)
            )
        for role in ("A", "B"):
            np.testing.assert_array_equal(
                fact_counts(worlds[role], batches[role], 2),
                fact_counts(worlds["rep" + role], batches["rep" + role], 3),
            )
        # Since the paired role pool sizes agree, A/B consume identical indices.
        np.testing.assert_array_equal(batches["A"], batches["B"])


@pytest.mark.parametrize("arm", ARMS)
def test_resume_restores_all_strata_and_interleaving_without_state_aliasing(arm):
    world = build_arm_world(experiment(), arm)
    stream = UsageStream(world, 14601, 3, 5, 7)
    for _ in range(4):
        stream.take()
    saved = stream.state_dict()
    expected = [stream.take() for _ in range(13)]
    restored = UsageStream(world, 987, 3, 5, 7)
    restored.load_state_dict(saved)
    for rows in expected:
        np.testing.assert_array_equal(restored.take(), rows)
    assert restored.batches == stream.batches
    assert saved["batches"] == 4
    with pytest.raises(ValueError, match="batch"):
        stream.take(1)


def test_auditor_rejects_truth_leakage_and_recorded_role_index_corruption():
    exp = experiment()
    broken = copy.deepcopy(exp)
    broken["test_composite"][0, -1] = 999
    with pytest.raises(ValueError, match="target"):
        audit_experiment(broken)
    broken = copy.deepcopy(exp)
    broken["role_A_composite"][0] = broken["test_composite"][0]
    with pytest.raises(ValueError, match="leaked"):
        audit_experiment(broken)
    broken = copy.deepcopy(exp)
    broken["role_B_fact_indices"][0, 0] ^= 1
    with pytest.raises(ValueError, match="fact indices"):
        audit_experiment(broken)


def test_full_graph_training_pool_changes_do_not_reselect_test_or_random_roles():
    low = build_experiment({"world_seed": 146001, "test_fraction": 0.1})
    high = build_experiment({"world_seed": 146001, "test_fraction": 0.2})
    np.testing.assert_array_equal(low["atomic"], high["atomic"])
    np.testing.assert_array_equal(low["fact_groups"], high["fact_groups"])
    assert np.all(~low["heldout_mask"] | high["heldout_mask"])
