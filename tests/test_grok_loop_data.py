"""Truth, composition experience, fixed exposure and restoration contracts."""

import copy

import numpy as np
import pytest

from llm_memory_editability import grok_multihop_data
from llm_memory_editability.grok_loop_data import (
    SPLITS,
    StratifiedStream,
    audit_world,
    build_world,
    majority_tail_diagnostics,
    suffix_target_diagnostics,
)


def spec(**updates):
    return {
        "world_seed": 41,
        "hops": 3,
        "entities": 16,
        "relations": 8,
        "degree": 4,
        "phi": 2.0,
        "id_fraction": 0.75,
        "id_test_fraction": 0.2,
        "evaluation_size": 9,
        **updates,
    }


@pytest.mark.parametrize("hops", [2, 3, 4])
@pytest.mark.parametrize("id_fraction", [0.75, 0.95])
def test_adapter_preserves_graph_query_splits_and_hash(hops, id_fraction):
    config = spec(hops=hops, id_fraction=id_fraction)
    world = build_world(config)
    old = grok_multihop_data.build_world(
        config["world_seed"], **{key: value for key, value in config.items() if key != "world_seed"}
    )
    for name in SPLITS:
        np.testing.assert_array_equal(world[name], old[name])
    assert world["metadata"]["dataset_sha256"] == old["metadata"]["dataset_sha256"]
    assert audit_world(world)["relevant_atomic_indices_checked"]
    assert len(world["ood_atomic"]) == round(len(world["atomic"]) * (1 - id_fraction))


def test_default_fraction_retains_historical_95_5_and_requires_explicit_seed():
    config = spec()
    del config["id_fraction"]
    assert build_world(config)["metadata"]["id_fraction"] == 0.95
    with pytest.raises(ValueError, match="seed"):
        build_world({})


def test_all_query_fact_indices_reconstruct_true_paths_and_ood_exposure():
    world = build_world(spec())
    for name, indices in world["relevant_atomic_indices"].items():
        rows = world[name]
        atoms = world["atomic"][indices]
        np.testing.assert_array_equal(atoms[:, 0, 0], rows[:, 0])
        np.testing.assert_array_equal(atoms[:, :, 1], rows[:, 1:-1])
        np.testing.assert_array_equal(atoms[:, -1, -1], rows[:, -1])
        np.testing.assert_array_equal(atoms[:, :-1, -1], atoms[:, 1:, 0])
    diag = world["metadata"]["loop_data_diagnostics"]
    assert diag["ood_facts_seen_in_composite_training"] == 0
    ood = diag["splits"]["ood_composite"]
    assert ood["all_ood_queries"] == len(world["ood_composite"]) > 0
    assert ood["any_relevant_fact_in_composite_training_n"] == 0
    assert ood["all_relevant_facts_in_atomic_training_fraction"] == 1.0


def test_audit_rejects_wrong_truth_wrong_split_and_wrong_recorded_indices():
    world = build_world(spec())
    broken = copy.deepcopy(world)
    broken["test_full_composite"][0, -1] = 2 + (int(broken["test_full_composite"][0, -1]) - 1) % 16
    with pytest.raises(ValueError, match="target"):
        audit_world(broken)
    broken = copy.deepcopy(world)
    train = broken["train_composite"][0].copy()
    broken["train_composite"][0] = broken["ood_composite"][0]
    broken["ood_composite"][0] = train
    with pytest.raises(ValueError, match="atomic split"):
        audit_world(broken)
    broken = copy.deepcopy(world)
    broken["relevant_atomic_indices"]["test_composite"][0, 0] ^= 1
    with pytest.raises(ValueError, match="relevant atomic indices"):
        audit_world(broken)


def test_phi_keeps_reserve_and_fixed_probe_unchanged():
    low, high = build_world(spec(phi=0.5)), build_world(spec(phi=99))
    for name in ("atomic", "ood_atomic", "test_composite", "test_full_composite", "ood_composite"):
        np.testing.assert_array_equal(low[name], high[name])
    assert high["metadata"]["train_composite_capped"]


def test_suffix_counts_are_descriptive_and_expose_singleton_inflation():
    rows = np.array(
        [[2, 10, 11, 3], [3, 10, 11, 3], [4, 10, 11, 4], [5, 12, 11, 4], [6, 12, 13, 4]],
        dtype=np.int64,
    )
    last, full = suffix_target_diagnostics(rows)
    assert last["empirical_modal_target_fraction"] == 3 / 5
    assert last["singleton_group_query_fraction"] == 1 / 5
    assert last["multi_query_empirical_modal_target_fraction"] == 1 / 2
    assert full["empirical_modal_target_fraction"] == 4 / 5
    assert full["singleton_group_query_fraction"] == 2 / 5
    assert full["multi_query_empirical_modal_target_fraction"] == 2 / 3
    assert full["multi_target_groups"] == 1
    assert full["same_suffix_different_target_example"] == [[2, 10, 11, 3], [4, 10, 11, 4]]
    assert majority_tail_diagnostics(rows) == {
        "target_token": 4,
        "count": 3,
        "fraction": 3 / 5,
        "distinct_targets": 2,
    }
    empty = rows[:0]
    assert all(
        item["empirical_modal_target_fraction"] is None for item in suffix_target_diagnostics(empty)
    )
    assert majority_tail_diagnostics(empty)["fraction"] is None


def test_stratified_batches_fix_exposure_and_shuffle_uniform_epochs():
    stream = StratifiedStream(5, 7, 7, 3, seed=47)
    batches = np.array([stream.take(7) for _ in range(35)])
    assert np.all((batches < 5).sum(axis=1) == 3)
    np.testing.assert_array_equal(np.bincount(batches[batches < 5], minlength=5), [21] * 5)
    np.testing.assert_array_equal(np.bincount(batches[batches >= 5] - 5, minlength=7), [20] * 7)
    assert len(np.unique(batches < 5, axis=0)) > 1


def test_composite_pool_changes_cannot_change_atomic_draws_or_positions():
    low = StratifiedStream(11, 13, 8, 3, seed=43)
    high = StratifiedStream(11, 157, 8, 3, seed=43)
    for _ in range(31):
        left, right = low.take(), high.take()
        np.testing.assert_array_equal(left < 11, right < 11)
        np.testing.assert_array_equal(left[left < 11], right[right < 11])


def test_stream_restore_is_exact_and_saved_state_does_not_alias_live_state():
    stream = StratifiedStream(5, 7, 7, 3, seed=49)
    for _ in range(4):
        stream.take()
    state = stream.state_dict()
    reference = [stream.take() for _ in range(19)]
    restored = StratifiedStream(5, 7, 7, 3, seed=987)
    restored.load_state_dict(state)
    for expected in reference:
        np.testing.assert_array_equal(restored.take(), expected)
    assert restored.batches == stream.batches
    mismatch = StratifiedStream(5, 7, 7, 4, seed=49)
    with pytest.raises(ValueError, match="n_atomic mismatch"):
        mismatch.load_state_dict(state)


@pytest.mark.parametrize(
    "args", [(0, 7, 7, 3), (5, 0, 7, 3), (5, 7, 0, 0), (5, 7, 7, 8), (5, -1, 7, 3)]
)
def test_stream_rejects_invalid_or_empty_sampled_strata(args):
    with pytest.raises(ValueError):
        StratifiedStream(*args, seed=0)


def test_stream_handles_unused_empty_stratum_without_hanging():
    all_atomic = StratifiedStream(5, 0, 7, 7, seed=5)
    assert np.all(all_atomic.take() < 5)
    all_composite = StratifiedStream(0, 5, 7, 0, seed=5)
    assert np.all(all_composite.take() < 5)
    with pytest.raises(ValueError, match="batch_size"):
        all_atomic.take(6)
