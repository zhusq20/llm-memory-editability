"""Truth, actual exposure, paired denominators and donor-prefix isolation."""

import copy

import numpy as np
import pytest

from llm_memory_editability import grok_loop_same_bridge as same_bridge
from llm_memory_editability.grok_loop_data import StratifiedStream, build_world


def make_spec(seed=87, **updates):
    return {
        "world_seed": seed,
        "hops": 2,
        "entities": 16,
        "relations": 4,
        "degree": 3,
        "id_fraction": 0.65,
        "phi": 0.3,
        "id_test_fraction": 0.2,
        "evaluation_size": 24,
        "steps": 500,
        "batch_size": 11,
        "n_atomic_per_batch": 3,
        "stream_seed": 8133,
        **updates,
    }


@pytest.mark.parametrize("steps", [0, 1, 17, 500])
def test_exposure_replay_matches_real_training_sampler_and_position_counts(steps):
    spec = make_spec()
    world = build_world(spec)
    exposure = same_bridge.actual_exposure_counts(world, {"spec": spec}, steps=steps)
    stream = StratifiedStream(
        len(world["atomic"]),
        len(world["train_composite"]),
        spec["batch_size"],
        spec["n_atomic_per_batch"],
        spec["stream_seed"],
    )
    count = np.zeros(len(world["atomic"]) + len(world["train_composite"]), dtype=np.int64)
    for _ in range(steps):
        np.add.at(count, stream.take(), 1)
    np.testing.assert_array_equal(exposure["atomic_record_counts"], count[: len(world["atomic"])])
    np.testing.assert_array_equal(
        exposure["composite_record_counts"], count[len(world["atomic"]) :]
    )
    graph = {(int(h), int(r)): (int(t), i) for i, (h, r, t) in enumerate(world["atomic"])}
    expected = np.zeros((len(world["atomic"]), 2), dtype=np.int64)
    table = np.zeros_like(expected)
    for row, repetitions in zip(
        world["train_composite"], exposure["composite_record_counts"], strict=True
    ):
        head, r1, r2, _ = row
        bridge, first = graph[head, r1]
        _, second = graph[bridge, r2]
        table[first, 0] += 1
        table[second, 1] += 1
        expected[first, 0] += repetitions
        expected[second, 1] += repetitions
    np.testing.assert_array_equal(exposure["actual_position_counts"], expected)
    np.testing.assert_array_equal(exposure["table_position_counts"], table)


@pytest.mark.parametrize("steps", [0, 1, 17, 500])
def test_all_qualified_donors_are_retained_and_pairs_have_recipient_denominators(steps):
    spec = make_spec()
    world = build_world(spec)
    selected = same_bridge.select_same_bridge_donors(world, spec, steps=steps)
    graph = {(int(h), int(r)): (int(t), i) for i, (h, r, t) in enumerate(world["atomic"])}
    ids = {tuple(map(int, row)) for row in world["id_atomic"]}
    exposures = selected["atomic_composite_actual_position_exposure"]
    np.testing.assert_array_equal(selected["original_rows"], world["ood_composite"])
    for i, (head, r1, r2, target) in enumerate(world["ood_composite"]):
        bridge, first = graph[head, r1]
        tail, second = graph[bridge, r2]
        assert tail == target
        np.testing.assert_array_equal(selected["original_atomic_indices"][i], [first, second])
        expected = {"id": [], "ood": []}
        second_only = unexposed = 0
        for donor in sorted(map(tuple, world["atomic"])):
            dh, dr, db = donor
            if dr != r1 or db != bridge or dh in (head, target):
                continue
            fact = graph[dh, dr][1]
            if donor in ids:
                if exposures[fact, 0]:
                    expected["id"].append(donor)
                elif exposures[fact, 1]:
                    second_only += 1
                else:
                    unexposed += 1
            else:
                expected["ood"].append(donor)
        for family in ("id", "ood"):
            offsets = selected[family + "_candidate_offsets"]
            actual = selected[family + "_donor_rows"][offsets[i] : offsets[i + 1]]
            assert list(map(tuple, actual)) == expected[family]
            assert selected[family + "_candidate_count"][i] == len(expected[family])
        assert selected["id_second_only_candidate_count"][i] == second_only
        assert selected["id_unexposed_candidate_count"][i] == unexposed
        common = bool(expected["id"] and expected["ood"])
        assert bool(selected["common_valid"][i]) == common
        pair_mask = selected["pair_recipient_indices"] == i
        assert pair_mask.sum() == len(expected["id"]) * len(expected["ood"])
        assert selected["pair_weights"][pair_mask].sum() == pytest.approx(float(common))
        if common:
            assert selected["reason"][i] == "eligible"
            observed_pairs = set(
                zip(
                    map(tuple, selected["pair_id_donor_rows"][pair_mask]),
                    map(tuple, selected["pair_ood_donor_rows"][pair_mask]),
                    strict=True,
                )
            )
            assert observed_pairs == {(a, b) for a in expected["id"] for b in expected["ood"]}
        else:
            assert selected["reason"][i] != "eligible"


def test_same_bridge_counterfactuals_keep_truth_successor_and_are_untrained():
    spec = make_spec(seed=71, phi=1.0)
    world = build_world(spec)
    selected = same_bridge.select_same_bridge_donors(world, spec)
    trained = {tuple(row[:-1]) for row in world["train_composite"]}
    assert len(selected["pair_rows"]), "Fixture must exercise nonempty paired selection"
    recipients = selected["pair_recipient_indices"]
    for family in ("id", "ood"):
        donors = selected["pair_" + family + "_donor_rows"]
        counterfactual = selected["pair_" + family + "_counterfactual_rows"]
        np.testing.assert_array_equal(donors[:, 1], selected["pair_rows"][:, 1])
        np.testing.assert_array_equal(donors[:, 2], selected["original_bridge"][recipients])
        np.testing.assert_array_equal(counterfactual[:, 1:], selected["pair_rows"][:, 1:])
        assert (donors[:, 0] != selected["pair_rows"][:, 0]).all()
        assert (donors[:, 0] != selected["pair_rows"][:, -1]).all()
        assert not any(tuple(row[:-1]) in trained for row in counterfactual)
        cf_edges = selected["pair_" + family + "_counterfactual_atomic_indices"]
        np.testing.assert_array_equal(
            cf_edges[:, 1], selected["original_atomic_indices"][recipients, 1]
        )
        assert not selected["atomic_is_id"][cf_edges[:, 1]].any()
        np.testing.assert_array_equal(selected["atomic_is_id"][cf_edges[:, 0]], family == "id")
        expected_split = "mixed_id_ood" if family == "id" else "pure_ood"
        assert set(selected["pair_" + family + "_counterfactual_split"]) == {expected_split}
    np.testing.assert_array_equal(
        selected["pair_self_donor_rows"],
        world["atomic"][selected["original_atomic_indices"][recipients, 0]],
    )
    audit = selected["audit"]
    common = selected["common_valid"]
    assert audit["unique_common_first_facts_n"] == len(
        set(map(tuple, world["ood_composite"][common, :2]))
    )
    assert audit["unique_common_second_facts_n"] == len(
        {
            (int(selected["original_bridge"][i]), int(row[2]))
            for i, row in enumerate(world["ood_composite"])
            if common[i]
        }
    )
    for family in ("id", "ood"):
        assert audit["unique_common_" + family + "_donor_facts_n"] == len(
            set(map(tuple, selected["pair_" + family + "_donor_rows"]))
        )
        assert audit["common_" + family + "_donor_prefix_contains_answer_n"] == 0
    assert audit["common_prefix_contains_answer_n"] == int(
        (world["ood_composite"][common, 0] == world["ood_composite"][common, -1]).sum()
    )


def test_selection_does_not_read_predictions_or_mutate_world_and_ignores_global_rng():
    spec = make_spec(seed=71, phi=1.0)
    world = build_world(spec)
    before = copy.deepcopy(world)
    first = same_bridge.select_same_bridge_donors(world, spec)
    world["predictions"] = {"arbitrary_model_score": float("nan")}
    np.random.seed(937)
    np.random.random(100)
    second = same_bridge.select_same_bridge_donors(world, {"spec": spec})
    for key in first:
        if isinstance(first[key], np.ndarray):
            np.testing.assert_array_equal(first[key], second[key])
        else:
            assert first[key] == second[key]
    for key in before:
        if isinstance(before[key], np.ndarray):
            np.testing.assert_array_equal(world[key], before[key])
        elif key == "relevant_atomic_indices":
            for name in before[key]:
                np.testing.assert_array_equal(world[key][name], before[key][name])
        else:
            assert world[key] == before[key]


def test_cache_shared_between_architectures_but_not_changed_sample_stream_or_step():
    spec = make_spec()
    world = build_world(spec)
    exposure = same_bridge.actual_exposure_counts(world, spec)
    with_architecture = {**spec, "architecture": "l2", "initialization": 123}
    cached = same_bridge.select_same_bridge_donors(world, with_architecture, exposure=exposure)
    direct = same_bridge.select_same_bridge_donors(world, spec)
    np.testing.assert_array_equal(cached["pair_rows"], direct["pair_rows"])
    for changed in ({**spec, "stream_seed": 918}, {**spec, "batch_size": 12}):
        with pytest.raises(ValueError, match="another world or sample stream"):
            same_bridge.select_same_bridge_donors(world, changed, exposure=exposure)
    with pytest.raises(ValueError, match="another world or sample stream"):
        same_bridge.select_same_bridge_donors(world, spec, steps=1, exposure=exposure)


def test_reject_bad_truth_world_identity_and_unregistered_budget():
    spec = make_spec()
    world = build_world(spec)
    with pytest.raises(ValueError, match="registered training endpoint"):
        same_bridge.select_same_bridge_donors(world, spec, steps=spec["steps"] + 1)
    with pytest.raises(ValueError, match="world_seed"):
        same_bridge.select_same_bridge_donors(world, {**spec, "world_seed": 999})
    world["metadata"]["dataset_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        same_bridge.select_same_bridge_donors(world, spec)
    world = build_world(spec)
    world["ood_composite"][0, -1] = world["metadata"]["entity_offset"]
    with pytest.raises(ValueError, match="target"):
        same_bridge.select_same_bridge_donors(world, spec)


@pytest.mark.parametrize(
    "seed,expected",
    [
        (145001, (540, 172, 68, 30, 34)),
        (145011, (515, 161, 25, 16, 28)),
        (145012, (521, 182, 60, 37, 45)),
        (145013, (515, 175, 64, 42, 50)),
    ],
)
def test_frozen_world_coverage_uses_all_donors_without_model_conditioning(seed, expected):
    spec = make_spec(
        seed=seed,
        entities=128,
        relations=16,
        degree=8,
        id_fraction=0.75,
        phi=6,
        id_test_fraction=0.1,
        evaluation_size=1024,
        steps=128000,
        batch_size=256,
        n_atomic_per_batch=32,
        stream_seed=seed * 100 + 2 if seed == 145001 else seed * 1000 + 21,
    )
    selected = same_bridge.select_same_bridge_donors(build_world(spec), spec)
    audit = selected["audit"]
    assert (
        tuple(
            audit[key]
            for key in (
                "all_recipient_n",
                "id_available_n",
                "ood_available_n",
                "common_n",
                "pair_n",
            )
        )
        == expected
    )
