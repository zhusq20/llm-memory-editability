"""Statistical weighting, continuous scores, and independently replayed exposures."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.grok_usage_data import UsageStream, build_arm_world, build_experiment


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[1] / "scripts/report_grok_usage.py"
    spec = importlib.util.spec_from_file_location("usage_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scores_keep_answer_eos_and_arithmetic_probability_separate(report):
    rows = np.array([[2, 10, 11, 4], [3, 10, 12, 5]])
    predictions = {
        "test_answer": np.array([4, 5]),
        "test_stop": np.array([1, 0]),
        "test_target": np.array([4, 5]),
        "test_nll": -np.log(np.array([[0.9, 0.7], [0.1, 0.8]])),
    }
    scored = report.rescore(predictions, "test", rows)
    assert scored["accuracy"] == 0.5
    assert scored["answer_accuracy"] == 1
    assert scored["answer_probability"] == pytest.approx(0.5)
    predictions["test_target"][0] = 999
    with pytest.raises(ValueError, match="targets"):
        report.rescore(predictions, "test", rows)


def test_cohort_effects_are_equal_weight_not_query_pooled_and_worlds_stay_separate(report):
    rows = []
    for world in (1, 2):
        for initialization in (1, 2) if world == 1 else (1,):
            for arm, aa, bb in (
                ("A", 0.9, 0.1),
                ("B", 0.1, 0.3),
                ("repA", 0.2, 0.1),
                ("repB", 0.1, 0.2),
            ):
                for cohort, value, n in (("A", aa, 100), ("B", bb, 10)):
                    rows.append(
                        {
                            "phase": "development",
                            "world_seed": world,
                            "initialization": initialization,
                            "step": 128000,
                            "arm": arm,
                            "split": f"test_{cohort}_{cohort}",
                            "n": n,
                            **{metric: value for metric in report.METRICS},
                        }
                    )
    effects = report.paired_effects(rows)
    primary = [
        row
        for row in effects
        if row["metric"] == "accuracy" and row["comparison"] == "main_role_equal_cohorts"
    ]
    assert len(primary) == 3
    assert all(row["effect"] == pytest.approx(0.5) for row in primary)
    world_effects = report.world_average_effects(effects)
    primary_worlds = [
        row
        for row in world_effects
        if row["metric"] == "accuracy" and row["comparison"] == "main_role_equal_cohorts"
    ]
    assert [row["world_seed"] for row in primary_worlds] == [1, 2]
    assert [row["initializations"] for row in primary_worlds] == [2, 1]
    assert all(row["effect"] == pytest.approx(0.5) for row in primary_worlds)


@pytest.mark.parametrize("arm", ["A", "B", "repA", "repB"])
def test_independent_rng_replay_matches_real_stream_exposure_at_epoch_boundaries(report, arm):
    exp = build_experiment({"world_seed": 146001, "entities": 16, "relations": 8, "degree": 4})
    world = build_arm_world(exp, arm)
    spec = {"arm": arm, "stream_seed": 441, "n_atomic": 3, "n_background": 5, "n_role": 7}
    steps = [0, 1, 2, 9, 37, 128]
    expected = report.expected_exposures(world, spec, steps)
    stream = UsageStream(world, 441, 3, 5, 7)
    actual = np.zeros((4, len(world["atomic"])), dtype=np.int64)
    kinds = np.zeros(4, dtype=np.int64)
    for step in range(129):
        if step:
            indices = stream.take()
            row_kinds = world["table_row_kinds"][indices]
            kinds += np.bincount(row_kinds, minlength=4)
            for kind in range(4):
                facts = world["table_fact_indices"][indices[row_kinds == kind]].ravel()
                actual[kind] += np.bincount(facts[facts >= 0], minlength=actual.shape[1])
        if step in expected:
            np.testing.assert_array_equal(actual, expected[step][0])
            np.testing.assert_array_equal(kinds, expected[step][1])


def test_role_alignment_averages_cohorts_then_worlds_and_retains_missing_nodes(report):
    rows = []
    scores = {
        "A": (0.95, 0.7, 0.9, 0.2),
        "B": (0.9, 0.85, 0.1, 0.8),
        "repA": (0.99, 0.6, 0.3, 0.07),
        "repB": (0.92, 0.9, 0.05, 0.4),
    }
    for phase, world, scale in (
        ("development", 11, 1),
        ("confirmation", 21, 1),
        ("confirmation", 22, 0.8),
        ("confirmation", 23, 0.6),
    ):
        for step in (0, 128000):
            for arm, values in scores.items():
                for split, value in zip(
                    ("atomic_A", "atomic_B", "test_A_A", "test_B_B"), values, strict=True
                ):
                    rows.append(
                        {
                            "phase": phase,
                            "world_seed": world,
                            "initialization": 1,
                            "step": step,
                            "arm": arm,
                            "split": split,
                            "n": 100 if split.endswith("A") else 1,
                            **{metric: value * scale for metric in report.METRICS},
                        }
                    )
    aligned = report.align_learning_roles(rows)
    configured = {"development": {11}, "confirmation": {21, 22, 23}}
    means = report.average_aligned_worlds(aligned, configured)
    selected = {
        (row["phase"], row["task"], row["role"]): row["value"]
        for row in means
        if row["step"] == 128000 and row["metric"] == "accuracy"
    }
    assert selected["development", "composition", "composition_used"] == pytest.approx(0.85)
    assert selected["confirmation", "composition", "composition_used"] == pytest.approx(0.68)
    assert selected["confirmation", "composition", "atomic_only"] == pytest.approx(0.12)
    assert selected["confirmation", "composition", "matching_repetition"] == pytest.approx(0.28)
    assert selected["confirmation", "composition", "other_repetition"] == pytest.approx(0.048)
    assert selected["confirmation", "atomic", "composition_used"] == pytest.approx(0.72)
    incomplete = [row for row in aligned if not (row["world_seed"] == 23 and row["step"] == 128000)]
    partial_means = report.average_aligned_worlds(incomplete, configured)
    assert all(
        row["value"] is None
        for row in partial_means
        if row["phase"] == "confirmation" and row["step"] == 128000
    )
    assert all(row["value"] is not None for row in partial_means if row["step"] == 0)
