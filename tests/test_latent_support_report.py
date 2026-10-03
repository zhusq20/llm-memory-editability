"""Conclusion-bearing contracts for support pairing, exposures and graph strata."""

import copy
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_latent_support.py"
MODULE_SPEC = importlib.util.spec_from_file_location("latent_support_report", SCRIPT)
REPORT = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(REPORT)


def scores(n=1000, accuracy=0.1):
    result = {
        task: {"n": n, "accuracy": accuracy, "answer_accuracy": accuracy + 0.01, "answer_nll": 2.0}
        for task in REPORT.TASKS
    }
    for task in ("familiar_test", "strict_test"):
        result[task].update(
            atomic_correct_coverage=1.0,
            conditional_accuracy=accuracy,
            autonomous_two_calls=1.0,
        )
    return result


@pytest.fixture
def rows():
    result = []
    for world_index, world in enumerate(REPORT.WORLDS):
        for seed_index, initialization in enumerate(REPORT.INITIALIZATIONS):
            for support in REPORT.SUPPORTS:
                score = 0.1 + 0.1 * world_index + 0.01 * seed_index
                if support == "connected":
                    score += 0.02 + 0.02 * seed_index + 0.02 * world_index
                result.append(
                    {
                        "name": f"w{world}-i{initialization}-{support}",
                        "world": world,
                        "initialization": initialization,
                        "support": support,
                        **REPORT.metric_counts(scores(n=1000 * (world_index + 1), accuracy=score)),
                    }
                )
    return result


def select(contrasts, level="all_worlds", metric="accuracy", task="familiar_test"):
    return [
        r for r in contrasts if r["level"] == level and r["metric"] == metric and r["task"] == task
    ]


def test_primary_sign_and_all_six_pairs_are_public(rows):
    contrasts = REPORT.paired_contrasts(rows)
    primary = select(contrasts)[0]
    assert primary["priority"] == "primary"
    assert primary["effect_pp"] == pytest.approx(5.0)
    assert primary["n_worlds"] == 3
    assert primary["n_initializations"] == 6
    worlds = select(contrasts, level="world")
    assert [r["effect_pp"] for r in worlds] == pytest.approx([3.0, 5.0, 7.0])
    pairs = select(contrasts, level="seed")
    assert [r["effect_pp"] for r in pairs] == pytest.approx([2, 4, 4, 6, 6, 8])
    assert [term["coefficient"] for term in pairs[0]["terms"]] == [1, -1]
    assert [term["n"] for term in pairs[0]["terms"]] == [1000, 1000]
    assert select(contrasts, metric="answer_accuracy")[0]["priority"] == "secondary"
    assert select(contrasts, task="common_atomic")[0]["priority"] == "secondary"


def test_world_weighting_never_pools_queries_or_initializations(rows):
    selected = [r for r in rows if r["support"] == "split"]
    selected = [
        r
        for r in selected
        if not (r["world"] == REPORT.WORLDS[0] and r["initialization"] == REPORT.INITIALIZATIONS[1])
    ]
    condition = REPORT.condition_means(selected)[0]
    assert condition["means"]["familiar_test.accuracy"] == pytest.approx((0.1 + 0.205 + 0.305) / 3)
    assert condition["means"]["familiar_test.accuracy"] != pytest.approx(
        np.mean([r["familiar_test.accuracy"] for r in selected])
    )
    assert [w["denominators"]["familiar_test"] for w in condition["worlds"]] == [
        [1000],
        [2000, 2000],
        [3000, 3000],
    ]


def test_missing_or_duplicate_pairs_fail(rows):
    with pytest.raises(ValueError, match="Missing paired endpoint"):
        REPORT.paired_contrasts(rows[1:])
    with pytest.raises(ValueError, match="Duplicate endpoint"):
        REPORT.paired_contrasts([*rows, rows[0]])


def test_actual_role_exposures_cannot_be_replaced_by_unweighted_degrees():
    a = np.array([[1, 11, 100, 17, 1000], [2, 11, 100, 18, 1001]])
    b = np.array([[1, 11, 100, 18, 1001], [2, 11, 100, 17, 1000]])
    assert REPORT.role_exposures(a, [16000, 16000]) == REPORT.role_exposures(b, [16000, 16000])
    assert REPORT.role_exposures(a, [16000, 15999]) != REPORT.role_exposures(b, [16000, 15999])
    first, second = REPORT.role_exposures(a, [16000, 16000])
    assert sum(first.values()) == sum(second.values()) == 32000
    with pytest.raises(ValueError, match="vector length"):
        REPORT.role_exposures(a, [1])


def test_bfs_partition_retains_different_and_uncovered_test_queries():
    connected = np.array([[1, 11, 100, 17, 1000], [2, 11, 100, 17, 1000], [2, 11, 100, 18, 1001]])
    split = np.array([[1, 11, 100, 17, 1000], [2, 11, 100, 18, 1001]])
    test = np.array(
        [
            [1, 11, 100, 17, 1000],
            [1, 11, 100, 18, 1001],
            [3, 11, 100, 18, 1001],
            [1, 11, 200, 17, 1002],
        ]
    )
    groups = REPORT.structural_subsets(connected, split, test)
    assert [groups[name].sum() for name in REPORT.SUBSETS] == [1, 1, 2]
    assert np.all(sum(mask.astype(int) for mask in groups.values()) == 1)
    assert list(groups["neither_same_or_uncovered"]) == [False, False, True, True]


def test_impossible_split_only_state_fails_instead_of_dropping_queries():
    connected = np.array([[1, 11, 100, 17, 1000]])
    split = np.array([[1, 11, 100, 18, 1001]])
    with pytest.raises(ValueError, match="outside the connected support partition"):
        REPORT.structural_subsets(connected, split, split)


def test_structural_groups_from_real_frozen_training_graphs():
    from llm_memory_editability.latent_support import build_world

    config_path = SCRIPT.parents[1] / "configs/latent-support-v1.json"
    specs = json.loads(config_path.read_text())["specs"]
    for world, expected_changed in zip(REPORT.WORLDS, (30, 24, 36), strict=True):
        worlds = {
            arm: build_world(next(s for s in specs if s["world"] == world and s["support"] == arm))
            for arm in REPORT.SUPPORTS
        }
        a, b = (worlds[arm] for arm in REPORT.SUPPORTS)
        np.testing.assert_array_equal(a["familiar_test"], b["familiar_test"])
        groups = REPORT.structural_subsets(
            a["train_composite"], b["train_composite"], a["familiar_test"]
        )
        assert groups["connected_only_same_component"].sum() == expected_changed
        assert sum(mask.sum() for mask in groups.values()) == len(a["familiar_test"])


@pytest.fixture
def raw_predictions():
    rows = np.array(
        [
            [1, 11, 100, 17, 1000],
            [2, 11, 100, 18, 1001],
            [3, 11, 100, 17, 1000],
            [4, 11, 100, 18, 1001],
        ]
    )
    predictions = {
        "familiar_test_generated": np.array(
            [[1000, 5, 1], [1001, 5, 1], [1000, 5, 9], [9999, 5, 1]]
        ),
        "familiar_test_correct": np.array([True, True, False, False]),
        "familiar_test_answer_nll": np.array([1, 2, 3, 4], dtype=float),
        "familiar_test_coverage": np.array([True, False, True, False]),
        "familiar_test_two_calls": np.array([True, True, True, False]),
    }
    return rows, predictions


def test_subsets_use_complete_predictions_and_report_n_format_and_prerequisites(raw_predictions):
    rows, predictions = raw_predictions
    values = REPORT.scores_from_predictions(predictions, "familiar_test", rows)
    flat = REPORT.metric_counts({"familiar_test": values})
    assert flat["familiar_test.n"] == 4
    assert flat["familiar_test.accuracy_count"] == 2
    assert flat["familiar_test.answer_accuracy_count"] == 3
    assert flat["familiar_test.format_gap_count"] == 1
    assert flat["familiar_test.conditional_n"] == 2
    assert flat["familiar_test.conditional_accuracy"] == 0.5
    subset = REPORT.scores_from_predictions(
        predictions, "familiar_test", rows, np.array([False, False, True, True])
    )
    assert subset["n"] == 2
    assert subset["accuracy"] == 0.0
    assert subset["answer_accuracy"] == 0.5
    assert subset["answer_nll"] == 3.5


def test_empty_subset_is_undefined_and_does_not_silently_select_worlds(raw_predictions, rows):
    world_rows, predictions = raw_predictions
    values = REPORT.scores_from_predictions(
        predictions, "familiar_test", world_rows, np.zeros(4, dtype=bool)
    )
    flat = REPORT.metric_counts({"familiar_test": values}, allow_empty=True)
    assert flat["familiar_test.n"] == 0
    assert flat["familiar_test.accuracy"] is None
    assert flat["familiar_test.accuracy_count"] == 0
    changed = copy.deepcopy(rows)
    changed[0]["familiar_test.accuracy"] = None
    assert REPORT.condition_means(changed)[0]["means"]["familiar_test.accuracy"] is None
    assert select(REPORT.paired_contrasts(changed, descriptive=True))[0]["effect"] is None
    assert (
        select(REPORT.paired_contrasts(changed, descriptive=True))[0]["priority"]
        == "descriptive secondary"
    )


def test_forged_correctness_and_incomplete_predictions_fail(raw_predictions):
    rows, predictions = raw_predictions
    predictions["familiar_test_correct"][2] = True
    with pytest.raises(AssertionError, match="Correctness differs from tokens"):
        REPORT.scores_from_predictions(predictions, "familiar_test", rows)
    predictions["familiar_test_generated"] = predictions["familiar_test_generated"][:3]
    with pytest.raises(ValueError, match="Incomplete generation"):
        REPORT.scores_from_predictions(predictions, "familiar_test", rows)


def test_matrix_requires_exact_pairing_and_only_support_changes():
    specs = json.loads((SCRIPT.parents[1] / "configs/latent-support-v1.json").read_text())["specs"]
    REPORT.validate_matrix(specs)
    with pytest.raises(ValueError, match="exactly the 12"):
        REPORT.validate_matrix(specs[:-1])
    changed = copy.deepcopy(specs)
    changed[0]["lr"] *= 2
    with pytest.raises(ValueError, match="nonexperimental"):
        REPORT.validate_matrix(changed)
    changed = copy.deepcopy(specs)
    changed[0]["composition_indices"][0] = changed[0]["composition_indices"][1]
    with pytest.raises(ValueError, match="512 unique"):
        REPORT.validate_matrix(changed)


def test_repeat_report_retains_all_fixed_budgets_without_selecting_best(rows):
    repeated = {}
    for row in rows:
        repeated[row["name"]] = {
            str(node): {
                str(repeat): scores(n=1000, accuracy=0.1 + repeat / 100) for repeat in (1, 2, 3, 4)
            }
            for node in (32000, 128000)
        }
    summaries = REPORT.repeat_report(rows, repeated)
    assert [(r["step"], r["test_repeats"]) for r in summaries] == list(
        itertools.product((32000, 128000), (1, 2, 3, 4))
    )
    assert all(select(r["contrasts"])[0]["priority"] == "descriptive secondary" for r in summaries)
    assert all(select(r["contrasts"])[0]["effect_pp"] == 0 for r in summaries)
