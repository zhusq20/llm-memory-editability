"""Saved-array alignment, EOS scoring, missingness and equal-world weighting."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[1] / "scripts/report_grok_loop_atomic_timing.py"
    spec = importlib.util.spec_from_file_location("loop_atomic_timing_report_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_spec(world=1, seed=1, **kwargs):
    return {
        "world_seed": world,
        "initialization": seed,
        "stream_seed": world,
        "phase": "confirmation",
        "hops": 2,
        "architecture": "l1",
        "width": 16,
        "layers": 1,
        "repeats": 4,
        "init_scheme": "scaled_effective",
        "steps": 100,
        **kwargs,
    }


def record(report, world=1, seed=1, value=1.0, **kwargs):
    return {
        **report.identity(model_spec(world, seed, **kwargs), f"w{world}-s{seed}", 100),
        "evaluated_repeats": 1,
        "split": "id_atomic",
        "kind": "atomic",
        "available": True,
        "missing_reason": "",
        "population_n": 3,
        "population_pool_n": 4,
        "partition_fraction": 0.75,
        "prediction_n": 3,
        "prediction_coverage": 1.0,
        "answer_accuracy": value,
        "complete_accuracy": value / 2,
    }


def test_atomic_membership_follows_complete_facts_in_shuffled_order(report):
    atoms = np.array([[9, 2, 4], [7, 2, 3], [2, 3, 9], [7, 3, 4]])
    world = {"atomic": atoms, "id_atomic": atoms[[3, 1]], "ood_atomic": atoms[[0, 2]]}
    masks = report.atomic_masks(world)
    assert masks["id_atomic"].tolist() == [False, True, False, True]
    archive = {
        "r1_atomic_answer": np.array([8, 3, 9, 4]),
        "r1_atomic_stop": np.array([1, 1, 1, 5]),
        "r1_atomic_target": atoms[:, -1],
    }
    score = report.recount(archive, "r1_atomic", atoms[:, -1], masks["id_atomic"])
    assert score["answer_accuracy"] == 1.0
    assert score["complete_accuracy"] == 0.5
    assert score["population_n"] == 2 and score["prediction_coverage"] == 1.0
    world["id_atomic"] = atoms[[0, 1]]
    with pytest.raises(ValueError, match="disjoint partition"):
        report.atomic_masks(world)


def test_missing_arrays_are_preserved_but_label_misalignment_is_rejected(report):
    target = np.array([4, 8])
    mask = np.ones(2, dtype=bool)
    empty = report.recount({}, "r8_atomic", target, mask)
    assert not empty["available"] and empty["prediction_coverage"] == 0
    assert empty["answer_accuracy"] is None and empty["population_n"] == 2
    archive = {"r1_atomic_" + key: target.copy() for key in ("answer", "stop", "target")}
    absent = report.recount(archive, "r1_atomic", target, mask, declared=False)
    assert absent["missing_reason"] == "repeat absent from scan summary"
    archive["r1_atomic_target"][0] += 1
    with pytest.raises(ValueError, match="aligned"):
        report.recount(archive, "r1_atomic", target, mask)


def test_initializations_average_within_world_then_worlds_equally(report):
    rows = [record(report, 1, seed, value) for seed, value in enumerate((0, 0, 1))]
    rows += [record(report, 2, seed, value) for seed, value in enumerate((0.8, 1.0))]
    rows[-1]["population_n"] = 1000
    _, worlds, groups, _ = report.summarize(rows, [])
    assert len(worlds) == 2 and len(groups) == 1
    metric = groups[0]["metrics"]["answer_accuracy"]
    assert metric["mean"] == pytest.approx((1 / 3 + 0.9) / 2)
    assert metric["min"] == pytest.approx(1 / 3)
    assert metric["max"] == pytest.approx(0.9)
    assert metric["worlds_with_denominator"] == 2


def test_phase_architecture_init_recipe_and_checkpoint_never_merge(report):
    rows = [record(report)]
    for changed in (
        {"phase": "development"},
        {"architecture": "l2", "layers": 2, "repeats": 2},
        {"init_scheme": "legacy_unique"},
        {"hops": 3},
    ):
        rows.append(record(report, **changed))
    rows.append({**rows[0], "checkpoint_step": 50})
    rows.append({**rows[0], "evaluated_repeats": 8})
    _, _, groups, _ = report.summarize(rows, [])
    assert len(groups) == 7


def test_missing_whole_models_and_repeats_are_explicit(report):
    expected = [
        report.identity(model_spec(world, seed), f"w{world}-s{seed}", 100)
        for world in (1, 2)
        for seed in (1, 2)
    ]
    row = record(report)
    _, _, groups, missing = report.summarize([row], expected)
    assert len(missing) == 4 * 8
    partial = next(r for r in missing if r["run"] == "w1-s1" and r["evaluated_repeats"] == 1)
    assert partial["missing_splits"] == ["ood_atomic", "test_composite", "ood_composite"]
    assert groups[0]["missing_world_initializations"] == [(1, 2), (2, 1), (2, 2)]
    assert groups[0]["expected_worlds"] == 2
    assert set(r["evaluated_repeats"] for r in missing) == set(range(1, 9))


def test_duplicate_scan_is_not_extra_evidence_and_conflicts_are_rejected(report):
    row = record(report)
    rows, _, _, _ = report.summarize([row, copy.deepcopy(row)], [])
    assert len(rows) == 1
    other = {**row, "complete_accuracy": 0}
    with pytest.raises(ValueError, match="Conflicting"):
        report.summarize([row, other], [])


def test_available_false_does_not_count_as_completed_model(report):
    expected = [report.identity(model_spec(1, seed), f"w1-s{seed}", 100) for seed in (1, 2)]
    missing = {
        **record(report, 1, 2),
        "available": False,
        "prediction_coverage": 0.0,
        "answer_accuracy": None,
        "complete_accuracy": None,
    }
    _, _, groups, _ = report.summarize([record(report), missing], expected)
    assert groups[0]["missing_world_initializations"] == [(1, 2)]
    assert groups[0]["registered_model_coverage"] == 0.5
    assert groups[0]["available_model_count"] == 1
    assert groups[0]["metrics"]["answer_accuracy"]["mean"] == 1.0
