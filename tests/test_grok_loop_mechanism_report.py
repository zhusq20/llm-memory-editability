"""World weighting, complete matrices, and no performance-based scan selection."""

import copy
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[1] / "scripts/report_grok_loop_mechanism.py"
    spec = importlib.util.spec_from_file_location("loop_mechanism_report_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_spec(world=1, seed=1, phase="confirmation"):
    return {
        "world_seed": world,
        "initialization": seed,
        "stream_seed": world,
        "phase": phase,
        "hops": 2,
        "architecture": "l1",
        "width": 16,
        "layers": 1,
        "repeats": 4,
        "init_scheme": "scaled_effective",
        "steps": 100,
    }


def record(report, world, seed, value, phase="confirmation", n=10):
    return {
        **report.common(model_spec(world, seed, phase), f"{phase}-w{world}-s{seed}", 100),
        "kind": "intervention",
        "split": "test_composite",
        "donor_seed": 5,
        "analysis_version": "version",
        "group": "all_rows",
        "condition": "e000_different_mlp",
        "family": "different",
        "component": "mlp",
        "execution_layer": 1,
        "evaluated_repeats": -1,
        "metrics": {"target_complete_accuracy": value, "n": n, "coverage": 1.0},
    }


def test_initializations_nest_within_equally_weighted_worlds(report):
    rows = [record(report, 1, seed, value) for seed, value in enumerate((0.0, 0.0, 1.0))]
    rows += [record(report, 2, seed, value, n=1000) for seed, value in enumerate((0.8, 1.0))]
    worlds, groups = report.aggregate(rows)
    assert len(worlds) == 2 and len(groups) == 1
    metric = groups[0]["metrics"]["target_complete_accuracy"]
    assert metric["mean"] == pytest.approx((1 / 3 + 0.9) / 2)
    assert metric["min"] == pytest.approx(1 / 3)
    assert metric["max"] == pytest.approx(0.9)
    assert metric["worlds_with_denominator"] == 2


def test_phase_node_donor_seed_and_analysis_version_never_merge(report):
    base = record(report, 1, 1, 0.2)
    rows = [base, record(report, 1, 1, 0.9, phase="development")]
    for field, value in (("step", 50), ("donor_seed", 9), ("analysis_version", "other")):
        rows.append({**base, field: value})
    _, groups = report.aggregate(rows)
    assert len(groups) == 5
    with pytest.raises(ValueError, match="Duplicate"):
        report.aggregate([base, copy.deepcopy(base)])


def test_overlapping_identical_scans_count_once_and_conflicts_are_rejected(report):
    row = record(report, 1, 1, 0.2)
    unique, count = report.deduplicate_records([row, copy.deepcopy(row)])
    assert unique == [row] and count == 1
    conflicting = copy.deepcopy(row)
    conflicting["metrics"]["target_complete_accuracy"] = 0.3
    with pytest.raises(ValueError, match="Conflicting"):
        report.deduplicate_records([row, conflicting])


def test_missing_worlds_seeds_and_denominators_stay_explicit(report):
    expected = [
        report.common(model_spec(world, seed), f"w{world}-s{seed}", 100)
        for world in (1, 2, 3)
        for seed in (1, 2)
    ]
    rows = [record(report, 1, 1, 0.75), record(report, 2, 1, None, n=0)]
    _, groups = report.aggregate(rows, expected)
    group = groups[0]
    assert group["expected_worlds"] == 3 and group["worlds_observed"] == 2
    assert group["registration_complete"] is False
    assert group["missing_world_initializations"] == [(1, 2), (2, 2), (3, 1), (3, 2)]
    assert group["metrics"]["target_complete_accuracy"]["mean"] == 0.75
    assert group["metrics"]["target_complete_accuracy"]["worlds_with_denominator"] == 1
    observed = [{**expected[0], "kind": "mechanism", "split": "test_composite"}]
    complete = report.completeness(expected, observed, ["test_composite", "ood_composite"])
    assert complete["expected_endpoints"] == 12
    assert complete["completed_expected_endpoints"] == 1
    assert len(complete["missing"]) == 11
    assert report.completeness([], observed, ["test_composite"])["expected_endpoints"] is None


def test_recurrence_keeps_every_count_and_separates_answer_from_eos(report):
    summary = {
        "source": {"scan.py": "sourcehash"},
        "runs": [
            {
                "run": "toy",
                "spec": model_spec(),
                "checkpoint": {"weights-0000100.pt": "hash"},
                "measurements": [
                    {
                        "repeats": repeat,
                        "effective_depth": repeat,
                        "atomic": {"n": 20, "answer_accuracy": value, "accuracy": value / 2},
                        "test_composite": {"n": 10, "answer_accuracy": 1.0, "accuracy": value},
                        "ood_composite": {"n": 3, "answer_accuracy": value, "accuracy": 0.0},
                    }
                    for repeat, value in ((1, 0.2), (4, 0.8), (8, 0.4))
                ],
            }
        ],
    }
    rows = report.recurrence_records(summary)
    assert len(rows) == 9
    assert {r["evaluated_repeats"] for r in rows} == {1, 4, 8}
    assert all(r["training_repeats"] == 4 for r in rows)
    atomic = next(r for r in rows if r["split"] == "atomic" and r["evaluated_repeats"] == 4)
    assert atomic["metrics"]["answer_accuracy"] == 0.8
    assert atomic["metrics"]["accuracy"] == 0.4


def test_coverage_zeros_and_all_component_layer_rows_are_retained(report):
    conditions = {
        f"e{layer:03d}_different_{component}": {
            "family": "different",
            "patch_layer": layer,
            "component": component,
        }
        for layer in range(4)
        for component in report.COMPONENTS
    }
    summary = {
        "split": "test_composite",
        "conditions": conditions,
        "scores": {
            "all_rows": {
                "conditions": {
                    name: {"n": 0, "coverage": 0.0, "target_complete_accuracy": None}
                    for name in conditions
                }
            }
        },
        "donor_coverage": {
            "different": {"n": 0, "total_n": 10, "counterfactual_split": {"missing": 10}}
        },
        "atomic_preconditions": {},
    }
    metadata = {
        "spec": model_spec(),
        "step": 100,
        "donor_seed": 2,
        "analysis_source_hashes": {
            "src/grok_loop_mechanism.py": "implementation",
            "scripts/analyze_grok_loop_mechanism.py": "cli",
            "tests/test_grok_loop_mechanism.py": "tests",
        },
    }
    rows = report.mechanism_records(summary, metadata, "toy")
    assert len(rows) == 17
    interventions = [r for r in rows if r["kind"] == "intervention"]
    assert all(r["analysis_version"] == "implementation" for r in rows)
    assert {r["execution_layer"] for r in interventions} == {1, 2, 3, 4}
    donor = next(r for r in rows if r["kind"] == "donor_coverage")
    assert donor["metrics"]["counterfactual_train_n"] == 0
    assert donor["metrics"]["counterfactual_reserved_test_n"] == 0
    assert donor["metrics"]["missing_n"] == 10


def test_mechanism_plot_shows_unpatched_score_on_matched_new_targets(report, tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib.figure")
    patch = record(report, 1, 1, 0.17)
    matched = {
        **copy.deepcopy(patch),
        "condition": "different_matched_baseline",
        "component": "none",
        "execution_layer": 0,
    }
    counterfactual = {
        **copy.deepcopy(matched),
        "condition": "different_counterfactual_input",
        "metrics": {**matched["metrics"], "target_complete_accuracy": 0.6},
    }
    unmatched = {
        **copy.deepcopy(matched),
        "condition": "baseline",
        "family": "none",
        "metrics": {**matched["metrics"], "target_complete_accuracy": 0.99},
    }
    _, groups = report.aggregate([patch, matched, counterfactual, unmatched])
    before = copy.deepcopy(groups)
    figures = []
    original = matplotlib.Figure.savefig

    def capture(figure, path, **kwargs):
        if "mechanism-different" in str(path) and str(path).endswith(".png"):
            figures.append(figure)
        return original(figure, path, **kwargs)

    monkeypatch.setattr(matplotlib.Figure, "savefig", capture)
    report.plot_mechanisms(groups, tmp_path)
    assert groups == before
    axis = figures[0].axes[0]
    lines = {line.get_label(): line for line in axis.lines}
    assert lines["Unpatched response scored on new target"].get_ydata() == [17.0, 17.0]
    assert lines["Unpatched response scored on new target"].get_linestyle() == ":"
    assert lines["Full counterfactual input"].get_ydata() == [60.0, 60.0]
    assert lines["Full counterfactual input"].get_linestyle() == "--"
