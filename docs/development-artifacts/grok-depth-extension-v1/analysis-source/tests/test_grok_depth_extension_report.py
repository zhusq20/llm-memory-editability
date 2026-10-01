"""Analysis safeguards: censoring survives aggregation and ratio reporting."""

import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/report_grok_depth_extension.py"
SPEC = importlib.util.spec_from_file_location("depth_extension_report", MODULE)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def run(world, seed, reached, depth=2, reused=True):
    node = {
        "step": 128000,
        "estimated_training_flops": 128e12,
        "parameters": 100,
        **{split: {"accuracy": 0.96} for split in report.SPLITS},
    }
    threshold = {
        "reached": reached,
        "step": 100000 if reached else None,
        "atomic_exposures": 3730 if reached else None,
        "estimated_training_flops": 100e12 if reached else None,
    }
    return {
        "run_id": f"{world}/{seed}/{depth}",
        "baseline_id": f"{world}/{seed}/2",
        "reused_baseline": reused,
        "spec": {"layers": depth, "world_seed": world, "initialization": seed},
        "status": "complete",
        "learning": [node],
        "fixed_budget_endpoint": node,
        "thresholds": {name: threshold.copy() for name in report.THRESHOLDS},
    }


def test_censored_initialization_is_not_dropped_from_world_mean():
    rows = [run(w, s, not (w == 1 and s == 1)) for w in (1, 2, 3) for s in (0, 1)]
    result = report.aggregate(rows)[0]
    assert result["thresholds"]["t95"]["reached_runs"] == 5
    assert result["thresholds"]["t95"]["world_mean_step"] is None
    assert result["worlds"][0]["thresholds"]["t95"]["mean_step"] is None
    assert result["world_mean_endpoint_accuracy"]["test_composite"] == 0.96


def test_endpoint_budget_ratio_does_not_become_uncensored_speed_ratio():
    baseline, new = run(1, 0, False), run(1, 0, True, depth=3, reused=False)
    comparison = report.comparisons([baseline, new])[0]
    assert (
        comparison["threshold_comparisons"]["paired_final"]["new_over_baseline_step_ratio"] is None
    )
    assert (
        comparison["paired_target_fraction_of_baseline_fixed_128k_budget"]["steps"]
        == 100000 / 128000
    )
    assert comparison["paired_target_fraction_of_baseline_fixed_128k_budget"][
        "not_an_exact_learning_speed_ratio"
    ]
