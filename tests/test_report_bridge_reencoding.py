"""Selection and independent-unit contracts for the re-encoding report."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts" / "report_bridge_reencoding.py"
SPEC = importlib.util.spec_from_file_location("report_bridge_reencoding", MODULE)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def row(alpha, familiar, strict=0.0, drop=0.0, arm="bridge_ce", world=1, initialization=1):
    return dict(
        arm=arm,
        world=world,
        initialization=initialization,
        variant="norm_matched",
        condition="self_decode",
        alpha=alpha,
        values=dict(
            familiar_accuracy=familiar, strict_accuracy=strict, atomic_delta_from_baseline=drop
        ),
    )


def test_selection_uses_only_bridge_ce_familiar_and_bridge_ce_atomic_retention():
    rows = [
        row(0, 0.1, 1),
        row(0.1, 0.7, 0),
        row(0.5, 0.9, 1, drop=-0.02),
        row(1, 0.7, 1),
        row(0.1, 0.99, 1, drop=-1, arm="baseline"),
    ]
    decision = reporter.development_decision(rows)
    assert decision["selected_alpha"] == 0.1
    assert not decision["strict_used_for_selection"]
    assert not decision["other_training_arms_used_for_selection"]


def test_hierarchical_world_average_does_not_weight_world_by_initialization_count():
    rows = [
        row(0.1, 1, world=1, initialization=1),
        row(0.1, 1, world=1, initialization=2),
        row(0.1, 0, world=2, initialization=1),
    ]
    result = reporter.hierarchical_mean(rows)
    assert result["values"]["familiar_accuracy"] == 0.5
    assert result["worlds"] == 2 and result["models"] == 3


def test_missing_alpha_in_one_development_model_is_ineligible():
    rows = [row(0, 0.1), row(0.1, 1), row(0, 0.2, world=2)]
    assert reporter.development_decision(rows)["selected_alpha"] == 0


def test_confirmation_configuration_rejects_ambiguous_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"selected_alpha": 0.1}))
    assert reporter.fixed_alpha_from_config(path) == 0.1
    path.write_text(json.dumps({"selected_alpha": 0.1, "confirmation_alpha": 0.5}))
    with pytest.raises(ValueError, match="unambiguous"):
        reporter.fixed_alpha_from_config(path)


def test_report_retains_all_alphas_and_uses_parent_model_counts(tmp_path):
    for arm in ("baseline", "bridge_ce", "aligned"):
        directory = tmp_path / arm
        directory.mkdir()
        cases = {"baseline": dict(condition="baseline", alpha=0, variant="norm_matched")}
        cases.update(
            {
                f"a{alpha}": dict(condition="self_decode", alpha=alpha, variant="norm_matched")
                for alpha in (0, 0.1)
            }
        )
        metrics = {}
        for name, case in cases.items():
            metrics[name] = {}
            for group in ("common_atomic", "train_composite", "familiar_test", "strict_test"):
                metrics[name][group] = dict(
                    n=10,
                    accuracy=0.8 if case["alpha"] else 0.1,
                    fixed_baseline_atoms_subset=dict(n=10, coverage=1, accuracy=0.5),
                )
                if group == "common_atomic":
                    metrics[name][group]["accuracy"] = 1
        (directory / "run.json").write_text(
            json.dumps(
                dict(
                    parent_spec=dict(phase="development-v2", arm=arm, world=1, initialization=2),
                    cases=cases,
                    spec={},
                )
            )
        )
        (directory / "metrics.json").write_text(json.dumps(metrics))
        (directory / "complete.json").write_text(json.dumps(dict(independently_reloaded=True)))
    summary, decision = reporter.report(tmp_path, "development")
    assert decision["selected_alpha"] == 0.1 and decision["ready_for_frozen_comparison"]
    assert summary["parent_models"] == 3 and summary["evaluated_cases"] == 9
    assert len(summary["all_endpoints"]) == 9
    assert (tmp_path / "report-development" / "decision.json").exists()
