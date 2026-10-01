"""Guard against a reversed matching contrast and mismatched paired controls."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_followup.py"
SPEC = importlib.util.spec_from_file_location("followup_summary", PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def cell(condition, chain, value, weight="uniform", kind="exception"):
    return {
        "world": 0,
        "seed": 0,
        "step": 512,
        "condition": condition,
        "chain": chain,
        "kind": kind,
        "weight": weight,
        "alpha": 0.25,
        **{metric: value for metric in summary.METRICS},
    }


def test_matching_contrast_is_within_query_then_averaged_and_keeps_neutral_control():
    rows = [
        cell("company", "company", 0.9),
        cell("project", "company", 0.5),
        cell("company", "project", 0.2),
        cell("project", "project", 0.4),
        cell("neither", "company", 0.1),
        cell("neither", "project", 0.3),
    ]
    result = summary.organization_effects(rows)[0]
    assert result["matching_D_probe_conflict_heldout_accuracy"] == pytest.approx(0.3)
    assert result["mismatched_minus_neither_D_probe_conflict_heldout_accuracy"] == pytest.approx(
        0.15
    )
    with pytest.raises(ValueError, match="Missing"):
        summary.organization_effects(rows[:-1])


def test_pairing_never_crosses_update_types_or_query_chains():
    rows = [
        cell("company", "company", 0.2),
        cell("company", "company", 0.7, kind="coherent"),
        cell("company", "project", 0.9),
        cell("company", "company", 0.3, weight="root250"),
        cell("company", "company", 0.6, weight="root250", kind="coherent"),
    ]
    differences = summary.paired_effects(rows)
    assert differences[0]["delta_D_probe_conflict_heldout_accuracy"] == pytest.approx(0.1)
    assert differences[1]["delta_D_probe_conflict_heldout_accuracy"] == pytest.approx(-0.1)


def test_unknown_retention_is_not_scored_as_zero_damage():
    baseline = cell("company", "company", 0.2)
    treatment = cell("company", "company", 0.3, weight="root250")
    baseline["U_unseen_strata_0_rate"] = None
    result = summary.paired_effects([baseline, treatment])[0]
    assert result["delta_U_unseen_strata_0_rate"] is None
    assert summary.average([None, None]) is None


def test_recomputed_scores_reject_changed_integers():
    with pytest.raises(AssertionError):
        summary.assert_metrics({"D": {"n": 9, "correct": 2}}, {"D": {"n": 9, "correct": 3}})


def test_update_interaction_is_exception_minus_coherent_on_same_block():
    points = []
    for kind, value in (("coherent", 0.1), ("exception", -0.2)):
        points.append(
            {
                "world": 1,
                "seed": 0,
                "weight": "root250",
                "step": 512,
                "kind": kind,
                "matching_D_probe_conflict_heldout_accuracy": value,
                "matching_D_heldout_accuracy": value / 2,
            }
        )
    result = summary.update_interactions(points)[0]
    assert result[
        "exception_minus_coherent_matching_D_probe_conflict_heldout_accuracy"
    ] == pytest.approx(-0.3)
    assert result["exception_minus_coherent_matching_D_heldout_accuracy"] == pytest.approx(-0.15)
