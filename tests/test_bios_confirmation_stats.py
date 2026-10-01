"""Artificial endpoint counts only: never generate confirmation worlds."""

import copy
import csv
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from llm_memory_editability.bios_confirmation_stats import (
    analyze_endpoints,
    holm,
    sign_flip_sensitivity,
    t_critical,
    t_two_sided_p,
    validate_endpoints,
    world_t_statistics,
)


def fake_endpoints():
    learning, editing = [], []
    for world, seed, condition, chain, prevalence in itertools.product(
        range(100, 108),
        (0, 1),
        ("company", "project", "neither"),
        ("company", "project"),
        ("low", "high"),
    ):
        base = {
            "world": world,
            "seed": seed,
            "condition": condition,
            "chain": chain,
            "prevalence": prevalence,
        }
        increase = world - 99 if prevalence == "high" else 0
        learning.append(
            {
                **base,
                "step": 15360,
                "cohort": "original_exception",
                "split": "heldout",
                "n": 64,
                "correct": 30 + increase,
            }
        )
        for support, kind in itertools.product((0, 1), ("coherent", "exception")):
            baseline = 1 if kind == "exception" else 8
            direction = 1 if kind == "exception" else -1
            editing.append(
                {
                    **base,
                    "support": support,
                    "kind": kind,
                    "step": 512,
                    "n": 9,
                    "correct": baseline + direction * increase,
                }
            )
    return learning, editing


def test_t_distribution_known_values_and_symmetry():
    assert t_two_sided_p(0, 7) == 1
    assert t_two_sided_p(1, 1) == pytest.approx(0.5, abs=1e-12)
    assert t_two_sided_p(-2.364624251592785, 7) == pytest.approx(0.05, abs=1e-12)
    assert t_critical(0.05, 7) == pytest.approx(2.364624251592785, abs=1e-11)
    assert t_critical(0.05 / 3, 7) == pytest.approx(3.127552274246371, abs=1e-11)
    with pytest.raises(ValueError):
        t_critical(0, 7)


def test_holm_known_values_ties_and_unavailable():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.02, 0.02, 0.9]) == pytest.approx([0.06, 0.06, 0.9])
    assert holm([0.01, None, 0.02]) == pytest.approx([0.03, 1, 0.04])
    with pytest.raises(ValueError):
        holm([float("nan")])


def test_degenerate_statistics_are_explicit_and_conservative():
    zero = world_t_statistics([0] * 8)
    assert zero["status"] == "degenerate_zero"
    assert (zero["t"], zero["p"], zero["ci95"]) == (0, 1, [0, 0])
    assert zero["sign_flip_sensitivity"]["p"] == 1
    nonzero = world_t_statistics([0.25] * 8)
    assert nonzero["status"] == "degenerate_nonzero"
    assert nonzero["t"] is nonzero["p"] is nonzero["ci95"] is None
    assert holm([nonzero["p"], 0.01, 0.02])[0] == 1
    assert nonzero["sign_flip_sensitivity"]["p"] == 2 / 256


def test_world_statistics_and_sign_flips():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    result = world_t_statistics(values)
    assert result["mean"] == 4.5
    assert result["sd"] == pytest.approx(6**0.5)
    assert result["t"] == pytest.approx(4.5 / (6 / 8) ** 0.5)
    assert result["ci_bonferroni"][0] < result["ci95"][0]
    assert result["ci_bonferroni"][1] > result["ci95"][1]
    assert sign_flip_sensitivity(values) == {"p": 2 / 256, "extreme": 2, "enumerated": 256}
    assert sign_flip_sensitivity([-x for x in values]) == sign_flip_sensitivity(values)
    assert sign_flip_sensitivity([-1, 1] * 4)["p"] == 1
    with pytest.raises(ValueError):
        world_t_statistics([0] * 7)


def test_complete_nested_aggregation_never_replicates_learning_by_support():
    learning, editing = fake_endpoints()
    result = analyze_endpoints(learning, editing)
    assert result["input_rows"] == {"learning": 192, "editing": 768}
    assert len(result["world_effects"]) == 24
    assert len(result["paired_cases"]) == 480
    for row in result["world_effects"]:
        if row["endpoint"] == "learning_original_exception":
            assert row["paired_cases"] == 12
            assert row["nested_query_count_descriptive"] == 768
            assert row["difference"] == (row["world"] - 99) / 64
        else:
            assert row["paired_cases"] == 24
            assert row["nested_query_count_descriptive"] == 216
            expected = (row["world"] - 99) / 9
            assert row["difference"] == (expected if "exception" in row["endpoint"] else -expected)
    assert {row["endpoint"] for row in result["results"]} == {
        "editing_exception_fixed9",
        "learning_original_exception",
        "editing_coherent_fixed9",
    }
    assert all(row["df"] == 7 for row in result["results"])


@pytest.mark.parametrize("editing", [False, True])
def test_duplicate_missing_unexpected_denominator_and_step_rejected(editing):
    rows = fake_endpoints()[int(editing)]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_endpoints([*rows, rows[0]], editing=editing)
    with pytest.raises(ValueError, match="missing 1"):
        validate_endpoints(rows[:-1], editing=editing)
    for field, value in [("n", 63 if not editing else 8), ("step", 128)]:
        changed = copy.deepcopy(rows)
        changed[0][field] = value
        with pytest.raises(ValueError):
            validate_endpoints(changed, editing=editing)
    changed = copy.deepcopy(rows)
    changed[0]["world"] = 99
    with pytest.raises(ValueError, match="unexpected 1"):
        validate_endpoints(changed, editing=editing)


def test_learning_support_and_wrong_cohort_rejected():
    rows = fake_endpoints()[0]
    rows[0]["support"] = 0
    with pytest.raises(ValueError, match="support"):
        validate_endpoints(rows)
    rows[0].pop("support")
    rows[0]["cohort"] = "all"
    with pytest.raises(ValueError, match="cohort"):
        validate_endpoints(rows)


def test_stats_entrypoint_requires_complete_audit_hashes_and_frozen_design(tmp_path):
    script_path = Path(__file__).parents[1] / "scripts/analyze_bios_confirmation_stats.py"
    spec = importlib.util.spec_from_file_location("confirmation_stats_entrypoint", script_path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    summary, output, design = (tmp_path / name for name in ("summary", "stats", "design"))
    summary.mkdir()
    script.write_design(design)
    rows = fake_endpoints()
    audit = {
        "complete": True,
        "outputs_sha256": {},
        "weight_archive_verified": True,
        "archive_index_sha256": "a" * 64,
    }
    for name, table in zip(("learning-endpoint.csv", "editing-endpoint.csv"), rows, strict=True):
        with (summary / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=table[0])
            writer.writeheader()
            writer.writerows(table)
        audit["outputs_sha256"][name] = script.digest(summary / name)
    (summary / "audit.json").write_text(json.dumps(audit))
    assert script.analyze(summary, output, design)["complete"]
    assert json.loads((output / "statistics-audit.json").read_text())["world_count"] == 8
    audit["complete"] = False
    (summary / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="incomplete"):
        script.analyze(summary, output, design)
    audit["complete"] = True
    audit["weight_archive_verified"] = False
    (summary / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="weight-archive"):
        script.analyze(summary, output, design)
    audit["weight_archive_verified"] = True
    audit["outputs_sha256"]["learning-endpoint.csv"] = "wrong"
    (summary / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="hash mismatch"):
        script.analyze(summary, output, design)
    (design / "statistics-contract.json").write_text("{}")
    with pytest.raises(ValueError, match="design changed"):
        script.analyze(summary, output, design)
