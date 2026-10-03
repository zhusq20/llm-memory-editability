"""Conclusion-bearing contracts for world weighting, pairing and denominators."""

import copy
import importlib.util
import itertools
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_latent_confirmation.py"
MODULE_SPEC = importlib.util.spec_from_file_location("latent_confirmation_report", SCRIPT)
REPORT = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(REPORT)


def scores(n=1000, accuracy=0.1, coverage=1.0):
    result = {
        task: {
            "n": n,
            "accuracy": accuracy,
            "answer_accuracy": accuracy + 0.01,
            "answer_nll": 2.0,
        }
        for task in REPORT.TASKS
    }
    for task in ("familiar_test", "strict_test"):
        result[task].update(
            atomic_correct_coverage=coverage,
            conditional_accuracy=accuracy if coverage else None,
            autonomous_two_calls=coverage,
        )
    return result


@pytest.fixture
def rows():
    result = []
    for world_index, world in enumerate(REPORT.WORLDS):
        for seed_index, initialization in enumerate(REPORT.INITIALIZATIONS):
            for count, (layers, repeats) in itertools.product(REPORT.COUNTS, REPORT.ARCHITECTURES):
                base = 0.1 + 0.1 * world_index + 0.01 * seed_index
                delta_r2 = (
                    0.02 + 0.02 * seed_index + (0.02 * (world_index + 1) if count == "all" else 0)
                )
                delta_r3 = (
                    0.03 + 0.02 * seed_index + (0.03 * (world_index + 1) if count == "all" else 0)
                )
                value = base
                if layers == 2:
                    value += delta_r2 + 0.05
                elif repeats == 2:
                    value += delta_r2
                elif repeats == 3:
                    value += delta_r3
                row = {
                    "name": f"w{world}-i{initialization}-n{count}-l{layers}-r{repeats}",
                    "world": world,
                    "initialization": initialization,
                    "composition_count": count,
                    "layers": layers,
                    "repeats": repeats,
                    "steps": 128000,
                    **REPORT.metric_counts(scores(n=1000 * (world_index + 1), accuracy=value)),
                    "role_coverage.first_atoms_used": 150,
                }
                result.append(row)
    return result


def select(contrasts, name, level="all_worlds", task="familiar_test", metric="accuracy"):
    return [
        row
        for row in contrasts
        if row["contrast"] == name
        and row["level"] == level
        and row["task"] == task
        and row["metric"] == metric
    ]


def test_interaction_sign_and_both_replication_levels_are_retained(rows):
    contrasts = REPORT.paired_contrasts(rows)
    primary = select(contrasts, "interaction_R2")[0]
    assert primary["priority"] == "primary"
    assert select(contrasts, "interaction_R2", task="common_atomic")[0]["priority"] == "diagnostic"
    assert (
        select(contrasts, "interaction_R2", metric="answer_accuracy")[0]["priority"] == "diagnostic"
    )
    assert primary["effect_pp"] == pytest.approx(4.0)
    assert primary["n_worlds"] == 3
    assert primary["n_initializations"] == 6
    worlds = select(contrasts, "interaction_R2", level="world")
    assert [r["effect_pp"] for r in worlds] == pytest.approx([2.0, 4.0, 6.0])
    seeds = select(contrasts, "interaction_R2", level="seed")
    assert len(seeds) == 6
    assert [t["coefficient"] for t in seeds[0]["terms"]] == [1, -1, -1, 1]
    assert [t["n"] for t in seeds[0]["terms"]] == [1000] * 4
    assert [t["conditional_n"] for t in seeds[0]["terms"]] == [1000] * 4
    assert select(contrasts, "interaction_R3")[0]["effect_pp"] == pytest.approx(6.0)


def test_execution_depth_comparator_uses_standard_minus_loop(rows):
    contrasts = REPORT.paired_contrasts(rows)
    for count in REPORT.COUNTS:
        value = select(contrasts, f"standard2-Loop2_n{count}")[0]
        assert value["effect_pp"] == pytest.approx(5.0)


def test_world_weighting_does_not_pool_queries_or_initializations(rows):
    selected = [
        r for r in rows if r["composition_count"] == 256 and r["layers"] == 1 and r["repeats"] == 1
    ]
    selected = [
        r
        for r in selected
        if not (r["world"] == REPORT.WORLDS[0] and r["initialization"] == REPORT.INITIALIZATIONS[1])
    ]
    condition = REPORT.condition_means(selected)[0]
    assert [w["n_initializations"] for w in condition["worlds"]] == [1, 2, 2]
    assert condition["means"]["familiar_test.accuracy"] == pytest.approx((0.1 + 0.205 + 0.305) / 3)
    assert [w["denominators"]["familiar_test"] for w in condition["worlds"]] == [
        [1000],
        [2000, 2000],
        [3000, 3000],
    ]
    assert condition["means"]["familiar_test.accuracy"] != pytest.approx(
        sum(r["familiar_test.accuracy"] for r in selected) / len(selected)
    )


def test_pairing_rejects_missing_and_duplicate_endpoints(rows):
    with pytest.raises(ValueError, match="Missing paired endpoint"):
        REPORT.paired_contrasts(rows[1:])
    with pytest.raises(ValueError, match="Duplicate endpoint"):
        REPORT.paired_contrasts([*rows, rows[0]])


def test_denominators_and_format_failures_are_separate():
    metrics = scores(n=100, accuracy=0.6, coverage=0.5)
    for task in REPORT.TASKS:
        metrics[task]["answer_accuracy"] = 0.8
    metrics["familiar_test"]["conditional_accuracy"] = 0.4
    flattened = REPORT.metric_counts(metrics)
    assert flattened["familiar_test.n"] == 100
    assert flattened["familiar_test.conditional_n"] == 50
    assert flattened["familiar_test.conditional_accuracy_count"] == 20
    assert flattened["familiar_test.accuracy_count"] == 60
    assert flattened["familiar_test.format_gap"] == pytest.approx(0.2)
    assert flattened["familiar_test.format_gap_count"] == 20


def test_undefined_conditionals_do_not_silently_select_seeds(rows):
    changed = copy.deepcopy(rows)
    changed[0]["familiar_test.conditional_accuracy"] = None
    changed[0]["familiar_test.conditional_n"] = 0
    aggregate = REPORT.condition_means(changed)[0]
    assert aggregate["means"]["familiar_test.conditional_accuracy"] is None
    contrasts = REPORT.paired_contrasts(changed)
    assert select(contrasts, "interaction_R2", metric="conditional_accuracy")[0]["effect"] is None
    assert select(contrasts, "interaction_R2")[0]["effect"] == pytest.approx(0.04)
    flattened = REPORT.metric_counts(scores(coverage=0))
    assert flattened["strict_test.conditional_n"] == 0
    assert flattened["strict_test.conditional_accuracy_count"] is None


@pytest.mark.parametrize(
    "change", [{"accuracy": 0.99999}, {"accuracy": 0.2, "answer_accuracy": 0.1}]
)
def test_inconsistent_fractional_counts_or_format_scores_fail(change):
    metrics = scores()
    metrics["common_atomic"].update(change)
    with pytest.raises(ValueError):
        REPORT.metric_counts(metrics)


def test_fixed_weight_repeat_comparisons_do_not_choose_best_repeat(rows):
    repeated = {}
    for row in rows:
        if row["layers"] != 1:
            continue
        values = {}
        for tested_repeat in (1, 2, 3, 4):
            value = row["familiar_test.accuracy"] + {1: 0, 2: 0.01, 3: -0.02, 4: 0.2}[tested_repeat]
            values[str(tested_repeat)] = scores(n=row["familiar_test.n"], accuracy=value)
        repeated[row["name"]] = {"128000": values}
    summary = REPORT.repeat_secondary(rows, repeated)
    for count in REPORT.COUNTS:
        fixed = select(
            summary["trained_R2_fixed_weight_R3_minus_R2"], f"trainedR2_testR3-testR2_n{count}"
        )[0]
        assert fixed["effect_pp"] == pytest.approx(-3.0)
    uniform = select(summary["uniform_test_R2_contrasts"], "testR2_interaction_R2")[0]
    assert uniform["effect_pp"] == pytest.approx(4.0)


def test_matrix_requires_all_frozen_conditions():
    runner_spec = importlib.util.spec_from_file_location(
        "latent_confirmation_runner", SCRIPT.with_name("run_latent_confirmation.py")
    )
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    specs = runner.specifications()
    REPORT.validate_matrix(specs)
    with pytest.raises(ValueError, match="exactly the 48"):
        REPORT.validate_matrix(specs[:-1])
    broken = copy.deepcopy(specs)
    broken[0]["lr"] *= 2
    with pytest.raises(ValueError, match="nonexperimental"):
        REPORT.validate_matrix(broken)
