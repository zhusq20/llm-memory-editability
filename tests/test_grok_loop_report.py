"""Synthetic audits of scoring, paired compute, world weighting and prerequisites."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.grok_loop_data import build_world
from llm_memory_editability.grok_multihop_data import path_details

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_script("report_grok_loop")
audit = load_script("audit_grok_loop")


def make_run(arch="c2", world=1, initialization=7, scheme="scaled_effective"):
    layers, repeats = {"c1": (1, 1), "c2": (2, 1), "cd": (4, 1), "l1": (1, 4), "l2": (2, 2)}[arch]
    spec = {
        "phase": "confirmation",
        "world_seed": world,
        "initialization": initialization,
        "stream_seed": 7,
        "hops": 2,
        "layers": layers,
        "repeats": repeats,
        "architecture": arch,
        "init_scheme": scheme,
        "steps": 12,
    }
    history = [
        {
            "step": step,
            "estimated_training_flops": step * layers * repeats * 100,
            "parameters": layers * 100,
            "effective_depth": layers * repeats,
            **{
                name: {"accuracy": accuracy, "n": 10}
                for name in ("test_composite", "test_full_composite", "ood_composite")
            },
        }
        for step, accuracy in zip((0, 3, 5, 9, 12), (0.0, 0.99, 0.5, 0.7, 0.8), strict=True)
    ]
    return {
        "run": f"{arch}-w{world}-i{initialization}",
        "config": "confirmation.json",
        "spec": spec,
        "history": history,
        "endpoint": history[-1],
        "dataset_sha256": f"world-{world}",
    }


def test_paired_common_compute_uses_latest_eligible_node_not_best_accuracy():
    left, right = make_run("c2"), make_run("l2")
    pair = report.paired_comparisons([right, left])["pairs"][0]
    assert pair["pair"] == "l2_minus_c2"
    assert pair["common_training_flop_cap"] == 2400
    assert pair["left"]["common_budget_step"] == 12
    assert pair["right"]["common_budget_step"] == 5
    assert pair["right"]["common_budget_flops"] == 2000
    assert pair["right"]["unused_common_flops"] == 400
    assert pair["common_budget_id_probe_difference"] == pytest.approx(-0.3)
    assert pair["endpoint_id_full_difference"] == 0


@pytest.mark.parametrize(
    "mismatch", ["world", "initialization", "scheme", "stream", "data", "phase"]
)
def test_pairing_does_not_mix_unmatched_conditions(mismatch):
    left, right = make_run("c2"), make_run("l2")
    if mismatch == "data":
        right["dataset_sha256"] = "other"
    else:
        key = {"world": "world_seed", "scheme": "init_scheme", "stream": "stream_seed"}.get(
            mismatch, mismatch
        )
        right["spec"][key] = "other"
    assert report.paired_comparisons([left, right])["pairs"] == []


def test_all_four_registered_pairs_and_architecture_constraints_are_checked():
    runs = [make_run(arch) for arch in ("c1", "c2", "cd", "l1", "l2")]
    assert len(report.paired_comparisons(runs)["pairs"]) == 4
    runs[-1]["endpoint"]["parameters"] += 1
    with pytest.raises(ValueError, match="parameter"):
        report.paired_comparisons(runs)


def test_world_weighting_nests_seed_repeats_and_retains_zero_denominators():
    pairs = []
    for world, initialization, difference in ((1, 7, 0.0), (1, 8, 0.2), (2, 7, 1.0)):
        left, right = make_run("cd", world, initialization), make_run("l2", world, initialization)
        left["endpoint"]["test_full_composite"]["accuracy"] = 0.0
        right["endpoint"]["test_full_composite"]["accuracy"] = difference
        pairs.extend((left, right))
    summary = report.paired_comparisons(pairs)["world_summary"][0]
    assert summary["worlds"] == 2
    assert summary["paired_runs"] == 3
    assert summary["endpoint_id_full_difference"]["mean"] == pytest.approx(0.55)
    rows = [{"world_seed": 1, "ood": None}, {"world_seed": 2, "ood": 0.2}]
    assert report.mean_across_worlds(rows, "ood") == {
        "mean": 0.2,
        "min": 0.2,
        "max": 0.2,
        "worlds_with_denominator": 1,
    }
    assert report.mean_across_worlds(rows[:1], "ood")["mean"] is None


def test_world_summary_keeps_distinct_initialization_schemes_separate():
    rows = []
    for scheme, score in (("scaled_effective", 0.1), ("zero_residual", 0.9)):
        rows.append(
            {
                "config": "dev",
                "condition_id": scheme,
                "init_scheme": scheme,
                "phase": "development",
                "hops": 2,
                "architecture": "l2",
                "world_seed": 1,
                "parameters": 200,
                "effective_depth": 4,
                **{name: score for name in report.METRICS},
            }
        )
    result = report.group_world_means(rows)
    assert len(result) == 2
    assert [item["ood_composite"] for item in result] == [0.1, 0.9]


def test_known_atomic_subset_checks_eos_indices_and_reports_coverage():
    world = {
        "atomic": np.array([[2, 8, 3], [3, 9, 4], [5, 8, 6], [6, 9, 7], [3, 8, 5]]),
        "ood_composite": np.array([[2, 8, 9, 4], [5, 8, 9, 7], [2, 8, 8, 5]]),
    }
    pred = {
        "atomic_answer": world["atomic"][:, -1].copy(),
        "atomic_target": world["atomic"][:, -1].copy(),
        "atomic_stop": np.array([1, 1, 1, 0, 1]),
        "ood_composite_answer": np.array([4, 7, 5]),
        "ood_composite_target": np.array([4, 7, 5]),
        "ood_composite_stop": np.array([1, 1, 0]),
    }
    spec = {"entities": 6, "relations": 2}
    result = report.conditional_metrics(world, pred, "ood_composite", spec)
    assert result["all_atoms_correct_n"] == 2
    assert result["all_atoms_correct_coverage"] == 2 / 3
    assert result["all_atoms_correct_accuracy"] == 1 / 2
    assert result["multiple_targets_per_relation_sequence_n"] == 2
    assert result["multiple_targets_per_relation_sequence_accuracy"] == 1.0
    assert result["all_atoms_correct_and_multiple_targets_n"] == 1
    assert result["all_atoms_correct_and_multiple_targets_accuracy"] == 1.0
    pred["atomic_target"] = pred["atomic_target"][::-1]
    with pytest.raises(ValueError, match="align"):
        report.conditional_metrics(world, pred, "ood_composite", spec)
    world["ood_composite"] = world["ood_composite"][:0]
    result = report.conditional_metrics(world, {}, "ood_composite", spec)
    assert result["all_atoms_correct_n"] == 0
    assert result["all_atoms_correct_coverage"] is None


def test_threshold_cost_is_first_observed_checkpoint_without_persistence_rule():
    history = make_run()["history"]
    for row, accuracy in zip(history, (None, 0.91, 0.20, 0.95, 0.50), strict=True):
        row["test_composite"]["accuracy"] = accuracy
    assert report.first_observed_threshold(history, 0.9) == {
        "step": 3,
        "training_flops": 600,
        "observed_accuracy": 0.91,
    }
    assert report.first_observed_threshold(history, 0.95) == {
        "step": 9,
        "training_flops": 1800,
        "observed_accuracy": 0.95,
    }
    for row in history:
        row["test_composite"]["accuracy"] = None
    assert report.first_observed_threshold(history, 0.9) == {
        "step": None,
        "training_flops": None,
        "observed_accuracy": None,
    }
    history[-1]["test_composite"]["accuracy"] = 0.89
    assert report.first_observed_threshold(history, 0.9)["step"] is None


def depth_variant(arch, depth, scheme="zero_residual"):
    run = make_run(arch, scheme=scheme)
    layers, repeats = (depth, 1) if arch == "cd" else (2, depth // 2)
    run["spec"].update(layers=layers, repeats=repeats, phase="development-sensitivity")
    run["run"] += f"-d{depth}-{scheme}"
    for row in run["history"]:
        row.update(effective_depth=depth, parameters=layers * 100)
        row["estimated_training_flops"] = row["step"] * depth * 100
    return run


def test_sensitivity_pairing_separates_depths_and_initialization_schemes():
    runs = [
        depth_variant(arch, depth, scheme)
        for depth, scheme in ((4, "zero_residual"), (8, "zero_residual"), (8, "scaled_effective"))
        for arch in ("cd", "l2")
    ]
    result = report.paired_comparisons(runs[::-1])
    assert len(result["pairs"]) == len(result["world_summary"]) == 3
    assert {(pair["comparison_depth"], pair["init_scheme"]) for pair in result["pairs"]} == {
        (4, "zero_residual"),
        (8, "zero_residual"),
        (8, "scaled_effective"),
    }
    for pair in result["pairs"]:
        assert (
            pair["left"]["effective_depth"]
            == pair["right"]["effective_depth"]
            == pair["comparison_depth"]
        )
    assert (
        report.paired_comparisons([depth_variant("cd", 4), depth_variant("l2", 8)])["pairs"] == []
    )
    with pytest.raises(ValueError, match="Duplicate"):
        report.paired_comparisons([runs[0], copy.deepcopy(runs[0])])


def test_equal_parameter_pairs_preserve_distinct_loop_depths():
    baseline = make_run("c2", scheme="zero_residual")
    baseline["spec"]["phase"] = "development-sensitivity"
    result = report.paired_comparisons(
        [
            baseline,
            depth_variant("l2", 4),
            depth_variant("l2", 8),
        ]
    )
    assert len(result["pairs"]) == len(result["world_summary"]) == 2
    assert {pair["comparison_depth"] for pair in result["pairs"]} == {4, 8}
    assert all(pair["comparison"] == "equal_parameters" for pair in result["pairs"])


def perfect_node(id_fraction=0.75):
    spec = {
        "world_seed": 42,
        "entities": 8,
        "relations": 4,
        "degree": 3,
        "id_fraction": id_fraction,
        "id_test_fraction": 0.3,
        "hops": 2,
        "phi": 2,
        "width": 8,
        "layers": 2,
        "repeats": 2,
        "batch_size": 256,
        "n_atomic_per_batch": 32,
        "steps": 12,
    }
    world = build_world(spec)
    step, width, vocab = 5, 8, 14
    row = {
        "step": step,
        "hops": 2,
        "unique_layers": 2,
        "repeats": 2,
        "effective_depth": 4,
        "parameters": (vocab + 10) * width + 2 * (12 * width**2 + 13 * width),
        "counts": {"atomic": 32 * step, "composite": 224 * step},
        "examples": 256 * step,
        "supervised_tokens": 512 * step,
        "effective_input_tokens": (32 * 3 + 224 * 4) * step,
        "estimated_training_flops": step * audit.independent_flops_per_step(spec),
    }
    pred = {}
    for name in audit.SCORING_SPLITS:
        n = len(world[name])
        row[name] = {
            "n": n,
            "accuracy": 1.0 if n else None,
            "answer_accuracy": 1.0 if n else None,
            "nll": 0.0 if n else None,
        }
        if n:
            pred.update(
                {
                    name + "_answer": world[name][:, -1].copy(),
                    name + "_target": world[name][:, -1].copy(),
                    name + "_stop": np.ones(n),
                    name + "_nll": np.zeros((n, 2)),
                }
            )
    nodes, _ = path_details(world["test_composite"], world["atomic"], 8, 4)
    for j in (1, 2):
        pred[f"autonomous_hop{j}_answer"] = nodes[:, j]
        pred[f"autonomous_hop{j}_stop"] = np.ones(len(nodes))
    row["autonomous_calls"] = {
        "n": len(nodes),
        "calls": 2,
        "accuracy": 1.0,
        "answer_accuracy": 1.0,
        "hop_accuracy": [1.0, 1.0],
        "all_intermediate_answers_and_eos_correct": 1.0,
    }
    return spec, world, row, pred


@pytest.mark.parametrize("id_fraction", [0.75, 1.0])
def test_all_node_audit_recounts_scoring_exposure_flops_and_empty_ood(id_fraction):
    spec, world, row, pred = perfect_node(id_fraction)
    assert all(item["passed"] for item in audit.audit_node(row, pred, world, spec))
    row["counts"]["atomic"] += 1
    row["estimated_training_flops"] //= 2
    pred["atomic_stop"][0] = 0
    pred["autonomous_hop1_stop"][0] = 0
    failures = {
        item["name"] for item in audit.audit_node(row, pred, world, spec) if not item["passed"]
    }
    assert {
        "atomic_exposure",
        "FLOPs_independent_formula",
        "atomic:answer_eos",
        "autonomous_calls:accuracy",
    } <= failures


def test_paired_sampling_audit_detects_equal_totals_with_different_stream_state():
    runs = [
        {
            "run": arch,
            "sampling_condition": {"world_seed": 1, "hops": 2},
            "dataset_sha256": "same",
            "terminal_stream_sha256": "same",
            "endpoint_counts": {"atomic": 320, "composite": 2240},
        }
        for arch in ("cd", "l2")
    ]
    assert audit.audit_sampling_pairs(runs)[0]["passed"]
    runs[1]["terminal_stream_sha256"] = "different"
    assert not audit.audit_sampling_pairs(runs)[0]["passed"]
    state = {"remaining": np.array([1, 2]), "rng": {"number": 31}}
    assert audit.stream_state_hash(state) == audit.stream_state_hash(copy.deepcopy(state))


def test_step_and_actual_flop_plots_include_world_aggregation(tmp_path):
    rows = [
        {
            "config": "development.json",
            "condition_id": "c2",
            "architecture": "c2",
            "init_scheme": "scaled_effective",
            "hops": 2,
            "world_seed": world,
            "step": step,
            "training_flops": step * 1e9,
            "id": 0.1 * step * world,
            "ood": None,
        }
        for world in (1, 2)
        for step in (0, 1, 2)
    ]
    paths = report.save_learning_plots(rows, tmp_path)
    assert len(paths) == 2
    assert all(Path(path).stat().st_size > 1000 for path in paths)


def test_curve_labels_and_line_styles_distinguish_depth_and_shared_block_size():
    rows = [
        {
            "condition_id": f"d{depth}-{scheme}",
            "architecture": "l2",
            "effective_depth": depth,
            "unique_layers": 2,
            "repeats": depth // 2,
            "init_scheme": scheme,
        }
        for depth, scheme in ((4, "zero_residual"), (8, "zero_residual"), (8, "scaled_effective"))
    ]
    before = copy.deepcopy(rows)
    styles = report.learning_curve_styles(rows)
    assert len({label for label, _ in styles.values()}) == 3
    assert len({style for _, style in styles.values()}) == 3
    assert "D4 (b×R=2×2)" in styles["d4-zero_residual"][0]
    assert "D8 (b×R=2×4)" in styles["d8-zero_residual"][0]
    assert rows == before
