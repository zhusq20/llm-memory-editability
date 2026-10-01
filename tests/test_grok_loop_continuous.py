"""Continuous-score conditioning, saved alignment and nested-world contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[1] / "scripts/report_grok_loop_continuous.py"
    spec = importlib.util.spec_from_file_location("loop_continuous_report_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predictions(target, p=(0.8, 0.2), eos=(0.9, 0.1), answers=None, stops=None, prefix="atomic"):
    target = np.asarray(target)
    return {
        prefix + "_answer": target.copy() if answers is None else np.asarray(answers),
        prefix + "_stop": np.ones(len(target), dtype=np.int64)
        if stops is None
        else np.asarray(stops),
        prefix + "_target": target.copy(),
        prefix + "_nll": -np.log(np.c_[p, eos]),
    }


def model_spec(world=1, seed=1, **changes):
    return {
        "phase": "confirmation",
        "world_seed": world,
        "initialization": seed,
        "stream_seed": world * 10 + seed,
        "architecture": "l1",
        "hops": 2,
        "layers": 1,
        "repeats": 4,
        "init_scheme": "scaled_effective",
        "width": 16,
        "phi": 6,
        "steps": 20,
        "nodes": [0, 10, 20],
        **changes,
    }


def record(report, world=1, seed=1, value=0.5, step=0, **changes):
    return {
        **report.identity(model_spec(world, seed, **changes), f"w{world}-s{seed}", "config.json"),
        "step": step,
        "split": "ood_composite",
        "available": True,
        "population_n": 2,
        "prediction_n": 2,
        "training_flops": step * 10,
        **{metric: value for metric in report.METRICS},
    }


def test_mean_item_probability_is_not_exponentiated_mean_nll(report):
    archive = predictions([4, 8], p=(0.9, 0.1), eos=(0.8, 0.4))
    result = report.score_saved(archive, "atomic", np.array([4, 8]))
    assert result["p_answer"] == pytest.approx(0.5)
    assert np.exp(-result["answer_nll"]) == pytest.approx(0.3)
    assert result["p_eos_given_gold"] == pytest.approx(0.6)
    assert result["p_gold_sequence"] == pytest.approx((0.9 * 0.8 + 0.1 * 0.4) / 2)
    assert result["p_gold_sequence"] != pytest.approx(
        result["p_answer"] * result["p_eos_given_gold"]
    )


def test_teacher_forced_eos_is_separate_from_generated_answer_eos(report):
    archive = predictions([4, 8], p=(0.8, 0.1), eos=(0.9, 0.9), answers=[4, 9], stops=[5, 1])
    result = report.score_saved(archive, "atomic", np.array([4, 8]))
    assert result["p_eos_given_gold"] == pytest.approx(0.9)
    assert result["answer_accuracy"] == 0.5
    assert result["complete_accuracy"] == 0
    assert result["generated_eos_accuracy"] == 0.5
    assert result["generated_eos_given_correct_answer_accuracy"] == 0
    assert result["answer_correct_eos_wrong_fraction"] == 0.5
    assert result["answer_accuracy"] - result["complete_accuracy"] == 0.5


def test_zero_conditional_denominator_and_absent_population_are_null(report):
    archive = predictions([4, 8], answers=[6, 6])
    result = report.score_saved(archive, "atomic", np.array([4, 8]))
    assert result["correct_answer_n"] == 0
    assert result["generated_eos_given_correct_answer_accuracy"] is None
    empty = report.score_saved({}, "atomic", np.array([], dtype=int))
    assert empty["available"] and empty["population_n"] == 0
    assert all(empty[metric] is None for metric in report.METRICS)
    assert empty["missing_reason"] == "empty population"
    missing = report.score_saved({}, "atomic", np.array([4, 8]))
    assert not missing["available"] and missing["prediction_coverage"] == 0
    assert missing["p_answer"] is None


def test_labels_shape_and_invalid_losses_cannot_silently_change_score(report):
    target = np.array([4, 8])
    archive = predictions(target)
    archive["atomic_target"] = target[::-1]
    with pytest.raises(ValueError, match="aligned"):
        report.score_saved(archive, "atomic", target)
    archive = predictions(target)
    archive["atomic_nll"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        report.score_saved(archive, "atomic", target)
    archive["atomic_nll"][0, 0] = -0.001
    with pytest.raises(ValueError, match="nonnegative"):
        report.score_saved(archive, "atomic", target)
    with pytest.raises(ValueError, match="mask"):
        report.score_saved(archive, "atomic", target, np.array([0, 1]))


def test_atomic_masks_use_complete_fact_membership_and_duplicate_audit_order(report):
    atoms = np.array([[9, 2, 4], [7, 2, 3], [2, 3, 9], [7, 3, 4]])
    world = {"atomic": atoms, "id_atomic": atoms[[3, 1]], "ood_atomic": atoms[[0, 2]]}
    masks = report.atomic_masks(world)
    assert masks["id_atomic"].tolist() == [False, True, False, True]
    archive = predictions(atoms[:, -1], p=(0.1, 0.2, 0.3, 0.4), eos=(0.9, 0.8, 0.7, 0.6))
    for split, indices in (("id_atomic", [3, 1]), ("ood_atomic", [0, 2])):
        for key in ("answer", "stop", "target", "nll"):
            archive[split + "_" + key] = archive["atomic_" + key][indices]
    report.verify_atomic_duplicates(archive, world)
    score = report.score_saved(archive, "atomic", atoms[:, -1], masks["id_atomic"])
    assert score["p_answer"] == pytest.approx(0.3)
    archive["id_atomic_answer"][0] = 7
    with pytest.raises(ValueError, match="disagree"):
        report.verify_atomic_duplicates(archive, world)
    world["id_atomic"] = atoms[[0, 1]]
    with pytest.raises(ValueError, match="disjoint"):
        report.atomic_masks(world)


def test_initializations_nest_within_equally_weighted_worlds(report):
    rows = [record(report, 1, seed, value) for seed, value in enumerate((0, 0, 1))]
    rows += [record(report, 2, seed, value) for seed, value in enumerate((0.8, 1))]
    rows[-1]["population_n"] = 10000
    worlds, groups = report.nested_summary(rows)
    assert len(worlds) == 2 and len(groups) == 1
    group = groups[0]
    assert group["p_answer"] == pytest.approx((1 / 3 + 0.9) / 2)
    assert group["p_answer_world_min"] == pytest.approx(1 / 3)
    assert group["p_answer_world_max"] == pytest.approx(0.9)
    assert group["p_answer_worlds_with_denominator"] == 2


def test_missing_initializations_are_explicit_and_conditions_never_merge(report):
    missing = record(report, 1, 2)
    missing.update(available=False, **{metric: None for metric in report.METRICS})
    rows = [
        record(report),
        missing,
        record(report, init_scheme="zero_residual"),
        record(report, phase="development"),
        record(report, hops=3),
        record(report, step=10),
    ]
    worlds, groups = report.nested_summary(rows)
    assert len(groups) == 5
    first = groups[0]
    assert first["registered_runs"] == 2 and first["available_runs"] == 1
    assert worlds[0]["expected_initializations"] == 2
    assert worlds[0]["available_initializations"] == 1
    assert worlds[0]["p_answer_initializations_with_denominator"] == 1


def test_decline_rebound_records_actual_irregular_saved_intervals(report):
    rows = [
        record(report, value=value, step=step)
        for step, value in ((0, 0.8), (100, 0.3), (8000, 0.9))
    ]
    adjacent, reversals, summary = report.dynamics(rows)
    assert [row["step_gap"] for row in adjacent] == [100, 7900]
    soft = next(row for row in reversals if row["metric"] == "p_answer")
    assert soft["decline"] == 0.5 and soft["rebound"] == pytest.approx(0.6)
    assert soft["decline_step_gap"] == 100 and soft["rebound_step_gap"] == 7900
    details = next(row for row in summary if row["metric"] == "p_answer")
    assert details["endpoint"] == 0.9
    assert details["decline_rebounds_both_ge_5pp"] == 1
    rows[1].update(available=False, **{metric: None for metric in report.METRICS})
    adjacent, reversals, _ = report.dynamics(rows)
    assert not reversals
    assert all(row["p_answer_delta"] is None for row in adjacent)


def test_autonomous_complete_requires_eos_at_every_external_call(report):
    archive = {
        "autonomous_hop1_answer": np.array([6, 7]),
        "autonomous_hop1_stop": np.array([1, 9]),
        "autonomous_hop2_answer": np.array([4, 8]),
        "autonomous_hop2_stop": np.array([1, 1]),
    }
    result = report.autonomous_hard(archive, np.array([4, 8]), 2)
    assert result["answer_accuracy"] == 1
    assert result["complete_accuracy"] == 0.5
    assert not result["continuous_available"]


def test_saved_matrix_collection_validates_all_nodes_and_endpoint_full_split(report, tmp_path):
    cfg = {"output_root": "runs", "base": model_spec(), "runs": {"toy": {}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    directory = tmp_path / "runs/confirmation/toy"
    directory.mkdir(parents=True)
    atoms = np.array([[2, 130, 3], [3, 130, 2], [4, 131, 5]], dtype=np.int64)
    comp = np.array([[2, 130, 130, 2], [3, 130, 130, 3]], dtype=np.int64)
    world = {
        "atomic": atoms,
        "id_atomic": atoms[:2],
        "ood_atomic": atoms[2:],
        "train_composite": comp,
        "test_composite": comp,
        "test_full_composite": comp[::-1],
        "ood_composite": comp,
        "unused_composite": np.empty((0, 4), dtype=np.int64),
    }
    np.savez(directory / "world.npz", **world)
    (directory / "metadata.json").write_text(
        json.dumps({"spec": model_spec(), "files": {str(path): report.digest(path)}})
    )
    (directory / "world-metadata.json").write_text(
        json.dumps(
            {"dataset_sha256": report.dataset_digest(world), "probe_indices_in_full_test": [1, 0]}
        )
    )
    history = []
    for step in cfg["base"]["nodes"]:
        archive, saved = (
            {},
            {
                "step": step,
                "estimated_training_flops": step * 10,
                "examples": step * 2,
                "supervised_tokens": step * 4,
            },
        )
        for split in report.SPLITS + (("test_full_composite",) if step == 20 else ()):
            target = world[split][:, -1]
            archive.update(
                predictions(target, p=[0.8] * len(target), eos=[0.9] * len(target), prefix=split)
            )
            score = report.score_saved(archive, split, target)
            saved[split] = {
                "n": len(target),
                "answer_accuracy": score["answer_accuracy"],
                "accuracy": score["complete_accuracy"],
                "nll": score["two_target_nll"],
            }
        for hop in (1, 2):
            archive[f"autonomous_hop{hop}_answer"] = world["test_composite"][:, -1]
            archive[f"autonomous_hop{hop}_stop"] = np.ones(2, dtype=int)
        saved["autonomous_calls"] = {"answer_accuracy": 1, "accuracy": 1}
        np.savez(directory / f"predictions-{step:07d}.npz", **archive)
        history.append(saved)
    (directory / "learning.json").write_text(json.dumps(history))
    rows, provenance, matrix, errors, autonomous = report.collect([path], root=tmp_path)
    assert len(rows) == 3 * 6 + 1 and len(autonomous) == 3
    assert not errors and matrix[0]["present_nodes"] == [0, 10, 20]
    assert len([row for row in provenance if row["kind"] == "predictions"]) == 3
    assert [row["step"] for row in rows if row["split"] == "test_full_composite"] == [20]
    (directory / "predictions-0000010.npz").unlink()
    rows, _, matrix, errors, _ = report.collect([path], root=tmp_path)
    assert matrix[0]["missing_nodes"] == [10]
    assert len(errors) == 6
    assert all(row["p_answer"] is None for row in rows if row["step"] == 10)
