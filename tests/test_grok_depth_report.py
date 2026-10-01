"""Synthetic report audits; no training or GPU is needed."""

import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPORT_PATH = Path(__file__).resolve().parents[1] / "scripts/report_grok_depth.py"
MODULE_SPEC = importlib.util.spec_from_file_location("grok_depth_report", REPORT_PATH)
report = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(report)


def make_run(world=1, layers=1, width=64, phase="confirmation", initialization=7, complete=True):
    spec = {
        "phase": phase,
        "world_seed": world,
        "initialization": initialization,
        "stream_seed": 8,
        "layers": layers,
        "width": width,
        "heads": 4,
        "steps": 12,
    }
    condition = {k: v for k, v in spec.items() if k not in report.NON_CONDITION}
    rows = [
        {
            "step": step,
            "examples": step * 10,
            "estimated_training_flops": step * layers * width / 64 * 100,
            "counts": {"atomic": step * 3, "composite": step * 7},
            **{
                split: {"n": 10, "accuracy": accuracy, "answer_accuracy": accuracy, "nll": 0.3}
                for split in report.SPLITS
            },
        }
        for step, accuracy in zip([0, 3, 5, 9, 12], [0.0, 0.99, 0.5, 0.7, 0.8], strict=True)
    ]
    counts = {"atomic": 10, "train_composite": 20}
    return {
        "phase": phase,
        "run_id": f"{phase}-W{world}-L{layers}-D{width}-I{initialization}",
        "status": "complete" if complete else "incomplete",
        "spec": spec,
        "condition": condition,
        "condition_id": report.stable_key(condition),
        "dataset_sha256": f"world-{world}",
        "budget_steps": 12,
        "data_counts": counts,
        "learning": rows,
        "fixed_budget_endpoint": rows[-1] if complete else None,
        "t90": report.t90(rows, counts, 12),
    }


def test_t90_requires_three_consecutive_nodes_and_records_confirmation_time():
    run = make_run()
    rows = run["learning"]
    for row, accuracy in zip(rows, [0.8, 0.95, 0.9, 0.91, 0.95], strict=True):
        row["test_composite"]["accuracy"] = accuracy
    threshold = report.t90(rows, run["data_counts"], 12)
    assert threshold["step"] == 3
    assert threshold["confirmed_at_step"] == 9
    assert threshold["previous_evaluation_step"] == 0
    rows[2]["test_composite"]["accuracy"] = 0.89
    censored = report.t90(rows, run["data_counts"], 12)
    assert not censored["reached"]
    assert censored["step"] is None
    assert censored["budget_steps"] == censored["last_observed_step"] == 12


def test_equal_compute_uses_last_eligible_node_even_when_an_earlier_node_is_better():
    left, right = make_run(layers=1), make_run(layers=2)
    result = report.paired_equal_compute([right, left])
    pair = result[0]["pairs"][0]
    assert pair["common_flop_cap"] == 1200
    assert pair["left"]["step"] == 12
    assert pair["right"]["step"] == 5  # Step 3 is 99%, but selection cannot use accuracy.
    assert pair["right"]["accuracy"]["test_composite"] == 0.5
    assert pair["right"]["estimated_training_flops"] == 1000
    assert pair["right"]["unused_common_cap_flops"] == 200
    assert pair["right"]["unrepresented_full_budget_flops"] == 1400
    assert pair["actual_flop_difference_right_minus_left"] == -200
    assert pair["accuracy_difference_right_minus_left"]["test_composite"] == pytest.approx(-0.3)
    # Fixed endpoint reporting remains distinct and unchanged.
    assert (
        report.paired_differences([left, right])[0]["pairs"][0][
            "accuracy_difference_right_minus_left"
        ]["test_composite"]
        == 0
    )


def test_equal_compute_compares_width_controls_and_keeps_exact_registered_nodes():
    narrow, wide = make_run(width=64), make_run(width=128)
    pair = report.paired_equal_compute([narrow, wide])[0]["pairs"][0]
    assert pair["left"]["step"] == 12
    assert pair["right"]["step"] == 5
    assert pair["left"]["unused_common_cap_flops"] == 0
    assert pair["right"]["unused_common_cap_flops"] == 200


@pytest.mark.parametrize(
    "mismatch", ["phase", "world", "initialization", "stream", "data", "status"]
)
def test_pairing_rejects_unmatched_or_incomplete_runs(mismatch):
    left, right = make_run(layers=1), make_run(layers=2)
    if mismatch == "phase":
        right["phase"] = right["spec"]["phase"] = "development"
    elif mismatch == "world":
        right["spec"]["world_seed"] = 2
    elif mismatch == "initialization":
        right["spec"]["initialization"] = 9
    elif mismatch == "stream":
        right["spec"]["stream_seed"] = 9
    elif mismatch == "data":
        right["dataset_sha256"] = "different-world"
    else:
        right["status"] = "incomplete"
    assert report.paired_equal_compute([left, right]) == []
    assert report.paired_differences([left, right]) == []


def test_world_means_do_not_treat_replicates_as_independent_worlds():
    runs = [
        make_run(world=1, initialization=7),
        make_run(world=1, initialization=8),
        make_run(world=2, initialization=7),
    ]
    for run, accuracy in zip(runs, [0.0, 0.2, 1.0], strict=True):
        run["fixed_budget_endpoint"]["test_composite"]["accuracy"] = accuracy
    grouped = report.group_world_means(runs)[0]
    assert grouped["independent_completed_world_count"] == 2
    assert grouped["world_mean_fixed_budget_accuracy"]["test_composite"] == pytest.approx(0.55)
    partners = [
        make_run(
            world=run["spec"]["world_seed"], layers=2, initialization=run["spec"]["initialization"]
        )
        for run in runs
    ]
    for partner, accuracy in zip(partners, [0.1, 0.3, 0.9], strict=True):
        partner["learning"][2]["test_composite"]["accuracy"] = accuracy
    paired = report.paired_equal_compute(runs + partners)[0]
    assert paired["independent_world_count"] == 2
    assert [world["paired_runs"] for world in paired["worlds"]] == [2, 1]
    assert paired["world_mean_accuracy_difference_right_minus_left"][
        "test_composite"
    ] == pytest.approx(0.0)


def test_saved_atomic_split_diagnostic_counts_eos_and_checks_row_alignment(tmp_path):
    np.savez(
        tmp_path / "world.npz",
        atomic=np.array([[2, 5, 3], [3, 5, 4], [4, 5, 2]]),
        id_atomic=np.array([[2, 5, 3], [4, 5, 2]]),
        ood_atomic=np.array([[3, 5, 4]]),
    )
    path = tmp_path / "predictions-0000010.npz"
    arrays = {
        "atomic_answer": np.array([3, 4, 2]),
        "atomic_stop": np.array([1, 0, 0]),
        "atomic_target": np.array([3, 4, 2]),
    }
    np.savez(path, **arrays)
    result = report.atomic_split_endpoint(tmp_path, {"step": 10})
    assert result["id_atomic"] == {"n": 2, "answer_accuracy": 1.0, "accuracy": 0.5}
    assert result["ood_atomic"] == {"n": 1, "answer_accuracy": 1.0, "accuracy": 0.0}
    arrays["atomic_target"] = arrays["atomic_target"][::-1]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="do not align"):
        report.atomic_split_endpoint(tmp_path, {"step": 10})


def test_load_keeps_incomplete_results_but_excludes_them_from_fixed_budget(tmp_path):
    for phase in ("development", "confirmation"):
        directory = tmp_path / phase / "example"
        directory.mkdir(parents=True)
        run = make_run(phase=phase)
        (directory / "metadata.json").write_text(json.dumps({"spec": run["spec"]}))
        rows = copy.deepcopy(run["learning"])
        if phase == "development":
            rows = rows[:-1]
        (directory / "learning.json").write_text(json.dumps(rows))
        (directory / "complete.json").write_text(json.dumps({"endpoint": run["learning"][-1]}))
    runs = report.load_runs(tmp_path)
    assert [run["status"] for run in runs] == ["incomplete", "complete"]
    assert runs[0]["fixed_budget_endpoint"] is None
    assert runs[0]["last_observed_step"] == 9
    assert runs[0]["warnings"]
    assert len(report.group_world_means(runs)) == 2


def test_plot_and_csv_artifacts_can_be_created_without_gpu(tmp_path):
    pytest.importorskip("matplotlib")
    runs = [make_run(world=world, layers=layers) for world in (1, 2, 3) for layers in (1, 2)]
    report.save_csv(runs, tmp_path / "learning.csv")
    plots = report.save_confirmation_primary(runs, tmp_path)
    assert len(plots) == 2
    assert all(Path(path).stat().st_size > 1000 for path in plots)
    with (tmp_path / "learning.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert all(row["phase"] == "confirmation" for row in rows)
