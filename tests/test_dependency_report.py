"""Prediction-only contracts for the complete dependency-gate comparison."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report_dependency_gate as report  # noqa: E402


def scores(prediction, mask=None):
    if mask is None:
        mask = np.ones(len(prediction["ids"]), dtype=bool)
    correct = prediction["answer"][mask] == prediction["labels"][mask, 0]
    exact = correct & (prediction["stop"][mask] == prediction["labels"][mask, 1])
    return {
        "n": int(mask.sum()),
        "correct": int(correct.sum()),
        "answer_accuracy": float(correct.mean()) if len(correct) else None,
        "accuracy": float(exact.mean()) if len(exact) else None,
    }


def metrics(prediction):
    value = scores(prediction)
    for feature in report.FEATURES:
        value[f"by_{feature}"] = {
            str(item): scores(prediction, prediction[feature] == item)
            for item in np.unique(prediction[feature])
        }
    value["by_join_presence"] = {
        "0": scores(prediction, prediction["join_count"] == 0),
        "ge1": scores(prediction, prediction["join_count"] >= 1),
    }
    return value


@pytest.fixture
def matrix(tmp_path):
    config = {
        "batch": "report-fixture",
        "arms": {name: {} for name in report.ARMS},
        "tasks": {},
    }
    for task, native, extra in (("listops", 20, 24), ("arithmetic", 15, 20)):
        config["tasks"][task] = {
            "training": {"steps": 100000},
            "repeats": native,
            "inference_repeats": [1, native, extra],
            "input_origin": {
                "world_sha256": f"world-{task}",
                "panels_sha256": f"panels-{task}",
                "train_majority_token": 5,
            },
        }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    output = tmp_path / "execution"
    correct_native = {
        "loop": [0, 1],
        "only-u": [0, 1, 2],
        "disco": [1, 2],
        "fixed-q-plus-u": [1, 2, 3],
    }
    correct_extra = {
        "loop": [0, 2],
        "only-u": [0],
        "disco": [1, 2, 3],
        "fixed-q-plus-u": [0, 1, 2, 3],
    }
    for task, settings in config["tasks"].items():
        for arm in report.ARMS:
            run = output / task / "runs" / arm
            run.mkdir(parents=True)
            complete = {
                "state": "complete",
                "step": 100000,
                "budget": 100000,
                "arm": arm,
                "engineering_only": False,
                "model_sha256": f"model-{task}-{arm}",
                "config_sha256": config_sha,
            }
            (run / "complete.json").write_text(json.dumps(complete))
            grid = {
                "task": task,
                "arm": arm,
                "step": 100000,
                "training_repeats": settings["repeats"],
                "engineering_only": False,
                "model_sha256": complete["model_sha256"],
                "config_sha256": config_sha,
                "world_sha256": settings["input_origin"]["world_sha256"],
                "panels_sha256": settings["input_origin"]["panels_sha256"],
                "inference_repeats": settings["inference_repeats"],
                "metrics": {},
            }
            for count in settings["inference_repeats"]:
                correct_rows = (
                    correct_extra[arm] if count == report.OVER[task] else correct_native[arm]
                )
                truth = np.array([4, 4, 4, 5])
                prediction = {
                    "ids": np.array([b"d", b"a", b"c", b"b"], dtype="S64"),
                    "labels": np.column_stack((truth, np.full(4, 2))),
                    "answer": np.full(4, 9),
                    "stop": np.full(4, 2),
                    "nll": np.zeros((4, 2), dtype=np.float32),
                    "depth": np.array([1, 2, 2, 3]),
                    "tree_depth": np.array([2, 3, 3, 4]),
                    "length": np.array([8, 10, 12, 14]),
                    "branch_imbalance": np.array([0, 1, 1, 2]),
                    "computed_branch_imbalance": np.array([0, 0, 1, 1]),
                    "join_count": np.array([0, 1, 2, 0]),
                }
                prediction["answer"][correct_rows] = truth[correct_rows]
                if arm == "only-u":
                    prediction["stop"][2] = 0  # Answer improvement need not improve EOS exactness.
                if count == report.OVER[task]:
                    permutation = np.array([3, 1, 0, 2])
                    prediction = {key: value[permutation] for key, value in prediction.items()}
                grid["metrics"][str(count)] = {
                    name: metrics(prediction) for name in report.ALL_PANELS
                }
                folder = run / "grid" / f"r{count}"
                folder.mkdir(parents=True)
                for pool in report.TESTS:
                    np.savez_compressed(folder / f"{pool}.npz", **prediction)
            (run / "grid.json").write_text(json.dumps(grid))
    return config_path, config, output


def test_complete_report_matches_ids_keeps_denominators_and_uses_training_majority(matrix):
    config_path, _, output = matrix
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in output.rglob("*.npz")
    }
    result = report.build_report(config_path, output)
    assert result["complete"] and result["scientific_runs"] == 8
    assert result["supplied_config_bytes_verified"]
    for task in report.TASKS:
        item = result["tasks"][task]
        assert item["primary"]["test_iid"]["loop"]["n"] == 4
        pair = item["paired"]["only-u-minus-loop"]["test_iid"]
        assert pair["answer"]["gained"] == 1 and pair["answer"]["lost"] == 0
        assert pair["answer"]["delta_accuracy"] == 0.25
        assert pair["answer_eos"]["delta_accuracy"] == 0
        transition = item["extra_rounds"]["loop"]["test_iid"]["answer"]
        assert transition["original_correct"] == transition["final_correct"] == 2
        assert transition["kept"] == transition["lost"] == transition["gained"] == 1
        assert transition["retention_denominator"] == 2 and transition["retention_rate"] == 0.5
        assert transition["denominator"] == 4
        assert item["training_majority_token"] == 5
        assert item["training_majority_baseline"]["test_iid"]["answer_accuracy"] == 0.25
    assert (output / "comparison.json").is_file() and (output / "report.md").is_file()
    assert "八臂固定终点" in (output / "report.md").read_text()
    assert report.build_report(config_path, output) == result
    assert all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )


@pytest.mark.parametrize("missing", ("grid.json", "complete.json", "grid/r1/test_iid.npz"))
def test_any_missing_run_or_test_grid_prevents_a_completion_report(matrix, missing):
    config_path, _, output = matrix
    (output / "arithmetic" / "runs" / "fixed-q-plus-u" / missing).unlink()
    with pytest.raises(FileNotFoundError):
        report.build_report(config_path, output)
    assert not (output / "comparison.json").exists()
    assert not (output / "report.md").exists()


def test_unfinished_run_cannot_be_summarized(matrix):
    config_path, _, output = matrix
    path = output / "listops" / "runs" / "only-u" / "complete.json"
    value = json.loads(path.read_text())
    value["step"] = 75000
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unfinished"):
        report.build_report(config_path, output)
    assert not (output / "comparison.json").exists()


def test_duplicate_ids_fail_before_output(matrix):
    config_path, _, output = matrix
    path = output / "listops" / "runs" / "disco" / "grid" / "r20" / "test_depth.npz"
    with np.load(path) as saved:
        values = dict(saved)
    values["ids"][1] = values["ids"][0]
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="Nonunique"):
        report.build_report(config_path, output)
    assert not (output / "comparison.json").exists()


def test_saved_accuracy_disagreement_fails_before_output(matrix):
    config_path, _, output = matrix
    path = output / "arithmetic" / "runs" / "disco" / "grid.json"
    value = json.loads(path.read_text())
    value["metrics"]["15"]["test_depth"]["answer_accuracy"] = 1.0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="answer_accuracy mismatch"):
        report.build_report(config_path, output)
    assert not (output / "comparison.json").exists()


def test_changed_truth_between_arms_is_rejected_even_when_scores_match(matrix):
    config_path, _, output = matrix
    run = output / "listops" / "runs" / "disco"
    path = run / "grid" / "r20" / "test_iid.npz"
    with np.load(path) as saved:
        values = dict(saved)
    values["labels"][0, 0] = 8
    np.savez_compressed(path, **values)
    grid = json.loads((run / "grid.json").read_text())
    grid["metrics"]["20"]["test_iid"] = metrics(values)
    (run / "grid.json").write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="Unpaired examples or truth"):
        report.build_report(config_path, output)
    assert not (output / "comparison.json").exists()


def test_missing_train_majority_never_falls_back_to_test_majority(matrix):
    _, config, output = matrix
    config = copy.deepcopy(config)
    del config["tasks"]["arithmetic"]["input_origin"]["train_majority_token"]
    with pytest.raises(ValueError, match="majority token"):
        report.build_report(config, output)
    assert not (output / "comparison.json").exists()


def test_config_file_identity_is_checked(matrix):
    config_path, config, output = matrix
    config["batch"] = "changed"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="supplied file"):
        report.build_report(config_path, output)
    assert not (output / "comparison.json").exists()


def test_zero_correct_transition_preserves_empty_conditional_denominator():
    value = report._transitions(np.array([False, False]), np.array([True, False]))
    assert value["denominator"] == 2 and value["retention_denominator"] == 0
    assert value["retention_rate"] is None and value["gain_rate_given_original_wrong"] == 0.5
    json.dumps(value, allow_nan=False)
