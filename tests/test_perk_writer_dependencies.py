"""A finished training flag cannot substitute for independent audit evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_perk_writer_dependencies.py"
spec = importlib.util.spec_from_file_location("perk_writer_dependencies", SCRIPT)
dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependencies)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    base = tmp_path / "original"
    write(base / "controller-state.json", {"state": "complete", "scientific_training": "complete"})
    write(base / "frozen/freeze.json", {"source_files": {}})
    freeze_sha = dependencies.sha(base / "frozen/freeze.json")
    write(
        base / "completion-manifest.json",
        {
            "passed": True,
            "scientific_training": True,
            "independent_full_test_audits": True,
            "cloud_verified": True,
            "freeze_sha256": freeze_sha,
            "controller_state_sha256": dependencies.sha(base / "controller-state.json"),
        },
    )
    cloud = {"passed": True, "runs": {}}
    comparison = {
        "rows": 999,
        "predictions_sha256": "predictions",
        "generated_tokens": 1234,
        "metrics": {"accuracy": 0.5},
    }
    for arm in dependencies.ARMS:
        root = base / "runs" / arm
        checkpoint = root / "train/checkpoint"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixed reader")
        write(
            root / "train/completion.json",
            {"global_step": 18000, "best_checkpoint": str(checkpoint)},
        )
        write(
            root / "native-training-endpoint.json",
            {
                "training_complete": True,
                "completion_sha256": dependencies.sha(root / "train/completion.json"),
            },
        )
        write(
            root / "checkpoint-selection.json",
            {
                "checkpoint": str(checkpoint),
                "selected_by": "validation_loss_only",
                "files": {str(checkpoint): dependencies.sha(checkpoint)},
            },
        )
        write(
            root / "audit.json",
            comparison
            | {
                "passed": True,
                "freeze_sha256": freeze_sha,
                "checkpoint_files": {str(checkpoint): dependencies.sha(checkpoint)},
            },
        )
        cloud["runs"][arm] = {
            "finished": True,
            "exact_step_verified": True,
            "local_optimizer_steps": 18000,
            "cloud_final_step": 17999,
        }
    write(base / "wandb-cloud-verification.json", cloud)
    monkeypatch.setattr(
        dependencies,
        "frozen_executor",
        lambda *args: SimpleNamespace(
            verify_freeze=lambda *args, **kwargs: None,
            compare_replay=lambda *args: comparison.copy(),
        ),
    )
    return base


def test_training_complete_without_replay_remains_pending(completed):
    (completed / "runs/perk/audit.json").unlink()
    result = dependencies.inspect_dependencies(completed)
    assert not result["ready_for_calibration"]
    assert "perk/audit.json is absent" in result["pending"]
    assert not result["launch_performed"]


def test_complete_audited_endpoint_allows_calibration_only(completed):
    result = dependencies.inspect_dependencies(completed)
    assert result["ready_for_calibration"]
    assert not result["scientific_extension_frozen"]
    assert not result["launch_performed"]
    assert result["arms"]["perk"]["audited_rows"] == 999


@pytest.mark.parametrize(
    "artifact", ("checkpoint", "freeze", "cloud", "predictions", "completion", "selection")
)
def test_changed_or_partial_evidence_cannot_unlock_calibration(completed, artifact):
    if artifact == "checkpoint":
        (completed / "runs/perk/train/checkpoint").write_bytes(b"changed reader")
    elif artifact == "freeze":
        write(completed / "frozen/freeze.json", {"source_files": {}, "changed": True})
    elif artifact == "cloud":
        path = completed / "wandb-cloud-verification.json"
        row = dependencies.read(path)
        row["runs"]["perk"]["cloud_final_step"] = 100
        write(path, row)
    elif artifact == "predictions":
        path = completed / "runs/perk/audit.json"
        row = dependencies.read(path)
        row["predictions_sha256"] = "changed predictions"
        write(path, row)
    elif artifact == "completion":
        path = completed / "runs/perk/train/completion.json"
        row = dependencies.read(path)
        row["best_checkpoint"] = "another checkpoint with the same training step"
        write(path, row)
    else:
        path = completed / "runs/perk/checkpoint-selection.json"
        row = dependencies.read(path)
        row["files"] = {"another checkpoint": "another hash"}
        write(path, row)
    result = dependencies.inspect_dependencies(completed)
    assert not result["ready_for_calibration"]
    assert result["invalid"]


def test_execution_failure_cannot_be_hidden_by_completed_training(completed):
    path = completed / "controller-state.json"
    write(path, {"state": "finished_with_failures", "scientific_training": "complete"})
    result = dependencies.inspect_dependencies(completed)
    assert not result["ready_for_calibration"]
    assert "original execution has recorded failures" in result["invalid"]


def test_loading_frozen_executor_does_not_write_original_batch(tmp_path):
    relative = "scripts/execute_perk_reproduction.py"
    path = tmp_path / "frozen/source" / relative
    path.parent.mkdir(parents=True)
    path.write_text("def verify_freeze(*args, **kwargs):\n    return 'original'\n")
    frozen = {"source_files": {relative: dependencies.sha(path)}}
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    module = dependencies.frozen_executor(tmp_path, frozen)
    assert module.verify_freeze() == "original"
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    path.write_text("# changed source\n")
    with pytest.raises(ValueError, match="source changed"):
        dependencies.frozen_executor(tmp_path, frozen)
