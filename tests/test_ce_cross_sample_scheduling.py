"""A scheduling handoff must never duplicate an active or completed write."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ce_scheduling",
    Path(__file__).resolve().parents[1] / "scripts/resume_ce_cross_sample_scheduling.py",
)
scheduler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scheduler)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def batch(tmp_path):
    for arm in ("transformer", "loop"):
        save(
            tmp_path / arm / "completion-manifest.json",
            {"state": "complete", "passed": True, "scientific_updates": 4480},
        )
    save(tmp_path / "discoloop/controller-state.json", {"state": "waiting_for_resources"})
    save(
        tmp_path / "discoloop/output/prepare/result.json",
        {"passed": True, "scientific_optimizer_updates": 0},
    )
    return tmp_path


def test_only_finished_arms_and_unstarted_science_are_eligible(batch):
    outcomes, waiting = scheduler.eligible(batch)
    assert set(outcomes) == {"transformer", "loop"}
    assert waiting["state"] == "waiting_for_resources"


@pytest.mark.parametrize("arm", ["transformer", "loop"])
def test_any_incomplete_architecture_blocks_handoff(batch, arm):
    save(batch / arm / "completion-manifest.json", {"state": "running", "passed": False})
    with pytest.raises(ValueError, match="finish all four stages"):
        scheduler.eligible(batch)


def test_started_discoloop_science_cannot_repeat(batch):
    save(batch / "discoloop/launch-evaluate.json", {"id": "existing"})
    with pytest.raises(FileExistsError, match="never be resubmitted"):
        scheduler.eligible(batch)


def test_running_discoloop_blocks_handoff(batch):
    save(batch / "discoloop/controller-state.json", {"state": "running", "stage": "evaluate"})
    with pytest.raises(ValueError, match="still be waiting"):
        scheduler.eligible(batch)


def test_failed_preparation_cannot_be_repeated(batch):
    save(batch / "discoloop/output/prepare/result.json", {"passed": False})
    with pytest.raises(ValueError, match="preparation must already have passed"):
        scheduler.eligible(batch)


def test_wrong_budget_blocks_handoff(batch):
    save(
        batch / "loop/completion-manifest.json",
        {"state": "complete", "passed": True, "scientific_updates": 4479},
    )
    with pytest.raises(ValueError, match="budget"):
        scheduler.eligible(batch)


def test_comparison_cannot_be_replaced(batch):
    save(batch / "report/launch-report.json", {"id": "existing"})
    with pytest.raises(FileExistsError, match="comparison"):
        scheduler.eligible(batch)
