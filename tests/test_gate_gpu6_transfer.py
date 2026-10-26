"""The transfer must win the original scheduler's guard before owning a job."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from migrate_gate_extension_gpu6 import claim_job, select_jobs  # noqa: E402


@pytest.fixture
def plan():
    return {
        "jobs": [
            {"id": name, "family": "confirmation", "start_step": 0, "steps": 300000}
            for name in ("confirmation-2--fixed-update", "confirmation-3--paper")
        ]
    }


def test_selection_preserves_frozen_jobs_and_requested_order(plan):
    identifiers = [job["id"] for job in reversed(plan["jobs"])]
    selected = select_jobs(plan, identifiers)
    assert selected == list(reversed(plan["jobs"]))
    assert selected[0] is plan["jobs"][1]


@pytest.mark.parametrize("identifiers", [[], ["new-job"], ["confirmation-3--paper"] * 2])
def test_invalid_selection_is_rejected(plan, identifiers):
    with pytest.raises(ValueError):
        select_jobs(plan, identifiers)


@pytest.mark.parametrize("change", [{"family": "continuation"}, {"start_step": 300000}])
def test_continuation_cannot_be_claimed_as_fresh_training(plan, change):
    plan["jobs"][0].update(change)
    with pytest.raises(ValueError, match="Only existing 300k confirmation"):
        select_jobs(plan, [plan["jobs"][0]["id"]])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    job = {"id": "confirmation-1--paper", "case": "confirmation-1", "arm": "paper"}
    write(old / "controller-state.json", {"jobs": {job["id"]: {"state": "pending"}}})
    return old, new, job


def test_claim_blocks_original_executor_and_preserves_portable_output_links(setup):
    old, new, job = setup
    claim = Path(claim_job(old, new, job, write))
    with pytest.raises(FileExistsError):
        claim.mkdir(parents=True, exist_ok=False)  # The frozen original executor's guard.
    linked = old / "cases/confirmation-1/runs/paper"
    destination = new / "cases/confirmation-1/runs/paper"
    write(destination / "run.json", {"origin": "receiver"})
    assert json.loads((linked / "run.json").read_text()) == {"origin": "receiver"}
    assert not Path(linked.readlink()).is_absolute()
    with pytest.raises(FileExistsError):
        claim_job(old, new, job, write)


def test_running_job_is_never_claimed(setup):
    old, new, job = setup
    write(old / "controller-state.json", {"jobs": {job["id"]: {"state": "running"}}})
    with pytest.raises(ValueError, match="no longer pending"):
        claim_job(old, new, job, write)
    assert not (old / "cases").exists()


def test_race_lost_to_original_scheduler_never_overwrites_its_stage(setup):
    old, new, job = setup
    stage = old / "cases/confirmation-1/execution/paper/train"
    write(stage / "command.json", {"original": True})
    with pytest.raises(FileExistsError):
        claim_job(old, new, job, write)
    assert json.loads((stage / "command.json").read_text()) == {"original": True}
    assert not (stage / "handoff.json").exists()


def test_existing_scientific_run_is_preserved(setup):
    old, new, job = setup
    write(old / "cases/confirmation-1/runs/paper/run.json", {"original": True})
    with pytest.raises(FileExistsError, match="Preserve existing run"):
        claim_job(old, new, job, write)
    assert not (old / "cases/confirmation-1/execution").exists()
