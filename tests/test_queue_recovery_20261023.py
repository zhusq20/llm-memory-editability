import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lm_resource_snapshot as R
from recover_knowledge_queue_20261023 import choose_waiting, validate_adopted


def test_recovery_bundle_rejects_mutated_code(tmp_path):
    code = b"frozen"
    (tmp_path / "entry.py").write_bytes(code)
    (tmp_path / "code-manifest.json").write_text(
        json.dumps({"entry.py": hashlib.sha256(code).hexdigest()})
    )
    R.verify_code_bundle(tmp_path)
    (tmp_path / "entry.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="code changed"):
        R.verify_code_bundle(tmp_path)


def test_removed_container_is_skipped_only_after_fresh_active_list(monkeypatch):
    calls = iter([["gone"], []])
    monkeypatch.setattr(R, "active_ids", lambda _: next(calls))

    def fail(_):
        raise R.ResourceUnavailable("inspect failed")

    monkeypatch.setattr(R, "call", fail)
    assert R.containers("default") == []


def test_unknown_docker_failure_blocks_admission(monkeypatch):
    monkeypatch.setattr(R, "active_ids", lambda _: ["still-there"])

    def fail(_):
        raise R.ResourceUnavailable("inspect failed")

    monkeypatch.setattr(R, "call", fail)
    with pytest.raises(R.ResourceUnavailable):
        R.containers("default")


def test_released_reservation_reblocks_if_started_or_restarting():
    c = dict(
        id="released",
        state=dict(Status="created", Pid=0, Running=False, StartedAt="0001-01-01T00:00:00Z"),
        restart={"Name": "no"},
    )
    assert R.released_created(c, ["released"])
    assert not R.released_created(c, [])
    for key, value in [
        ("Pid", 123),
        ("Running", True),
        ("Status", "running"),
        ("StartedAt", "2026-10-23"),
    ]:
        assert not R.released_created({**c, "state": {**c["state"], key: value}}, ["released"])
    assert not R.released_created({**c, "restart": {"Name": "always"}}, ["released"])


def test_exclusive_evaluation_does_not_block_pending_training():
    jobs = {
        "done": {"state": "waiting_resource", "stage": "audit"},
        "new": {"state": "waiting_resource", "stage": "train"},
        "live": {"state": "running", "stage": "train"},
    }
    assert choose_waiting(jobs, list(jobs), ["live"]) == ["new"]
    assert choose_waiting(jobs, list(jobs), []) == ["new", "done"]


def test_adoption_rejects_replacement_ambiguous_and_oom():
    job = {"container_id": "same"}
    validate_adopted(job, {"id": "same", "Status": "running", "OOMKilled": False})
    validate_adopted(job, {"id": "same", "Status": "exited", "OOMKilled": False})
    for patch in [{"id": "different"}, {"Status": "created"}, {"OOMKilled": True}]:
        with pytest.raises(ValueError):
            validate_adopted(job, {"id": "same", "Status": "running", "OOMKilled": False, **patch})


def test_unknown_external_pid_and_reservation_are_not_discarded():
    snap = dict(
        rows={4: {"uuid": "GPU-four"}},
        process_details=[{"uuid": "GPU-four", "pid": 99}],
        containers=[
            {
                "id": "foreign",
                "context": "default",
                "name": "/reserved",
                "requests": [],
                "state": {
                    "Status": "created",
                    "Pid": 0,
                    "Running": False,
                    "StartedAt": "0001-01-01T00:00:00Z",
                },
                "restart": {"Name": "no"},
            }
        ],
    )
    _, blocked = R.knowledge_view(snap, lambda requests, rows: [4])
    assert blocked[4] == ["active CUDA process 99", "default:/reserved"]
