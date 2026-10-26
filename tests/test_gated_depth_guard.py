import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/guard_gated_depth_launch.py"
SPEC = importlib.util.spec_from_file_location("guard_gated_depth_launch", PATH)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_waits_if_another_process_claims_gpu_after_admission(tmp_path, monkeypatch):
    states = iter([({1: {"uuid": "gpu"}}, {1: ["other process"]}), ({1: {"uuid": "gpu"}}, {1: []})])
    receipts, sleeps = [], []
    module = SimpleNamespace(
        resource_snapshot=lambda: next(states),
        previous_queue_blockers=lambda rt: {},
        utc=lambda: "time",
        idle_candidates=lambda rt, rows, blocked: [1] if not blocked[1] else [],
        write=lambda path, value: receipts.append(value),
    )
    monkeypatch.setattr(guard.time, "sleep", lambda value: sleeps.append(value))
    guard.wait_for_gpu(
        module,
        {"runtime": {"poll_seconds": 30}},
        tmp_path,
        "job",
        "smoke",
        {"gpu": 1, "uuid": "gpu"},
    )
    assert sleeps == [30]
    assert [r["available"] for r in receipts] == [False, True]


def test_rechecks_before_create_and_start_but_not_existing_claim(tmp_path, monkeypatch):
    checks, calls = [], []
    module = SimpleNamespace(
        container_command=lambda *args: ("name", ["command"]),
        docker=lambda *args, **kwargs: calls.append(args),
    )
    monkeypatch.setattr(guard, "wait_for_gpu", lambda *args: checks.append(args))
    guard.install(module)
    row = {"gpu": 1}
    module.container_command({}, tmp_path, "job", "smoke", 0, row)
    assert len(checks) == 1
    module.docker("start", "name")
    assert len(checks) == 2 and calls == [("start", "name")]
    claim = tmp_path / "execution/job/audit/claim.json"
    claim.parent.mkdir(parents=True)
    claim.write_text("{}")
    module.container_command({}, tmp_path, "job", "audit", 0, row)
    assert len(checks) == 2


def test_changed_gpu_identity_is_rejected(tmp_path):
    module = SimpleNamespace(
        resource_snapshot=lambda: ({1: {"uuid": "new"}}, {}), previous_queue_blockers=lambda rt: {}
    )
    with pytest.raises(ValueError, match="identity"):
        guard.wait_for_gpu(
            module, {"runtime": {}}, tmp_path, "job", "stage", {"gpu": 1, "uuid": "old"}
        )
