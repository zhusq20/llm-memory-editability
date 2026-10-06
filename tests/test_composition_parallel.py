"""Scheduling recovery must preserve registered work and occupied GPU barriers."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "composition_parallel", ROOT / "scripts/execute_composition_parallel.py"
)
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_deferred_cards_require_terminal_state_and_all_audits(tmp_path):
    schedule = {
        "immediate_gpus": [0, 1, 2, 3, 4],
        "deferred_gpus": [
            {"results_root": str(tmp_path), "required_runs": ["a", "b"], "gpus": [6, 7]}
        ],
    }
    state = {"state": "running", "completed": ["a", "b"], "failed": [], "active": {}, "queued": []}
    for name in ["a", "b"]:
        write(tmp_path / "runs" / name / "audit.json", {"passed": True})
    write(tmp_path / "controller-state.json", state)
    assert parallel.eligible_gpus(schedule) == [0, 1, 2, 3, 4]
    state["state"] = "complete"
    write(tmp_path / "controller-state.json", state)
    assert parallel.eligible_gpus(schedule) == [0, 1, 2, 3, 4, 6, 7]
    write(tmp_path / "runs/b/audit.json", {"passed": False})
    assert parallel.eligible_gpus(schedule) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "change",
    [
        {"failed": ["b"]},
        {"active": {"6": {"name": "b"}}},
        {"queued": ["b"]},
        {"completed": ["a"]},
    ],
)
def test_free_memory_cannot_bypass_old_batch_barrier(tmp_path, change):
    state = {
        "state": "complete",
        "completed": ["a", "b"],
        "failed": [],
        "active": {},
        "queued": [],
        **change,
    }
    write(tmp_path / "controller-state.json", state)
    schedule = {
        "immediate_gpus": [0, 1],
        "deferred_gpus": [
            {"results_root": str(tmp_path), "required_runs": ["a", "b"], "gpus": [6, 7]}
        ],
    }
    assert parallel.eligible_gpus(schedule) == [0, 1]


def test_pending_queue_excludes_surviving_and_completed_workers():
    specs = [
        {"name": n, "entities": size}
        for n, size in [("done", 128), ("live", 256), ("small", 512), ("large", 4096)]
    ]
    assert parallel.pending_specs(specs, ["done"], ["live"], "support") == specs[2:]
    assert parallel.pending_specs(specs, ["done"], ["live"], "load") == specs[2:][::-1]
    with pytest.raises(ValueError, match="completed run"):
        parallel.pending_specs(specs, ["live"], ["live"], "support")
    with pytest.raises(ValueError, match="outside"):
        parallel.pending_specs(specs, [], ["unknown"], "support")


def test_recovery_adopts_live_container_without_stopping_it(monkeypatch):
    info = {
        "Config": {"Labels": {"batch": "curves", "run": "live"}},
        "Image": "fixed",
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["0"]}]},
    }
    monkeypatch.setattr(parallel.subprocess, "check_output", lambda *a, **kw: "container\n")
    monkeypatch.setattr(parallel, "read_container", lambda name: info)
    scheduler = parallel.ParallelScheduler(
        SimpleNamespace(start_tracking=None), {"immediate_gpus": [0, 1, 2], "deferred_gpus": []}
    )
    state = {"active": {"0": {"name": "live", "container": "container"}}}
    scheduler.recover({"batch": "curves", "runtime": {"image": "fixed"}}, state)
    assert state["active"] == {"0": {"name": "live", "container": "container"}}
    info["Image"] = "wrong"
    with pytest.raises(RuntimeError, match="identity"):
        scheduler.recover({"batch": "curves", "runtime": {"image": "fixed"}}, state)


def test_completed_support_revisit_preserves_active_load(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {"source_root": str(source), "results_root": str(tmp_path)}
    specs = [{"name": "support", "entities": 128}]
    materialized = {**config, "runs": specs, "phase": "support"}
    write(tmp_path / "support-config.json", materialized)
    scheduler = parallel.ParallelScheduler(SimpleNamespace(start_tracking=None), {})
    state = {"completed": ["support"], "active": {"2": {"name": "load", "container": "live"}}}
    scheduler.stage(config, specs, "support", state)
    assert state["active"]["2"]["container"] == "live"
    assert json.loads((tmp_path / "support-config.json").read_text()) == materialized
