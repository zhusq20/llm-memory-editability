"""Admission, immutable execution and failure-stop contracts for the new single arm."""

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "state_launcher", ROOT / "scripts/launch_loop_state_handoff.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


@pytest.fixture
def config():
    return json.loads((ROOT / launch.CONFIG).read_text())


def empty_snapshot():
    return {
        "rows": {0: {"uuid": "GPU-zero", "used": 0, "free": 90000, "utilization": 0}},
        "processes": [],
        "containers": [],
    }


def test_active_cuda_process_blocks_even_when_memory_looks_idle(config):
    snap = empty_snapshot()
    snap["processes"] = ["GPU-zero"]
    assert "active CUDA process" in launch.available(snap, config["runtime"], 0)


def test_created_container_reservation_is_not_silently_overridden(config):
    snap = empty_snapshot()
    snap["containers"] = [
        {
            "id": "another-owner",
            "name": "/reserved-job",
            "state": {"Status": "created"},
            "restart": {"Name": "no"},
            "requests": [{"DeviceIDs": ["GPU-zero"]}],
        }
    ]
    assert "/reserved-job" in launch.available(snap, config["runtime"], 0)


def test_priority_queue_keeps_its_gpus_until_every_job_completes(config, tmp_path):
    runtime = copy.deepcopy(config["runtime"])
    path = tmp_path / "old-state.json"
    runtime["priority_controller_state"] = str(path)
    path.write_text(json.dumps({"state": "running", "jobs": {"old": {"state": "pending"}}}))
    assert launch.priority_reasons(runtime, 5)
    assert launch.priority_reasons(runtime, 0) == []
    path.write_text(json.dumps({"state": "complete", "jobs": {"old": {"state": "failed"}}}))
    assert launch.priority_reasons(runtime, 5)
    path.write_text(json.dumps({"state": "complete", "jobs": {"old": {"state": "complete"}}}))
    assert launch.priority_reasons(runtime, 5) == []
    path.unlink()
    assert launch.priority_reasons(runtime, 5)


def test_new_arm_cannot_expand_into_old_arms_or_change_budget(config):
    launch.validate(config)
    for key, value in (("steps", 10), ("steps", 1500001)):
        changed = copy.deepcopy(config)
        changed["training"][key] = value
        with pytest.raises(ValueError):
            launch.validate(changed)
    changed = copy.deepcopy(config)
    changed["arms"]["invariant"] = {"handoff": "invariant"}
    with pytest.raises(ValueError):
        launch.validate(changed)


def test_container_uses_frozen_readonly_source_and_isolated_output(config, tmp_path):
    label, argv = launch.stage_command(config, tmp_path, "train", 0, "GPU-zero")
    assert label == "lsh-1022-state-mlp-train"
    assert argv[:3] == ["docker", "--context", "lm-memory"]
    assert "device=GPU-zero" in argv
    assert f"type=bind,source={tmp_path / 'source'},target=/project,readonly" in argv
    assert f"type=bind,source={tmp_path / 'artifacts'},target=/output" in argv
    assert "run_loop_native_handoff.py" not in argv
    assert argv[-1] == "state-mlp"
    _, cloud = launch.stage_command(config, tmp_path, "cloud", 0, "GPU-zero")
    assert "--gpus" not in cloud


def test_changed_frozen_file_is_rejected(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    launch.write(
        tmp_path / "freeze.json",
        {
            "files": {"source.py": launch.digest(source)},
            "inputs": {},
        },
    )
    launch.verify_freeze(tmp_path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        launch.verify_freeze(tmp_path)


def test_failure_stops_without_retry_and_duplicate_submission_refuses(
    config, tmp_path, monkeypatch
):
    launch.write(tmp_path / "source" / launch.CONFIG, config)
    launch.write(tmp_path / "freeze.json", {"files": {}, "inputs": {}})
    launch.write(
        tmp_path / "engineering/cpu-checks.json",
        {
            "passed": True,
            "freeze_sha256": launch.digest(tmp_path / "freeze.json"),
            "image": config["runtime"]["image"],
        },
    )
    monkeypatch.setattr(launch, "verify_freeze", lambda _: {})
    monkeypatch.setattr(launch, "credential_archive", lambda _: b"not-a-real-secret")
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=config["runtime"]["image"] + "\n"),
    )
    controller = launch.Controller(tmp_path, tmp_path / "secret")
    monkeypatch.setattr(controller, "admit", lambda gpu=None: (0, "GPU-zero"))
    executed = []

    def execute(stage, *_):
        executed.append(stage)
        if stage == "smoke":
            raise RuntimeError("engineering failure")

    monkeypatch.setattr(controller, "execute", execute)
    with pytest.raises(RuntimeError, match="engineering failure"):
        controller.run()
    assert executed == ["environment", "smoke"]
    assert json.loads((tmp_path / "controller-state.json").read_text())["state"] == "failed"
    with pytest.raises(FileExistsError, match="Already submitted"):
        controller.run()
    assert executed == ["environment", "smoke"]
