"""Resource reassignment must retain the frozen science and exclude busy GPUs."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ttcd_resource_reschedule", ROOT / "scripts/reschedule_ttcd_verified.py"
)
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "ttcd_resource_test_repair", ROOT / "scripts/repair_ttcd_verified_preflight.py"
)
repair = importlib.util.module_from_spec(REPAIR_SPEC)
REPAIR_SPEC.loader.exec_module(repair)


def config():
    return json.loads((ROOT / "configs/ttcd-successful-base-v1.json").read_text())


def row():
    return {
        "uuid": "GPU-idle",
        "memory": 0,
        "free_mib": 97251,
        "total_mib": 97887,
        "utilization": 0,
    }


@pytest.mark.parametrize("reason", ["process", "memory", "utilization", "capacity", "container"])
def test_excludes_occupied_or_insufficient_gpu(reason, monkeypatch):
    value, processes, containers = row(), [], []
    if reason == "process":
        processes = [{"uuid": value["uuid"], "pid": 123}]
    elif reason == "memory":
        value["memory"] = 2000
    elif reason == "utilization":
        value["utilization"] = 10
    elif reason == "capacity":
        value["total_mib"] = 24000
    else:
        containers = [{}]
        monkeypatch.setattr(repair.controller, "container_uses_gpu", lambda *args: True)
    with pytest.raises(repair.controller.ResourceUnavailable):
        schedule.idle_row(
            repair.controller, config(), "common", 4, {4: value}, processes, containers, 10**9
        )


def test_actual_uuid_changes_device_without_mutating_scientific_config(tmp_path, monkeypatch):
    settings, before = config(), config()
    test_repair = SimpleNamespace(controller=SimpleNamespace(), build_command=repair.build_command)
    schedule.install_resource_policy(test_repair, tmp_path / "schedule.json", tmp_path / "locks")
    admission = {**row(), "gpu": 4}
    name, command = test_repair.build_command(
        settings, "common", "train", tmp_path, tmp_path / "no-token", admission
    )
    assert command[command.index("--gpus") + 1] == "device=GPU-idle"
    assert "LM_ACTUAL_GPU_INDEX=4" in command
    assert command[command.index("--config") + 1] == "/frozen/configs/experiment.json"
    assert any("source-v2,target=/frozen,readonly" in item for item in command)
    assert settings == before
    assert name.endswith("-v2")


def test_paired_threads_cannot_acquire_same_gpu(tmp_path, monkeypatch):
    controller = SimpleNamespace(
        ResourceUnavailable=repair.controller.ResourceUnavailable,
        admit_cpu=lambda *args: ([], 10**9),
        gpu_snapshot=lambda: ({4: row(), 7: {**row(), "uuid": "GPU-other"}}, []),
        utc_now=lambda: "fixed",
        container_uses_gpu=lambda *args: False,
    )
    test_repair = SimpleNamespace(controller=controller, build_command=lambda *args: None)
    schedule.install_resource_policy(test_repair, tmp_path / "schedule.json", tmp_path / "locks")
    deadline = time.monotonic() + 2
    first, a = controller.wait_gpu(tmp_path, config(), "single", deadline)
    try:
        # Prefer the held GPU first to exercise flock exclusion, then fall back.
        monkeypatch.setitem(schedule.PREFERENCES, "stream", [4, 7])
        second, b = controller.wait_gpu(tmp_path, config(), "stream", deadline)
        with second:
            assert (a["gpu"], b["gpu"]) == (4, 7)
    finally:
        first.close()


def test_common_gate_and_scientific_pipeline_are_not_replaced(tmp_path):
    original_phase = repair.run_phase
    original_pipeline = repair.execute_pipeline
    test_repair = SimpleNamespace(
        controller=SimpleNamespace(),
        build_command=repair.build_command,
        run_phase=original_phase,
        execute_pipeline=original_pipeline,
    )
    schedule.install_resource_policy(test_repair, tmp_path / "schedule.json")
    assert test_repair.run_phase is original_phase
    assert test_repair.execute_pipeline is original_pipeline


def test_default_plan_does_not_access_experiment_or_credentials(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/reschedule_ttcd_verified.py"),
            "--workspace",
            str(tmp_path / "absent"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(result.stdout)
    assert plan["executed"] is False
    assert plan["common_gate_before_paired_arms"] is True
    assert list(tmp_path.iterdir()) == []


def test_refuses_live_waiting_controller(tmp_path, monkeypatch):
    directory = tmp_path / schedule.AMENDMENT
    directory.mkdir()
    (directory / "stop-waiting-controller.json").write_text(
        json.dumps(
            {
                "scientific_updates": 0,
                "previous_state": {"state": "waiting_for_resources"},
                "previous_pid": 123,
            }
        )
    )
    test_repair = SimpleNamespace(
        validate_zero_updates=lambda *args: None,
        verify_repaired_source=lambda *args: None,
        controller=repair.controller,
    )
    monkeypatch.setattr(schedule.os, "kill", lambda *args: None)
    with pytest.raises(ValueError, match="still alive"):
        schedule.validate_reschedule(test_repair, tmp_path, config(), None, None)
