"""The user-designated GPU7 exception must not broaden other admissions."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def context(monkeypatch):
    original = script("launch_lr_control")
    resumed = script("resume_lr_control_gpu7")
    monkeypatch.setattr(original.os, "sched_getaffinity", lambda _: set(range(96)))
    config = json.loads((ROOT / "configs/loop-discoloop-lr-control-v1.json").read_text())
    snapshot = {
        "utc": "test",
        "available_kib": 900 * 1024**2,
        "gpus": {
            7: {
                "gpu": 7,
                "uuid": resumed.GPU7_UUID,
                "used_mib": 6393,
                "free_mib": 90858,
                "total_mib": 97887,
                "utilization": 59,
            }
        },
        "processes": [{"uuid": resumed.GPU7_UUID, "pid": 123}],
        "containers": [
            {
                "Name": next(iter(resumed.STALE_DEFAULT)),
                "inspection_context": "default",
                "HostConfig": {
                    "CpusetCpus": "",
                    "DeviceRequests": [{"DeviceIDs": [resumed.GPU7_UUID]}],
                },
            }
        ],
    }
    return original, resumed, config, snapshot


def test_designated_first_arm_uses_gpu7_and_records_real_occupancy(context):
    original, resumed, config, snapshot = context
    row = resumed.admit(original, config, resumed.FIRST_JOB, snapshot, 7)
    assert row["used_mib"] == 6393 and row["utilization"] == 59
    assert row["observed_existing_gpu_pids"] == snapshot["processes"]
    assert row["exclusive_idle_gpu"] is False
    assert row["existing_tasks_stopped"] is False
    assert snapshot["gpus"][7]["used_mib"] == 6393


def test_other_arms_still_wait_for_actual_idle_gpu(context):
    original, resumed, config, snapshot = context
    assert resumed.admit(original, config, "loop-lr5e-4", snapshot, 7) is None
    snapshot["processes"] = []
    snapshot["gpus"][7].update(used_mib=0, utilization=0)
    row = resumed.admit(original, config, "loop-lr5e-4", snapshot, 7)
    assert row is not None and row["exclusive_idle_gpu"]


def test_designated_arm_requires_reviewed_24gib_free_memory(context):
    original, resumed, config, snapshot = context
    snapshot["gpus"][7]["free_mib"] = resumed.GPU7_MIN_FREE_MIB - 1
    assert resumed.admit(original, config, resumed.FIRST_JOB, snapshot, 7) is None


def test_designated_sharing_preserves_existing_exclusive_lock(context, tmp_path):
    import fcntl

    _, resumed, _, _ = context
    with (tmp_path / "gpu-7.lock").open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = resumed.acquire_gpu_lock(tmp_path, 7, resumed.FIRST_JOB)
        assert first is not None
        assert resumed.acquire_gpu_lock(tmp_path, 7, "loop-lr5e-4") is None
        first.close()
        assert resumed.acquire_gpu_lock(tmp_path, 7, "loop-lr5e-4") is None
    other = resumed.acquire_gpu_lock(tmp_path, 7, "loop-lr5e-4")
    assert other is not None
    other.close()


def test_designation_does_not_allow_cpu_collision(context):
    original, resumed, config, snapshot = context
    snapshot["containers"][0]["HostConfig"]["CpusetCpus"] = "0-3"
    assert resumed.admit(original, config, resumed.FIRST_JOB, snapshot, 7) is None


def test_non7_gpu_admission_remains_unchanged(context):
    original, resumed, config, snapshot = context
    snapshot["gpus"][4] = {**snapshot["gpus"][7], "gpu": 4, "uuid": "GPU-other"}
    snapshot["processes"].append({"uuid": "GPU-other", "pid": 456})
    assert resumed.admit(original, config, resumed.FIRST_JOB, snapshot, 4) is None
