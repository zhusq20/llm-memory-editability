"""Sharing authorization covers named frozen containers, never unknown jobs."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpu4_resources", ROOT / "scripts/ttcd_gpu4_resources.py"
)
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


class Busy(RuntimeError):
    pass


def fixture():
    row = {
        "uuid": "GPU-four",
        "total_mib": 97887,
        "free_mib": 80000,
        "memory": 17000,
        "utilization": 99,
    }
    container = {
        "Name": "/ttcd-successful-base-v1-single-train-v2",
        "Id": "own-single",
        "Mounts": [
            {
                "Destination": "/frozen",
                "Source": str(resources.COHORTS["ttcd-successful-base-v1-"][0]),
                "RW": False,
            }
        ],
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["GPU-four"]}]},
        "Config": {
            "Cmd": ["python", "/frozen/scripts/run_ttcd_verified_base.py", "train"],
            "User": "1015:1016",
        },
    }
    helper = SimpleNamespace(
        ResourceUnavailable=Busy,
        gpu_snapshot=lambda: (
            {4: row},
            [{"uuid": "GPU-four", "pid": 123}, {"uuid": "other", "pid": 999}],
        ),
        running_containers=lambda _: [container],
        container_uses_gpu=lambda *args: True,
        docker=lambda *args: SimpleNamespace(stdout="PID\n123\n124\n"),
        admit_cpu=lambda *args: ([], 10**9),
        utc_now=lambda: "fixed",
    )
    return helper, row, container


def test_known_frozen_cohort_can_share_busy_gpu4():
    helper, _, _ = fixture()
    runtime = {"cpu_sets": {"single": [24, 25, 26, 27]}}
    row = resources.admission(helper, runtime, "single")
    assert row["gpu"] == 4 and row["uuid"] == "GPU-four" and row["gpu_sharing"]
    assert row["utilization"] == 99
    assert row["cohort"][0]["host_pids"] == [123, 124]


@pytest.mark.parametrize(
    "change",
    [
        "foreign_name",
        "source",
        "writable_source",
        "script",
        "user",
        "all_gpus",
        "unknown_pid",
        "capacity",
        "memory",
    ],
)
def test_rejects_foreign_or_unverified_workload(change, monkeypatch):
    helper, row, container = fixture()
    monkeypatch.setattr(resources.time, "sleep", lambda _: None)
    if change == "foreign_name":
        container["Name"] = "/other-project"
    elif change == "source":
        container["Mounts"][0]["Source"] = "/tmp/unfrozen"
    elif change == "writable_source":
        container["Mounts"][0]["RW"] = True
    elif change == "script":
        container["Config"]["Cmd"] = ["python", "foreign.py"]
    elif change == "user":
        container["Config"]["User"] = "0"
    elif change == "all_gpus":
        container["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = []
    elif change == "unknown_pid":
        helper.docker = lambda *args: SimpleNamespace(stdout="PID\n456\n")
    elif change == "capacity":
        row["total_mib"] = 24000
    else:
        row["free_mib"] = 20000
    with pytest.raises(Busy):
        resources.admission(helper, {"cpu_sets": {"single": [24]}}, "single")


def test_both_parent_arms_are_recognized_without_sharing_with_foreign_gpu():
    helper, _, first = fixture()
    second = copy.deepcopy(first)
    second["Name"] = "/ttcd-successful-base-v1-stream-train-v2"
    second["Id"] = "own-stream"
    helper.running_containers = lambda _: [first, second]
    helper.docker = lambda *args: SimpleNamespace(
        stdout="PID\n123\n" if args[1] == "own-single" else "PID\n456\n"
    )
    helper.gpu_snapshot = lambda: (
        {
            4: {
                "uuid": "GPU-four",
                "total_mib": 97887,
                "free_mib": 70000,
                "memory": 27000,
                "utilization": 99,
            }
        },
        [
            {"uuid": "GPU-four", "pid": 123},
            {"uuid": "GPU-four", "pid": 456},
            {"uuid": "foreign-other-gpu", "pid": 999},
        ],
    )
    result = resources.verify_gpu4_cohort(helper)
    assert result["authorized_host_pids"] == [123, 456]


def test_transient_nvml_container_snapshot_is_rechecked(monkeypatch):
    helper, _, _ = fixture()
    calls = []

    def top(*args):
        calls.append(args)
        return SimpleNamespace(stdout="PID\n" if len(calls) == 1 else "PID\n123\n")

    helper.docker = top
    monkeypatch.setattr(resources.time, "sleep", lambda _: None)
    assert resources.verify_gpu4_cohort(helper)["authorized_host_pids"] == [123]
    assert len(calls) == 2
