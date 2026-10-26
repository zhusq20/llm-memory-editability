"""Prevent duplicate work and accidental adoption of unrelated training."""

import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "continuation_reschedule",
    Path(__file__).resolve().parents[1]
    / "scripts/reschedule_paper_update_continuation.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
REMOTE_SPEC = importlib.util.spec_from_file_location(
    "continuation_handoff_receiver",
    Path(__file__).resolve().parents[1] / "scripts/launch_continuation_handoff.py",
)
receiver = importlib.util.module_from_spec(REMOTE_SPEC)
REMOTE_SPEC.loader.exec_module(receiver)


def example():
    command = [
        "docker",
        "create",
        "--cpuset-cpus",
        "0,1",
        "--mount",
        "type=bind,source=/frozen,target=/project,readonly",
        "--mount",
        "type=bind,source=/output,target=/output",
        "sha256:frozen",
        "python",
        "train.py",
    ]
    launch = {"id": "original-id", "container": "original-name"}
    item = {
        "Id": "original-id",
        "Name": "/original-name",
        "Image": "sha256:frozen",
        "Config": {"Image": "sha256:frozen", "Cmd": ["python", "train.py"]},
        "HostConfig": {
            "CpusetCpus": "0,1",
            "DeviceRequests": [{"DeviceIDs": ["GPU-original"]}],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Destination": "/project",
                "Source": "/frozen",
                "RW": False,
            },
            {"Type": "bind", "Destination": "/output", "Source": "/output", "RW": True},
        ],
        "State": {"Running": True, "ExitCode": 0},
    }
    return item, launch, command


def check(item, launch, command):
    runner.validate_container(item, launch, command, "sha256:frozen", "GPU-original")


def test_handoff_removes_both_pending_arms_without_removing_active_arms():
    schedule = runner.lanes(None)
    assert set(schedule) == {1, 2}
    assert [arm for names in schedule.values() for arm in names] == [
        "token-update",
        "token-update-half-lr",
    ]


def test_additional_lane_preserves_exactly_four_unique_arms():
    schedule = runner.lanes(3)
    arms = [arm for names in schedule.values() for arm in names]
    assert len(arms) == len(set(arms)) == 4
    assert schedule[3] == ("paper",)
    assert schedule[2] == ("token-update-half-lr", "fixed-update")


@pytest.mark.parametrize("gpu", [-1, 1, 2])
def test_reject_conflicting_gpu(gpu):
    with pytest.raises(ValueError):
        runner.lanes(gpu)


def test_exact_running_container_adoptable():
    check(*example())


@pytest.mark.parametrize(
    "field",
    [
        "Id",
        "Name",
        "Image",
        "config_image",
        "command",
        "cpu",
        "gpu",
        "mount_source",
        "mount_write",
    ],
)
def test_reject_identity_or_execution_changes(field):
    item, launch, command = example()
    item = copy.deepcopy(item)
    if field in ("Id", "Name", "Image"):
        item[field] = "different"
    elif field == "config_image":
        item["Config"]["Image"] = "different"
    elif field == "command":
        item["Config"]["Cmd"] = ["python", "other.py"]
    elif field == "cpu":
        item["HostConfig"]["CpusetCpus"] = "4,5"
    elif field == "gpu":
        item["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU-other"]
    elif field == "mount_source":
        item["Mounts"][0]["Source"] = "/different"
    else:
        item["Mounts"][0]["RW"] = True
    with pytest.raises(ValueError):
        check(item, launch, command)


def test_failed_existing_training_must_not_restart():
    item, launch, command = example()
    item["State"] = {"Running": False, "ExitCode": 1}
    with pytest.raises(RuntimeError, match="never retried"):
        check(item, launch, command)


def test_completed_existing_training_can_finish_monitoring():
    item, launch, command = example()
    item["State"] = {"Running": False, "ExitCode": 0}
    check(item, launch, command)


@pytest.mark.parametrize(
    "field",
    ["ready", "old_scheduler_stopped", "arms", "freeze_sha256", "config_sha256"],
)
def test_receiver_rejects_missing_or_mismatched_handoff(field):
    receipt = {
        "ready": True,
        "old_scheduler_stopped": True,
        "arms": ["paper", "fixed-update"],
        "freeze_sha256": "frozen",
        "config_sha256": "config",
    }
    receiver.validate_receipt(receipt, "frozen", "config")
    receipt.pop(field)
    with pytest.raises(ValueError, match="No matching completed handoff"):
        receiver.validate_receipt(receipt, "frozen", "config")
