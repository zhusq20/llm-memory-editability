"""Admission must preserve running baselines and reject unrelated GPU occupants."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_loop_feedback_controls as launcher  # noqa: E402


def fixtures():
    containers = [
        {
            "id": f"baseline-{gpu}",
            "name": f"/baseline-{gpu}",
            "state": {"Status": "running", "Pid": 100 + gpu, "StartedAt": "unchanged"},
            "requests": [{"DeviceIDs": [str(gpu)]}],
        }
        for gpu in (4, 6, 7, 8)
    ]
    snap = {
        "rows": {str(gpu): {"uuid": f"GPU-{gpu}", "free": 90000} for gpu in (4, 6, 7, 8)},
        "containers": containers,
        "process_details": [{"uuid": f"GPU-{gpu}", "pid": 100 + gpu} for gpu in (4, 6, 7, 8)],
    }
    auth = {
        "assignments": {"infiloop-feedback": 4, "invariant-no-feedback": 8},
        "baseline_containers": {c["id"]: copy.deepcopy(c) for c in containers},
    }
    return snap, auth


def test_sharing_allows_only_unchanged_baselines_and_this_launch():
    baseline, auth = fixtures()
    current = copy.deepcopy(baseline)
    launcher.check_shared(current, baseline, auth)
    current["containers"].append(
        {
            "id": "own",
            "state": {"Pid": 999, "Status": "running"},
            "requests": [{"DeviceIDs": ["4"]}],
        }
    )
    current["process_details"].append({"uuid": "GPU-4", "pid": 999})
    launcher.check_shared(current, baseline, auth, {"own"})


@pytest.mark.parametrize("change", ["memory", "process", "reservation", "restart", "exit"])
def test_conflicting_changes_block_admission(change):
    baseline, auth = fixtures()
    current = copy.deepcopy(baseline)
    if change == "memory":
        current["rows"]["4"]["free"] = 39999
    elif change == "process":
        current["process_details"].append({"uuid": "GPU-8", "pid": 999})
    elif change == "reservation":
        current["containers"].append(
            {
                "id": "new",
                "name": "/new",
                "state": {"Status": "created", "Pid": 0},
                "requests": [{"DeviceIDs": ["4"]}],
            }
        )
    elif change == "restart":
        current["containers"][0]["state"]["Pid"] += 1
    else:
        current["containers"][1]["state"]["Status"] = "exited"
    with pytest.raises(RuntimeError):
        launcher.check_shared(current, baseline, auth)


def test_preexisting_unrelated_cuda_process_is_not_authorized_by_snapshot():
    baseline, auth = fixtures()
    baseline["process_details"].append({"uuid": "GPU-8", "pid": 999})
    with pytest.raises(RuntimeError, match="unrelated"):
        launcher.check_shared(copy.deepcopy(baseline), baseline, auth)
