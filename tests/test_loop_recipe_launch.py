"""Prevent ablation admission from displacing protected or foreign work."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch_loop_recipe_ablation as launch  # noqa: E402
from run_loop_recipe_pipeline import reload_matches  # noqa: E402


def test_independent_reload_requires_identical_panels_and_numerical_outputs():
    expected = {"atomic": {"n": 5000, "accuracy": 0.5, "nll": 0.1234567}}
    assert reload_matches(expected, copy.deepcopy(expected))
    nearby = copy.deepcopy(expected)
    nearby["atomic"]["nll"] += 1e-8
    assert reload_matches(expected, nearby)
    changed = copy.deepcopy(expected)
    changed["atomic"]["accuracy"] += 1 / 5000
    assert not reload_matches(expected, changed)
    assert not reload_matches(expected, {})


def fixture():
    containers = [
        {
            "id": f"protected-{i}",
            "name": f"/protected-{i}",
            "state": {"Status": "running", "Pid": 100 + i, "StartedAt": "fixed"},
            "requests": [{"DeviceIDs": [str(gpu)]}],
        }
        for i, gpu in enumerate([4, 6, 7, 8, 4, 8])
    ]
    snap = {
        "rows": {str(g): {"uuid": f"GPU-{g}", "free": 85000} for g in [4, 6, 7, 8]},
        "containers": containers,
        "process_details": [
            {"pid": 100 + i, "uuid": f"GPU-{g}"} for i, g in enumerate([4, 6, 7, 8, 4, 8])
        ],
    }
    auth = {
        "assignments": {"supervision": 6, "truncate": 7, "supervision-truncate": 4, "floor4": 8},
        "baseline_containers": {c["id"]: copy.deepcopy(c) for c in containers},
    }
    return snap, auth


def test_unchanged_six_and_our_new_process_are_allowed():
    baseline, auth = fixture()
    current = copy.deepcopy(baseline)
    launch.admit(current, baseline, auth)
    current["containers"].append({"id": "own", "state": {"Pid": 999}, "requests": []})
    current["process_details"].append({"pid": 999, "uuid": "GPU-6"})
    launch.admit(current, baseline, auth, {"own"})


@pytest.mark.parametrize(
    "conflict", ["memory", "foreign", "reservation", "restart", "exit", "wrong_gpu", "wrong_arms"]
)
def test_conflicts_fail_closed(conflict):
    baseline, auth = fixture()
    current = copy.deepcopy(baseline)
    if conflict == "memory":
        current["rows"]["4"]["free"] = 39999
    elif conflict == "foreign":
        current["process_details"].append({"pid": 999, "uuid": "GPU-7"})
    elif conflict == "reservation":
        current["containers"].append(
            {"id": "other", "name": "/other", "requests": [{"DeviceIDs": ["8"]}]}
        )
    elif conflict == "restart":
        current["containers"][0]["state"]["StartedAt"] = "restarted"
    elif conflict == "exit":
        current["containers"].pop()
    elif conflict == "wrong_gpu":
        auth["assignments"]["floor4"] = 9
    else:
        auth["assignments"]["old-held-init"] = auth["assignments"].pop("floor4")
    with pytest.raises((ValueError, RuntimeError)):
        launch.admit(current, baseline, auth)


def test_foreign_process_in_original_snapshot_is_not_an_exception():
    baseline, auth = fixture()
    baseline["process_details"].append({"pid": 999, "uuid": "GPU-7"})
    with pytest.raises(RuntimeError, match="foreign"):
        launch.admit(copy.deepcopy(baseline), baseline, auth)
