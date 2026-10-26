"""Migration must not add scientific jobs or use occupied GPUs."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transfer", ROOT / "scripts/launch_loop_transfer.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)


@pytest.fixture
def schedule():
    manifest = json.loads((ROOT / "results/loop-queue-transfer-20261021/tasks.json").read_text())
    seeds, early = [], []
    for task in manifest["training_jobs"]:
        (seeds if task["condition"] in ("init-1", "init-2") else early).append(task["job"])
    return {
        "phases": [early + [j["job"] for j in manifest["evaluation_only_jobs"]], seeds],
        "gpus": [5, 6, 8],
        "cpu_sets": ["70,71", "72,73", "74,75"],
    }, manifest


def test_whitelist_and_seeds_last(schedule):
    plan, manifest = schedule
    transfer.validate_schedule(plan, manifest)
    for mutation in ("duplicate", "omit", "seed_early", "gpu", "budget"):
        p, m = copy.deepcopy(plan), copy.deepcopy(manifest)
        if mutation == "duplicate":
            p["phases"][0].append(p["phases"][0][0])
        elif mutation == "omit":
            p["phases"][0].pop()
        elif mutation == "seed_early":
            p["phases"][0].append(p["phases"][1].pop())
        elif mutation == "gpu":
            p["gpus"] = [5, 6, 9]
        else:
            m["training_jobs"][0]["steps"] = 32
        with pytest.raises(ValueError):
            transfer.validate_schedule(p, m)


def test_gpu_process_and_active_container_cannot_be_ignored():
    state = {"Status": "created", "Pid": 0, "Running": False, "StartedAt": "0001-01-01T00:00:00Z"}
    snap = {
        "rows": {5: {"uuid": "GPU-test", "used": 0, "free": 90000, "utilization": 0}},
        "processes": [],
        "containers": [
            {
                "id": "old",
                "name": "old",
                "state": state,
                "restart": {"Name": "no"},
                "requests": [{"DeviceIDs": ["GPU-test"]}],
            }
        ],
    }
    assert not transfer.blockers(snap, 5, ["old"])
    assert transfer.blockers(snap, 5, [])
    snap["processes"].append("GPU-test")
    assert transfer.blockers(snap, 5, ["old"])
    snap["processes"].clear()
    state.update(Status="running", Pid=1, Running=True)
    assert transfer.blockers(snap, 5, ["old"])


def test_controller_refuses_existing_submission(tmp_path, schedule):
    plan, manifest = schedule
    plan["source_controller_locks"] = []
    (tmp_path / "schedule.json").write_text(json.dumps(plan))
    (tmp_path / "tasks.json").write_text(json.dumps(manifest))
    (tmp_path / "controller-process.json").write_text("{}")
    with pytest.raises(FileExistsError, match="Already submitted"):
        transfer.Controller(tmp_path, tmp_path / "unused-token").run()


def test_tracking_repair_rejects_scientific_changes(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from repair_loop_transfer_tracking import empty_dictionary_omissions

    expected = {"training": {"steps": 1500000}, "reused_comparators": {}}
    assert empty_dictionary_omissions(expected, expected) == []
    assert empty_dictionary_omissions(expected, {"training": {"steps": 1500000}}) == [
        "experiment.reused_comparators"
    ]
    for actual in ({"training": {"steps": 32}}, {}, {**expected, "extra": 1}):
        with pytest.raises(ValueError):
            empty_dictionary_omissions(expected, actual)
    sys.modules.pop("repair_loop_transfer_tracking", None)
