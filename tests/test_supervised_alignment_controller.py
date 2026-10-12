"""Scheduling contracts protect the running predecessor and freeze the matrix."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "supervised_controller", SCRIPTS / "execute_supervised_alignment.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


@pytest.fixture
def predecessor(tmp_path):
    config_path = tmp_path / "source/configs/self-decode-alignment-development-v1.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"runtime": {"gpus": {"baseline": 0, "alignment": 2}}}))
    return {"baseline_batch": str(tmp_path)}


def test_no_container_gap_cannot_release_old_training_or_audit(predecessor):
    assert not controller.predecessor_released(0, predecessor)
    assert not controller.predecessor_released(2, predecessor)
    assert controller.predecessor_released(6, predecessor)


@pytest.mark.parametrize(
    "state,released",
    [("running", False), ("auditing", False), ("complete", True), ("failed", True)],
)
def test_only_terminal_old_task_releases_reservation(predecessor, state, released):
    marker = Path(predecessor["baseline_batch"]) / "task-state/baseline.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"state": state}))
    assert controller.predecessor_released(0, predecessor) is released
    assert not controller.predecessor_released(2, predecessor)


def test_cpuset_accounts_for_ranges():
    assert controller.cpuset("0-3,8,16-18") == {0, 1, 2, 3, 8, 16, 17, 18}
    assert controller.cpuset("") == set()


@pytest.fixture
def idle(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "BASE", tmp_path / "new-batch")
    cfg = {"runtime": {"gpu_preference": [6, 7]}}
    observed = {
        "gpus": {
            6: {"uuid": "u6", "memory": 0, "utilization": 0},
            7: {"uuid": "u7", "memory": 0, "utilization": 0},
        },
        "occupied_gpus": [],
        "occupied_cpus": [],
        "process_uuids": [],
        "memory_available_kib": 100 * 1024**2,
    }
    monkeypatch.setattr(controller, "config", lambda: cfg)
    monkeypatch.setattr(controller, "snapshot", lambda: observed)
    monkeypatch.setattr(controller, "predecessor_released", lambda gpu, cfg: True)
    monkeypatch.setattr(controller.os, "sched_getaffinity", lambda _: set(range(128)))
    return observed


@pytest.mark.parametrize(
    "reason", ["container", "memory", "utilization", "process", "cpus", "held", "old_audit"]
)
def test_busy_gpu_is_skipped_not_preempted(idle, monkeypatch, reason):
    held = set()
    if reason == "container":
        idle["occupied_gpus"] = [6]
    elif reason == "memory":
        idle["gpus"][6]["memory"] = 1025
    elif reason == "utilization":
        idle["gpus"][6]["utilization"] = 6
    elif reason == "process":
        idle["process_uuids"] = ["u6"]
    elif reason == "cpus":
        idle["occupied_cpus"] = [64]
    elif reason == "held":
        held = {6}
    else:
        monkeypatch.setattr(controller, "predecessor_released", lambda gpu, cfg: gpu == 7)
    slot = controller.acquire_slot(held)
    try:
        assert slot["gpu"] == 7
        assert slot["cpus"] == list(range(72, 80))
    finally:
        slot["lock"].close()


def test_cooperative_gpu_lock_blocks_second_allocation(idle):
    first = controller.acquire_slot(set())
    second = controller.acquire_slot(set())
    try:
        assert (first["gpu"], second["gpu"]) == (6, 7)
        assert controller.acquire_slot(set()) is None
    finally:
        first["lock"].close()
        second["lock"].close()


def test_memory_shortage_prevents_allocation(idle):
    idle["memory_available_kib"] = 47 * 1024**2
    assert controller.acquire_slot(set()) is None


def test_fixed_matrix_has_only_six_authorized_new_arms():
    cfg = json.loads(
        (SCRIPTS.parent / "configs/supervised-alignment-development-v1.json").read_text()
    )
    assert set(cfg["arms"]) == {
        "alignment",
        "same_kv_alignment",
        "label_ce",
        "same_kv_label_ce",
        "same_kv",
        "discoloop",
    }
    assert len(cfg["arms"]) * cfg["spec"]["steps"] == 1800000
    assert cfg["baseline_arm"] == "baseline"
    assert cfg["runtime"]["automatic_retry"] is False
    for arm in cfg["arms"].values():
        assert arm["scope"] == "supervised"
    for prefix in ("", "same_kv_"):
        align, ce = cfg["arms"][prefix + "alignment"], cfg["arms"][prefix + "label_ce"]
        assert {k: v for k, v in align.items() if k != "kind"} == {
            k: v for k, v in ce.items() if k != "kind"
        }


def test_smoke_audit_targets_smoke_output_without_changing_config():
    args = controller.arguments("audit", "alignment", smoke=True)
    assert args[args.index("--base") + 1] == str(controller.BASE / "runs")
    assert args[-1] == "--smoke"
    assert str(controller.BASE / "source" / controller.CONFIG) in args


def test_failed_baseline_audit_cannot_produce_comparison(tmp_path, monkeypatch):
    cfg = {"baseline_batch": str(tmp_path / "old"), "baseline_arm": "baseline"}
    marker = tmp_path / "old/runs/baseline/audit.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"passed":false}')
    monkeypatch.setattr(controller, "config", lambda: cfg)
    monkeypatch.setattr(controller, "BASE", tmp_path / "new")
    assert controller.factorial_summary() is False
    assert not (tmp_path / "new/factorial-summary.json").exists()
