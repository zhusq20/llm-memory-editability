"""Protect the scientific matrix and the two-resource scheduling contract."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dependency_launcher", ROOT / "scripts" / "launch_dependency_gate.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


def config():
    return json.loads((ROOT / launch.CONFIG).read_text())


def test_complete_paired_lanes():
    left, right = launch.lane_jobs(0), launch.lane_jobs(1)
    assert len(set(left + right)) == 8
    for (task_a, arm_a), (task_b, arm_b) in zip(left, right, strict=True):
        assert task_a == task_b
        assert (arm_a, arm_b) in (("loop", "only-u"), ("disco", "fixed-q-plus-u"))
    launch.validate(config())


@pytest.mark.parametrize("kind", ["steps", "gate", "budget", "context", "tracking"])
def test_reject_scientific_or_runtime_drift(kind):
    value = copy.deepcopy(config())
    if kind == "steps":
        value["tasks"]["listops"]["training"]["steps"] = 1500000
    elif kind == "gate":
        value["arms"]["only-u"]["injection"] = "fixed1"
    elif kind == "budget":
        value["budget"]["new_scientific_updates"] += 1
    else:
        value["runtime"]["docker_context" if kind == "context" else "tracking"] = "wrong"
    with pytest.raises(ValueError):
        launch.validate(value)


@pytest.mark.parametrize("lane", [0, 1])
def test_every_job_gets_cpu_pair_and_frozen_mounts(tmp_path, lane, monkeypatch):
    value = config()
    # Unit tests can run on hosts with an arbitrary CPU affinity range.
    value["runtime"].pop("cpu_first", None)
    monkeypatch.setattr(launch.common.os, "sched_getaffinity", lambda _: {0, 1, 2, 3})
    base = tmp_path / "frozen"
    base.mkdir()
    (base / "freeze.json").write_text("{}")
    owner = launch.Controller(
        value, base, tmp_path / "out", {2: "u2", 5: "u5"}, tmp_path / "private-token", "science"
    )
    for task, arm in launch.lane_jobs(lane):
        controller = launch.JobController(owner, task, arm, lane)
        _, argv, cpus = controller.command(
            arm, "train", 2 if lane == 0 else 5, "u2" if lane == 0 else "u5"
        )
        assert len(cpus) == 2
        assert argv[argv.index("--cpuset-cpus") + 1] == ",".join(map(str, cpus))
        assert "scripts/run_dependency_gate.py" in argv
        assert argv[argv.index("--stage") + 1] == "train"
        assert argv[argv.index("--task") + 1] == task
        assert f"type=bind,source={base / 'data'},target=/inputs,readonly" in argv
        assert argv[1:3] == ["--context", "lm-memory"]


def test_receipt_must_match_config(tmp_path):
    base, output = tmp_path / "base", tmp_path / "out"
    path = base / "source" / launch.CONFIG
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    target = output / "listops" / "preflight" / "loop.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"passed": True, "config_sha256": "wrong"}))
    with pytest.raises(ValueError):
        launch.verify_receipt(base, output, "listops", "loop")
    target.write_text(json.dumps({"passed": True, "config_sha256": launch.common.sha(path)}))
    launch.verify_receipt(base, output, "listops", "loop")


def test_preflight_pairing_rejects_backbone_drift(tmp_path):
    for task in launch.TASKS:
        for arm in launch.ARMS:
            path = tmp_path / task / "preflight" / f"{arm}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "initial_backbone_sha256": task,
                        "world_sha256": "data-" + task,
                        "panels_sha256": "panels-" + task,
                    }
                )
            )
    launch.verify_pairing(tmp_path)
    changed = tmp_path / "listops/preflight/only-u.json"
    receipt = json.loads(changed.read_text())
    receipt["initial_backbone_sha256"] = "different"
    changed.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="initial_backbone"):
        launch.verify_pairing(tmp_path)
