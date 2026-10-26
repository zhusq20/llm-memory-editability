"""Prevent resource preemption, implicit launches and mutable source recovery."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ttcd_controller", ROOT / "scripts/launch_ttcd_second_hop.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_no_gpu_reassignment_or_preemption():
    runtime = {"gpu_memory_max_mib": 1024, "gpu_utilization_max": 5}
    row = {"uuid": "GPU-eight", "memory": 0, "utilization": 0}
    assert controller.check_gpu(8, runtime, ({8: row}, set())) == row
    for snapshot in (({8: {**row, "memory": 3000}}, set()), ({8: row}, {"GPU-eight"})):
        with pytest.raises(RuntimeError, match="occupied"):
            controller.check_gpu(8, runtime, snapshot)
    with pytest.raises(ValueError, match="does not exist"):
        controller.check_gpu(9, runtime, ({8: row}, set()))


def test_plan_does_not_touch_docker_or_gpu(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["launch_ttcd_second_hop.py", "calibrate"])

    def forbidden(*args, **kwargs):
        raise AssertionError("Plan must not invoke a subprocess")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    controller.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"]
    assert plan["jobs"] == [{"model": "discoloop", "gpu": 8}, {"model": "loop", "gpu": 9}]


def test_frozen_source_is_not_silently_refreshed(tmp_path, monkeypatch):
    project = tmp_path / "project"
    for name in ("src", "scripts", "configs"):
        (project / name).mkdir(parents=True, exist_ok=True)
    (project / "src/example.py").write_text("original = 1\n")
    for name in ("run_ttcd_second_hop.py", "launch_ttcd_second_hop.py"):
        (project / "scripts" / name).write_text("pass\n")
    (project / "configs/experiment-tracking-defaults.json").write_text("{}")
    inputs = project / "data"
    inputs.mkdir()
    for name in ("world.npz", "panels.npz", "complete.json", "training-split.json", "model.json"):
        (inputs / name).write_text("original fixture bytes\n")
    config = project / "configs/experiment.json"
    config.write_text(
        json.dumps(
            {
                "batch": "original",
                "scratch": {"version": 1},
                "data_dir": str(inputs),
                "model_config": str(inputs / "model.json"),
            }
        )
    )
    batch = project / "results/new"
    batch.mkdir(parents=True)
    monkeypatch.setattr(controller, "ROOT", project)
    frozen = controller.freeze_source(batch, config)
    assert frozen.is_file()
    (project / "src/example.py").write_text("changed = 2\n")
    (inputs / "world.npz").write_text("changed fixture bytes\n")
    controller.freeze_source(batch, config)
    assert (batch / "source/src/example.py").read_text() == "original = 1\n"
    assert (batch / "source/input-backup/world.npz").read_text() == "original fixture bytes\n"
    config.write_text('{"batch": "changed"}')
    with pytest.raises(ValueError, match="Configuration changed"):
        controller.freeze_source(batch, config)


def test_formal_outputs_cannot_silently_use_local_disk(tmp_path, monkeypatch):
    config = {"batch": "fixture", "runtime": {"persistent_root": str(tmp_path)}}
    monkeypatch.setattr(Path, "is_mount", lambda self: False)
    with pytest.raises(RuntimeError, match="persistent mount"):
        controller.persistent_batch(config)
    monkeypatch.setattr(Path, "is_mount", lambda self: True)
    with pytest.raises(ValueError, match="/ossfs/workspace"):
        controller.persistent_batch(config)
