"""Successful-base adaptation gates and freezes must precede paired updates."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "discoloop_paired_controller", ROOT / "scripts/launch_discoloop_paired.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def configuration():
    return json.loads((ROOT / "configs/ttcd-successful-base-v1.json").read_text())


def test_dryrun_is_readonly_and_shows_two_stage_gating(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration()))
    monkeypatch.setattr(sys, "argv", ["launch_discoloop_paired.py", "--config", str(path)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Planning cannot touch Docker, GPU or persistent output")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    monkeypatch.setattr(controller, "persistent_batch", forbidden)
    controller.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"] and not plan["automatic_retry"] and not plan["gpu_sharing"]
    assert plan["foundation_gate_required"] and plan["common_gate_required_before_paired"]
    assert plan["prepare_device"] == "cpu"
    assert plan["common_gpu_preference"] == [8, 9]
    assert plan["paired_gpus"] == {"single": 8, "stream": 9}
    assert plan["minimum_persistent_free_gib"] >= 200


@pytest.mark.parametrize(
    "change", ["budget", "phi", "gpu", "sharing", "cpu", "disk", "parent", "panel"]
)
def test_rejects_changes_outside_frozen_comparison(change):
    config = configuration()
    if change == "budget":
        config["common"]["steps"] = 20000
    elif change == "phi":
        config["phi"] = 2.0
    elif change == "gpu":
        config["runtime"]["gpus"]["stream"] = 8
    elif change == "sharing":
        config["runtime"]["shared_gpu_allowlist"] = ["other-task"]
    elif change == "cpu":
        config["runtime"]["cpu_sets"]["stream"][0] = config["runtime"]["cpu_sets"]["single"][0]
    elif change == "disk":
        config["runtime"]["minimum_persistent_free_gib"] = 100
    elif change == "parent":
        config["foundation"]["checkpoint"] = "results/old-unverified/checkpoint.pt"
    else:
        config["continual"]["panel_source_path"] = "configs/live-panel.json"
    with pytest.raises(ValueError):
        controller.validate_config(config)


def make_foundation(tmp_path, config, passed=True):
    batch = tmp_path / controller.BATCH
    checkpoint = controller.result_host_path(batch, config["foundation"]["checkpoint"])
    gate = controller.result_host_path(batch, config["foundation"]["gate_receipt"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified foundation endpoint")
    gate.write_text(
        json.dumps({"passed": passed, "checkpoint_sha256": controller.sha256(checkpoint)})
    )
    return batch, checkpoint, gate


def test_foundation_gate_binds_actual_checkpoint_before_any_adaptation(tmp_path):
    config = configuration()
    batch, checkpoint, gate = make_foundation(tmp_path, config)
    assert controller.verify_foundation(batch, config)["passed"]
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="another checkpoint"):
        controller.verify_foundation(batch, config)
    gate.write_text('{"passed": false}')
    with pytest.raises(RuntimeError, match="no common reader or write training"):
        controller.verify_foundation(batch, config)


def make_project(tmp_path, config):
    project = tmp_path / "project"
    for directory in ("src", "scripts", "configs", "tests"):
        (project / directory).mkdir(parents=True)
    (project / "src/example.py").write_text("original = True\n")
    (project / "tests/test_example.py").write_text("assert True\n")
    for name in controller.SOURCE_SCRIPTS:
        (project / "scripts" / name).write_text("pass\n")
    for name in ("pyproject.toml", "CONTRIBUTING.md"):
        (project / name).write_text("original metadata\n")
    (project / "configs/experiment-tracking-defaults.json").write_text("{}")
    (project / "configs/ttcd-second-hop-scratch-v1.json").write_text('{"fixture_only": true}')
    for relative in controller.PROTOCOLS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen330k design and two gates\n")
    for relative in controller.raw_inputs(config):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"original {relative}".encode())
    config_path = project / "configs/experiment.json"
    config_path.write_text(json.dumps(config))
    return project, config_path


def test_prepare_freezes_source_then_adds_only_generated_panel_and_data(tmp_path, monkeypatch):
    config = configuration()
    project, config_path = make_project(tmp_path, config)
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    monkeypatch.setattr(controller, "ROOT", project)
    foundation = {"passed": True, "checkpoint_sha256": "a" * 64}
    controller.freeze_preparation(batch, config_path, foundation)
    original_manifest = (batch / "preparation-source-manifest.json").read_bytes()
    controller.setup_raw_overlay(batch, config)
    for relative in controller.prepared_inputs(config):
        path = batch / "input-mount" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"prepared {relative}".encode())
    panel = controller.result_host_path(batch, config["continual"]["panel_source_path"])
    assert panel.parent.is_dir()
    assert panel.parent.stat().st_mode & 0o777 == 0o755
    panel.write_text('{"graph_only_panel": true}\n')
    controller.freeze_prepared_inputs(batch, config)
    assert (batch / "preparation-source-manifest.json").read_bytes() == original_manifest
    assert (batch / "source/configs/continual-panel.json").read_bytes() == panel.read_bytes()
    assert (batch / "source/tests/test_example.py").is_file()
    assert (batch / "source/configs/ttcd-second-hop-scratch-v1.json").is_file()
    for relative in controller.PROTOCOLS:
        assert (batch / "source" / relative).is_file()
    controller.verify_input_overlay(batch, config)
    (project / "src/example.py").write_text("changed = True\n")
    controller.freeze_preparation(batch, config_path, foundation)
    assert (batch / "source/src/example.py").read_text() == "original = True\n"
    prepared = batch / "input-mount" / config["model_config"]
    prepared.write_bytes(b"changed prepared config")
    with pytest.raises(ValueError, match="Prepared input overlay changed"):
        controller.verify_input_overlay(batch, config)


def test_prepare_mounts_cpu_only_writable_new_data_and_readonly_original(tmp_path):
    config = configuration()
    batch = tmp_path / controller.BATCH
    name, command = controller.build_command(config, "common", "prepare", batch, tmp_path / "token")
    assert name.endswith("preparation-prepare")
    assert "--gpus" not in command
    assert command[command.index("--device") + 1] == "cpu"
    assert f"type=bind,source={batch / 'input-mount/data'},target=/project/data" in command
    original = Path(config["original_data_dir"]).parent
    assert (
        f"type=bind,source={batch / 'input-mount' / original},target=/project/{original},readonly"
    ) in command
    assert "/frozen/scripts/run_ttcd_verified_base.py" in command


def test_arm_mounts_own_outputs_rw_and_foundation_common_other_arm_readonly(tmp_path):
    config = configuration()
    batch = tmp_path / controller.BATCH
    _, command = controller.build_command(
        config, "stream", "train", batch, tmp_path / "token", {"uuid": "GPU-nine"}
    )
    assert f"type=bind,source={tmp_path},target=/project/results,readonly" in command
    assert (
        f"type=bind,source={batch},target=/project/results/{controller.BATCH},readonly" in command
    )
    assert (
        f"type=bind,source={batch / 'stream'},target=/project/results/{controller.BATCH}/stream"
    ) in command
    assert f"type=bind,source={batch / 'input-mount/data'},target=/project/data,readonly" in command
    assert "device=GPU-nine" in command


def test_common_gate_failure_creates_no_paired_scientific_updates(tmp_path, monkeypatch):
    config = configuration()
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    called = []
    monkeypatch.setattr(controller, "admit_cpu", lambda *args: ([], 1024**3))
    monkeypatch.setattr(controller, "run_stage", lambda *args, **kwargs: 0)
    monkeypatch.setattr(controller, "freeze_prepared_inputs", lambda *args: None)

    def common(*args):
        called.append(args[2])
        return 3

    monkeypatch.setattr(controller, "run_phase", common)
    assert controller.execute_pipeline(batch, config, tmp_path / "token") == 3
    assert called == ["common"]
    state = json.loads((batch / "controller-state.json").read_text())
    assert state["paired_scientific_updates"] == 0
    assert not (batch / "single").exists() and not (batch / "stream").exists()


def test_common_gate_cannot_be_bypassed_by_standalone_arm_entry(tmp_path, monkeypatch):
    config = configuration()
    batch = tmp_path / controller.BATCH
    batch.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("No GPU admission before a passing common gate")

    monkeypatch.setattr(controller, "wait_gpu", forbidden)
    assert controller.run_phase(batch, config, "single", ["train"], tmp_path / "token") == 1
    assert not (batch / "single/claim-train.json").exists()


def test_common_uses_any_idle_reviewed_gpu_but_stream_never_switches_gpu(tmp_path, monkeypatch):
    config = configuration()
    attempts = []

    def admit(batch, config, phase, gpu):
        attempts.append(gpu)
        if gpu == 8:
            raise controller.ResourceUnavailable("GPU8 occupied")
        return {"gpu": gpu, "uuid": "GPU-nine"}

    monkeypatch.setattr(controller, "admit_gpu", admit)
    monkeypatch.setattr(controller.fcntl, "flock", lambda *args: None)
    held, row = controller.wait_gpu(tmp_path, config, "common", float("inf"))
    held.close()
    assert row["gpu"] == 9 and attempts == [8, 9]
    attempts.clear()
    held, row = controller.wait_gpu(tmp_path, config, "stream", float("inf"))
    held.close()
    assert attempts == [9]


def test_space_floor_is_checked_before_starting_persistent_state_work(tmp_path, monkeypatch):
    config = configuration()
    monkeypatch.setattr(
        controller.shutil, "disk_usage", lambda _: SimpleNamespace(free=199 * 1024**3)
    )
    with pytest.raises(RuntimeError, match="at least200GiB"):
        controller.check_space(tmp_path, config)


def test_failed_arm_does_not_cancel_other_parallel_arm(tmp_path, monkeypatch):
    config = configuration()
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    calls = []
    monkeypatch.setattr(controller, "admit_cpu", lambda *args: ([], 1024**3))
    monkeypatch.setattr(controller, "run_stage", lambda *args, **kwargs: 0)
    monkeypatch.setattr(controller, "freeze_prepared_inputs", lambda *args: None)
    monkeypatch.setattr(controller, "require_common_gate", lambda *args: None)

    def phase(batch, config, name, stages, token):
        calls.append(name)
        return 3 if name == "single" else 0

    monkeypatch.setattr(controller, "run_phase", phase)
    assert controller.execute_pipeline(batch, config, tmp_path / "token") == 1
    assert set(calls) == {"common", "single", "stream"}
    state = json.loads((batch / "controller-state.json").read_text())
    assert state["exit_codes"] == {"single": 3, "stream": 0}
