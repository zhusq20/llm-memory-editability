"""The branch controller must preserve parent jobs and fail closed on sharing."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ttcd_branch_controller", ROOT / "scripts/launch_ttcd_branch.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def configuration():
    batch = "ttcd-second-hop-reader-control-v1"
    parent = "ttcd-second-hop-scratch-v1"
    return {
        "batch": batch,
        "branch": {
            "parent_batch": parent,
            "parent_source_dir": "/parent-frozen",
            "parent_manifest_path": "/parent-source-manifest.json",
            "parent_manifest_sha256": "manifest-digest",
            "parent_config_sha256": "config-digest",
            "parent_image_sha256": "sha256:original-image",
            "checkpoint_step": 20000,
            "steps": 10000,
            "objective": "old_atoms_old_chains_context_reader_only",
            "checkpoints": {
                name: f"results/{parent}/{name}/training/checkpoint-0020000.pt"
                for name in controller.PARENTS
            },
        },
        "models": {
            name: {"checkpoint": f"results/{batch}/{name}/training/parent.pt"}
            for name in controller.PARENTS
        },
        "data_dir": "data/ttcd-second-hop-scratch-v1",
        "model_config": "data/ttcd-second-hop-scratch-v1/model.json",
        "runtime": {
            "gpus": {"discoloop": 8, "loop": 9},
            "cpu_sets": {"discoloop": [8, 9, 10, 11], "loop": [12, 13, 14, 15]},
            "cpu_threads": 4,
            "shared_gpu_allowlist": {
                name: [parent_name] for name, parent_name in controller.PARENTS.items()
            },
            "shared_gpu_min_free_mib": 8192,
            "memory_gib": 48,
            "persistent_root": "/ossfs/workspace/llm-memory-editability",
            "timeout_hours": 12,
            "tracking": "online",
        },
    }


def test_only_explicit_parent_cuda_pids_are_admitted():
    runtime = configuration()["runtime"]
    row = {"uuid": "GPU-eight", "free_mib": 20000, "total_mib": 24000}
    known = {"uuid": "GPU-eight", "pid": 101}
    other_gpu = {"uuid": "GPU-nine", "pid": 900}
    assert controller.check_shared_gpu(8, runtime, ({8: row}, [known, other_gpu]), {101}) == row
    unrelated = {"uuid": "GPU-eight", "pid": 102}
    with pytest.raises(RuntimeError, match="unrelated CUDA processes"):
        controller.check_shared_gpu(8, runtime, ({8: row}, [known, unrelated]), {101})
    with pytest.raises(RuntimeError, match="reserved free memory"):
        controller.check_shared_gpu(8, runtime, ({8: {**row, "free_mib": 1000}}, []), set())
    with pytest.raises(ValueError, match="does not exist"):
        controller.check_shared_gpu(9, runtime, ({8: row}, []), set())


def test_cpu_isolation_covers_running_parents_and_other_jobs():
    running = {
        "Name": "/parent",
        "State": {"Running": True},
        "HostConfig": {"CpusetCpus": "0-3,6"},
    }
    controller.check_cpu_sets([8, 9, 10, 11], [running], range(16))
    with pytest.raises(RuntimeError, match="collides"):
        controller.check_cpu_sets([3, 4, 5, 7], [running], range(16))
    with pytest.raises(RuntimeError, match="available affinity"):
        controller.check_cpu_sets([16], [running], range(16))
    with pytest.raises(RuntimeError, match="Cannot verify CPU isolation"):
        controller.check_cpu_sets([8], [{**running, "HostConfig": {"CpusetCpus": ""}}], range(16))


def test_waiter_ignores_partial_and_stops_on_parent_failure(tmp_path):
    checkpoint = tmp_path / "checkpoint-0020000.pt"
    checkpoint.with_suffix(".partial").write_bytes(b"incomplete")
    with pytest.raises(RuntimeError, match="Parent stopped"):
        controller.wait_for_checkpoint(
            checkpoint, lambda: {"State": {"Running": False}}, float("inf")
        )
    assert not checkpoint.exists()


def test_waiter_waits_for_final_atomic_publication(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-0020000.pt"
    partial = checkpoint.with_suffix(".partial")
    partial.write_bytes(b"final checkpoint bytes")
    states = []
    monkeypatch.setattr(controller.time, "sleep", lambda _: partial.replace(checkpoint))
    receipt = controller.wait_for_checkpoint(
        checkpoint, lambda: {"State": {"Running": True}}, float("inf"), states.append
    )
    assert receipt["sha256"] == controller.sha256(checkpoint)
    assert receipt["bytes"] == len(b"final checkpoint bytes")
    assert states[0]["state"] == "waiting_for_parent_20000"
    # A parent which has subsequently finished does not invalidate its 20k bytes.
    assert controller.wait_for_checkpoint(checkpoint, lambda: None, 0) == receipt


def test_waiter_hard_timeout_does_not_touch_parent(tmp_path):
    def forbidden():
        raise AssertionError("A timeout must not inspect or modify the parent")

    with pytest.raises(TimeoutError, match="Hard timeout"):
        controller.wait_for_checkpoint(tmp_path / "checkpoint.pt", forbidden, 0)


def test_default_plan_never_invokes_docker_or_gpu(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration()))
    monkeypatch.setattr(sys, "argv", ["launch_ttcd_branch.py", "--config", str(config_path)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only planning cannot run a subprocess or freeze inputs")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    monkeypatch.setattr(controller, "freeze_source", forbidden)
    controller.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"]
    assert [job["gpu"] for job in plan["jobs"]] == [8, 9]
    assert plan["stages"] == list(controller.STAGES)
    assert plan["docker_context"] == "lm-memory"


def test_stage_claims_and_existing_training_prevent_resubmission(tmp_path):
    controller.reserve_stage(tmp_path, "train")
    with pytest.raises(FileExistsError):
        controller.reserve_stage(tmp_path, "train")
    assert json.loads((tmp_path / "claim-train.json").read_text())["stage"] == "train"
    (tmp_path / "claim-train.json").unlink()
    (tmp_path / "training").mkdir()
    with pytest.raises(FileExistsError, match="no retry or overwrite"):
        controller.reserve_stage(tmp_path, "train")


@pytest.mark.parametrize("condition", ["wrong_parent", "wrong_step", "cpu_collision", "overwrite"])
def test_configuration_rejects_unapproved_or_overwriting_branches(condition):
    config = configuration()
    if condition == "wrong_parent":
        config["runtime"]["shared_gpu_allowlist"]["loop"].append("unrelated-job")
    elif condition == "wrong_step":
        config["branch"]["checkpoint_step"] = 19000
    elif condition == "cpu_collision":
        config["runtime"]["cpu_sets"]["loop"][0] = 8
    else:
        config["models"]["loop"]["checkpoint"] = config["branch"]["checkpoints"]["loop"]
    with pytest.raises(ValueError):
        controller.validate_config(config)


def test_parent_inputs_readonly_and_control_outputs_separate(tmp_path):
    config = configuration()
    batch = tmp_path / config["batch"]
    parent = tmp_path / config["branch"]["parent_batch"]
    out = batch / "discoloop"
    name, command = controller.build_command(
        config,
        "discoloop",
        "train",
        "sha256:pinned",
        {"uuid": "GPU-eight"},
        out,
        batch,
        parent,
        tmp_path / "missing-token",
    )
    assert name == f"{config['batch']}-discoloop-train"
    assert command[:4] == ["docker", "--context", "lm-memory", "run"]
    assert "--detach" in command
    assert f"type=bind,source={parent / 'source'},target=/parent-frozen,readonly" in command
    assert f"type=bind,source={tmp_path},target=/project/results,readonly" in command
    assert f"type=bind,source={batch / 'input-mount/data'},target=/project/data,readonly" in command
    assert not any("source/input-backup" in value for value in command)
    assert f"type=bind,source={out},target=/project/results/{config['batch']}/discoloop" in command
    assert "PYTHONPATH=/frozen/src" in command
    assert "sha256:pinned" in command
    assert not any("gpu-8.lock" in value for value in command)


def test_parent_manifest_and_config_are_pinned(tmp_path):
    config = configuration()
    parent = tmp_path / config["branch"]["parent_batch"]
    source = parent / "source/configs"
    source.mkdir(parents=True)
    frozen_config = source / "experiment.json"
    frozen_config.write_text("original parent configuration\n")
    config_digest = controller.sha256(frozen_config)
    manifest = parent / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_sha256": config_digest,
                "files": {"configs/experiment.json": config_digest},
            }
        )
    )
    config["branch"]["parent_manifest_sha256"] = controller.sha256(manifest)
    config["branch"]["parent_config_sha256"] = config_digest
    assert controller.verify_parent(tmp_path / config["batch"], config) == parent
    frozen_config.write_text("changed parent configuration\n")
    with pytest.raises(ValueError, match="Parent frozen source changed"):
        controller.verify_parent(tmp_path / config["batch"], config)


@pytest.mark.parametrize("private_umask", [0o022, 0o077])
def test_freeze_copies_parent_input_bytes_and_never_refreshes(tmp_path, monkeypatch, private_umask):
    project = tmp_path / "project"
    parent = tmp_path / "parent"
    for relative in ("src", "scripts", "configs", "data/ttcd-second-hop-scratch-v1"):
        (project / relative).mkdir(parents=True, exist_ok=True)
    backup = parent / "source/input-backup"
    backup.mkdir(parents=True)
    for name in ("world.npz", "panels.npz", "complete.json", "training-split.json"):
        (backup / name).write_bytes(b"parent original input")
        (project / "data/ttcd-second-hop-scratch-v1" / name).write_bytes(b"parent original input")
    (backup / "model-config.json").write_text("parent model configuration")
    (project / "data/ttcd-second-hop-scratch-v1/model.json").write_text(
        "parent model configuration"
    )
    (project / "src/example.py").write_text("original = True\n")
    for name in ("run_ttcd_second_hop.py", "prepare_ttcd_scratch.py", "launch_ttcd_branch.py"):
        (project / "scripts" / name).write_text("pass\n")
    (project / "configs/experiment-tracking-defaults.json").write_text("{}")
    config_path = project / "configs/new.json"
    config_path.write_text(json.dumps(configuration()))
    batch = tmp_path / "new-batch"
    batch.mkdir()
    monkeypatch.setattr(controller, "ROOT", project)
    old_umask = os.umask(private_umask)
    try:
        controller.freeze_source(batch, config_path, parent)
    finally:
        os.umask(old_umask)
    assert (batch / "source/input-backup").stat().st_mode & 0o777 == 0o755
    assert (batch / "source/scripts/prepare_ttcd_scratch.py").is_file()
    contract = json.loads((batch / "source/controller-contract.json").read_text())
    assert contract["stages"] == list(controller.STAGES)
    assert not contract["automatic_retry"]
    (project / "src/example.py").write_text("changed = True\n")
    (project / "data/ttcd-second-hop-scratch-v1/world.npz").write_bytes(b"changed live data")
    controller.freeze_source(batch, config_path, parent)
    assert (batch / "source/src/example.py").read_text() == "original = True\n"
    assert (batch / "source/input-backup/world.npz").read_bytes() == b"parent original input"
    config_path.write_text("{}")
    with pytest.raises(ValueError, match="Configuration changed"):
        controller.freeze_source(batch, config_path, parent)


@pytest.mark.parametrize("private_umask", [0o022, 0o077])
def test_input_overlay_copies_exact_bytes_without_touching_frozen_source(tmp_path, private_umask):
    config = configuration()
    batch = tmp_path / config["batch"]
    frozen = batch / "source"
    backup = frozen / "input-backup"
    backup.mkdir(parents=True)
    backup.chmod(0o700)
    originals = (
        "world.npz",
        "panels.npz",
        "complete.json",
        "training-split.json",
        "model-config.json",
    )
    for name in originals:
        (backup / name).write_bytes(f"immutable {name}".encode())
    (batch / "source-manifest.json").write_text('{"original": true}\n')
    frozen_hashes = {name: controller.sha256(backup / name) for name in originals}
    old_umask = os.umask(private_umask)
    try:
        data_root = controller.prepare_input_mount(batch, config)
    finally:
        os.umask(old_umask)
    assert data_root == batch / "input-mount/data"
    assert backup.stat().st_mode & 0o777 == 0o700
    for directory in (batch / "input-mount", data_root, data_root / "ttcd-second-hop-scratch-v1"):
        assert directory.stat().st_mode & 0o777 == 0o755
    assert frozen_hashes == {name: controller.sha256(backup / name) for name in originals}
    manifest_path = batch / "input-mount-manifest.json"
    manifest_before = manifest_path.read_bytes()
    assert len(json.loads(manifest_before)["files"]) == 5
    assert controller.prepare_input_mount(batch, config) == data_root
    assert manifest_path.read_bytes() == manifest_before
    target = batch / "input-mount" / config["data_dir"] / "world.npz"
    assert target.read_bytes() == (backup / "world.npz").read_bytes()
    target.write_bytes(b"changed overlay input")
    with pytest.raises(ValueError, match="Input mount bytes changed"):
        controller.prepare_input_mount(batch, config)
    assert target.read_bytes() == b"changed overlay input"
    assert frozen_hashes == {name: controller.sha256(backup / name) for name in originals}


def test_pipeline_stops_arm_on_calibration_exit_three(tmp_path, monkeypatch):
    config = configuration()
    parent = tmp_path / config["branch"]["parent_batch"]
    checkpoint = parent / "discoloop/training/checkpoint-0020000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"published checkpoint")
    batch = tmp_path / config["batch"]
    batch.mkdir()
    launched = []

    def fake_run(command, **kwargs):
        launched.append(command[command.index("/frozen/scripts/run_ttcd_second_hop.py") + 1])
        return SimpleNamespace(returncode=0, stdout="new-container-id\n", stderr="")

    monkeypatch.setattr(controller, "inspect_container", lambda _: None)
    monkeypatch.setattr(controller, "prepare_input_mount", lambda *args: None)
    monkeypatch.setattr(controller, "admit_shared_job", lambda *args: {"uuid": "GPU-eight"})
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    monkeypatch.setattr(controller.fcntl, "flock", lambda *args: None)
    exits = iter((0, 0, 3))
    monkeypatch.setattr(controller, "monitor_container", lambda *args: next(exits))
    code = controller.run_arm(
        config,
        "discoloop",
        list(controller.STAGES),
        "sha256:pinned",
        batch,
        parent,
        tmp_path / "no-token",
    )
    assert code == 3
    assert launched == ["train", "train-audit", "calibrate"]
    out = batch / "discoloop"
    assert json.loads((out / "controller-state.json").read_text())["state"] == "calibration_failed"
    assert json.loads((out / "exit-calibrate.json").read_text())["returncode"] == 3
    assert not (out / "claim-evaluate.json").exists()
