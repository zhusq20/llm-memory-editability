"""The foundation restoration must preserve the recipe and wait without preemption."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "discoloop_restore_controller", ROOT / "scripts/launch_discoloop_restore.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def configuration():
    root = f"data/{controller.BATCH}/original"
    return {
        "batch": controller.BATCH,
        "spec": {
            "name": controller.BATCH,
            "unique_layers": 4,
            "repeats": 2,
            "phi": 7.2,
            "weight_decay": 0.1,
            "initialization": 791101,
            "stream_seed": 792101,
            "lr": 1e-4,
            "warmup": 2000,
            "batch_size": 512,
            "steps": 300000,
        },
        "arms": {
            "discoloop": {
                "kind": "none",
                "coefficient": 0.0,
                "scope": "supervised",
                "first_round_kv": False,
                "discoloop": True,
                "discoloop_alpha": 1.0,
                "discoloop_temperature": 1.0,
                "discoloop_eps": 1e-6,
            }
        },
        "input_root": root,
        "initial_checkpoint": f"{root}/development/loop4x2-phi7.2-wd0.1/checkpoint-0000000.pt",
        "evaluation_nodes": [0, 1000, 300000],
        "restore": {"mode": "rebuild"},
        "runtime": {
            "image": "sha256:" + "9" * 64,
            "docker_context": "lm-memory",
            "gpu_preference": [8, 9],
            "cpu_set": list(range(24, 32)),
            "memory_gib": 96,
            "min_total_gpu_mib": 90000,
            "min_free_gpu_mib": 80000,
            "tracking": "online",
            "persistent_root": "/ossfs/workspace/llm-memory-editability",
            "timeout_hours": 12,
            "wait_timeout_hours": 24,
        },
    }


def test_default_plan_never_touches_docker_gpu_or_persistent_outputs(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration()))
    monkeypatch.setattr(sys, "argv", ["launch_discoloop_restore.py", "--config", str(path)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Dryrun cannot touch runtime resources")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    monkeypatch.setattr(controller, "persistent_batch", forbidden)
    monkeypatch.setattr(controller, "freeze_source", forbidden)
    controller.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"] and not plan["gpu_sharing"] and not plan["automatic_retry"]
    assert plan["stages"] == list(controller.STAGES)
    assert plan["gpu_dependencies"] == {"8": "single", "9": "stream"}
    assert plan["original_image"] == controller.ORIGINAL_IMAGE
    assert plan["downstream_config"] is None


@pytest.mark.parametrize(
    "change",
    [
        "batch",
        "width_recipe",
        "loss",
        "kv",
        "gpu",
        "sharing",
        "checkpoint",
        "small_gpu",
        "tracking",
    ],
)
def test_configuration_preserves_one_arm_and_original_budget(change):
    config = configuration()
    if change == "batch":
        config["batch"] = "supervised-alignment-development-v1"
    elif change == "width_recipe":
        config["spec"]["unique_layers"] = 2
    elif change == "loss":
        config["arms"]["discoloop"]["coefficient"] = 0.3
    elif change == "kv":
        config["arms"]["discoloop"]["first_round_kv"] = True
    elif change == "gpu":
        config["runtime"]["gpu_preference"] = [0, 1]
    elif change == "sharing":
        config["runtime"]["shared_gpu_allowlist"] = ["old-task"]
    elif change == "checkpoint":
        config["initial_checkpoint"] = "results/old/checkpoint-0000000.pt"
    elif change == "small_gpu":
        config["runtime"]["min_total_gpu_mib"] = 20000
    else:
        config["runtime"]["tracking"] = "disabled"
    with pytest.raises(ValueError):
        controller.validate_config(config)


def test_new_spec_name_does_not_change_original_recipe_and_reload_never_trains():
    config = configuration()
    controller.validate_config(config)
    config["spec"]["name"] = "new-explicit-restoration"
    controller.validate_config(config)
    config["restore"]["mode"] = "reload"
    config["restore"]["recovered_checkpoint"] = config["input_root"] + "/checkpoint-0300000.pt"
    assert controller.pipeline_stages(config) == [
        "preflight",
        "audit",
        "foundation-gate",
        "cloud-audit",
    ]


def make_predecessor(tmp_path, arm, passed=True):
    receipt = tmp_path / "ttcd-cross-sample-v1" / arm / "audit.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"passed": passed, "answer_and_EOS_exact": passed}))
    return receipt


def test_matching_local_science_audit_releases_gpu_without_waiting_for_cloud_or_other_arm(tmp_path):
    config = configuration()
    batch = tmp_path / controller.BATCH
    path = make_predecessor(tmp_path, "single")
    receipt = controller.predecessor_receipt(batch, config, 8)
    assert receipt["passed"] and receipt["arm"] == "single"
    assert receipt["sha256"] == controller.sha256(path)
    assert not (tmp_path / "ttcd-cross-sample-v1/controller-state.json").exists()
    with pytest.raises(controller.ResourceUnavailable, match="stream"):
        controller.predecessor_receipt(batch, config, 9)


def test_failed_science_audit_cannot_release_its_gpu(tmp_path):
    make_predecessor(tmp_path, "single", passed=False)
    with pytest.raises(RuntimeError, match="science audit did not pass"):
        controller.predecessor_receipt(tmp_path / controller.BATCH, configuration(), 8)


def test_admission_checks_all_host_compute_pids_and_lm_gpu_reservations(tmp_path, monkeypatch):
    config = configuration()
    make_predecessor(tmp_path, "single")
    batch = tmp_path / controller.BATCH
    row = {
        "uuid": "GPU-eight",
        "memory": 0,
        "free_mib": 97000,
        "total_mib": 97887,
        "utilization": 0,
    }
    processes, containers = [], []
    monkeypatch.setattr(controller, "running_containers", lambda _: containers)
    monkeypatch.setattr(controller, "gpu_snapshot", lambda: ({8: row}, processes))
    monkeypatch.setattr(controller, "host_memory_available_kib", lambda: 1024**3)
    monkeypatch.setattr(controller.os, "sched_getaffinity", lambda _: set(range(96)))
    assert controller.admit_job(batch, config, 8)["gpu"] == 8
    processes.append({"uuid": "GPU-eight", "pid": 1047})
    with pytest.raises(controller.ResourceUnavailable, match="occupied"):
        controller.admit_job(batch, config, 8)
    processes.clear()
    containers.append(
        {
            "Name": "/reserved-not-yet-using-cuda",
            "State": {"Running": True},
            "HostConfig": {"CpusetCpus": "0-3", "DeviceRequests": [{"DeviceIDs": ["GPU-eight"]}]},
        }
    )
    with pytest.raises(controller.ResourceUnavailable, match="occupied"):
        controller.admit_job(batch, config, 8)
    containers[0]["HostConfig"]["DeviceRequests"] = [{"DeviceIDs": ["9"]}]
    assert controller.admit_job(batch, config, 8)["gpu"] == 8


def test_waiter_tries_only_reviewed_gpu_slots_without_launching_or_preemption(
    tmp_path, monkeypatch
):
    attempts, updates = [], []
    config = configuration()

    def admit(batch, config, gpu):
        attempts.append(gpu)
        if gpu == 8:
            raise controller.ResourceUnavailable("single science audit still pending")
        return {"gpu": 9, "uuid": "GPU-nine"}

    monkeypatch.setattr(controller, "admit_job", admit)
    monkeypatch.setattr(controller.fcntl, "flock", lambda *args: None)
    with controller.wait_for_slot(
        tmp_path / controller.BATCH, config, float("inf"), updates.append
    )[0] as held:
        assert not held.closed
    assert attempts == [8, 9]
    assert not updates
    assert not list(tmp_path.glob("claim-*"))


def test_wait_timeout_does_not_inspect_or_stop_existing_containers(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Timeout cannot touch another task")

    monkeypatch.setattr(controller, "admit_job", forbidden)
    monkeypatch.setattr(controller, "docker", forbidden)
    with pytest.raises(TimeoutError, match="no existing experiment was stopped"):
        controller.wait_for_slot(tmp_path / controller.BATCH, configuration(), 0)


def test_stage_claims_prevent_repeated_training(tmp_path):
    controller.reserve_stage(tmp_path, "train")
    with pytest.raises(FileExistsError):
        controller.reserve_stage(tmp_path, "train")


def make_project(tmp_path, config):
    project = tmp_path / "project"
    for relative in ("src", "scripts", "tests", "configs"):
        (project / relative).mkdir(parents=True)
    (project / "src/example.py").write_text("original = True\n")
    (project / "tests/test_example.py").write_text("assert True\n")
    for script in controller.SOURCE_SCRIPTS:
        (project / "scripts" / script).write_text("pass\n")
    for name in ("pyproject.toml", "CONTRIBUTING.md"):
        (project / name).write_text("metadata\n")
    for relative in controller.PROTOCOLS:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("frozen design and330k budget\n")
    (project / "configs/experiment-tracking-defaults.json").write_text("{}")
    (project / "configs/ttcd-second-hop-scratch-v1.json").write_text('{"fixture_only": true}')
    for relative in controller.input_files(config):
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"immutable {relative}".encode())
    path = project / "configs/experiment.json"
    path.write_text(json.dumps(config))
    return project, path


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_freeze_and_overlay_preserve_original_inputs_without_live_directory_mounts(
    tmp_path, monkeypatch, umask
):
    config = configuration()
    project, config_path = make_project(tmp_path, config)
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    monkeypatch.setattr(controller, "ROOT", project)
    previous = os.umask(umask)
    try:
        controller.freeze_source(batch, config_path)
        data_root = controller.prepare_input_mount(batch, config)
    finally:
        os.umask(previous)
    manifest = json.loads((batch / "source-manifest.json").read_text())
    assert set(manifest["inputs"]) == {str(path) for path in controller.input_files(config)}
    assert (batch / "source/tests/test_example.py").is_file()
    assert (batch / "source/scripts/run_supervised_alignment.py").is_file()
    assert (batch / "source/configs/ttcd-second-hop-scratch-v1.json").is_file()
    assert data_root.stat().st_mode & 0o777 == 0o755
    for relative, expected in manifest["inputs"].items():
        target = batch / "input-mount" / relative
        assert controller.sha256(target) == expected
        assert target.stat().st_mode & 0o777 == 0o644
    (project / "src/example.py").write_text("changed = True\n")
    controller.freeze_source(batch, config_path)
    assert (batch / "source/src/example.py").read_text() == "original = True\n"
    target = batch / "input-mount" / config["initial_checkpoint"]
    target.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="overlay bytes changed"):
        controller.prepare_input_mount(batch, config)


def test_missing_step_zero_stops_before_partial_freeze(tmp_path, monkeypatch):
    config = configuration()
    project, config_path = make_project(tmp_path, config)
    (project / config["initial_checkpoint"]).unlink()
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    monkeypatch.setattr(controller, "ROOT", project)
    with pytest.raises(FileNotFoundError, match="Required restored input"):
        controller.freeze_source(batch, config_path)
    assert not (batch / "source").exists()


@pytest.mark.parametrize(
    "message",
    [
        "error: no such object: discoloop-success-restore-v1-preflight",
        "Error response from daemon: No such container: discoloop-success-restore-v1-train",
    ],
)
def test_missing_container_is_recognized_without_retrying_daemon_errors(monkeypatch, message):
    monkeypatch.setattr(
        controller,
        "docker",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr=message,
            stdout="[]",
        ),
    )
    assert controller.inspect_container("new-container") is None
    monkeypatch.setattr(
        controller,
        "docker",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="permission denied connecting to LM daemon",
            stdout="",
        ),
    )
    with pytest.raises(RuntimeError, match="Cannot inspect container"):
        controller.inspect_container("new-container")


def test_container_mounts_pinned_image_readonly_inputs_and_only_new_batch_outputs(tmp_path):
    config = configuration()
    batch = tmp_path / controller.BATCH
    token = tmp_path / "token"
    token.write_text("must-not-be-logged")
    name, command = controller.build_command(config, "train", {"uuid": "GPU-eight"}, batch, token)
    assert name == f"{controller.BATCH}-train"
    assert command[:4] == ["docker", "--context", "lm-memory", "run"]
    assert config["runtime"]["image"] in command
    assert f"type=bind,source={batch / 'input-mount/data'},target=/project/data,readonly" in command
    assert f"type=bind,source={batch / 'source'},target=/frozen,readonly" in command
    assert (
        f"type=bind,source={batch / 'source'},"
        f"target=/project/results/{controller.BATCH}/source,readonly" in command
    )
    assert f"type=bind,source={batch},target=/project/results/{controller.BATCH}" in command
    assert "1015:1016" in command
    assert "/frozen/scripts/run_discoloop_restore.py" in command
    assert "HOME=/tmp/lm-home" in command
    assert "must-not-be-logged" not in " ".join(command)


@pytest.mark.parametrize("failed_stage", ["smoke-audit", "audit", "foundation-gate"])
def test_pipeline_stops_on_failure_without_later_stage_or_downstream_launch(
    tmp_path, monkeypatch, failed_stage
):
    config = configuration()
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    launched = []
    row = {"gpu": 8, "uuid": "GPU-eight"}
    monkeypatch.setattr(
        controller, "wait_for_slot", lambda *args: ((tmp_path / "gpu.lock").open("a"), row)
    )
    monkeypatch.setattr(controller, "prepare_input_mount", lambda *args: None)
    monkeypatch.setattr(controller, "admit_job", lambda *args: row)
    monkeypatch.setattr(controller, "inspect_container", lambda _: None)

    def launch(command, **kwargs):
        launched.append(command[command.index("/frozen/scripts/run_discoloop_restore.py") + 1])
        return SimpleNamespace(returncode=0, stderr="", stdout="own-container\n")

    monkeypatch.setattr(controller.subprocess, "run", launch)
    monkeypatch.setattr(
        controller,
        "monitor_container",
        lambda name, *args: 3 if name == f"{controller.BATCH}-{failed_stage}" else 0,
    )
    assert controller.run_pipeline(batch, config, list(controller.STAGES), tmp_path / "token") == 3
    assert launched == list(controller.STAGES[: controller.STAGES.index(failed_stage) + 1])
    assert not (batch / "claim-cloud-audit.json").exists()
    assert not (batch / "downstream-handoff.json").exists()


def test_downstream_requires_passing_gate_and_successful_gate_exit(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Downstream cannot run before a passing foundation gate")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    (tmp_path / "result-foundation-gate.json").write_text('{"passed": false}')
    (tmp_path / "exit-foundation-gate.json").write_text('{"returncode": 3}')
    with pytest.raises(ValueError, match="passing local foundation gate"):
        controller.run_downstream(
            tmp_path, tmp_path / "future.json", tmp_path / "future.py", tmp_path / "token"
        )


def test_downstream_runs_only_explicit_reviewed_launcher_after_passing_gate(tmp_path, monkeypatch):
    config_path, launcher = tmp_path / "paired.json", tmp_path / "paired.py"
    config_path.write_text('{"batch":"separate-paired-batch"}')
    launcher.write_text("pass\n")
    (tmp_path / "result-foundation-gate.json").write_text('{"passed": true}')
    (tmp_path / "exit-foundation-gate.json").write_text('{"returncode": 0}')
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(controller.subprocess, "run", run)
    assert controller.run_downstream(tmp_path, config_path, launcher, tmp_path / "token") == 0
    assert calls[0][:5] == [sys.executable, str(launcher), "pipeline", "--config", str(config_path)]
    handoff = json.loads((tmp_path / "downstream-handoff.json").read_text())
    assert handoff["config_sha256"] == controller.sha256(config_path)
    assert handoff["gate_sha256"] == controller.sha256(tmp_path / "result-foundation-gate.json")


def test_downstream_science_is_pinned_before_foundation_and_checked_at_handoff(
    tmp_path, monkeypatch
):
    config = configuration()
    project, config_path = make_project(tmp_path, config)
    for relative in controller.DOWNSTREAM_SCRIPTS:
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("pass\n")
    launcher = project / "scripts/paired.py"
    launcher.write_text("pass\n")
    monkeypatch.setattr(controller, "ROOT", project)
    pinned = controller.downstream_contract(config_path, launcher)
    assert pinned["launcher_sha256"] == controller.sha256(launcher)
    assert "src/example.py" in pinned["files"]
    assert all(relative in pinned["files"] for relative in controller.PROTOCOLS)
    (tmp_path / "result-foundation-gate.json").write_text('{"passed": true}')
    (tmp_path / "exit-foundation-gate.json").write_text('{"returncode": 0}')
    (project / "src/example.py").write_text("changed = True\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Changed downstream science must not be silently launched")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="scientific source changed after startup"):
        controller.run_downstream(tmp_path, config_path, launcher, tmp_path / "token", pinned)
