"""Cross-sample execution must preserve parents, isolation and stage gates."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ttcd_continual_controller", ROOT / "scripts/launch_ttcd_continual.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def configuration():
    batch = "ttcd-cross-sample-v1"
    parent = "ttcd-second-hop-scratch-v1"
    return {
        "batch": batch,
        "parent": {
            "parent_batch": parent,
            "parent_model": "discoloop",
            "parent_source_dir": "/parent-frozen",
            "parent_manifest_path": "/parent-source-manifest.json",
            "parent_manifest_sha256": "a" * 64,
            "parent_config_sha256": "b" * 64,
            "parent_image_sha256": "sha256:" + "c" * 64,
            "checkpoint_step": 20000,
            "checkpoint_sha256": "d" * 64,
            "steps": 10000,
            "checkpoints": {
                name: f"results/{parent}/discoloop/training/checkpoint-0020000.pt"
                for name in controller.ARMS
            },
        },
        "models": {
            name: {"checkpoint": f"results/{batch}/{name}/training/parent.pt"}
            for name in controller.ARMS
        },
        "data_dir": "data/ttcd-second-hop-scratch-v1",
        "model_config": "data/ttcd-second-hop-scratch-v1/model.json",
        "continual": {
            "panel_source_path": "configs/ttcd-cross-sample-v1-panel.json",
            "panel_path": "/frozen/configs/continual-panel.json",
        },
        "runtime": {
            "gpus": {"single": 8, "stream": 9},
            "cpu_sets": {"single": [16, 17, 18, 19], "stream": [20, 21, 22, 23]},
            "cpu_threads": 4,
            "memory_gib": 48,
            "persistent_root": "/ossfs/workspace/llm-memory-editability",
            "timeout_hours": 12,
            "resource_wait_timeout_hours": 12,
            "gpu_memory_max_mib": 1024,
            "gpu_utilization_max": 5,
            "docker_inspection_contexts": ["lm-memory"],
            "tracking": "online",
        },
    }


def shared_configuration():
    config = configuration()
    config["runtime"]["shared_gpu_allowlist"] = {
        name: list(containers) for name, containers in controller.ALLOWED_SHARED.items()
    }
    config["runtime"]["shared_gpu_min_free_mib"] = 16384
    return config


def make_checkpoint(parent, config):
    checkpoint = parent / "discoloop/training/checkpoint-0020000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"exact original parent checkpoint")
    config["parent"]["checkpoint_sha256"] = controller.sha256(checkpoint)
    return checkpoint


def test_default_plan_is_readonly_and_includes_preflight_and_cloud_audit(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration()))
    monkeypatch.setattr(sys, "argv", ["launch_ttcd_continual.py", "--config", str(config_path)])

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Read-only planning must not access Docker, GPUs or persistent outputs"
        )

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    monkeypatch.setattr(controller, "freeze_source", forbidden)
    monkeypatch.setattr(controller, "persistent_batch", forbidden)
    controller.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"]
    assert not plan["gpu_sharing"]
    assert not plan["automatic_retry"]
    assert [job["model"] for job in plan["jobs"]] == ["single", "stream"]
    assert [job["gpu"] for job in plan["jobs"]] == [8, 9]
    assert plan["stages"] == list(controller.STAGES)
    assert plan["stages"][0] == "preflight"
    assert plan["stages"][-1] == "cloud-audit"


@pytest.mark.parametrize(
    "condition",
    [
        "wrong_parent",
        "wrong_architecture",
        "wrong_step",
        "wrong_checkpoint",
        "gpu_collision",
        "shared_gpu",
        "cpu_collision",
        "overwrite_parent",
        "output_traversal",
        "unfixed_checkpoint",
        "live_panel",
    ],
)
def test_configuration_rejects_changes_outside_authorized_matrix(condition):
    config = configuration()
    if condition == "wrong_parent":
        config["parent"]["parent_batch"] = "another-batch"
    elif condition == "wrong_architecture":
        config["parent"]["parent_model"] = "loop"
    elif condition == "wrong_step":
        config["parent"]["checkpoint_step"] = 19000
    elif condition == "wrong_checkpoint":
        config["parent"]["checkpoints"]["stream"] = "results/other/checkpoint.pt"
    elif condition == "gpu_collision":
        config["runtime"]["gpus"]["stream"] = 8
    elif condition == "shared_gpu":
        config["runtime"]["shared_gpu_allowlist"] = {"single": ["old-parent"]}
    elif condition == "cpu_collision":
        config["runtime"]["cpu_sets"]["stream"][0] = 16
    elif condition == "overwrite_parent":
        config["models"]["single"]["checkpoint"] = config["parent"]["checkpoints"]["single"]
    elif condition == "output_traversal":
        config["models"]["single"]["checkpoint"] = (
            "results/ttcd-cross-sample-v1/single/../../parent/checkpoint.pt"
        )
    elif condition == "unfixed_checkpoint":
        config["parent"]["checkpoint_sha256"] = "not-fixed"
    elif condition == "live_panel":
        config["continual"]["panel_path"] = "configs/live-panel.json"
    with pytest.raises(ValueError):
        controller.validate_config(config)


def test_valid_configuration_preserves_exact_two_arm_matrix():
    controller.validate_config(configuration())
    controller.validate_config(shared_configuration())


@pytest.mark.parametrize(
    "message",
    [
        "error: no such object: ttcd-cross-sample-v1-single-preflight",
        "Error response from daemon: No such container: ttcd-cross-sample-v1-stream-preflight",
        "Error: No such object: ttcd-cross-sample-v1-single-train",
    ],
)
def test_inspect_recognizes_missing_container_case_insensitively(monkeypatch, message):
    monkeypatch.setattr(
        controller,
        "docker",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=message, stdout="[]\n"),
    )
    assert controller.inspect_container("new-container") is None


@pytest.mark.parametrize(
    "message",
    [
        "Cannot connect to the Docker daemon at unix:///run/docker-lm-memory/docker.sock",
        "permission denied while trying to connect to the Docker daemon socket",
        "no such file or directory: /run/docker-lm-memory/docker.sock",
    ],
)
def test_inspect_never_mistakes_daemon_failure_for_missing_container(monkeypatch, message):
    monkeypatch.setattr(
        controller,
        "docker",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=message, stdout=""),
    )
    with pytest.raises(RuntimeError, match="Cannot inspect container"):
        controller.inspect_container("new-container")


@pytest.mark.parametrize("change", ["wrong_original", "third_gpu", "less_free_memory"])
def test_shared_configuration_is_scoped_to_latest_explicit_authorization(change):
    config = shared_configuration()
    if change == "wrong_original":
        config["runtime"]["shared_gpu_allowlist"]["stream"] = ["unrelated-project"]
    elif change == "third_gpu":
        config["runtime"]["gpus"]["stream"] = 7
    else:
        config["runtime"]["shared_gpu_min_free_mib"] = 8192
    with pytest.raises(ValueError):
        controller.validate_config(config)


def test_shared_gpu_accepts_only_whitelisted_host_pids_and_reserved_free_memory():
    runtime = shared_configuration()["runtime"]
    row = {"uuid": "GPU-eight", "free_mib": 20000, "memory": 4000, "utilization": 80}
    old_parent = {"uuid": "GPU-eight", "pid": 101}
    other_gpu = {"uuid": "GPU-nine", "pid": 900}
    assert (
        controller.check_shared_gpu(8, runtime, ({8: row}, [old_parent, other_gpu]), {101}) == row
    )
    with pytest.raises(controller.ResourceUnavailable, match="unrelated CUDA processes"):
        controller.check_shared_gpu(
            8, runtime, ({8: row}, [old_parent, {"uuid": "GPU-eight", "pid": 102}]), {101}
        )
    with pytest.raises(controller.ResourceUnavailable, match="reserved free memory"):
        controller.check_shared_gpu(8, runtime, ({8: {**row, "free_mib": 16383}}, []), set())


def test_shared_admission_derives_pid_allowlist_from_named_lm_image(tmp_path, monkeypatch):
    config = shared_configuration()
    parent = {
        "Name": "/" + controller.ALLOWED_SHARED["single"][0],
        "Id": "original-container-id",
        "Image": config["parent"]["parent_image_sha256"],
        "InspectionContext": "lm-memory",
        "State": {"Running": True, "Pid": 100},
        "HostConfig": {"CpusetCpus": "0-3"},
    }
    row = {"uuid": "GPU-eight", "free_mib": 20000, "memory": 4000, "utilization": 80}
    monkeypatch.setattr(controller, "running_containers", lambda _: [parent])
    monkeypatch.setattr(controller.os, "sched_getaffinity", lambda _: set(range(24)))
    monkeypatch.setattr(controller, "host_memory_available_kib", lambda: 1024**3)
    monkeypatch.setattr(controller, "container_pids", lambda _: {100, 101})
    monkeypatch.setattr(
        controller, "gpu_snapshot", lambda: ({8: row}, [{"uuid": "GPU-eight", "pid": 101}])
    )
    admission = controller.admit_job(config, "single")
    assert admission["gpu_sharing"]
    assert admission["allowed_host_pids"] == [100, 101]
    assert admission["observed_shared_containers"][0]["name"] == parent["Name"]
    parent["Image"] = "sha256:unapproved-image"
    with pytest.raises(ValueError, match="pinned parent image"):
        controller.admit_job(config, "single")


def test_any_compute_pid_or_memory_or_utilization_blocks_gpu_admission():
    runtime = configuration()["runtime"]
    row = {"uuid": "GPU-eight", "memory": 0, "utilization": 0}
    assert controller.check_idle_gpu(8, runtime, ({8: row}, [])) == row
    with pytest.raises(controller.ResourceUnavailable, match="occupied"):
        controller.check_idle_gpu(8, runtime, ({8: row}, [{"uuid": "GPU-eight", "pid": 101}]))
    for busy in ({**row, "memory": 1025}, {**row, "utilization": 6}):
        with pytest.raises(controller.ResourceUnavailable, match="no sharing"):
            controller.check_idle_gpu(8, runtime, ({8: busy}, []))
    with pytest.raises(ValueError, match="does not exist"):
        controller.check_idle_gpu(9, runtime, ({8: row}, []))


def test_cpu_isolation_covers_running_jobs_and_available_affinity():
    running = {
        "Name": "/old-parent",
        "State": {"Running": True},
        "HostConfig": {"CpusetCpus": "0-3,6"},
    }
    controller.check_cpu_sets([16, 17, 18, 19], [running], range(24))
    with pytest.raises(controller.ResourceUnavailable, match="collides"):
        controller.check_cpu_sets([3, 4, 5, 7], [running], range(24))
    with pytest.raises(RuntimeError, match="available affinity"):
        controller.check_cpu_sets([24], [running], range(24))
    with pytest.raises(RuntimeError, match="Cannot verify CPU isolation"):
        controller.check_cpu_sets([16], [{**running, "HostConfig": {"CpusetCpus": ""}}], range(24))


def test_resource_queue_waits_without_claim_or_stop(tmp_path, monkeypatch):
    config = configuration()
    states, locks, admissions = [], [], []

    def fake_admit(*args):
        admissions.append(True)
        if len(admissions) == 1:
            raise controller.ResourceUnavailable("Original parent still occupies GPU8")
        return {"uuid": "GPU-eight"}

    monkeypatch.setattr(controller, "admit_job", fake_admit)
    monkeypatch.setattr(controller.fcntl, "flock", lambda _, mode: locks.append(mode))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    with (tmp_path / "gpu.lock").open("a") as lock:
        row = controller.wait_for_resources(config, "single", lock, float("inf"), states.append)
    assert row == {"uuid": "GPU-eight"}
    assert states[0]["state"] == "waiting_for_resources"
    assert controller.fcntl.LOCK_UN in locks
    assert len(admissions) == 2
    assert not list(tmp_path.glob("claim-*"))


def test_resource_queue_does_not_retry_inspection_failures(tmp_path, monkeypatch):
    def broken_daemon(*args):
        raise RuntimeError("Cannot reach LM Docker daemon")

    def forbidden(*args):
        raise AssertionError("Infrastructure failures must not be silently retried")

    monkeypatch.setattr(controller, "admit_job", broken_daemon)
    monkeypatch.setattr(controller.time, "sleep", forbidden)
    with (tmp_path / "gpu.lock").open("a") as lock:
        with pytest.raises(RuntimeError, match="Cannot reach"):
            controller.wait_for_resources(configuration(), "single", lock, float("inf"))


def test_resource_wait_timeout_never_touches_docker_or_existing_jobs(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Resource timeout must not touch another job")

    monkeypatch.setattr(controller, "admit_job", forbidden)
    monkeypatch.setattr(controller, "docker", forbidden)
    with (tmp_path / "gpu.lock").open("a") as lock:
        with pytest.raises(TimeoutError, match="no experiment was launched or stopped"):
            controller.wait_for_resources(configuration(), "single", lock, 0)


def test_exclusive_claim_and_existing_training_prevent_resubmission(tmp_path):
    controller.reserve_stage(tmp_path, "train")
    with pytest.raises(FileExistsError):
        controller.reserve_stage(tmp_path, "train")
    assert json.loads((tmp_path / "claim-train.json").read_text())["stage"] == "train"
    (tmp_path / "claim-train.json").unlink()
    (tmp_path / "training").mkdir()
    with pytest.raises(FileExistsError, match="no retry or overwrite"):
        controller.reserve_stage(tmp_path, "train")


def test_parent_checkpoint_identity_and_source_manifest_are_pinned(tmp_path):
    config = configuration()
    parent = tmp_path / config["parent"]["parent_batch"]
    source = parent / "source/configs"
    source.mkdir(parents=True)
    frozen_config = source / "experiment.json"
    frozen_config.write_text("original parent configuration\n")
    digest = controller.sha256(frozen_config)
    manifest = parent / "source-manifest.json"
    manifest.write_text(
        json.dumps({"config_sha256": digest, "files": {"configs/experiment.json": digest}})
    )
    config["parent"]["parent_config_sha256"] = digest
    config["parent"]["parent_manifest_sha256"] = controller.sha256(manifest)
    checkpoint = make_checkpoint(parent, config)
    assert controller.verify_parent(tmp_path / config["batch"], config) == parent
    checkpoint.write_bytes(b"different step20000 bytes")
    with pytest.raises(ValueError, match="checkpoint identity changed"):
        controller.verify_parent(tmp_path / config["batch"], config)
    checkpoint.write_bytes(b"exact original parent checkpoint")
    frozen_config.write_text("changed parent source\n")
    with pytest.raises(ValueError, match="Parent frozen source changed"):
        controller.verify_parent(tmp_path / config["batch"], config)


def test_build_command_uses_readonly_parent_and_frozen_inputs_separate_output_and_secret(tmp_path):
    config = configuration()
    batch = tmp_path / config["batch"]
    parent = tmp_path / config["parent"]["parent_batch"]
    out = batch / "single"
    token = tmp_path / "private-token"
    token.write_text("test-credential-must-not-appear-in-argv")
    name, command = controller.build_command(
        config, "single", "train", "sha256:pinned", {"uuid": "GPU-eight"}, out, batch, parent, token
    )
    assert name == f"{config['batch']}-single-train"
    assert command[:4] == ["docker", "--context", "lm-memory", "run"]
    assert command[command.index("--user") + 1] == "1015:1016"
    assert f"type=bind,source={parent / 'source'},target=/parent-frozen,readonly" in command
    assert f"type=bind,source={tmp_path},target=/project/results,readonly" in command
    assert f"type=bind,source={batch / 'input-mount/data'},target=/project/data,readonly" in command
    assert f"type=bind,source={out},target=/project/results/{config['batch']}/single" in command
    assert f"type=bind,source={token},target=/run/secrets/wandb,readonly" in command
    assert "HOME=/tmp/lm-home" in command
    assert "PYTHONPATH=/frozen/src" in command
    assert "test-credential-must-not-appear-in-argv" not in " ".join(command)
    assert "/frozen/scripts/run_ttcd_continual.py" in command
    assert not any("source/input-backup" in arg for arg in command)


@pytest.mark.parametrize("private_umask", [0o022, 0o077])
def test_freeze_copies_parent_inputs_and_panel_and_never_refreshes(
    tmp_path, monkeypatch, private_umask
):
    project, parent = tmp_path / "project", tmp_path / "parent"
    config = configuration()
    for relative in ("src", "scripts", "tests", "configs"):
        (project / relative).mkdir(parents=True)
    backup = parent / "source/input-backup"
    backup.mkdir(parents=True)
    for name in (*controller.INPUTS, "model-config.json"):
        (backup / name).write_bytes(f"immutable parent {name}".encode())
    (project / "src/example.py").write_text("original = True\n")
    (project / "tests/test_example.py").write_text("assert True\n")
    for name in controller.SOURCE_SCRIPTS:
        (project / "scripts" / name).write_text("pass\n")
    for name in ("pyproject.toml", "CONTRIBUTING.md"):
        (project / name).write_text("original project metadata\n")
    (project / "configs/experiment-tracking-defaults.json").write_text("{}")
    (project / config["continual"]["panel_source_path"]).write_text('{"panel": "original"}\n')
    make_checkpoint(parent, config)
    config_path = project / "configs/new.json"
    config_path.write_text(json.dumps(config))
    batch = tmp_path / "new-batch"
    batch.mkdir()
    monkeypatch.setattr(controller, "ROOT", project)
    old_umask = os.umask(private_umask)
    try:
        controller.freeze_source(batch, config_path, parent)
    finally:
        os.umask(old_umask)
    assert (batch / "source/input-backup").stat().st_mode & 0o777 == 0o755
    assert (batch / "source/configs/continual-panel.json").read_text() == '{"panel": "original"}\n'
    assert (batch / "source/tests/test_example.py").is_file()
    assert (batch / "source/scripts/run_ttcd_second_hop.py").is_file()
    assert (batch / "source/scripts/prepare_ttcd_scratch.py").is_file()
    contract = json.loads((batch / "source/controller-contract.json").read_text())
    assert contract["stages"] == list(controller.STAGES)
    assert not contract["automatic_retry"]
    assert not contract["gpu_sharing"]
    (project / "src/example.py").write_text("changed = True\n")
    (project / config["continual"]["panel_source_path"]).write_text('{"panel": "changed"}\n')
    controller.freeze_source(batch, config_path, parent)
    assert (batch / "source/src/example.py").read_text() == "original = True\n"
    assert (batch / "source/configs/continual-panel.json").read_text() == '{"panel": "original"}\n'
    config_path.write_text("{}")
    with pytest.raises(ValueError, match="Configuration changed"):
        controller.freeze_source(batch, config_path, parent)


@pytest.mark.parametrize("private_umask", [0o022, 0o077])
def test_input_overlay_is_exact_traversable_and_detects_tampering(tmp_path, private_umask):
    config = configuration()
    batch = tmp_path / config["batch"]
    backup = batch / "source/input-backup"
    backup.mkdir(parents=True)
    backup.chmod(0o700)
    originals = (*controller.INPUTS, "model-config.json")
    for name in originals:
        (backup / name).write_bytes(f"immutable {name}".encode())
    (batch / "source-manifest.json").write_text('{"original": true}\n')
    frozen = {name: controller.sha256(backup / name) for name in originals}
    old_umask = os.umask(private_umask)
    try:
        data_root = controller.prepare_input_mount(batch, config)
    finally:
        os.umask(old_umask)
    assert backup.stat().st_mode & 0o777 == 0o700
    for directory in (batch / "input-mount", data_root, data_root / "ttcd-second-hop-scratch-v1"):
        assert directory.stat().st_mode & 0o777 == 0o755
    assert frozen == {name: controller.sha256(backup / name) for name in originals}
    manifest = batch / "input-mount-manifest.json"
    before = manifest.read_bytes()
    controller.prepare_input_mount(batch, config)
    assert manifest.read_bytes() == before
    target = batch / "input-mount" / config["data_dir"] / "world.npz"
    target.write_bytes(b"changed overlay input")
    with pytest.raises(ValueError, match="Input mount bytes changed"):
        controller.prepare_input_mount(batch, config)
    assert target.read_bytes() == b"changed overlay input"
    assert frozen == {name: controller.sha256(backup / name) for name in originals}


@pytest.mark.parametrize("failure_stage, exit_code", [("preflight", 1), ("calibrate", 3)])
def test_pipeline_stops_failed_arm_without_retry_or_later_stages(
    tmp_path, monkeypatch, failure_stage, exit_code
):
    config = configuration()
    parent = tmp_path / config["parent"]["parent_batch"]
    make_checkpoint(parent, config)
    batch = tmp_path / config["batch"]
    batch.mkdir()
    launched = []

    def fake_run(command, **kwargs):
        launched.append(command[command.index("/frozen/scripts/run_ttcd_continual.py") + 1])
        return SimpleNamespace(returncode=0, stdout="new-container-id\n", stderr="")

    monkeypatch.setattr(controller, "inspect_container", lambda _: None)
    monkeypatch.setattr(controller, "prepare_input_mount", lambda *args: None)
    monkeypatch.setattr(controller, "admit_job", lambda *args: {"uuid": "GPU-eight"})
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    monkeypatch.setattr(controller.fcntl, "flock", lambda *args: None)
    monkeypatch.setattr(
        controller,
        "monitor_container",
        lambda *args: exit_code if args[0].endswith(failure_stage) else 0,
    )
    code = controller.run_arm(
        config,
        "single",
        list(controller.STAGES),
        "sha256:pinned",
        batch,
        parent,
        tmp_path / "no-token",
    )
    assert code == exit_code
    assert launched == list(controller.STAGES[: controller.STAGES.index(failure_stage) + 1])
    out = batch / "single"
    expected_state = "calibration_failed" if failure_stage == "calibrate" else "needs_attention"
    assert json.loads((out / "controller-state.json").read_text())["state"] == expected_state
    assert json.loads((out / f"exit-{failure_stage}.json").read_text())["returncode"] == exit_code
    assert json.loads((out / f"command-{failure_stage}.json").read_text())["argv"]
    assert not (out / "claim-evaluate.json").exists()
    assert not (batch / "stream").exists()


def test_shared_pipeline_releases_short_admission_lock_and_never_touches_original_lock(
    tmp_path, monkeypatch
):
    config = shared_configuration()
    parent = tmp_path / config["parent"]["parent_batch"]
    make_checkpoint(parent, config)
    batch = tmp_path / config["batch"]
    batch.mkdir()
    events = []

    def flock(handle, mode):
        events.append((Path(handle.name).name, mode))

    def launch(command, **kwargs):
        events.append(
            ("launch", command[command.index("/frozen/scripts/run_ttcd_continual.py") + 1])
        )
        return SimpleNamespace(returncode=0, stdout="new-id\n", stderr="")

    def monitor(*args):
        assert events[-1] == ("shared-admission-gpu-8.lock", controller.fcntl.LOCK_UN)
        events.append(("monitor", args[0]))
        return 0

    monkeypatch.setattr(controller, "inspect_container", lambda _: None)
    monkeypatch.setattr(controller, "prepare_input_mount", lambda *args: None)
    monkeypatch.setattr(controller, "admit_job", lambda *args: {"uuid": "GPU-eight"})
    monkeypatch.setattr(controller.subprocess, "run", launch)
    monkeypatch.setattr(controller.fcntl, "flock", flock)
    monkeypatch.setattr(controller, "monitor_container", monitor)
    code = controller.run_arm(
        config,
        "single",
        ["preflight", "train"],
        "sha256:pinned",
        batch,
        parent,
        tmp_path / "no-token",
    )
    assert code == 0
    assert not any(name == "gpu-8.lock" for name, _ in events)
    assert sum(name == "launch" for name, _ in events) == 2
    assert (
        sum(
            name == "shared-admission-gpu-8.lock" and mode == controller.fcntl.LOCK_UN
            for name, mode in events
        )
        == 2
    )


def test_timeout_stops_only_the_own_named_container(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        returncode = None

        def communicate(self, timeout):
            self.returncode = 0
            return "", ""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout):
            self.returncode = 0

    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(controller, "docker", lambda *args, **kwargs: calls.append(args))
    with (tmp_path / "train.log").open("w") as log:
        with pytest.raises(TimeoutError, match="own container"):
            controller.monitor_container("ttcd-cross-sample-v1-single-train", log, 0)
    assert calls == [("stop", "--time", "30", "ttcd-cross-sample-v1-single-train")]
    assert (
        json.loads((tmp_path / "hard-timeout.json").read_text())["action"] == "stop own container"
    )


def test_monitor_records_alive_and_resources_without_changing_container(tmp_path, monkeypatch):
    updates, calls = [], []

    class FakeProcess:
        returncode = None
        attempts = 0

        def communicate(self, timeout):
            self.attempts += 1
            if self.attempts == 1:
                raise subprocess.TimeoutExpired("docker wait", timeout)
            self.returncode = 0
            return "0\n", ""

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0

    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(controller, "inspect_container", lambda _: {"State": {"Running": True}})

    def stats(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='{"MemUsage": "1GiB / 48GiB", "CPUPerc": "120%"}\n')

    monkeypatch.setattr(controller, "docker", stats)
    with (tmp_path / "train.log").open("w") as log:
        code = controller.monitor_container("own-container", log, float("inf"), updates.append)
    assert code == 0
    assert updates[0]["container_running"]
    assert updates[0]["container_resources"]["CPUPerc"] == "120%"
    assert calls == [("stats", "--no-stream", "--format", "{{json .}}", "own-container")]
