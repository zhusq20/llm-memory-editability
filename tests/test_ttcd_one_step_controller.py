"""Evaluation-only submission, immutable parent sources and exclusive admission."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "one_step_controller", ROOT / "scripts/launch_ttcd_one_step.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def plan():
    return json.loads((ROOT / f"configs/{controller.BATCH}.json").read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen(tmp_path):
    settings = plan()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "persistent"
    root.mkdir()
    batch = root / controller.BATCH
    batch.mkdir()
    data_parent = root / "ttcd-successful-base-v1"
    input_file = data_parent / "input-mount/data/example.bin"
    input_file.parent.mkdir(parents=True)
    input_file.write_bytes(b"immutable original input")
    inputs = {"data/example.bin": controller.sha256(input_file)}
    panel_bytes = b'{"fixed": "facts"}\n'
    settings["panel_sha256"] = controller.digest(panel_bytes)
    for arm in controller.ARCHITECTURES:
        spec = settings["architectures"][arm]
        source = root / spec["parent_batch"] / spec["source_directory"]
        source.mkdir(parents=True, exist_ok=True)
        parent_config = {
            "writer": {"steps": 4},
            "runtime": {"image": settings["runtime"]["image"]},
            "models": {
                "stream": {
                    "step": 10000,
                    "checkpoint": (
                        f"results/{spec['parent_batch']}/{spec['endpoint_directory']}"
                        "/training/parent.pt"
                    ),
                }
            },
        }
        write(source / spec["parent_config"], parent_config)
        (source / "configs/continual-panel.json").write_bytes(panel_bytes)
        (source / "parent.py").write_bytes(b"# exact parent bytes\n")
    for parent_batch, source_name, manifest_name in (
        ("ttcd-architecture-comparison-v1", "source", "source-manifest.json"),
        ("ttcd-successful-base-v1", "source-v2", "source-v2-manifest.json"),
    ):
        parent = root / parent_batch
        write(
            parent / manifest_name,
            {
                "files": controller.resources.source_hashes(parent / source_name),
                "inputs" if source_name == "source-v2" else "parent_immutable_inputs": inputs,
            },
        )
    for relative in controller.ADDITIONS:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# new evaluation-only source\n")
    config_path = workspace / "settings.json"
    write(config_path, settings)
    controller.freeze(batch, settings, config_path, workspace)
    return batch, settings, workspace, config_path


def test_fixed_one_step_matrix_contains_no_training_stage():
    controller.validate_plan(plan())
    assert controller.STAGES == ("preflight", "calibrate", "evaluate", "audit", "cloud-audit")
    assert not any("train" in stage for stage in controller.STAGES)


@pytest.mark.parametrize("change", ["steps", "training", "endpoint", "cpus", "image", "gpus"])
def test_rejects_scope_or_resource_drift(change):
    settings = plan()
    if change == "steps":
        settings["writer_steps"] = 4
    elif change == "training":
        settings["new_base_parameter_updates"] = 1
    elif change == "endpoint":
        settings["architectures"]["loop"]["endpoint_directory"] = "loop/common"
    elif change == "cpus":
        settings["runtime"]["cpu_sets"]["loop"] = [12, 13, 14, 15]
    elif change == "image":
        settings["runtime"]["image"] = "latest"
    else:
        settings["runtime"]["gpu_preferences"]["loop"] = [8, 9]
    with pytest.raises(ValueError):
        controller.validate_plan(settings)


def test_freezes_each_complete_parent_manifest_and_new_settings(frozen):
    batch, settings, _, config_path = frozen
    for arm in controller.ARCHITECTURES:
        snapshot = controller.verify_snapshot(batch, arm)
        assert snapshot["base_parameter_updates"] == 0
        assert not snapshot["host_permissions_changed"]
        assert (
            batch / arm / "source/configs/one-step.json"
        ).read_bytes() == config_path.read_bytes()
        for relative, expected in snapshot["parent_files"].items():
            assert controller.sha256(batch / arm / "source" / relative) == expected
        assert snapshot["parent_config"] == settings["architectures"][arm]["parent_config"]
    with pytest.raises(FileExistsError):
        controller.freeze(batch, settings, config_path)


def test_changed_original_byte_is_rejected(frozen):
    batch, _, _, _ = frozen
    (batch / "loop/source/parent.py").write_bytes(b"altered")
    with pytest.raises(ValueError, match="bytes changed"):
        controller.verify_snapshot(batch, "loop")


def test_source_directory_symlink_is_rejected(frozen):
    batch, _, _, _ = frozen
    source = batch / "loop/source"
    source.rename(batch / "loop/original-source")
    source.symlink_to(batch / "loop/original-source", target_is_directory=True)
    with pytest.raises(ValueError, match="redirect"):
        controller.verify_snapshot(batch, "loop")


def test_intermediate_directory_symlink_is_rejected(tmp_path):
    (tmp_path / "outside").mkdir()
    (tmp_path / "inside").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        controller.safe_path(tmp_path, "inside/file.py")
    with pytest.raises(ValueError):
        controller.safe_path(tmp_path, "../outside")


def test_runtime_archive_is_exact_root_owned_and_readonly(frozen):
    import tarfile

    batch, _, _, _ = frozen
    source = batch / "discoloop/source"
    original_modes = {path: path.stat().st_mode for path in source.rglob("*")}
    archive = controller.runtime_archive(batch, "discoloop")
    with tarfile.open(fileobj=io.BytesIO(archive)) as content:
        for member in content.getmembers():
            assert member.uid == member.gid == 0
            assert member.mode == (0o555 if member.isdir() else 0o444)
            assert not member.issym() and not member.islnk()
        snapshot = controller.verify_snapshot(batch, "discoloop")
        for relative, expected in snapshot["files"].items():
            assert controller.digest(content.extractfile(f"frozen/{relative}").read()) == expected
    assert all(path.stat().st_mode == mode for path, mode in original_modes.items())


def parent_endpoint(frozen, arm):
    batch, settings, _, _ = frozen
    spec = settings["architectures"][arm]
    parent = batch.parent / spec["parent_batch"]
    endpoint = parent / spec["endpoint_directory"]
    checkpoint = endpoint / "training/parent.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"independently audited stream10000 terminal")
    write(
        endpoint / "training/audit.json",
        {
            "passed": True,
            "phase": "stream",
            "checkpoint_sha256": controller.sha256(checkpoint),
        },
    )
    if arm == "discoloop":
        exits = [endpoint / f"exit-{stage}.json" for stage in ("train", "train-audit")]
        code_field = "returncode"
    else:
        exits = [
            parent / arm / "stages" / f"stream-{stage}" / "exit.json"
            for stage in ("train", "train-audit")
        ]
        code_field = "code"
    for path in exits:
        write(path, {code_field: 0})
    return endpoint, exits, code_field


@pytest.mark.parametrize("arm", controller.ARCHITECTURES)
def test_parent_readiness_uses_actual_exit_layout_and_does_not_wait_for_evaluation(frozen, arm):
    batch, settings, _, _ = frozen
    assert controller.endpoint_receipt(batch, settings, arm) is None
    endpoint, _, _ = parent_endpoint(frozen, arm)
    receipt = controller.endpoint_receipt(batch, settings, arm)
    assert receipt["passed"]
    assert receipt["checkpoint_sha256"] == controller.sha256(endpoint / "training/parent.pt")
    assert not (endpoint / "result-evaluate.json").exists()


@pytest.mark.parametrize("arm", controller.ARCHITECTURES)
def test_parent_failed_exit_stops_without_retry(frozen, arm):
    batch, settings, _, _ = frozen
    _, exits, code_field = parent_endpoint(frozen, arm)
    write(exits[-1], {code_field: 1})
    with pytest.raises(RuntimeError, match="Parent stage failed"):
        controller.endpoint_receipt(batch, settings, arm)


def test_parent_audit_must_bind_exact_checkpoint(frozen):
    batch, settings, _, _ = frozen
    endpoint, _, _ = parent_endpoint(frozen, "loop")
    (endpoint / "training/parent.pt").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="exact checkpoint"):
        controller.endpoint_receipt(batch, settings, "loop")


def test_later_parent_evaluation_failure_does_not_block_audited_endpoint(frozen):
    batch, settings, _, _ = frozen
    parent_endpoint(frozen, "loop")
    write(
        batch.parent / "ttcd-architecture-comparison-v1/loop/controller-state.json",
        {
            "state": "needs_attention",
            "stage": "stream-evaluate",
        },
    )
    assert controller.endpoint_receipt(batch, settings, "loop")["passed"]


def test_waiting_parent_training_failure_is_terminal(frozen):
    batch, settings, _, _ = frozen
    write(
        batch.parent / "ttcd-architecture-comparison-v1/loop/controller-state.json",
        {
            "state": "needs_attention",
            "stage": "common-train",
        },
    )
    with pytest.raises(RuntimeError, match="needs attention"):
        controller.endpoint_receipt(batch, settings, "loop")


def row(gpu):
    return {
        "uuid": f"GPU-{gpu}",
        "total_mib": 100000,
        "free_mib": 99000,
        "memory": 0,
        "utilization": 0,
    }


def test_admission_skips_occupied_gpu_and_holds_exclusive_lock(monkeypatch, tmp_path):
    import fcntl

    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([], 999999999))
    monkeypatch.setattr(
        controller.resources,
        "gpu_snapshot",
        lambda: ({0: row(0), 1: row(1)}, [{"uuid": "GPU-0", "pid": 123}]),
    )
    held, admitted = controller.acquire_resources(
        plan(), "transformer", "evaluate", lambda _: None, tmp_path
    )
    assert admitted["gpu"] == 1
    assert not admitted["gpu_sharing"]
    with (tmp_path / "gpu-1.lock").open("a") as other:
        with pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    held.close()


def test_cloud_audit_never_requests_or_inspects_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([], 999999999))
    monkeypatch.setattr(
        controller.resources, "gpu_snapshot", lambda: pytest.fail("CPU cloud audit touched GPU")
    )
    held, admitted = controller.acquire_resources(
        plan(), "loop", "cloud-audit", lambda _: None, tmp_path
    )
    assert admitted["device"] == "cpu"
    assert "gpu" not in admitted
    held.close()


def test_container_argv_uses_readonly_parent_inputs_and_separate_output(frozen):
    batch, settings, _, _ = frozen
    _, command = controller.build_command(
        settings, "loop", "evaluate", batch, Path("/private/token"), {"gpu": 1, **row(1)}
    )
    assert command[:4] == ["docker", "--context", "lm-memory", "create"]
    assert "run_ttcd_one_step.py" in " ".join(command)
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--parent-config") + 1] == "/frozen/configs/architecture-loop.json"
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert all(mount.endswith(",readonly") for mount in mounts if "target=/output" not in mount)
    assert len([mount for mount in mounts if "target=/output" in mount]) == 1
    assert not any("/source," in mount for mount in mounts)
    assert any("target=/frozen/results,readonly" in mount for mount in mounts)
    assert any("target=/frozen/data,readonly" in mount for mount in mounts)
    assert any("target=/inputs/data,readonly" in mount for mount in mounts)


def test_aggregate_container_has_no_gpu_and_readonly_comparison(frozen):
    batch, settings, _, _ = frozen
    _, command = controller.build_command(
        settings, "transformer", "aggregate", batch, Path("/private/token"), {"device": "cpu"}
    )
    assert "--gpus" not in command
    assert command[command.index("--comparison-root") + 1] == "/comparison"
    assert command[command.index("--out") + 1] == "/aggregate"
    assert any("target=/comparison,readonly" in arg for arg in command)


@pytest.mark.parametrize("copy_code", [0, 1])
def test_create_copy_start_order_and_copy_failure_preservation(monkeypatch, tmp_path, copy_code):
    calls = []

    def docker(*args):
        calls.append(args[0])
        return SimpleNamespace(stdout="container-id\n", returncode=0)

    def copy_runner(command, **kwargs):
        calls.append("copy")
        assert command[-1] == "test-container:/"
        assert kwargs["input"] == b"exact archive"
        return SimpleNamespace(returncode=copy_code, stderr=b"copy failure" if copy_code else b"")

    monkeypatch.setattr(controller.resources, "docker", docker)
    command = [*controller.resources.DOCKER, "create", "--name", "test-container"]
    if copy_code:
        with pytest.raises(RuntimeError, match="copy failed"):
            controller.launch_copied_runtime(
                command, b"exact archive", tmp_path, copy_runner=copy_runner
            )
    else:
        controller.launch_copied_runtime(
            command, b"exact archive", tmp_path, copy_runner=copy_runner
        )
    assert calls == (["create", "copy"] if copy_code else ["create", "copy", "start"])
    receipt = json.loads((tmp_path / "transport.json").read_text())
    assert receipt["state"] == ("copy_failed" if copy_code else "started")
    assert not receipt["host_permissions_changed"]


def test_arm_stops_at_failed_stage_without_retry(monkeypatch, tmp_path):
    (tmp_path / "loop").mkdir()
    called = []
    monkeypatch.setattr(controller, "wait_endpoint", lambda *_: {"passed": True})

    def run_stage(_plan, _arm, stage, *_):
        called.append(stage)
        if stage == "calibrate":
            raise ValueError("calibration failed")

    monkeypatch.setattr(controller, "run_stage", run_stage)
    with pytest.raises(ValueError, match="calibration failed"):
        controller.run_arm(plan(), "loop", tmp_path, Path("/private/token"))
    assert called == ["preflight", "calibrate"]
    state = json.loads((tmp_path / "loop/controller-state.json").read_text())
    assert state["state"] == "needs_attention"
    assert state["completed_stages"] == ["preflight"]


def test_nonzero_container_exit_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "runtime_archive", lambda *_: b"archive")
    monkeypatch.setattr(
        controller, "acquire_resources", lambda *_: (contextlib.ExitStack(), {"device": "cpu"})
    )
    monkeypatch.setattr(controller, "build_command", lambda *_: ("test-container", ["ignored"]))
    monkeypatch.setattr(controller.resources, "inspect_container", lambda _: None)
    monkeypatch.setattr(
        controller, "launch_copied_runtime", lambda *_: SimpleNamespace(stdout="id")
    )
    monkeypatch.setattr(controller.resources, "monitor_container", lambda *_: 2)
    with pytest.raises(RuntimeError, match="exited 2"):
        controller.run_stage(
            plan(), "loop", "audit", tmp_path, Path("/private/token"), lambda _: None
        )
    exit_path = tmp_path / "loop/stages/audit/exit.json"
    assert json.loads(exit_path.read_text())["returncode"] == 2
    with pytest.raises(FileExistsError):
        controller.run_stage(
            plan(), "loop", "audit", tmp_path, Path("/private/token"), lambda _: None
        )
