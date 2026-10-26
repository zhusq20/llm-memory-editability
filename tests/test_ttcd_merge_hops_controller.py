"""Frozen-parent reuse, independent queueing and evaluation-only merge/hop stages."""

import contextlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_hops_controller", ROOT / "scripts/launch_ttcd_merge_hops.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def plan():
    return json.loads((ROOT / f"configs/{controller.BATCH}.json").read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def material(tmp_path):
    settings = plan()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "persistent"
    root.mkdir()
    parent = root / controller.PARENT
    parent.mkdir()
    data = root / "original/input-mount/data"
    data.mkdir(parents=True)
    (data / "world.bin").write_bytes(b"unaltered graph input")
    inputs = {"data/world.bin": controller.sha256(data / "world.bin")}
    panel_bytes = b'{"fixed":"32-facts"}\n'
    settings["panel_sha256"] = controller.parent_controller.digest(panel_bytes)
    records = {}
    for arm in controller.ARCHITECTURES:
        spec = settings["architectures"][arm]
        source = parent / spec["source_directory"]
        source.mkdir(parents=True)
        write(
            source / spec["parent_config"],
            {
                "writer": {"steps": 4},
                "runtime": {"image": settings["runtime"]["image"]},
                "models": {"stream": {"step": 10000}},
            },
        )
        (source / "configs/continual-panel.json").write_bytes(panel_bytes)
        write(source / "configs/one-step.json", {"batch": controller.PARENT, "writer_steps": 1})
        (source / "parent.py").write_bytes(b"# unchanged one-step parent\n")
        manifest_path = parent / spec["source_manifest"]
        write(
            manifest_path,
            {
                "architecture": arm,
                "parent_config": spec["parent_config"],
                "parent_inputs": inputs,
                "data_directory": str(data),
                "files": controller.resources.source_hashes(source),
            },
        )
        records[arm] = controller.sha256(manifest_path)
    write(parent / "source-manifest.json", {"arms": records})
    for relative in controller.ADDITIONS:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# new merge and hop evaluation\n")
    config_path = workspace / "settings.json"
    write(config_path, settings)
    batch = root / controller.BATCH
    batch.mkdir()
    return batch, settings, workspace, config_path


@pytest.fixture
def frozen(material):
    batch, settings, workspace, config_path = material
    controller.freeze(batch, settings, config_path, workspace)
    return material


def test_authorized_methods_and_hop_roles_with_zero_training():
    settings = plan()
    controller.validate_plan(settings)
    assert settings["methods"] == ["continuous", "independent_merge"]
    assert settings["roles"] == ["first", "second"]
    assert settings["new_base_parameter_updates"] == 0
    assert controller.STAGES == ("preflight", "evaluate", "audit", "cloud-audit")
    assert "calibrate" not in controller.STAGES
    old = controller.parent_controller.ROOT / "configs/ttcd-one-step-second-hop-v1.json"
    old_cpus = json.loads(old.read_text())["runtime"]["cpu_sets"]
    assert not set(sum(settings["runtime"]["cpu_sets"].values(), [])).intersection(
        sum(old_cpus.values(), [])
    )


@pytest.mark.parametrize(
    "change", ["methods", "roles", "steps", "updates", "endpoint", "cpus", "image"]
)
def test_rejects_scope_and_resource_drift(change):
    settings = plan()
    if change == "methods":
        settings["methods"].append("orthogonal")
    elif change == "roles":
        settings["roles"] = ["second"]
    elif change == "steps":
        settings["writer_steps"] = 4
    elif change == "updates":
        settings["new_base_parameter_updates"] = 100
    elif change == "endpoint":
        settings["architectures"]["loop"]["evaluation_directory"] = "transformer/evaluation"
    elif change == "cpus":
        settings["runtime"]["cpu_sets"]["loop"] = [12, 13, 14, 15]
    else:
        settings["runtime"]["image"] = "latest"
    with pytest.raises(ValueError):
        controller.validate_plan(settings)


def test_freeze_preserves_all_parent_bytes_and_original_configuration(frozen):
    batch, settings, _, config_path = frozen
    for arm in controller.ARCHITECTURES:
        manifest = controller.verify_snapshot(batch, arm)
        assert not manifest["host_permissions_changed"]
        assert manifest["base_parameter_updates"] == 0
        source = batch / arm / "source"
        for relative, expected in manifest["parent_files"].items():
            assert controller.sha256(source / relative) == expected
        assert (source / "configs/merge-hops.json").read_bytes() == config_path.read_bytes()
        assert (
            json.loads((source / "configs/one-step.json").read_text())["batch"] == controller.PARENT
        )
        assert (
            json.loads((source / settings["architectures"][arm]["parent_config"]).read_text())[
                "writer"
            ]["steps"]
            == 4
        )
    with pytest.raises(FileExistsError):
        controller.freeze(batch, settings, config_path)


def test_parent_original_byte_mutation_is_rejected(material):
    batch, settings, workspace, config_path = material
    parent = batch.parent / controller.PARENT
    (parent / "loop/source/parent.py").write_bytes(b"mutated parent")
    with pytest.raises(ValueError, match="bytes changed"):
        controller.freeze(batch, settings, config_path, workspace)


def test_parent_manifest_identity_mutation_is_rejected(material):
    batch, settings, workspace, config_path = material
    write(batch.parent / controller.PARENT / "loop/source-manifest.json", {"files": {}})
    with pytest.raises(ValueError, match="parent index"):
        controller.freeze(batch, settings, config_path, workspace)


def test_parent_source_directory_symlink_is_rejected(material):
    batch, settings, workspace, config_path = material
    source = batch.parent / controller.PARENT / "loop/source"
    source.rename(source.with_name("saved"))
    source.symlink_to(source.with_name("saved"), target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        controller.freeze(batch, settings, config_path, workspace)


def test_new_overlay_must_not_replace_existing_parent_science(material):
    batch, settings, workspace, config_path = material
    parent = batch.parent / controller.PARENT
    source = parent / "transformer/source"
    relative = controller.ADDITIONS[0]
    target = source / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"different preexisting parent science")
    manifest_path = parent / "transformer/source-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"] = controller.resources.source_hashes(source)
    write(manifest_path, manifest)
    index = json.loads((parent / "source-manifest.json").read_text())
    index["arms"]["transformer"] = controller.sha256(manifest_path)
    write(parent / "source-manifest.json", index)
    with pytest.raises(ValueError, match="Cannot replace parent"):
        controller.freeze(batch, settings, config_path, workspace)


def parent_ready(frozen, arm):
    batch, settings, _, _ = frozen
    parent = batch.parent / controller.PARENT
    evaluation = parent / settings["architectures"][arm]["evaluation_directory"]
    paths = []
    for stage in ("audit", "cloud-audit"):
        result = evaluation / f"result-{stage}.json"
        exit_path = parent / f"{arm}/stages/{stage}/exit.json"
        write(result, {"passed": True, "accuracy": 0.0})
        write(exit_path, {"returncode": 0})
        paths.extend([result, exit_path])
    return paths


@pytest.mark.parametrize("arm", controller.ARCHITECTURES)
def test_waits_for_own_local_and_cloud_audit_without_accuracy_selection(frozen, arm):
    batch, settings, _, _ = frozen
    assert controller.parent_receipt(batch, settings, arm) is None
    paths = parent_ready(frozen, arm)
    receipt = controller.parent_receipt(batch, settings, arm)
    assert receipt["passed"] and receipt["capability_thresholds"] is None
    assert len(receipt["receipts"]) == 4
    paths[-1].unlink()
    assert controller.parent_receipt(batch, settings, arm) is None


@pytest.mark.parametrize("failed", ["audit", "cloud-audit"])
def test_failed_parent_audit_is_terminal(frozen, failed):
    batch, settings, _, _ = frozen
    parent_ready(frozen, "loop")
    write(
        batch.parent / controller.PARENT / f"loop/evaluation/result-{failed}.json",
        {"passed": False},
    )
    with pytest.raises(ValueError, match="validation failed"):
        controller.parent_receipt(batch, settings, "loop")


def test_failed_parent_exit_is_terminal(frozen):
    batch, settings, _, _ = frozen
    parent_ready(frozen, "loop")
    write(batch.parent / controller.PARENT / "loop/stages/cloud-audit/exit.json", {"returncode": 1})
    with pytest.raises(RuntimeError, match="unsuccessfully"):
        controller.parent_receipt(batch, settings, "loop")


def test_does_not_inspect_parent_pid_or_wait_for_other_architectures(frozen, monkeypatch):
    batch, settings, _, _ = frozen
    parent_ready(frozen, "loop")
    monkeypatch.setattr(
        controller.os, "kill", lambda *_: pytest.fail("PID inspection crosses namespace")
    )
    assert controller.parent_receipt(batch, settings, "loop")["passed"]
    assert controller.parent_receipt(batch, settings, "transformer") is None


def test_preparation_precedes_parent_wait_and_remaining_stages(monkeypatch, tmp_path):
    (tmp_path / "loop").mkdir()
    events = []
    monkeypatch.setattr(controller, "run_stage", lambda _p, _a, stage, *_: events.append(stage))
    monkeypatch.setattr(
        controller, "wait_parent", lambda *_: events.append("parent-wait") or {"passed": True}
    )
    result = controller.run_arm(plan(), "loop", tmp_path, Path("/private/token"))
    assert events == ["prepare", "parent-wait", "preflight", "evaluate", "audit", "cloud-audit"]
    assert result["completed_stages"] == ["prepare", *controller.STAGES]
    assert result["state"] == "complete"


def test_failed_preparation_never_waits_or_runs_gpu_stages(monkeypatch, tmp_path):
    (tmp_path / "loop").mkdir()

    def fail(*_):
        raise ValueError("graph contract failure")

    monkeypatch.setattr(controller, "run_stage", fail)
    monkeypatch.setattr(
        controller, "wait_parent", lambda *_: pytest.fail("Failed prepare reached parent wait")
    )
    with pytest.raises(ValueError, match="graph contract failure"):
        controller.run_arm(plan(), "loop", tmp_path, Path("/private/token"))
    assert (
        json.loads((tmp_path / "loop/controller-state.json").read_text())["state"]
        == "needs_attention"
    )


def test_preparation_resources_are_cpu_only(monkeypatch, tmp_path):
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([], 999999999))
    monkeypatch.setattr(
        controller.resources, "gpu_snapshot", lambda: pytest.fail("Prepare requested GPU")
    )
    held, row = controller.acquire_resources(plan(), "loop", "prepare", lambda _: None, tmp_path)
    assert row["device"] == "cpu" and "gpu" not in row
    held.close()


def test_argv_has_readonly_parent_states_and_input_mounts(frozen):
    batch, settings, _, _ = frozen
    _, command = controller.build_command(
        settings, "loop", "evaluate", batch, Path("/private/token"), {"gpu": 5, "uuid": "GPU-5"}
    )
    assert command[:4] == ["docker", "--context", "lm-memory", "create"]
    assert command[command.index("--parent-evaluation") + 1] == "/parent-evaluation"
    assert command[command.index("--parent-config") + 1] == "/frozen/configs/architecture-loop.json"
    assert command[command.index("--settings") + 1] == "/frozen/configs/merge-hops.json"
    assert command[command.index("--device") + 1] == "cuda:0"
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert any("target=/parent-evaluation,readonly" in value for value in mounts)
    assert any("target=/frozen/results,readonly" in value for value in mounts)
    assert any("target=/inputs/data,readonly" in value for value in mounts)
    assert any("target=/frozen/data,readonly" in value for value in mounts)
    assert all(value.endswith(",readonly") for value in mounts if "target=/output" not in value)
    assert not any("/source," in value for value in mounts)


def test_prepare_tolerates_parent_output_not_created_yet(frozen):
    batch, settings, _, _ = frozen
    _, command = controller.build_command(
        settings, "loop", "prepare", batch, Path("/private/token"), {"device": "cpu"}
    )
    assert "--gpus" not in command
    assert not any("target=/parent-evaluation" in value for value in command)
    assert command[command.index("--device") + 1] == "cpu"


def test_aggregate_mounts_comparison_readonly_and_uses_cpu(frozen):
    batch, settings, _, _ = frozen
    _, command = controller.build_command(
        settings, "transformer", "aggregate", batch, Path("/private/token"), {"device": "cpu"}
    )
    assert "--gpus" not in command
    assert command[command.index("--comparison-root") + 1] == "/comparison"
    assert command[command.index("--out") + 1] == "/aggregate"
    assert any("target=/comparison,readonly" in value for value in command)


def test_archive_copies_exact_frozen_bytes_with_root_readonly_ownership(frozen):
    batch, _, _, _ = frozen
    manifest = controller.verify_snapshot(batch, "loop")
    archive = controller.runtime_archive(batch, "loop")
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            assert member.uid == member.gid == 0
            assert member.mode == (0o555 if member.isdir() else 0o444)
        for relative, expected in manifest["files"].items():
            assert (
                controller.parent_controller.digest(bundle.extractfile(f"frozen/{relative}").read())
                == expected
            )


def test_failed_container_exit_is_preserved_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "runtime_archive", lambda *_: b"exact tar")
    monkeypatch.setattr(
        controller, "acquire_resources", lambda *_: (contextlib.ExitStack(), {"device": "cpu"})
    )
    monkeypatch.setattr(controller, "build_command", lambda *_: ("merge-test", ["ignored"]))
    monkeypatch.setattr(controller.resources, "inspect_container", lambda _: None)
    monkeypatch.setattr(
        controller, "launch_copied_runtime", lambda *_: SimpleNamespace(stdout="container-id")
    )
    monkeypatch.setattr(controller.resources, "monitor_container", lambda *_: 3)
    with pytest.raises(RuntimeError, match="exited 3"):
        controller.run_stage(
            plan(), "loop", "audit", tmp_path, Path("/private/token"), lambda _: None
        )
    assert json.loads((tmp_path / "loop/stages/audit/exit.json").read_text())["returncode"] == 3
    with pytest.raises(FileExistsError):
        controller.run_stage(
            plan(), "loop", "audit", tmp_path, Path("/private/token"), lambda _: None
        )
