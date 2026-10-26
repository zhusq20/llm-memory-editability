"""Contracts for immutable execution, exclusive GPUs and nonrepeated science."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cross_sample_controller", ROOT / "scripts/launch_ce_cross_sample.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def plan(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "INPUT_ROOT", tmp_path)
    parent, repository = tmp_path / "parent", tmp_path / "repository"
    parent.mkdir()
    repository.mkdir()
    (parent / "old.py").write_text("unchanged parent\n")
    manifest = {"files": {"old.py": controller.sha(parent / "old.py")}}
    controller.write(parent / "source-manifest.json", manifest)
    extras = [
        "scripts/launch_ce_cross_sample.py",
        "scripts/run_ce_cross_sample.py",
        "scripts/report_ce_cross_sample.py",
        f"configs/{controller.BATCH}.json",
        f"docs/development-artifacts/{controller.BATCH}/protocol.md",
        f"docs/development-artifacts/{controller.BATCH}/panel.json",
        "tests/test_ce_cross_sample_controller.py",
    ]
    for relative in extras:
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("new source\n")
    result = {
        "batch": controller.BATCH,
        "panel": extras[5],
        "architectures": dict.fromkeys(controller.ARCHITECTURES, {}),
        "budget": {
            "new_scientific_updates_total": 13440,
            "new_scientific_updates_per_architecture": 4480,
        },
        "execution": {
            "parent_source": str(parent),
            "parent_source_manifest_sha256": controller.sha(parent / "source-manifest.json"),
            "parent_source_file_count": 1,
            "extra_files": extras,
            "persistent_root": str(tmp_path / controller.BATCH),
            "image": "sha256:immutable",
            "timeout_seconds_per_architecture": 43200,
            "resources": {"preferred_gpu_indices": [6, 9]},
        },
    }
    return result, repository, tmp_path / controller.BATCH


def test_freeze_keeps_parent_and_adds_only_new_files(plan):
    recipe, repository, batch = plan
    source, manifest = controller.freeze(recipe, batch, repository=repository)
    original = json.loads((source / "source-manifest.json").read_text())
    assert (source / "old.py").read_text() == "unchanged parent\n"
    assert original["parent_file_count"] == 1
    assert set(original["files"]) == {"old.py", *recipe["execution"]["extra_files"]}
    assert controller.verify_source(source, manifest) == manifest
    assert controller.freeze(recipe, batch, repository=repository) == (source, manifest)


def test_frozen_source_cannot_follow_later_working_edits(plan):
    recipe, repository, batch = plan
    controller.freeze(recipe, batch, repository=repository)
    (repository / recipe["panel"]).write_text("changed scientific panel\n")
    with pytest.raises(ValueError, match="differs from already frozen"):
        controller.freeze(recipe, batch, repository=repository)


def test_modified_parent_is_rejected_before_copy(plan):
    recipe, repository, batch = plan
    (Path(recipe["execution"]["parent_source"]) / "old.py").write_text("changed\n")
    with pytest.raises(ValueError, match="parent changed"):
        controller.freeze(recipe, batch, repository=repository)


def test_no_new_file_can_replace_original_science(plan):
    recipe, repository, batch = plan
    recipe["execution"]["extra_files"].append("old.py")
    with pytest.raises(ValueError, match="must not replace"):
        controller.freeze(recipe, batch, repository=repository)


def test_manifest_rejects_unrecorded_files(plan):
    recipe, repository, batch = plan
    source, manifest = controller.freeze(recipe, batch, repository=repository)
    (source / "unrecorded.py").write_text("extra\n")
    with pytest.raises(ValueError, match="Unmanifested"):
        controller.verify_source(source, manifest)


def test_container_archive_is_root_owned_and_read_only(plan):
    recipe, repository, batch = plan
    source, _ = controller.freeze(recipe, batch, repository=repository)
    with tarfile.open(fileobj=io.BytesIO(controller.archive(source))) as archive:
        for member in archive:
            assert (member.uid, member.gid) == (0, 0)
            assert member.mode == (0o555 if member.isdir() else 0o444)
            assert Path(member.name).parts[0] == "frozen"
            assert not member.issym()


def test_output_is_direct_persistent_bind_and_source_is_read_only(plan, monkeypatch):
    recipe, _, batch = plan
    output = batch / "transformer/output"
    output.mkdir(parents=True)
    monkeypatch.setattr(controller, "private_output", lambda *args: output)
    name, argv = controller.container_arguments(
        recipe, batch, "manifest", "transformer", "evaluate", {"uuid": "GPU-id"}
    )
    assert name.endswith("transformer-evaluate")
    assert f"type=bind,source={controller.INPUT_ROOT},target=/persist" in argv
    assert not any(f"source={batch}" in value for value in argv)
    assert not any(value.startswith("type=volume") for value in argv)
    assert (
        f"type=bind,source={controller.INPUT_ROOT},target={controller.INPUT_ROOT},readonly" in argv
    )
    assert f"type=bind,source={controller.TOKEN},target=/run/secrets/wandb,readonly" in argv
    assert "--read-only" not in argv
    assert argv[argv.index("--user") + 1] == "1015:1016"
    assert argv[argv.index("--gpus") + 1] == "device=GPU-id"
    assert argv[-1] == "cuda:0"


def test_cloud_has_no_gpu_request_and_uses_cpu(plan, monkeypatch):
    recipe, _, batch = plan
    monkeypatch.setattr(controller, "private_output", lambda *args: batch / "loop/output")
    _, argv = controller.container_arguments(recipe, batch, "manifest", "loop", "cloud", None)
    assert "--gpus" not in argv
    assert argv[-1] == "cpu"


@pytest.mark.parametrize("stage", ["prepare", "evaluate", "audit", "cloud"])
def test_runner_interface_is_fixed(plan, stage):
    recipe, _, _ = plan
    command = controller.command_for(recipe, stage, "discoloop")
    assert command[:3] == ["python", "scripts/run_ce_cross_sample.py", stage]
    assert command[command.index("--architecture") + 1] == "discoloop"
    assert command[command.index("--root") + 1] == f"/persist/{controller.BATCH}/discoloop/output"


def test_existing_science_launch_without_exit_cannot_repeat(tmp_path):
    controller.write(tmp_path / "launch-evaluate.json", {"source_sha256": "expected"})
    with pytest.raises(FileExistsError, match="never repeat"):
        controller.completed_stage(tmp_path, "evaluate", "expected")


def test_failed_science_exit_cannot_repeat(tmp_path):
    controller.write(tmp_path / "launch-evaluate.json", {"source_sha256": "expected"})
    controller.write(tmp_path / "exit-evaluate.json", {"ExitCode": 1})
    with pytest.raises(FileExistsError, match="never repeat"):
        controller.completed_stage(tmp_path, "evaluate", "expected")


def test_completed_stage_requires_identical_source(tmp_path):
    controller.write(tmp_path / "launch-evaluate.json", {"source_sha256": "expected"})
    controller.write(tmp_path / "exit-evaluate.json", {"ExitCode": 0})
    assert controller.completed_stage(tmp_path, "evaluate", "expected")
    with pytest.raises(FileExistsError, match="never repeat"):
        controller.completed_stage(tmp_path, "evaluate", "changed")


def gpu(index, *, pids=None, used=0, free=81920, utilization=0):
    return {
        "index": index,
        "uuid": f"GPU-{index}",
        "compute_pids": pids or [],
        "used_mib": used,
        "free_mib": free,
        "utilization_percent": utilization,
    }


@pytest.mark.parametrize(
    "record", [gpu(4, pids=[100]), gpu(4, used=1025), gpu(4, free=10000), gpu(4, utilization=6)]
)
def test_busy_or_insufficient_gpu_is_never_admitted(plan, record):
    recipe, _, _ = plan
    assert not controller.idle(record, recipe)


def test_idle_gpu_double_checked_after_lock(plan, tmp_path, monkeypatch):
    recipe, _, _ = plan
    snapshots = iter([[gpu(6)], [gpu(6, pids=[1])]])
    monkeypatch.setattr(controller, "gpu_inventory", lambda: next(snapshots))
    assert controller.acquire_gpu(recipe) == (None, None)


def test_shared_gpu_lock_excludes_another_architecture(plan, monkeypatch):
    recipe, _, _ = plan
    monkeypatch.setattr(controller, "gpu_inventory", lambda: [gpu(99999)])
    first, lock = controller.acquire_gpu(recipe)
    try:
        assert first["index"] == 99999
        assert controller.acquire_gpu(recipe) == (None, None)
    finally:
        lock.close()


def test_gpu_timeout_is_total_across_stages(plan, monkeypatch):
    recipe, _, batch = plan
    recipe["execution"]["timeout_seconds_per_architecture"] = 100
    calls = []

    class Lock:
        def close(self):
            pass

    monkeypatch.setattr(controller, "acquire_gpu", lambda _: (gpu(4), Lock()))

    def fake_stage(*args, **kwargs):
        calls.append((args[5], kwargs.get("timeout"), kwargs.get("gpu")))
        return 20

    monkeypatch.setattr(controller, "run_stage", fake_stage)
    result = controller.execute_arm(recipe, batch, Path("source"), "manifest", "loop")
    assert [(stage, timeout) for stage, timeout, _ in calls] == [
        ("prepare", 100),
        ("evaluate", 80),
        ("audit", 60),
        ("cloud", None),
    ]
    assert calls[-1][2] is None
    assert result["scientific_updates"] == 4480


def test_failed_arm_is_preserved_without_following_stages(plan, monkeypatch):
    recipe, _, batch = plan
    calls = []

    class Lock:
        def close(self):
            pass

    monkeypatch.setattr(controller, "acquire_gpu", lambda _: (gpu(4), Lock()))

    def fake_stage(*args, **kwargs):
        calls.append(args[5])
        if args[5] == "evaluate":
            raise RuntimeError("failure retained")
        return 1

    monkeypatch.setattr(controller, "run_stage", fake_stage)
    with pytest.raises(RuntimeError, match="failure retained"):
        controller.execute_arm(recipe, batch, Path("source"), "manifest", "loop")
    assert calls == ["prepare", "evaluate"]
    state = json.loads((batch / "loop/controller-state.json").read_text())
    assert state["state"] == "needs_attention"
    assert state["automatic_retry"] is False
    assert not (batch / "loop/completion-manifest.json").exists()


@pytest.mark.parametrize("value", ["/absolute.py", "../escaped.py", "dir/../../escaped.py"])
def test_source_path_cannot_escape(value):
    with pytest.raises(ValueError, match="escaped"):
        controller.safe_relative(value)


def test_report_reads_batch_mount_and_writes_only_output(plan, monkeypatch):
    recipe, _, batch = plan
    output = batch / "report/output"
    monkeypatch.setattr(controller, "private_output", lambda *args: output)
    _, argv = controller.container_arguments(recipe, batch, "manifest", "report", "report", None)
    assert not any("target=/batch" in value for value in argv)
    assert f"type=bind,source={controller.INPUT_ROOT},target=/persist" in argv
    assert not any(f"source={batch}" in value for value in argv)
    assert argv[argv.index("--batch") + 1] == f"/persist/{controller.BATCH}"
    assert argv[argv.index("--out") + 1] == f"/persist/{controller.BATCH}/report/output"
    assert "--gpus" not in argv


def test_deployment_copies_source_before_start_without_read_only_rootfs(plan, monkeypatch):
    recipe, repository, batch = plan
    source, manifest = controller.freeze(recipe, batch, repository=repository)
    calls = []
    monkeypatch.setattr(controller, "private_output", lambda *args: batch / "contracts/output")

    def fake_docker(*arguments, **kwargs):
        calls.append((arguments, kwargs))
        if arguments[0] == "create":
            output = b"container-id\n"
        elif arguments[0] == "inspect":
            output = json.dumps({"Running": False, "ExitCode": 0}).encode()
        else:
            output = b""
        return subprocess.CompletedProcess(arguments, 0, stdout=output, stderr=b"")

    monkeypatch.setattr(controller, "docker", fake_docker)
    controller.run_stage(recipe, batch, source, manifest, "contracts", "contracts")
    operations = [arguments[0] for arguments, _ in calls]
    assert operations[:3] == ["create", "cp", "start"]
    creation = calls[0][0]
    assert "--read-only" not in creation
    assert creation[creation.index("--user") + 1] == "1015:1016"
    assert (
        f"type=bind,source={controller.INPUT_ROOT},target={controller.INPUT_ROOT},readonly"
        in creation
    )
    assert f"type=bind,source={controller.TOKEN},target=/run/secrets/wandb,readonly" in creation
    with tarfile.open(fileobj=io.BytesIO(calls[1][1]["input"])) as copied:
        for member in copied:
            assert (member.uid, member.gid) == (0, 0)
            assert member.mode == (0o555 if member.isdir() else 0o444)


def test_private_batch_uses_traversable_project_parent_mount(plan):
    recipe, _, batch = plan
    batch.mkdir(mode=0o700)
    _, arguments = controller.container_arguments(
        recipe, batch, "manifest", "loop", "evaluate", None
    )
    assert batch.stat().st_mode & 0o777 == 0o700
    assert (batch / "loop/output").stat().st_mode & 0o777 == 0o700
    assert f"type=bind,source={controller.INPUT_ROOT},target=/persist" in arguments
    assert not any("target=/out" in value or "target=/batch" in value for value in arguments)
    assert arguments[arguments.index("--root") + 1] == f"/persist/{controller.BATCH}/loop/output"


@pytest.mark.parametrize("arm", ["../escaped", "../../other", "/absolute", "unknown"])
def test_private_output_rejects_unapproved_destination(plan, arm):
    _, _, batch = plan
    with pytest.raises(ValueError, match="architecture escaped"):
        controller.private_output(batch, arm)
    with pytest.raises(ValueError, match="architecture escaped"):
        controller.command_for(plan[0], "evaluate", arm)


def test_private_output_rejects_symlink_escape(plan, tmp_path):
    _, _, batch = plan
    batch.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (batch / "loop").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink escaped"):
        controller.private_output(batch, "loop")


def test_runtime_batch_rejects_another_project_destination(plan, tmp_path):
    recipe, _, _ = plan
    recipe["execution"]["persistent_root"] = str(tmp_path / "another")
    with pytest.raises(ValueError, match="batch path escaped"):
        controller.runtime_batch(recipe)


def test_source_revision_preserves_the_original_frozen_evidence(plan):
    recipe, repository, batch = plan
    original, original_sha = controller.freeze(recipe, batch, repository=repository)
    original_receipt = (batch / "freeze-receipt.json").read_bytes()
    (repository / recipe["panel"]).write_text("deployment amendment\n")
    recipe["execution"]["source_revision"] = "source-v2"
    revision, revision_sha = controller.freeze(recipe, batch, repository=repository)
    assert revision == batch / "source-v2"
    assert revision_sha != original_sha
    assert controller.verify_source(original, original_sha) == original_sha
    assert (batch / "freeze-receipt.json").read_bytes() == original_receipt
    receipt = json.loads((batch / "freeze-receipt-source-v2.json").read_text())
    assert receipt["source_revision"] == "source-v2"
    assert receipt["source_sha256"] == revision_sha
    assert controller.freeze(recipe, batch, repository=repository) == (revision, revision_sha)


@pytest.mark.parametrize(
    "revision", ["../source-v2", "/tmp/source", "source/other", "other", "source-v0", "source-v02"]
)
def test_source_revision_cannot_escape_or_overwrite_arbitrary_directories(plan, revision):
    recipe, repository, batch = plan
    recipe["execution"]["source_revision"] = revision
    with pytest.raises(ValueError, match="source revision"):
        controller.freeze(recipe, batch, repository=repository)
