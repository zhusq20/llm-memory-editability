"""The mount repair changes orchestration only and retains failed evidence."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mount_recovery", PROJECT / "scripts/resume_path_overlap_mount.py"
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_worker_command_mounts_complete_provenance_bases_readonly(tmp_path):
    origin, previous, attempt = tmp_path / "v3", tmp_path / "v2", tmp_path / "v3/attempts/fix"
    recovery.write(origin / "selection-adoption.json", {"source_base": str(previous)})
    image = "sha256:fixed"

    def command(base, run, stage, gpu, cpus):
        return "lm-z1-v3-parent-select", [
            "docker",
            "--context",
            "lm-memory",
            "run",
            "--name",
            "lm-z1-v3-parent-select",
            "--network",
            "none",
            "--mount",
            f"type=bind,source={attempt},target={attempt}",
            image,
            "python",
            str(origin / "frozen/source/scripts/execute_path_overlap.py"),
            "worker",
            "--base",
            str(attempt),
        ]

    def source(base):
        return Path(base) / "frozen/source"

    def inputs(base):
        return Path(base) / "inputs/grokking-reproduction-v1"

    legacy = SimpleNamespace(
        source=source,
        input_root=inputs,
        container_command=command,
        write=recovery.write,
        IMAGE=image,
        resource_check=lambda *args, **kwargs: {},
    )
    wrapped = recovery.install_wrappers(legacy, origin, attempt)
    name, argv = wrapped.container_command(attempt, "parent", "select", 2, list(range(8)))
    assert name == "lm-z1-v3-mountfix-parent-select"
    assert f"type=bind,source={previous},target={previous},readonly" in argv
    assert f"type=bind,source={origin},target={origin},readonly" in argv
    assert argv.index(f"type=bind,source={origin},target={origin},readonly") < argv.index(
        f"type=bind,source={attempt},target={attempt}"
    )
    assert str(attempt / "orchestration/resume_path_overlap_mount.py") in argv
    assert argv[-2:] == ["--origin-base", str(origin)]
    assert argv[argv.index("--network") + 1] == "none"
    assert wrapped.source(attempt) == origin / "frozen/source"
    assert wrapped.input_root(attempt) == origin / "inputs/grokking-reproduction-v1"
    assert wrapped.source(previous) == previous / "frozen/source"


def test_tracking_uses_scientific_batch_with_separate_attempt_identity(tmp_path):
    origin, attempt = tmp_path / "v3", tmp_path / "v3/attempts/fix"
    recovery.write(origin / "selection-adoption.json", {"source_base": str(tmp_path / "v2")})
    legacy = SimpleNamespace(
        source=lambda base: Path(base) / "frozen/source",
        input_root=lambda base: Path(base) / "inputs",
        container_command=lambda *args: None,
        write=recovery.write,
        contract=lambda base: {"batch": "zero-training-path-overlap-v3"},
        resource_check=lambda *args, **kwargs: {},
    )
    wrapped = recovery.install_wrappers(legacy, origin, attempt)
    value = {
        "batch": "zero-training-path-overlap-v3",
        "spec": {"parent_run": "parent", "optimizer_updates": 0},
    }
    wrapped.write(attempt / "runs/parent/run.json", value)
    recorded = recovery.read(attempt / "runs/parent/run.json")
    assert recorded["batch"] == "zero-training-path-overlap-v3"
    assert recorded["tracking_group"] == "zero-training-path-overlap-v3"
    assert recorded["spec"]["execution_attempt"] == "source-base-mount-v1"
    assert recorded["spec"]["parent_run"] == "parent"
    assert "execution_attempt" not in value["spec"]


def test_prepare_keeps_scientific_freeze_and_failure_receipts(tmp_path, monkeypatch):
    origin, attempt = tmp_path / "v3", tmp_path / "v3/execution-attempts/source-base-mount-v1"
    frozen = origin / "frozen/freeze.json"
    recovery.write(frozen, {"original": "scientific-freeze"})
    recovery.write(origin / "input-receipt.json", {"original": "input"})
    (origin / "frozen/source").mkdir()
    (origin / "inputs/grokking-reproduction-v1").mkdir(parents=True)
    (origin / "validation").mkdir()
    controller = origin / "frozen/source/scripts/execute_path_overlap.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("# immutable frozen controller\n")
    recovery.write(
        origin / "controller-state.json", {"state": "finished_with_failures", "active": []}
    )
    recovery.write(origin / "runs/parent/failure.json", {"exit_code": 1, "cause": "missing mount"})
    recovery.write(
        origin / "tracking-completion.json", {"passed": True, "failure_records_preserved": True}
    )
    recovery.write(origin / "tracking-wandb/parent/state.json", {"run_id": "old-wandb-id"})
    nodes = []
    for node in range(111):
        path = f"selection-stage/parent/{node:07d}/selection.json"
        recovery.write(
            origin / path, {"complete": True, "node": node, "contract_sha256": "original"}
        )
        nodes.append({"run": "parent", "step": node, "destination": path})
    adoption = {"source_base": str(tmp_path / "v2"), "nodes": nodes}
    recovery.write(origin / "selection-adoption.json", adoption)
    before = recovery.sha(frozen)
    failure_before = (origin / "runs/parent/failure.json").read_bytes()
    progress = []
    legacy = SimpleNamespace(
        verify_freeze=lambda *args: None,
        contract=lambda base: {
            "batch": "zero-training-path-overlap-v3",
            "source": {"runs": ["parent"], "nodes": list(range(112))},
        },
        record_progress=lambda *args, **kwargs: progress.append(args[3]),
        utc=lambda: "time",
    )
    monkeypatch.setattr(recovery, "load_controller", lambda *args: legacy)
    monkeypatch.setattr(recovery, "original_attempt_evidence", lambda *args: adoption)
    monkeypatch.setattr(recovery, "install_wrappers", lambda *args: legacy)
    args = argparse.Namespace(origin_base=origin, attempt_base=attempt)
    recovery.prepare(args)
    assert recovery.sha(frozen) == before
    assert recovery.sha(attempt / "frozen/freeze.json") == before
    assert (origin / "runs/parent/failure.json").read_bytes() == failure_before
    assert not (attempt / "runs/parent/failure.json").exists()
    snapshot = origin / "scheduling-intervention/source-base-mount-v1/original-attempt"
    assert (snapshot / "runs/parent/failure.json").read_bytes() == failure_before
    assert recovery.read(snapshot / "tracking-wandb/parent/state.json")["run_id"] == "old-wandb-id"
    assert len(progress) == 111
    assert len(list((attempt / "selection-stage").glob("*/*/selection.json"))) == 111
    assert not (attempt / "selection-stage/parent/0000111").exists()
    assert (
        recovery.read(attempt / "recovery-receipt.json")["failed_attempt_new_selection_nodes"] == 0
    )
    assert (attempt / "orchestration/resume_path_overlap_mount.py").is_file()
    with pytest.raises(RuntimeError, match="already exists"):
        recovery.prepare(args)


def test_cpu_mapping_is_fixed_at_16_to_63_and_checks_existing_ownership(tmp_path, monkeypatch):
    origin, attempt = tmp_path / "v3", tmp_path / "v3/attempts/fix"
    recovery.write(origin / "selection-adoption.json", {"source_base": str(tmp_path / "v2")})
    checked = []
    scientific_worker = object()

    def resources(base, gpus, reserved_cpus=None):
        checked.append((gpus, reserved_cpus))
        return {"gpus": gpus, "cpus": [[99]]}

    legacy = SimpleNamespace(
        source=lambda base: Path(base) / "frozen/source",
        input_root=lambda base: Path(base) / "inputs",
        container_command=lambda *args: None,
        write=recovery.write,
        resource_check=resources,
        worker=scientific_worker,
    )
    monkeypatch.setattr(recovery.os, "sched_getaffinity", lambda pid: set(range(80)))
    wrapped = recovery.install_wrappers(legacy, origin, attempt)
    value = wrapped.resource_check(attempt, list(range(2, 8)))
    assert value["cpus"] == [list(range(16 + 8 * index, 24 + 8 * index)) for index in range(6)]
    assert checked == [(list(range(2, 8)), list(range(16, 64)))]
    assert wrapped.worker is scientific_worker
    with pytest.raises(RuntimeError, match="differs from the fixed"):
        wrapped.resource_check(attempt, [2], reserved_cpus=list(range(32, 40)))
