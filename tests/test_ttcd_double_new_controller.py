"""Supplementary execution must preserve parent identities and add no training."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ttcd_double_new_controller", ROOT / "scripts/launch_ttcd_double_new_eval.py"
)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def settings():
    return {
        "batch": controller.BATCH,
        "parent_batch": controller.PARENT,
        "pair_seed": 20261015,
        "pairs_per_type": 4,
        "rotations": 32,
        "primary_t": 32,
        "runtime": {
            "persistent_root": "/ossfs/workspace/llm-memory-editability",
            "image": "sha256:" + "9" * 64,
            "cpu_sets": {"single": list(range(32, 36)), "stream": list(range(36, 40))},
            "memory_gib": 96,
            "cpu_threads": 4,
            "timeout_hours": 24,
            "wait_timeout_hours": 24,
            "minimum_persistent_free_gib": 200,
        },
    }


def make_parent(tmp_path):
    parent = tmp_path / controller.PARENT
    source = parent / "source-v2"
    base = {
        "batch": controller.PARENT,
        "runtime": {"image": settings()["runtime"]["image"]},
        "common": {
            "gate_receipt": f"results/{controller.PARENT}/common/gate.json",
            "checkpoint": f"results/{controller.PARENT}/common/training/parent.pt",
        },
        "models": {
            arm: {"checkpoint": f"results/{controller.PARENT}/{arm}/training/parent.pt"}
            for arm in controller.ARMS
        },
    }
    for name, contents in {
        "configs/experiment.json": json.dumps(base),
        "src/llm_memory_editability/ttcd_verified_base.py": "original_science = True\n",
        "scripts/launch_discoloop_paired.py": "RESOURCE_HELPERS = True\n",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    files = {
        str(path.relative_to(source)): controller.sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    data = parent / "input-mount/data/ttcd-successful-base-v1/world.npz"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"original immutable world")
    manifest = {
        "files": files,
        "inputs": {"data/ttcd-successful-base-v1/world.npz": controller.sha256(data)},
        "config_sha256": controller.sha256(source / "configs/experiment.json"),
    }
    (parent / "source-v2-manifest.json").write_text(json.dumps(manifest))
    cache = source / "scripts/__pycache__/unregistered.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"not part of frozen science")
    return parent, base, manifest


def make_project(tmp_path):
    project = tmp_path / "workspace"
    for relative in controller.OVERLAYS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"new supplementary file {relative}\n")
    # A changed live copy must never replace the parent's frozen scientific code.
    old = project / "src/llm_memory_editability/ttcd_verified_base.py"
    old.write_text("wrong_live_science = True\n")
    path = project / "configs/ttcd-double-new-eval-v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(settings()))
    return project, path


def complete_parent(parent, base, arm):
    checkpoint = controller.result_path(parent, base["models"][arm]["checkpoint"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"original completed arm weights")
    common = controller.result_path(parent, base["common"]["checkpoint"])
    common.parent.mkdir(parents=True, exist_ok=True)
    common.write_bytes(b"original complete common weights")
    gate = controller.result_path(parent, base["common"]["gate_receipt"])
    gate.write_text(json.dumps({"passed": True, "checkpoint_sha256": controller.sha256(common)}))
    out = parent / arm
    for stage in controller.PARENT_STAGES:
        (out / f"exit-{stage}.json").write_text('{"returncode": 0}')
    (out / "audit.json").write_text('{"passed": true}')
    (out / "result-cloud-audit.json").write_text('{"passed": true}')


def test_default_plan_never_launches_and_contains_no_training_stage(tmp_path, monkeypatch, capsys):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings()))
    monkeypatch.setattr(sys, "argv", ["launch_ttcd_double_new_eval.py", "--config", str(path)])

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only planning cannot touch runtime or frozen outputs")

    monkeypatch.setattr(controller.subprocess, "run", forbidden)
    monkeypatch.setattr(controller, "persistent_batch", forbidden)
    controller.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan["executed"] and not plan["gpu_sharing"] and not plan["automatic_retry"]
    assert plan["new_base_parameter_updates"] == 0
    assert plan["prepare_device"] == "cpu"
    assert plan["stages"] == ["preflight", "evaluate", "audit", "cloud-audit"]
    assert plan["parent_local_and_cloud_completion_required"]
    assert plan["all_parent_arms_completion_required"]
    assert "train" not in controller.STAGES


@pytest.mark.parametrize("change", ["parent", "pairs", "seed", "cpu", "sharing", "space"])
def test_settings_reject_scope_and_resource_changes(change):
    config = settings()
    if change == "parent":
        config["parent_batch"] = "another-parent"
    elif change == "pairs":
        config["pairs_per_type"] = 8
    elif change == "seed":
        config["pair_seed"] = 999
    elif change == "cpu":
        config["runtime"]["cpu_sets"]["single"] = [24, 25, 26, 27]
    elif change == "sharing":
        config["runtime"]["shared_gpu_allowlist"] = ["parent-common"]
    else:
        config["runtime"]["minimum_persistent_free_gib"] = 100
    with pytest.raises(ValueError):
        controller.validate_settings(config)


def test_frozen_helper_import_cannot_write_parent_bytecode(tmp_path):
    module = tmp_path / "frozen_helpers.py"
    module.write_text("RESOURCE_HELPERS = True\n")
    previous = sys.dont_write_bytecode
    assert controller.load_module(module, "test_double_new_frozen_helper").RESOURCE_HELPERS
    assert sys.dont_write_bytecode == previous
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_freeze_copies_exact_parent_then_only_new_supplementary_files(tmp_path, monkeypatch, umask):
    parent, _, original = make_parent(tmp_path)
    project, path = make_project(tmp_path)
    batch = tmp_path / controller.BATCH
    old_umask = os.umask(umask)
    try:
        batch.mkdir()
        monkeypatch.setattr(controller, "ROOT", project)
        frozen = controller.freeze_source(batch, parent, path)
    finally:
        os.umask(old_umask)
    expected = (
        set(original["files"])
        | set(controller.OVERLAYS)
        | {
            "configs/double-new-eval.json",
            f"configs/{controller.BATCH}.json",
        }
    )
    assert set(frozen["files"]) == expected
    assert "docs/development-artifacts/ttcd-double-new-eval-v1/verify_cuda.py" in expected
    assert frozen["protocol"]["sha256"] == controller.sha256(batch / "source" / controller.PROTOCOL)
    assert frozen["new_base_parameter_updates"] == 0
    assert frozen["parent_config_sha256"] == original["config_sha256"]
    for relative, digest in original["files"].items():
        assert controller.sha256(batch / "source" / relative) == digest
    assert not (batch / "source/scripts/__pycache__").exists()
    assert batch.stat().st_mode & 0o777 == 0o755
    assert (batch / "source").stat().st_mode & 0o777 == 0o755
    assert (batch / "source/configs/experiment.json").read_bytes() == (
        parent / "source-v2/configs/experiment.json"
    ).read_bytes()
    (project / "src/llm_memory_editability/ttcd_double_new_eval.py").write_text(
        "changed live extension"
    )
    controller.freeze_source(batch, parent, path)
    assert (
        (batch / "source/src/llm_memory_editability/ttcd_double_new_eval.py")
        .read_text()
        .startswith("new supplementary")
    )


def test_parent_and_panel_hash_changes_block_followup_stages(tmp_path, monkeypatch):
    parent, _, _ = make_parent(tmp_path)
    project, path = make_project(tmp_path)
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    monkeypatch.setattr(controller, "ROOT", project)
    controller.freeze_source(batch, parent, path)
    (batch / "preparation").mkdir()
    (batch / "preparation/result-prepare.json").write_text('{"passed": true}')
    (batch / "panel.json").write_text('{"graph_only": true}')
    controller.record_panel(batch)
    controller.verify_snapshot(batch, parent, panel_required=True)
    (batch / "panel.json").write_text('{"graph_only": "changed"}')
    with pytest.raises(ValueError, match="panel changed"):
        controller.verify_snapshot(batch, parent, panel_required=True)
    (parent / "source-v2/src/llm_memory_editability/ttcd_verified_base.py").write_text(
        "changed parent"
    )
    with pytest.raises(ValueError, match="Parent repaired source changed"):
        controller.verify_snapshot(batch, parent)


def test_each_arm_waits_for_all_original_local_and_cloud_receipts(tmp_path):
    parent, base, _ = make_parent(tmp_path)
    with pytest.raises(controller.DependencyPending):
        controller.dependency_receipt(parent, base, "single")
    complete_parent(parent, base, "single")
    receipt = controller.dependency_receipt(parent, base, "single")
    assert receipt["passed"] and receipt["new_base_parameter_updates"] == 0
    assert not (parent / "stream").exists()
    (parent / "single/result-cloud-audit.json").unlink()
    with pytest.raises(controller.DependencyPending, match="cloud completion"):
        controller.dependency_receipt(parent, base, "single")


def test_supplement_waits_for_both_parent_arms_before_gpu_admission(tmp_path, monkeypatch):
    parent, base, _ = make_parent(tmp_path)
    complete_parent(parent, base, "single")
    updates = []

    def waiting(seconds):
        raise TimeoutError("test stopped while the other original arm is pending")

    monkeypatch.setattr(controller.time, "sleep", waiting)
    with pytest.raises(TimeoutError, match="other original arm is pending"):
        controller.wait_dependencies(parent, base, "single", float("inf"), updates.append)
    assert updates[-1]["state"] == "waiting_for_parent_completion"
    assert "stream local and cloud completion" in updates[-1]["reason"]
    complete_parent(parent, base, "stream")
    receipt = controller.wait_dependencies(parent, base, "single", float("inf"))
    assert receipt == controller.dependency_receipt(parent, base, "single")


def test_other_parent_failure_stops_wait_even_when_own_receipts_are_missing(tmp_path):
    parent, base, _ = make_parent(tmp_path)
    complete_parent(parent, base, "stream")
    (parent / "stream/result-cloud-audit.json").write_text('{"passed": false}')
    with pytest.raises(RuntimeError, match="Required original gate/audit failed"):
        controller.wait_dependencies(parent, base, "single", float("inf"))


@pytest.mark.parametrize("failure", ["common", "audit", "cloud", "stage"])
def test_failed_original_receipt_never_admits_evaluation(tmp_path, failure):
    parent, base, _ = make_parent(tmp_path)
    complete_parent(parent, base, "single")
    if failure == "common":
        path = parent / "common/gate.json"
        path.write_text('{"passed": false}')
    elif failure == "audit":
        (parent / "single/audit.json").write_text('{"passed": false}')
    elif failure == "cloud":
        (parent / "single/result-cloud-audit.json").write_text('{"passed": false}')
    else:
        (parent / "single/exit-evaluate.json").write_text('{"returncode": 1}')
        (parent / "single/result-cloud-audit.json").unlink()
    with pytest.raises(RuntimeError):
        controller.dependency_receipt(parent, base, "single")


def test_cpu_prepare_and_gpu_stage_have_isolated_readonly_parent_mounts(tmp_path):
    config = settings()
    batch, parent = tmp_path / controller.BATCH, tmp_path / controller.PARENT
    token = tmp_path / "secret"
    token.write_text("never-print-this-value")
    _, preparation = controller.build_command(config, "single", "prepare", batch, parent, token)
    assert "--gpus" not in preparation
    assert preparation[preparation.index("--device") + 1] == "cpu"
    assert preparation[preparation.index("--out") + 1].endswith("/preparation")
    _, evaluation = controller.build_command(
        config, "stream", "evaluate", batch, parent, token, {"gpu": 7, "uuid": "GPU-seven"}
    )
    assert evaluation[:4] == ["docker", "--context", "lm-memory", "run"]
    assert evaluation[evaluation.index("--gpus") + 1] == "device=GPU-seven"
    assert "LM_ACTUAL_GPU_INDEX=7" in evaluation
    assert f"type=bind,source={tmp_path},target=/project/results,readonly" in evaluation
    assert (
        f"type=bind,source={parent / 'input-mount/data'},target=/project/data,readonly"
        in evaluation
    )
    assert (
        f"type=bind,source={batch},target=/project/results/{controller.BATCH},readonly"
        in evaluation
    )
    assert (
        f"type=bind,source={batch / 'stream'},target=/project/results/{controller.BATCH}/stream"
    ) in evaluation
    assert "/frozen/configs/experiment.json" in evaluation
    assert "/frozen/configs/double-new-eval.json" in evaluation
    assert "never-print-this-value" not in " ".join(evaluation)


def test_reused_idle_policy_rejects_unknown_pids_and_reserved_containers():
    policy = controller.load_module(
        ROOT / "scripts/reschedule_ttcd_verified.py", "test_double_new_idle"
    )

    class Busy(RuntimeError):
        pass

    helpers = SimpleNamespace(
        ResourceUnavailable=Busy,
        utc_now=controller.utc_now,
        container_uses_gpu=lambda container, gpu, uuid: container["uuid"] == uuid,
    )
    config = controller.resource_config(settings())
    assert config["runtime"]["gpu_memory_max_mib"] == 1024
    assert config["runtime"]["gpu_utilization_max"] == 5
    row = {"uuid": "GPU-four", "memory": 0, "utilization": 0, "total_mib": 97887, "free_mib": 97000}
    assert policy.idle_row(helpers, config, "single", 4, {4: row}, [], [], 1024**3)["gpu"] == 4
    with pytest.raises(Busy, match="occupied"):
        policy.idle_row(
            helpers, config, "single", 4, {4: row}, [{"uuid": "GPU-four", "pid": 1047}], [], 1024**3
        )
    with pytest.raises(Busy, match="occupied"):
        policy.idle_row(helpers, config, "single", 4, {4: row}, [], [{"uuid": "GPU-four"}], 1024**3)


def test_gpu_waiter_uses_preferred_then_idle_fallback_uuid_with_exclusive_locks(
    tmp_path, monkeypatch
):
    calls, locks = [], []

    class Busy(RuntimeError):
        pass

    helpers = SimpleNamespace(
        ResourceUnavailable=Busy, gpu_snapshot=lambda: ({4: {}, 5: {}, 7: {}}, [])
    )

    def admit(config, arm, gpu, helpers, policy):
        calls.append(gpu)
        if gpu in (4, 7):
            raise Busy("parent or other process still running")
        return {"gpu": 5, "uuid": "GPU-five"}

    monkeypatch.setattr(controller, "admit_gpu", admit)
    monkeypatch.setattr(
        controller.fcntl, "flock", lambda handle, mode: locks.append(Path(handle.name).name)
    )
    held, row = controller.wait_gpu(settings(), "single", helpers, None, float("inf"))
    held.close()
    assert calls == [4, 7, 5]
    assert locks == ["gpu-4.lock", "gpu-7.lock", "gpu-5.lock"]
    assert row["uuid"] == "GPU-five"


@pytest.mark.parametrize("failed_stage", ["preflight", "evaluate"])
def test_failed_stage_never_retries_or_starts_later_stages(tmp_path, monkeypatch, failed_stage):
    batch, parent = tmp_path / controller.BATCH, tmp_path / controller.PARENT
    batch.mkdir()
    receipt = {"passed": True}
    calls = []
    monkeypatch.setattr(controller, "wait_dependencies", lambda *args: receipt)
    monkeypatch.setattr(controller, "dependency_receipt", lambda *args: receipt)
    monkeypatch.setattr(
        controller,
        "wait_gpu",
        lambda *args: ((tmp_path / "gpu.lock").open("a"), {"gpu": 4, "uuid": "GPU-four"}),
    )
    monkeypatch.setattr(controller, "admit_gpu", lambda *args: {"gpu": 4, "uuid": "GPU-four"})

    def stage(config, arm, name, *args):
        calls.append(name)
        return int(name == failed_stage)

    monkeypatch.setattr(controller, "run_stage", stage)
    assert (
        controller.run_arm(
            settings(),
            "single",
            list(controller.STAGES),
            batch,
            parent,
            tmp_path / "token",
            None,
            None,
            {},
        )
        == 1
    )
    assert calls == list(controller.STAGES[: controller.STAGES.index(failed_stage) + 1])
    state = json.loads((batch / "single/controller-state.json").read_text())
    assert state["state"] == "needs_attention" and state["new_base_parameter_updates"] == 0


def test_false_preflight_receipt_stops_even_with_zero_container_exit(tmp_path, monkeypatch):
    batch, parent = tmp_path / controller.BATCH, tmp_path / controller.PARENT
    batch.mkdir()
    receipt = {"passed": True, "files": {"parent-checkpoint": "unchanged"}}
    monkeypatch.setattr(controller, "wait_dependencies", lambda *args: receipt)
    monkeypatch.setattr(controller, "dependency_receipt", lambda *args: receipt)
    monkeypatch.setattr(controller, "verify_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        controller,
        "wait_gpu",
        lambda *args: ((tmp_path / "gpu.lock").open("a"), {"gpu": 4, "uuid": "GPU-four"}),
    )
    calls = []

    def launch(command, **kwargs):
        calls.append(command[command.index("/frozen/scripts/run_ttcd_double_new_eval.py") + 1])
        (batch / "single/result-preflight.json").write_text('{"passed": false}')
        return SimpleNamespace(returncode=0, stdout="test-container-id", stderr="")

    monkeypatch.setattr(controller.subprocess, "run", launch)
    helpers = SimpleNamespace(
        inspect_container=lambda name: None, monitor_container=lambda *args: 0
    )
    assert (
        controller.run_arm(
            settings(),
            "single",
            list(controller.STAGES),
            batch,
            parent,
            tmp_path / "token",
            helpers,
            None,
            {},
        )
        == 3
    )
    assert calls == ["preflight"]
    assert not (batch / "single/claim-evaluate.json").exists()
    exit_record = json.loads((batch / "single/exit-preflight.json").read_text())
    assert exit_record["returncode"] == 3 and exit_record["container_returncode"] == 0
    assert json.loads((batch / "single/parent-completion-receipt.json").read_text()) == receipt


def test_standalone_evaluation_cannot_bypass_cuda_preflight(tmp_path, monkeypatch):
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    monkeypatch.setattr(controller, "verify_snapshot", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="passing development CUDA preflight"):
        controller.run_stage(
            settings(),
            "single",
            "evaluate",
            batch,
            tmp_path / controller.PARENT,
            tmp_path / "token",
            None,
            float("inf"),
            {"gpu": 4, "uuid": "GPU-four"},
        )
    assert not (batch / "single/claim-evaluate.json").exists()


def test_parent_identity_changed_after_preflight_blocks_evaluation(tmp_path, monkeypatch):
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    receipt = {"passed": True, "files": {"checkpoint": "original"}}
    calls = []
    monkeypatch.setattr(controller, "wait_dependencies", lambda *args: receipt)
    monkeypatch.setattr(
        controller,
        "dependency_receipt",
        lambda *args: (
            receipt if not calls else {"passed": True, "files": {"checkpoint": "changed"}}
        ),
    )
    monkeypatch.setattr(
        controller,
        "wait_gpu",
        lambda *args: ((tmp_path / "gpu.lock").open("a"), {"gpu": 4, "uuid": "GPU-four"}),
    )

    def stage(config, arm, name, *args):
        calls.append(name)
        return 0

    monkeypatch.setattr(controller, "run_stage", stage)
    assert (
        controller.run_arm(
            settings(),
            "single",
            list(controller.STAGES),
            batch,
            tmp_path / controller.PARENT,
            tmp_path / "token",
            None,
            None,
            {},
        )
        == 1
    )
    assert calls == ["preflight"]
    failure = json.loads((batch / "single/controller-failure.json").read_text())
    assert "Original parent completion identity changed" in failure["traceback"]


def test_original_wait_timeout_does_not_touch_parent_jobs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A wait timeout must not inspect, stop or retry a parent")

    monkeypatch.setattr(controller, "dependency_receipt", forbidden)
    with pytest.raises(TimeoutError, match="no parent job was changed"):
        controller.wait_dependencies(tmp_path, {}, "single", 0)
