"""No-restart handoff and future-command provenance contracts; no model execution."""

import contextlib
import copy
import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ttcd_speedup_controller", ROOT / "scripts/accelerate_ttcd_controllers.py"
)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)
IMAGE = "sha256:" + "1" * 64


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


def command(script="run_ttcd_verified_base.py"):
    return [
        "docker",
        "--context",
        "lm-memory",
        "run",
        "--detach",
        "--name",
        "old-stage",
        "--gpus",
        "device=GPU-original",
        "--cpuset-cpus",
        "24,25,26,27",
        "--memory",
        "96g",
        "--mount",
        "type=bind,source=/old/source,target=/frozen,readonly",
        "--env",
        "PYTHONPATH=/frozen/src",
        IMAGE,
        "bash",
        "-c",
        'exec "$@"',
        "old-entry",
        "python",
        f"/frozen/scripts/{script}",
        "train",
        "--model",
        "stream",
        "--config",
        "/frozen/configs/experiment.json",
        "--out",
        "/old/output",
        "--tracking",
        "online",
    ]


def speedup(tmp_path):
    source = tmp_path / "source"
    entry = source / "scripts/run_ttcd_speedup.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("# Frozen runtime fixture; never executed\n")
    manifest = source.parent / "source-manifest.json"
    write(
        manifest,
        {
            "files": {str(entry.relative_to(source)): controller.sha256(entry)},
            "numerical_verification": {"passed": True},
        },
    )
    return source, manifest


@pytest.mark.parametrize("target", controller.TARGETS)
def test_default_plan_does_not_read_or_write_runtime(tmp_path, target):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/accelerate_ttcd_controllers.py"),
            "--target",
            target,
            "--workspace",
            str(tmp_path / "absent"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    value = json.loads(result.stdout)
    assert value["wrapper_applies_to_future_TTCD_containers_only"]
    assert not value["executed"] and not value["model_restart"] and not value["signals_sent"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("script", sorted(controller.WRAPPED_SCRIPTS))
def test_wrap_preserves_every_scientific_argument_resource_and_source_mount(tmp_path, script):
    source, manifest = speedup(tmp_path)
    original = command(script)
    before = copy.deepcopy(original)
    wrapped = controller.wrap_command(
        original, IMAGE, source, manifest, "parent:stream:train-audit"
    )
    index = wrapped.index("/run/ttcd-speedup/scripts/run_ttcd_speedup.py")
    assert wrapped[index + 1] == "--"
    assert wrapped[index + 2 :] == original[original.index("python") + 1 :]
    for option in ("--name", "--gpus", "--cpuset-cpus", "--memory"):
        assert wrapped[wrapped.index(option) + 1] == original[original.index(option) + 1]
    assert "type=bind,source=/old/source,target=/frozen,readonly" in wrapped
    assert "PYTHONPATH=/frozen/src" in wrapped
    assert "LM_TTCD_SPEEDUP_MANIFEST_SHA256=" + controller.sha256(manifest) in wrapped
    assert original == before
    with pytest.raises(ValueError, match="already wrapped"):
        controller.wrap_command(wrapped, IMAGE, source, manifest, "again")


def test_native_foundation_commands_are_not_wrapped(tmp_path):
    source, manifest = speedup(tmp_path)
    original = command("run_ttcd_architecture_foundation.py")
    assert (
        controller.wrap_command(original, IMAGE, source, manifest, "foundation:audit") == original
    )


def test_wrong_docker_context_and_nonfrozen_ttcd_entry_are_rejected(tmp_path):
    source, manifest = speedup(tmp_path)
    value = command()
    value[2] = "default"
    with pytest.raises(ValueError, match="explicit LM"):
        controller.wrap_command(value, IMAGE, source, manifest, "bad")
    value = command()
    value[value.index("python") + 1] = "/mutable/scripts/run_ttcd_verified_base.py"
    with pytest.raises(ValueError, match="frozen"):
        controller.wrap_command(value, IMAGE, source, manifest, "bad")


@pytest.mark.parametrize("change", ["unverified", "source", "escape", "receipt"])
def test_runtime_manifest_requires_numerical_proof_and_exact_source(tmp_path, change):
    source, manifest = speedup(tmp_path)
    value = json.loads(manifest.read_text())
    if change == "unverified":
        value["numerical_verification"]["passed"] = False
    elif change == "source":
        (source / "scripts/run_ttcd_speedup.py").write_text("changed")
    elif change == "escape":
        outside = source.parent / "outside.py"
        outside.write_text("outside")
        value["files"]["../outside.py"] = controller.sha256(outside)
    else:
        write(
            source.parent / "engineering-receipt.json",
            {"passed": True, "source_manifest_sha256": "other"},
        )
    write(manifest, value)
    with pytest.raises(ValueError):
        controller.verify_speedup(source)


def test_runtime_manifest_can_bind_an_independent_engineering_receipt(tmp_path):
    source, manifest = speedup(tmp_path)
    write(
        source.parent / "engineering-receipt.json",
        {"passed": True, "source_manifest_sha256": controller.sha256(manifest)},
    )
    assert controller.verify_speedup(source) == manifest


def test_deadline_translation_accounts_for_elapsed_time_and_never_resets():
    now = dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc)
    end = controller.expiration(now.isoformat(), 24)
    assert (
        controller.deadline_from_utc(end, wall_now=now + dt.timedelta(hours=2), monotonic_now=50)
        == 50 + 22 * 3600
    )
    with pytest.raises(TimeoutError):
        controller.deadline_from_utc(end, wall_now=now + dt.timedelta(days=2))
    with pytest.raises(ValueError, match="timezone"):
        controller.parse_utc("2026-10-15T00:00:00")


def handoff_fixture(tmp_path):
    anchor = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    batch = tmp_path / controller.PARENT
    write(batch / "common/exit-cloud-audit.json", {"returncode": 0, "utc": anchor})
    protected = batch / "source-manifest.json"
    protected.write_text("immutable original manifest")
    containers = {}
    active = []
    for i, arm in enumerate(("single", "stream")):
        name = f"{arm}-original-train"
        containers[name] = {"Id": name + "-id", "State": {"Running": True, "Pid": 100 + i}}
        active.append(
            {
                "arm": arm,
                "stage": "train",
                "container": name,
                "container_id": name + "-id",
                "host_model_pid": 100 + i,
                "deadline_utc": controller.expiration(anchor, 24),
                "state": {"phase": arm},
            }
        )
    context = SimpleNamespace(
        target="parent",
        batch=batch,
        record={"pid": 123},
        config={"runtime": {"timeout_hours": 24}},
        helpers=SimpleNamespace(inspect_container=lambda name: containers[name]),
    )
    snapshot = {
        "schema": "ttcd-speedup-handoff-v1",
        "target": "parent",
        "batch": str(batch),
        "old_controller": {"pid": 123, "state": "T", "start_ticks": 10},
        "active": active,
        "wait_deadline_utc": None,
        "protected_files": {str(protected): controller.sha256(protected)},
    }
    return context, snapshot, containers, protected


@pytest.mark.parametrize(
    "change",
    [
        "old_alive",
        "not_stopped",
        "model_pid",
        "container",
        "failed",
        "source",
        "deadline",
        "missing_arm",
    ],
)
def test_handoff_rejects_restarts_failures_changed_sources_or_deadlines(
    tmp_path, monkeypatch, change
):
    context, snapshot, containers, protected = handoff_fixture(tmp_path)
    monkeypatch.setattr(controller, "process_identity", lambda _: None)
    if change == "old_alive":
        monkeypatch.setattr(controller, "process_identity", lambda _: {"state": "T"})
    elif change == "not_stopped":
        snapshot["old_controller"]["state"] = "R"
    elif change == "model_pid":
        containers["single-original-train"]["State"]["Pid"] += 1
    elif change == "container":
        containers["single-original-train"]["Id"] = "replacement"
    elif change == "failed":
        containers["single-original-train"]["State"].update(
            Running=False, Status="exited", ExitCode=1
        )
    elif change == "source":
        protected.write_text("changed")
    elif change == "deadline":
        snapshot["active"][0]["deadline_utc"] = controller.expiration(controller.utc_now(), 48)
    else:
        snapshot["active"].pop()
    with pytest.raises(ValueError):
        controller.validate_handoff(snapshot, context)


def test_natural_completion_during_handoff_is_adopted_without_relaunch(tmp_path, monkeypatch):
    context, snapshot, containers, _ = handoff_fixture(tmp_path)
    monkeypatch.setattr(controller, "process_identity", lambda _: None)
    containers["single-original-train"]["State"] = {
        "Running": False,
        "Pid": 0,
        "Status": "exited",
        "ExitCode": 0,
    }
    controller.validate_handoff(snapshot, context)


def test_adopt_monitor_does_not_call_admission_or_builders(tmp_path):
    calls = []
    job = {"arm": "stream", "container": "original", "container_id": "exact", "host_model_pid": 9}
    context = SimpleNamespace(
        helpers=SimpleNamespace(
            monitor_container=lambda name, log, deadline, update: (
                calls.append((name, deadline)) or 0
            )
        )
    )
    assert controller.monitor_adopted(context, job, 123.0, tmp_path, lambda _: None) == 0
    assert calls == [("original", 123.0)]
    with pytest.raises(FileExistsError):
        controller.monitor_adopted(context, job, 123.0, tmp_path, lambda _: None)


@pytest.mark.parametrize("train_exit", [0, 1])
def test_parent_adopts_train_once_then_uses_original_stage_order_and_deadline(tmp_path, train_exit):
    context, snapshot, _, _ = handoff_fixture(tmp_path)
    batch = context.batch
    (batch / "stream").mkdir()
    events = []
    helpers = context.helpers
    helpers.ARM_STAGES = ("train", "train-audit", "calibrate", "evaluate", "audit", "cloud-audit")
    helpers.require_common_gate = lambda *args: events.append("gate")
    helpers.write_json = controller.write_json
    helpers.monitor_container = lambda name, log, deadline, update: (
        events.append(("adopt", name, deadline)) or train_exit
    )
    context.policy = SimpleNamespace(
        admission=lambda helpers, runtime, phase: {"gpu": 4, "uuid": "original-GPU"}
    )

    def stage(batch, config, phase, stage, token, deadline, row, update):
        events.append(("stage", phase, stage, deadline, row["gpu"]))
        return 0

    context.controller = SimpleNamespace(
        verify_repaired_source=lambda *args: events.append("verify"), run_stage=stage
    )
    context.config_path = tmp_path / "config"
    context.receipt = tmp_path / "receipt"
    code = controller.run_parent_arm(context, snapshot["active"][1], tmp_path / "token", tmp_path)
    assert code == train_exit
    stages = [e for e in events if isinstance(e, tuple) and e[0] == "stage"]
    assert [e[2] for e in stages] == ([] if train_exit else list(helpers.ARM_STAGES[1:]))
    adopted = next(e for e in events if isinstance(e, tuple) and e[0] == "adopt")
    assert all(e[3] == adopted[2] and e[4] == 4 for e in stages)
    assert json.loads((batch / "stream/exit-train.json").read_text())["adopted"]
    assert not any(e[0] == "stage" and e[2] == "train" for e in stages)


def test_architecture_finishes_existing_foundation_before_original_remaining_stages(
    tmp_path, monkeypatch
):
    batch = tmp_path / controller.ARCHITECTURE
    stage_dir = batch / "loop/stages/foundation-train"
    stage_dir.mkdir(parents=True)
    output = batch / "loop/foundation"
    write(output / "result-train.json", {"step": 300000})
    events = []
    specs = [
        {"label": "foundation-preflight", "stage": "preflight"},
        {"label": "foundation-train", "stage": "train"},
        {"label": "foundation-audit", "stage": "audit"},
        {"label": "common-train", "stage": "train"},
    ]
    helpers = SimpleNamespace(
        monitor_container=lambda name, log, deadline, update: events.append("adopt") or 0,
        inspect_container=lambda _: {"Id": "original", "State": {"ExitCode": 0}},
    )
    fake = SimpleNamespace(
        stage_plan=lambda arm: specs,
        output_path=lambda *args: output,
        write_json=controller.write_json,
        run_stage=lambda plan, arm, spec, batch, token, update: events.append(spec["label"]),
    )
    context = SimpleNamespace(
        controller=fake, helpers=helpers, batch=batch, config={"runtime": {"gpus": {"loop": 9}}}
    )
    job = {
        "arm": "loop",
        "stage": "foundation-train",
        "container": "original",
        "deadline_utc": controller.expiration(controller.utc_now(), 48),
        "state": {"completed_stages": ["foundation-preflight"]},
    }

    @contextlib.contextmanager
    def no_real_gpu_locks(paths):
        yield

    monkeypatch.setattr(controller, "locks", no_real_gpu_locks)
    state = controller.run_architecture_arm(context, job, tmp_path / "token", tmp_path)
    assert events == ["adopt", "foundation-audit", "common-train"]
    assert state["completed_stages"] == [s["label"] for s in specs]
    assert json.loads((stage_dir / "exit.json").read_text())["adopted"]


def test_lock_conflict_and_exclusive_claim_prevent_duplicate_adoption(tmp_path):
    with controller.locks([tmp_path / "controller.lock"]):
        with pytest.raises(BlockingIOError):
            with controller.locks([tmp_path / "controller.lock"]):
                pass
    claim = tmp_path / "claim.json"
    controller.write_json(claim, {"first": True}, exclusive=True)
    with pytest.raises(FileExistsError):
        controller.write_json(claim, {"first": False}, exclusive=True)
    assert json.loads(claim.read_text()) == {"first": True}
