"""One-step training scope, immutable sources and safe within-batch sharing."""

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
    "one_step_training_controller", ROOT / "scripts/launch_ttcd_one_step_training.py"
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
    workspace, persistent = tmp_path / "workspace", tmp_path / "persistent"
    workspace.mkdir()
    persistent.mkdir()
    batch = persistent / controller.BATCH
    batch.mkdir()
    inputs_file = persistent / "ttcd-successful-base-v1/input-mount/data/example.bin"
    inputs_file.parent.mkdir(parents=True)
    inputs_file.write_bytes(b"original immutable graph")
    inputs = {"data/example.bin": controller.sha256(inputs_file)}
    panel = b'{"immutable": "facts"}\n'
    settings["panel_sha256"] = controller.transport.digest(panel)
    for arm in controller.ARCHITECTURES:
        spec = settings["architectures"][arm]
        source = persistent / spec["parent_batch"] / spec["source_directory"]
        common = persistent / spec["parent_batch"] / arm / "common"
        common.mkdir(parents=True)
        checkpoint, gate = common / "parent.pt", common / "gate.json"
        checkpoint.write_bytes(f"original {arm} common10000 weights".encode())
        write(gate, {"passed": True})
        config = {
            "writer": {"steps": 4},
            "training": {"steps": 10000},
            "runtime": {"image": settings["runtime"]["image"]},
            "common": {
                "checkpoint": "results/" + str(checkpoint.relative_to(persistent)),
                "gate_receipt": "results/" + str(gate.relative_to(persistent)),
            },
        }
        write(source / spec["parent_config"], config)
        (source / "configs/continual-panel.json").write_bytes(panel)
        (source / "original.py").write_bytes(b"# frozen parent science\n")
    for parent_batch, source_name, manifest_name in (
        ("ttcd-architecture-comparison-v1", "source", "source-manifest.json"),
        ("ttcd-successful-base-v1", "source-v2", "source-v2-manifest.json"),
    ):
        parent = persistent / parent_batch
        write(
            parent / manifest_name,
            {
                "files": controller.resources.source_hashes(parent / source_name),
                "inputs" if source_name == "source-v2" else "parent_immutable_inputs": inputs,
            },
        )
    for arm in controller.ARCHITECTURES:
        spec = settings["architectures"][arm]
        parent = persistent / spec["parent_batch"]
        config_path = parent / spec["source_directory"] / spec["parent_config"]
        config = json.loads(config_path.read_text())
        task = {
            "batch": controller.BATCH,
            "architecture_name": arm,
            "training_method": "stream",
            "parent_config": "configs/parent.json",
            "parent_config_sha256": controller.sha256(config_path),
            "parent_source_manifest_sha256": controller.sha256(parent / spec["source_manifest"]),
            "parent_common_checkpoint": config["common"]["checkpoint"],
            "parent_common_gate": config["common"]["gate_receipt"],
            "parent_common_checkpoint_sha256": controller.sha256(parent / arm / "common/parent.pt"),
            "parent_common_gate_sha256": controller.sha256(parent / arm / "common/gate.json"),
            "training_config": {"writer": {"steps": 1}, "training": {"steps": 10000}},
        }
        write(controller.task_path(workspace, arm), task)
    for relative in controller.ADDITIONS:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# new batch code\n")
    config_path = workspace / "settings.json"
    write(config_path, settings)
    controller.freeze(batch, settings, config_path, workspace)
    return batch, settings, workspace, config_path


def test_three_stream_models_only_and_full_pipeline():
    controller.validate_plan(plan())
    assert controller.STAGES == (
        "prepare",
        "preflight",
        "train",
        "train-audit",
        "train-cloud-audit",
        "eval-preflight",
        "calibrate",
        "evaluate",
        "audit",
        "cloud-audit",
    )
    assert controller.STAGES.count("train") == 1


@pytest.mark.parametrize(
    "change",
    [
        "writer",
        "training_writer",
        "updates",
        "foundation",
        "common",
        "methods",
        "roles",
        "cpus",
        "free",
        "gpu8",
        "sharing",
        "image",
    ],
)
def test_rejects_scope_and_resource_drift(change):
    settings = plan()
    if change == "writer":
        settings["writer_steps"] = 4
    elif change == "training_writer":
        settings["training_writer_steps"] = 4
    elif change == "updates":
        settings["new_base_parameter_updates"] = 90000
    elif change == "foundation":
        settings["new_foundation_updates"] = 300000
    elif change == "common":
        settings["new_common_updates"] = 10000
    elif change == "methods":
        settings["training_methods"] = ["stream", "single", "independent_merge"]
    elif change == "roles":
        settings["roles"] = ["second"]
    elif change == "cpus":
        settings["runtime"]["cpu_sets"]["loop"] = [64, 65, 66, 67]
    elif change == "free":
        settings["runtime"]["minimum_shared_free_mib"] = 1
    elif change == "gpu8":
        settings["runtime"]["gpu_preferences"]["loop"] = [8]
    elif change == "sharing":
        settings["runtime"]["allow_own_batch_sharing"] = False
    else:
        settings["runtime"]["image"] = "latest"
    with pytest.raises(ValueError):
        controller.validate_plan(settings)


def test_freeze_preserves_parent_bytes_and_binds_separate_task(frozen):
    batch, _, _, config = frozen
    for arm in controller.ARCHITECTURES:
        snapshot = controller.transport.verify_snapshot(batch, arm)
        source = batch / arm / "source"
        assert snapshot["base_parameter_updates"] == 10000
        assert snapshot["writer_steps"] == 1
        assert not snapshot["host_permissions_changed"]
        assert json.loads((source / "configs/parent.json").read_text())["writer"]["steps"] == 4
        assert (
            json.loads((source / "configs/task.json").read_text())["training_config"]["writer"][
                "steps"
            ]
            == 1
        )
        assert (source / "configs/one-step-training.json").read_bytes() == config.read_bytes()
        controller.verify_entries(source, snapshot["parent_files"])
    with pytest.raises(FileExistsError):
        controller.freeze(batch, plan(), config)


def test_snapshot_rejects_changed_original_byte(frozen):
    batch, _, _, _ = frozen
    (batch / "loop/source/original.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        controller.transport.verify_snapshot(batch, "loop")


def test_runtime_tar_keeps_source_immutable_without_host_permission_changes(frozen):
    batch, _, _, _ = frozen
    original = {p: p.stat().st_mode for p in (batch / "discoloop/source").rglob("*")}
    archive = controller.runtime_archive(batch, "discoloop")
    with tarfile.open(fileobj=io.BytesIO(archive)) as content:
        assert (
            content.extractfile("frozen/source-manifest.json").read()
            == (batch / "discoloop/source-manifest.json").read_bytes()
        )
        for member in content:
            assert member.uid == member.gid == 0
            assert member.mode == (0o555 if member.isdir() else 0o444)
    assert all(p.stat().st_mode == mode for p, mode in original.items())


def test_common_checkpoint_change_is_rejected(frozen):
    batch, settings, workspace, _ = frozen
    task = json.loads(controller.task_path(workspace, "loop").read_text())
    parent = batch.parent / settings["architectures"]["loop"]["parent_batch"]
    parent_path = parent / "source/configs/architecture-loop.json"
    (parent / "loop/common/parent.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="common checkpoint"):
        controller.verify_task(
            task, "loop", json.loads(parent_path.read_text()), parent_path, batch
        )


def row(gpu, free=99000, memory=0):
    return {
        "uuid": f"GPU-{gpu}",
        "total_mib": 100000,
        "free_mib": free,
        "memory": memory,
        "utilization": 0,
    }


def test_slots_share_single_global_lock_until_last_arm_exits(tmp_path):
    import fcntl

    slots = controller.CohortSlots(tmp_path)
    first = slots.take(5, "transformer")
    second = slots.take(5, "loop")
    first.close()
    with (tmp_path / "gpu-5.lock").open("a") as outside:
        with pytest.raises(BlockingIOError):
            fcntl.flock(outside, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second.close()
        fcntl.flock(outside, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_slots_reject_duplicate_arm_and_limit(tmp_path):
    slots = controller.CohortSlots(tmp_path)
    with slots.take(5, "transformer", maximum=1):
        with pytest.raises(ValueError):
            slots.take(6, "transformer")
        with pytest.raises(controller.resources.ResourceUnavailable):
            slots.take(5, "loop", maximum=1)


def own_container(batch, settings, arm, stage, gpu=5):
    output = controller.output_directory(batch, arm, stage)
    manifest_sha = controller.sha256(batch / arm / "source-manifest.json")
    return {
        "Id": f"{arm}-id",
        "Name": f"/{controller.BATCH}-{arm}-{stage}",
        "Config": {
            "User": "1015:1016",
            "Image": settings["runtime"]["image"],
            "Cmd": ["/frozen/scripts/run_ttcd_one_step_training.py"],
            "Env": [f"LM_TTCD_SOURCE_MANIFEST_SHA256={manifest_sha}"],
        },
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": [f"GPU-{gpu}"]}]},
        "Mounts": [{"Destination": "/output", "Source": str(output), "RW": True}],
    }


def test_cohort_allows_exact_own_container_and_known_cuda_pid(frozen, monkeypatch):
    batch, settings, _, _ = frozen
    container = own_container(batch, settings, "loop", "preflight")
    monkeypatch.setattr(
        controller.resources, "docker", lambda *_: SimpleNamespace(stdout="PID\n123\n")
    )
    members = controller.gpu_cohort(
        batch, settings, 5, row(5, memory=4000), [{"uuid": "GPU-5", "pid": 123}], [container]
    )
    assert members[0]["host_pids"] == [123]


@pytest.mark.parametrize(
    "change", ["other_project", "source_bind", "wrong_owner", "wrong_manifest", "unknown_pid"]
)
def test_cohort_refuses_unrelated_or_unverified_workload(frozen, monkeypatch, change):
    batch, settings, _, _ = frozen
    container = own_container(batch, settings, "loop", "train")
    processes = [{"uuid": "GPU-5", "pid": 123}]
    if change == "other_project":
        container["Name"] = "/other-project"
    elif change == "source_bind":
        container["Mounts"].append({"Destination": "/frozen", "Source": "/other", "RW": False})
    elif change == "wrong_owner":
        container["Config"]["User"] = "root"
    elif change == "wrong_manifest":
        container["Config"]["Env"] = []
    else:
        processes[0]["pid"] = 999
    monkeypatch.setattr(
        controller.resources, "docker", lambda *_: SimpleNamespace(stdout="PID\n123\n")
    )
    with pytest.raises(controller.resources.ResourceUnavailable):
        controller.gpu_cohort(batch, settings, 5, row(5, memory=4000), processes, [container])


def test_admission_uses_idle_card_before_shared_and_excludes_gpu8(frozen, monkeypatch, tmp_path):
    batch, settings, _, _ = frozen
    container = own_container(batch, settings, "loop", "train")
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([container], 999999999))
    monkeypatch.setattr(
        controller.resources, "docker", lambda *_: SimpleNamespace(stdout="PID\n123\n")
    )
    monkeypatch.setattr(
        controller.resources,
        "gpu_snapshot",
        lambda: ({5: row(5, 70000, 30000), 6: row(6), 8: row(8)}, [{"uuid": "GPU-5", "pid": 123}]),
    )
    held, admitted = controller.acquire_resources(
        settings, "transformer", "train", lambda _: None, batch, controller.CohortSlots(tmp_path)
    )
    assert admitted["gpu"] == 6
    assert not admitted["gpu_sharing"]
    held.close()


def test_admission_never_uses_idle_gpus_outside_authorized_cuda5_6(frozen, monkeypatch, tmp_path):
    batch, settings, _, _ = frozen
    updates = []
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([], 999999999))
    monkeypatch.setattr(
        controller.resources,
        "gpu_snapshot",
        lambda: ({gpu: row(gpu) for gpu in (0, 7, 8, 9)}, []),
    )
    ticks = iter((0, 0, 400000, 400000))
    monkeypatch.setattr(controller.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="No safe resource"):
        controller.acquire_resources(
            settings,
            "transformer",
            "train",
            updates.append,
            batch,
            controller.CohortSlots(tmp_path),
        )
    assert updates[0]["state"] == "waiting_for_resources"
    assert not list(tmp_path.glob("gpu-*.lock"))


def test_admission_shares_own_batch_when_other_cards_occupied(frozen, monkeypatch, tmp_path):
    batch, settings, _, _ = frozen
    container = own_container(batch, settings, "loop", "train")
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([container], 999999999))
    monkeypatch.setattr(
        controller.resources, "docker", lambda *_: SimpleNamespace(stdout="PID\n123\n")
    )
    monkeypatch.setattr(
        controller.resources,
        "gpu_snapshot",
        lambda: ({5: row(5, 70000, 30000), 8: row(8)}, [{"uuid": "GPU-5", "pid": 123}]),
    )
    held, admitted = controller.acquire_resources(
        settings, "transformer", "train", lambda _: None, batch, controller.CohortSlots(tmp_path)
    )
    assert admitted["gpu"] == 5
    assert admitted["gpu_sharing"]
    assert not admitted["wallclock_is_exclusive_gpu_benchmark"]
    held.close()


@pytest.mark.parametrize("stage", controller.CPU_STAGES)
def test_cpu_stages_do_not_touch_gpu(frozen, monkeypatch, tmp_path, stage):
    batch, settings, _, _ = frozen
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([], 999999999))
    monkeypatch.setattr(
        controller.resources, "gpu_snapshot", lambda: pytest.fail("CPU stage queried GPU")
    )
    held, admitted = controller.acquire_resources(
        settings, "loop", stage, lambda _: None, batch, controller.CohortSlots(tmp_path)
    )
    assert admitted["device"] == "cpu"
    held.close()


@pytest.mark.parametrize("stage", ["preflight", "train", "evaluate", "aggregate"])
def test_container_mounts_only_own_arm_writable_and_no_source_binding(frozen, stage):
    batch, settings, _, _ = frozen
    gpu = {} if stage == "aggregate" else {"gpu": 5, **row(5)}
    _, command = controller.build_command(
        settings, "loop", stage, batch, Path("/private/token"), gpu
    )
    assert command[:4] == ["docker", "--context", "lm-memory", "create"]
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    writable = [mount for mount in mounts if not mount.endswith(",readonly")]
    assert len(writable) == 2
    assert all(
        str(batch / "loop") in mount or stage == "aggregate" and str(batch / "report") in mount
        for mount in writable
    )
    assert any("target=/frozen/results,readonly" in mount for mount in mounts)
    assert any(f"target=/frozen/results/{controller.BATCH}/loop" in mount for mount in writable)
    assert any("target=/inputs/data,readonly" in mount for mount in mounts)
    assert any("target=/frozen/data,readonly" in mount for mount in mounts)
    assert not any("target=/frozen," in mount for mount in mounts)
    assert command[command.index("--task") + 1] == "/frozen/configs/task.json"
    assert ("--gpus" in command) is (stage != "aggregate")


def test_arm_failure_stops_pipeline_without_retry(monkeypatch, tmp_path):
    (tmp_path / "loop").mkdir()
    called = []

    def run_stage(_plan, _arm, stage, *_):
        called.append(stage)
        if stage == "train-audit":
            raise ValueError("audit mismatch")

    monkeypatch.setattr(controller, "run_stage", run_stage)
    with pytest.raises(ValueError, match="audit mismatch"):
        controller.run_arm(plan(), "loop", tmp_path, Path("/private/token"))
    assert called == ["prepare", "preflight", "train", "train-audit"]
    state = json.loads((tmp_path / "loop/controller-state.json").read_text())
    assert state["state"] == "needs_attention"
    assert state["completed_stages"] == ["prepare", "preflight", "train"]


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
    monkeypatch.setattr(controller.resources, "monitor_container", lambda *_: 3)
    with pytest.raises(RuntimeError, match="exited 3"):
        controller.run_stage(
            plan(), "loop", "train", tmp_path, Path("/private/token"), lambda _: None
        )
    exit_path = tmp_path / "loop/stages/train/exit.json"
    assert json.loads(exit_path.read_text())["returncode"] == 3
    with pytest.raises(FileExistsError):
        controller.run_stage(
            plan(), "loop", "train", tmp_path, Path("/private/token"), lambda _: None
        )


def test_four_step_controls_wait_for_all_audits_and_fail_without_retry(tmp_path):
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    assert controller.four_step_control_receipt(batch) is None
    parent = tmp_path / "ttcd-merge-hops-v1"
    for arm in controller.ARCHITECTURES:
        for stage in ("audit", "cloud-audit"):
            write(parent / arm / f"evaluation/result-{stage}.json", {"passed": True})
            write(parent / arm / f"stages/{stage}/exit.json", {"returncode": 0})
    receipt = controller.four_step_control_receipt(batch)
    assert receipt["passed"]
    assert len(receipt["receipts"]) == 12
    write(parent / "loop/stages/audit/exit.json", {"returncode": 1})
    with pytest.raises(RuntimeError, match="four-step control loop/audit failed"):
        controller.four_step_control_receipt(batch)


def test_pending_controls_produce_separate_report_without_overwriting_main(tmp_path, monkeypatch):
    batch = tmp_path / controller.BATCH
    batch.mkdir()
    calls, updates = [], []

    def stage(_plan, _arm, _stage, _batch, _token, _update, **kwargs):
        calls.append(kwargs)
        directory = kwargs.get("report_directory", "report")
        write(
            batch / directory / "result-aggregate.json",
            {
                "four_step_training_control": {
                    "status": "complete" if directory == "control-report" else "pending"
                }
            },
        )

    def wait(_plan, _batch, update):
        update({"state": "main_complete_waiting_four_step_control"})

    monkeypatch.setattr(controller, "run_stage", stage)
    monkeypatch.setattr(controller, "wait_four_step_controls", wait)
    evidence = controller.finish_reports(plan(), batch, Path("/private/token"), updates.append)
    assert calls == [
        {},
        {"report_directory": "control-report", "report_variant": "control-aggregate"},
    ]
    assert (
        json.loads((batch / "report/result-aggregate.json").read_text())[
            "four_step_training_control"
        ]["status"]
        == "pending"
    )
    assert evidence["main_report"]["path"] != evidence["control_report"]["path"]
    assert any(u.get("state") == "main_complete_waiting_four_step_control" for u in updates)


def test_control_aggregate_container_has_distinct_name_and_output(frozen):
    batch, settings, _, _ = frozen
    name, command = controller.build_command(
        settings,
        "transformer",
        "aggregate",
        batch,
        Path("/private/token"),
        {"device": "cpu"},
        report_directory="control-report",
        report_variant="control-aggregate",
    )
    assert name.endswith("-control-aggregate")
    assert any(f"source={batch / 'control-report'},target=/output" in arg for arg in command)
    assert (
        command[command.index("/frozen/scripts/run_ttcd_one_step_training.py") + 1] == "aggregate"
    )
