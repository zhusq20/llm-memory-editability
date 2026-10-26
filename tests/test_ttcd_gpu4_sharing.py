"""GPU sharing changes scheduling without repeating any scientific stage."""

import contextlib
import copy
import datetime
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "share_ttcd_gpu4", ROOT / "scripts/share_ttcd_gpu4.py"
)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def stop_record():
    return {
        "old_pid": 123456789,
        "phase": "common",
        "stage": "train",
        "container": f"{adapter.PARENT}-common-train-v2",
        "container_id": "original-container-id",
        "deadline_remaining_seconds": 3600,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "science_updates_snapshot": {"common": 123},
        "no_model_restart": True,
        "reason": "Explicit user instruction to run in parallel on GPU4",
    }


def config():
    return json.loads((ROOT / "configs/ttcd-successful-base-v1.json").read_text())


def fake_resources(events):
    def admission(helpers, runtime, phase):
        events.append(("admit", phase))
        return {
            "gpu": 4,
            "uuid": "GPU-four",
            "gpu_sharing": True,
            "cpu_set": runtime["cpu_sets"][phase],
        }

    def verify(helpers):
        events.append(("verify",))
        return {"row": {"uuid": "GPU-four"}, "containers": [], "authorized_host_pids": []}

    return SimpleNamespace(admission=admission, verify_gpu4_cohort=verify)


def make_repair(tmp_path, *, failure=None, train_exit=0, parallel=False):
    repair = adapter.load_module(
        ROOT / "scripts/repair_ttcd_verified_preflight.py", "gpu4_test_repair"
    )
    batch = tmp_path / adapter.PARENT
    (batch / "common").mkdir(parents=True)
    (batch / adapter.AMENDMENT).mkdir()
    write_json(
        batch / "common/admission-train.json", {"gpu": 4, "uuid": "GPU-four", "gpu_sharing": False}
    )
    events, gate = [], {"passed": False}
    barrier = threading.Barrier(2) if parallel else None
    helpers = repair.controller
    helpers.write_json = write_json
    helpers.utc_now = lambda: "test-utc"

    def monitor(name, log, deadline, update):
        events.append(("monitor", name, deadline))
        assert not any(e[0] == "admit" for e in events)
        return train_exit

    def stage(batch, settings, phase, stage, token, deadline, row, update):
        events.append(("stage", phase, stage, row["gpu"], row["uuid"]))
        if phase != "common":
            assert gate["passed"]
            if barrier is not None and stage == "train":
                barrier.wait(timeout=3)
        if phase == "common" and stage == "gate":
            gate["passed"] = failure != "gate"
        return 3 if stage == failure else 0

    def require_gate(*args):
        assert gate["passed"], "Paired training cannot enter before the unchanged common gate"

    helpers.monitor_container = monitor
    helpers.require_common_gate = require_gate
    repair.verify_repaired_source = lambda *args: events.append(("source-check",))
    repair.run_stage = stage
    return repair, batch, events


@pytest.mark.parametrize("target", ["parent", "supplement"])
def test_default_plan_is_readonly_even_without_workspace(tmp_path, target):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/share_ttcd_gpu4.py"),
            "--target",
            target,
            "--workspace",
            str(tmp_path / "absent"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert not plan["executed"] and plan["gpu"] == 4 and plan["gpu_sharing"]
    assert (
        not plan["model_restarted"]
        and not plan["preparation_repeated"]
        and not plan["automatic_retry"]
    )
    assert list(tmp_path.iterdir()) == []


def test_adoption_never_restarts_common_and_original_gate_precedes_both_threads(tmp_path):
    repair, batch, events = make_repair(tmp_path, parallel=True)
    settings, before = config(), config()
    original_pipeline, original_arm_stage = repair.execute_pipeline, repair.controller.ARM_STAGES
    deadline = time.monotonic() + 120
    resources = fake_resources(events)
    adapter.install_parent_resources(
        repair, resources, tmp_path / "amendment.json", stop_record(), deadline
    )
    assert repair.execute_pipeline is original_pipeline
    assert repair.controller.ARM_STAGES is original_arm_stage
    assert repair.execute_pipeline(batch, settings, None, None, tmp_path / "token") == 0
    stages = [e[1:3] for e in events if e[0] == "stage"]
    assert stages[:3] == [("common", "train-audit"), ("common", "gate"), ("common", "cloud-audit")]
    assert ("common", "preflight") not in stages and ("common", "train") not in stages
    assert sum(e[0] == "monitor" for e in events) == 1
    assert next(e for e in events if e[0] == "monitor")[2] == deadline
    for phase in ("single", "stream"):
        assert [s for p, s in stages if p == phase] == list(original_arm_stage)
    assert all(e[3:] == (4, "GPU-four") for e in events if e[0] == "stage")
    assert settings == before
    assert read_exit(batch)["adopted"] and read_exit(batch)["returncode"] == 0
    assert not adapter.read_json(batch / "common/admission-train.json")["gpu_sharing"]
    adopted = adapter.read_json(batch / adapter.AMENDMENT / "adopted-common-admission.json")
    assert adopted["gpu_sharing"] and adopted["cpu_readmission"] is False


def read_exit(batch):
    return adapter.read_json(batch / "common/exit-train.json")


@pytest.mark.parametrize("failure,train_exit", [("gate", 0), (None, 7)])
def test_failure_preserved_without_paired_updates_or_retry(tmp_path, failure, train_exit):
    repair, batch, events = make_repair(tmp_path, failure=failure, train_exit=train_exit)
    adapter.install_parent_resources(
        repair,
        fake_resources(events),
        tmp_path / "amendment.json",
        stop_record(),
        time.monotonic() + 120,
    )
    assert repair.execute_pipeline(batch, config(), None, None, tmp_path / "token") != 0
    assert not any(e[0] == "stage" and e[1] in ("single", "stream") for e in events)
    saved = (batch / "common/exit-train.json").read_bytes()
    assert repair.run_phase(batch, config(), None, None, "common", tmp_path / "token") == 1
    assert (batch / "common/exit-train.json").read_bytes() == saved
    assert sum(e[0] == "monitor" for e in events) == 1


def test_foreign_gpu_pid_refusal_never_monitors_or_stops_any_container(tmp_path):
    repair, batch, events = make_repair(tmp_path)

    def foreign(*args):
        raise repair.controller.ResourceUnavailable("Foreign CUDA PID rejected")

    resources = fake_resources(events)
    resources.verify_gpu4_cohort = foreign
    adapter.install_parent_resources(
        repair, resources, tmp_path / "amendment.json", stop_record(), time.monotonic() + 120
    )
    assert repair.run_phase(batch, config(), None, None, "common", tmp_path / "token") == 1
    assert not any(e[0] in ("monitor", "stage", "admit") for e in events)
    assert (
        "Foreign CUDA PID"
        in adapter.read_json(batch / "common/controller-failure-v2.json")["traceback"]
    )
    assert not (batch / "common/exit-train.json").exists()


@pytest.mark.parametrize("state", ["missing", "live", "restart", "wrong-stage"])
def test_refuses_missing_or_live_old_controller_and_invalid_stop_record(
    tmp_path, monkeypatch, state
):
    directory = tmp_path / adapter.AMENDMENT
    directory.mkdir()
    record = stop_record()
    if state == "restart":
        record["no_model_restart"] = False
    if state == "wrong-stage":
        record["stage"] = "preflight"
    if state != "missing":
        write_json(directory / "stop-host-controller.json", record)

    def dead(pid, signal):
        assert signal == 0
        raise ProcessLookupError

    monkeypatch.setattr(adapter.os, "kill", (lambda *args: None) if state == "live" else dead)
    with pytest.raises(ValueError):
        adapter.stopped_controller(tmp_path, "parent")


def test_remaining_deadline_deducts_host_handoff_time_without_reset(monkeypatch):
    stop = stop_record()
    stop["deadline_remaining_seconds"] = 100
    stop["utc"] = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=20)
    ).isoformat()
    monkeypatch.setattr(adapter.time, "monotonic", lambda: 1000)
    assert 1079 < adapter.remaining_deadline(stop) <= 1080
    stop["deadline_remaining_seconds"] = 1
    with pytest.raises(TimeoutError, match="cannot be reset"):
        adapter.remaining_deadline(stop)


@pytest.mark.parametrize("invalid", [None, "identity", "preflight", "finished"])
def test_parent_adoption_requires_exact_existing_container_and_original_receipts(tmp_path, invalid):
    repair, batch, _ = make_repair(tmp_path)
    settings, stop = config(), stop_record()
    preflight = batch / "common/preflight-v2"
    preflight.mkdir()
    write_json(preflight / "exit-preflight.json", {"returncode": 0})
    write_json(preflight / "result-preflight.json", {"passed": invalid != "preflight"})
    write_json(batch / "common/claim-train.json", {"original": True})
    write_json(
        batch / "common/command-train.json", {"argv": ["docker", "--name", stop["container"]]}
    )
    repair.controller.inspect_container = lambda name: {
        "Id": "foreign-container-id" if invalid == "identity" else stop["container_id"],
        "State": {"Running": True},
        "Config": {"Image": settings["runtime"]["image"]},
    }
    if invalid == "finished":
        write_json(batch / "common/exit-train.json", {"returncode": 0})
    if invalid:
        with pytest.raises((ValueError, FileExistsError)):
            adapter.validate_parent(repair, batch, settings, None, None, stop)
    else:
        deadline = adapter.validate_parent(repair, batch, settings, None, None, stop)
        assert 3590 < deadline - time.monotonic() <= 3600
        assert not (batch / "common/exit-train.json").exists()


def test_resource_command_uses_same_uuid_and_unchanged_frozen_scientific_arguments(tmp_path):
    repair = adapter.load_module(
        ROOT / "scripts/repair_ttcd_verified_preflight.py", "gpu4_command_repair"
    )
    settings, before = config(), config()
    row = {"gpu": 4, "uuid": "GPU-four"}
    _, original = repair.build_command(
        settings, "stream", "train", tmp_path, tmp_path / "no-token", row
    )
    command = adapter.add_resource_metadata(
        original, settings["runtime"]["image"], row, tmp_path / "amendment.json"
    )
    assert command[command.index("--gpus") + 1] == "device=GPU-four"
    image_index = original.index(settings["runtime"]["image"])
    assert command[command.index(settings["runtime"]["image"]) :] == original[image_index:]
    assert "/frozen/configs/experiment.json" in command
    assert any("source-v2,target=/frozen,readonly" in x for x in command)
    assert "LM_ACTUAL_GPU_INDEX=4" in command and settings == before
    with pytest.raises(ValueError, match="GPU4 only"):
        adapter.add_resource_metadata(
            original, settings["runtime"]["image"], {"gpu": 7, "uuid": "GPU-seven"}, tmp_path
        )


def test_supplement_waits_for_both_parents_before_lock_and_never_prepares_again(
    tmp_path, monkeypatch
):
    settings = json.loads((ROOT / "configs/ttcd-double-new-eval-v1.json").read_text())
    settings["runtime"]["persistent_root"] = str(tmp_path / "persistent")
    workspace, batch = tmp_path / "workspace", tmp_path / "persistent" / adapter.SUPPLEMENT
    (workspace / "configs").mkdir(parents=True)
    (batch / "source/configs").mkdir(parents=True)
    (batch / "preparation").mkdir()
    (batch / adapter.AMENDMENT).mkdir()
    write_json(workspace / f"configs/{adapter.SUPPLEMENT}.json", settings)
    write_json(batch / "source/configs/double-new-eval.json", settings)
    write_json(batch / "source/configs/experiment.json", {"batch": adapter.PARENT})
    write_json(batch / "preparation/result-prepare.json", {"passed": True})
    events, barrier = [], threading.Barrier(2)
    helpers = SimpleNamespace(
        write_json=write_json, utc_now=lambda: "test", monitor_container=lambda *args: 0
    )

    def run_arm(settings, arm, stages, *args):
        events.append(("arm", arm, stages))
        barrier.wait(timeout=3)
        return int(arm == "stream")

    controller = SimpleNamespace(
        ARMS=("single", "stream"),
        STAGES=("preflight", "evaluate", "audit", "cloud-audit"),
        validate_settings=lambda *args: None,
        verify_snapshot=lambda *args, **kwargs: events.append(("verify",)),
        load_resources=lambda *args: (helpers, None),
        wait_dependencies=lambda *args: events.append(("both-parents-complete",)),
        build_command=lambda *args: None,
        run_arm=run_arm,
    )

    @contextlib.contextmanager
    def acquire():
        assert ("both-parents-complete",) in events
        events.append(("lock",))
        yield

    resources = fake_resources(events)
    resources.acquire_gpu4_lock = acquire
    monkeypatch.setattr(adapter, "load_module", lambda *args: controller)
    monkeypatch.setattr(adapter, "stopped_controller", lambda *args: stop_record())
    monkeypatch.setattr(adapter, "record_amendment", lambda *args: tmp_path / "amendment.json")
    # A missing prepare/freeze API is deliberate: adopting a prepared batch cannot call it.
    assert (
        adapter.execute_supplement(workspace, tmp_path / "token", resources, tmp_path / "resources")
        == 1
    )
    assert {e[1] for e in events if e[0] == "arm"} == {"single", "stream"}
    assert all(e[2] == list(controller.STAGES) for e in events if e[0] == "arm")
    state = adapter.read_json(batch / adapter.AMENDMENT / "controller-state.json")
    assert state["state"] == "needs_attention" and state["exit_codes"] == {"single": 0, "stream": 1}
    assert state["scientific_training_updates"] == 0


def test_supplement_threads_share_same_gpu4_without_nested_exclusive_lock(tmp_path):
    controller = adapter.load_module(
        ROOT / "scripts/launch_ttcd_double_new_eval.py", "gpu4_supplement_test"
    )
    helpers = SimpleNamespace(monitor_container=lambda *args: 0)
    events, resources = [], None
    resources = fake_resources(events)
    original_stage = controller.run_stage
    adapter.install_supplement_resources(
        controller, helpers, resources, tmp_path / "amendment.json"
    )
    settings = json.loads((ROOT / "configs/ttcd-double-new-eval-v1.json").read_text())
    before = copy.deepcopy(settings)
    for arm in controller.ARMS:
        held, row = controller.wait_gpu(settings, arm, helpers, None, time.monotonic() + 10)
        assert isinstance(held, contextlib.nullcontext)
        assert (row["gpu"], row["uuid"]) == (4, "GPU-four")
        assert row["cpu_set"] == settings["runtime"]["cpu_sets"][arm]
    assert controller.run_stage is original_stage and settings == before


def test_cohort_lock_wait_handles_parent_completion_release_race(monkeypatch):
    entered, updates = [], []

    @contextlib.contextmanager
    def acquire():
        entered.append(True)
        if len(entered) == 1:
            raise BlockingIOError("parent is just finishing")
        yield

    monkeypatch.setattr(adapter.time, "sleep", lambda *args: None)
    with adapter.wait_cohort_lock(
        SimpleNamespace(acquire_gpu4_lock=acquire), time.monotonic() + 10, updates.append
    ):
        pass
    assert len(entered) == 2 and updates == [{"state": "waiting_for_parent_gpu4_cohort_lock"}]
