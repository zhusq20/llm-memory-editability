"""Shared scheduling must adopt existing writes and reject external GPU processes."""

from __future__ import annotations

import concurrent.futures
import copy
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shared_handoff", ROOT / "scripts/resume_ce_cross_sample_ablation_shared.py"
)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    controller = shared.load_controller(ROOT)
    monkeypatch.setattr(controller.transport, "INPUT_ROOT", tmp_path)
    plan = json.loads((ROOT / "configs/ce-cross-sample-ablation-v1.json").read_text())
    batch = tmp_path / shared.BATCH
    plan["execution"]["persistent_root"] = str(batch)
    return shared.Handoff(controller, plan, batch, batch / "source-v1")


def launch(handoff, *, stage="science", running=False, exit_code=0):
    arm = "loop-no-bound"
    receipt = {
        "name": f"{shared.BATCH}-{arm}-{stage}",
        "id": "exact-container-id",
        "source_sha256": shared.SOURCE_SHA,
        "gpu": shared.GPU,
    }
    container = {
        "Id": receipt["id"],
        "Name": "/" + receipt["name"],
        "Config": {
            "Labels": {"lm.batch": shared.BATCH, "lm.architecture": arm, "lm.stage": stage},
            "User": "1015:1016",
            "Image": handoff.plan["execution"]["image"],
            "Env": [f"LM_SOURCE_MANIFEST_SHA256={shared.SOURCE_SHA}"],
            "Cmd": handoff.controller.command_for(handoff.plan, stage, arm),
        },
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": [shared.GPU["uuid"]]}]},
        "State": {
            "Running": running,
            "Paused": False,
            "Restarting": False,
            "Pid": 123,
            "ExitCode": exit_code,
            "StartedAt": "2026-10-17T01:00:00Z",
            "FinishedAt": "2026-10-17T01:01:00Z",
        },
    }
    handoff.controller.write(handoff.batch / arm / f"launch-{stage}.json", receipt)
    return arm, receipt, container


def test_adoption_never_recreates_or_starts_existing_science(handoff, monkeypatch):
    arm, _, container = launch(handoff)
    calls = []
    monkeypatch.setattr(handoff, "inspect", lambda _: copy.deepcopy(container))
    monkeypatch.setattr(
        handoff, "original_run", lambda *args, **kwargs: pytest.fail("never run old stage again")
    )

    def docker(*args, **kwargs):
        calls.append(args)
        assert args[0] == "logs"
        return subprocess.CompletedProcess(
            args, 0, stdout=b"original output", stderr=b"original stderr"
        )

    monkeypatch.setattr(handoff.transport, "docker", docker)
    assert (
        handoff.dispatch(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            arm,
            "science",
            gpu=shared.GPU,
            timeout=100,
        )
        == 60
    )
    assert [args[0] for args in calls] == ["logs"]
    duration = json.loads((handoff.batch / arm / "duration-science.json").read_text())
    assert duration["adopted_existing"] is True
    assert duration["exclusive_gpu_wall_benchmark"] is False
    assert (handoff.batch / arm / "science.log").read_bytes() == b"original outputoriginal stderr"


def test_completed_stage_reuses_exact_exit_and_duration(handoff, monkeypatch):
    arm, _, _ = launch(handoff)
    handoff.controller.write(handoff.batch / arm / "exit-science.json", {"ExitCode": 0})
    handoff.controller.write(
        handoff.batch / arm / "duration-science.json", {"elapsed_seconds": 12.5}
    )
    monkeypatch.setattr(
        handoff, "inspect", lambda _: pytest.fail("completed stage needs no Docker call")
    )
    assert (
        handoff.dispatch(
            handoff.plan, handoff.batch, handoff.source, shared.SOURCE_SHA, arm, "science"
        )
        == 12.5
    )


def test_failed_stage_is_never_retried(handoff, monkeypatch):
    arm, _, _ = launch(handoff)
    handoff.controller.write(handoff.batch / arm / "exit-science.json", {"ExitCode": 1})
    monkeypatch.setattr(handoff, "original_run", lambda *args, **kwargs: pytest.fail("no retry"))
    with pytest.raises(FileExistsError, match="never repeat"):
        handoff.dispatch(
            handoff.plan, handoff.batch, handoff.source, shared.SOURCE_SHA, arm, "science"
        )


def test_adopted_failure_records_original_exit_without_restart(handoff, monkeypatch):
    arm, _, container = launch(handoff, exit_code=2)
    monkeypatch.setattr(handoff, "inspect", lambda _: container)
    monkeypatch.setattr(
        handoff.transport,
        "docker",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=b"failed", stderr=b""),
    )
    with pytest.raises(RuntimeError, match="never retry"):
        handoff.dispatch(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            arm,
            "science",
            timeout=100,
        )
    assert json.loads((handoff.batch / arm / "exit-science.json").read_text())["ExitCode"] == 2


@pytest.mark.parametrize("changed", ["id", "name", "source", "label", "command", "gpu"])
def test_existing_container_identity_must_match_frozen_receipt(handoff, changed):
    arm, receipt, container = launch(handoff)
    if changed == "id":
        container["Id"] = "foreign"
    elif changed == "name":
        container["Name"] = "/foreign"
    elif changed == "source":
        container["Config"]["Env"] = ["LM_SOURCE_MANIFEST_SHA256=foreign"]
    elif changed == "label":
        container["Config"]["Labels"]["lm.batch"] = "foreign"
    elif changed == "command":
        container["Config"]["Cmd"] = ["other", "training"]
    else:
        container["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU-foreign"]
    with pytest.raises(ValueError):
        handoff.validate_container(container, arm, "science", receipt)


def test_admission_rejects_external_compute_pid(handoff, monkeypatch):
    monkeypatch.setattr(
        handoff, "owned_running", lambda: ({10, 20}, [("loop-no-bound", "science")])
    )
    gpu = {**shared.GPU, "compute_pids": [10, 999], "free_mib": 90000}
    result = handoff.admission(gpu)
    assert result["passed"] is False
    assert result["external_pids"] == [999]


def test_admission_allows_only_owned_processes_with_memory_headroom(handoff, monkeypatch):
    monkeypatch.setattr(
        handoff, "owned_running", lambda: ({10, 20}, [("loop-no-bound", "science")])
    )
    gpu = {**shared.GPU, "compute_pids": [10, 20], "free_mib": 90000}
    assert handoff.admission(gpu)["passed"] is True
    gpu["free_mib"] = 16000
    assert handoff.admission(gpu)["passed"] is False


def test_reservation_of_adopted_stage_leaves_two_slots():
    pool = shared.Pool()
    pool.reserve(("loop-no-bound", "science"))
    assert pool.active == 1
    with pool.slot(("second", "gate")), pool.slot(("third", "gate")):
        assert pool.active == 3
    with pool.slot(("loop-no-bound", "science")):
        assert pool.active == 1
    assert pool.active == 0
    assert pool.maximum_observed == 3


def test_pool_never_admits_more_than_three_concurrent_stages():
    pool = shared.Pool()
    barrier = threading.Barrier(3)

    def work(index):
        with pool.slot((str(index), "gate")):
            if index < 3:
                barrier.wait(timeout=2)
            time.sleep(0.01)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(work, range(6)))
    assert pool.maximum_observed == 3
    assert pool.active == 0


def test_unreceipted_outputs_cannot_be_repeated(handoff):
    output = handoff.batch / "loop-no-bound/output/science"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="Unreceipted stage output"):
        handoff.dispatch(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            "loop-no-bound",
            "science",
        )


def test_never_started_container_is_not_implicitly_started():
    with pytest.raises(ValueError, match="never started"):
        shared.epoch("0001-01-01T00:00:00Z")


def test_owned_container_exit_between_inspect_and_top_is_not_an_arm_failure(handoff, monkeypatch):
    _, _, container = launch(handoff, running=True)
    exited = copy.deepcopy(container)
    exited["State"]["Running"] = False
    inspections = iter([container, exited])
    monkeypatch.setattr(handoff, "inspect", lambda _: next(inspections))

    def docker(*args, **kwargs):
        if args[0] == "ps":
            return subprocess.CompletedProcess(args, 0, stdout=b"exact-container-id\n", stderr=b"")
        assert args[0] == "top"
        raise subprocess.CalledProcessError(1, args, stderr=b"container is not running")

    monkeypatch.setattr(handoff.transport, "docker", docker)
    assert handoff.owned_running() == (set(), [])


def test_stale_just_exited_gpu_pid_waits_without_starting_new_work(handoff, monkeypatch):
    snapshots = iter(
        [
            {**shared.GPU, "compute_pids": [999], "free_mib": 90000},
            {**shared.GPU, "compute_pids": [], "free_mib": 90000},
        ]
    )
    events = []
    monkeypatch.setattr(handoff.transport, "gpu_inventory", lambda: [next(snapshots)])
    monkeypatch.setattr(handoff, "owned_running", lambda: (set(), []))
    monkeypatch.setattr(shared.time, "sleep", lambda seconds: events.append("wait"))
    monkeypatch.setattr(
        handoff, "dispatch", lambda *args, **kwargs: events.append("launch_after_admission") or 5
    )
    assert (
        handoff.gpu_stage(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            "loop-no-bound",
            "gate",
            100,
        )
        == 5
    )
    assert events == ["wait", "launch_after_admission"]
    state = json.loads((handoff.batch / "loop-no-bound/controller-state.json").read_text())
    assert state["external_sharing_admitted"] is False
    assert state["admission"]["external_pids"] == [999]


def test_loading_frozen_controllers_cannot_add_bytecode_files(tmp_path, monkeypatch):
    source = tmp_path / "frozen-source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    for filename in ("launch_ce_cross_sample_ablation.py", "launch_ce_cross_sample.py"):
        (scripts / filename).write_bytes((ROOT / "scripts" / filename).read_bytes())
    before = {str(path.relative_to(source)) for path in source.rglob("*")}
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    module = shared.load_controller(source)
    assert module.BATCH == shared.BATCH
    assert sys.dont_write_bytecode is True
    assert {str(path.relative_to(source)) for path in source.rglob("*")} == before
    assert not list(source.rglob("*.pyc"))
    assert not list(source.rglob("__pycache__"))


@pytest.mark.parametrize(
    "prefix", [b"error: no such object: ", b"Error response from daemon: No such container: "]
)
def test_actual_missing_container_response_allows_one_new_launch(handoff, monkeypatch, prefix):
    calls = []
    name = f"{shared.BATCH}-loop-carry-adam-gate"

    def inspect_missing(actual):
        assert actual == name
        raise subprocess.CalledProcessError(
            1, ["docker", "inspect", actual], stderr=prefix + actual.encode() + b"\n"
        )

    monkeypatch.setattr(handoff, "inspect", inspect_missing)
    monkeypatch.setattr(
        handoff, "original_run", lambda *args, **kwargs: calls.append((args, kwargs)) or 7
    )
    assert handoff.container_exists(name) is False
    assert (
        handoff.dispatch(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            "loop-carry-adam",
            "gate",
            gpu=shared.GPU,
            timeout=100,
        )
        == 7
    )
    assert len(calls) == 1
    assert calls[0][0][4:6] == ("loop-carry-adam", "gate")


def test_unavailable_docker_daemon_never_allows_a_new_launch(handoff, monkeypatch):
    def inspect_unavailable(name):
        raise subprocess.CalledProcessError(
            1,
            ["docker", "inspect", name],
            stderr=b"Cannot connect to the Docker daemon at unix:///run/docker-lm-memory/docker.sock\n",
        )

    monkeypatch.setattr(handoff, "inspect", inspect_unavailable)
    monkeypatch.setattr(
        handoff,
        "original_run",
        lambda *args, **kwargs: pytest.fail("daemon error must never be treated as absence"),
    )
    with pytest.raises(subprocess.CalledProcessError):
        handoff.dispatch(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            "loop-carry-adam",
            "gate",
            gpu=shared.GPU,
            timeout=100,
        )
    assert not (handoff.batch / "loop-carry-adam/launch-gate.json").exists()


def test_existing_unreceipted_container_is_never_recreated(handoff, monkeypatch):
    monkeypatch.setattr(handoff, "inspect", lambda name: {"Name": name})
    monkeypatch.setattr(
        handoff, "original_run", lambda *args, **kwargs: pytest.fail("never recreate")
    )
    with pytest.raises(FileExistsError, match="Unreceipted container"):
        handoff.dispatch(
            handoff.plan,
            handoff.batch,
            handoff.source,
            shared.SOURCE_SHA,
            "loop-carry-adam",
            "gate",
            gpu=shared.GPU,
            timeout=100,
        )
