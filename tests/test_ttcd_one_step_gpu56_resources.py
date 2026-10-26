"""Bound external identities and reservations, using the actual controller's leases."""

import fcntl
import importlib.util
import json
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load("gpu56_policy", "scripts/ttcd_one_step_gpu56_resources.py")


def snapshot():
    return {
        "schema": "ttcd-one-step-gpu56-sharing-v1",
        "batch": policy.BATCH,
        "allowed_gpus": [5, 6],
        "external_processes": [
            {
                "gpu": gpu,
                "uuid": f"GPU-{gpu}",
                "pid": pid,
                "starttime": pid + 1,
                "uid": 1047,
                "exe": "/workspace/envs/python/bin/python",
                "exe_source": "nvidia-smi process_name",
                "proc_exe_readable": False,
            }
            for pid, gpu in policy.EXTERNAL_PIDS.items()
        ],
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    controller = load("gpu56_real_controller", "scripts/launch_ttcd_one_step_training.py")
    plan = json.loads((ROOT / f"configs/{policy.BATCH}.json").read_text())
    batch = tmp_path / policy.BATCH
    for arm in controller.ARCHITECTURES:
        (batch / arm).mkdir(parents=True)
        (batch / arm / "source-manifest.json").write_text("{}")
    identities = snapshot()
    indexed = {record["pid"]: record for record in identities["external_processes"]}

    def identity(pid, process_name):
        return {**indexed[pid], "exe": process_name}

    monkeypatch.setattr(policy, "process_identity", identity)
    memory = {
        (record["uuid"], record["pid"]): {
            "uuid": record["uuid"],
            "pid": record["pid"],
            "used_memory_mib": {3168776: 53262, 3164122: 35480, 3165961: 6312}[record["pid"]],
            "process_name": record["exe"],
        }
        for record in identities["external_processes"]
    }
    rows = {
        gpu: {
            "uuid": f"GPU-{gpu}",
            "total_mib": 97887,
            "free_mib": {5: 43980, 6: 55445}[gpu],
            "memory": {5: 53907, 6: 42442}[gpu],
            "utilization": 50,
        }
        for gpu in (5, 6)
    }
    processes = [{"uuid": uuid, "pid": pid} for uuid, pid in memory]
    monkeypatch.setattr(policy, "memory_snapshot", lambda: memory)
    monkeypatch.setattr(controller.resources, "gpu_snapshot", lambda: (rows, processes))
    monkeypatch.setattr(controller.resources, "admit_cpu", lambda *_: ([], 999999999))
    slots = controller.CohortSlots(tmp_path / "locks")
    # Exercise compatibility even with the non-reentrant lock used by older helpers.
    slots.guard = threading.Lock()
    policy.install(controller, identities)
    return controller, plan, batch, slots, rows, processes, memory, identities


def test_actual_controller_leases_spread_across_user_authorized_gpus(setup):
    controller, plan, batch, slots, *_ = setup
    held_t, transformer = controller.acquire_resources(
        plan, "transformer", "train", lambda _: None, batch, slots
    )
    held_l, loop = controller.acquire_resources(plan, "loop", "train", lambda _: None, batch, slots)
    assert (transformer["gpu"], loop["gpu"]) == (5, 6)
    assert transformer["external_memory_mib"] == 53262
    assert transformer["new_reservation_mib"] == 28672
    assert not transformer["wallclock_is_exclusive_gpu_benchmark"]
    held_l.close()
    held_t.close()
    assert not slots.arms and not slots.handles


def test_launching_leases_are_reserved_once_and_global_lock_cleans_up(setup, monkeypatch):
    controller, plan, batch, slots, rows, *_ = setup
    rows.pop(5)
    first, _ = controller.acquire_resources(plan, "loop", "train", lambda _: None, batch, slots)
    second, admitted = controller.acquire_resources(
        plan, "discoloop", "train", lambda _: None, batch, slots
    )
    assert admitted["existing_reservations_mib"] == 20480
    assert admitted["reserved_architectures"] == ["discoloop", "loop"]
    ticks = iter((0, 0, 400000, 400000))
    monkeypatch.setattr(policy.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(policy.time, "sleep", lambda _: None)
    updates = []
    with pytest.raises(TimeoutError):
        controller.acquire_resources(plan, "transformer", "train", updates.append, batch, slots)
    assert "reservations exceed capacity" in updates[0]["reason"]
    with (slots.directory / "gpu-6.lock").open("a") as outside:
        with pytest.raises(BlockingIOError):
            fcntl.flock(outside, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second.close()
        first.close()
        fcntl.flock(outside, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not slots.arms


@pytest.mark.parametrize("changed", ["starttime", "uid", "exe"])
def test_external_pid_identity_changes_are_refused(setup, monkeypatch, changed):
    controller, plan, batch, _, rows, processes, _, identities = setup
    expected = identities["external_processes"][0]

    def wrong_identity(pid, name):
        observed = {**expected, "pid": pid, "exe": name}
        observed[changed] = "different-path" if changed == "exe" else observed[changed] + 1
        return observed

    monkeypatch.setattr(policy, "process_identity", wrong_identity)
    with pytest.raises(controller.resources.ResourceUnavailable, match="replaced or changed"):
        controller.gpu_cohort(batch, plan, 5, rows[5], processes, [])


def test_new_unknown_cuda_process_is_refused(setup):
    controller, plan, batch, _, rows, processes, memory, _ = setup
    processes.append({"uuid": "GPU-5", "pid": 999999})
    memory[("GPU-5", 999999)] = {
        "uuid": "GPU-5",
        "pid": 999999,
        "used_memory_mib": 100,
        "process_name": "/other/python",
    }
    with pytest.raises(controller.resources.ResourceUnavailable, match="outside CUDA processes"):
        controller.gpu_cohort(batch, plan, 5, rows[5], processes, [])


def test_own_container_validation_remains_mandatory(setup):
    controller, plan, batch, _, rows, processes, _, _ = setup
    container = {
        "Name": f"/{policy.BATCH}-loop-train",
        "Id": "our-container",
        "Config": {"User": "root", "Image": plan["runtime"]["image"]},
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["GPU-5"]}]},
        "Mounts": [],
    }
    with pytest.raises(controller.resources.ResourceUnavailable, match="differs in identity"):
        controller.gpu_cohort(batch, plan, 5, rows[5], processes, [container])


def test_scope_excludes_other_gpus_and_cpu_stage_keeps_original_logic(setup, monkeypatch):
    controller, plan, batch, slots, rows, processes, *_ = setup
    with pytest.raises(ValueError, match="outside CUDA5/6"):
        policy.observed_external(snapshot(), 8, rows[5], processes, {})
    monkeypatch.setattr(policy, "memory_snapshot", lambda: pytest.fail("CPU stage queried CUDA"))
    held, admitted = controller.acquire_resources(
        plan, "loop", "cloud-audit", lambda _: None, batch, slots
    )
    assert admitted["device"] == "cpu"
    held.close()


def test_process_identity_reports_nvml_path_provenance_without_readlink(tmp_path):
    process = tmp_path / "123"
    process.mkdir()
    fields = ["S", *["0"] * 18, "987654"]
    (process / "stat").write_text("123 (command with spaces) " + " ".join(fields))
    identity = policy.process_identity(123, "/nvml/reported/python", tmp_path)
    assert identity["starttime"] == 987654
    assert identity["exe"] == "/nvml/reported/python"
    assert identity["exe_source"] == "nvidia-smi process_name"
    assert identity["proc_exe_readable"] is False
