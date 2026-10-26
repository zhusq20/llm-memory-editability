"""Monitor existing model processes while admitting the one waiting architecture."""

import contextlib
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


handoff = load("gpu56_handoff", ROOT / "scripts/share_ttcd_one_step_training_gpu56.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_24gib_transformer_reservation_fits_gpu6_with_running_loop():
    total, external, loop = 97251, 41792, 20480
    assert external + loop + 28672 + handoff.HEADROOM_MIB > total
    assert external + loop + handoff.RESERVATIONS_MIB["transformer"] + handoff.HEADROOM_MIB <= total
    assert handoff.RESERVATIONS_MIB["loop"] == handoff.RESERVATIONS_MIB["discoloop"] == 20480
    assert 46406 >= handoff.RESERVATIONS_MIB["transformer"] + handoff.HEADROOM_MIB


def test_real_gpu_and_cpu_leases_are_taken_before_workers_without_reentrant_lock(tmp_path):
    controller = load(
        "gpu56_handoff_real_controller", ROOT / "scripts/launch_ttcd_one_step_training.py"
    )
    controller.SLOTS = controller.CohortSlots(tmp_path / "locks")
    controller.SLOTS.guard = threading.Lock()
    plan = {"runtime": {"cpu_sets": {"loop": [68, 69], "discoloop": [72, 73]}}}
    snapshot = {"models": {"loop": {"gpu": 6}, "discoloop": {"gpu": 5}}}
    leases = handoff.adopt_leases(controller, plan, tmp_path, snapshot)
    assert controller.SLOTS.arms == {6: {"loop"}, 5: {"discoloop"}}
    assert len(list((tmp_path / "locks").glob("cpu-*.lock"))) == 4
    for held in leases.values():
        held.close()
    assert not controller.SLOTS.arms and not controller.SLOTS.handles


def test_existing_train_containers_are_only_monitored_and_never_resubmitted(tmp_path, monkeypatch):
    called, monitored = [], []
    controller = SimpleNamespace(
        STAGES=("prepare", "preflight", "train", "train-audit", "cloud-audit"),
        run_arm=lambda _p, arm, *_: called.append((arm, "train")),
        run_stage=lambda _p, arm, stage, *_: called.append((arm, stage)),
        resources=SimpleNamespace(utc_now=lambda: "now"),
        write_json=write,
    )
    for arm in handoff.RUNNING:
        (tmp_path / arm).mkdir()
    monkeypatch.setattr(handoff, "monitor_existing", lambda _c, _p, arm, *_: monitored.append(arm))
    leases = {arm: contextlib.ExitStack() for arm in handoff.RUNNING}
    handoff.install_adopted_arms(controller, {}, leases)
    for arm in (*handoff.RUNNING, "transformer"):
        controller.run_arm({}, arm, tmp_path, Path("/token"))
    assert monitored == ["loop", "discoloop"]
    assert ("loop", "train") not in called and ("discoloop", "train") not in called
    assert ("transformer", "train") in called
    assert not leases


@pytest.fixture
def captured(tmp_path, monkeypatch):
    batch = tmp_path / handoff.BATCH
    write(batch / "controller-process.json", {"pid": handoff.OLD_PID})
    write(batch / "source-manifest.json", {"same": "science"})
    tasks = {}
    for arm in ("transformer", *handoff.RUNNING):
        path = batch / arm / "source/configs/task.json"
        write(path, {"arm": arm})
        tasks[arm] = handoff.sha256(path)
    models = {
        arm: {"id": f"{arm}-container", "state_pid": 123 + i, "progress": {"step": 20}}
        for i, arm in enumerate(handoff.RUNNING)
    }
    snapshot = {
        "batch": handoff.BATCH,
        "old_host": {"pid": handoff.OLD_PID},
        "old_process_sha256": handoff.sha256(batch / "controller-process.json"),
        "science_manifest_sha256": handoff.sha256(batch / "source-manifest.json"),
        "source_tasks": tasks,
        "models": models,
        "plan": {},
        "transformer": {"claim_sha256": "same-claim"},
    }
    controller = SimpleNamespace(
        ARCHITECTURES=("transformer", "loop", "discoloop"),
        transport=SimpleNamespace(verify_snapshot=lambda *_: None),
    )
    monkeypatch.setattr(handoff, "alive", lambda _: False)
    monkeypatch.setattr(
        handoff, "check_transformer_waiting", lambda *_: {"claim_sha256": "same-claim"}
    )
    return batch, controller, snapshot


@pytest.mark.parametrize("key", ["id", "state_pid"])
def test_container_or_process_replacement_is_refused(captured, monkeypatch, key):
    batch, controller, snapshot = captured

    def changed(_c, _p, _b, arm):
        return {**snapshot["models"][arm], key: "replaced"}

    monkeypatch.setattr(handoff, "binding", changed)
    with pytest.raises(ValueError, match="Running model identity changed"):
        handoff.verify_snapshot(controller, batch, snapshot)


def test_model_progress_may_advance_but_never_roll_back(captured, monkeypatch):
    batch, controller, snapshot = captured
    monkeypatch.setattr(
        handoff,
        "binding",
        lambda _c, _p, _b, arm: {**snapshot["models"][arm], "progress": {"step": 21}},
    )
    handoff.verify_snapshot(controller, batch, snapshot)
    monkeypatch.setattr(
        handoff,
        "binding",
        lambda _c, _p, _b, arm: {**snapshot["models"][arm], "progress": {"step": 19}},
    )
    with pytest.raises(ValueError, match="progress moved backwards"):
        handoff.verify_snapshot(controller, batch, snapshot)


def test_transformer_pending_claim_is_retained_and_existing_science_untouched(tmp_path):
    batch = tmp_path / handoff.BATCH
    for name in ("controller-process.json", "controller-state.json", "schedule-amendment.json"):
        write(batch / name, {"original": True})
    write(batch / "transformer/stages/train/claim.json", {"pending": True})
    model = batch / "loop/training-run/training/checkpoint-0000000.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"existing model must remain")
    (batch / handoff.HANDOFF).mkdir(parents=True)
    before = model.read_bytes()
    archive = handoff.archive_pending_claim(batch)
    assert read(archive / "transformer-pending-train/claim.json") == {"pending": True}
    assert model.read_bytes() == before
    assert not (batch / "transformer/stages/train").exists()


def read(path):
    return json.loads(path.read_text())


def test_handoff_does_not_reset_container_deadline(monkeypatch):
    monkeypatch.setattr(handoff.time, "time", lambda: 3600)
    monkeypatch.setattr(handoff.time, "monotonic", lambda: 100)
    assert handoff.remaining_deadline("1970-01-01T00:00:00Z") == 100 + 47 * 3600


def test_mount_order_is_normalized_without_changing_source_or_write_permissions():
    mounts = [
        {"Source": "/same/output", "Destination": "/output", "RW": True},
        {"Source": "/original/data", "Destination": "/frozen/data", "RW": False},
        {"Source": "/original/results", "Destination": "/frozen/results", "RW": False},
    ]
    normalized = handoff.canonical_mounts(mounts)
    assert normalized == handoff.canonical_mounts(list(reversed(mounts)))
    assert {tuple(sorted(mount.items())) for mount in normalized} == {
        tuple(sorted(mount.items())) for mount in mounts
    }
    assert mounts[0]["Destination"] == "/output"
    assert [mount["Destination"] for mount in normalized] == [
        "/frozen/data",
        "/frozen/results",
        "/output",
    ]
