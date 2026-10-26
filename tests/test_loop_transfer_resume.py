"""Resource expansion must adopt work, preserve failures, and keep seeds last."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "resume", ROOT / "scripts/resume_loop_transfer.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_admission_can_resume_but_ambiguous_launch_cannot(tmp_path, recovery):
    stage = tmp_path / "grid"
    assert recovery.stage_action(stage) == "unstarted"
    stage.mkdir()
    assert recovery.stage_action(stage) == "unstarted"
    (stage / "command.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Ambiguous"):
        recovery.stage_action(stage)
    (stage / "container.json").write_text('{"id":"retained"}')
    assert recovery.stage_action(stage) == "adopt"
    (stage / "exit.json").write_text('{"code":1}')
    with pytest.raises(RuntimeError, match="no automatic retry"):
        recovery.stage_action(stage)
    (stage / "exit.json").write_text('{"code":0}')
    assert recovery.stage_action(stage) == "complete"


def test_capacity_counts_other_batches_and_pending_admissions(recovery):
    assert recovery.capacity_available("new", {"state", "a", "b"}, set(), 4)
    assert not recovery.capacity_available("fifth", {"state", "a", "b"}, {"new"}, 4)
    assert recovery.capacity_available("a", {"state", "a", "b", "new"}, set(), 4)


def test_adopt_running_container_only_waits_never_restarts(tmp_path, recovery, monkeypatch):
    controller = object.__new__(recovery.Controller)
    controller.base = tmp_path
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        return type("Result", (), {"stdout": "0\n"})()

    monkeypatch.setattr(recovery.original, "command", command)
    controller.adopt("existing", "train", {"Id": "kept", "State": {"Status": "running"}})
    assert calls == [["docker", "--context", "lm-memory", "wait", "kept"]]
    receipt = json.loads((tmp_path / "execution/existing/train/exit.json").read_text())
    assert receipt["adopted_without_restart"]
    assert receipt["code"] == 0


def test_created_or_failed_container_never_restarted(tmp_path, recovery):
    controller = object.__new__(recovery.Controller)
    controller.base = tmp_path
    with pytest.raises(RuntimeError, match="Refusing to restart"):
        controller.adopt("j", "train", {"Id": "kept", "State": {"Status": "created"}})
    with pytest.raises(RuntimeError, match="failed"):
        controller.adopt("j", "train", {"Id": "kept", "State": {"Status": "exited", "ExitCode": 1}})


def test_resume_reuses_successful_train_and_only_returns_unstarted_grid(
    tmp_path, recovery, monkeypatch
):
    controller = object.__new__(recovery.Controller)
    controller.base = tmp_path
    controller.tasks = {"j": {"stages": ["train", "grid"]}}
    controller.checked = set()
    stage = tmp_path / "execution/j/train"
    stage.mkdir(parents=True)
    (stage / "container.json").write_text('{"id":"kept"}')
    (stage / "exit.json").write_text('{"code":0}')
    monkeypatch.setattr(
        controller, "existing_container", lambda *a: {"State": {"Status": "exited", "ExitCode": 0}}
    )
    assert controller.next_stage("j") == ("grid", "unstarted")


def test_amendment_preserves_whitelist_budget_phase_order_and_cpu_separation(tmp_path, recovery):
    manifest = json.loads((ROOT / "results/loop-queue-transfer-20261021/tasks.json").read_text())
    early, seeds = [], []
    for task in manifest["training_jobs"]:
        (seeds if task["condition"] in ("init-1", "init-2") else early).append(task["job"])
    plan = {
        "gpus": [5, 6, 8],
        "cpu_sets": ["70,71", "72,73", "74,75"],
        "phases": [early + [j["job"] for j in manifest["evaluation_only_jobs"]], seeds],
    }
    for name, data in [("schedule.json", plan), ("tasks.json", manifest)]:
        (tmp_path / name).write_text(json.dumps(data))
    amendment = {
        "original_sha256": {
            n: recovery.original.digest(tmp_path / n) for n in ("schedule.json", "tasks.json")
        },
        "additional_gpus": [2, 3],
        "additional_cpu_sets": {"2": "78,79", "3": "80,81"},
    }
    assert recovery.validate_amendment(tmp_path, amendment) == plan
    amendment["additional_cpu_sets"]["2"] = "70,71"
    with pytest.raises(ValueError, match="overlap"):
        recovery.validate_amendment(tmp_path, amendment)
    amendment["additional_cpu_sets"]["2"] = "78,79"
    plan["phases"][0].append(plan["phases"][1].pop())
    (tmp_path / "schedule.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        recovery.validate_amendment(tmp_path, amendment)
