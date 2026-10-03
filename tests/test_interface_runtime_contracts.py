"""Docker isolation and W&B identity/audit contracts, without launching Docker."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_memory_editability.experiment_tracking import write_json
from llm_memory_editability.interface_tracking import AuditedRunTracker, drain_runs

RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_memory_interface_next.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_memory_interface_next", RUNNER_PATH)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


class FakeRun:
    def __init__(self, options):
        self.options, self.config, self.summary = options, dict(options["config"]), {}
        self.logs, self.exit_code, self.url = [], None, "https://wandb.invalid/test"

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, metrics, step):
        self.logs.append((step, metrics))

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class FakeSDK:
    def Settings(self, **options):
        return options

    def init(self, **options):
        return FakeRun(options)


def make_tracker(tmp_path):
    path = tmp_path / "runs" / "run1"
    write_json(
        path / "run.json",
        {
            "spec": {"alphas": [0.1]},
            "pid": 1,
            "gpu": 0,
            "parent_spec": {"world": 42, "arm": "bridge_ce"},
            "cases": {"main": {"condition": "self_decode", "alpha": 0.1}},
        },
    )
    write_json(path / "container-runtime.json", {"host_pid": 9876, "physical_gpu": 3})
    write_json(
        path / "learning.json",
        [
            {
                "step": 1,
                "optimizer_updates": 0,
                "case": "main",
                "condition": "self_decode",
                "variant": "norm_matched",
                "alpha": 0.1,
                "wall_seconds": 1,
                "metrics": {},
            },
            {
                "step": 2,
                "optimizer_updates": 0,
                "case": "control",
                "condition": "wrong_entity",
                "variant": "paper_unit",
                "alpha": 0.1,
                "wall_seconds": 2,
                "metrics": {},
            },
        ],
    )
    write_json(path / "status.json", {"state": "running"})
    return AuditedRunTracker(
        FakeSDK(),
        tmp_path,
        path,
        {
            "mode": "offline",
            "entity": "zhusq20",
            "project": "llm-memory-editability",
            "system_metrics": True,
        },
    )


def test_tracking_uses_host_pid_and_physical_gpu_and_retains_parent_config(tmp_path):
    tracker = make_tracker(tmp_path)
    settings = tracker.run.options["settings"]
    assert settings["x_stats_pid"] == 9876
    assert settings["x_stats_gpu_device_ids"] == (3,)
    assert tracker.run.config["parent_spec"]["world"] == 42
    assert tracker.run.config["cases"]["main"]["condition"] == "self_decode"


def test_tracking_keeps_intervention_labels_and_never_calls_evaluation_training(tmp_path):
    tracker = make_tracker(tmp_path)
    assert not tracker.poll()
    metrics = tracker.run.logs[-1][1]
    assert metrics["condition/case"] == "control"
    assert metrics["condition/condition"] == "wrong_entity"
    assert metrics["condition/variant"] == "paper_unit"
    assert metrics["training/optimizer_updates"] == 0
    assert metrics["evaluation/index"] == 2
    assert metrics["condition/step_unit"] == "evaluation_index"
    assert metrics["perf/updates_per_second"] == 0
    assert "perf/training_ms_per_update" not in metrics


def test_missing_container_runtime_cannot_silently_monitor_the_sidecar(tmp_path):
    tracker = make_tracker(tmp_path)
    (tracker.path / "container-runtime.json").unlink()
    with pytest.raises(FileNotFoundError):
        AuditedRunTracker(
            FakeSDK(),
            tmp_path,
            tracker.path,
            {
                "mode": "offline",
                "entity": "zhusq20",
                "project": "llm-memory-editability",
                "system_metrics": True,
            },
        )


def test_failed_audit_file_does_not_mark_run_successful(tmp_path):
    tracker = make_tracker(tmp_path)
    write_json(tracker.path / "complete.json", {"state": "complete"})
    write_json(tracker.path / "audit.json", {"passed": False})
    assert not tracker.poll()
    assert tracker.run.summary["independently_reloaded"] is False
    assert tracker.run.exit_code is None
    write_json(tracker.path / "audit.json", {"passed": True})
    assert tracker.poll()
    assert tracker.run.exit_code == 0


def test_container_mounts_only_one_output_writable_and_maps_physical_gpu_to_zero(
    tmp_path, monkeypatch
):
    root, source, results = tmp_path / "repo", tmp_path / "repo/source", tmp_path / "repo/results"
    out = results / "runs/run1"
    out.mkdir(parents=True)
    config = {
        "batch": "test-batch",
        "repository": str(root),
        "source_root": str(source),
        "results_root": str(results),
        "specs": [{"name": "run1"}],
        "runtime": {
            "docker_host": "unix:///tmp/test.sock",
            "image": "sha256:abc",
            "python": "/opt/python/bin/python",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    commands = []

    def popen(command):
        commands.append(command)
        return SimpleNamespace(poll=lambda: None, wait=lambda: 0)

    inspected = [
        {"State": {"Running": True, "Pid": 9876}, "Id": "container1", "Image": "sha256:abc"}
    ]
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(inspected)),
    )
    runner.container_work(config_path, 0, 3)
    command = commands[0]
    mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
    assert mounts == [f"type=bind,src={root},dst={root},readonly", f"type=bind,src={out},dst={out}"]
    assert "--read-only" in command and "--network=none" in command
    assert command[command.index("--gpus") + 1] == "device=3"
    assert command[-2:] == ["--gpu", "0"]
    runtime = json.loads((out / "container-runtime.json").read_text())
    assert runtime["host_pid"] == 9876 and runtime["physical_gpu"] == 3


def test_audit_is_new_python_process_in_same_container_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "results_root": str(tmp_path / "results"),
                "source_root": str(tmp_path / "source"),
                "specs": [{"name": "run1"}],
                "module": "bridge_reencoding",
            }
        )
    )
    operations, launched = [], []
    module = SimpleNamespace(run=lambda *args: operations.append(args))
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(
        runner.subprocess, "run", lambda command, **kwargs: launched.append((command, kwargs))
    )
    monkeypatch.setenv("MEMORY_INTERFACE_CONTAINER", "1")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/container/cuda/lib")
    runner.work(config_path, 0, 0)
    assert str(operations[0][2]) == "cuda:0"
    command, options = launched[0]
    assert "audit" in command and command[-2:] == ["--gpu", "0"]
    assert "docker" not in command
    assert options["check"] is True
    assert options["env"]["MEMORY_INTERFACE_CONTAINER"] == "1"
    assert options["env"]["LD_LIBRARY_PATH"] == "/container/cuda/lib"


def test_terminal_drain_registers_run_created_during_sdk_poll_once(tmp_path):
    runs = tmp_path / "runs"

    def create(name):
        path = runs / name
        write_json(path / "run.json", {"pid": 1})
        write_json(path / "learning.json", [{"step": 7}])

    create("early")
    write_json(tmp_path / "controller-state.json", {"state": "complete"})
    registered = []

    class Tracker:
        def __init__(self, sdk, root, path, settings):
            registered.append(path.name)
            self.path, self.metadata, self.cursor = path, {"pid": 1}, -1
            self.run = SimpleNamespace(url="mock://run")

        def poll(self):
            self.cursor = 7
            if self.path.name == "early":
                create("late")
            return True

        def close(self):
            pass

    result = drain_runs(None, tmp_path, runs, {"poll_seconds": 0}, tracker_factory=Tracker)
    assert registered == ["early", "late"]
    assert result["registered_runs"] == 2
    assert all(r["last_logged_step"] >= r["final_step"] for r in result["runs"].values())


def test_terminal_drain_reopens_only_a_finished_run_with_unlogged_final_node(tmp_path):
    runs = tmp_path / "runs"
    for name in ("growing", "stable"):
        write_json(runs / name / "run.json", {"pid": 1})
        write_json(runs / name / "learning.json", [{"step": 1}])
    write_json(tmp_path / "controller-state.json", {"state": "complete"})
    registered = []

    class Tracker:
        def __init__(self, sdk, root, path, settings):
            registered.append(path.name)
            self.path, self.metadata, self.cursor = path, {"pid": 1}, -1
            self.run = SimpleNamespace(url="mock://run")

        def poll(self):
            history = json.loads((self.path / "learning.json").read_text())
            self.cursor = history[-1]["step"]
            if self.path.name == "growing" and self.cursor == 1:
                write_json(self.path / "learning.json", [{"step": 1}, {"step": 2}])
            return True

        def close(self):
            pass

    result = drain_runs(None, tmp_path, runs, {"poll_seconds": 0}, tracker_factory=Tracker)
    assert registered.count("growing") == 2
    assert registered.count("stable") == 1
    assert result["runs"]["growing"] == {"last_logged_step": 2, "final_step": 2}
