import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_memory_editability.experiment_tracking import (
    ArtifactRunTracker,
    learning_metrics,
    write_json,
)


class FakeRun:
    def __init__(self, sdk, args):
        self.sdk, self.args = sdk, args
        self.step = sdk.cloud.get(args["id"], -1) + 1
        self.url = f"https://wandb.invalid/{args['id']}"
        self.summary = {}
        self.definitions = []
        self.exit_code = None

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, values, step):
        assert step >= self.step
        self.sdk.logs.append((self.args["id"], step, values))
        self.sdk.cloud[self.args["id"]] = step
        self.step = step + 1

    def finish(self, exit_code=0):
        self.exit_code = exit_code


class FakeSDK:
    def __init__(self):
        self.cloud, self.logs, self.runs = {}, [], []

    def Settings(self, **kwargs):
        return kwargs

    def init(self, **kwargs):
        run = FakeRun(self, kwargs)
        self.runs.append(run)
        return run

    def Api(self, timeout):
        return self

    def run(self, path):
        return SimpleNamespace(lastHistoryStep=self.cloud.get(path.rsplit("/", 1)[-1], -1))


@pytest.fixture
def settings():
    return {
        "entity": "zhusq20",
        "project": "llm-memory-editability",
        "mode": "online",
        "system_metrics": True,
    }


def fixture_run(batch, name="standard8", world="world-A"):
    path = batch / "development" / name
    write_json(
        path / "run.json",
        {
            "spec": {"name": name, "phi": 7.2, "api_key": "not-a-real-key", "tokens": 4096},
            "world_sha256": world,
            "initial_model_sha256": "initial-A",
            "gpu": 6,
            "pid": 123,
        },
    )
    write_json(
        path / "learning.json",
        [{"step": s, "metrics": {"test_ii": {"accuracy": 0.1, "n": 3000}}} for s in [0, 256, 512]],
    )
    write_json(path / "status.json", {"state": "running"})
    return path


def test_actual_exposure_and_sparse_optimizer_steps_drive_throughput():
    previous = {"step": 256, "examples": 130000, "training_seconds": 4}
    record = {
        "step": 512,
        "examples": 259000,
        "training_seconds": 8,
        "metrics": {"test_ii": {"accuracy": 0.2, "n": 3000, "nll": 2.5}},
    }
    values = learning_metrics(record, previous)
    assert values["training/step"] == 512
    assert values["perf/examples_per_second"] == 32250
    assert values["perf/updates_per_second"] == 64
    assert values["perf/training_ms_per_update"] == 15.625
    assert values["eval/test_ii/n"] == 3000
    assert values["eval/test_ii/nll"] == 2.5


def test_recording_preserves_trainer_files_and_omits_credentials(tmp_path, settings):
    path = fixture_run(tmp_path)
    original = {p: p.read_bytes() for p in path.iterdir()}
    sdk = FakeSDK()
    tracker = ArtifactRunTracker(sdk, tmp_path, path, settings)
    assert not tracker.poll()
    assert {p: p.read_bytes() for p in path.iterdir()} == original
    assert "not-a-real-key" not in json.dumps(tracker.run.args)
    assert tracker.run.args["config"]["spec"]["tokens"] == 4096
    assert tracker.run.args["settings"]["x_stats_pid"] == 123
    assert tracker.run.args["settings"]["x_stats_gpu_device_ids"] == (6,)
    assert [row[1] for row in sdk.logs] == [0, 256, 512]


def test_explicit_scientific_group_is_preserved_for_confirmation_batch(tmp_path, settings):
    path = fixture_run(tmp_path)
    metadata = json.loads((path / "run.json").read_text())
    metadata["tracking_group"] = "realworld-composition-v1"
    write_json(path / "run.json", metadata)
    tracker = ArtifactRunTracker(FakeSDK(), tmp_path, path, settings)
    assert tracker.run.args["group"] == "realworld-composition-v1"


def test_online_resume_skips_cloud_points_during_checkpoint_replay(tmp_path, settings):
    path = fixture_run(tmp_path)
    sdk = FakeSDK()
    first = ArtifactRunTracker(sdk, tmp_path, path, settings)
    first.poll()
    first.close()
    metadata = json.loads((path / "run.json").read_text())
    metadata["pid"] = 456
    write_json(path / "run.json", metadata)
    history = json.loads((path / "learning.json").read_text())
    write_json(path / "learning.json", history[:2])
    resumed = ArtifactRunTracker(sdk, tmp_path, path, settings)
    resumed.poll()
    write_json(path / "learning.json", history + [{"step": 1024, "metrics": {}}])
    resumed.poll()
    assert first.run.args["id"] == resumed.run.args["id"]
    assert [row[1] for row in sdk.logs] == [0, 256, 512, 1024]
    assert resumed.run.args["settings"]["x_stats_pid"] == 456


def test_missing_local_cursor_replays_unsynced_points_from_cloud_step(tmp_path, settings):
    path = fixture_run(tmp_path)
    sdk = FakeSDK()
    first = ArtifactRunTracker(sdk, tmp_path, path, settings)
    first.poll()
    sdk.cloud[first.run.args["id"]] = 256
    resumed = ArtifactRunTracker(sdk, tmp_path, path, settings)
    resumed.poll()
    assert sdk.logs[-1][1] == 512


def test_changed_scientific_identity_cannot_reuse_run(tmp_path, settings):
    path = fixture_run(tmp_path)
    ArtifactRunTracker(FakeSDK(), tmp_path, path, settings)
    metadata = json.loads((path / "run.json").read_text())
    metadata["world_sha256"] = "world-B"
    write_json(path / "run.json", metadata)
    with pytest.raises(AssertionError, match="trajectory identity changed"):
        ArtifactRunTracker(FakeSDK(), tmp_path, path, settings)


def test_parallel_conditions_have_distinct_runs_in_same_batch(tmp_path, settings):
    a = fixture_run(tmp_path, "standard8")
    b = fixture_run(tmp_path, "loop4x2")
    sdk = FakeSDK()
    left = ArtifactRunTracker(sdk, tmp_path, a, settings)
    right = ArtifactRunTracker(sdk, tmp_path, b, settings)
    assert left.run.args["id"] != right.run.args["id"]
    assert left.run.args["group"] == right.run.args["group"] == tmp_path.name
    assert left.run.args["reinit"] == right.run.args["reinit"] == "create_new"


def test_offline_resume_uses_explicit_segments(tmp_path, settings):
    path = fixture_run(tmp_path)
    sdk = FakeSDK()
    settings = {**settings, "mode": "offline"}
    first = ArtifactRunTracker(sdk, tmp_path, path, settings)
    first.poll()
    second = ArtifactRunTracker(sdk, tmp_path, path, settings)
    assert first.run.args["id"] != second.run.args["id"]
    assert (
        first.run.args["config"]["scientific_run_id"]
        == second.run.args["config"]["scientific_run_id"]
    )
    assert second.run.args["resume"] is None


def test_completed_retry_keeps_failure_record_and_successful_exit(tmp_path, settings):
    path = fixture_run(tmp_path)
    write_json(path / "failure.json", {"exit_code": 1})
    write_json(path / "complete.json", {"state": "complete"})
    tracker = ArtifactRunTracker(FakeSDK(), tmp_path, path, settings)
    assert tracker.poll()
    assert tracker.run.summary["has_failure_record"] is True
    assert tracker.run.exit_code == 0


def test_future_defaults_record_user_preference():
    defaults = json.loads(Path("configs/experiment-tracking-defaults.json").read_text())
    assert defaults["enabled"] and defaults["system_metrics"]
    assert defaults["entity"] == "zhusq20"
    assert defaults["project"] == "llm-memory-editability"
