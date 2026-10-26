"""Scheduling contracts: shared preparation, GPU ownership and immutable inputs."""

import copy
import importlib.util
import json
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "knowledge_launcher", ROOT / "scripts/launch_knowledge_method_diagnosis.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture
def plan(tmp_path):
    value = json.loads((ROOT / "configs/knowledge-method-diagnosis-launch-v1.json").read_text())
    value["output"] = str(tmp_path)
    return value


def test_gpu_reservations_fail_closed():
    rows = {3: {"uuid": "GPU-three"}, 4: {"uuid": "GPU-four"}}
    assert M.reserved_devices([{"Capabilities": [["gpu"]], "Count": -1}], rows) == {3, 4}
    assert M.reserved_devices([{"Capabilities": [["gpu"]], "DeviceIDs": ["9"]}], rows) == set()
    assert M.reserved_devices([{"Capabilities": [["gpu"]], "DeviceIDs": ["GPU-four"]}], rows) == {4}
    assert M.reserved_devices([{"Capabilities": [["gpu"]], "DeviceIDs": ["unknown"]}], rows) == {
        3,
        4,
    }


def test_busy_or_wrong_device_is_never_admitted(plan):
    row = {"uuid": "expected", "used": 0, "free": 97000, "utilization": 0}
    assert M.idle(row, "expected", [], plan["runtime"])
    assert not M.idle(row, "wrong", [], plan["runtime"])
    assert not M.idle(row, "expected", ["created-container"], plan["runtime"])
    assert not M.idle({**row, "used": 513}, "expected", [], plan["runtime"])


def test_failed_prepare_blocks_all_science(plan, monkeypatch):
    controller = M.Controller(plan)
    calls = []

    def worker(arm, stages, lane):
        calls.append((arm, stages))
        return False

    monkeypatch.setattr(controller, "worker", worker)
    assert controller.run() == 1
    assert calls == [("shared", ["prepare"])]
    assert all(controller.state["jobs"][arm]["state"] == "blocked_by_prepare" for arm in M.ARMS)


def test_success_without_prepare_marker_still_blocks_science(plan, monkeypatch):
    controller = M.Controller(plan)
    calls = []
    monkeypatch.setattr(controller, "worker", lambda arm, stages, lane: calls.append(arm) or True)
    assert controller.run() == 1
    assert calls == ["shared"]


def test_all_arms_wait_for_shared_marker_then_audit_separately(plan, monkeypatch):
    controller = M.Controller(plan)
    calls = []

    def worker(arm, stages, lane):
        calls.append((arm, stages))
        if arm == "shared":
            M.write(Path(plan["output"]) / "shared/prepare-complete.json", {"passed": True})
        else:
            assert M.prepared(Path(plan["output"]))
        return True

    monkeypatch.setattr(controller, "worker", worker)
    summaries = []
    monkeypatch.setattr(M, "call", lambda argv, **kwargs: summaries.append(argv))
    assert controller.run() == 0
    assert len(summaries) == 1
    assert summaries[0][1].endswith("/source/scripts/summarize_knowledge_method_diagnosis.py")
    assert calls[0] == ("shared", ["prepare"])
    assert {arm for arm, _ in calls[1:]} == set(M.ARMS)
    assert all(stages == ["train", "evaluate", "audit"] for _, stages in calls[1:])


def test_existing_attempt_cannot_be_replaced(plan):
    controller = M.Controller(plan)
    (Path(plan["output"]) / "execution/pd_lora/train").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        controller.run_stage("pd_lora", "train", plan["lanes"]["pd_lora"])


def test_mounts_freeze_source_and_dependencies_and_do_not_expose_token(plan):
    _, argv = M.container_command(plan, "audit", "pd_lora", plan["lanes"]["pd_lora"])
    mounts = [argv[i + 1] for i, word in enumerate(argv) if word == "--mount"]
    assert any("/source,target=/project,readonly" in mount for mount in mounts)
    assert any("target=/hf,readonly" in mount for mount in mounts)
    assert any("target=/batch/deps,readonly" in mount for mount in mounts)
    assert plan["wandb_token_file"] not in " ".join(argv)
    assert argv[argv.index("--cpus") + 1] == "10"
    assert argv[argv.index("--memory") + 1] == "128g"
    assert M.runner_command(plan, "audit", "pd_lora")[-2:] == ["--arm", "pd_lora"]


def test_review_and_coherent_lanes_required(plan):
    M.validate(plan)
    altered = copy.deepcopy(plan)
    altered["cli_reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        M.validate(altered, execute=True)
    altered = copy.deepcopy(plan)
    altered["lanes"]["sft_lora"]["gpu"] = altered["lanes"]["pd_lora"]["gpu"]
    with pytest.raises(ValueError, match="identical"):
        M.validate(altered)


def test_two_arms_cannot_hold_same_gpu_lease(plan, monkeypatch, tmp_path):
    lane = plan["lanes"]["sft_lora"]
    assert lane == plan["lanes"]["sft_down"]
    row = {"uuid": lane["gpu_uuid"], "used": 0, "free": 97000, "utilization": 0}
    monkeypatch.setattr(M, "snapshot", lambda: ({lane["gpu"]: row}, {}))
    runtime = {**plan["runtime"], "poll_seconds": 0.001}
    directory = tmp_path / "locks"
    lease = M.acquire_gpu(lane, runtime, lambda **kwargs: None, time.monotonic() + 1, directory)
    try:
        with pytest.raises(TimeoutError):
            M.acquire_gpu(lane, runtime, lambda **kwargs: None, time.monotonic() + 0.03, directory)
    finally:
        lease.close()
    released = M.acquire_gpu(lane, runtime, lambda **kwargs: None, time.monotonic() + 1, directory)
    released.close()


def test_snapshot_is_independent_and_cannot_be_overwritten(plan, monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "src/llm_memory_editability").mkdir(parents=True)
    (checkout / "scripts").mkdir()
    (checkout / "configs").mkdir()
    source = checkout / "src/llm_memory_editability/model.py"
    source.write_text("original")
    (checkout / plan["scientific_config"]).write_text("{}")
    launch = checkout / "configs/launch.json"
    launch.write_text("{}")
    (checkout / "pyproject.toml").write_text("")
    monkeypatch.setattr(M, "ROOT", checkout)
    manifest = M.freeze_source(plan, launch)
    source.write_text("changed")
    assert (
        Path(plan["output"]) / "source/src/llm_memory_editability/model.py"
    ).read_text() == "original"
    assert "src/llm_memory_editability/model.py" in manifest
    with pytest.raises(FileExistsError):
        M.freeze_source(plan, launch)
