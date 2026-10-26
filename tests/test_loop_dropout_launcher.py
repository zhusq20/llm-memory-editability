"""Non-preemption, quota and original-recipe contracts for the dropout queue."""

import copy
import importlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
launcher = importlib.import_module("launch_loop_dropout")


def config():
    return json.loads((ROOT / launcher.CONFIG).read_text())


def original():
    return json.loads((ROOT / "configs/loop-state-handoff-development-v1.json").read_text())


def snap():
    return {
        "rows": {
            g: {"uuid": f"GPU-{g}", "used": 0, "free": 97000, "utilization": 0} for g in range(10)
        },
        "processes": [],
        "own_processes": [],
        "containers": [],
    }


def container(gpu, owner="external", state="created"):
    return {
        "id": f"container-{gpu}",
        "name": f"/reserved-{gpu}",
        "context": "default",
        "state": {"Status": state},
        "requests": [{"DeviceIDs": [str(gpu)]}],
        "restart": {"Name": "no"},
        "owner": owner,
        "user": "9999:9999",
        "cpu_set": "",
    }


def runtime():
    result = copy.deepcopy(config()["runtime"])
    result["priority_batches"] = []
    return result


def test_only_dropout_and_execution_cap_may_change_science():
    launcher.validate(config(), original())
    for field, key, value in [
        ("training", "steps", 500000),
        ("training", "stream_seed", 4),
        ("model", "n_embd", 384),
        ("model", "attn_pdrop", 0.1),
        ("runtime", "maximum_local_user_gpus", 5),
    ]:
        changed = config()
        changed[field][key] = value
        with pytest.raises(ValueError):
            launcher.validate(changed, original())


def test_even_an_idle_gpu_is_blocked_when_four_user_cards_are_in_use():
    current, limits = snap(), runtime()
    current["containers"] = [container(g, owner=limits["owner"], state="running") for g in range(4)]
    assert launcher.owned_gpus(current, limits) == {0, 1, 2, 3}
    assert any("quota" in why for why in launcher.available(current, limits, 5))
    current["containers"].pop()
    assert launcher.available(current, limits, 5) == []


def test_created_reservations_memory_and_native_cuda_processes_all_block():
    limits = runtime()
    current = snap()
    current["containers"] = [container(5)]
    assert "/reserved-5" in launcher.available(current, limits, 5)
    current["containers"] = []
    current["processes"] = ["GPU-5"]
    assert "active CUDA process" in launcher.available(current, limits, 5)
    current["processes"] = []
    current["rows"][5]["used"] = 513
    assert "GPU not idle" in launcher.available(current, limits, 5)
    assert launcher.available(snap(), limits, 3)


def test_quota_counts_other_batches_numeric_uid_and_native_processes():
    current, limits = snap(), runtime()
    c = container(0)
    c["user"] = f"{os.getuid()}:1"
    current["containers"] = [c, container(1, limits["owner"]), container(2, limits["owner"])]
    current["own_processes"] = ["GPU-7"]
    assert launcher.owned_gpus(current, limits) == {0, 1, 2, 7}
    assert any("quota" in why for why in launcher.available(current, limits, 5))


def test_old_queue_reserves_cards_during_stage_gaps_until_all_jobs_complete(tmp_path):
    state, amendment = tmp_path / "state.json", tmp_path / "amendment.json"
    state.write_text(json.dumps({"state": "running", "jobs": {"old": {"state": "running"}}}))
    amendment.write_text(json.dumps({"effective_gpu_pool": [1, 6, 3, 2]}))
    limits = runtime()
    limits["priority_batches"] = [
        {
            "controller_state": str(state),
            "schedule_amendment": str(amendment),
            "fallback_gpu_ids": [5, 6, 8],
        }
    ]
    assert launcher.priority_reasons(limits, 1)
    assert launcher.priority_reasons(limits, 5) == []
    state.write_text(json.dumps({"state": "complete", "jobs": {"old": {"state": "running"}}}))
    assert launcher.priority_reasons(limits, 1)
    state.write_text(json.dumps({"state": "complete", "jobs": {"old": {"state": "complete"}}}))
    assert launcher.priority_reasons(limits, 1) == []
    state.unlink()
    assert launcher.priority_reasons(limits, 5)


def test_container_uses_fixed_image_readonly_inputs_and_no_source_bind(tmp_path):
    current = config()
    name, argv = launcher.stage_command(current, tmp_path, "train", "GPU-exact")
    assert name == "ld0-1022-state-mlp-v2-train"
    assert argv[:3] == ["docker", "--context", "lm-memory"]
    assert argv[argv.index("--gpus") + 1] == "device=GPU-exact"
    assert argv[argv.index("--cpuset-cpus") + 1] == "84,85"
    assert argv[argv.index("--memory") + 1] == "16g"
    assert argv[argv.index("--user") + 1] == "0:0"
    bootstrap = argv[argv.index("-c") + 1]
    assert f"os.setgid({os.getgid()})" in bootstrap
    assert f"os.setuid({os.getuid()})" in bootstrap
    assert bootstrap.index("os.setuid") < bootstrap.index("read_text()")
    assert current["runtime"]["image"] in argv
    assert "scripts/run_loop_dropout.py" in argv
    assert not any("target=/project" in word for word in argv)
    assert not any("WANDB_API_KEY=" in word for word in argv)
    _, cpu = launcher.stage_command(current, tmp_path, "contracts")
    assert "--gpus" not in cpu
    assert cpu[cpu.index("--network") + 1] == "none"
    assert "--gpus" not in launcher.stage_command(current, tmp_path, "cloud")[1]


def test_copy_archive_is_root_owned_readonly_without_host_permission_changes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "test.py"
    file.write_text("original code")
    file.chmod(0o644)
    with tarfile.open(fileobj=io.BytesIO(launcher.code_archive(tmp_path))) as archive:
        for item in archive.getmembers():
            assert item.uid == item.gid == 0
            assert item.mode == (0o555 if item.isdir() else 0o444)
        assert archive.extractfile("project/test.py").read() == b"original code"
    assert file.stat().st_mode & 0o777 == 0o644


def test_cpu_reservations_cover_created_and_running_containers():
    current = snap()
    c = container(7)
    c["cpu_set"] = "80-84,86"
    current["containers"] = [c]
    assert launcher.cpu_reasons(current, runtime())
    assert launcher.available(current, runtime(), 5)
    c["cpu_set"] = "80-83"
    assert launcher.cpu_reasons(current, runtime()) == []


def test_receipt_exemption_never_ignores_started_or_other_containers():
    c = container(5)
    c["state"].update(Pid=0, Running=False, StartedAt="0001-01-01T00:00:00Z")
    assert launcher.is_unstarted_receipt(c, c["id"])
    assert not launcher.is_unstarted_receipt(c, "another-container")
    for key, value in [("Pid", 1), ("Running", True), ("StartedAt", "2026-10-22T01:00:00Z")]:
        changed = copy.deepcopy(c)
        changed["state"][key] = value
        assert not launcher.is_unstarted_receipt(changed, c["id"])


def test_two_consecutive_idle_checks_required_for_admission(tmp_path, monkeypatch):
    controller = object.__new__(launcher.Controller)
    controller.base = tmp_path
    controller.config = {"runtime": {**runtime(), "gpu_candidates": [5]}}
    controller.state = {"stage": "environment"}
    snapshots = []

    def snapshot():
        snapshots.append(True)
        return snap()

    monkeypatch.setattr(launcher, "resource_snapshot", snapshot)
    monkeypatch.setattr(launcher.time, "sleep", lambda _: None)
    assert controller.admit() == (5, "GPU-5")
    assert len(snapshots) == 2
    assert (tmp_path / "resource-admission/environment.json").exists()
