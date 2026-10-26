"""Scheduling contracts for the authorized extension, without Docker calls."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "extension_launcher", ROOT / "scripts/launch_gate_extension.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def inactive_placeholder():
    return {
        "Id": launcher.INACTIVE_PLACEHOLDER,
        "Name": "/large-fora-k7-three-sigma-0-to500-20261013-gpu7-unused",
        "State": {
            "Status": "created",
            "StartedAt": "0001-01-01T00:00:00Z",
            "Running": False,
            "Pid": 0,
        },
        "HostConfig": {
            "RestartPolicy": {"Name": "no"},
            "DeviceRequests": [{"DeviceIDs": ["GPU-7"]}],
        },
    }


@pytest.mark.parametrize(
    ("section", "key", "changed"),
    [
        (None, "Id", "other-container"),
        (None, "Name", "/another-unused-container"),
        ("State", "Running", True),
        ("State", "StartedAt", "2026-10-17T20:00:00Z"),
        ("State", "Status", "running"),
        ("State", "Pid", 1234),
        ("RestartPolicy", "Name", "always"),
    ],
)
def test_inactive_placeholder_exception_requires_exact_never_started_identity(
    section, key, changed
):
    container = inactive_placeholder()
    assert launcher.inactive_placeholder(container)
    target = (
        container
        if section is None
        else (
            container["HostConfig"][section] if section == "RestartPolicy" else container[section]
        )
    )
    target[key] = changed
    assert not launcher.inactive_placeholder(container)


@pytest.mark.parametrize("context", ("default", "lm-memory"))
def test_placeholder_exception_is_default_context_only_and_never_erases_active_block(
    monkeypatch,
    context,
):
    container = inactive_placeholder()
    rows = {7: {"uuid": "GPU-7", "used": 0, "utilization": 0, "free": 97000}}
    active = []
    monkeypatch.setattr(launcher, "_original_snapshot", lambda: (rows, {7: active.copy()}))

    def docker(*args, context: str):
        if args[0] == "ps":
            return SimpleNamespace(stdout=container["Id"] if context == selected_context else "")
        assert args[0] == "inspect" and context == selected_context
        return SimpleNamespace(stdout=json.dumps([container]))

    selected_context = context
    monkeypatch.setattr(launcher.common, "docker", docker)
    actual_rows, blocked = launcher.snapshot()
    assert launcher.idle(7, "GPU-7", actual_rows, blocked) is (context == "default")
    active.append("active CUDA process 123")
    actual_rows, blocked = launcher.snapshot()
    assert "active CUDA process 123" in blocked[7]
    assert not launcher.idle(7, "GPU-7", actual_rows, blocked)


def test_idle_rejects_created_container_reservations_and_utilization():
    rows = {5: {"uuid": "abc", "used": 0, "utilization": 0, "free": 97000}}
    assert launcher.idle(5, "abc", rows, {})
    assert not launcher.idle(5, "abc", rows, {5: ["created-other-project"]})
    assert not launcher.idle(5, "different", rows, {})
    rows[5]["utilization"] = 100
    assert not launcher.idle(5, "abc", rows, {})


def test_snapshot_protects_both_context_reservations_without_blocking_exited_own_jobs(
    monkeypatch,
):
    rows = {
        gpu: {"uuid": f"GPU-{gpu}", "used": 0, "utilization": 0, "free": 97000}
        for gpu in (2, 3, 4, 5, 7)
    }

    def container(name, state, device):
        return {
            "Id": name,
            "Name": f"/{name}",
            "State": {"Status": state},
            "HostConfig": {"DeviceRequests": [{"DeviceIDs": [device]}]},
        }

    containers = {
        "default": [
            container("default-created-uuid", "created", "GPU-5"),
            container("default-created-numeric", "created", "7"),
            container("default-paused", "paused", "GPU-2"),
            container("gate-generalization-extension-v1-prior-train", "exited", "GPU-4"),
        ],
        "lm-memory": [
            container("lm-created-numeric", "created", "5"),
            container("lm-created-uuid", "created", "GPU-7"),
            container("lm-restarting", "restarting", "2"),
            container("gate-generalization-extension-v1-prior-grid", "exited", "4"),
        ],
    }
    calls = []

    def docker(*args, context):
        calls.append((context, args))
        if args[0] == "ps":
            assert "-a" in args and "-q" in args
            filters = {args[index + 1] for index, arg in enumerate(args) if arg == "--filter"}
            assert filters == {"status=created", "status=paused", "status=restarting"}
            # Docker combines multiple values of the same filter as OR. Exited
            # completed/failed containers remain recorded but reserve no GPU.
            identifiers = [
                row["Id"]
                for row in containers[context]
                if f"status={row['State']['Status']}" in filters
            ]
            return SimpleNamespace(stdout="\n".join(identifiers))
        assert args[0] == "inspect"
        selected = [row for row in containers[context] if row["Id"] in args[1:]]
        assert all(row["State"]["Status"] != "exited" for row in selected)
        return SimpleNamespace(stdout=json.dumps(selected))

    monkeypatch.setattr(launcher, "_original_snapshot", lambda: (rows, {3: ["active CUDA 123"]}))
    monkeypatch.setattr(launcher.common, "docker", docker)
    actual_rows, blocked = launcher.snapshot()
    assert actual_rows == rows
    assert set(blocked[5]) == {"/default-created-uuid", "/lm-created-numeric"}
    assert set(blocked[7]) == {"/default-created-numeric", "/lm-created-uuid"}
    assert set(blocked[2]) == {"/default-paused", "/lm-restarting"}
    assert blocked[3] == ["active CUDA 123"]
    for gpu in (2, 3, 5, 7):
        assert not launcher.idle(gpu, f"GPU-{gpu}", actual_rows, blocked)
    assert 4 not in blocked
    assert launcher.idle(4, "GPU-4", actual_rows, blocked)
    assert {context for context, args in calls if args[0] == "ps"} == {"default", "lm-memory"}


def test_frozen_matrix_has_no_duplicate_parent_continuation():
    plan = launcher.read(ROOT / "configs/gate-generalization-extension-v1.json")
    launcher.validate(plan)
    assert sum(j["steps"] - j["start_step"] for j in plan["jobs"]) == 12300000
    assert all(j["arm"] != "paper" for j in plan["jobs"] if j["family"] == "continuation")
    plan["jobs"][0]["arm"] = "paper"
    with pytest.raises(ValueError, match="reused"):
        launcher.validate(plan)


def test_job_command_has_correct_cpu_dataset_and_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.os, "sched_getaffinity", lambda _pid: set(range(80, 86)))
    plan = launcher.read(ROOT / "configs/gate-generalization-extension-v1.json")
    job = plan["jobs"][0]
    source = tmp_path / "source" / job["config"]
    source.parent.mkdir(parents=True)
    config = launcher.read(ROOT / job["config"])
    config["runtime"].pop("cpu_first", None)
    launcher.write(source, config)
    launcher.write(tmp_path / "freeze.json", {})
    owner = type(
        "Owner",
        (),
        {
            "base": tmp_path,
            "output": tmp_path / "out",
            "token_file": tmp_path / "private-token",
            "cpus": [80, 81, 82, 83, 84, 85],
        },
    )()
    controller = launcher.JobController(owner, job, 5, "uuid5", 1)
    _, argv, cpus = controller.command(job["arm"], "train", 5, "uuid5")
    assert cpus == [82, 83]
    assert argv[argv.index("--cpuset-cpus") + 1] == "82,83"
    assert f"type=bind,source={tmp_path / 'data'},target=/inputs,readonly" in argv
    assert job["runner"] in argv
    assert argv[-2:] == ["--output", "/output"]
    assert "--context" in argv and argv[argv.index("--context") + 1] == "lm-memory"
