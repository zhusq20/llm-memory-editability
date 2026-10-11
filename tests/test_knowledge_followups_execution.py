"""Resource, provenance and independent-process gates for six development lanes.

Docker and process launch are mocked: these tests never start containers or CUDA.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "knowledge_execution", PROJECT / "scripts/execute_knowledge_followups.py"
)
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


@pytest.fixture
def frozen(tmp_path):
    base = tmp_path / "batch"
    config = execution.read(PROJECT / "configs/edit-scope-development-v1.json")
    execution.write(base / "frozen/config.json", config)
    code = base / "frozen/scripts/execute_knowledge_followups.py"
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("# frozen source fixture\n")
    data = tmp_path / "world.npz"
    data.write_bytes(b"frozen graph fixture")
    execution.write(
        base / "frozen/freeze.json",
        {
            "source_files": {
                str(code.relative_to(base / "frozen")): execution.sha(code),
                "config.json": execution.sha(base / "frozen/config.json"),
            },
            "inputs": {str(data): execution.sha(data)},
            "image": execution.IMAGE,
        },
    )
    return base


@pytest.mark.parametrize("kind", ("source", "config", "input"))
def test_source_config_and_input_hashes_are_rechecked(frozen, kind):
    execution.verify(frozen)
    meta = execution.read(frozen / "frozen/freeze.json")
    path = {
        "source": frozen / "frozen/scripts/execute_knowledge_followups.py",
        "config": frozen / "frozen/config.json",
        "input": Path(next(iter(meta["inputs"]))),
    }[kind]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        execution.verify(frozen)


@pytest.fixture
def mocked_resources(monkeypatch):
    active = [
        {
            "Name": "existing-perk",
            "HostConfig": {"CpusetCpus": "0-7", "DeviceRequests": [{"DeviceIDs": ["0"]}]},
        },
        {
            "Name": "existing-fticr",
            "HostConfig": {"CpusetCpus": "8-15", "DeviceRequests": [{"DeviceIDs": ["1"]}]},
        },
    ]
    memory = [5000, 6000, 0, 0, 0, 0, 0, 0]
    calls = []

    def docker(arguments, **kwargs):
        calls.append(arguments)
        if arguments[0] == "ps":
            return SimpleNamespace(stdout="existing0 existing1")
        if arguments[0] == "inspect":
            return SimpleNamespace(stdout=json.dumps(active))
        assert arguments[:2] == ["image", "inspect"]
        return SimpleNamespace(stdout=json.dumps([{"Id": execution.IMAGE}]))

    monkeypatch.setattr(execution, "docker", docker)
    monkeypatch.setattr(
        execution.subprocess, "check_output", lambda *a, **kw: "\n".join(map(str, memory))
    )
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: set(range(80)))
    read_text = Path.read_text

    def file_text(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemAvailable: {512 * 1024 * 1024} kB\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", file_text)
    return active, memory, calls


def test_exact_six_idle_gpus_and_disjoint_cpu_allocations_leave_existing_lanes(mocked_resources):
    allocation = execution.resources(list(range(2, 8)))
    assert allocation["gpus"] == list(range(2, 8))
    cpus = allocation["cpus"]
    assert len(cpus) == 6 and all(len(part) == 8 for part in cpus)
    assert len(set().union(*(set(part) for part in cpus))) == 48
    assert set().union(*(set(part) for part in cpus)).isdisjoint(range(16))
    assert all(call[0] in ("ps", "inspect", "image") for call in mocked_resources[2])


@pytest.mark.parametrize("gpus", ([0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]))
def test_resource_matrix_cannot_use_other_gpu_sets(mocked_resources, gpus):
    with pytest.raises(ValueError, match="only GPUs"):
        execution.resources(gpus)


@pytest.mark.parametrize(
    "device_request", ({"DeviceIDs": ["4"]}, {"Count": -1}, {"Count": 1, "DeviceIDs": []})
)
def test_active_or_unknown_gpu_assignment_prevents_launch(mocked_resources, device_request):
    active, _, _ = mocked_resources
    active.append(
        {
            "Name": "unrelated-live-work",
            "HostConfig": {"CpusetCpus": "", "DeviceRequests": [device_request]},
        }
    )
    with pytest.raises(RuntimeError, match="GPU collision"):
        execution.resources(list(range(2, 8)))


def test_existing_gpu_memory_without_a_known_container_also_prevents_launch(mocked_resources):
    mocked_resources[1][7] = 501
    with pytest.raises(RuntimeError, match="existing allocation"):
        execution.resources(list(range(2, 8)))


def test_six_commands_bind_frozen_inputs_readonly_and_separate_outputs_and_tmp(frozen):
    names, outputs = set(), set()
    for index in range(6):
        cpus = list(range(16 + index * 8, 24 + index * 8))
        command, name, output = execution.command(frozen, index, "science", cpus, index + 2)
        names.add(name)
        outputs.add(output)
        assert command[command.index("--gpus") + 1] == f"device={index + 2}"
        assert command[command.index("--cpuset-cpus") + 1] == ",".join(map(str, cpus))
        assert command[command.index("--memory") + 1] == "48g"
        assert command[command.index("--tmpfs") + 1].startswith("/tmp:rw,")
        mounts = [command[i + 1] for i, word in enumerate(command) if word == "--mount"]
        assert (
            f"type=bind,source={execution.RECOVERY},target={execution.RECOVERY},readonly" in mounts
        )
        assert f"type=bind,source={frozen / 'frozen'},target={frozen / 'frozen'},readonly" in mounts
        assert f"type=bind,source={frozen},target={frozen}" in mounts
        assert any("target=/root/.netrc,readonly" in mount for mount in mounts)
        assert not any("WANDB_API_KEY" in word for word in command)
        audit, audit_name, audit_output = execution.command(frozen, index, "audit", cpus, index + 2)
        assert audit_name != name and audit_output == output
        assert audit[audit.index("--audit-dir") + 1] == str(output)
        assert "{out}" not in audit
        smoke, smoke_name, smoke_output = execution.command(
            frozen, index, "science", cpus, 2, smoke=True
        )
        assert smoke_name != name and smoke_output != output and "--smoke" in smoke
    assert len(names) == len(outputs) == 6


@pytest.mark.parametrize("gpu", (0, 1))
def test_direct_command_cannot_target_reserved_gpu(frozen, gpu):
    with pytest.raises(ValueError):
        execution.command(frozen, 0, "science", list(range(16, 24)), gpu)


def preflight(base):
    receipts = []
    for stage in ("science", "audit"):
        path = base / "preflight/fixture" / f"{stage}-process.json"
        execution.write(
            path,
            {
                "stage": stage,
                "state": "complete",
                "exit_code": 0,
                "physical_gpu": 2,
                "container_id": f"fresh-{stage}",
            },
        )
        receipts.append({"path": str(path), "sha256": execution.sha(path)})
    receipt = {
        "passed": True,
        "freeze_sha256": execution.sha(base / "frozen/freeze.json"),
        "receipts": receipts,
    }
    execution.write(base / "preflight-passed.json", receipt)
    return receipt


@pytest.mark.parametrize(
    "failure",
    ("missing", "boolean_only", "wrong_freeze", "changed_receipt", "failed_stage", "missing_audit"),
)
def test_launch_needs_actual_completed_cuda_and_replay_receipts(frozen, monkeypatch, failure):
    receipt = preflight(frozen)
    path = frozen / "preflight-passed.json"
    if failure == "missing":
        path.unlink()
    elif failure == "boolean_only":
        execution.write(path, {"passed": True})
    elif failure == "wrong_freeze":
        receipt["freeze_sha256"] = "wrong"
        execution.write(path, receipt)
    elif failure == "changed_receipt":
        Path(receipt["receipts"][0]["path"]).write_text("{}")
    elif failure == "failed_stage":
        evidence = Path(receipt["receipts"][0]["path"])
        value = execution.read(evidence)
        value["exit_code"] = 1
        execution.write(evidence, value)
        receipt["receipts"][0]["sha256"] = execution.sha(evidence)
        execution.write(path, receipt)
    else:
        receipt["receipts"] = receipt["receipts"][:1]
        execution.write(path, receipt)
    monkeypatch.setattr(
        execution.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("Invalid preflight launched a controller"),
    )
    with pytest.raises(ValueError):
        execution.launch(frozen)


def test_duplicate_launch_is_rejected_even_after_valid_preflight(frozen, monkeypatch):
    preflight(frozen)
    execution.write(frozen / "controller-process.json", {"pid": 123})
    monkeypatch.setattr(
        execution.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate controller launched")
    )
    with pytest.raises(RuntimeError, match="already been submitted"):
        execution.launch(frozen)


def test_science_and_replay_use_distinct_container_processes(frozen, monkeypatch):
    launched = []

    def docker(arguments, **kwargs):
        if arguments[0] == "run":
            launched.append(arguments)
            return SimpleNamespace(returncode=0, stdout=f"container-{len(launched)}", stderr="")
        if arguments[0] == "inspect":
            return SimpleNamespace(stdout=json.dumps({"Running": False, "ExitCode": 0}))
        assert arguments[0] == "logs"
        return SimpleNamespace(stdout="retained stage log", stderr="")

    monkeypatch.setattr(execution, "docker", docker)
    result = execution.lane(frozen, 0, list(range(16, 24)), 2, smoke=True)
    assert result["passed"]
    assert len(launched) == 2
    reports = result["reports"]
    assert {report["stage"] for report in reports} == {"science", "audit"}
    assert reports[0]["container_id"] != reports[1]["container_id"]
    assert reports[0]["container"] != reports[1]["container"]
    assert all(report["physical_gpu"] == 2 and report["exit_code"] == 0 for report in reports)


def test_failed_science_is_retained_without_audit_or_automatic_retry(frozen, monkeypatch):
    launches = []

    def docker(arguments, **kwargs):
        if arguments[0] == "run":
            launches.append(arguments)
            return SimpleNamespace(returncode=0, stdout="failed-science", stderr="")
        if arguments[0] == "inspect":
            return SimpleNamespace(stdout=json.dumps({"Running": False, "ExitCode": 7}))
        return SimpleNamespace(stdout="original failure", stderr="")

    monkeypatch.setattr(execution, "docker", docker)
    result = execution.lane(frozen, 0, list(range(16, 24)), 2)
    assert not result["passed"] and len(launches) == 1
    assert result["reports"][0]["exit_code"] == 7
    assert (frozen / "runs/first-joint/science.log").read_text() == "original failure"
