"""Resource and execution isolation for the zero-training readout matrix."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/execute_parallel_readouts.py"
spec = importlib.util.spec_from_file_location("parallel_executor", SCRIPT)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


def config():
    return {
        "batch": "parallel-readouts-development-v1",
        "new_optimizer_updates": 0,
        "execution": {
            "lanes": {str(gpu): list(range(gpu * 8 + 8, gpu * 8 + 16)) for gpu in range(1, 8)},
            "tasks": [
                {
                    "name": f"task{i}",
                    "gpu": i % 7 + 1,
                    "kind": "restoration" if i < 4 else "phrasing",
                    "script": "scripts/readout.py",
                    "science_arguments": ["--config", "{config}", "--out", "{out}"],
                    "audit_arguments": ["--out", "{out}", "--audit-out", "{audit_out}"],
                    "smoke_arguments": ["--limit", "1"],
                }
                for i in range(12)
            ],
        },
    }


def test_rejects_optimizer_updates():
    cfg = config()
    cfg["new_optimizer_updates"] = 1
    with pytest.raises(ValueError, match="zero"):
        executor.validate_config(cfg)


def test_rejects_duplicate_tasks_and_cpu_overlaps():
    cfg = config()
    cfg["execution"]["tasks"][1]["name"] = cfg["execution"]["tasks"][0]["name"]
    with pytest.raises(ValueError, match="unique"):
        executor.validate_config(cfg)
    cfg = config()
    cfg["execution"]["lanes"]["2"] = cfg["execution"]["lanes"]["1"]
    with pytest.raises(ValueError, match="exclusive"):
        executor.validate_config(cfg)


def test_never_allocates_perk_gpu():
    cfg = config()
    cfg["execution"]["lanes"]["0"] = cfg["execution"]["lanes"].pop("1")
    with pytest.raises(ValueError, match="idle"):
        executor.validate_config(cfg)


def test_commands_mount_sources_readonly_and_only_new_outputs_writable(tmp_path):
    cfg = config()
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "config.json").write_text(json.dumps(cfg))
    command, _, out = executor.task_command(tmp_path, cfg["execution"]["tasks"][0], "science")
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert f"type=bind,source={executor.ARCHIVE},target={executor.ARCHIVE},readonly" in mounts
    assert f"type=bind,source={executor.RECOVERY},target={executor.RECOVERY},readonly" in mounts
    assert f"type=bind,source={tmp_path},target={tmp_path}" in mounts
    assert "device=1" in command
    assert str(frozen / "scripts/track_parallel_readout.py") in command
    assert out == tmp_path / "runs/task0"


def test_smoke_has_separate_directory_and_no_online_science_wrapper(tmp_path):
    cfg = config()
    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen/config.json").write_text(json.dumps(cfg))
    command, name, out = executor.task_command(
        tmp_path, cfg["execution"]["tasks"][0], "science", True
    )
    assert name.endswith("-smoke")
    assert out == tmp_path / "preflight/task0"
    assert "--limit" in command
    assert not any("track_parallel_readout.py" in x for x in command)


def test_rejects_task_outside_frozen_matrix(tmp_path):
    cfg = config()
    (tmp_path / "frozen").mkdir()
    (tmp_path / "frozen/config.json").write_text(json.dumps(cfg))
    task = copy.deepcopy(cfg["execution"]["tasks"][0])
    task["gpu"] = 0
    with pytest.raises(ValueError, match="outside"):
        executor.task_command(tmp_path, task, "science")
