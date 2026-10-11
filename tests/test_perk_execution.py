"""Scientific freeze, coverage, resource isolation, and no-repeat contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/execute_perk_reproduction.py"
spec = importlib.util.spec_from_file_location("perk_execution", SCRIPT)
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


def configs():
    common = {
        "reproduction_kind": "method_reconstruction",
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "wandb_run_id": "public-run-id",
    }
    return {
        "perk": {**common, "lora_rank": 256, "n_inner_iter": 4, "num_train_epochs": 2},
        "fticr": {**common, "max_seq_length": 9216, "max_position_embeddings": 10240},
    }


@pytest.fixture
def frozen(tmp_path):
    values = configs()
    for arm in execution.ARMS:
        execution.write(execution.config_path(tmp_path, arm), values[arm])
    source = execution.source(tmp_path) / "src/llm_memory_editability/core.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen scientific source\n")
    data = tmp_path / "input.json"
    data.write_text("[]\n")
    value = {
        "image_sha256": "sha256:" + "9" * 64,
        "source_files": {"src/llm_memory_editability/core.py": execution.sha(source)},
        "configs": {
            arm: execution.sha(execution.config_path(tmp_path, arm)) for arm in execution.ARMS
        },
        "inputs": {str(data): execution.sha(data)},
    }
    execution.write(tmp_path / "frozen/freeze.json", value)
    return tmp_path


def test_full_two_arm_scientific_contract_rejects_development_limits():
    execution.validate_science_configs(configs())
    for bad_field in ("smoke_max_train_steps", "limit_evaluation", "wandb_api_key"):
        values = configs()
        values["perk"][bad_field] = 2
        with pytest.raises(ValueError):
            execution.validate_science_configs(values)
    with pytest.raises(ValueError, match="exactly"):
        execution.validate_science_configs({"perk": configs()["perk"]})


@pytest.mark.parametrize("kind", ("source", "config", "data"))
def test_every_frozen_dependency_is_checked(frozen, kind):
    execution.verify_freeze(frozen)
    path = {
        "source": execution.source(frozen) / "src/llm_memory_editability/core.py",
        "config": execution.config_path(frozen, "perk"),
        "data": frozen / "input.json",
    }[kind]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        execution.verify_freeze(frozen)


def populate_predictions(folder, arm, count=2):
    prediction, summary = execution.prediction_files(folder, arm)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text(
        "".join(
            json.dumps(
                {
                    "guid": f"id-{i}",
                    "prompt": "prompt",
                    "answer": "gold",
                    "output": "generated",
                    "generated_token_ids": [7, 8, 9],
                }
            )
            + "\n"
            for i in range(count)
        )
    )
    if arm == "perk":
        value = {
            "checkpoint_sha256": "checkpoint",
            "parameter_sha256": "parameters",
            "metrics": {"accuracy": 0.0, "f1": 0.0},
            "mean_outer_loss": 2.0,
            "persistent_parameters_unchanged": True,
            "evaluation_exposure": {"problems": count},
        }
    else:
        value = {
            "accuracy": 0.0,
            "f1": 0.0,
            "evaluated_rows": count,
            "expected_rows": 999,
            "elapsed_seconds": 12.0,
        }
    execution.write(summary, value)
    return prediction, summary


@pytest.mark.parametrize("arm", execution.ARMS)
def test_replay_requires_exact_generated_tokens_and_complete_coverage(tmp_path, arm):
    populate_predictions(tmp_path / "a", arm)
    path, _ = populate_predictions(tmp_path / "b", arm)
    assert execution.compare_replay(tmp_path / "a", tmp_path / "b", arm, 2)["passed"]
    with pytest.raises(ValueError, match="coverage"):
        execution.compare_replay(tmp_path / "a", tmp_path / "b", arm, 999)
    rows = path.read_text().replace("[7, 8, 9]", "[7, 8, 10]")
    path.write_text(rows)
    with pytest.raises(ValueError, match="replay differs"):
        execution.compare_replay(tmp_path / "a", tmp_path / "b", arm, 2)


def test_fticr_volatile_wall_clock_does_not_change_scientific_replay(tmp_path):
    populate_predictions(tmp_path / "a", "fticr")
    _, summary = populate_predictions(tmp_path / "b", "fticr")
    value = execution.read(summary)
    value["elapsed_seconds"] = 67.0
    execution.write(summary, value)
    assert execution.compare_replay(tmp_path / "a", tmp_path / "b", "fticr", 2)["passed"]


def test_container_isolation_and_credentials_never_enter_arguments(frozen):
    for arm, gpu in (("perk", 0), ("fticr", 1)):
        _, command = execution.container_command(frozen, arm, "train", list(range(64, 96)))
        assert command[command.index("--gpus") + 1] == f"device={gpu}"
        assert command[command.index("--memory") + 1] == "96g"
        assert command[command.index("--shm-size") + 1] == "16g"
        assert command[command.index("--network") + 1] == "host"
        assert command[command.index("--cpuset-cpus") + 1] == ",".join(map(str, range(64, 96)))
        assert any("target=/root/.netrc,readonly" in word for word in command)
        assert not any("WANDB_API_KEY" in word for word in command)
        assert "--context" in command and "lm-memory" in command
    with pytest.raises(ValueError, match="outside"):
        execution.container_command(frozen, "perk", "train", list(range(32)))


@pytest.mark.parametrize("z1_running", (True, False))
def test_cpu_allocation_reserves_z1_cores_before_and_after_its_launch(
    frozen, monkeypatch, z1_running
):
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda _: set(range(128)))
    monkeypatch.setattr(execution, "available_memory_kib", lambda: 512 * 1024 * 1024)

    def check_output(command, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return ""
        return "0, gpu-zero, 0\n1, gpu-one, 0\n"

    def docker(arguments, **kwargs):
        if arguments[0] == "ps":
            return SimpleNamespace(stdout="z1-id\n" if z1_running else "")
        if arguments[0] == "inspect":
            return SimpleNamespace(
                stdout=json.dumps(
                    [
                        {
                            "Name": "z1",
                            "HostConfig": {
                                "CpusetCpus": "16-63",
                                "DeviceRequests": [{"DeviceIDs": ["2"]}],
                            },
                        }
                    ]
                )
            )
        return SimpleNamespace(
            stdout=json.dumps(
                [
                    execution.read(frozen / "frozen/freeze.json")
                    | {
                        "Id": "sha256:" + "9" * 64,
                    }
                ]
            )
        )

    monkeypatch.setattr(execution.subprocess, "check_output", check_output)
    monkeypatch.setattr(execution, "docker", docker)
    allocation = execution.resource_check(frozen)
    assert allocation["cpus"]["perk"] == list(range(16)) + list(range(64, 80))
    assert allocation["cpus"]["fticr"] == list(range(80, 112))
    assert not (set(allocation["cpus"]["perk"]) | set(allocation["cpus"]["fticr"])) & set(
        range(16, 64)
    )


def test_launch_refuses_existing_controller_even_if_prerequisites_pass(frozen, monkeypatch):
    execution.write(
        frozen / "development/cuda-validation.json",
        {
            "passed": True,
            "arms": {arm: {"passed": True} for arm in execution.ARMS},
            "freeze_sha256": execution.sha(frozen / "frozen/freeze.json"),
            "artifacts": {str(frozen / "input.json"): execution.sha(frozen / "input.json")},
        },
    )
    execution.write(frozen / "controller-process.json", {"pid": 1})
    monkeypatch.setattr(execution, "compare_replay", lambda *args: {"passed": True})
    monkeypatch.setattr(execution, "resource_check", lambda base: {})
    with pytest.raises(RuntimeError, match="already submitted"):
        execution.launch(frozen)


def test_completed_training_is_preserved_when_a_later_audit_stage_fails(frozen, monkeypatch):
    execution.write(
        frozen / "launch-resources.json",
        {"cpus": {arm: list(range(64, 96)) for arm in execution.ARMS}},
    )
    started = []

    def exit_code(arm, stage):
        return int(arm == "perk" and stage == "evaluation")

    def start(base, arm, stage, cpus, environment):
        started.append((arm, stage))
        return {
            "process": SimpleNamespace(poll=lambda: exit_code(arm, stage)),
            "arm": arm,
            "stage": stage,
            "container": f"{arm}/{stage}",
            "cpus": cpus,
            "started_unix": 0,
        }

    def docker(arguments, **kwargs):
        arm, stage = arguments[-1].split("/")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ExitCode": exit_code(arm, stage), "Running": False}),
        )

    monkeypatch.setattr(execution, "start_container", start)
    monkeypatch.setattr(execution, "docker", docker)
    monkeypatch.setattr(execution, "audit_science", lambda *args: {"passed": True})
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    execution.controller(frozen)
    final = execution.read(frozen / "controller-state.json")
    assert final["state"] == "finished_with_failures"
    assert set(final["training_completed"]) == {"perk", "fticr"}
    assert final["scientific_training"] == "complete"
    assert final["independent_audit"] == "partial"
    assert final["cloud_verification"] == "pending"
    assert final["completed"] == ["fticr"]
    assert ("perk", "replay") not in started
    assert len(started) == len(set(started)) == 5


@pytest.fixture
def cuda_frozen(frozen):
    receipt = execution.read(frozen / "frozen/freeze.json")
    receipt["development_sources"] = {
        arm: str(frozen / "development" / name)
        for arm, name in execution.DEVELOPMENT_SOURCE_NAMES.items()
    }
    for relative in execution.METHOD_SOURCE_FILES:
        for root in (
            execution.source(frozen),
            *(Path(path) for path in receipt["development_sources"].values()),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# original computation\n")
        receipt["source_files"][relative] = execution.sha(execution.source(frozen) / relative)
    execution.write(frozen / "frozen/freeze.json", receipt)
    execution.write(
        frozen / "development/perk/train/execution-manifest.json",
        {
            "adapter_sha256": receipt["source_files"][
                "src/llm_memory_editability/perk_reproduction.py"
            ],
        },
    )
    for arm in execution.ARMS:
        root = frozen / "development" / arm
        execution.write(
            root / "train/completion.json",
            {
                "global_step" if arm == "perk" else "optimizer_steps": 2,
                "validation_exposure": {"problems": 2},
            },
        )
        execution.write(
            root / "train/training-progress.json",
            {
                "inner_loss": 1.0,
                "outer_loss": 2.0,
                "validation_samples_exposed": 2,
            },
        )
        for stage in execution.STAGES:
            execution.write(
                root / f"{stage}-exit.json",
                {
                    "exit_code": 0,
                    "device": "cuda",
                    "image_sha256": receipt["image_sha256"],
                    "source_root": receipt["development_sources"][arm],
                    "source_files": {
                        path: receipt["source_files"][path]
                        for path in execution.METHOD_SOURCE_FILES
                    },
                },
            )
        populate_predictions(root / "evaluation", arm)
        populate_predictions(root / "replay", arm)
    return frozen


def test_cuda_prerequisite_binds_actual_steps_image_source_and_replay(cuda_frozen):
    frozen = cuda_frozen
    assert execution.audit_preflight(frozen)["passed"]
    progress = frozen / "development/fticr/train/training-progress.json"
    original = execution.read(progress)
    execution.write(progress, original | {"validation_samples_exposed": 1})
    with pytest.raises(ValueError, match="validate exactly two"):
        execution.audit_preflight(frozen)
    execution.write(progress, original)
    exit_path = frozen / "development/perk/evaluation-exit.json"
    original = execution.read(exit_path)
    execution.write(exit_path, original | {"source_files": {}})
    with pytest.raises(ValueError, match="stage source hashes"):
        execution.audit_preflight(frozen)
    execution.write(exit_path, original)
    bad = frozen / "development/fticr/replay-exit.json"
    execution.write(bad, {"exit_code": 0, "device": "cuda", "image_sha256": "other"})
    with pytest.raises(ValueError, match="image differs"):
        execution.audit_preflight(frozen)
    with pytest.raises(ValueError, match="artifact changed"):
        execution.launch(frozen)


@pytest.mark.parametrize("arm", execution.ARMS)
def test_successful_short_run_can_retain_unused_other_arm_source(cuda_frozen, arm):
    frozen = cuda_frozen
    receipt = execution.read(frozen / "frozen/freeze.json")
    other_arm = next(value for value in execution.ARMS if value != arm)
    relative = execution.ARM_SOURCE_FILES[other_arm][0]
    snapshot = Path(receipt["development_sources"][arm]) / relative
    snapshot.write_text("# Earlier snapshot of the unused other arm.\n")
    for stage in execution.STAGES:
        path = frozen / "development" / arm / f"{stage}-exit.json"
        value = execution.read(path)
        value["source_files"][relative] = execution.sha(snapshot)
        execution.write(path, value)
    assert execution.audit_preflight(frozen)["passed"]


@pytest.mark.parametrize("arm", execution.ARMS)
def test_short_run_rejects_changed_current_arm_dependency(cuda_frozen, arm):
    frozen = cuda_frozen
    receipt = execution.read(frozen / "frozen/freeze.json")
    relative = execution.ARM_SOURCE_FILES[arm][0]
    snapshot = Path(receipt["development_sources"][arm]) / relative
    snapshot.write_text("# Changed actual training dependency.\n")
    with pytest.raises(ValueError, match="method differs"):
        execution.audit_preflight(frozen)
