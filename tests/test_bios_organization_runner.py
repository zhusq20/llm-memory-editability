"""A completion filename must never silently certify a different or partial study."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm_memory_editability.bios_data import N_QUERIES, array_hash

pytest.importorskip("torch")
from llm_memory_editability import bios_organization, bios_organization_train  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "organization_runner", Path(__file__).parents[1] / "scripts/run_bios_organization.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def study():
    return json.loads(
        (Path(__file__).parents[1] / "configs/bios-organization-development-v1.json").read_text()
    )


def test_frozen_configuration_and_subsets(study):
    runner.validate_study_config(study)
    assert runner.selected_jobs(study, [1], [0], ["C"]) == [(1, 0, "C")]
    for selectors in (([2], None, None), (None, [0, 0], None), (None, None, ["SA"])):
        with pytest.raises(ValueError, match="subset"):
            runner.selected_jobs(study, *selectors)
    variants = []
    for key, value in (
        ("document_batch_size", 32),
        ("warmup_epochs", 8),
        ("learning_checkpoints", [0, 14336]),
    ):
        changed = copy.deepcopy(study)
        changed[key] = value
        variants.append(changed)
    changed = copy.deepcopy(study)
    changed["model"]["width"] = 512
    variants.append(changed)
    changed = copy.deepcopy(study)
    changed["organization_measurement"]["full_answer_behavior"] = False
    variants.append(changed)
    changed = copy.deepcopy(study)
    changed["editing"]["types"] = ["coherent"]
    variants.append(changed)
    for changed in variants:
        with pytest.raises(ValueError):
            runner.validate_study_config(changed)


@pytest.fixture
def completed_learning(tmp_path, monkeypatch, study):
    run, world_path = tmp_path / "world-0-seed-0-A", tmp_path / "data/world-0"
    run.mkdir()
    world = SimpleNamespace(
        seed=0,
        vocab_size=17,
        answers=np.array([11, 12]),
        prompts=np.array([[1, 5, 6, 2]]),
        lengths=np.array([4]),
    )
    documents = np.arange(14).reshape(2, 7)
    derived = np.arange(28).reshape(1, 28)
    monkeypatch.setattr(runner, "load_world", lambda _path: world)
    monkeypatch.setattr(bios_organization, "make_documents", lambda *_args: documents)
    monkeypatch.setattr(bios_organization_train, "derived_schedule", lambda *_args: derived)
    monkeypatch.setattr(bios_organization_train, "schedule_hash", lambda *_args: "schedule")
    monkeypatch.setattr(bios_organization_train, "core_sources", lambda: {"source": "frozen"})
    steps, width, layers = study["world_steps"], 768, 8
    model = {"vocab_size": 17, "width": width, "layers": layers, "heads": 12, "context": 128}
    config = {
        "protocol": "v2.4",
        "dataset": "bios-organization-v1",
        "phase": "development",
        "world_seed": 0,
        "seed": 0,
        "condition": "A",
        "organization_seed": 0,
        "steps": steps,
        "model": model,
        "parameters": (17 + 128) * width + layers * (12 * width * width + 13 * width) + 2 * width,
        "lr": 1e-4,
        "optimizer": "AdamW",
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": 16,
        "warmup_steps": 2048,
        "steps_per_epoch": 128,
        "epochs": steps / 128,
        "documents_per_step": 16,
        "facts_per_document": 7,
        "derived_queries_per_step": 28,
        "document_weight": 0.8,
        "derived_weight": 0.2,
        "cross_document_attention": False,
        "document_shape": {"sequence": 42, "supervised_positions": 14},
        "derived_shape": {"sequence": 6, "supervised_positions": 2},
        "checkpoints": study["learning_checkpoints"],
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "lengths_sha256": array_hash(world.lengths),
        "documents_sha256": array_hash(documents),
        "document_schedule_sha256": "schedule",
        "derived_schedule_sha256": array_hash(derived),
        "core_source_sha256": {"source": "frozen"},
        "planned_atomic_presentations": steps * 112,
        "planned_derived_presentations": steps * 28,
        "planned_total_presentations": steps * 140,
        "planned_document_presentations": steps * 16,
        "planned_input_tokens_including_padding": steps * 840,
        "planned_supervised_tokens": steps * 280,
        "world": str(world_path),
        "output": str(run),
        "model_initial_sha256": "state-0",
    }
    write(run / "config.json", config)
    np.savez(run / "schedule.npz", documents=documents, derived=derived)
    points = []
    for step in study["learning_checkpoints"]:
        points.append(
            {
                "step": step,
                "presentations": step * 140,
                "base_presentations": step * 112,
                "derived_presentations": step * 28,
                "supervised_tokens": step * 280,
                "input_tokens_including_padding": step * 840,
                "exposure_sha256": f"exposure-{step}",
                "slot_exposure_sha256": f"slots-{step}",
            }
        )
        (run / f"model-{step}.pt").write_bytes(b"mockcheckpoint")
        np.savez(
            run / f"predictions-{step}.npz",
            prediction=np.arange(N_QUERIES),
            ended=np.ones(N_QUERIES, dtype=bool),
            correct=np.ones(N_QUERIES, dtype=bool),
        )
    write(run / "learning.json", points)
    write(
        run / "complete.json",
        {
            "status": "complete",
            "step": steps,
            "final": points[-1],
            "exposure_sha256": f"exposure-{steps}",
            "slot_exposure_sha256": f"slots-{steps}",
            "model_final_sha256": f"state-{steps}",
        },
    )
    monkeypatch.setattr(
        runner,
        "checkpoint_facts",
        lambda _path, _config, step: {
            "sha256": f"file-{step}",
            "state_sha256": f"state-{step}",
        },
    )
    return run, study, (0, 0, "A"), world_path


def test_learning_skip_requires_matching_identity_and_real_completion(completed_learning):
    run, study, identity, world_path = completed_learning
    assert runner.validate_learning(*completed_learning)["sha256"] == "file-14336"
    with pytest.raises(ValueError, match="condition"):
        runner.validate_learning(run, study, (0, 0, "B"), world_path)
    marker = json.loads((run / "complete.json").read_text())
    marker["step"] = 896
    write(run / "complete.json", marker)
    with pytest.raises(ValueError, match="step"):
        runner.validate_learning(*completed_learning)
    (run / "complete.json").write_text('{"status":')
    with pytest.raises(ValueError, match="malformed"):
        runner.validate_learning(*completed_learning)


def test_learning_skip_rejects_missing_checkpoint_and_mismatched_budget(completed_learning):
    run, _, _, _ = completed_learning
    (run / "model-896.pt").unlink()
    with pytest.raises(ValueError, match="Missing or empty"):
        runner.validate_learning(*completed_learning)
    (run / "model-896.pt").write_bytes(b"mockcheckpoint")
    config = json.loads((run / "config.json").read_text())
    config["planned_supervised_tokens"] -= 2
    write(run / "config.json", config)
    with pytest.raises(ValueError, match="planned_supervised_tokens"):
        runner.validate_learning(*completed_learning)


@pytest.fixture
def completed_edits(completed_learning):
    run, study, _, world_path = completed_learning
    directory, edit = run / "edits", study["editing"]
    write(
        directory / "run.json",
        {
            "supports": edit["support_indices"],
            "scopes": edit["scopes"],
            "steps": edit["max_steps"],
            "window": 3,
            "lr": edit["learning_rate"],
            "retention": edit["retention_kl_weight"],
            "branches": False,
            "world_step": study["world_steps"],
            "world": str(world_path),
            "output": str(directory),
            "checkpoint": str(run / "model-14336.pt"),
        },
    )
    cases = []
    for support in edit["support_indices"]:
        for kind in edit["types"]:
            for scope in edit["scopes"]:
                case = directory / f"support-{support}-{kind}-{scope}"
                case_config = {
                    "support": support,
                    "kind": kind,
                    "scope": scope,
                    "branch": None,
                    "branch_parameters": 0,
                    "steps": edit["max_steps"],
                    "lr": edit["learning_rate"],
                    "retention_kl_weight": 1.0,
                    "window_zero_based": [3, 4, 5],
                    "selection": "prespecified_unmatched",
                }
                points = [{"step": step, "E": 1.0, "D": 0.5} for step in edit["checkpoints"]]
                result = {**case_config, "status": "complete", "final": points[-1]}
                write(case / "config.json", case_config)
                write(case / "trajectory.json", points)
                write(case / "complete.json", result)
                np.savez(
                    case / "sets.npz",
                    E=np.arange(93),
                    D=np.arange(96),
                    target=np.arange(N_QUERIES),
                    old_correct=np.ones(N_QUERIES, dtype=bool),
                )
                for step in edit["checkpoints"]:
                    np.savez(
                        case / f"predictions-{step}.npz",
                        prediction=np.arange(N_QUERIES),
                        ended=np.ones(N_QUERIES, dtype=bool),
                    )
                cases.append(result)
    write(directory / "complete.json", {"status": "complete", "cases": cases})
    return run, study, world_path


def test_editing_skip_requires_exact_cases_and_budget(completed_edits):
    run, study, world_path = completed_edits
    runner.validate_editing(*completed_edits)
    path = run / "edits/complete.json"
    complete = json.loads(path.read_text())
    complete["cases"].pop()
    write(path, complete)
    with pytest.raises(ValueError, match="cases"):
        runner.validate_editing(*completed_edits)
    path.unlink()
    runner.validate_editing(run, study, world_path, allow_partial=True)
    # A cached individual case must be checked even without an aggregate marker.
    case = run / "edits/support-0-coherent-mlp/complete.json"
    result = json.loads(case.read_text())
    result["steps"] = 256
    write(case, result)
    with pytest.raises(ValueError, match="steps"):
        runner.validate_editing(run, study, world_path, allow_partial=True)


def test_editing_skip_rejects_different_model_predictions(completed_edits):
    run, _, _ = completed_edits
    path = run / "edits/support-0-coherent-mlp/predictions-0.npz"
    np.savez(path, prediction=np.arange(N_QUERIES) + 1, ended=np.ones(N_QUERIES, dtype=bool))
    with pytest.raises(ValueError, match="step-zero"):
        runner.validate_editing(*completed_edits)


def test_measurement_skip_requires_matching_checkpoint_and_complete_behavior(tmp_path, study):
    run = tmp_path / "run"
    directory = run / "organization-final"
    marker = {
        "step": 14336,
        "behavior": True,
        "checkpoint_sha256": "expected",
        "checkpoint": str(run / "model-14336.pt"),
        "geometry": [{"layer": layer} for layer in range(8)],
        "probes": [
            {
                "recipient": recipient,
                "layer": layer,
                "intervention": kind,
                "generation": {
                    name: {}
                    for name in (
                        "same_company_nonexception",
                        "old_exception",
                        "independent_attribute",
                        "unrelated_company",
                    )
                },
            }
            for recipient in range(8)
            for layer in range(8)
            for kind in (
                "donor",
                "same_answer_donor",
                "wrong_donor",
                "random_equal_norm",
                "ablation",
            )
        ],
    }
    write(directory / "measurements.json", marker)
    for name in ("actual-city-activations.npz", "input-activation-output-position-sample.npz"):
        np.savez(directory / name, data=np.zeros(1))
    runner.validate_measurement(run, study, "expected")
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        runner.validate_measurement(run, study, "stale")
    marker["probes"].pop()
    write(directory / "measurements.json", marker)
    with pytest.raises(ValueError, match="probes"):
        runner.validate_measurement(run, study, "expected")


def test_checkpoint_facts_validate_serialized_step_and_weights(tmp_path):
    import torch

    from llm_memory_editability.bios_organization_train import state_hash

    path = tmp_path / "model.pt"
    model = {"width": 1}
    state = {"weight": torch.tensor([1.0, 2.0])}
    torch.save({"step": 10, "config": model, "model": state}, path)
    result = runner.checkpoint_facts(path, {"model": model, "parameters": 2}, 10)
    assert result["state_sha256"] == state_hash(state)
    with pytest.raises(ValueError, match="step"):
        runner.checkpoint_facts(path, {"model": model, "parameters": 2}, 11)


def test_runner_does_not_take_over_active_run(tmp_path, monkeypatch, study):
    root = tmp_path / "results"
    config = tmp_path / "config.json"
    write(config, study)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--gpus",
            "0",
            "--root",
            str(root),
            "--config",
            str(config),
            "--worlds",
            "1",
            "--seeds",
            "0",
            "--conditions",
            "C",
        ],
    )
    monkeypatch.setattr(runner, "active_processes", lambda _run: [123])
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched"))
    runner.main()
    ledger = json.loads(next(root.glob("execution-*.json")).read_text())
    assert ledger["planned"] == [[1, 0, "C"]]
    assert ledger["records"][0]["status"] == "already_running"


def test_runner_skips_only_valid_completed_phase(completed_learning, monkeypatch):
    run, study, _, world_path = completed_learning
    config = run.parent / "study-input.json"
    write(config, study)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--gpus",
            "0",
            "--root",
            str(run.parent),
            "--config",
            str(config),
            "--world-root",
            str(world_path.parent),
            "--phase",
            "learning",
            "--worlds",
            "0",
            "--seeds",
            "0",
            "--conditions",
            "A",
        ],
    )
    monkeypatch.setattr(runner, "active_processes", lambda _run: [])
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("launched"))
    runner.main()
    ledger = json.loads(next(run.parent.glob("execution-*.json")).read_text())
    assert ledger["records"][0]["status"] == "already_complete"
    (run / "complete.json").write_text("not valid json")
    with pytest.raises(SystemExit) as failure:
        runner.main()
    assert failure.value.code == 1
    ledgers = sorted(run.parent.glob("execution-*.json"))
    assert len(ledgers) == 2  # Separate invocations never overwrite the execution ledger.
    records = json.loads(ledgers[-1].read_text())["records"]
    assert records[-1]["status"] == "failed" and "malformed" in records[-1]["reason"]
