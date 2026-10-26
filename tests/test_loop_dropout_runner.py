"""Paired initialization, single recipe change and stop-versus-LR-horizon contracts."""

import copy
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_loop_dropout")


def config():
    return json.loads((ROOT / "configs/loop-dropout-development-v1.json").read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_scientific_recipe_has_exactly_three_probability_changes_and_separate_stop():
    current = config()
    runner.validate_config(current)
    baseline = json.loads((ROOT / "configs/loop-state-handoff-development-v1.json").read_text())
    assert current["training"] == baseline["training"]
    for key in ("data", "handoff", "score_design", "arms", "dropout_seed", "evaluation_rounds"):
        assert current[key] == baseline[key]
    differences = {
        key: (baseline["model"][key], current["model"][key])
        for key in baseline["model"]
        if baseline["model"][key] != current["model"][key]
    }
    assert differences == dict.fromkeys(runner.DROPOUT_KEYS, (0.1, 0.0))
    assert current["evaluation_nodes"] == [s for s in baseline["evaluation_nodes"] if s <= 500000]
    assert current["execution_stop_step"] == current["evaluation_nodes"][-1] == 500000
    assert current["training"]["steps"] == 1500000
    assert current["training"]["batch_size"] == 512
    assert current["budget"]["new_scientific_updates"] == 500000
    assert current["budget"]["learning_rate_horizon_steps"] == 1500000
    expected_lr = runner.native.learning_rate(baseline["training"], 500000)
    assert runner.native.learning_rate(current["training"], 500000) == expected_lr
    assert expected_lr > current["training"]["lr"] * current["training"]["final_lr_fraction"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("model", "attn_pdrop", 0.1),
        ("model", "embd_pdrop", 0.1),
        ("model", "resid_pdrop", 0.1),
        ("model", "n_embd", 12),
        ("training", "steps", 500000),
        ("training", "batch_size", 64),
        ("training", "initialization", 123),
        ("training", "stream_seed", 123),
        ("training", "schedule_seed", 123),
        ("training", "segments", 1),
        ("training", "weight_decay", 0.0),
        ("training", "lr", 0.00005),
        ("handoff", "score_seed", 123),
        ("paired_recipe", "initial_model_sha256", "backbone-only"),
        ("budget", "new_scientific_updates", 1500000),
        ("budget", "learning_rate_horizon_steps", 500000),
    ],
)
def test_extra_scientific_changes_are_rejected(section, key, value):
    current = config()
    current[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_config(current)


@pytest.mark.parametrize(
    "change",
    [
        {"execution_stop_step": 1500000},
        {"evaluation_nodes": [0, 500001]},
        {"evaluation_batch_size": 64},
        {"smoke_steps": 1},
        {"extra_tracking_config": {}},
    ],
)
def test_changed_endpoint_coverage_or_empty_tracking_config_is_rejected(change):
    with pytest.raises(ValueError):
        runner.validate_config({**config(), **change})


def arguments(stage, *extra):
    return runner.parser().parse_args(
        [stage, "--config", "config.json", "--base", "/output", "--arm", runner.ARM, *extra]
    )


@pytest.mark.parametrize("stage", ["environment", "train", "grid", "audit"])
def test_scientific_model_stages_reject_cpu(stage):
    with pytest.raises(ValueError, match="CUDA"):
        runner.validate_arguments(arguments(stage, "--device", "cpu"))


def test_scientific_stages_reject_smoke_limits_and_smoke_budget_is_32():
    for option in ("--limit-rows", "--smoke-steps", "--smoke-grid-rounds", "--threads"):
        with pytest.raises(ValueError, match="overrides"):
            runner.validate_arguments(arguments("train", option, "1"))
    runner.validate_arguments(arguments("smoke", "--smoke-steps", "32", "--device", "cpu"))
    with pytest.raises(ValueError):
        runner.validate_arguments(arguments("smoke", "--smoke-steps", "31"))


def tiny_model(dropout=0.0):
    torch.manual_seed(71)
    hf = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        attn_pdrop=dropout,
        embd_pdrop=dropout,
        resid_pdrop=dropout,
    )
    return runner.native.LoopNativeHandoffGPT(hf, 3, handoff=runner.ARM, score_seed=11)


def test_full_initialization_catches_scorer_change_with_unchanged_backbone():
    model, original = tiny_model(), tiny_model(dropout=0.1)
    expected = runner.native.model_digest(original)
    assert runner.assert_full_initialization(model, original.state_dict(), expected) > 0
    backbone = runner.native.backbone_digest(model)
    with torch.no_grad():
        model.handoff_score_out.add_(1)
    assert runner.native.backbone_digest(model) == backbone
    with pytest.raises(AssertionError, match="handoff_score_out"):
        runner.assert_full_initialization(model, original.state_dict(), expected)


def test_full_width_initialization_matches_reused_checkpoint_zero_on_cpu():
    current = config()
    _, _, metadata = runner.original._load_data(current)
    model = runner._INITIALIZE(current, runner.ARM, metadata, torch.device("cpu"))
    comparator = current["reused_comparators"][runner.ARM]
    saved = torch.load(
        Path(comparator["run"]) / "checkpoint-0000000.pt", map_location="cpu", weights_only=False
    )
    assert saved["step"] == 0 and saved["arm"] == runner.ARM
    assert saved["config_sha256"] == comparator["baseline_config_sha256"]
    assert saved["model_sha256"] == runner.INITIAL_MODEL
    assert runner.assert_full_initialization(model, saved["model"]) > 0
    runner.assert_zero_dropout(model)


def test_zero_dropout_covers_modules_functional_config_and_train_eval_segment_replay():
    model = tiny_model()
    result = runner.assert_zero_dropout(model)
    assert result["module_probabilities"]
    assert result["functional_probabilities"] == dict.fromkeys(runner.DROPOUT_KEYS, 0.0)
    tokens = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5]])
    positions = torch.tensor([[3, 4], [3, 4]])
    with torch.no_grad():
        model.train()
        train = model.run(tokens, positions, rounds=5, carry=False)[0]
        _, state = model.run(tokens, positions, rounds=2)
        continued = model.run(tokens, positions, rounds=3, state=state, carry=False)[0]
        model.eval()
        evaluate = model.run(tokens, positions, rounds=5, carry=False)[0]
    assert torch.equal(train, continued) and torch.equal(train, evaluate)
    model.config.attn_pdrop = 0.1
    with pytest.raises(AssertionError):
        runner.assert_zero_dropout(model)
    model.config.attn_pdrop = 0.0
    next(m for m in model.modules() if isinstance(m, torch.nn.Dropout)).p = 0.1
    with pytest.raises(AssertionError):
        runner.assert_zero_dropout(model)


def test_frozen_loop_source_and_function_identity_and_restoration():
    runner.verify_frozen_sources()
    train = runner.native.train
    before = (
        runner.native.validate_config,
        runner.native.step_flops,
        runner.native.initialize,
        runner.native.write_json,
    )
    with pytest.raises(RuntimeError):
        with runner.native_contract():
            assert runner.native.train is train is runner._NATIVE_TRAIN
            assert runner.native.validate_config is runner.validate_config
            raise RuntimeError("engineered failure")
    assert before == (
        runner.native.validate_config,
        runner.native.step_flops,
        runner.native.initialize,
        runner.native.write_json,
    )
    assert runner.native.train is train


def test_output_budget_adaptation_does_not_shorten_training_horizon():
    current = config()
    status = {"budget": current["training"]["steps"], "step": 500000}
    assert runner.report_budget(status) is status
    assert status["budget"] == status["execution_stop_step"] == 500000
    assert status["lr_horizon_steps"] == current["training"]["steps"] == 1500000
    smoke = {"budget": 32}
    assert runner.report_budget(smoke) == {"budget": 32}
    previous = runner.original.TRAINING
    with runner.endpoint_contract():
        assert runner.original.TRAINING["steps"] == 500000
        assert current["training"]["steps"] == runner.TRAINING["steps"] == 1500000
    assert runner.original.TRAINING is previous


def test_training_rejects_old_environment_without_complete_initialization_proof(tmp_path):
    write_json(
        tmp_path / "environment.json", {"passed": True, "initial_backbone_sha256": runner.BACKBONE}
    )
    with pytest.raises(ValueError, match="full-initialization"):
        runner.check_training_admission(config(), tmp_path, tmp_path / "config.json")


def test_training_admission_checks_reference_source_dropout_and_independent_smoke(
    tmp_path, monkeypatch
):
    current = config()
    reference = tmp_path / "reference"
    reference.mkdir()
    checkpoint = reference / "checkpoint-0000000.pt"
    checkpoint.write_bytes(b"original reference")
    current["reused_comparators"][runner.ARM]["run"] = str(reference)
    config_path = tmp_path / "config.json"
    write_json(config_path, current)
    config_sha = runner.native.digest(config_path)
    environment = {
        "passed": True,
        "config_sha256": config_sha,
        "initial_backbone_sha256": runner.BACKBONE,
        "initial_model_sha256": runner.INITIAL_MODEL,
        "reference_initial_model_sha256": runner.INITIAL_MODEL,
        "initial_full_model_exact": True,
        "train_eval_logits_exact": True,
        "train_segmented_logits_exact": True,
        "frozen_source_sha256": runner.FROZEN_SOURCES,
        "reference_step_zero_checkpoint_sha256": runner.native.digest(checkpoint),
        "dropout_contract": {
            "module_probabilities": {"mlp.dropout": 0.0},
            "functional_probabilities": dict.fromkeys(runner.DROPOUT_KEYS, 0.0),
        },
    }
    write_json(tmp_path / "environment.json", environment)
    # The original guard must still require all 32 smoke updates and 33-file replay.
    with pytest.raises(FileNotFoundError):
        runner.check_training_admission(current, tmp_path, config_path)
    calls = []
    monkeypatch.setattr(runner, "_ADMISSION", lambda *args: calls.append(args))
    runner.check_training_admission(current, tmp_path, config_path)
    assert len(calls) == 1
    for key, value in (
        ("initial_full_model_exact", False),
        ("train_eval_logits_exact", False),
        ("frozen_source_sha256", {"modified": "source"}),
        ("reference_step_zero_checkpoint_sha256", "changed"),
    ):
        write_json(tmp_path / "environment.json", {**environment, key: value})
        with pytest.raises(ValueError):
            runner.check_training_admission(current, tmp_path, config_path)
    changed = copy.deepcopy(environment)
    changed["dropout_contract"]["module_probabilities"]["mlp.dropout"] = 0.1
    write_json(tmp_path / "environment.json", changed)
    with pytest.raises(ValueError, match="dropout"):
        runner.check_training_admission(current, tmp_path, config_path)


def test_independent_audit_invokes_new_adapter(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    runner.independent_audit(arguments("smoke-grid", "--device", "cpu"), smoke=True)
    command, options = calls[0]
    assert command[0] == sys.executable
    assert Path(command[1]).name == "run_loop_dropout.py" and command[2] == "smoke-audit"
    assert options == {"check": True}


def test_scientific_replay_accepts_500k_and_checks_all_33_prediction_files(tmp_path, monkeypatch):
    current = config()
    config_path = tmp_path / "config.json"
    write_json(config_path, current)
    config_sha = runner.native.digest(config_path)
    out = tmp_path / "runs" / runner.ARM
    out.mkdir(parents=True)
    checkpoint = out / "checkpoint-0500000.pt"
    torch.save(
        {
            "config_sha256": config_sha,
            "world_sha256": runner.original.DATA_HASHES["world_sha256"],
            "step": 500000,
            "handoff": runner.ARM,
            "arm": runner.ARM,
            "model": {"weight": 3},
            "ema": {"weight": 2},
            "model_sha256": "endpoint-model",
        },
        checkpoint,
    )
    write_json(
        out / "complete.json",
        {
            "config_sha256": config_sha,
            "step": 500000,
            "engineering_only": False,
            "final_checkpoint_sha256": runner.native.digest(checkpoint),
            "model_sha256": "endpoint-model",
        },
    )
    write_json(
        out / "run.json",
        {
            "config_sha256": config_sha,
            "pid": os.getpid() + 1,
            "initial_model_sha256": runner.INITIAL_MODEL,
        },
    )
    prediction = {
        "rows": np.array([[0, 2, 3]], dtype=np.int64),
        "answer": np.array([12], dtype=np.int64),
        "stop": np.array([1], dtype=np.int64),
        "nll": np.array([[0.2, 0.1]], dtype=np.float32),
    }
    names = runner.native.panel_names()
    panels = dict.fromkeys(names, prediction["rows"])
    monkeypatch.setattr(
        runner.original, "_load_data", lambda _: ({}, panels, runner.original.DATA_HASHES)
    )
    model = SimpleNamespace(loaded=False, eval=lambda: None)

    def load_state(values, strict):
        assert strict and values == {"weight": 3}
        model.loaded = True

    model.load_state_dict = load_state
    monkeypatch.setattr(runner.native, "initialize", lambda *args: model)
    monkeypatch.setattr(runner.native, "backbone_digest", lambda _: runner.BACKBONE)
    monkeypatch.setattr(
        runner.native,
        "model_digest",
        lambda m: "endpoint-model" if m.loaded else runner.INITIAL_MODEL,
    )
    monkeypatch.setattr(
        runner.native,
        "ParameterAverage",
        lambda *args: SimpleNamespace(
            load_state_dict=lambda value: None,
            applied=lambda _: runner.native._nothing(),
        ),
    )
    calls = []

    def evaluate(*args):
        calls.append(args[-1])
        return {str(r): ({}, dict.fromkeys(names, prediction)) for r in args[-1]}

    monkeypatch.setattr(runner.native, "evaluate_rounds", evaluate)
    for kind in ("ema-r4", "ema-r8", "raw-r4"):
        folder = out / "predictions" / "0500000" / kind
        folder.mkdir(parents=True)
        for name in names:
            np.savez_compressed(folder / f"{name}.npz", **prediction)
    result = runner.audit(current, tmp_path, config_path, torch.device("cpu"), smoke=False)
    assert result["passed"] and result["independent_process"]
    assert result["step"] == 500000 and result["prediction_files"] == 33
    assert result["queries"] == 33 and calls == [[4, 8], [4]]
    changed = copy.deepcopy(prediction)
    changed["stop"][0] = 5
    np.savez_compressed(out / "predictions" / "0500000" / "raw-r4" / f"{names[-1]}.npz", **changed)
    model.loaded = False
    with pytest.raises(AssertionError):
        runner.audit(current, tmp_path, config_path, torch.device("cpu"), smoke=False)


def test_scientific_cloud_endpoint_is_500k_with_unchanged_1500k_config(tmp_path, monkeypatch):
    current = config()
    config_path = tmp_path / "config.json"
    write_json(config_path, current)
    config_sha = runner.native.digest(config_path)
    out = tmp_path / "runs" / runner.ARM
    endpoint = {"step": 500000, "metrics": {"ema": {"8": {"ood_4": {"answer_accuracy": 0.5}}}}}
    write_json(
        out / "complete.json",
        {
            "step": 500000,
            "engineering_only": False,
            "config_sha256": config_sha,
            "endpoint": endpoint,
        },
    )
    write_json(
        out / "reload-audit.json",
        {
            "passed": True,
            "prediction_files": 33,
            "config_sha256": config_sha,
        },
    )
    write_json(out / "tracking.json", {"id": "paired", "url": "https://wandb.ai/run"})
    remote = SimpleNamespace(
        state="finished",
        config={"experiment": current},
        summary={"_step": 500000, **runner.native.learning_metrics(endpoint)},
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Api=lambda **kw: SimpleNamespace(run=lambda _: remote),
        ),
    )
    assert runner.cloud(current, tmp_path, config_path)["step"] == 500000
    assert current["training"]["steps"] == 1500000
    remote.summary["_step"] = 1500000
    with pytest.raises(AssertionError):
        runner.cloud(current, tmp_path, config_path)
