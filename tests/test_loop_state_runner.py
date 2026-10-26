"""Scientific admission, frozen recipe and independent state-handoff replay contracts."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_loop_state_handoff")


def config():
    return json.loads((ROOT / "configs/loop-state-handoff-development-v1.json").read_text())


def test_one_raw_state_arm_has_original_scientific_recipe_and_no_tracking_empty_dicts():
    current = config()
    runner.validate_config(current)
    original = json.loads((ROOT / "configs/loop-native-handoff-development-v1.json").read_text())
    for key in ("training", "model", "handoff", "data", "evaluation_nodes", "evaluation_rounds"):
        assert current[key] == original[key]
    assert current["paired_recipe"] == original["paired_recipe"]
    assert current["input_origin"] == original["input_origin"]
    assert current["budget"]["new_scientific_updates"] == 1500000
    assert current["tracking"]["entity"] == "zsqzz"


@pytest.mark.parametrize(
    "change",
    [
        {"batch": "loop-native-handoff-development-v1"},
        {"arms": {"state-mlp": {"handoff": "invariant"}}},
        {"arms": {"state-mlp": {"handoff": "state-mlp"}, "ema": {"handoff": "ema"}}},
        {"training": {**runner.TRAINING, "steps": 32}},
        {"training": {**runner.TRAINING, "initialization": 2026101603}},
        {"training": {**runner.TRAINING, "batch_size": 8}},
        {"training": {**runner.TRAINING, "stream_seed": 2026101604}},
        {"model": {**runner.MODEL, "n_embd": 12}},
        {"handoff": {**runner.HANDOFF, "score_hidden": 8}},
        {"score_design": {**runner.SCORE_DESIGN, "additional_normalization": True}},
        {"evaluation_nodes": [0, 1500000]},
        {"smoke_steps": 1},
        {"reused_comparators": {}},
        {"source_references": []},
        {"budget": {"new_scientific_arms": 2, "new_scientific_updates": 3000000}},
        {"extra_tracking_config": {}},
    ],
)
def test_changed_science_matrix_recipe_budget_and_tracking_loss_are_rejected(change):
    with pytest.raises(ValueError):
        runner.validate_config({**config(), **change})


@pytest.mark.parametrize(
    "key,value",
    [
        ("image", "sha256:unverified"),
        ("docker_context", "default"),
    ],
)
def test_unverified_image_or_docker_context_is_rejected(key, value):
    current = config()
    current["runtime"][key] = value
    with pytest.raises(ValueError):
        runner.validate_config(current)


def arguments(stage, *extra):
    return runner.parser().parse_args(
        [
            stage,
            "--config",
            "config.json",
            "--base",
            "/output",
            "--arm",
            "state-mlp",
            *extra,
        ]
    )


@pytest.mark.parametrize("stage", ["environment", "train", "grid", "audit"])
def test_scientific_model_stages_and_environment_cannot_use_cpu(stage):
    with pytest.raises(ValueError, match="CUDA"):
        runner.validate_arguments(arguments(stage, "--device", "cpu"))


@pytest.mark.parametrize("stage", ["train", "grid", "audit", "cloud", "environment"])
@pytest.mark.parametrize(
    "option", ["--limit-rows", "--smoke-steps", "--smoke-grid-rounds", "--threads"]
)
def test_scientific_stages_reject_all_engineering_overrides_even_zero(stage, option):
    with pytest.raises(ValueError, match="overrides"):
        runner.validate_arguments(arguments(stage, option, "0"))


def test_smoke_requires_full_32_updates_and_positive_engineering_limits():
    runner.validate_arguments(arguments("smoke", "--smoke-steps", "32", "--device", "cpu"))
    runner.validate_arguments(arguments("smoke-grid", "--smoke-grid-rounds", "8"))
    with pytest.raises(ValueError):
        runner.validate_arguments(arguments("smoke", "--smoke-steps", "1"))
    with pytest.raises(ValueError):
        runner.validate_arguments(arguments("smoke", "--limit-rows", "0"))


def test_legacy_runner_contract_and_flops_restore_on_success_and_failure():
    original = runner.native.validate_config, runner.native.step_flops
    for raises in (False, True):
        try:
            with runner.native_contract():
                assert runner.native.validate_config is runner.validate_config
                assert runner.native.step_flops is runner.step_flops
                if raises:
                    raise RuntimeError("engineering failure")
        except RuntimeError:
            pass
        assert (runner.native.validate_config, runner.native.step_flops) == original


def test_state_flops_include_both_score_matrices_and_leave_legacy_handoffs_unchanged(monkeypatch):
    monkeypatch.setattr(runner, "_NATIVE_FLOPS", lambda *args: 117)
    model = SimpleNamespace(
        handoff="state-mlp",
        config=SimpleNamespace(n_embd=768),
        handoff_score_in=torch.empty(16, 1536),
    )
    score = 3 * 2 * 512 * 6 * (1536 * 16 + 16) * 9
    assert runner.step_flops(model, {"sequence_length": 6}, 512, [5, 4]) == 117 + score
    for handoff in runner.native.HANDOFFS:
        model.handoff = handoff
        assert runner.step_flops(model, {"sequence_length": 6}, 512, [5, 4]) == 117


def prediction():
    return {
        "rows": np.array([[0, 2, 3], [1, 4, 5]], dtype=np.int64),
        "answer": np.array([12, 14], dtype=np.int64),
        "stop": np.array([1, 1], dtype=np.int64),
        "nll": np.array([[0.2, 0.1], [0.3, 0.4]], dtype=np.float32),
    }


def test_replay_compares_all_fields_and_not_merely_aggregate_accuracy():
    original = prediction()
    evidence = runner.compare_prediction(original, copy.deepcopy(original))
    assert set(evidence) == set(original)
    assert all(value["exact"] for value in evidence.values())
    # Permuting two wrong answers leaves accuracy unchanged and must still fail.
    replay = copy.deepcopy(original)
    replay["answer"] = replay["answer"][::-1].copy()
    with pytest.raises(AssertionError):
        runner.compare_prediction(original, replay)
    replay = copy.deepcopy(original)
    replay["stop"][1] = 2
    with pytest.raises(AssertionError):
        runner.compare_prediction(original, replay)


@pytest.mark.parametrize("field", ["rows", "answer", "stop", "nll"])
def test_replay_rejects_dropped_fields(field):
    replay = prediction()
    del replay[field]
    with pytest.raises(AssertionError):
        runner.compare_prediction(prediction(), replay)


def test_replay_rejects_extra_fields_nonfinite_and_nll_changes():
    replay = {**prediction(), "unchecked": np.array([1])}
    with pytest.raises(AssertionError):
        runner.compare_prediction(prediction(), replay)
    for value in (np.nan, 20.0):
        replay = prediction()
        replay["nll"][0, 0] = value
        with pytest.raises(AssertionError):
            runner.compare_prediction(prediction(), replay)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def admission_files(tmp_path):
    config_path = tmp_path / "config.json"
    write_json(config_path, config())
    config_sha = runner.native.digest(config_path)
    out = tmp_path / "smoke" / "state-mlp"
    out.mkdir(parents=True)
    checkpoint = out / "checkpoint-0000032.pt"
    checkpoint.write_bytes(b"immutable engineering checkpoint")
    checkpoint_sha = runner.native.digest(checkpoint)
    write_json(
        tmp_path / "environment.json",
        {
            "passed": True,
            "config_sha256": config_sha,
            "initial_backbone_sha256": runner.BACKBONE,
        },
    )
    write_json(
        out / "complete.json",
        {
            "config_sha256": config_sha,
            "step": 32,
            "engineering_only": True,
            "final_checkpoint_sha256": checkpoint_sha,
        },
    )
    write_json(
        out / "grid.json",
        {
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
        },
    )
    write_json(
        out / "reload-audit.json",
        {
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint_sha,
            "passed": True,
            "independent_process": True,
            "prediction_files": 33,
            "step": 32,
        },
    )
    return config_path, out


def test_train_accepts_only_same_configuration_cuda_smoke_grid_and_independent_replay(tmp_path):
    config_path, out = admission_files(tmp_path)
    runner.check_training_admission(config(), tmp_path, config_path)
    replay = runner._read(out / "reload-audit.json")
    replay["prediction_files"] = 32
    write_json(out / "reload-audit.json", replay)
    with pytest.raises(ValueError, match="coverage"):
        runner.check_training_admission(config(), tmp_path, config_path)


@pytest.mark.parametrize(
    "artifact,change",
    [
        ("environment.json", {"passed": False}),
        ("environment.json", {"config_sha256": "different"}),
        ("environment.json", {"initial_backbone_sha256": "different"}),
        ("smoke/state-mlp/complete.json", {"step": 1}),
        ("smoke/state-mlp/grid.json", {"checkpoint_sha256": "different"}),
        ("smoke/state-mlp/reload-audit.json", {"passed": False}),
        ("smoke/state-mlp/reload-audit.json", {"independent_process": False}),
    ],
)
def test_training_admission_rejects_missing_failed_or_mismatched_engineering_evidence(
    tmp_path,
    artifact,
    change,
):
    config_path, _ = admission_files(tmp_path)
    path = tmp_path / artifact
    write_json(path, {**runner._read(path), **change})
    with pytest.raises(ValueError):
        runner.check_training_admission(config(), tmp_path, config_path)


def test_changed_smoke_checkpoint_is_rejected(tmp_path):
    config_path, out = admission_files(tmp_path)
    (out / "checkpoint-0000032.pt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.check_training_admission(config(), tmp_path, config_path)


def test_audit_is_a_fresh_interpreter_process(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    args = arguments("smoke-grid", "--device", "cpu")
    runner.independent_audit(args, smoke=True)
    command, options = calls.pop()
    assert command[0] == sys.executable
    assert command[2] == "smoke-audit"
    assert command[-2:] == ["--device", "cpu"]
    assert options == {"check": True}


def test_environment_requires_cuda_before_loading_data_or_starting_tracking(monkeypatch):
    monkeypatch.setattr(runner, "_load_data", lambda _: pytest.fail("must reject CPU first"))
    with pytest.raises(ValueError, match="real CUDA"):
        runner.environment(config(), Path("/output"), Path("config.json"), torch.device("cpu"))


def test_endpoint_audit_reloads_all_33_prediction_files_and_detects_changed_eos(
    tmp_path, monkeypatch
):
    current = config()
    config_path = tmp_path / "config.json"
    write_json(config_path, current)
    config_sha = runner.native.digest(config_path)
    out = tmp_path / "smoke" / "state-mlp"
    out.mkdir(parents=True)
    checkpoint = out / "checkpoint-0000032.pt"
    torch.save(
        {
            "config_sha256": config_sha,
            "world_sha256": runner.DATA_HASHES["world_sha256"],
            "step": 32,
            "handoff": "state-mlp",
            "arm": "state-mlp",
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
            "step": 32,
            "engineering_only": True,
            "final_checkpoint_sha256": runner.native.digest(checkpoint),
            "model_sha256": "endpoint-model",
        },
    )
    write_json(
        out / "run.json",
        {
            "config_sha256": config_sha,
            "pid": os.getpid() + 1,
            "initial_model_sha256": "initial-model",
            "limited_composition_rows": 8,
        },
    )
    names = runner.native.panel_names()
    panels = {name: prediction()["rows"] for name in names}
    monkeypatch.setattr(runner, "_load_data", lambda _: ({}, panels, runner.DATA_HASHES))
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
        lambda m: "endpoint-model" if m.loaded else "initial-model",
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
        return {str(r): ({}, {name: prediction() for name in names}) for r in args[-1]}

    monkeypatch.setattr(runner.native, "evaluate_rounds", evaluate)
    for kind in ("ema-r4", "ema-r8", "raw-r4"):
        folder = out / "predictions" / "0000032" / kind
        folder.mkdir(parents=True)
        for name in names:
            np.savez_compressed(folder / f"{name}.npz", **prediction())
    result = runner.audit(current, tmp_path, config_path, torch.device("cpu"), smoke=True)
    assert result["passed"] and result["independent_process"]
    assert result["prediction_files"] == 33 and result["queries"] == 66
    assert calls == [[4, 8], [4]]
    changed = prediction()
    changed["stop"][0] = 5
    np.savez_compressed(out / "predictions" / "0000032" / "raw-r4" / f"{names[-1]}.npz", **changed)
    model.loaded = False
    with pytest.raises(AssertionError):
        runner.audit(current, tmp_path, config_path, torch.device("cpu"), smoke=True)


def cloud_files(tmp_path):
    config_path = tmp_path / "config.json"
    current = config()
    write_json(config_path, current)
    config_sha = runner.native.digest(config_path)
    out = tmp_path / "runs" / "state-mlp"
    endpoint = {
        "step": 1500000,
        "metrics": {"ema": {"8": {"ood_4": {"answer_accuracy": 0.8}}}},
        "examples": 768000000,
    }
    write_json(
        out / "complete.json",
        {
            "config_sha256": config_sha,
            "step": 1500000,
            "engineering_only": False,
            "endpoint": endpoint,
        },
    )
    write_json(
        out / "reload-audit.json",
        {
            "config_sha256": config_sha,
            "passed": True,
            "prediction_files": 33,
        },
    )
    write_json(out / "tracking.json", {"id": "state-run-id", "url": "https://wandb.ai/run"})
    remote = SimpleNamespace(
        state="finished",
        config={"experiment": current},
        summary={"_step": 1500000, **runner.native.learning_metrics(endpoint)},
    )
    return current, config_path, remote


def test_cloud_checks_finished_exact_step_config_and_endpoint_metrics(tmp_path, monkeypatch):
    current, config_path, remote = cloud_files(tmp_path)
    requested = []

    def lookup(path):
        requested.append(path)
        return remote

    fake = SimpleNamespace(Api=lambda **kwargs: SimpleNamespace(run=lookup))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    result = runner.cloud(current, tmp_path, config_path)
    assert result["passed"] and result["config_exact"]
    assert requested == ["zsqzz/llm-memory-editability/state-run-id"]
    remote.state = "running"
    with pytest.raises(AssertionError):
        runner.cloud(current, tmp_path, config_path)
    remote.state = "finished"
    remote.summary["_step"] = 1499999
    with pytest.raises(AssertionError):
        runner.cloud(current, tmp_path, config_path)
    remote.summary["_step"] = 1500000
    remote.config = {"experiment": {**current, "score_design": "different"}}
    with pytest.raises(AssertionError):
        runner.cloud(current, tmp_path, config_path)
    remote.config = {"experiment": current}
    remote.summary["eval/ema/8/ood_4/answer_accuracy"] = 0.9
    with pytest.raises(AssertionError):
        runner.cloud(current, tmp_path, config_path)
