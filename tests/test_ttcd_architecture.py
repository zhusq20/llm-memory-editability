"""Ordinary/Loop architecture adapters preserve the original TTCD rules."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_ttcd_double_new_eval import material as double_fixture
from test_ttcd_verified_base import fixture as verified_fixture
from transformers import GPT2Config

from llm_memory_editability import ttcd_architecture as experiment
from llm_memory_editability import ttcd_verified_base as verified
from llm_memory_editability.grokking_reproduction import (
    MergedBatchStream,
    ReproductionGPT,
    model_digest,
)
from llm_memory_editability.path_overlap import sha256
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_ttcd_architecture", ROOT / "scripts/run_ttcd_architecture.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(directory, name):
    config = verified_fixture(directory)
    config["architecture"] = {
        "name": name,
        "unique_layers": 8 if name == "transformer" else 4,
        "repeats": 1 if name == "transformer" else 2,
        "discoloop": False,
    }
    config["models"] = {"stream": {**config["models"]["stream"], "discoloop": False}}
    config["reading_gate"]["minimum_accuracy"] = 0.6
    config["capability_gate"].update(
        minimum_atomic_and_ii=0.98, minimum_io_oi_oo=0.9, maximum_drop=0.02
    )
    hf = GPT2Config.from_json_file(config["model_config"])
    hf.n_layer = config["architecture"]["unique_layers"]
    hf.embd_pdrop = hf.resid_pdrop = hf.attn_pdrop = 0.0
    torch.manual_seed(91)
    backbone = ReproductionGPT(hf, config["architecture"]["repeats"]).eval()
    saved = {
        "model": backbone.state_dict(),
        "model_sha256": model_digest(backbone),
        "step": 300000,
        "spec": {"unique_layers": hf.n_layer, "repeats": backbone.repeats, "phi": config["phi"]},
        # Native historical checkpoints need not have a Disco architecture field.
    }
    torch.save(saved, config["foundation"]["checkpoint"])
    Path(config["foundation"]["gate_receipt"]).write_text(
        json.dumps(
            {
                "passed": True,
                "checkpoint_sha256": sha256(config["foundation"]["checkpoint"]),
                "model_sha256": saved["model_sha256"],
                "native_accuracy": 0.0,
                "quality_selection": False,
            }
        )
    )
    return config, saved


@pytest.mark.parametrize("name,layers,repeats", [("transformer", 8, 1), ("loop", 4, 2)])
def test_native_checkpoint_wrapper_and_physical_fast_capacity(tmp_path, name, layers, repeats):
    config, saved = fixture(tmp_path, name)
    before = copy.deepcopy(config)
    reader, world, normalized, provenance = experiment.foundation_model(config, "cpu")
    assert model_digest(reader.backbone) == saved["model_sha256"]
    assert normalized["architecture"]["discoloop"] is False
    assert provenance["quality_selection"] is False
    fast = reader.empty_fast()
    assert len(fast) == layers and len({value.data_ptr() for value in fast.values()}) == layers
    accounting = experiment.architecture_accounting(reader)
    assert accounting["effective_block_executions"] == layers * repeats == 8
    assert accounting["fast_matrices"] == layers
    assert accounting["fast_elements"] == layers * fast[0].numel()
    assert not accounting["equal_compute_or_capacity_claim"]
    checks = verified.wrapper_agreement(reader, config, world, "cpu")
    assert all(item["tokens_exact"] for item in checks)
    assert config == before
    assert set(experiment.identity(config)["config"]["models"]) == {"stream"}


def test_two_engines_have_private_globals_without_mutating_original_module(tmp_path):
    a, _ = fixture(tmp_path / "ordinary", "transformer")
    b, _ = fixture(tmp_path / "loop", "loop")
    original = (
        verified.identity,
        verified.validate_inputs,
        verified._backbone,
        verified.foundation_inputs,
    )
    first, second = experiment.build_engine(a), experiment.build_engine(b)
    assert first.train.__globals__ is not second.train.__globals__
    assert first.train.__globals__["foundation_inputs"] is experiment.foundation_model
    with ThreadPoolExecutor(max_workers=2) as pool:
        readers = list(
            pool.map(
                lambda item: item[0].foundation_inputs(item[1], "cpu")[0], [(first, a), (second, b)]
            )
        )
    assert [len(r.empty_fast()) for r in readers] == [8, 4]
    assert (
        verified.identity,
        verified.validate_inputs,
        verified._backbone,
        verified.foundation_inputs,
    ) == original


@pytest.mark.parametrize("change", ["kv", "disco", "depth", "digest", "receipt"])
def test_rejects_backbone_identity_or_unapproved_architecture_changes(tmp_path, change):
    config, saved = fixture(tmp_path, "loop")
    if change in ("kv", "disco"):
        saved["architecture"] = {"first_round_kv": change == "kv", "discoloop": change == "disco"}
    elif change == "depth":
        saved["spec"]["unique_layers"] = 8
    elif change == "digest":
        saved["model_sha256"] = "wrong"
    else:
        Path(config["foundation"]["gate_receipt"]).write_text('{"passed": false}')
    if change != "receipt":
        torch.save(saved, config["foundation"]["checkpoint"])
        Path(config["foundation"]["gate_receipt"]).write_text(
            json.dumps(
                {"passed": True, "checkpoint_sha256": sha256(config["foundation"]["checkpoint"])}
            )
        )
    with pytest.raises(ValueError):
        experiment.foundation_model(config, "cpu")


@pytest.mark.parametrize("name", ["transformer", "loop"])
def test_common_stream_training_replay_and_advisory_quality_continue_fixed_budget(tmp_path, name):
    config, _ = fixture(tmp_path, name)
    engine = experiment.build_engine(config)
    common, stream = tmp_path / "common", tmp_path / "stream"
    common.mkdir()
    stream.mkdir()
    initial = engine.preflight(config, common, "cpu")
    assert initial["passed"] and all(c["tokens_exact"] for c in initial["wrapper_agreement"])
    trained = engine.train(config, "common", common, "cpu")
    assert trained["counts"]["base_parameter_updates"] == config["common"]["steps"]
    assert trained["counts"]["corrected_two_hop_training_examples"] == 0
    assert engine.audit_training(config, "common", common, "cpu")["passed"]
    gate = engine.common_gate(config, common, "cpu")
    assert gate["passed"] and gate["numerical_audit_passed"] and not gate["quality_passed"]
    assert gate["quality_is_advisory"] and all(
        v["report_only"] for v in gate["native_checks"].values()
    )
    trained = engine.train(config, "stream", stream, "cpu")
    assert trained["counts"]["base_parameter_updates"] == config["training"]["steps"]
    assert trained["counts"]["corrected_two_hop_training_examples"] == 0
    assert engine.audit_training(config, "stream", stream, "cpu")["passed"]
    calibration = experiment.calibrate(engine, config, "stream", stream, "cpu")
    assert calibration["passed"] and calibration["quality_is_advisory"]
    reader, _ = engine.load_reader(config, "stream", "cpu")
    assert len(reader.empty_fast()) == config["architecture"]["unique_layers"]
    saved = torch.load(
        config["models"]["stream"]["checkpoint"], map_location="cpu", weights_only=False
    )
    assert set(saved["identity"]["config"]["models"]) == {"stream"}
    assert saved["architecture"]["discoloop"] is False
    summary = runner.dispatch(engine, config, "stream", "evaluate", stream, "cpu")
    assert summary["audit_state"] == "pending_fresh_process_reload"
    audit = runner.dispatch(engine, config, "stream", "audit", stream, "cpu")
    assert audit["passed"]


def test_numerical_training_audit_remains_required_despite_advisory_quality(tmp_path):
    config, _ = fixture(tmp_path, "loop")
    engine = experiment.build_engine(config)
    common = tmp_path / "common"
    common.mkdir()
    engine.train(config, "common", common, "cpu")
    with pytest.raises(FileNotFoundError):
        engine.common_gate(config, common, "cpu")
    engine.audit_training(config, "common", common, "cpu")
    receipt = Path(config["common"]["checkpoint"]).parent / "audit.json"
    failed = json.loads(receipt.read_text())
    failed["passed"] = False
    receipt.write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="Fresh training reload audit"):
        engine.common_gate(config, common, "cpu")


def test_cli_preflight_uses_fresh_process_and_refuses_single_arm(tmp_path):
    config, _ = fixture(tmp_path, "loop")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command = [
        sys.executable,
        str(ROOT / "scripts/run_ttcd_architecture.py"),
        "preflight",
        "--config",
        str(path),
        "--model",
        "common",
        "--out",
        str(tmp_path / "preflight"),
        "--device",
        "cpu",
        "--tracking",
        "disabled",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=90,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((tmp_path / "preflight/result-preflight.json").read_text())
    assert receipt["passed"] and receipt["architecture_accounting"]["fast_matrices"] == 4
    command[command.index("--model") + 1] = "single"
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and "invalid choice" in result.stderr
    failed_out = tmp_path / "preserved-failure"
    failed_out.mkdir()
    failure = failed_out / "failure-train.json"
    failure.write_text('{"preserved": true}')
    command[command.index("--model") + 1] = "common"
    command[2] = "train"
    command[command.index("--out") + 1] = str(failed_out)
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and "no automatic retry" in result.stderr
    assert failure.read_text() == '{"preserved": true}' and not (failed_out / "training").exists()


def test_standard_and_loop_fast_state_use_existing_write_rule_and_frozen_base(tmp_path):
    from llm_memory_editability.ttcd_continual import write_states

    for name in ("transformer", "loop"):
        config, _ = fixture(tmp_path / name, name)
        reader, world, _, _ = experiment.foundation_model(config, "cpu", trainable=True)
        rng = np.random.RandomState(config["seed"] + 2)
        stream = MergedBatchStream(
            len(verified.native_table(world, config["phi"])[0][0]), config["seed"] + 3
        )
        batch = verified.sample_step(config, world, rng, stream)
        before = tensor_digest(reader.backbone.state_dict())
        fast = write_states(
            reader,
            torch.as_tensor(batch["packets"]),
            config["writer"],
            config["continual"]["train_length"],
            "stream",
        )
        assert len(fast) == config["architecture"]["unique_layers"]
        assert all(value.requires_grad and torch.isfinite(value).all() for value in fast.values())
        assert tensor_digest(reader.backbone.state_dict()) == before


@pytest.mark.parametrize("name,layers,repeats", [("transformer", 8, 1), ("loop", 4, 2)])
def test_double_new_and_retention_metrics_replay_through_architecture_dispatch(
    tmp_path, name, layers, repeats
):
    config, settings, world, previous = double_fixture.__wrapped__(tmp_path)
    hf = copy.deepcopy(previous.backbone.config)
    hf.n_layer = layers
    reader = TTCDReader(
        SupervisedAlignmentGPT(hf, repeats, discoloop=False), native_causal_attention=True
    ).eval()
    engine = SimpleNamespace(
        load_reader=lambda *args: (reader, world),
        evaluation_identity=lambda *args: {"architecture": name, "fixture": True},
    )
    panel = runner.double_new.build_panel(config, settings, world)
    panel_path = tmp_path / "double-panel.json"
    panel_path.write_text(json.dumps(panel))
    out = tmp_path / "double-evaluation"
    out.mkdir()
    initial = tensor_digest(reader.backbone.state_dict())
    result = runner.dispatch(
        engine, config, "stream", "double-evaluate", out, "cpu", settings=settings, panel=panel_path
    )
    # Reload with an independent reader and the same architecture and state.
    fresh = copy.deepcopy(reader)
    engine.load_reader = lambda *args: (fresh, world)
    audit = runner.dispatch(
        engine, config, "stream", "double-audit", out, "cpu", settings=settings, panel=panel_path
    )
    assert audit["passed"] and audit["states"] == 81
    assert (
        result["primary"]["total_pair_readouts"]
        == result["primary"]["eligible_pair_readouts"]
        == 32
    )
    assert result["scientific_training_updates"] == 0
    assert result["by_fact_age"] and result["by_pair_completion_age"]
    assert set(result["primary"]["U"]) == {"U_atoms", "U_chains"}
    assert result["primary"]["earlier_atoms_retention_from_immediate"]["total"] > 0
    assert tensor_digest(reader.backbone.state_dict()) == initial
