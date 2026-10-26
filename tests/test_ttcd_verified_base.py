"""Successful-base TTCD contracts; fixtures are engineering, never science."""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ttcd_scratch import fixture_config
from transformers import GPT2Config

from llm_memory_editability import ttcd_continual_eval as evaluation
from llm_memory_editability import ttcd_verified_base as experiment
from llm_memory_editability.grokking_reproduction import MergedBatchStream, model_digest
from llm_memory_editability.path_overlap import sha256
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_scratch import prepare_scratch_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(directory, *, original=None, model_config=None, full_size=False):
    if original is None:
        legacy = fixture_config(directory / "legacy")
        legacy["cases"]["evaluation_cases"] = 2
        prepare_scratch_data(legacy)
        original, model_config, phi = legacy["data_dir"], legacy["model_config"], legacy["phi"]
    else:
        phi = 7.2
    checkpoint = directory / "foundation.pt"
    gate = directory / "foundation-gate.json"
    steps = 16 if full_size else 2
    config = {
        "batch": "ttcd-verified-engineering-fixture",
        "engineering_fixture_only": True,
        "seed": 20261015,
        "phi": phi,
        "original_data_dir": str(original),
        "original_model_config": str(model_config),
        "data_dir": str(directory / "adaptation-data"),
        "model_config": str(directory / "adaptation-data/gpt2-config.json"),
        "cases": {
            "seed": 20261015,
            "calibration_cases": 1,
            "evaluation_cases": 2,
            "affected": 2,
            "controls": 2,
        },
        "writer": {
            "steps": 4,
            "learning_rate": 0.001,
            "gap": 17,
            "teacher_window": 32,
            "student_window": 16,
        },
        "foundation": {"checkpoint": str(checkpoint), "gate_receipt": str(gate), "step": 300000},
        "common": {
            "steps": steps,
            "checkpoint": str(directory / "common/training/parent.pt"),
            "gate_receipt": str(directory / "common/gate.json"),
        },
        "models": {
            name: {
                "checkpoint": str(directory / name / "training/parent.pt"),
                "step": steps,
                "discoloop": True,
            }
            for name in ("single", "stream")
        },
        "continual": {
            "train_length": 8 if full_size else 2,
            "eval_length": 4,
            "rotations": 1,
            "affected": 2,
            "controls": 2,
            "order_seed": 1729,
        },
        "training": {
            "steps": steps,
            "native_batch_size": 512 if full_size else 8,
            "correction_batch_size": 128 if full_size else 4,
            "micro_native": 64 if full_size else 2,
            "micro_facts": 8 if full_size else 2,
            "learning_rate": 1e-5,
            "weight_decay": 0.1,
            "gradient_clip": 1.0,
            "validation_every": steps,
            "evaluation_batch_size": 64 if full_size else 4,
        },
        "reading_gate": {"minimum_cases_per_group": 1, "minimum_accuracy": 0.0},
        "capability_gate": {
            "minimum_atomic_and_ii": 0.0,
            "minimum_io_oi_oo": 0.0,
            "maximum_drop": 1.0,
        },
        "runtime": {"cpu_threads": 1},
    }
    if full_size:
        config["cases"]["evaluation_cases"] = 8
        config["continual"]["eval_length"] = 16
    experiment.prepare_data(config)
    world = experiment.validate_inputs(config)
    hf = GPT2Config.from_json_file(config["model_config"])
    hf.vocab_size = world.metadata["vocab_size"]
    hf.n_layer = 4 if full_size else 2
    torch.manual_seed(31)
    model = SupervisedAlignmentGPT(hf, repeats=2, discoloop=True).eval()
    saved = {
        "model": model.state_dict(),
        "model_sha256": model_digest(model),
        "step": 300000,
        "spec": {"unique_layers": hf.n_layer, "repeats": 2, "phi": phi},
        "architecture": {
            "discoloop": True,
            "first_round_kv": False,
            "discoloop_alpha": 1.0,
            "discoloop_temperature": 1.0,
            "discoloop_eps": 1e-6,
        },
    }
    torch.save(saved, checkpoint)
    gate.write_text(
        json.dumps(
            {
                "passed": True,
                "checkpoint_sha256": sha256(checkpoint),
                "model_sha256": saved["model_sha256"],
                "engineering_fixture_only": True,
            }
        )
    )
    del model
    return config


def test_preparation_preserves_original_bytes_and_excludes_all_reserved_keys(tmp_path):
    config = fixture(tmp_path)
    world = experiment.validate_inputs(config)
    for name in ("world.npz", "panels.npz", "complete.json"):
        assert sha256(Path(config["data_dir"]) / name) == sha256(
            Path(config["original_data_dir"]) / name
        )
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    reserved = {case["atom"] for case in split["case_bank"]}
    assert not reserved.intersection(split["train_edit_atoms"])
    table, atomic_count = experiment.native_table(world, config["phi"])
    assert atomic_count == len(world.atoms)
    assert len(table[0]) == len(world.atoms) + int(world.support(config["phi"]).sum())
    stream = MergedBatchStream(len(table[0]), 42)
    rng = np.random.RandomState(12)
    for _ in range(8):
        batch = experiment.sample_step(config, world, rng, stream)
        assert not reserved.intersection(batch["edit_ids"])
        for start in (0, 2):
            assert not set(batch["edit_ids"][start : start + 2]).intersection(
                batch["retention_ids"][start : start + 2]
            )
    before = sha256(Path(config["data_dir"]) / "complete.json")
    experiment.prepare_data(config)
    assert before == sha256(Path(config["data_dir"]) / "complete.json")
    config["continual"]["panel_source_path"] = str(tmp_path / "prepared-panel.json")
    experiment.prepare_data(config)
    assert json.loads((tmp_path / "prepared-panel.json").read_text()) == evaluation.build_panel(
        config, world
    )
    changed = copy.deepcopy(config)
    changed["phi"] = 2.0
    with pytest.raises(ValueError, match="contract changed"):
        experiment.validate_inputs(changed)


def test_foundation_is_gate_bound_and_wrapper_preserves_native_logits(tmp_path):
    config = fixture(tmp_path)
    reader, world, _, _ = experiment.foundation_inputs(config, "cpu", trainable=True)
    agreement = experiment.wrapper_agreement(reader, config, world, "cpu")
    assert len(agreement) == 7 and all(row["tokens_exact"] for row in agreement)
    original_config_bytes = Path(config["model_config"]).read_bytes()
    assert reader.backbone.config.resid_pdrop == 0.0
    assert Path(config["model_config"]).read_bytes() == original_config_bytes
    gate = Path(config["foundation"]["gate_receipt"])
    receipt = json.loads(gate.read_text())
    receipt["checkpoint_sha256"] = "bad"
    gate.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact checkpoint"):
        experiment.foundation_inputs(config, "cpu")


def test_native_table_matches_original_trainer_order_and_all_encoded_values(tmp_path):
    config = fixture(tmp_path)
    world = experiment.validate_inputs(config)
    arrays = dict(np.load(Path(config["data_dir"]) / "world.npz"))
    module_spec = importlib.util.spec_from_file_location(
        "native_supervised_for_table_contract", ROOT / "scripts/run_supervised_alignment.py"
    )
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    expected, selected = native.make_table(arrays, world.metadata, {"phi": config["phi"]}, "cpu")
    actual, atomic_count = experiment.native_table(world, config["phi"])
    assert atomic_count == len(world.atoms)
    assert not np.array_equal(selected, np.sort(selected))
    np.testing.assert_array_equal(selected, world.train_order[: len(selected)])
    for encoded, original in zip(actual, expected, strict=True):
        # Original CUDA graphs add one loss-masked sentinel row; adaptation
        # executes only the valid original final-epoch examples.
        np.testing.assert_array_equal(encoded, original[:-1].numpy())
    assert expected[-1][-1].tolist() == [-100, -100]


@pytest.mark.parametrize("phase", ["common", "single", "stream"])
def test_microbatches_match_full_objective_all_gradients_and_adam_update(tmp_path, phase):
    config = fixture(tmp_path)
    reader, world, _, _ = experiment.foundation_inputs(config, "cpu", trainable=True)
    whole = copy.deepcopy(reader)
    table, _ = experiment.native_table(world, config["phi"])
    batch = experiment.sample_step(
        config, world, np.random.RandomState(17), MergedBatchStream(len(table[0]), 18)
    )
    micro_loss, micro_parts = experiment.backward_objective(
        reader, config, world, table, batch, phase
    )
    full_loss, full_parts = experiment.backward_objective(
        whole, config, world, table, batch, phase, micro=False
    )
    np.testing.assert_allclose(micro_loss, full_loss, rtol=1e-6, atol=1e-6)
    for key in full_parts:
        np.testing.assert_allclose(micro_parts[key], full_parts[key], rtol=1e-6, atol=1e-6)
    for (name, parameter), (other_name, other) in zip(
        reader.named_parameters(), whole.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5)
    for candidate in (reader, whole):
        optimizer = experiment.optimizer_for(candidate, config)
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), config["training"]["gradient_clip"])
        optimizer.step()
        assert optimizer.param_groups[1]["weight_decay"] == 0.0
    for parameter, other in zip(reader.parameters(), whole.parameters(), strict=True):
        torch.testing.assert_close(parameter, other, atol=2e-7, rtol=2e-5)


def cli(config, stage, phase, directory, device="cpu", timeout=240):
    path = directory / "experiment.json"
    path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_verified_base.py"),
            stage,
            "--config",
            str(path),
            "--model",
            phase,
            "--out",
            str(directory / phase),
            "--device",
            device,
            "--tracking",
            "disabled",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((directory / phase / f"result-{stage}.json").read_text())


def test_common_reload_gate_and_both_branches_full_independent_replay(tmp_path):
    config = fixture(tmp_path)
    cli(config, "preflight", "common", tmp_path)
    cli(config, "train", "common", tmp_path)
    audit = cli(config, "train-audit", "common", tmp_path)
    assert audit["passed"] and audit["formal_D_used"] is False
    gate = cli(config, "gate", "common", tmp_path)
    assert gate["passed"] and gate["formal_D_used"] is False
    results = {}
    for phase in ("single", "stream"):
        results[phase] = cli(config, "train", phase, tmp_path)
        cli(config, "train-audit", phase, tmp_path)
        cli(config, "calibrate", phase, tmp_path)
        evaluation_result = cli(config, "evaluate", phase, tmp_path)
        replay = cli(config, "audit", phase, tmp_path)
        assert replay["passed"] and replay["write_states_exact"]
        assert evaluation_result["states"] == 1 + 4 + 4 + 2
        assert results[phase]["counts"]["corrected_two_hop_training_examples"] == 0
        saved = torch.load(config["models"][phase]["checkpoint"], weights_only=False)
        assert saved["common_parent_sha256"] == sha256(config["common"]["checkpoint"])
    assert results["single"]["counts"] == results["stream"]["counts"]
    assert results["single"]["trace_sha256"] == results["stream"]["trace_sha256"]
    path = Path(config["common"]["gate_receipt"])
    changed = json.loads(path.read_text())
    changed["passed"] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="passing common"):
        experiment._parent_for_phase(config, "single", "cpu")


def test_common_gate_failure_is_preserved_and_blocks_branch(tmp_path):
    config = fixture(tmp_path)
    config["capability_gate"] = {
        "minimum_accuracy": {
            name: 1.0 if name == "atomic_id" else 0.0 for name in experiment.PANELS
        },
        "maximum_reference_drop": 1.0,
    }
    experiment.train(config, "common", tmp_path / "common", "cpu")
    experiment.audit_training(config, "common", tmp_path / "common", "cpu")
    gate = experiment.common_gate(config, tmp_path / "common", "cpu")
    assert not gate["passed"] and gate["status"] == "common_gate_failed_no_branch"
    assert gate["formal_D_used"] is False
    with pytest.raises(ValueError, match="passing common"):
        experiment.train(config, "stream", tmp_path / "stream", "cpu")
    assert not (tmp_path / "stream/training").exists()


@pytest.mark.skipif(
    os.environ.get("TTCD_VERIFIED_TEST_CUDA") != "1",
    reason="Explicit full-size CUDA engineering fixture only",
)
def test_full_successful_architecture_cuda_16_updates_per_phase(tmp_path):
    assert torch.cuda.is_available()
    original = Path(os.environ["TTCD_VERIFIED_ORIGINAL_DATA"])
    model_config = Path(os.environ["TTCD_VERIFIED_ORIGINAL_CONFIG"])
    config = fixture(tmp_path, original=original, model_config=model_config, full_size=True)
    world = experiment.validate_inputs(config)
    assert (
        world.entities,
        world.metadata["relations"],
        len(world.atoms),
        world.metadata["vocab_size"],
    ) == (
        2000,
        200,
        40000,
        52463,
    )
    reader, _, _, _ = experiment.foundation_inputs(config, "cuda:0", trainable=True)
    assert reader.backbone.config.n_embd == 768 and len(reader.backbone.transformer.h) == 4
    assert sum(p.numel() for p in reader.parameters()) == 69431040
    assert sum(value.numel() for value in reader.empty_fast().values()) == 9437184
    del reader
    torch.cuda.empty_cache()
    experiment.preflight(config, tmp_path / "common", "cuda:0")
    experiment.train(config, "common", tmp_path / "common", "cuda:0")
    experiment.audit_training(config, "common", tmp_path / "common", "cuda:0")
    assert experiment.common_gate(config, tmp_path / "common", "cuda:0")["passed"]
    results = {}
    for phase in ("single", "stream"):
        results[phase] = experiment.train(config, phase, tmp_path / phase, "cuda:0")
        experiment.audit_training(config, phase, tmp_path / phase, "cuda:0")
        kwargs = {
            "reader_loader": experiment.load_reader,
            "identity_factory": experiment.evaluation_identity,
        }
        assert evaluation.calibrate(config, phase, tmp_path / phase, "cuda:0", **kwargs)["passed"]
        evaluation.evaluate(config, phase, tmp_path / phase, "cuda:0", **kwargs)
        assert evaluation.audit(config, phase, tmp_path / phase, "cuda:0", **kwargs)["passed"]
        assert results[phase]["counts"]["base_parameter_updates"] == 16
    assert results["single"]["trace_sha256"] == results["stream"]["trace_sha256"]
    assert results["single"]["counts"] == results["stream"]["counts"]
