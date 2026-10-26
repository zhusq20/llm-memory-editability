"""End-to-end engineering fixtures, not evidence about learned capabilities."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.path_overlap import sha256
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_experiment import audit, calibrate, evaluate, load_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_inputs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    atoms = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0], [4, 0, 1], [5, 0, 3]])
    chains = np.array(
        [
            [0, 0, 1, 2],
            [4, 0, 1, 2],
            [2, 0, 1, 0],
            [5, 0, 1, 0],
            [1, 1, 0, 3],
            [3, 1, 0, 1],
        ]
    )
    np.savez(
        data / "world.npz",
        atoms=atoms,
        chains=chains,
        first=np.array([0, 4, 2, 5, 1, 3]),
        second=np.array([1, 1, 3, 3, 2, 0]),
        ood=np.array([False, False, True, True, False, False]),
        kind=np.array([0, 0, 3, 3, 2, 2]),
        train_order=np.array([], dtype=np.int64),
    )
    np.savez(data / "panels.npz", atomic_id=atoms)
    metadata = {
        "entities": 6,
        "relations": 2,
        "entity_offset": 3,
        "relation_offset": 11,
        "end_marker": 2,
        "pad_token": 0,
        "vocab_size": 29,
        "atomic_id": 4,
        "world_sha256": sha256(data / "world.npz"),
        "panels_sha256": sha256(data / "panels.npz"),
    }
    (data / "complete.json").write_text(json.dumps(metadata))
    torch.manual_seed(10)
    cfg = GPT2Config(
        vocab_size=29,
        n_positions=40,
        n_embd=16,
        n_layer=2,
        n_head=2,
        embd_pdrop=0,
        resid_pdrop=0,
        attn_pdrop=0,
    )
    parent = SupervisedAlignmentGPT(cfg, repeats=2, discoloop=True).eval()
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "step": 300000,
            "model": parent.state_dict(),
            "spec": {"unique_layers": 2, "repeats": 2, "phi": 7.2},
            "architecture": {"discoloop": True, "first_round_kv": False},
        },
        checkpoint,
    )
    hf_config = tmp_path / "config.json"
    cfg.to_json_file(hf_config)
    config = copy.deepcopy(
        json.loads((ROOT / "configs/ttcd-second-hop-development-v1.json").read_text())
    )
    config.update(data_dir=str(data), model_config=str(hf_config))
    config["models"] = {
        "discoloop": {"checkpoint": str(checkpoint), "step": 300000, "discoloop": True}
    }
    config["cases"].update(calibration_cases=1, evaluation_cases=1, affected=2, controls=2)
    config["reading_gate"].update(minimum_cases_per_group=1, minimum_accuracy=0.0)
    config["writer"]["steps"] = 1
    config["atomic_ce"]["steps"] = 1
    config["engineering_fixture_only"] = True
    out = tmp_path / "out"
    out.mkdir()
    return config, out


def test_reading_gate_failure_retained_and_cannot_edit(tmp_path):
    config, out = fixture_inputs(tmp_path)
    # Deliberately impossible threshold in this internal fixture forces failure.
    config["reading_gate"]["minimum_cases_per_group"] = 2
    result = calibrate(config, "discoloop", out, "cpu")
    assert not result["passed"]
    assert result["status"] == "reading_needs_attention"
    assert len(result["records"]) == 2
    with pytest.raises(ValueError, match="Passing reading calibration"):
        evaluate(config, "discoloop", out, "cpu")
    assert not (out / "cases").exists()


def test_full_fixture_matrix_and_fresh_process_audit(tmp_path):
    config, out = fixture_inputs(tmp_path)
    assert calibrate(config, "discoloop", out, "cpu")["passed"]
    result = evaluate(config, "discoloop", out, "cpu")
    assert result["records"] == 6  # Two cases, three writer conditions.
    paths = list((out / "cases").glob("*/*/readout.json"))
    assert len(paths) == 6
    for path in paths:
        record = json.loads(path.read_text())
        assert record["exposure"]["base_parameter_updates"] == 0
        assert set(record["open_context"]) == {"atomic", "two_hop"}
        if record["method"] == "sham":
            assert record["before"] == record["after"]
    cfg_path = tmp_path / "experiment.json"
    cfg_path.write_text(json.dumps(config))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_second_hop.py"),
            "audit",
            "--config",
            str(cfg_path),
            "--model",
            "discoloop",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--tracking",
            "disabled",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((out / "audit.json").read_text())["passed"]
    # A full bank cannot be claimed audited after losing a saved result.
    paths[0].unlink()
    with pytest.raises(ValueError, match="Incomplete execution matrix"):
        audit(config, "discoloop", out, "cpu")


def test_parent_architecture_and_hash_mismatches_fail(tmp_path):
    config, _ = fixture_inputs(tmp_path)
    config["models"]["discoloop"]["discoloop"] = False
    with pytest.raises(ValueError, match="architecture"):
        load_inputs(config, "discoloop", "cpu")
    config["models"]["discoloop"]["discoloop"] = True
    config["models"]["discoloop"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        load_inputs(config, "discoloop", "cpu")


def test_changed_configuration_invalidates_reading_receipt(tmp_path):
    config, out = fixture_inputs(tmp_path)
    calibrate(config, "discoloop", out, "cpu")
    config["writer"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="identical code/config/inputs"):
        evaluate(config, "discoloop", out, "cpu")


@pytest.mark.skipif(
    os.environ.get("TTCD_TEST_CUDA") != "1", reason="Explicit isolated GPU test only"
)
def test_cuda_actual_write_gradient_and_reload(tmp_path):
    config, out = fixture_inputs(tmp_path)
    assert calibrate(config, "discoloop", out, "cuda:0")["passed"]
    evaluate(config, "discoloop", out, "cuda:0")
    assert audit(config, "discoloop", out, "cuda:0")["passed"]
    torch.cuda.synchronize()
    config_path = tmp_path / "cuda-fixture.json"
    config_path.write_text(json.dumps(config))
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_second_hop.py"),
            "audit",
            "--config",
            str(config_path),
            "--out",
            str(out),
            "--model",
            "discoloop",
            "--device",
            "cuda:0",
            "--tracking",
            "disabled",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stderr
