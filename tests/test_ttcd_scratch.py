"""Scratch training contracts and fresh-process replay; fixtures are not science."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest
from llm_memory_editability.ttcd_experiment import load_inputs
from llm_memory_editability.ttcd_scratch import (
    audit_scratch_training,
    new_scratch_reader,
    outer_objective,
    prepare_scratch_data,
    sample_batch,
    session_write,
    train_scratch,
    validate_scratch_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_config(tmp_path, *, full_size=False):
    config = copy.deepcopy(
        json.loads((ROOT / "configs/ttcd-second-hop-scratch-v1.json").read_text())
    )
    config.update(batch="scratch-engineering-fixture", engineering_fixture_only=True)
    config["data_dir"] = str(tmp_path / "data")
    config["model_config"] = str(tmp_path / "data/gpt2-config.json")
    config["training"].update(
        warmup_steps=2,
        adaptation_steps=2,
        batch_size=2,
        validation_every=4,
        validation_panel_size=4,
        lr_warmup_steps=1,
    )
    if full_size:
        config["training"].update(
            warmup_steps=16, adaptation_steps=16, batch_size=128, validation_every=16
        )
    else:
        config["world"].update(entities=64, relations=8, degree=4, holdout=0.2)
        config["phi"] = 1.0
        config["model"].update(width=16, heads=2, intermediate=32, positions=40)
    config["cases"].update(calibration_cases=1, evaluation_cases=1, affected=2, controls=2)
    config["reading_gate"].update(minimum_cases_per_group=1, minimum_accuracy=0.0)
    config["atomic_ce"]["steps"] = 1
    for name, spec in config["models"].items():
        spec["checkpoint"] = str(tmp_path / name / "training/parent.pt")
        spec["step"] = config["training"]["warmup_steps"] + config["training"]["adaptation_steps"]
    return config


def cli(stage, config_path, name, out, *, device="cpu"):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_second_hop.py"),
            stage,
            "--config",
            str(config_path),
            "--model",
            name,
            "--out",
            str(out),
            "--device",
            device,
            "--tracking",
            "disabled",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_data_and_training_labels_exclude_edited_compositions(tmp_path):
    config = fixture_config(tmp_path)
    prepare_scratch_data(config)
    world = validate_scratch_inputs(config)
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    reserved = {c["atom"] for c in split["case_bank"]}
    support = world.support(config["phi"])
    assert reserved.isdisjoint(split["train_edit_atoms"])
    assert {c["group"] for c in split["case_bank"]} == {"id", "ood"}
    assert not any(support[c["D"]].any() for c in split["case_bank"])
    rng = np.random.RandomState(99)
    for _ in range(25):
        batch = sample_batch(config, world, rng)
        assert reserved.isdisjoint(batch["edit_ids"])
        np.testing.assert_array_equal(
            batch["corrected_atoms"][:, :2], world.atoms[batch["edit_ids"], :2]
        )
        assert (batch["corrected_atoms"][:, 2] != world.atoms[batch["edit_ids"], 2]).all()
        assert (
            (batch["retention_atoms"][:, :2] != batch["corrected_atoms"][:, :2]).any(axis=1).all()
        )
        assert {tuple(r) for r in batch["old_chains"]}.issubset(
            {tuple(r) for r in world.chains[support]}
        )
    assert prepare_scratch_data(config) == world.metadata
    changed = copy.deepcopy(config)
    changed["model"]["width"] *= 2
    with pytest.raises(ValueError, match="contract changed"):
        validate_scratch_inputs(changed)


@pytest.mark.parametrize("disco", [False, True])
def test_outer_write_matches_inference_and_never_mixes_sessions(tmp_path, disco):
    config = fixture_config(tmp_path)
    prepare_scratch_data(config)
    world = validate_scratch_inputs(config)
    reader = new_scratch_reader(config, "discoloop" if disco else "loop", world, "cpu")
    batch = sample_batch(config, world, np.random.RandomState(27))
    packets = torch.as_tensor(batch["packets"])
    fast = None
    for _ in range(config["writer"]["steps"]):
        fast = session_write(reader, packets, fast, config["writer"])
    assert all(v.requires_grad and v.grad_fn is not None for v in fast.values())
    frozen = TTCDReader(copy.deepcopy(reader.backbone)).eval()
    for index in range(len(packets)):
        state = frozen.empty_fast()
        for _ in range(config["writer"]["steps"]):
            state, _ = frozen.write(
                packets[index : index + 1],
                state,
                **{
                    k: config["writer"][k]
                    for k in ("teacher_window", "student_window", "learning_rate")
                },
            )
        for layer in state:
            torch.testing.assert_close(fast[layer][index], state[layer], atol=1e-9, rtol=1e-5)
    altered = packets.clone()
    altered[1, 2] = 2 + ((int(altered[1, 2]) - 2 + 1) % world.entities)
    changed = session_write(reader, altered, None, config["writer"])
    single = session_write(reader, packets, None, config["writer"])
    for layer in changed:
        torch.testing.assert_close(changed[layer][0], single[layer][0], rtol=0, atol=0)
    assert any(not torch.equal(changed[k][1], single[k][1]) for k in changed)
    initial = tensor_digest(reader.backbone.state_dict())
    loss, losses = outer_objective(reader, config, world, batch, adaptation=True)
    loss.backward()
    assert set(losses) == set(config["training"]["loss_weights"])
    assert (
        tensor_digest(reader.backbone.state_dict()) == initial
    )  # forward/write never changes base
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())
    assert sum(float(p.grad.square().sum()) for p in reader.parameters()) > 0


def test_matching_initialization_and_reproducible_data_stream(tmp_path):
    config = fixture_config(tmp_path)
    prepare_scratch_data(config)
    world = validate_scratch_inputs(config)
    a = new_scratch_reader(config, "discoloop", world, "cpu")
    b = new_scratch_reader(config, "loop", world, "cpu")
    assert tensor_digest(a.backbone.state_dict()) == tensor_digest(b.backbone.state_dict())
    assert not torch.equal(
        a.backbone.transformer.h[0].mlp.c_fc.weight, a.backbone.transformer.h[1].mlp.c_fc.weight
    )
    rng_a, rng_b = np.random.RandomState(7), np.random.RandomState(7)
    for _ in range(5):
        paired = sample_batch(config, world, rng_b)
        for key, value in sample_batch(config, world, rng_a).items():
            np.testing.assert_array_equal(value, paired[key])


def test_outer_gradient_stops_teacher_and_preserves_student_write_path():
    z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    teacher = torch.tensor([[4.0, 5.0], [6.0, 7.0]], requires_grad=True)
    student = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

    class LocalFeatures:
        trainable_backbone = True
        compute = {
            "estimated_dense_forward_matmul_flops": 0,
            "estimated_backward_matmul_flops_proxy": 0,
        }

        def empty_fast(self):
            return {0: torch.zeros(2, 2)}

        def __call__(self, *args, capture, **kwargs):
            capture["features"] = {0: (z, teacher, student)}

        def validate_fast(self, fast):
            pass

    rate = 0.03
    fast = session_write(
        LocalFeatures(),
        torch.zeros(2, 3, dtype=torch.long),
        None,
        {"teacher_window": 32, "student_window": 16, "learning_rate": rate},
    )
    fast[0].sum().backward()
    assert teacher.grad is None
    torch.testing.assert_close(
        z.grad, rate * (teacher.detach() - student.detach()).sum(-1, keepdim=True).expand_as(z)
    )
    torch.testing.assert_close(
        student.grad, -rate * z.detach().sum(-1, keepdim=True).expand_as(student)
    )


def test_fresh_process_training_audit_then_complete_edit_pipeline(tmp_path):
    config = fixture_config(tmp_path)
    prepare_scratch_data(config)
    name, out = "discoloop", tmp_path / "discoloop"
    out.mkdir()
    result = train_scratch(config, name, out, "cpu")
    assert result["counts"]["base_parameter_updates"] == 4
    assert result["counts"]["corrected_two_hop_training_examples"] == 0
    payload = torch.load(config["models"][name]["checkpoint"], weights_only=True)
    assert payload["initial_tensor_digest"] != tensor_digest(payload["model"])
    assert payload["optimizer"]["state"]
    with pytest.raises(ValueError, match="reload audit is required"):
        load_inputs(config, name, "cpu")
    config_path = tmp_path / "fixture.json"
    config_path.write_text(json.dumps(config))
    for stage in ("train-audit", "calibrate", "evaluate", "audit"):
        cli(stage, config_path, name, out)
    assert json.loads((out / "training/audit.json").read_text())["passed"]
    assert json.loads((out / "audit.json").read_text())["passed"]
    with pytest.raises(FileExistsError, match="already exists"):
        train_scratch(config, name, out, "cpu")
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="provenance"):
        load_inputs(changed, name, "cpu")


def test_invalid_budget_and_corrupt_split_are_rejected(tmp_path):
    config = fixture_config(tmp_path)
    prepare_scratch_data(config)
    changed = copy.deepcopy(config)
    changed["training"]["warmup_steps"] = 0
    with pytest.raises(ValueError, match="positive integers"):
        validate_scratch_inputs(changed)
    path = Path(config["data_dir"]) / "training-split.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="split hash"):
        validate_scratch_inputs(config)


@pytest.mark.skipif(
    os.environ.get("TTCD_TEST_CUDA") != "1", reason="Explicit isolated GPU test only"
)
def test_cuda_full_size_scratch_and_fresh_process_replay(tmp_path):
    config = fixture_config(tmp_path, full_size=True)
    prepare_scratch_data(config)
    assert validate_scratch_inputs(config).metadata["vocab_size"] == 290
    name = os.environ.get("TTCD_SCRATCH_TEST_MODEL", "discoloop")
    out = tmp_path / name
    out.mkdir()
    result = train_scratch(config, name, out, "cuda:0")
    assert result["step"] == 32
    assert audit_scratch_training(config, name, out, "cuda:0")["passed"]
    config_path = tmp_path / "cuda-fixture.json"
    config_path.write_text(json.dumps(config))
    for stage in ("train-audit", "calibrate", "evaluate", "audit"):
        cli(stage, config_path, name, out, device="cuda:0")
    torch.cuda.synchronize()
