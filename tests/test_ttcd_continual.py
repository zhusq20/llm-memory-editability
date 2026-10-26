"""Paired continual TTCD contracts; all training here is engineering fixtures."""

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ttcd_branch import assert_state_equal
from test_ttcd_scratch import fixture_config

from llm_memory_editability.path_overlap import sha256
from llm_memory_editability.ttcd_continual import (
    load_reader,
    outer_objective,
    pack_rng,
    restore_data_rng,
    sample_batch,
    train,
    validate_config,
    validate_parent,
    write_states,
)
from llm_memory_editability.ttcd_continual_eval import build_panel
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest
from llm_memory_editability.ttcd_scratch import (
    new_scratch_reader,
    prepare_scratch_data,
    row_loss,
    train_scratch,
)
from llm_memory_editability.ttcd_scratch import (
    outer_objective as old_outer_objective,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_parent(directory, *, full_size=False, device="cpu"):
    """Real inherited AdamW/RNG checkpoint with an independently frozen source."""
    config = fixture_config(directory, full_size=full_size)
    config["models"] = {"discoloop": config["models"]["discoloop"]}
    config["runtime"]["cpu_threads"] = 1
    if full_size:
        config["cases"].update(calibration_cases=4, evaluation_cases=8)
        config["reading_gate"]["minimum_cases_per_group"] = 4
    else:
        config["training"].update(
            warmup_steps=4, adaptation_steps=2, batch_size=4, validation_every=2
        )
        config["cases"]["evaluation_cases"] = 2
    for spec in config["models"].values():
        spec["step"] = config["training"]["warmup_steps"] + config["training"]["adaptation_steps"]
    prepare_scratch_data(config)
    train_scratch(config, "discoloop", directory / "discoloop", device)
    step = config["training"]["warmup_steps"]
    checkpoint = directory / f"discoloop/training/checkpoint-{step:07d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = directory / "frozen-source"
    for filename, digest in payload["training_identity"]["code"].items():
        destination = source / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, destination)
        assert sha256(destination) == digest
    frozen_config = source / "configs/experiment.json"
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    frozen_config.write_text(json.dumps(config))
    manifest = directory / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_sha256": sha256(frozen_config),
                "files": {
                    **payload["training_identity"]["code"],
                    "configs/experiment.json": sha256(frozen_config),
                },
            }
        )
    )
    return {
        "config": config,
        "checkpoint": checkpoint,
        "source": source,
        "manifest": manifest,
    }


@pytest.fixture(scope="module")
def parent_material(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return make_parent(tmp_path_factory.mktemp("ttcd-continual-parent"))
    finally:
        torch.set_num_threads(previous)


def make_continual_config(parent, directory, *, full_size=False):
    config = copy.deepcopy(parent["config"])
    config.update(batch="ttcd-continual-engineering-fixture", engineering_fixture_only=True)
    spec = parent["config"]["models"]["discoloop"]
    config["models"] = {
        name: {**spec, "checkpoint": str(directory / name / "training/parent.pt")}
        for name in ("single", "stream")
    }
    config["parent"] = {
        "parent_model": "discoloop",
        "parent_batch": parent["config"]["batch"],
        "parent_source_dir": str(parent["source"]),
        "parent_manifest_path": str(parent["manifest"]),
        "parent_manifest_sha256": sha256(parent["manifest"]),
        "parent_config_sha256": sha256(parent["source"] / "configs/experiment.json"),
        "checkpoint_step": config["training"]["warmup_steps"],
        "steps": config["training"]["adaptation_steps"],
        "checkpoint_sha256": sha256(parent["checkpoint"]),
        "checkpoints": {name: str(parent["checkpoint"]) for name in config["models"]},
    }
    config["continual"] = {
        "train_length": 8 if full_size else 2,
        "eval_length": 16 if full_size else 4,
        "rotations": 1,
        "affected": 2,
        "controls": 2,
        "order_seed": 1729,
    }
    world = validate_config(config)
    directory.mkdir(parents=True, exist_ok=True)
    panel_path = directory / "frozen-panel.json"
    panel_path.write_text(json.dumps(build_panel(config, world)))
    config["continual"]["panel_path"] = str(panel_path)
    return config


@pytest.fixture
def config(parent_material, tmp_path):
    return make_continual_config(parent_material, tmp_path / "continual")


def cli(stage, config_path, name, out, *, device="cpu", timeout=180):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_continual.py"),
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
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((out / f"result-{stage}.json").read_text())


@pytest.mark.parametrize(("length", "batch_size"), [(2, 4), (8, 16)])
def test_sampler_pairs_all_labels_and_excludes_entire_stream(config, length, batch_size):
    config["continual"]["train_length"] = length
    config["training"]["batch_size"] = batch_size
    world = validate_config(config)
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    reserved = {c["atom"] for c in split["case_bank"]}
    left, right = np.random.RandomState(723), np.random.RandomState(723)
    for _ in range(20):
        a, b = sample_batch(config, world, left), sample_batch(config, world, right)
        for key, value in a.items():
            np.testing.assert_array_equal(value, b[key])
        ids = a["edit_ids"].reshape(-1, config["continual"]["train_length"])
        controls = a["retention_ids"].reshape(ids.shape)
        for written, retained in zip(ids, controls, strict=True):
            assert len(set(written)) == len(written)
            assert set(written).isdisjoint(retained)
        assert reserved.isdisjoint(a["edit_ids"])
        assert (a["corrected_atoms"][:, -1] != world.atoms[a["edit_ids"], -1]).all()
        assert {tuple(r) for r in a["old_chains"]}.issubset(
            {tuple(r) for r in world.chains[world.support(config["phi"])]}
        )
    restored = restore_data_rng(pack_rng(left))
    expected, actual = sample_batch(config, world, left), sample_batch(config, world, restored)
    for key, value in expected.items():
        np.testing.assert_array_equal(value, actual[key])


def test_K1_both_arms_match_original_objective_and_every_parameter_gradient(config):
    config = copy.deepcopy(config)
    config["continual"]["train_length"] = 1
    world = validate_config(config)
    batch = sample_batch(config, world, np.random.RandomState(94))
    original = new_scratch_reader(config, "single", world, "cpu")
    readers = [copy.deepcopy(original) for _ in range(2)]
    before = tensor_digest(original.backbone.state_dict())
    reference, losses = old_outer_objective(original, config, world, batch, adaptation=True)
    reference.backward()
    for arm, reader in zip(("single", "stream"), readers, strict=True):
        actual, parts = outer_objective(reader, config, world, batch, arm)
        actual.backward()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert parts == losses
        for (name, p), (other, expected) in zip(
            reader.named_parameters(), original.named_parameters(), strict=True
        ):
            assert name == other and p.grad is not None and expected.grad is not None
            torch.testing.assert_close(p.grad, expected.grad, rtol=1e-6, atol=1e-9)
        assert tensor_digest(reader.backbone.state_dict()) == before


def test_stream_writer_matches_per_stream_inference_and_isolates_streams(config):
    world = validate_config(config)
    reader = new_scratch_reader(config, "stream", world, "cpu")
    batch = sample_batch(config, world, np.random.RandomState(27))
    packets = torch.as_tensor(batch["packets"])
    before = tensor_digest(reader.backbone.state_dict())
    length = config["continual"]["train_length"]
    states = write_states(reader, packets, config["writer"], length, "stream")
    frozen = TTCDReader(copy.deepcopy(reader.backbone)).eval()
    for start in range(0, len(packets), length):
        fast = frozen.empty_fast()
        for index in range(start, start + length):
            for _ in range(config["writer"]["steps"]):
                fast, _ = frozen.write(
                    packets[index : index + 1],
                    fast,
                    **{
                        k: config["writer"][k]
                        for k in ("teacher_window", "student_window", "learning_rate")
                    },
                )
        for layer, expected in fast.items():
            for i in range(start, start + length):
                torch.testing.assert_close(states[layer][i], expected, atol=1e-9, rtol=1e-5)
    altered = packets.clone()
    eo = world.eo
    altered[0, 2] = eo + (int(altered[0, 2]) - eo + 1) % world.entities
    changed = write_states(reader, altered, config["writer"], length, "stream")
    for layer in states:
        torch.testing.assert_close(changed[layer][length:], states[layer][length:], rtol=0, atol=0)
    assert any(not torch.equal(changed[k][:length], states[k][:length]) for k in states)
    assert tensor_digest(reader.backbone.state_dict()) == before


def test_first_fact_fast_state_receives_final_future_read_gradient(config):
    world = validate_config(config)
    reader = new_scratch_reader(config, "stream", world, "cpu")
    batch = sample_batch(config, world, np.random.RandomState(91))
    packets = torch.as_tensor(batch["packets"])
    first = {}

    def observe(position, state):
        if position == 0:
            first.update(state)
            for value in state.values():
                value.retain_grad()

    final = write_states(
        reader,
        packets,
        config["writer"],
        config["continual"]["train_length"],
        "stream",
        observe=observe,
    )
    # Only later facts are read, so this cannot be a direct loss on the first write.
    later = np.arange(
        config["continual"]["train_length"] - 1, len(packets), config["continual"]["train_length"]
    )
    loss = row_loss(
        reader,
        batch["corrected_atoms"][later],
        world.metadata,
        fast={k: value[later] for k, value in final.items()},
    )
    loss.backward()
    for value in first.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.square().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())


def test_parent_identity_budget_recipe_and_optimizer_are_required(config, tmp_path):
    parent, provenance, _ = validate_parent(config, "single")
    assert provenance["step"] == 4
    for update, message in (
        ({"checkpoint_step": 3}, "budget"),
        ({"parent_manifest_sha256": "0" * 64}, "manifest"),
    ):
        changed = copy.deepcopy(config)
        changed["parent"].update(update)
        with pytest.raises(ValueError, match=message):
            validate_parent(changed, "single")
    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="recipe"):
        validate_parent(changed, "single")
    for field, value, message in (
        ("optimizer", {"state": {}, "param_groups": []}, "AdamW"),
        ("counts", {**parent["counts"], "fast_state_updates": 1}, "pre-write"),
    ):
        checkpoint = tmp_path / f"broken-{field}.pt"
        torch.save({**parent, field: value}, checkpoint)
        changed = copy.deepcopy(config)
        changed["parent"]["checkpoints"]["single"] = str(checkpoint)
        changed["parent"]["checkpoint_sha256"] = sha256(checkpoint)
        with pytest.raises(ValueError, match=message):
            validate_parent(changed, "single")


def check_pipeline(config, parent_material, config_path, *, device="cpu", timeout=180):
    original = torch.load(parent_material["checkpoint"], map_location="cpu", weights_only=True)
    preserved = {
        str(path): sha256(path)
        for directory in (parent_material["checkpoint"].parents[1], parent_material["source"])
        for path in directory.rglob("*")
        if path.is_file()
    }
    preserved[str(parent_material["manifest"])] = sha256(parent_material["manifest"])
    count = config["parent"]["steps"]
    warmup = config["parent"]["checkpoint_step"]
    n = config["training"]["batch_size"]
    finals = []
    for arm in ("single", "stream"):
        out = Path(config["models"][arm]["checkpoint"]).parents[1]
        cli("preflight", config_path, arm, out, device=device, timeout=timeout)
        summary = cli("train", config_path, arm, out, device=device, timeout=timeout)
        fork = torch.load(
            out / f"training/checkpoint-{warmup:07d}.pt", map_location="cpu", weights_only=True
        )
        for key in ("model", "optimizer", "data_rng", "torch_rng", "cuda_rng"):
            assert_state_equal(fork[key], original[key])
        assert fork["parent"]["inherited_counts"] == original["counts"]
        assert summary["step"] == warmup + count
        assert summary["additional_counts"] == {
            "base_parameter_updates": count,
            "old_atomic_examples": n * count,
            "old_chain_examples": n * count,
            "correction_examples": n * count,
            "fast_state_updates": n * count * config["writer"]["steps"],
            "fast_atomic_examples": n * count,
            "fast_retention_examples": n * count,
            "supervised_target_tokens": n * count * 10,
            "corrected_two_hop_training_examples": 0,
        }
        records = [
            json.loads(line) for line in (out / "training/curves.jsonl").read_text().splitlines()
        ]
        assert [r["step"] for r in records] == list(range(warmup + 1, warmup + count + 1))
        assert all(set(r["losses"]) == set(config["training"]["loss_weights"]) for r in records)
        with pytest.raises((FileNotFoundError, ValueError)):
            load_reader(config, arm, device)
        for stage in ("train-audit", "calibrate", "evaluate", "audit"):
            result = cli(stage, config_path, arm, out, device=device, timeout=timeout)
            if stage != "evaluate":
                assert result["passed"]
        reader, _ = load_reader(config, arm, device)
        assert not any(p.requires_grad for p in reader.parameters())
        final = torch.load(
            config["models"][arm]["checkpoint"], map_location="cpu", weights_only=True
        )
        assert all(torch.isfinite(value).all() for value in final["model"].values())
        assert tensor_digest(final["model"]) != tensor_digest(original["model"])
        assert json.loads((out / "audit.json").read_text())["write_states_exact"]
        finals.append(final)
    assert finals[0]["trace_sha256"] == finals[1]["trace_sha256"]
    assert finals[0]["data_rng"] == finals[1]["data_rng"]
    assert finals[0]["additional_counts"] == finals[1]["additional_counts"]
    assert {path: sha256(path) for path in preserved} == preserved


def test_exact_parent_restore_paired_budget_and_all_fresh_process_stages(
    config, parent_material, tmp_path
):
    path = tmp_path / "continual-fixture.json"
    path.write_text(json.dumps(config))
    check_pipeline(config, parent_material, path)
    out = Path(config["models"]["single"]["checkpoint"]).parents[1]
    with pytest.raises(FileExistsError):
        train(config, "single", out, "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("TTCD_CONTINUAL_TEST_CUDA") != "1",
    reason="Full-size CUDA engineering fixture requires CUDA and TTCD_CONTINUAL_TEST_CUDA=1",
)
def test_full_size_cuda_paired_16_step_training_and_independent_complete_pipeline(tmp_path):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = make_parent(tmp_path / "parent", full_size=True, device="cuda:0")
    config = make_continual_config(parent, tmp_path / "continual", full_size=True)
    assert config["world"]["entities"] == 256 and config["world"]["relations"] == 32
    assert config["model"]["width"] == 128
    assert config["model"]["unique_layers"] == config["model"]["repeats"] == 2
    assert config["training"]["batch_size"] == 128
    assert config["training"]["warmup_steps"] == config["training"]["adaptation_steps"] == 16
    assert config["continual"]["train_length"] == 8
    assert config["continual"]["eval_length"] == 16
    path = tmp_path / "cuda-continual-fixture.json"
    path.write_text(json.dumps(config))
    check_pipeline(config, parent, path, device="cuda:0", timeout=600)
    torch.cuda.synchronize()
