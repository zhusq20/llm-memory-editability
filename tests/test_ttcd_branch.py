"""Engineering fixtures for an exact warmup branch; these are not science runs."""

import copy
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ttcd_scratch import fixture_config
from torch.nn import functional as F

from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.path_overlap import sha256
from llm_memory_editability.ttcd_editing import tensor_digest
from llm_memory_editability.ttcd_scratch import (
    new_scratch_reader,
    prepare_scratch_data,
    sample_batch,
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


@pytest.fixture(scope="module")
def parent_material(tmp_path_factory):
    """Create real optimizer/RNG checkpoints and a separately frozen parent source."""
    directory = tmp_path_factory.mktemp("ttcd-branch-parent")
    config = fixture_config(directory)
    config["training"].update(warmup_steps=4, adaptation_steps=2, validation_every=2)
    for spec in config["models"].values():
        spec["step"] = 6
    prepare_scratch_data(config)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in config["models"]:
            train_scratch(config, name, directory / name, "cpu")
    finally:
        torch.set_num_threads(previous)
    checkpoints = {
        name: directory / name / "training/checkpoint-0000004.pt" for name in config["models"]
    }
    payload = torch.load(checkpoints["loop"], weights_only=True)
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
        "source": source,
        "manifest": manifest,
        "checkpoints": checkpoints,
    }


@pytest.fixture
def branch_config(tmp_path, parent_material):
    parent = parent_material
    config = copy.deepcopy(parent["config"])
    config.update(
        batch="ttcd-second-hop-reader-control-v1",
        authorization="CPU engineering fixture only",
    )
    for name, spec in config["models"].items():
        spec["checkpoint"] = str(tmp_path / name / "training/parent.pt")
    config["branch"] = {
        "parent_batch": parent["config"]["batch"],
        "parent_source_dir": str(parent["source"]),
        "parent_manifest_path": str(parent["manifest"]),
        "parent_manifest_sha256": sha256(parent["manifest"]),
        "parent_config_sha256": sha256(parent["source"] / "configs/experiment.json"),
        "checkpoint_step": 4,
        "steps": 2,
        "objective": "old_atoms_old_chains_context_reader_only",
        "checkpoints": {k: str(v) for k, v in parent["checkpoints"].items()},
    }
    return config


def data_rng(state):
    rng = np.random.RandomState()
    rng.set_state(
        (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )
    return rng


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


def ordinary_continuation(config, name, checkpoint):
    """Independent loop: three ordinary losses, inherited AdamW/RNG/global LR."""
    world = validate_scratch_inputs(config)
    reader = new_scratch_reader(config, name, world, "cpu")
    reader.backbone.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    torch.set_rng_state(checkpoint["torch_rng"])
    rng = data_rng(checkpoint["data_rng"])
    settings = config["training"]
    total = settings["warmup_steps"] + settings["adaptation_steps"]
    losses = []

    def row_ce(rows):
        tokens, positions, labels = [torch.as_tensor(a) for a in encode_rows(rows, world.metadata)]
        logits = reader(tokens, positions)
        return F.cross_entropy(logits.flatten(0, 1), labels.flatten())

    for step in range(checkpoint["step"] + 1, total + 1):
        fraction = (step - 1) / max(1, total - 1)
        rate = settings["minimum_learning_rate"] + 0.5 * (
            settings["learning_rate"] - settings["minimum_learning_rate"]
        ) * (1 + math.cos(math.pi * fraction))
        rate *= min(1, step / max(1, settings["lr_warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        batch = sample_batch(config, world, rng)
        parts = {"atoms": row_ce(batch["old_atoms"]), "chains": row_ce(batch["old_chains"])}
        packets = torch.as_tensor(batch["packets"])
        new = torch.as_tensor(batch["corrected_atoms"][:, 2] + world.eo)
        tokens = torch.cat((packets, new[:, None]), dim=1)
        positions = torch.tensor([packets.shape[1] - 1, packets.shape[1]])
        positions = positions[None].expand(len(packets), -1)
        labels = torch.stack((new, torch.full_like(new, world.metadata["end_marker"])), dim=1)
        logits = reader(tokens, positions, window=config["writer"]["teacher_window"])
        parts["reader"] = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss = sum(settings["loss_weights"][key] * value for key, value in parts.items())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            reader.parameters(), settings["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        losses.append({key: float(value.detach()) for key, value in parts.items()})
    return reader, optimizer, rng, losses


@pytest.mark.parametrize("name", ["discoloop", "loop"])
def test_branch_exactly_restores_and_continues_ordinary_training(
    branch_config, parent_material, name
):
    from llm_memory_editability.ttcd_branch import train_branch, validate_branch_parent

    config = branch_config
    parent_path = parent_material["checkpoints"][name]
    parent_hash = sha256(parent_path)
    manifest_hash = sha256(parent_material["manifest"])
    source_hashes = {
        str(path): sha256(path) for path in parent_material["source"].rglob("*") if path.is_file()
    }
    parent, provenance = validate_branch_parent(config, name, parent_path)
    assert parent["step"] == 4
    assert provenance["checkpoint_step"] == 4
    assert provenance["checkpoint_sha256"] == parent_hash
    assert provenance["new_steps"] == 2
    reference, optimizer, rng, losses = ordinary_continuation(config, name, parent)
    reference_rng = torch.get_rng_state().clone()
    out = Path(config["models"][name]["checkpoint"]).parents[1]
    summary = train_branch(config, name, out, "cpu")
    child = torch.load(config["models"][name]["checkpoint"], weights_only=True)
    assert child["step"] == summary["step"] == 6
    assert child["branch"]
    assert_state_equal(child["model"], reference.backbone.state_dict())
    assert_state_equal(child["optimizer"], optimizer.state_dict())
    assert torch.equal(child["torch_rng"], reference_rng)
    actual_next = sample_batch(config, validate_scratch_inputs(config), data_rng(child["data_rng"]))
    expected_next = sample_batch(config, validate_scratch_inputs(config), rng)
    for key in actual_next:
        np.testing.assert_array_equal(actual_next[key], expected_next[key])
    fork = torch.load(out / "training/checkpoint-0000004.pt", weights_only=True)
    assert_state_equal(fork["model"], parent["model"])
    assert_state_equal(fork["optimizer"], parent["optimizer"])
    assert_state_equal(fork["data_rng"], parent["data_rng"])
    assert torch.equal(fork["torch_rng"], parent["torch_rng"])
    assert fork["training_identity"] != parent["training_identity"]
    assert tensor_digest(child["model"]) != tensor_digest(parent["model"])
    expected_additional = {
        "base_parameter_updates": 2,
        "old_atomic_examples": 4,
        "old_chain_examples": 4,
        "correction_examples": 4,
        "fast_state_updates": 0,
        "adaptation_episodes": 0,
        "supervised_target_tokens": 24,
        "corrected_two_hop_training_examples": 0,
    }
    assert summary["additional_counts"] == expected_additional
    assert summary["inherited_cost"] == {
        "counts": parent["counts"],
        "training_seconds": parent["training_seconds"],
        "wall_seconds": parent["wall_seconds"],
        "training_compute": parent["training_compute"],
        "compute_including_validation": parent["compute"],
    }
    assert (
        child["counts"]
        == summary["counts"]
        == {key: parent["counts"][key] + value for key, value in expected_additional.items()}
    )
    assert summary["branch_training_seconds"] > 0
    assert summary["branch_wall_seconds"] >= summary["branch_training_seconds"]
    assert summary["training_seconds"] == pytest.approx(
        parent["training_seconds"] + summary["branch_training_seconds"]
    )
    assert summary["training_compute"] == {
        key: parent["training_compute"][key] + value
        for key, value in summary["branch_training_compute"].items()
    }
    assert summary["branch_training_compute"] == reference.compute
    assert sha256(child["branch"]["preserved_checkpoint"]) == parent_hash
    records = [
        json.loads(line) for line in (out / "training/curves.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [5, 6]
    for record, expected_loss in zip(records, losses, strict=True):
        assert record["losses"] == expected_loss
        assert set(record["losses"]) == {"atoms", "chains", "reader"}
        assert record["counts"]["fast_state_updates"] == 0
    assert sha256(parent_path) == parent_hash
    assert sha256(parent_material["manifest"]) == manifest_hash
    assert {path: sha256(path) for path in source_hashes} == source_hashes
    with pytest.raises(FileExistsError):
        train_branch(config, name, out, "cpu")


@pytest.mark.parametrize(
    "change",
    [
        "architecture",
        "training",
        "checkpoint_step",
        "steps",
        "objective",
        "config_hash",
        "manifest_hash",
    ],
)
def test_branch_rejects_incompatible_or_unpinned_contract(branch_config, parent_material, change):
    from llm_memory_editability.ttcd_branch import validate_branch_parent

    name = "loop"
    if change == "architecture":
        branch_config["models"][name]["discoloop"] = True
    elif change == "training":
        branch_config["training"]["learning_rate"] *= 2
    elif change == "checkpoint_step":
        branch_config["branch"]["checkpoint_step"] = 2
    elif change == "steps":
        branch_config["branch"]["steps"] = 3
    elif change == "objective":
        branch_config["branch"]["objective"] = "fast_atomic"
    elif change == "config_hash":
        branch_config["branch"]["parent_config_sha256"] = "0" * 64
    elif change == "manifest_hash":
        branch_config["branch"]["parent_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_branch_parent(branch_config, name, parent_material["checkpoints"][name])


@pytest.mark.parametrize("corruption", ["step", "model", "optimizer", "rng", "input"])
def test_branch_rejects_internally_corrupt_checkpoint(
    monkeypatch, branch_config, parent_material, corruption
):
    from llm_memory_editability.ttcd_branch import validate_branch_parent

    name = "loop"
    payload = torch.load(parent_material["checkpoints"][name], weights_only=True)
    if corruption == "step":
        payload["step"] = 3
    elif corruption == "model":
        next(iter(payload["model"].values())).flatten()[0] += 1
    elif corruption == "optimizer":
        payload["optimizer"]["state"] = {}
    elif corruption == "rng":
        payload["data_rng"]["position"] = -1
    elif corruption == "input":
        key = next(iter(payload["training_identity"]["inputs"]))
        payload["training_identity"]["inputs"][key] = "0" * 64
    # Keep the original file/path and snapshot immutable. Inject the corrupt
    # deserialized payload so this reaches semantic checks beyond the path guard.
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises((ValueError, RuntimeError)):
        validate_branch_parent(branch_config, name, parent_material["checkpoints"][name])


def test_branch_rejects_changed_parent_source(tmp_path, branch_config, parent_material):
    from llm_memory_editability.ttcd_branch import validate_branch_parent

    source = tmp_path / "tampered-source"
    shutil.copytree(parent_material["source"], source)
    module = source / "src/llm_memory_editability/ttcd_scratch.py"
    module.write_text(module.read_text() + "\n# changed after freezing\n")
    branch_config["branch"]["parent_source_dir"] = str(source)
    with pytest.raises(ValueError):
        validate_branch_parent(branch_config, "loop", parent_material["checkpoints"]["loop"])


def test_branch_final_checkpoint_has_fresh_process_training_audit(branch_config, tmp_path):
    from llm_memory_editability.ttcd_branch import train_branch

    name = "loop"
    out = Path(branch_config["models"][name]["checkpoint"]).parents[1]
    train_branch(branch_config, name, out, "cpu")
    config_path = tmp_path / "branch-fixture.json"
    config_path.write_text(json.dumps(branch_config))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_second_hop.py"),
            "train-audit",
            "--config",
            str(config_path),
            "--model",
            name,
            "--out",
            str(out),
            "--device",
            "cpu",
            "--tracking",
            "disabled",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((out / "training/audit.json").read_text())
    assert receipt["passed"]
    assert receipt["step"] == 6
    assert receipt["parent_sha256"] == sha256(branch_config["models"][name]["checkpoint"])


@pytest.mark.skipif(
    os.environ.get("TTCD_BRANCH_TEST_CUDA") != "1",
    reason="Full-size CUDA engineering fixture requires TTCD_BRANCH_TEST_CUDA=1",
)
def test_full_size_cuda_branch_restore_and_fresh_process_audit(tmp_path, monkeypatch):
    """Opt-in engineering fixture; never reads or starts scientific training."""
    from llm_memory_editability.ttcd_branch import train_branch
    from llm_memory_editability.ttcd_scratch import (
        audit_scratch_training,
        outer_objective,
    )

    name = os.environ.get("TTCD_BRANCH_TEST_MODEL", "discoloop")
    assert name in {"discoloop", "loop"}
    assert torch.cuda.is_available(), "Opt-in CUDA fixture requires an actual CUDA device"
    device = "cuda:0"
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    config = fixture_config(tmp_path, full_size=True)
    assert config["world"]["entities"] == 256
    assert config["world"]["relations"] == 32
    assert config["model"]["width"] == 128
    assert config["model"]["unique_layers"] == config["model"]["repeats"] == 2
    assert config["training"]["batch_size"] == 128
    assert config["training"]["warmup_steps"] == config["training"]["adaptation_steps"] == 16
    world_metadata = prepare_scratch_data(config)
    assert world_metadata["vocab_size"] == 290
    parent_out = tmp_path / name
    train_scratch(config, name, parent_out, device)
    parent_path = parent_out / "training/checkpoint-0000016.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)

    source = tmp_path / "frozen-parent-source"
    for filename, digest in parent["training_identity"]["code"].items():
        destination = source / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, destination)
        assert sha256(destination) == digest
    frozen_config = source / "configs/experiment.json"
    frozen_config.parent.mkdir(parents=True, exist_ok=True)
    frozen_config.write_text(json.dumps(config))
    manifest = tmp_path / "parent-source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "config_sha256": sha256(frozen_config),
                "files": {
                    **parent["training_identity"]["code"],
                    "configs/experiment.json": sha256(frozen_config),
                },
            }
        )
    )
    preserved = {
        str(path): sha256(path)
        for directory in (parent_out, source)
        for path in directory.rglob("*")
        if path.is_file()
    }
    preserved[str(manifest)] = sha256(manifest)
    branch = copy.deepcopy(config)
    branch.update(
        batch="ttcd-second-hop-reader-control-v1",
        authorization="Full-size short CUDA engineering fixture only; tracking disabled",
    )
    for model, spec in branch["models"].items():
        spec["checkpoint"] = str(tmp_path / "reader-control" / model / "training/parent.pt")
    branch["branch"] = {
        "parent_batch": config["batch"],
        "parent_source_dir": str(source),
        "parent_manifest_path": str(manifest),
        "parent_manifest_sha256": sha256(manifest),
        "parent_config_sha256": sha256(frozen_config),
        "checkpoint_step": 16,
        "steps": 16,
        "objective": "old_atoms_old_chains_context_reader_only",
        "checkpoints": {
            model: str(Path(spec["checkpoint"]).with_name("checkpoint-0000016.pt"))
            for model, spec in config["models"].items()
        },
    }
    out = Path(branch["models"][name]["checkpoint"]).parents[1]
    summary = train_branch(branch, name, out, device)
    fork = torch.load(out / "training/checkpoint-0000016.pt", map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "data_rng", "counts"):
        assert_state_equal(fork[key], parent[key])
    assert len(fork["cuda_rng"]) == torch.cuda.device_count()
    assert fork["cuda_rng"]
    assert summary["step"] == 32
    assert summary["additional_counts"] == {
        "base_parameter_updates": 16,
        "old_atomic_examples": 2048,
        "old_chain_examples": 2048,
        "correction_examples": 2048,
        "fast_state_updates": 0,
        "adaptation_episodes": 0,
        "supervised_target_tokens": 12288,
        "corrected_two_hop_training_examples": 0,
    }
    records = [
        json.loads(line) for line in (out / "training/curves.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == list(range(17, 33))
    assert all(set(record["losses"]) == {"atoms", "chains", "reader"} for record in records)
    final = torch.load(branch["models"][name]["checkpoint"], map_location="cpu", weights_only=True)
    assert all(torch.isfinite(tensor).all() for tensor in final["model"].values())
    assert tensor_digest(final["model"]) != tensor_digest(parent["model"])
    world = validate_scratch_inputs(branch)
    reader = new_scratch_reader(branch, name, world, device)
    reader.backbone.load_state_dict(final["model"])
    batch = sample_batch(branch, world, data_rng(final["data_rng"]))
    loss, losses = outer_objective(reader, branch, world, batch, adaptation=False)
    assert torch.isfinite(loss)
    assert set(losses) == {"atoms", "chains", "reader"}
    loss.backward()
    torch.cuda.synchronize(device)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters())
    assert sum(float(p.grad.square().sum()) for p in reader.parameters()) > 0
    del reader
    assert audit_scratch_training(branch, name, out, device)["passed"]
    config_path = tmp_path / "cuda-branch-fixture.json"
    config_path.write_text(json.dumps(branch))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ttcd_second_hop.py"),
            "train-audit",
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
    receipt = json.loads((out / "training/audit.json").read_text())
    assert receipt["passed"] and receipt["step"] == 32
    assert receipt["parent_sha256"] == sha256(branch["models"][name]["checkpoint"])
    assert {path: sha256(path) for path in preserved} == preserved
