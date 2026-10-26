"""The one-step overlay reuses an audited parent without relabelling it."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ttcd_architecture import fixture as architecture_fixture
from test_ttcd_verified_base import fixture as disco_fixture

from llm_memory_editability import ttcd_continual as continual
from llm_memory_editability import ttcd_one_step_training as experiment
from llm_memory_editability.grokking_reproduction import MergedBatchStream, model_digest
from llm_memory_editability.path_overlap import sha256


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def material(directory, name="discoloop"):
    parent = (
        disco_fixture(directory / "original")
        if name == "discoloop"
        else architecture_fixture(directory / "original", name)[0]
    )
    source = directory / "original-config.json"
    source.write_text(json.dumps(parent))
    original = experiment._base_engine(parent)
    common = directory / "original/common"
    common.mkdir()
    original.train(parent, "common", common, "cpu")
    original.audit_training(parent, "common", common, "cpu")
    original.common_gate(parent, common, "cpu")
    config = copy.deepcopy(parent)
    config["batch"] = experiment.BATCH
    config["writer"]["steps"] = 1
    config["models"] = {
        "stream": {
            **parent["models"]["stream"],
            "checkpoint": str(directory / "new-stream/training/parent.pt"),
        }
    }
    task = {
        "batch": experiment.BATCH,
        "architecture_name": name,
        "training_method": "stream",
        "parent_config": str(source),
        "parent_config_sha256": sha256(source),
        "parent_common_checkpoint_sha256": sha256(parent["common"]["checkpoint"]),
        "parent_common_gate_sha256": sha256(parent["common"]["gate_receipt"]),
        "training_config": config,
    }
    return task, original


@pytest.mark.parametrize("name", ["transformer", "loop", "discoloop"])
def test_reuses_complete_common_parent_then_trains_one_step_and_independently_reloads(
    tmp_path, name
):
    task, original = material(tmp_path, name)
    config = task["training_config"]
    before = copy.deepcopy(task)
    old_identity = original.identity(json.loads(Path(task["parent_config"]).read_text()))
    engine = experiment.build_engine(task)
    reader, _, _, _, parent = engine._parent_for_phase(config, "stream", "cpu")
    assert model_digest(reader.backbone) == parent["model_sha256"]
    assert parent["identity"] == old_identity
    assert parent["phase"] == "common"
    assert engine.identity(config)["parent"]["identity"] == old_identity
    directory = tmp_path / "new-stream"
    directory.mkdir()
    result = engine.train(config, "stream", directory, "cpu")
    assert result["counts"]["base_parameter_updates"] == config["training"]["steps"]
    assert result["counts"]["fast_state_updates"] == (
        config["training"]["steps"] * config["training"]["correction_batch_size"]
    )
    assert result["counts"]["corrected_two_hop_training_examples"] == 0
    initial = engine._load(directory / "training/checkpoint-0000000.pt")
    assert initial["model_sha256"] == parent["model_sha256"]
    assert experiment.state_digest(initial["optimizer"]) == experiment.state_digest(
        parent["optimizer"]
    )
    assert initial["data_rng"] == parent["data_rng"]
    torch.testing.assert_close(initial["torch_rng"], parent["torch_rng"], rtol=0, atol=0)
    with pytest.raises(FileNotFoundError):
        engine.load_reader(config, "stream", "cpu")
    assert engine.audit_training(config, "stream", directory, "cpu")["passed"]
    reloaded, _ = experiment.build_engine(task).load_reader(config, "stream", "cpu")
    assert model_digest(reloaded.backbone) == result["model_sha256"]
    assert all(not parameter.requires_grad for parameter in reloaded.parameters())
    assert task == before
    assert original.identity(json.loads(Path(task["parent_config"]).read_text())) == old_identity


def test_each_fact_writes_once_with_connected_stream_state_and_outer_gradients(
    tmp_path, monkeypatch
):
    task, _ = material(tmp_path)
    engine = experiment.build_engine(task)
    config = engine.config
    reader, world, _, _, _ = engine._parent_for_phase(config, "stream", "cpu")
    table, _ = engine.native_table(world, config["phi"])
    batch = engine.sample_step(
        config, world, np.random.RandomState(17), MergedBatchStream(len(table[0]), 18)
    )
    calls = []
    original_write = continual.session_write

    def observed(candidate, packets, fast, writer):
        calls.append((packets.detach().clone(), fast))
        return original_write(candidate, packets, fast, writer)

    monkeypatch.setattr(continual, "session_write", observed)
    packets = torch.as_tensor(batch["packets"])
    fast = engine.write_states(
        reader, packets, config["writer"], config["continual"]["train_length"], "stream"
    )
    length = config["continual"]["train_length"]
    assert len(calls) == length
    assert calls[0][1] is None and all(state is not None for _, state in calls[1:])
    grouped = packets.reshape(-1, length, packets.shape[-1])
    for position, (written, _) in enumerate(calls):
        torch.testing.assert_close(written, grouped[:, position], rtol=0, atol=0)
    assert all(state.grad_fn is not None for state in fast.values())
    loss = engine._row_loss(reader, batch["corrected_atoms"], world.metadata, fast)
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in reader.parameters()
    )


def test_one_step_microbatch_and_full_batch_losses_gradients_and_update_match(tmp_path):
    task, _ = material(tmp_path)
    engine = experiment.build_engine(task)
    config = engine.config
    reader, world, _, _, _ = engine._parent_for_phase(config, "stream", "cpu")
    whole = copy.deepcopy(reader)
    table, _ = engine.native_table(world, config["phi"])
    batch = engine.sample_step(
        config, world, np.random.RandomState(17), MergedBatchStream(len(table[0]), 18)
    )
    first = engine.backward_objective(reader, config, world, table, batch, "stream")
    second = engine.backward_objective(whole, config, world, table, batch, "stream", micro=False)
    np.testing.assert_allclose(first[0], second[0], rtol=1e-6, atol=1e-6)
    assert set(first[1]) == {"native", "reader", "fast_atomic", "fast_retention"}
    for key in first[1]:
        np.testing.assert_allclose(first[1][key], second[1][key], rtol=1e-6, atol=1e-6)
    for parameter, other in zip(reader.parameters(), whole.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-5)
    for candidate in (reader, whole):
        optimizer = engine.optimizer_for(candidate, config)
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), config["training"]["gradient_clip"])
        optimizer.step()
    for parameter, other in zip(reader.parameters(), whole.parameters(), strict=True):
        torch.testing.assert_close(parameter, other, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize(
    "change", ["steps", "method", "source", "checkpoint", "gate", "loss", "destination"]
)
def test_refuses_wrong_budget_source_parent_or_training_objective(tmp_path, change):
    task, _ = material(tmp_path)
    if change == "steps":
        task["training_config"]["writer"]["steps"] = 4
    elif change == "method":
        task["training_method"] = "merge"
    elif change == "source":
        task["parent_config_sha256"] = "wrong"
    elif change in ("checkpoint", "gate"):
        task[f"parent_common_{change}_sha256"] = "wrong"
    elif change == "loss":
        task["training_config"]["training"]["learning_rate"] *= 2
    else:
        task["training_config"]["models"]["stream"]["checkpoint"] = json.loads(
            Path(task["parent_config"]).read_text()
        )["common"]["checkpoint"]
    with pytest.raises(ValueError):
        experiment.build_engine(task)


def test_cannot_retrain_common_or_load_old_phase_and_engines_have_private_globals(tmp_path):
    task, _ = material(tmp_path)
    first, second = experiment.build_engine(task), experiment.build_engine(task)
    assert first.train.__globals__ is not second.train.__globals__
    for phase in ("common", "single"):
        with pytest.raises(ValueError, match="Only stream adaptation"):
            first.train(first.config, phase, tmp_path / phase, "cpu")
        with pytest.raises(ValueError, match="Only the new one-step stream"):
            first.load_reader(first.config, phase, "cpu")
    # A post-construction alteration is rejected rather than silently changing provenance.
    Path(task["parent_config"]).write_text("{}")
    with pytest.raises(ValueError, match="configuration bytes"):
        first.identity(first.config)


def test_zero_update_preflight_checks_real_parent_losses_states_and_all_gradients(tmp_path):
    task, _ = material(tmp_path)
    checkpoint = Path(task["training_config"]["common"]["checkpoint"])
    before = sha256(checkpoint)
    result = experiment.preflight(task, tmp_path / "preflight", "cpu")
    assert result["passed"] and result["all_parameter_gradients_exact"]
    assert result["all_fast_states_exact"] and result["all_four_losses_exact"]
    assert result["base_parameter_updates"] == result["optimizer_updates"] == 0
    assert result["formal_D_queries"] == 0
    assert result["writer_steps"] == 1
    assert result["backbone_unchanged"]
    assert result["speedup"]["pruned_calls"] > 0
    assert sha256(checkpoint) == before
    assert not Path(task["training_config"]["models"]["stream"]["checkpoint"]).exists()
