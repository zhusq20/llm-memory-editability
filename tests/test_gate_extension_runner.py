"""Contracts that distinguish new data/NoPE behavior from the native runner."""

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.dual_gate import DualGateGPT
from llm_memory_editability.grokking_reproduction import model_digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_gate_extension")


@pytest.fixture(scope="module", autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return {
        "batch": "gate-extension-contracts",
        "family": "depth",
        "position_encoding": "none",
        "input_root": "unused",
        "dropout_seed": 19,
        "model": {
            "n_embd": 8,
            "n_head": 2,
            "n_positions": 6,
            "n_ctx": 6,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "embd_pdrop": 0.1,
        },
        "training": {
            "initialization": 13,
            "stream_seed": 17,
            "weight_decay": 0.1,
            "lr": 0.001,
            "warmup": 2,
            "batch_size": 8,
            "steps": 300000,
        },
        "arms": {
            mode: {
                "architecture": "discoloop",
                "unique_layers": 1,
                "repeats": 4,
                "update_mode": mode,
                "initial_update": 0.95,
            }
            for mode in ("full", "fixed", "token")
        },
        "evaluation_nodes": [0, 1000, 300000],
        "inference_repeats": [1, 2, 3, 4, 6, 8],
    }


@pytest.fixture
def metadata():
    return {
        "kind": "depth",
        "vocab_size": 17,
        "pad_token": 0,
        "end_marker": 2,
        "entity_offset": 3,
        "relation_offset": 11,
        "sequence_length": 6,
        "primary_score": "answer_accuracy",
    }


def test_only_registered_position_policy_and_u95_can_execute(config):
    runner.validate_config(config)
    for change in (
        {"position_encoding": "learned"},
        {"family": "wang"},
        {"evaluation_nodes": [0, 300000, 1000]},
        {"inference_repeats": [1, 4, 4]},
    ):
        with pytest.raises(ValueError):
            runner.validate_config({**config, **change})
    invalid = copy.deepcopy(config)
    invalid["arms"]["token"]["initial_update"] = 0.90
    with pytest.raises(ValueError):
        runner.validate_config(invalid)
    invalid = copy.deepcopy(config)
    invalid["arms"]["token"]["repeats"] = 8
    with pytest.raises(ValueError, match="same block and loop"):
        runner.validate_config(invalid)
    del invalid["arms"]["token"]
    with pytest.raises(ValueError, match="complete full/fixed/token"):
        runner.validate_config(invalid)


def test_mounted_data_must_match_frozen_family_and_identity(config, monkeypatch):
    config["data"] = {"kind": "depth_extrapolation", "world_seed": 123}
    config["input_origin"] = {"world_sha256": "expected"}
    metadata = {"kind": "depth_extrapolation", "world_seed": 123, "world_sha256": "expected"}
    monkeypatch.setattr(runner, "load", lambda _: ({}, {}, metadata))
    assert runner.verify_input(config) is metadata
    metadata["world_sha256"] = "different"
    with pytest.raises(ValueError, match="input identity"):
        runner.verify_input(config)
    metadata["world_sha256"] = "expected"
    metadata["world_seed"] = 456
    with pytest.raises(ValueError, match="specification"):
        runner.verify_input(config)
    metadata["kind"] = "wang"
    with pytest.raises(ValueError, match="Dataset kind"):
        runner.verify_input(config)


@pytest.mark.parametrize("position", ["none", "learned"])
def test_paired_initialization_and_rng_include_exact_original_p(config, metadata, position):
    config.update(
        position_encoding=position, family="depth" if position == "none" else "confirmation"
    )
    models, rngs = {}, []
    for name in config["arms"]:
        model, _, arm, initial = runner.initialize(config, name, metadata, torch.device("cpu"))
        assert model_digest(model) == initial
        assert arm["position_encoding"] == position
        models[name] = model
        rngs.append(torch.get_rng_state())
    assert all(torch.equal(rngs[0], rng) for rng in rngs)
    full = models["full"]
    for key, value in full.state_dict().items():
        assert torch.equal(value, models["fixed"].state_dict()[key])
        assert torch.equal(value, models["token"].state_dict()[key])
    assert model_digest(models["fixed"]) == model_digest(models["token"])
    reference = DualGateGPT(full.config, 4, gate_variant="paper", discoloop=True)
    reference.load_state_dict(full.state_dict(), strict=True)
    tokens = torch.tensor([[3, 11, 12, 4, 0, 0], [5, 11, 11, 6, 0, 0]])
    positions = torch.tensor([[2, 3], [2, 3]])
    full.eval()
    reference.eval()
    assert torch.equal(full(tokens, positions), reference(tokens, positions))


def test_nope_is_exact_zero_addition_and_excluded_from_optimizer_after_update(config, metadata):
    model, spec, _, _ = runner.initialize(config, "token", metadata, torch.device("cpu"))
    model.eval()
    tokens = torch.tensor([[3, 11, 12, 4, 0, 0], [5, 11, 11, 6, 0, 0]])
    positions = torch.tensor([[2, 3], [2, 3]])
    assert torch.equal(model._embedded(tokens), model.transformer.wte(tokens))
    registered = runner.optimizer_for(model, spec["lr"], spec["weight_decay"])
    ids = [id(p) for group in registered.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.transformer.wpe.weight) not in ids
    gate_ids = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
    assert all(
        group["weight_decay"] == 0
        for group in registered.param_groups
        if any(id(p) in gate_ids for p in group["params"])
    )
    # CPU arithmetic checks the freeze contract; actual fused captured CUDA is
    # checked separately by the execution preflight.
    optimizer = torch.optim.AdamW(
        [
            {"params": group["params"], "weight_decay": group["weight_decay"]}
            for group in registered.param_groups
        ],
        lr=spec["lr"],
    )
    before = model_digest(model)
    F.cross_entropy(model(tokens, positions).flatten(0, 1), torch.tensor([4, 2, 6, 2])).backward()
    assert model.transformer.wpe.weight.grad is None
    optimizer.step()
    assert torch.count_nonzero(model.transformer.wpe.weight) == 0
    assert model_digest(model) != before


def test_fixed_and_learned_initial_outputs_and_shared_gradients_match(config, metadata):
    tokens = torch.tensor([[3, 11, 12, 4, 0, 0], [5, 11, 11, 6, 0, 0]])
    positions = torch.tensor([[2, 3], [2, 3]])
    output, gradients = [], []
    for name in ("fixed", "token"):
        model, _, _, _ = runner.initialize(config, name, metadata, torch.device("cpu"))
        torch.manual_seed(config["dropout_seed"])
        logits = model(tokens, positions)
        F.cross_entropy(logits.flatten(0, 1), torch.tensor([4, 2, 6, 2])).backward()
        output.append(logits.detach())
        gradients.append({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    assert torch.equal(*output)
    assert set(gradients[0]).issubset(gradients[1])
    assert all(torch.equal(gradient, gradients[1][name]) for name, gradient in gradients[0].items())
    assert "transformer.wpe.weight" not in gradients[0]
    assert "gate_update_weight" in gradients[1]


def test_wang_full_atomic_table_has_correct_global_order_without_mutating_original(monkeypatch):
    atoms = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 0]])
    panels = {"atomic_id": atoms[[2]], "atomic_ood": atoms[[1]]}
    metadata = {"kind": "wang", "entity_offset": 3}
    monkeypatch.setattr(runner.data, "load", lambda _: ({"atoms": atoms}, panels, metadata))
    _, observed, _ = runner.load("unused")
    assert "atomic_all" not in panels
    assert np.array_equal(observed["atomic_all"], atoms)
    prediction = {
        "atomic_id": {"answer": np.array([0])},
        "atomic_ood": {"answer": np.array([0])},
        "atomic_all": {"answer": np.array([4, 0, 3])},
    }
    assert runner.atomic_correctness(prediction, observed, metadata).tolist() == [True, False, True]


def test_conditioned_exact_score_keeps_answer_and_eos_distinct(monkeypatch):
    rows = np.array([[0, 0, 0, 2], [1, 0, 0, 3]])
    monkeypatch.setattr(
        runner.data,
        "path_details",
        lambda *args: (np.array([[0, 1, 2], [1, 2, 3]]), np.array([[0, 1], [1, 2]])),
    )
    prediction = {"answer": np.array([5, 6]), "stop": np.array([1, 2])}
    result = runner.premise_metrics(
        rows,
        prediction,
        np.array([True, True, False]),
        {},
        {
            "entity_offset": 3,
            "end_marker": 2,
        },
    )
    assert result["premise_correct_n"] == 1
    assert result["premise_coverage"] == 0.5
    assert result["answer_accuracy_given_premises"] == 1
    assert result["accuracy_given_premises"] == 0


def test_registered_sweep_restores_horizon_rng_mode_and_does_not_pick_best(
    config, metadata, monkeypatch
):
    model, _, _, _ = runner.initialize(config, "token", metadata, torch.device("cpu"))
    calls = []
    panels = {"atomic_id": np.array([[0, 0, 1]]), "atomic_ood": np.array([[1, 0, 0]])}

    def measure(current, world, pools, meta, device, batch_size):
        calls.append(current.repeats)
        return (
            {name: {} for name in pools},
            {name: {"answer": rows[:, -1] + meta["entity_offset"]} for name, rows in pools.items()},
        )

    monkeypatch.setattr(runner, "measure_panels", measure)
    before = model_digest(model), torch.get_rng_state()
    metrics, _ = runner.sweep(model, {}, panels, metadata, torch.device("cpu"), 8, [1, 4, 8])
    assert list(metrics) == ["1", "4", "8"]
    assert calls == [1, 4, 8]
    assert model.repeats == 4 and model.training
    assert model_digest(model) == before[0]
    assert torch.equal(torch.get_rng_state(), before[1])


def test_scoped_adapter_restores_nested_mutations_even_on_failure(config, monkeypatch):
    monkeypatch.setattr(runner, "load", lambda _: ({}, {"atomic_all": np.zeros((1, 3))}, {}))
    names = ("initialize", "load", "pack_rows", "optimizer_for", "measure_panels", "graph_values")
    before = {name: getattr(runner.native, name) for name in names}
    before_depth = runner.depth.load, runner.depth.sweep, runner.depth.panel_names
    with pytest.raises(RuntimeError, match="controlled"):
        with runner.installed(config):
            assert runner.native.initialize is runner.initialize
            installed = {name: getattr(runner.native, name) for name in names}
            with runner.installed(config):
                assert {name: getattr(runner.native, name) for name in names} == installed
            assert {name: getattr(runner.native, name) for name in names} == installed
            raise RuntimeError("controlled")
    assert {name: getattr(runner.native, name) for name in names} == before
    assert (runner.depth.load, runner.depth.sweep, runner.depth.panel_names) == before_depth


def test_nonfinite_training_diagnostic_stops_run(monkeypatch):
    monkeypatch.setattr(runner, "_native_graph_values", lambda *args: {"loss": float("nan")})
    with pytest.raises(FloatingPointError):
        runner.graph_values(None, 0)
