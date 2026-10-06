"""Inference information boundaries and matched training-budget contracts."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability.capacity_controls import (
    decode_bridge,
    first_queries,
    second_queries,
    training_indices,
)
from llm_memory_editability.capacity_scaling import generate_world, pack


@pytest.fixture
def world():
    return generate_world(
        {
            "world_seed": 913,
            "heads_n": 32,
            "max_heads": 256,
            "values_n": 256,
            "relations": 4,
            "evaluation_per_pool": 40,
        }
    )


def test_replay_preserves_atomic_slots_and_total_update_budget(world):
    atoms, comps = len(world["atomic"]), len(world["train_composition"])
    spec = {"stream_seed": 42}
    mixed = training_indices(world, spec, 9)
    replay = training_indices(world, {**spec, "training_mode": "atomic_replay"}, 9)
    only = training_indices(world, {**spec, "training_mode": "atomic"}, 9)
    assert len(mixed) == len(replay) == atoms + comps
    assert np.array_equal(replay[mixed < atoms], mixed[mixed < atoms])
    assert replay.max() < atoms
    counts = np.bincount(replay, minlength=atoms)
    assert counts.min() >= 1 and counts.max() - counts.min() <= 1
    assert np.array_equal(np.sort(only), np.arange(atoms))
    assert np.array_equal(
        replay, training_indices(world, {**spec, "training_mode": "atomic_replay"}, 9)
    )


def test_query_builder_never_uses_answers_or_hidden_fact_ids(world):
    rows = world["II"]
    changed = rows.copy()
    changed[:, 4:] = 999
    assert np.array_equal(first_queries(rows), first_queries(changed))
    bridges = np.arange(len(rows)) % 256
    assert np.array_equal(second_queries(rows, bridges), second_queries(changed, bridges))
    assert np.all(first_queries(rows)[:, 4] == 0)
    assert np.all(second_queries(rows, bridges)[:, 4] == 0)


def test_bridge_rejects_malformed_or_out_of_range_identifiers(world):
    labels = pack(world["atomic"][:5])[2]
    decoded, valid = decode_bridge(labels)
    assert valid.all()
    assert np.array_equal(decoded, world["atomic"][:5, 4])
    labels[0, 0] = 6
    labels[1, 1] = 8  # Valid hex digit but value is outside B's 0..255 domain.
    labels[2, 3] = 30
    labels[3, -1] = 0
    _, valid = decode_bridge(labels)
    assert valid.tolist() == [False, False, False, False, True]


def test_serial_reference_and_wrong_bridge_with_exact_atomic_model(world):
    path = Path(__file__).resolve().parents[1] / "scripts/capacity_serial_recall.py"
    spec = importlib.util.spec_from_file_location("serial_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def lookup(model, rows, device, batch):
        # The evaluator's query answer column must remain uninformative.
        assert not rows[:, 4].any()
        resolved = rows.copy()
        for i, row in enumerate(rows):
            table = world["first"] if row[0] == 0 else world["second"]
            resolved[i, 4] = table[row[1], row[2]]
        return pack(resolved)[2], None

    native = {
        p + "_both_atomic": np.ones(len(world[p]), dtype=bool) for p in ("II", "IO", "OI", "OO")
    }
    metrics, _ = module.serial_recall(
        None, world, {"evaluation_batch_size": 32, "values_n": 256}, None, lookup, native
    )
    for p in ("II", "IO", "OI", "OO"):
        if len(world[p]):
            assert metrics["serial_" + p]["accuracy"] == 1
            assert metrics["oracle_bridge_" + p]["accuracy"] == 1
            assert metrics["wrong_bridge_" + p]["accuracy"] < 0.2


def test_continuation_and_replay_train_audit(tmp_path, monkeypatch):
    import torch

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    import run_capacity_controls as runner

    # This test concerns actual optimizer/exposure/resume behavior. GPU-scored
    # decoding is independently exercised by the inference and legacy tests.
    monkeypatch.setattr(runner, "evaluate", lambda *a, **kw: ({}, {"predictions": np.zeros(1)}))
    spec = {
        "name": "small",
        "world_seed": 123,
        "initialization": 4,
        "stream_seed": 6,
        "width": 8,
        "heads_n": 4,
        "max_heads": 4,
        "values_n": 256,
        "relations": 4,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "warmup_steps": 1,
        "evaluation_per_pool": 4,
        "evaluation_epochs": [0, 1, 2],
        "checkpoint_epochs": 1,
        "log_steps": 1000,
        "max_wall_seconds": 60,
        "epochs": 2,
        "evaluation_batch_size": 64,
        "training_mode": "atomic_replay",
    }
    config = {"batch": "unit", "source_files": {}}
    runner.train(config, spec, tmp_path / "base", torch.device("cpu"))
    runner.audit(config, spec, tmp_path / "base", torch.device("cpu"))
    before = torch.load(tmp_path / "base/latest.pt", weights_only=False)
    runner.train(config, spec, tmp_path / "base", torch.device("cpu"))
    after = torch.load(tmp_path / "base/latest.pt", weights_only=False)
    assert before["step"] == after["step"]
    assert all(torch.equal(v, after["model"][k]) for k, v in before["model"].items())
    import hashlib

    child = {
        **spec,
        "name": "child",
        "epochs": 1,
        "parent": str(tmp_path / "base"),
        "parent_checkpoint_sha256": hashlib.sha256(
            (tmp_path / "base/latest.pt").read_bytes()
        ).hexdigest(),
        "weight_decay": 0.1,
    }
    runner.train(config, child, tmp_path / "child", torch.device("cpu"))
    runner.audit(config, child, tmp_path / "child", torch.device("cpu"))
    state = torch.load(tmp_path / "child/latest.pt", weights_only=False)
    assert state["optimizer"]["param_groups"][0]["weight_decay"] == 0.1
    old_step = next(iter(before["optimizer"]["state"].values()))["step"]
    new_step = next(iter(state["optimizer"]["state"].values()))["step"]
    assert new_step == old_step + state["step"]
