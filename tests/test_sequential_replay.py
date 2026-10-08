"""Frozen stream identity, fresh optimization, and exact branch continuation."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability import sequential_replay as replay
from llm_memory_editability import sequential_transfer as original
from llm_memory_editability.realworld_composition import construct, model_digest
from llm_memory_editability.realworld_composition_data import sha256


def fixture_data():
    def atom(key, subset):
        return {
            "id": key,
            "subset": subset,
            "answer": "answer",
            "question": "question",
            "encoded": {"prefix": [1, 2], "target": [3, 30], "input": [1, 2, 3]},
        }

    def chain(key, ids, role):
        row = atom(key, "unused")
        del row["subset"]
        return {**row, "atom_ids": ids, "sequential_role": role}

    return {
        "atoms": [
            atom("a0", "A"),
            atom("a1", "A"),
            atom("a2", "A"),
            atom("b0", "B"),
            atom("b1", "B"),
        ],
        "train_compositions": [chain("train-aa", ["a0", "a1"], "AA")],
        "evaluation_compositions": [
            chain("test-aa", ["a1", "a2"], "AA"),
            chain("test-ba", ["b0", "a0"], "BA"),
            chain("test-ab", ["a0", "b1"], "AB"),
            chain("test-bb", ["b0", "b1"], "BB"),
        ],
    }


def parent_spec():
    return {
        "history": "sequential_composition",
        "stage_a_steps": 4,
        "stage_b_steps": 4,
        "batch_size": 6,
        "microbatch_size": 6,
        "sampling_seed": 17,
        "initialization": 23,
        "unique_layers": 2,
        "repeats": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "warmup_steps": 2,
        "evaluation_nodes": [0, 4, 8],
        "evaluation_batch_size": 3,
        "model": {
            "vocab_size": 31,
            "positions": 32,
            "hidden_size": 8,
            "attention_heads": 2,
            "dropout": 0.1,
        },
    }


def replay_spec(arm="atom_replay"):
    return {
        "phase": "engineering",
        "replay_arm": arm,
        "original_stage_a_steps": 4,
        "stage_b_steps": 4,
        "batch_size": 6,
        "microbatch_size": 6,
        "learning_rate": 3e-4,
        "warmup_steps": 1,
        "minimum_lr_fraction": 0.1,
        "weight_decay": 0.1,
        "optimizer_betas": [0.9, 0.95],
        "evaluation_nodes": [0, 2, 4],
        "evaluation_batch_size": 3,
    }


def make_parent(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = fixture_data()
    source = tmp_path / "source"
    source.mkdir()
    data_path = source / "data.json"
    data_path.write_text(json.dumps(data))
    spec = {
        **parent_spec(),
        "data_file": str(data_path),
        "data_sha256": sha256(data_path),
        "tokenizer": str(source / "tokenizer"),
    }

    def setup(value, device):
        return (
            data,
            SimpleNamespace(eos_token_id=30),
            construct(value["model"], value, device),
            torch.device(device),
        )

    def evaluation(model, *args, **kwargs):
        digest = model_digest(model)
        return {"fixture": {"n": 1, "digest": digest}}, {"fixture": [{"digest": digest}]}

    monkeypatch.setattr(original, "_setup", setup)
    monkeypatch.setattr(original, "evaluate_transfer", evaluation)
    original.run(spec, source, "cpu")
    (source / "audit.json").write_text(json.dumps({"passed": True}))
    new_spec = {
        **replay_spec(),
        "parent_run_dir": str(source),
        "parent_checkpoint": str(source / "stage-a.pt"),
        "parent_checkpoint_sha256": sha256(source / "stage-a.pt"),
        "parent_run_sha256": sha256(source / "run.json"),
    }
    return source, new_spec


def test_two_arms_copy_identical_original_b_stream_and_declared_old_order():
    data, parent = fixture_data(), parent_spec()
    records, old = original.training_plan(data, parent)
    _r, atomic, atomic_manifest, _p = replay.continuation_plan(data, parent, replay_spec())
    _r, full, full_manifest, _p = replay.continuation_plan(data, parent, replay_spec("full_replay"))
    np.testing.assert_array_equal(atomic[:, 3:], old[4:, 3:])
    np.testing.assert_array_equal(full[:, 3:], old[4:, 3:])
    np.testing.assert_array_equal(atomic[:, :3].flatten(), old[:4, :3].flatten())
    np.testing.assert_array_equal(full[:, :3].flatten(), old[:4].flatten()[:12])
    assert atomic_manifest["new_stream_sha256"] == full_manifest["new_stream_sha256"]
    assert atomic_manifest["plan_sha256"] != full_manifest["plan_sha256"]
    assert all(records[i].get("subset") == "A" for i in atomic[:, :3].flatten())
    assert sum(records[i].get("subset") == "A" for i in full[:, :3].flatten()) == 6
    assert [records[i].get("subset") for i in full[:, 0]] == ["A", None, "A", None]
    assert full_manifest["examples"] == 24
    assert all(not record["id"].startswith("test-") for record in records)
    assert all("B" not in record.get("sequential_role", "") for record in records)


@pytest.mark.parametrize(
    "change",
    [
        {"replay_arm": "unknown"},
        {"original_stage_a_steps": 3},
        {"stage_b_steps": 5},
        {"stage_b_steps": 6},
        {"batch_size": 4},
    ],
)
def test_incompatible_suffix_contracts_are_rejected(change):
    with pytest.raises(ValueError):
        replay.continuation_plan(fixture_data(), parent_spec(), {**replay_spec(), **change})


@pytest.mark.parametrize("steps,warmup", [(4000, 40), (7000, 70)])
def test_new_schedule_has_one_percent_warmup_and_tenth_peak_endpoint(steps, warmup):
    spec = {**replay_spec(), "stage_b_steps": steps, "warmup_steps": warmup}
    assert replay.learning_rate(spec, 0) == 0
    assert replay.learning_rate(spec, 1) == pytest.approx(3e-4 / warmup)
    assert replay.learning_rate(spec, warmup) == pytest.approx(3e-4)
    assert replay.learning_rate(spec, steps) == pytest.approx(3e-5)


def test_source_restores_model_rng_but_optimizer_is_fresh_and_hashes_are_required(
    tmp_path, monkeypatch
):
    source, spec = make_parent(tmp_path, monkeypatch)
    checkpoint = torch.load(source / "stage-a.pt", map_location="cpu", weights_only=False)
    loaded = replay._load_source(spec, "cpu")
    model = loaded[2]
    assert all(
        torch.equal(value, checkpoint["model"][name]) for name, value in model.state_dict().items()
    )
    torch.testing.assert_close(torch.get_rng_state(), checkpoint["cpu_rng"], atol=0, rtol=0)
    optimizer = replay.fresh_optimizer(model, spec)
    assert not optimizer.state and checkpoint["optimizer"]["state"]
    assert all(group["betas"] == (0.9, 0.95) for group in optimizer.param_groups)
    assert all(
        tuple(group["betas"]) == (0.9, 0.999) for group in checkpoint["optimizer"]["param_groups"]
    )
    with pytest.raises(ValueError, match="checkpoint hash"):
        replay._load_source({**spec, "parent_checkpoint_sha256": "wrong"}, "cpu")
    with pytest.raises(ValueError, match="metadata hash"):
        replay._load_source({**spec, "parent_run_sha256": "wrong"}, "cpu")
    changed = copy.deepcopy(spec)
    changed["model"] = {**parent_spec()["model"], "dropout": 0.0}
    with pytest.raises(ValueError, match="inherited model"):
        replay._load_source(changed, "cpu")


def test_stage_b_branch_interrupt_resume_exact_and_new_only_accounting(tmp_path, monkeypatch):
    source, spec = make_parent(tmp_path, monkeypatch)
    source_hash = sha256(source / "stage-a.pt")
    uninterrupted = replay.run(spec, tmp_path / "uninterrupted", "cpu")
    original_update = replay.update
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("stop after a saved B node")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(replay, "update", interrupt)
    with pytest.raises(RuntimeError, match="saved B node"):
        replay.run(spec, tmp_path / "resumed", "cpu")
    monkeypatch.setattr(replay, "update", original_update)
    resumed = replay.run(spec, tmp_path / "resumed", "cpu")
    assert resumed["model_sha256"] == uninterrupted["model_sha256"]
    assert resumed["examples"] == uninterrupted["examples"] == 24
    assert resumed["local_optimizer_updates"] == 4 and resumed["step"] == 8
    left = torch.load(tmp_path / "uninterrupted/latest.pt", weights_only=False)
    right = torch.load(tmp_path / "resumed/latest.pt", weights_only=False)
    np.testing.assert_array_equal(left["counts"], right["counts"])
    torch.testing.assert_close(left["cpu_rng"], right["cpu_rng"], atol=0, rtol=0)
    for key, state in left["optimizer"]["state"].items():
        assert int(state["step"]) == 4
        for name, value in state.items():
            torch.testing.assert_close(
                value, right["optimizer"]["state"][key][name], atol=0, rtol=0
            )
    history = json.loads((tmp_path / "resumed/learning.json").read_text())
    assert [row["step"] for row in history] == [4, 6, 8]
    assert [row["examples"] for row in history] == [0, 12, 24]
    assert history[0]["training_seconds"] == 0 and history[0]["composition_epochs"] == 0
    assert sha256(source / "stage-a.pt") == source_hash
    with pytest.raises(ValueError, match="different process"):
        replay.audit(tmp_path / "resumed", "cpu")
    trained_pid = resumed["pid"]
    monkeypatch.setattr(replay.os, "getpid", lambda: trained_pid + 1000)
    result = replay.audit(tmp_path / "resumed", "cpu")
    assert result["passed"] and result["fresh_optimizer_verified"]
    assert result["parent_boundary_recomputed"] and result["new_only_exposures_exact"]
    assert (tmp_path / "resumed/complete.json").exists()


def test_full_replay_keeps_b_and_exposes_training_compositions(tmp_path, monkeypatch):
    _source, spec = make_parent(tmp_path, monkeypatch)
    spec["replay_arm"] = "full_replay"
    out = tmp_path / "full"
    replay.run(spec, out, "cpu")
    history = json.loads((out / "learning.json").read_text())
    assert history[-1]["composition_epochs"] == 6
    manifest = json.loads((out / "sampling-plan.json").read_text())
    assert manifest["counts"]["train-aa"] == 6
    assert sum(count for key, count in manifest["counts"].items() if key.startswith("b")) == 12
    assert manifest["supervised_tokens"] == 48
