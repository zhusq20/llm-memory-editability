"""Scientific contracts for learning order, held-out knowledge and restoration."""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability import sequential_transfer as st
from llm_memory_editability.realworld_composition import (
    construct,
    model_digest,
    optimizer_for,
    update,
)


def fixture_data():
    def row(key, subset=None, atoms=None, role=None):
        value = {
            "id": key,
            "answer": "answer",
            "question": "question",
            "encoded": {"prefix": [1, 2], "target": [3, 30], "input": [1, 2, 3]},
        }
        if subset:
            value["subset"] = subset
        if atoms:
            value.update(atom_ids=atoms, sequential_role=role)
        return value

    atoms = [row("a0", "A"), row("a1", "A"), row("a2", "A"), row("b0", "B"), row("b1", "B")]
    chains = [row("train", atoms=["a0", "a1"], role="AA")]
    evaluation = [
        row("test-aa", atoms=["a1", "a2"], role="AA"),
        row("test-ba", atoms=["b0", "a0"], role="BA"),
        row("test-ab", atoms=["a0", "b1"], role="AB"),
        row("test-bb", atoms=["b0", "b1"], role="BB"),
    ]
    return {
        "atoms": atoms,
        "train_compositions": chains,
        "evaluation_compositions": evaluation,
        "panels": {
            "atomic_A": ["a0"],
            "atomic_B": [],
            "train_composition": ["train"],
            **{r["sequential_role"]: [r["id"]] for r in evaluation},
        },
    }


def spec(history="sequential_composition"):
    return {
        "history": history,
        "stage_a_steps": 7,
        "stage_b_steps": 5,
        "batch_size": 6,
        "microbatch_size": 6,
        "sampling_seed": 17,
        "initialization": 23,
        "unique_layers": 2,
        "repeats": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "warmup_steps": 2,
    }


def test_joint_is_exact_example_and_supervision_multiset():
    data = fixture_data()
    records, sequential = st.training_plan(data, spec())
    _, joint = st.training_plan(data, spec("joint"))
    assert Counter(sequential.flatten()) == Counter(joint.flatten())
    assert not np.array_equal(sequential, joint)
    a = st.plan_manifest(records, sequential, spec())
    b = st.plan_manifest(records, joint, spec("joint"))
    for key in ("multiset_sha256", "supervised_tokens", "input_tokens", "counts", "examples"):
        assert a[key] == b[key]
    assert a["plan_sha256"] != b["plan_sha256"]


def test_new_facts_and_all_B_combinations_absent_from_A():
    records, plan = st.training_plan(fixture_data(), spec())
    assert all(records[i].get("subset") != "B" for i in plan[:7].flatten())
    assert all(records[i].get("subset") in {"A", "B"} for i in plan[7:].flatten())
    assert all(not row["id"].startswith("test") for row in records)
    assert all("B" not in row.get("sequential_role", "") for row in records)


def test_atomic_control_has_identical_new_fact_and_replay_stream():
    records, original = st.training_plan(fixture_data(), spec())
    _, control = st.training_plan(fixture_data(), spec("sequential_atomic"))
    np.testing.assert_array_equal(original[7:], control[7:])
    assert all(records[i].get("subset") == "A" for i in control[:7].flatten())
    np.testing.assert_array_equal(original[:7, :3], control[:7, :3])


def test_sham_has_identical_parent_and_only_A_continuation():
    records, original = st.training_plan(fixture_data(), spec())
    _, sham = st.training_plan(fixture_data(), spec("sequential_composition_sham"))
    np.testing.assert_array_equal(original[:7], sham[:7])
    np.testing.assert_array_equal(original[7:, :3], sham[7:, :3])
    assert all(records[i].get("subset") == "A" for i in sham[7:].flatten())


def test_evaluation_panel_includes_every_necessary_atom():
    groups = st.evaluation_groups(fixture_data(), full=False)
    available = {row["id"] for name in ("atomic_A", "atomic_B") for row in groups[name]}
    needed = {
        key for role in ("AA", "BA", "AB", "BB") for row in groups[role] for key in row["atom_ids"]
    }
    assert needed <= available
    assert {row["id"] for row in groups["BB"]} == {"test-bb"}


@pytest.mark.parametrize("leak", ("new_fact", "heldout_chain", "wrong_role"))
def test_data_contract_rejects_leakage(leak):
    data = fixture_data()
    if leak == "new_fact":
        data["train_compositions"][0]["atom_ids"] = ["a0", "b0"]
    elif leak == "heldout_chain":
        data["evaluation_compositions"][0]["atom_ids"] = ["a0", "a1"]
    else:
        data["evaluation_compositions"][0]["sequential_role"] = "BB"
    with pytest.raises(ValueError):
        st.validate_data(data)


def test_learning_rate_has_same_global_history_and_explicit_stage_reset():
    for step in range(1, 13):
        assert len({st.learning_rate(spec(history), step) for history in st.HISTORIES}) == 1
    assert st.learning_rate(spec(), 7) == pytest.approx(0.0001)
    assert st.learning_rate(spec(), 8) == pytest.approx(0.0005)
    assert st.learning_rate(spec(), 12) == pytest.approx(0.0001)


def test_resume_restores_optimizer_rng_and_next_update(tmp_path):
    torch.set_num_threads(1)
    model_config = dict(vocab_size=31, positions=32, hidden_size=8, attention_heads=2, dropout=0.1)
    run_spec = spec()
    records, plan = st.training_plan(fixture_data(), run_spec)
    manifest = st.plan_manifest(records, plan, run_spec)
    model = construct(model_config, run_spec, "cpu")
    optimizer = optimizer_for(model, 0.001, 0.1)
    update(model, optimizer, [records[i] for i in plan[0]], run_spec, "cpu", 30)
    counts = np.zeros(len(records), dtype=np.int64)
    np.add.at(counts, plan[0], 1)
    st.save_checkpoint(
        tmp_path / "resume.pt", model, optimizer, 1, counts, {"examples": 6}, [], manifest
    )
    update(model, optimizer, [records[i] for i in plan[1]], run_spec, "cpu", 30)
    expected = model_digest(model)
    restored = construct(model_config, run_spec, "cpu")
    restored_optimizer = optimizer_for(restored, 0.001, 0.1)
    state = st.restore_checkpoint(tmp_path / "resume.pt", restored, restored_optimizer, manifest)
    assert state["step"] == 1 and state["counts"].sum() == 6
    update(restored, restored_optimizer, [records[i] for i in plan[1]], run_spec, "cpu", 30)
    assert model_digest(restored) == expected
    changed = copy.deepcopy(manifest)
    changed["plan_sha256"] = "wrong"
    with pytest.raises(ValueError, match="sampling plan"):
        st.restore_checkpoint(tmp_path / "resume.pt", restored, restored_optimizer, changed)


def test_training_worker_resumes_without_repeating_or_skipping_updates(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    run_spec = {
        **spec(),
        "stage_a_steps": 2,
        "stage_b_steps": 2,
        "model": dict(vocab_size=31, positions=32, hidden_size=8, attention_heads=2, dropout=0.1),
        "data_sha256": "unit-test",
        "evaluation_nodes": [0, 2, 4],
    }
    data = fixture_data()

    def setup(value, device):
        return (
            data,
            SimpleNamespace(eos_token_id=30),
            construct(value["model"], value, device),
            torch.device(device),
        )

    monkeypatch.setattr(st, "_setup", setup)
    monkeypatch.setattr(
        st,
        "evaluate_transfer",
        lambda model, *args, **kwargs: (
            {"fixture": {"n": 1, "digest": model_digest(model)}},
            {},
        ),
    )
    uninterrupted = st.run(run_spec, tmp_path / "uninterrupted", "cpu")
    original_update = st.update
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption after saved stage A")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(st, "update", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        st.run(run_spec, tmp_path / "resumed", "cpu")
    monkeypatch.setattr(st, "update", original_update)
    resumed = st.run(run_spec, tmp_path / "resumed", "cpu")
    assert resumed["model_sha256"] == uninterrupted["model_sha256"]
    assert resumed["examples"] == uninterrupted["examples"] == 24
    left = torch.load(tmp_path / "uninterrupted/latest.pt", weights_only=False)
    right = torch.load(tmp_path / "resumed/latest.pt", weights_only=False)
    np.testing.assert_array_equal(left["counts"], right["counts"])
    history = json.loads((tmp_path / "resumed/learning.json").read_text())
    assert [row["step"] for row in history] == [0, 2, 4]
    assert history[1]["atomic_B_epochs"] == 0
    assert history[-1]["atomic_B_epochs"] > 0
