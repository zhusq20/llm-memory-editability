"""Stateful continuation checks: unchanged streams, exact optimizer/RNG replay, scope."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_cross import documents, epoch_documents, qa_schedule
from llm_memory_editability.bios_cross_continue import (
    capture_state,
    continuation_lr,
    edit_update,
    expected_exposure,
    learning_update,
    restore_state,
    tree_hash,
    validate_optimizer,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig, select_parameters


@pytest.fixture(scope="module")
def world():
    from test_bios_cross import cross_world

    return cross_world.__wrapped__()


def test_extended_schedule_is_identical_prefix_and_excludes_heldout(world):
    short, long = qa_schedule(world, 15360), qa_schedule(world, 30720)
    np.testing.assert_array_equal(short, long[:15360])
    assert not np.isin(long, world.heldout_ids).any()
    for chain in range(2):
        assert np.isin(long[:, chain * 20 : (chain + 1) * 20], world.train_ids[chain]).all()


def test_exposure_reconstruction_includes_warmup_and_rotated_slots(world):
    step, lr = 13 * 128, 0.0001
    docs, schedule = documents(world, "project"), qa_schedule(world, step)
    counts = np.zeros(len(world.answers), dtype=np.int64)
    slots = np.zeros((world.n_base, 10), dtype=np.int64)
    weighted = np.zeros(len(world.answers), dtype=np.float64)
    for epoch in range(13):
        arranged = epoch_documents(docs, world.seed, epoch)
        for offset in range(128):
            current = epoch * 128 + offset
            facts = arranged[offset * 16 : (offset + 1) * 16]
            np.add.at(counts, facts.ravel(), 1)
            np.add.at(counts, schedule[current], 1)
            np.add.at(slots, (facts, np.arange(10)[None, :]), 1)
            np.add.at(weighted, facts.ravel(), continuation_lr(lr, current))
            np.add.at(weighted, schedule[current], continuation_lr(lr, current))
    reconstructed = expected_exposure(world, docs, schedule, step, lr)
    np.testing.assert_array_equal(counts, reconstructed[0])
    np.testing.assert_array_equal(slots, reconstructed[1])
    np.testing.assert_allclose(weighted, reconstructed[2], rtol=0, atol=1e-12)


def small_setup():
    torch.manual_seed(721)
    model = CausalLM(ModelConfig(vocab_size=32, width=8, layers=8, heads=1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    tokens = torch.randint(0, 32, (8, 6))
    batch = {
        "tokens": tokens,
        "positions": torch.tensor([[3, 4]]).repeat(8, 1),
        "labels": tokens[:, 4:6],
    }
    return model, optimizer, batch


def random_update(model, optimizer, batch, lr=0.0001):
    # Sampling exercises CPU RNG restoration as well as the AdamW moment state.
    qa = torch.randint(0, 8, (4,))
    learning_update(model, optimizer, batch, batch, qa, 0, 4, lr, torch.device("cpu"))


def test_resume_matches_uninterrupted_weights_moments_and_rng(tmp_path):
    torch.set_num_threads(1)
    model, optimizer, batch = small_setup()
    for _ in range(3):
        random_update(model, optimizer, batch)
    path = tmp_path / "resume.pt"
    torch.save(capture_state(model, optimizer, torch.device("cpu"), step=3), path)
    for _ in range(4):
        random_update(model, optimizer, batch)
    expected = tree_hash(capture_state(model, optimizer, torch.device("cpu"), step=7))
    restored, resumed_optimizer, _ = small_setup()
    saved = torch.load(path, weights_only=False)
    restore_state(restored, resumed_optimizer, saved, torch.device("cpu"))
    for _ in range(4):
        random_update(restored, resumed_optimizer, batch)
    actual = tree_hash(capture_state(restored, resumed_optimizer, torch.device("cpu"), step=7))
    assert actual == expected


def test_sensitivity_branch_keeps_parent_moments_and_does_not_restart_warmup():
    torch.set_num_threads(1)
    model, optimizer, batch = small_setup()
    for _ in range(2):
        random_update(model, optimizer, batch)
    parent = copy.deepcopy(capture_state(model, optimizer, torch.device("cpu"), step=2))
    branches = []
    for rate in (0.0001, 0.0003):
        child, child_opt, _ = small_setup()
        restore_state(child, child_opt, parent, torch.device("cpu"))
        assert tree_hash(child_opt.state_dict()) == tree_hash(parent["optimizer"])
        applied = continuation_lr(rate, 15360)
        assert applied == rate
        random_update(child, child_opt, batch, applied)
        assert all(int(value["step"]) == 3 for value in child_opt.state_dict()["state"].values())
        branches.append((tree_hash(child.state_dict()), tree_hash(child_opt.state_dict()["state"])))
    assert branches[0][0] != branches[1][0]
    assert branches[0][1] == branches[1][1]


def test_optimizer_audit_rejects_reset_or_incomplete_moments():
    model, optimizer, batch = small_setup()
    random_update(model, optimizer, batch)
    state = optimizer.state_dict()
    validate_optimizer(state, 1, 0.0001)
    with pytest.raises(ValueError, match="count"):
        validate_optimizer(state, 2, 0.0001)
    altered = copy.deepcopy(state)
    del next(iter(altered["state"].values()))["exp_avg_sq"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_optimizer(altered, 1, 0.0001)


def test_mlp_selection_never_enables_embedding_attention_or_other_layers():
    model, _, _ = small_setup()
    selected = select_parameters(model, "mlp", 3)
    wanted = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert wanted and len(selected) == len(wanted)
    assert all(
        any(name.startswith(f"blocks.{layer}.mlp.") for layer in (3, 4, 5)) for name in wanted
    )
    assert not model.token.weight.requires_grad
    assert not model.blocks[3].attention.qkv.weight.requires_grad


def test_edit_resume_matches_uninterrupted_and_leaves_frozen_parameters_intact(tmp_path):
    torch.set_num_threads(1)
    model, _, data = small_setup()
    before = copy.deepcopy(model.state_dict())
    selected = select_parameters(model, "mlp", 3)
    optimizer = torch.optim.AdamW(selected, lr=0.00003, weight_decay=0.1)
    with torch.no_grad():
        references = model(data["tokens"], data["positions"]).detach().clone()
    new_data = {**data, "labels": (data["labels"] + 3) % 32}
    device = torch.device("cpu")

    def update(current, opt, params):
        choices = torch.randint(0, 8, (4,))
        edit_update(
            current, opt, params, data, new_data, references, choices, choices, choices, device
        )

    for _ in range(2):
        update(model, optimizer, selected)
    path = tmp_path / "edit-resume.pt"
    torch.save(capture_state(model, optimizer, device, step=2), path)
    for _ in range(3):
        update(model, optimizer, selected)
    expected = tree_hash(capture_state(model, optimizer, device, step=5))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, before[name])
    child, _, _ = small_setup()
    params = select_parameters(child, "mlp", 3)
    opt = torch.optim.AdamW(params, lr=0.00003, weight_decay=0.1)
    restore_state(child, opt, torch.load(path, weights_only=False), device)
    for _ in range(3):
        update(child, opt, params)
    assert tree_hash(capture_state(child, opt, device, step=5)) == expected


def test_matrix_pairs_every_common_run_and_sensitivity_uses_same_parent(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_bios_cross_continue.py"
    spec = importlib.util.spec_from_file_location("continue_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    matrix = runner.jobs(tmp_path / "parents", tmp_path / "outputs")
    assert len(matrix) == 36 and len({job["output"] for job in matrix}) == 36
    common = {job["parent"]: job for job in matrix if job["branch"] == "common"}
    sensitivity = [job for job in matrix if job["branch"] == "sensitivity"]
    assert len(common) == 24 and len(sensitivity) == 12
    for job in sensitivity:
        assert job["seed"] == 0 and job["lr"] == 0.0003
        assert common[job["parent"]]["lr"] == 0.0001
        assert job["parent"] not in job["output"]


def summary_module():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_bios_cross_continue.py"
    spec = importlib.util.spec_from_file_location("continue_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_requires_correct_eos_and_rejects_forged_correct_flags(tmp_path):
    summary = summary_module()
    path = tmp_path / "prediction.npz"
    np.savez(
        path, prediction=[1, 2], ended=[True, False], correct=[True, True], value_nll=[0.1, 0.2]
    )
    with pytest.raises(AssertionError):
        summary.load_arrays(path, np.array([1, 2]))
    np.savez(
        path, prediction=[1, 2], ended=[True, False], correct=[True, False], value_nll=[0.1, 0.2]
    )
    assert summary.load_arrays(path, np.array([1, 2]))["correct"].sum() == 1


def test_summary_pairs_organization_within_branch_and_requires_complete_blocks():
    summary = summary_module()
    rows = []
    for condition, company, project in (
        ("company", 0.9, 0.6),
        ("project", 0.8, 0.85),
        ("neither", 0.5, 0.4),
    ):
        rows.append(
            {
                "branch": "common",
                "width": 128,
                "world": 0,
                "seed": 0,
                "step": 30720,
                "condition": condition,
                "company_heldout": company,
                "project_heldout": project,
                "mean_heldout": (company + project) / 2,
                "base_accuracy": 0.99,
            }
        )
    rows.extend({**row, "branch": "sensitivity"} for row in rows[:2])
    result = summary.paired_learning(rows)
    assert len(result) == 1 and result[0]["branch"] == "common"
    assert result[0]["matching_effect"] == pytest.approx(0.175)
    assert result[0]["mismatched_vs_neither"] == pytest.approx(0.25)
