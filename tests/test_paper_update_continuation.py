"""Check full-state continuation and isolate the new gate's LR intervention."""

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.paper_update_gate import PaperUpdateGateGPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_paper_update_continuation")
SPEC = {"lr": 1e-4, "weight_decay": 0.1}


def config():
    return {
        "batch": "paper-update-gate-continuation-v1",
        "continuation": {"start_step": 300000, "new_warmup": False},
        "training": {**SPEC, "steps": 1500000},
        "evaluation_nodes": list(range(300000, 1500001, 100000)),
        "budget": {"new_scientific_updates": 4800000},
        "arms": {
            name: {
                "architecture": "discoloop",
                "unique_layers": 1,
                "repeats": 4,
                "update_mode": mode,
                "initial_update": 0.9,
                "parent": parent,
                "update_gate_lr_factor": factor,
            }
            for name, mode, parent, factor in (
                ("paper", "full", "paper", 1.0),
                ("fixed-update", "fixed", "fixed-update", 1.0),
                ("token-update", "token", "token-update", 1.0),
                ("token-update-half-lr", "token", "token-update", 0.5),
            )
        },
    }


def tiny_model(mode="token"):
    torch.manual_seed(491)
    hf = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    return PaperUpdateGateGPT(hf, repeats=3, update_mode=mode, initial_update=0.9).double()


def assign_gradients(model, update):
    # This is an optimizer intervention test: every corresponding parameter gets
    # the exact same nonzero gradient, independently of its subsequent value.
    for index, parameter in enumerate(model.parameters()):
        values = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(parameter.shape)
        parameter.grad = (values.remainder(11) + index + update + 1) / 100


def trained_parent(mode="token"):
    model = tiny_model(mode)
    optimizer = runner.adapter.optimizer_for(model, torch.tensor(SPEC["lr"]), 0.1)
    for update in range(3):
        assign_gradients(model, update)
        optimizer.step()
    return model, optimizer


def named_state(model, optimizer):
    return {
        name: {key: value.clone() for key, value in optimizer.state[parameter].items()}
        for name, parameter in model.named_parameters()
    }


def restored(parent, optimizer, factor):
    model = copy.deepcopy(parent)
    # CPU load_state_dict can retain tensors from its input. Real checkpoint
    # loads are independent; duplicate the saved mapping for these CPU branches.
    restored_optimizer, lr, gate_lr = runner.restore_optimizer(
        model, copy.deepcopy(optimizer.state_dict()), SPEC, factor, torch.device("cpu")
    )
    return model, restored_optimizer, lr, gate_lr


@pytest.mark.parametrize("mode", ("full", "fixed", "token"))
def test_restore_preserves_every_nonzero_adam_moment_step_and_option(mode):
    parent, original = trained_parent(mode)
    expected = named_state(parent, original)
    model, optimizer, lr, gate_lr = restored(parent, original, 1.0)
    actual = named_state(model, optimizer)
    assert actual.keys() == expected.keys()
    for name, state in expected.items():
        assert state.keys() == {"step", "exp_avg", "exp_avg_sq"}
        assert state["step"].item() == 3
        assert torch.all(state["exp_avg"] != 0)
        assert torch.all(state["exp_avg_sq"] > 0)
        for key, value in state.items():
            assert torch.equal(actual[name][key], value), (name, key)
            assert actual[name][key].dtype == value.dtype
    assert runner.optimizer_digest(model, optimizer) == runner.optimizer_digest(parent, original)
    assert len(optimizer.param_groups) == (3 if mode == "token" else 2)
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    for index, group in enumerate(optimizer.param_groups):
        assert group["capturable"] is True
        assert group["fused"] is True
        assert group["betas"] == (0.9, 0.999)
        assert group["eps"] == 1e-8
        assert group["lr"] is (gate_lr if index == 2 else lr)
        if index:
            assert group["weight_decay"] == 0.0
    groups = runner.parameter_groups(model, optimizer)
    assert {"gate_paper_weight", "gate_paper_bias"} <= set(groups[1])
    if mode == "token":
        assert set(groups[2]) == runner.GATE_NAMES
        assert not runner.GATE_NAMES & set(groups[1])
    else:
        assert gate_lr is None


def test_factor_one_regroup_matches_original_two_groups_for_same_gradient():
    parent, original = trained_parent()
    model, optimizer, _, _ = restored(parent, original, 1.0)
    assign_gradients(parent, 8)
    assign_gradients(model, 8)
    original.step()
    optimizer.step()
    for name, expected in parent.named_parameters():
        assert torch.equal(model.get_parameter(name), expected), name
    assert runner.optimizer_digest(model, optimizer) == runner.optimizer_digest(parent, original)


def test_half_lr_changes_only_new_gate_update_and_preserves_zero_decay():
    parent, original = trained_parent()
    full, full_optimizer, full_lr, _ = restored(parent, original, 1.0)
    half, half_optimizer, half_lr, gate_lr = restored(parent, original, 0.5)
    assert torch.equal(full_lr, half_lr)
    assert torch.equal(gate_lr, half_lr * 0.5)
    before = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
    for model, optimizer in ((full, full_optimizer), (half, half_optimizer)):
        assign_gradients(model, 8)
        optimizer.step()
    for name, full_value in full.named_parameters():
        half_value = half.get_parameter(name)
        if name in runner.GATE_NAMES:
            assert not torch.equal(half_value, full_value), name
            torch.testing.assert_close(
                half_value - before[name],
                (full_value - before[name]) * 0.5,
                rtol=0,
                atol=1e-15,
            )
        else:
            assert torch.equal(half_value, full_value), name
    # Adam moments depend on gradients, not LR; both branches retain identical
    # moments after this controlled step, including the altered gate's moments.
    assert runner.optimizer_digest(full, full_optimizer) == runner.optimizer_digest(
        half, half_optimizer
    )
    for optimizer in (full_optimizer, half_optimizer):
        assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.0, 0.0]


@pytest.mark.parametrize("mode", ("full", "fixed"))
def test_reduced_gate_lr_requires_trainable_update_gate(mode):
    parent, optimizer = trained_parent(mode)
    with pytest.raises(ValueError, match="Only a learned update gate"):
        restored(parent, optimizer, 0.5)


@pytest.mark.parametrize("completed_batches", (2, 3))
def test_stream_resume_preserves_partial_epoch_next_batches_and_exposure(completed_batches):
    size, batch_size = 10, 4
    original = runner.MergedBatchStream(size, 184)
    counts = np.zeros(size, dtype=np.int64)
    for _ in range(completed_batches):
        rows, count = original.batch(batch_size)
        assert np.count_nonzero(rows != size) == count
        counts += np.bincount(rows, minlength=size + 1)[:size]
    source = copy.deepcopy(original.state_dict())
    resumed = runner.MergedBatchStream(size, 99999)
    resumed.load_state_dict(source)
    resumed_counts = counts.copy()
    for update in range(17):
        expected, expected_count = original.batch(batch_size)
        actual, actual_count = resumed.batch(batch_size)
        np.testing.assert_array_equal(actual, expected)
        assert actual_count == expected_count
        if update == 0 and completed_batches == 2:
            assert actual_count == 2
            np.testing.assert_array_equal(actual[2:], [size, size])
        if update == 0 and completed_batches == 3:
            assert actual_count == 4
        counts += np.bincount(expected, minlength=size + 1)[:size]
        resumed_counts += np.bincount(actual, minlength=size + 1)[:size]
        np.testing.assert_array_equal(resumed_counts, counts)
    assert original.state_dict()["rng"] == resumed.state_dict()["rng"]
    np.testing.assert_array_equal(original.remaining, resumed.remaining)
    # Loading a stream must not consume or mutate its inherited source state.
    assert source["rng"] != resumed.state_dict()["rng"]
    assert len(source["remaining"]) == (2 if completed_batches == 2 else 0)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("continuation", "start_step", 0),
        ("continuation", "new_warmup", True),
        ("training", "steps", 300000),
        ("training", "lr", 5e-5),
        ("training", "weight_decay", 0.2),
        ("budget", "new_scientific_updates", 6000000),
    ],
)
def test_validate_rejects_changed_budget_or_backbone_recipe(section, key, value):
    selected = config()
    runner.validate_config(selected)
    selected[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_config(selected)


@pytest.mark.parametrize(
    ("arm", "key", "value"),
    [
        ("paper", "update_mode", "fixed"),
        ("fixed-update", "initial_update", 0.95),
        ("token-update", "repeats", 8),
        ("token-update", "update_gate_lr_factor", 0.5),
        ("token-update-half-lr", "parent", "fixed-update"),
        ("token-update-half-lr", "update_gate_lr_factor", 1.0),
    ],
)
def test_validate_rejects_changed_arm_or_parent(arm, key, value):
    selected = config()
    selected["arms"][arm][key] = value
    with pytest.raises(ValueError, match="Changed authorized arm"):
        runner.validate_config(selected)


def test_validate_rejects_added_arm_and_changed_evaluation_nodes():
    selected = config()
    selected["arms"]["extra"] = copy.deepcopy(selected["arms"]["paper"])
    with pytest.raises(ValueError, match="four-arm"):
        runner.validate_config(selected)
    selected = config()
    selected["evaluation_nodes"].remove(400000)
    with pytest.raises(ValueError, match="evaluation nodes"):
        runner.validate_config(selected)
