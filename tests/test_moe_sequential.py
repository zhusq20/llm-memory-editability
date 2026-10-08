"""Sequential sparse-access comparison and reuse contracts."""

import copy

import numpy as np
import pytest
import torch
from test_sequential_transfer import fixture_data
from test_sequential_transfer import spec as data_spec

from llm_memory_editability import moe_sequential as ms
from llm_memory_editability import sequential_transfer as st
from llm_memory_editability.grokking_reproduction import model_digest
from llm_memory_editability.loop_kv_learning import construct as old_construct
from llm_memory_editability.realworld_composition import optimizer_for, update


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(arm="M4"):
    return {
        **data_spec(),
        "architecture": arm,
        "stage_b_steps": 4,
        "replay_source": "full_stage_a",
        "unique_layers": 4,
        "repeats": 1,
        "initialization_reference_layers": 8,
        "microbatch_size": 2,
        "balance_coefficient": 0.01,
        "model": {
            "vocab_size": 43,
            "positions": 32,
            "hidden_size": 16,
            "attention_heads": 4,
            "dropout": 0.0,
        },
    }


def records():
    return [
        {"encoded": {"input": [2, 5, 8, 1, 0], "prefix": [2, 5], "target": [8, 1]}},
        {"encoded": {"input": [3, 9, 1], "prefix": [3], "target": [9, 1]}},
    ] * 2


def test_ordinary_weights_forward_and_actual_update_match_reusable_baseline(tmp_path):
    spec = options("D4")
    old_spec = {**spec, "arm": "standard", "base_unique_blocks": 4}
    old, model = old_construct(old_spec, "cpu"), ms.construct(spec["model"], spec, "cpu")
    assert model_digest(old) == model_digest(model)
    x = torch.tensor([[1, 2, 3, 4]])
    assert torch.equal(old(x), model(x))
    old_optimizer, optimizer = [optimizer_for(m, 0.003, 0.1) for m in (old, model)]
    expected = update(old, old_optimizer, records(), spec, "cpu", 0)
    with ms.adapter(spec, tmp_path):
        actual = st.update(model, optimizer, records(), spec, "cpu", 0)
    assert expected == actual
    assert model_digest(old) == model_digest(model)


def test_sparse_total_and_active_parameters_have_separate_dense_controls():
    ledgers = {
        arm: ms.manifest(ms.construct(options(arm)["model"], options(arm), "cpu"))
        for arm in ms.ARMS
    }
    assert ledgers["M4"]["unique_parameters"] > ledgers["D4"]["unique_parameters"]
    assert (
        abs(ledgers["M4"]["unique_parameters"] - ledgers["W4"]["unique_parameters"])
        < 0.03 * ledgers["M4"]["unique_parameters"]
    )
    assert ledgers["M4"]["active_unique_parameters"] < ledgers["W4"]["active_unique_parameters"]
    common = [row["initialization"]["shared_tensor_sha256"] for row in ledgers.values()]
    base = {k: v for k, v in common[0].items() if ".mlp." not in k}
    assert base == common[1] == common[2]


def test_new_fact_exposure_is_identical_and_adapter_restores_after_exception(tmp_path):
    data = fixture_data()
    plans = [ms.training_plan(data, options(arm))[1] for arm in ms.ARMS]
    assert all(np.array_equal(plans[0], plan) for plan in plans[1:])
    originals = {
        key: getattr(st, key) for key in ("construct", "training_plan", "update", "save_checkpoint")
    }
    with pytest.raises(RuntimeError), ms.adapter(options(), tmp_path):
        raise RuntimeError("injected failure")
    assert all(getattr(st, key) is value for key, value in originals.items())


@pytest.mark.parametrize("arm", ["M4", "W4"])
def test_cost_and_optimizer_restore_use_actual_sparse_or_wide_path(tmp_path, arm):
    spec = options(arm)
    model = ms.construct(spec["model"], spec, "cpu")
    optimizer = optimizer_for(model, 0.003, 0.1)
    with ms.adapter(spec, tmp_path):
        result = st.update(model, optimizer, records(), spec, "cpu", 0)
        saved_model, saved_optimizer = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
        )
        st.update(model, optimizer, records(), spec, "cpu", 0)
        restored = ms.construct(spec["model"], spec, "cpu")
        restored.load_state_dict(saved_model)
        restored_optimizer = optimizer_for(restored, 0.003, 0.1)
        restored_optimizer.load_state_dict(saved_optimizer)
        st.update(restored, restored_optimizer, records(), spec, "cpu", 0)
    assert model_digest(restored) == model_digest(model)
    assert (
        result["estimated_matmul_training_flops"]
        == result["compute"]["estimated_total_training_matmul_flops"]
    )
    if arm == "M4":
        assert result["router"]["dropped_assignments"] == 0
        assert result["router"]["replay_assignment_count_error"] == 0
        assert result["routing_prepass_executed_input_tokens"] > 0
        assert (
            result["compute"]["selected_expert_assignments"]
            == 4 * sum(len(r["encoded"]["input"]) for r in records()) * 2
        )
    else:
        dense = ms.construct(options("D4")["model"], options("D4"), "cpu")
        normal = update(dense, optimizer_for(dense, 0.003, 0.1), records(), spec, "cpu", 0)
        assert result["estimated_matmul_training_flops"] > normal["estimated_matmul_training_flops"]
