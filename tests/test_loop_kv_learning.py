"""Causal access, ordinary initialization, historical equivalence and cost."""

import copy

import numpy as np
import pytest
import torch
from test_loop_learning import spec as tiny
from test_sequential_transfer import fixture_data
from test_sequential_transfer import spec as data_spec

from llm_memory_editability import loop_kv_learning_train as trainer
from llm_memory_editability import sequential_transfer as st
from llm_memory_editability.grokking_reproduction import model_digest
from llm_memory_editability.loop_kv_learning import (
    architecture_manifest,
    construct,
    extra_attention_flops,
)
from llm_memory_editability.loop_learning import construct as historical_construct
from llm_memory_editability.loop_learning_train import training_plan
from llm_memory_editability.realworld_composition import optimizer_for, update


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def options(arm="loop_shared_kv"):
    base = tiny("shared")
    base.update(arm=arm, repeats=2, unique_layers=2, initialization_reference_layers=8)
    if arm == "standard":
        base.update(base_unique_blocks=4, unique_layers=4, repeats=1)
    return base


def test_standard_layers_are_independent_random_draws():
    model = construct(options("standard"), "cpu")
    weights = [block.attn.c_attn.weight for block in model.transformer.h]
    assert len({w.data_ptr() for w in weights}) == 4
    assert all(not torch.equal(weights[i], weights[j]) for i in range(4) for j in range(i))
    assert architecture_manifest(model)["executed_blocks"] == 4


def test_disabling_access_exactly_restores_historical_forward_and_gradient():
    spec = options()
    model = construct(spec, "cpu")
    old = historical_construct({**spec, "arm": "shared"}, "cpu")
    assert model_digest(model) == model_digest(old)
    model.disable_shared = True
    tokens = torch.tensor([[2, 5, 8, 1, 3], [7, 3, 9, 2, 1]])
    positions = torch.tensor([[1, 3], [2, 4]])
    torch.testing.assert_close(model(tokens, positions), old(tokens, positions), rtol=0, atol=0)
    model(tokens).sum().backward()
    old(tokens).sum().backward()
    for p, q in zip(model.parameters(), old.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_duplicate_current_identity_and_position_selection():
    model = construct(options(), "cpu").eval()
    tokens = torch.tensor([[2, 5, 8, 1, 3]])
    model.duplicate_current = True
    duplicated = model(tokens)
    model.disable_shared = True
    torch.testing.assert_close(duplicated, model(tokens), rtol=2e-5, atol=2e-6)
    model.disable_shared, model.duplicate_current = False, False
    positions = torch.tensor([[0, 2, 4]])
    torch.testing.assert_close(model(tokens, positions), model(tokens)[:, [0, 2, 4]])


def test_both_kv_branches_are_causal_and_call_local():
    model = construct(options(), "cpu").eval()
    tokens = torch.tensor([[2, 5, 8, 1, 3], [7, 3, 9, 2, 1]])
    before = model(tokens)
    changed = tokens.clone()
    changed[:, 3:] = torch.tensor([[12, 13], [14, 15]])
    torch.testing.assert_close(before[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    model(torch.tensor([[10, 11, 12]]))
    torch.testing.assert_close(before, model(tokens), rtol=0, atol=0)
    assert not torch.allclose(before, construct(options("loop_local"), "cpu")(tokens))


def test_checkpoint_optimizer_continuation_and_shared_cost(tmp_path):
    spec = {**options(), "microbatch_size": 2}
    records = [{"encoded": {"input": [2, 5, 8, 1, 3], "prefix": [2, 5], "target": [8, 1]}}] * 3
    model = construct(spec, "cpu")
    optimizer = optimizer_for(model, 0.003, 0.1)
    update(model, optimizer, records, spec, "cpu", 0)
    path = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    expected = update(model, optimizer, records, spec, "cpu", 0)
    restored = construct(spec, "cpu")
    state = torch.load(path, weights_only=False)
    restored.load_state_dict(state["model"])
    restored_optimizer = optimizer_for(restored, 0.003, 0.1)
    restored_optimizer.load_state_dict(state["optimizer"])
    update(restored, restored_optimizer, records, spec, "cpu", 0)
    assert model_digest(restored) == model_digest(model)
    extra = 12 * 8 * 2 * 3 * 8**2
    assert extra_attention_flops(model, records, 2) == extra
    assert architecture_manifest(model)["attention_pair_multiples_per_sequence_square"] == 6
    with trainer.adapter(spec, tmp_path / "adapter"):
        another = construct(spec, "cpu")
        another.load_state_dict(copy.deepcopy(state["model"]))
        opt = optimizer_for(another, 0.003, 0.1)
        opt.load_state_dict(state["optimizer"])
        actual = st.update(another, opt, records, spec, "cpu", 0)
    assert (
        actual["estimated_matmul_training_flops"]
        == expected["estimated_matmul_training_flops"] + extra
    )


def test_new_models_have_identical_data_exposure_and_adapter_restores(tmp_path):
    spec = {**data_spec(), "stage_b_steps": 4, "replay_source": "full_stage_a"}
    data = fixture_data()
    before = copy.deepcopy(data)
    original_plan = training_plan(data, spec)[1]
    for arm in ("standard", "loop_shared_kv", "shared"):
        np.testing.assert_array_equal(training_plan(data, {**spec, "arm": arm})[1], original_plan)
    assert data == before
    original = {k: getattr(st, k) for k in ("construct", "training_plan", "update")}
    with pytest.raises(RuntimeError), trainer.adapter(spec, tmp_path):
        raise RuntimeError("injected")
    assert all(getattr(st, k) is v for k, v in original.items())


def test_invalid_standard_rounds_rejected():
    with pytest.raises(ValueError, match="independent layers once"):
        construct({**options("standard"), "repeats": 2}, "cpu")
