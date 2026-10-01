"""Scientific contracts: intervention Jacobians, supervision, and real transfer."""

import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_model import Attention, CausalLM, ModelConfig
from llm_memory_editability.bios_path_transfer import (
    ARMS,
    ROOT,
    assess,
    attention_surrogate,
    cases,
    example_data,
    measure,
    path_mode,
    relative,
    run_case,
    subset,
    verify_receipt,
)


@pytest.fixture
def tiny():
    torch.manual_seed(13)
    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(31, width=8, layers=3, heads=2, context=12)).double()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "blocks.0.mlp.down.weight")
    tokens = torch.tensor([[1, 7, 8, 2, 9, 3], [1, 6, 8, 2, 5, 3]])
    data = {
        "tokens": tokens,
        "prompts": tokens[:, :5].clone(),
        "lengths": torch.tensor([4, 4]),
        "positions": torch.tensor([[3, 4], [3, 4]]),
        "labels": torch.tensor([[9, 3], [5, 3]]),
    }
    return model, data


@pytest.mark.parametrize("arm", ARMS)
def test_forward_preservation_and_gradient_reconstruction(tiny, arm):
    model, data = tiny
    original = model(data["tokens"], data["positions"]).detach()
    measured = measure(model, data, 0, arm)
    assert torch.equal(original, measured["logits"])
    assert measured["reconstruction_error"] < 1e-12
    assert measured["decomposition_error"] < 1e-12
    # Per-example factors reconstruct individual autograd, not just the sum.
    for index in range(2):
        one = measure(model, subset(data, slice(index, index + 1)), 0, arm)
        torch.testing.assert_close(one["g"][0], measured["g"][index], rtol=1e-10, atol=1e-12)


def test_no_cross_is_the_true_diagonal_jacobian():
    torch.manual_seed(21)
    module = Attention(ModelConfig(23, width=8, layers=2, heads=2)).double()
    for p in module.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
    original = torch.autograd.functional.jacobian(module, x)
    actual = torch.autograd.functional.jacobian(
        lambda value: attention_surrogate(module, value, "no_cross"), x
    )
    expected = original.clone()
    for target in range(3):
        for source in range(3):
            if target != source:
                expected[:, target, :, :, source, :] = 0
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert (original - expected).norm() > 0.01


def test_fixed_qk_is_value_path_only():
    torch.manual_seed(22)
    module = Attention(ModelConfig(23, width=8, heads=2)).double()
    for p in module.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
    q, k, v = module.qkv(x).reshape(1, 3, 3, 2, 4).unbind(2)
    expected = module.proj(
        F.scaled_dot_product_attention(
            q.detach().transpose(1, 2),
            k.detach().transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        )
        .transpose(1, 2)
        .reshape(1, 3, 8)
    )
    actual = attention_surrogate(module, x, "fixed_qk")
    upstream = torch.randn_like(actual)
    g_expected = torch.autograd.grad((expected * upstream).sum(), x, retain_graph=True)[0]
    g_actual = torch.autograd.grad((actual * upstream).sum(), x)[0]
    torch.testing.assert_close(g_actual, g_expected, rtol=1e-10, atol=1e-12)


def test_cross_position_cut_eliminates_unsupervised_gradient(tiny):
    model, data = tiny
    full = measure(model, data, 0, "full")
    cut = measure(model, data, 0, "no_cross")
    assert full["unsupervised"].norm() > 1e-9
    assert torch.count_nonzero(cut["delta"][~cut["mask"]]) == 0
    assert cut["supervised"].norm() > 0


def test_mlp_stop_keeps_residual_and_restores_methods(tiny):
    model, data = tiny
    x = torch.randn(2, 6, 8, dtype=torch.float64, requires_grad=True)
    block = model.blocks[1]
    original = block.mlp.forward
    with path_mode(model, 0, "no_mlp"):
        residual = x + block.mlp(block.ln2(x))
        actual = torch.autograd.grad(residual.sum(), x)[0]
        torch.testing.assert_close(actual, torch.ones_like(x), rtol=0, atol=0)
    assert block.mlp.forward == original
    with pytest.raises(RuntimeError), path_mode(model, 0, "fixed_qk"):
        raise RuntimeError("restore even on failure")
    assert measure(model, data, 0, "full")["reconstruction_error"] < 1e-12


def test_fixed_attention_two_layer_formula():
    torch.manual_seed(23)
    r = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 5, dtype=torch.float64)
    a = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    e = torch.randn(5, dtype=torch.float64)
    h = r[-1] + b @ (a @ r)
    actual = torch.autograd.grad(h @ e, r)[0]
    expected = a[:, None] * (b.T @ e)[None]
    expected[-1] += e
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert expected[:-1].norm() > 0


def test_actual_step_uses_true_evaluation_gradient(tiny):
    model, data = tiny
    baseline = measure(model, data, 0, "full")
    source = measure(model, subset(data, slice(0, 1)), 0, "fixed_qk")
    w = model.blocks[0].mlp.down.weight
    parent = w.detach().clone()
    errors = []
    for eta in (1e-4, 5e-5):
        with torch.no_grad():
            w.copy_(parent - eta * source["g"][0])
        after = assess(model, data)[0]
        predicted = -eta * (baseline["g"] * source["g"][0]).sum((-1, -2))
        errors.append((after - baseline["loss"] - predicted).norm().item())
        with torch.no_grad():
            w.copy_(parent)
    assert 0 < errors[1] < 0.35 * errors[0]
    assert relative(w, parent) == 0


@pytest.mark.parametrize("world_id", [0, 1])
def test_case_selection_is_paired_and_heldout(world_id):
    world = make_cross_world(world_id, ROOT / "data/bios-organization-v1")
    selected = cases(world)
    assert len(selected) == 18
    for index in range(0, 18, 3):
        root, coherent, conflict = selected[index : index + 3]
        assert coherent["ids"] == conflict["ids"]
        assert coherent["targets"][0] != conflict["targets"][0]
        assert coherent["targets"][1:] == conflict["targets"][1:]
        for case in (root, coherent, conflict):
            assert np.isin(case["ids"][1:3], world.heldout_ids).all()
            assert case["ids"][0] not in case["ids"][1:]
            assert len(set(case["ids"])) == 6
            assert case["targets"][-2:] == case["old_targets"][-2:]
            data = example_data(world, case, torch.device("cpu"))
            assert data["labels"][:, 0].tolist() == case["targets"]


def test_saved_single_step_matrix_and_free_generation(tmp_path):
    world = make_cross_world(0, ROOT / "data/bios-organization-v1")
    torch.manual_seed(39)
    model = CausalLM(ModelConfig(world.vocab_size, width=8, layers=3, heads=2))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "blocks.0.mlp.down.weight")
    config = {
        "target_layer": 0,
        "target_parameter": "blocks.0.mlp.down.weight",
        "arms": ARMS,
        "forward_tolerance": 2e-6,
        "gradient_tolerance": 5e-5,
        "scales": ["shared_lr", "matched_norm"],
        "step_fractions": [0.0001, 0.001],
    }
    parent = {name: p.detach().clone() for name, p in model.named_parameters()}
    run_case(model, world, cases(world)[0], config, tmp_path)
    verify_receipt(tmp_path)
    saved = json.loads((tmp_path / "measurements.json").read_text())
    assert len(saved["updates"]) == 4 * 2 * 2 * 6
    assert len(saved["diagnostics"]) == 4 * 6
    for row in saved["updates"]:
        assert np.isfinite(row["predicted_change"])
        assert row["after"] - row["before"] == row["observed_change"]
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parent[name])
