"""Scientific contracts for supervision mass and the attention intervention."""

import numpy as np
import pytest
import torch

from llm_memory_editability import bios_mechanism_edit as original
from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_followup import (
    ROOT_MASSES,
    load_config,
    mixture_adapter,
    mixture_spec,
    uniform_first_layer,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig


@pytest.mark.parametrize("chain", [0, 1])
@pytest.mark.parametrize("kind", ["coherent", "exception"])
def test_mixture_changes_only_weights_and_preserves_expected_total(chain, kind):
    world = make_cross_world(0)
    baseline = original.make_arm(world, chain, kind)
    for tag, mass in ROOT_MASSES.items():
        spec = mixture_spec(world, chain, f"{kind}-{tag}")
        for key in baseline.keys() - {"S_weights"}:
            np.testing.assert_array_equal(spec[key], baseline[key])
        weights, roots = spec["S_weights"], spec["S_root_mask"]
        assert weights.mean() == pytest.approx(1)
        assert weights[roots].sum() / len(weights) == pytest.approx(mass)
    if kind == "exception":
        np.testing.assert_array_equal(
            mixture_spec(world, chain, "exception-root500")["S_weights"],
            original.make_arm(world, chain, "class-balanced-exception")["S_weights"],
        )


def test_adapter_restores_original_functions_even_on_failure():
    make_arm, sources = original.make_arm, original.source_hashes
    with pytest.raises(RuntimeError), mixture_adapter():
        assert original.make_arm is mixture_spec
        raise RuntimeError("test restoration")
    assert original.make_arm is make_arm
    assert original.source_hashes is sources


def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(27)
    return CausalLM(ModelConfig(31, width=8, layers=2, heads=2, context=12)).double()


def test_uniform_attention_equals_zero_qk_and_preserves_other_layers():
    model = tiny_model()
    attention = model.blocks[0].attention
    original_forward = attention.forward
    downstream = model.blocks[1].attention.forward
    x = torch.randn(2, 6, 8, dtype=torch.float64)
    with torch.no_grad():
        attention.qkv.weight[:16].zero_()
        attention.qkv.bias[:16].zero_()
    expected = attention(x)
    with uniform_first_layer(model):
        torch.testing.assert_close(attention(x), expected, rtol=1e-12, atol=1e-12)
        assert model.blocks[1].attention.forward == downstream
    assert attention.forward == original_forward


def test_uniform_attention_is_causal_and_retains_downstream_backpropagation():
    model = tiny_model()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = tokens.clone()
    changed[:, 4:] = 9
    with uniform_first_layer(model):
        first, second = model(tokens), model(changed)
        torch.testing.assert_close(first[:, :4], second[:, :4], rtol=0, atol=0)
        loss = first[:, 3].square().sum()
        loss.backward()
    qkv = model.blocks[0].attention.qkv.weight.grad
    assert torch.count_nonzero(qkv[:16]) == 0
    assert torch.count_nonzero(qkv[16:]) > 0
    assert model.blocks[1].mlp.down.weight.grad.norm() > 0
    assert model.blocks[0].attention.proj.weight.grad.norm() > 0


def test_declared_matrix_counts_and_coherent_people_are_paired():
    config = load_config()
    assert len(config["edit"]["arms"]) * 12 * 2 == 120
    assert len(config["gradient"]["states"]) * 4 * 3 == 36
    world = make_cross_world(0)
    coherent = mixture_spec(world, 0, "coherent-root250")
    exception = mixture_spec(world, 0, "exception-root250")
    np.testing.assert_array_equal(coherent["D_probe_conflict"], exception["D_probe_conflict"])
    np.testing.assert_array_equal(coherent["S"], exception["S"])
    np.testing.assert_array_equal(coherent["replay"], exception["replay"])
