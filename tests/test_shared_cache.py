"""Scientific contracts for cross-recursion KV sharing."""

import copy

import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.latent_scaling import model_digest
from llm_memory_editability.shared_cache import ARMS, SharedCacheGPT, executed_flops


def model(arm="shared_full", repeats=3):
    torch.manual_seed(51)
    return SharedCacheGPT(
        ModelConfig(vocab_size=39, width=16, layers=2, heads=2, context=16), repeats, arm=arm
    )


@pytest.mark.parametrize("arm", ARMS)
def test_parameter_initialization_and_R1_equal_historical_model(arm):
    actual = model(arm)
    torch.manual_seed(51)
    reference = LoopGPT(actual.config, 3, 0.0, "legacy_unique")
    assert model_digest(actual) == model_digest(reference)
    x = torch.tensor([[2, 3, 4, 5, 6]])
    torch.testing.assert_close(actual(x, repeats=1), reference(x, repeats=1), rtol=0, atol=0)
    if arm == "local":
        torch.testing.assert_close(actual(x), reference(x), rtol=0, atol=0)


@pytest.mark.parametrize("arm", ARMS)
def test_future_tokens_do_not_leak_through_either_cache(arm):
    m = model(arm).eval()
    a = torch.tensor([[2, 3, 4, 5, 6, 7]])
    b = torch.tensor([[2, 3, 4, 18, 19, 20]])
    torch.testing.assert_close(m(a)[:, :3], m(b)[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(m(a)[:, :3], m(a[:, :3]), rtol=1e-5, atol=1e-6)


def test_detach_preserves_forward_and_changes_parameter_gradient():
    normal = model()
    detached = copy.deepcopy(normal)
    detached.memory_arm = "shared_detached"
    x = torch.tensor([[2, 3, 4, 5]])
    torch.testing.assert_close(normal(x), detached(x), rtol=0, atol=0)
    normal(x)[:, -1].square().sum().backward()
    detached(x)[:, -1].square().sum().backward()
    a, b = normal.blocks[0].attention.qkv.weight.grad, detached.blocks[0].attention.qkv.weight.grad
    assert torch.linalg.vector_norm(a - b) > 1e-5
    assert all(p.grad is not None for p in detached.parameters())


def test_window_counts_preceding_tokens_and_current_entry():
    m = model("shared_window")
    assert torch.equal(m.local_window_mask[5].nonzero().flatten(), torch.tensor([3, 4, 5]))
    assert torch.equal(m.shared_causal_mask[5].nonzero().flatten(), torch.arange(6))


def test_disable_and_duplicate_controls_and_cache_is_per_call():
    m = model().eval()
    reference = model("local").eval()
    x = torch.tensor([[2, 3, 4, 5]])
    m.disable_shared = True
    torch.testing.assert_close(m(x), reference(x), rtol=1e-5, atol=1e-6)
    m.disable_shared = False
    m.duplicate_current = True
    torch.testing.assert_close(m(x), reference(x), rtol=1e-5, atol=1e-6)
    m.duplicate_current = False
    expected = m(x).detach()
    m(torch.tensor([[9, 10, 11, 12]]))
    torch.testing.assert_close(m(x), expected, rtol=0, atol=0)


def test_shared_attention_increases_executed_shape_budget_even_with_short_window():
    c = model().config
    ordinary = executed_flops(c, 3, 192, 9, "local")
    shared = executed_flops(c, 3, 192, 9, "shared_full")
    assert shared - ordinary == 3 * 4 * 192 * 2 * 2 * 9**2 * 16
    assert shared == executed_flops(c, 3, 192, 9, "shared_window")
    assert shared == executed_flops(c, 3, 192, 9, "shared_detached")
