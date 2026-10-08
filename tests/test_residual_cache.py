"""Mechanism checks for shared controllers, masks, scaling and historical identity."""

import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.residual_cache import ResidualCacheGPT, make_optimizer
from llm_memory_editability.shared_cache import SharedCacheGPT


@pytest.mark.parametrize("memory", ["local", "shared_full"])
def test_single_path_is_historical(memory):
    cfg = ModelConfig(vocab_size=40, width=16, layers=2, heads=2, context=8)
    torch.manual_seed(7)
    old = SharedCacheGPT(cfg, 4, arm=memory)
    torch.manual_seed(7)
    new = ResidualCacheGPT(cfg, 4, arm=memory)
    assert list(old.state_dict()) == list(new.state_dict())
    tokens = torch.tensor([[2, 3, 4, 5]])
    torch.testing.assert_close(old(tokens), new(tokens), rtol=0, atol=0)
    old(tokens).square().sum().backward()
    new(tokens).square().sum().backward()
    for a, b in zip(old.parameters(), new.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    new.diagnostic_records = []
    torch.testing.assert_close(old(tokens), new(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["identity_mhc", "mhc"])
def test_mhc_causality_parameter_sharing_and_optimizer(kind):
    cfg = ModelConfig(vocab_size=40, width=16, layers=2, heads=2, context=8)
    model = ResidualCacheGPT(cfg, 4, arm="shared_full", residual_kind=kind)
    before = [id(p) for p in model.parameters()]
    tokens = torch.tensor([[2, 3, 4, 5]])
    changed = torch.tensor([[2, 3, 8, 9]])
    torch.testing.assert_close(model(tokens)[:, :2], model(changed)[:, :2], rtol=0, atol=0)
    model(tokens, repeats=8).square().mean().backward()
    assert before == [id(p) for p in model.parameters()]
    assert len(model.connections) == 4
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = make_optimizer(model, 1e-3, 0.01)
    no_decay = {
        id(p) for g in optimizer.param_groups if g["weight_decay"] == 0 for p in g["params"]
    }
    for name, parameter in model.named_parameters():
        if name.endswith(".res_bias"):
            assert id(parameter) in no_decay
    model.diagnostic_records = []
    with torch.no_grad():
        model(tokens)
    assert len(model.diagnostic_records) == 16
    if kind == "mhc":
        assert max(r["max_column_error"] for r in model.diagnostic_records) < 1e-4


def test_branch_scale_preserves_residual_and_changes_writes():
    cfg = ModelConfig(vocab_size=40, width=16, layers=1, heads=2, context=8)
    model = ResidualCacheGPT(cfg, 2, residual_kind="mhc", gamma=0.1)
    x = torch.randn(1, 3, 4, 16)
    connection = model.connections[0]
    _, _, matrix = connection.coefficients(x)
    from llm_memory_editability.parametric_architecture import residual_mix

    expected = residual_mix(x, matrix)
    actual = model._connection(x, lambda z: torch.zeros_like(z), connection, 0, 0, "attention")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
