import torch

from llm_memory_editability.qwen_constraints import (
    feature_basis,
    functional_projection,
    local_gradients,
    module_projection,
    route_output,
    select_records,
    update_scale,
)


def test_function_preservation_can_allow_module_changes():
    # Readout sees only h_2. Preserving full h removes a useful h_1 direction.
    target = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    keep = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    projected, stats = functional_projection(target, keep, 1e-10)
    keys = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    basis, _ = feature_basis(keys, 1e-5)
    strict = module_projection(target.reshape(2, 2), basis).flatten()
    assert torch.allclose(keep @ projected, torch.zeros(1, dtype=torch.float64))
    assert stats["retained_energy"] == 1.0
    assert strict.norm() < projected.norm()
    assert torch.allclose(strict.reshape(2, 2) @ keys.T, torch.zeros(2, 1).double())


def test_all_positions_chain_rule_includes_silu_gate():
    torch.manual_seed(5)
    x = torch.randn(5, 3, dtype=torch.float64)
    a = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    g = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    d = torch.randn(5, 3, dtype=torch.float64)
    up, gate = x @ a.T, x @ g.T
    q = (((up * torch.nn.functional.silu(gate)) @ b.T) * d).sum()
    truth = torch.autograd.grad(q, (b, a, g))
    computed = local_gradients(x, up, gate, b, d)
    for group, expected in zip(["down", "up", "gate"], truth, strict=True):
        torch.testing.assert_close(computed[group], expected)


def test_forward_routes_partition_real_delta_and_restore_zero_update():
    original = torch.randn(2, 4, 3)
    current = original + torch.randn_like(original)
    attention = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    last = route_output(current, original, attention, "last")
    earlier = route_output(current, original, attention, "earlier")
    torch.testing.assert_close(
        (last + earlier - 2 * original)[attention.bool()], (current - original)[attention.bool()]
    )
    for route in ["all", "last", "earlier"]:
        torch.testing.assert_close(route_output(original, original, attention, route), original)


def test_equal_norm_and_target_comparisons_and_cap():
    g = torch.tensor([2.0, 2.0])
    d = torch.tensor([2.0, 0.0])
    scale, capped = update_scale(d, g, 0.1, 10.0, 0.1, "equal_norm")
    assert not capped
    assert abs(float((scale * d).norm()) - 0.1 / float(g.norm())) < 1e-7
    scale, capped = update_scale(d, g, 0.1, 10.0, 0.1, "equal_target")
    assert not capped
    assert abs(float(g @ (scale * d)) - 0.1) < 1e-7
    scale, capped = update_scale(d, g, 100.0, 10.0, 0.001, "equal_target")
    assert capped
    assert abs(float((scale * d).norm()) - 0.01) < 1e-7


def test_selection_is_input_order_invariant():
    records = [{"case_id": i} for i in range(50)]
    assert select_records(records, 12, 3, "E") == select_records(records[::-1], 12, 3, "E")
