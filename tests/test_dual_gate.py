"""Contracts that distinguish gate placement, supervision and baseline reuse."""

import copy
import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.dual_gate import VARIANTS, DualGateGPT, gate_parameter_count
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
launcher = importlib.import_module("launch_dual_gate")
runner = importlib.import_module("run_dual_gate")


def model(variant):
    config = GPT2Config(
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
    torch.manual_seed(71)
    return DualGateGPT(config, repeats=3, discoloop=True, gate_variant=variant).double()


@pytest.mark.parametrize("variant", VARIANTS)
def test_parameter_counts_no_rng_consumption_and_unique_physical_weights(variant):
    current = model(variant)
    actual_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    reference = SupervisedAlignmentGPT(current.config, repeats=3, discoloop=True).double()
    assert torch.equal(actual_rng, torch.get_rng_state())
    assert all(
        torch.equal(p, reference.state_dict()[n])
        for n, p in current.state_dict().items()
        if not n.startswith("gate_")
    )
    assert sum(
        p.numel() for n, p in current.named_parameters() if n.startswith("gate_")
    ) == gate_parameter_count(variant, 12)
    assert len(list(current.parameters())) == len({id(p) for p in current.parameters()})


@pytest.mark.parametrize("variant", ("scalar", "dual", "vector_injection"))
def test_initial_forward_and_every_backbone_gradient_equal_fixed_discoloop(variant):
    current = model(variant)
    reference = SupervisedAlignmentGPT(current.config, repeats=3, discoloop=True).double()
    reference.load_state_dict(
        {n: p for n, p in current.state_dict().items() if not n.startswith("gate_")}
    )
    tokens = torch.tensor([[2, 4, 7, 8], [3, 2, 9, 1]])
    a, b = reference(tokens), current(tokens)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter.grad, current.get_parameter(name).grad), name
    gate_grads = [p.grad for n, p in current.named_parameters() if n.startswith("gate_")]
    assert all(torch.isfinite(g).all() for g in gate_grads)
    assert all(torch.count_nonzero(g) for g in gate_grads)


@pytest.mark.parametrize("variant", VARIANTS)
def test_boundary_matches_declared_formula_with_nonconstant_gates(variant):
    current = model(variant)
    for name, p in current.named_parameters():
        if name.startswith("gate_"):
            with torch.no_grad():
                p.copy_(torch.linspace(-0.3, 0.4, p.numel()).reshape(p.shape))
    h = torch.randn(2, 4, 12, dtype=torch.float64)
    logits = current.transformer.ln_f(h) @ current.transformer.wte.weight.T
    v = logits.softmax(-1) @ current.transformer.wte.weight
    q = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-6)
    if variant == "paper":
        gate = torch.sigmoid(
            (v * current.gate_paper_weight).sum(-1, keepdim=True) + current.gate_paper_bias
        )
        expected = h + gate * q
    else:
        u = torch.nn.functional.layer_norm(h, (12,), eps=current.config.layer_norm_epsilon)
        g = 2 * torch.sigmoid(
            (u * current.gate_u + q * current.gate_q).sum(-1, keepdim=True) + current.gate_bias
        )
        if variant == "scalar":
            expected = h + g * q
        else:
            r = 2 * torch.sigmoid(current.gate_r_u * u + current.gate_r_q * q + current.gate_r_bias)
            expected = r * h + g * q if variant == "dual" else h + g * (r * q)
    assert torch.allclose(current._discoloop_boundary(h), expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("variant", VARIANTS)
def test_boundary_autograd_and_causal_prefix(variant):
    current = model(variant).eval()
    h = torch.randn(1, 1, 12, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(current._discoloop_boundary, (h,), atol=1e-5)
    a = torch.tensor([[2, 3, 4, 5]])
    b = torch.tensor([[2, 3, 9, 8]])
    assert torch.equal(current(a)[:, :2], current(b)[:, :2])


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_boundary_after_final_round_and_disabled_channel(variant):
    current = model(variant)
    calls = []
    original = current._discoloop_boundary
    current._discoloop_boundary = lambda x: (calls.append(1), original(x))[1]
    current(torch.tensor([[2, 3, 4]]))
    assert len(calls) == 2
    current.discoloop_alpha = 0
    h = torch.randn(1, 2, 12, dtype=torch.float64)
    assert current._discoloop_boundary(h) is h


@pytest.mark.parametrize("variant", VARIANTS)
def test_optimizer_gate_decay_zero_and_checkpoint_reload(variant):
    current = model(variant)
    optimizer = runner.optimizer_for(current, 0.001, 0.1)
    assert optimizer.defaults["capturable"]
    gates = {id(p) for n, p in current.named_parameters() if n.startswith("gate_")}
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(current.parameters()))
    for group in optimizer.param_groups:
        if any(id(p) in gates for p in group["params"]):
            assert group["weight_decay"] == 0
    clone = model(variant)
    clone.load_state_dict(current.state_dict(), strict=True)
    x = torch.tensor([[2, 3, 4]])
    assert torch.equal(clone(x), current(x))


def test_plan_five_conditions_only_four_new_runs_and_no_novel_fact_budget():
    config = json.loads((ROOT / "configs/dual-gate-multihop-v1.json").read_text())
    launcher.validate(config)
    assert config["budget"]["novel_fact_updates"] == 0
    assert config["budget"]["reused_fixed_baseline_updates"] == 0
    assert len(config["arms"]) == 4
    invalid = copy.deepcopy(config)
    invalid["arms"]["fixed"] = {"architecture": "discoloop", "unique_layers": 1, "repeats": 4}
    with pytest.raises(ValueError):
        launcher.validate(invalid)
    assert math.isclose(2 * torch.sigmoid(torch.tensor(0.0)).item(), 1)


def test_resource_reservations_block_foreign_tasks():
    rows = {9: {"uuid": "u9"}, 8: {"uuid": "u8"}}
    foreign = {
        "Name": "/other",
        "State": {"Pid": 123},
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["9"]}]},
    }
    blocked = launcher.blocked_gpus(rows, [("u8", "456")], [foreign], "dual-gate-multihop-v1-")
    assert blocked[9] and blocked[8]
