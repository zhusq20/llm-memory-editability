"""Contracts separating gate conditioning from the joint update itself."""

import copy
import json

import pytest
import run_candidate_update_gate as runner
import torch
from test_joint_update_gate import ROOT, make

from llm_memory_editability.candidate_update_gate import CandidateUpdateGateGPT
from llm_memory_editability.joint_update_gate import JointUpdateGateGPT


def test_initial_forward_and_backbone_gradients_match_but_gate_gradient_changes():
    old, new = make(JointUpdateGateGPT), make(CandidateUpdateGateGPT)
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    tokens = torch.tensor([[2, 3, 4, 5], [4, 2, 3, 6]])
    a, b = old(tokens), new(tokens)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, p in old.named_parameters():
        q = dict(new.named_parameters())[name]
        assert p.grad is not None and q.grad is not None
        if name == "gate_update_weight":
            assert not torch.allclose(p.grad, q.grad, atol=1e-12, rtol=1e-9)
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-12)


def test_gate_reads_candidate_and_differentiates_through_it():
    model = make(CandidateUpdateGateGPT)
    with torch.no_grad():
        model.gate_update_weight.copy_(torch.linspace(-0.5, 0.8, 12))
    x = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    h = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
    c = model._discoloop_boundary(h)
    normalized = torch.nn.functional.layer_norm(c, (12,), eps=1e-5)
    u = torch.sigmoid(
        (normalized * model.gate_update_weight).sum(-1, keepdim=True) + model.gate_update_bias
    )
    expected = x + u * (c - x)
    actual = model._update_boundary(x, h)
    assert torch.equal(actual, expected)
    wrong = x + model.update_gate(h) * (c - x)
    assert not torch.allclose(actual, wrong)
    parameters = (h, model.gate_paper_weight, model.transformer.wte.weight)
    ga = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), parameters)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-12, rtol=1e-12)
        assert torch.isfinite(a).all() and torch.count_nonzero(a)


@pytest.mark.parametrize("repeats", (1, 4))
def test_final_readout_causality_and_single_reencoding_per_boundary(repeats):
    model = make(CandidateUpdateGateGPT, repeats=repeats)
    calls = []
    original = model._discoloop_boundary
    model._discoloop_boundary = lambda h: (calls.append(h), original(h))[1]
    tokens = torch.tensor([[2, 3, 4, 5]])
    logits, _, final = model.forward_with_states(tokens)
    assert len(calls) == repeats - 1
    assert torch.equal(logits, model.transformer.ln_f(final) @ model.transformer.wte.weight.T)
    assert torch.equal(logits[:, :2], model(torch.tensor([[2, 3, 9, 8]]))[:, :2])
    if repeats == 1:
        logits.square().sum().backward()
        assert all(p.grad is None for n, p in model.named_parameters() if n.startswith("gate_"))


@pytest.mark.parametrize("gate", (0.0, 1.0))
def test_gate_endpoints(gate):
    model = make(CandidateUpdateGateGPT)
    x, h = (torch.randn(2, 3, 12, dtype=torch.float64) for _ in range(2))
    model.update_gate = lambda c: c.new_full((*c.shape[:-1], 1), gate)
    expected = x if gate == 0 else model._discoloop_boundary(h)
    torch.testing.assert_close(model._update_boundary(x, h), expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("field", ("training", "evaluation_nodes", "gate_condition", "runtime"))
def test_paired_recipe_rejects_drift(field):
    cfg = json.loads((ROOT / "configs/candidate-update-gate-multihop-v1.json").read_text())
    old = json.loads((ROOT / cfg["paired_recipe"]["config"]).read_text())
    runner.validate_config(cfg, old)
    cfg = copy.deepcopy(cfg)
    if field == "training":
        cfg[field]["stream_seed"] += 1
    elif field == "evaluation_nodes":
        cfg[field].pop()
    elif field == "runtime":
        cfg[field]["image"] = "changed"
    else:
        cfg[field] = "H"
    with pytest.raises(ValueError):
        runner.validate_config(cfg, old)


def test_runner_restores_original_model_and_hooks():
    names = ("initialize", "accounting", "optimizer_for", "preflight")
    before = {n: getattr(runner.native, n) for n in names}
    original = runner.original.PaperUpdateGateGPT
    with runner.installed():
        assert runner.original.PaperUpdateGateGPT is CandidateUpdateGateGPT
        assert runner.native.initialize is runner.initialize
    assert runner.original.PaperUpdateGateGPT is original
    assert all(getattr(runner.native, n) is value for n, value in before.items())
