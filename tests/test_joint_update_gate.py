"""Scientific contracts for the new placement of the update gate."""

import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.joint_update_gate import JointUpdateGateGPT
from llm_memory_editability.paper_update_gate import PaperUpdateGateGPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_joint_update_gate as runner  # noqa: E402


def make(cls=JointUpdateGateGPT, repeats=4, mode="token"):
    torch.manual_seed(17)
    config = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        attn_pdrop=0,
        resid_pdrop=0,
        embd_pdrop=0,
    )
    return cls(config, repeats, update_mode=mode, initial_update=0.95).double()


def test_initial_parameters_rng_and_parameter_count_match_existing_gate():
    old = make(PaperUpdateGateGPT)
    rng = torch.get_rng_state()
    new = make()
    assert torch.equal(rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())


def test_independent_formula_and_difference_from_continuous_only_gate():
    new, old = make(), make(PaperUpdateGateGPT)
    with torch.no_grad():
        new.gate_update_weight.copy_(torch.linspace(-0.4, 0.6, 12))
        new.gate_update_bias.fill_(0.3)
        new.gate_paper_weight.copy_(torch.linspace(0.7, -0.5, 12))
        new.gate_paper_bias.fill_(0.1)
    old.load_state_dict(new.state_dict())
    x, h = (torch.randn(2, 3, 12, dtype=torch.float64) for _ in range(2))
    v = (new.transformer.ln_f(h) @ new.transformer.wte.weight.T).softmax(-1)
    v = v @ new.transformer.wte.weight
    q = v / torch.sqrt(v.square().mean(-1, keepdim=True) + 1e-6)
    g = torch.sigmoid((v * new.gate_paper_weight).sum(-1, keepdim=True) + new.gate_paper_bias)
    u = new.update_gate(h)
    actual = new._update_boundary(x, h)
    torch.testing.assert_close(actual, x + u * (h - x) + u * g * q, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        old._update_boundary(x, h) - actual, (1 - u) * g * q, atol=1e-14, rtol=1e-14
    )
    assert not torch.allclose(old._update_boundary(x, h), actual)


@pytest.mark.parametrize("gate", (0.0, 1.0))
def test_exact_closed_gate_and_full_gate_endpoint(gate):
    model = make()
    x, h = (torch.randn(2, 3, 12, dtype=torch.float64) for _ in range(2))
    model.update_gate = lambda state: state.new_full((*state.shape[:-1], 1), gate)
    actual = model._update_boundary(x, h)
    if gate == 0:
        assert torch.equal(actual, x)
    else:
        torch.testing.assert_close(actual, model._discoloop_boundary(h), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("repeats", (1, 4))
def test_final_round_raw_readout_causal_and_gate_gradients(repeats):
    model = make(repeats=repeats)
    calls = []
    original = model._discoloop_boundary
    model._discoloop_boundary = lambda h: (calls.append(h), original(h))[1]
    tokens = torch.tensor([[2, 3, 4, 5]])
    logits, _, final = model.forward_with_states(tokens)
    assert len(calls) == repeats - 1
    assert torch.equal(logits, model.transformer.ln_f(final) @ model.transformer.wte.weight.T)
    changed = model(torch.tensor([[2, 3, 9, 8]]))
    assert torch.equal(logits[:, :2], changed[:, :2])
    logits.square().mean().backward()
    for name, p in model.named_parameters():
        if name.startswith("gate_") and repeats == 1:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            if name.startswith("gate_"):
                assert torch.count_nonzero(p.grad), name


def test_full_mode_exactly_matches_existing_paper_path():
    a, b = make(mode="full"), make(PaperUpdateGateGPT, mode="full")
    tokens = torch.tensor([[2, 3, 4, 5]])
    assert torch.equal(a(tokens), b(tokens))


def test_frozen_recipe_and_rejection_of_unpaired_changes():
    cfg = json.loads((ROOT / "configs/joint-update-gate-multihop-v1.json").read_text())
    old = json.loads((ROOT / cfg["paired_recipe"]["config"]).read_text())
    runner.validate_config(cfg, old)
    cfg["training"]["stream_seed"] += 1
    with pytest.raises(ValueError, match="training horizon"):
        runner.validate_config(cfg, old)


def test_adapter_restores_all_global_runner_hooks():
    names = ("initialize", "accounting", "optimizer_for", "preflight")
    before = {n: getattr(runner.native, n) for n in names}
    with runner.installed():
        assert runner.original.PaperUpdateGateGPT is JointUpdateGateGPT
        assert runner.native.initialize is runner.initialize
    assert runner.original.PaperUpdateGateGPT is PaperUpdateGateGPT
    assert all(getattr(runner.native, n) is v for n, v in before.items())
