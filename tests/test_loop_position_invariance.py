"""Contracts for rotary-position loops: exact absolute reuse, relative invariance,
backbone pairing and per-arm recipes."""

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import learning_rate as original_rate
from llm_memory_editability.loop_native_handoff import HANDOFFS, LoopNativeHandoffGPT
from llm_memory_editability.loop_position_invariance import LoopPositionGPT, rotate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TOKENS = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5]])
POSITIONS = torch.tensor([[3, 4], [3, 4]])


def config():
    return GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=24,
        n_ctx=24,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )


def build(handoff, position, repeats=3):
    torch.manual_seed(71)
    return LoopPositionGPT(config(), repeats, handoff=handoff, position=position).double().eval()


def perturb(model, seed=5):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("handoff_"):
                parameter.add_(
                    torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                )


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_absolute_position_is_exactly_the_loop_native_model(handoff):
    current = build(handoff, "absolute")
    torch.manual_seed(71)
    reference = LoopNativeHandoffGPT(config(), 3, handoff=handoff).double().eval()
    assert list(current.state_dict()) == list(reference.state_dict())
    a, b = reference(TOKENS, POSITIONS), current(TOKENS, POSITIONS)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, parameter in reference.named_parameters():
        expected = parameter.grad
        actual = current.get_parameter(name).grad
        assert (expected is None and actual is None) or torch.equal(expected, actual), name


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_rope_backbone_matches_and_consumes_no_global_rng(handoff):
    current = build(handoff, "rope")
    after = torch.get_rng_state().clone()
    torch.manual_seed(71)
    reference = LoopNativeHandoffGPT(config(), 3, handoff=handoff).double()
    assert torch.equal(after, torch.get_rng_state())
    for name, tensor in reference.state_dict().items():
        assert torch.equal(current.state_dict()[name], tensor), name


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_rope_outputs_do_not_depend_on_a_global_position_shift(handoff):
    model = build(handoff, "rope")
    perturb(model)
    base = model.run(TOKENS, POSITIONS, 4, carry=False)[0]
    model.position_offset = 7
    shifted = model.run(TOKENS, POSITIONS, 4, carry=False)[0]
    torch.testing.assert_close(base, shifted, rtol=1e-9, atol=1e-10)


def test_rope_ignores_learned_positions_and_differs_from_absolute():
    rope, absolute = build("ema", "rope"), build("ema", "absolute")
    out = rope(TOKENS, POSITIONS)
    assert not torch.allclose(out, absolute(TOKENS, POSITIONS))
    out.square().mean().backward()
    assert rope.transformer.wpe.weight.grad is None
    with torch.no_grad():
        rope.transformer.wpe.weight.normal_()
    torch.testing.assert_close(rope(TOKENS, POSITIONS), out.detach(), rtol=0, atol=0)


def test_rotation_preserves_norm_and_depends_only_on_offsets():
    generator = torch.Generator().manual_seed(2)
    q = torch.randn(1, 2, 6, 8, generator=generator, dtype=torch.double)
    k = torch.randn(1, 2, 6, 8, generator=generator, dtype=torch.double)
    positions = torch.arange(6)
    rq, rk = rotate(q, positions, 10000.0), rotate(k, positions, 10000.0)
    torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))
    sq, sk = rotate(q, positions + 11, 10000.0), rotate(k, positions + 11, 10000.0)
    # The rotation is computed in float32 by design.
    torch.testing.assert_close(
        rq @ rk.transpose(-1, -2), sq @ sk.transpose(-1, -2), rtol=1e-5, atol=1e-5
    )


def test_rope_trace_equals_fresh_calls():
    model = build("invariant", "rope")
    perturb(model)
    logits, _, _ = model.trace(TOKENS, POSITIONS, 5)
    for r in (1, 3, 5):
        torch.testing.assert_close(logits[r - 1], model.run(TOKENS, POSITIONS, r, carry=False)[0])


def test_config_arms_and_recipes():
    runner = importlib.import_module("run_loop_position_invariance")
    path = ROOT / "configs/loop-position-invariance-development-v1.json"
    config = json.loads(path.read_text())
    runner.validate_config(config)
    old = runner.arm_config(config, "disco-rope-old")["training"]
    new = runner.arm_config(config, "ema-rope")["training"]
    assert old["segments"] == 1 and old["first_segment_rounds"] == [4, 4]
    assert old["betas"] == [0.9, 0.999] and old["final_lr_fraction"] == 1.0
    assert new == config["training"]
    native = importlib.import_module("run_loop_native_handoff")
    spec = {"lr": old["lr"], "warmup": old["warmup"]}
    for step in (1, 2, 1000, 2000, 2001, 900000, old["steps"]):
        assert math.isclose(native.learning_rate(old, step), original_rate(spec, step))
    arms = {a["handoff"] + "/" + a["position"] for a in config["arms"].values()}
    assert arms == {"ema/rope", "additive/rope", "ema/absolute"}
    assert config["paired_recipe"]["initial_backbone_sha256"].startswith("39a48782")
