"""Preserve native recurrences and isolate receiver content from fixed bridge input."""

import numpy as np
import pytest
import torch
from diagnose_loop_receiver_state import (
    CONDITIONS,
    INVARIANT_EXTRA,
    make_donors,
    receiver,
    relation_counterfactuals,
    trace,
)
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT
from llm_memory_editability.loop_position_invariance import LoopPositionGPT


def test_counterfactual_panel_changes_only_last_relation_and_recomputes_targets():
    rows = np.zeros((10, 500, 7), dtype=np.int64)
    rows[:, :, 0] = np.arange(500) + 500
    for template in range(10):
        rows[template, :, 1:6] = np.array([template, 2, 3, 4, 5])
    world = {
        "permutations": np.stack(
            [
                np.stack([(np.arange(500) + relation + 1) % 500 for relation in range(10)])
                for _ in range(2)
            ]
        )
    }
    cf = relation_counterfactuals(rows.reshape(5000, 7), {"entities_per_graph": 500}, world)
    groups = cf.reshape(500, 10, 7)
    np.testing.assert_array_equal(groups[:, :, :5], np.repeat(groups[:, :1, :5], 10, axis=1))
    np.testing.assert_array_equal(groups[:, :, 5], np.tile(np.arange(10), (500, 1)))
    expected = (cf[:, 0] - 500 + (cf[:, 1:6] + 1).sum(1)) % 500 + 500
    np.testing.assert_array_equal(cf[:, -1], expected)


def setup(arm):
    torch.manual_seed(61)
    cfg = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=16,
        n_ctx=16,
        embd_pdrop=0,
        resid_pdrop=0,
        attn_pdrop=0,
        pad_token_id=0,
    )
    m = (
        (
            LoopNativeHandoffGPT(cfg, 4, handoff="invariant")
            if arm == "invariant"
            else LoopPositionGPT(cfg, 4, handoff="additive", position="rope")
        )
        .double()
        .eval()
    )
    x = torch.tensor([[3, 8, 9, 10, 11, 12], [4, 8, 10, 9, 12, 11]])
    b = torch.tensor([5, 6])
    d = make_donors(m, x, b, b + 1, 8, 5)
    return m, x, torch.tensor([[5], [5]]), d


@pytest.mark.parametrize("arm", ["rope", "invariant"])
def test_native_history_and_single_receiver_scope(arm):
    m, x, pos, d = setup(arm)
    plain = m.trace(x, pos, 6)[0][:, :, 0].argmax(-1).numpy()
    native = trace(m, x, pos, d, rounds=6)
    np.testing.assert_array_equal(native["answer"], plain)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    for condition in CONDITIONS + (INVARIANT_EXTRA if arm == "invariant" else ()):
        patched = trace(m, x, pos, d, condition, 4, 6)
        np.testing.assert_array_equal(patched["answer"][:3], native["answer"][:3])
        for stage in ["raw_prefix", "carry_prefix"]:
            np.testing.assert_array_equal(patched[stage][:, :, :-1], native[stage][:, :, :-1])
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    np.testing.assert_array_equal(m.trace(x, pos, 6)[0][:, :, 0].argmax(-1).numpy(), plain)


@pytest.mark.parametrize("arm", ["rope", "invariant"])
def test_norm_and_constant_do_not_supply_question_specific_relation(arm):
    m, x, _, d = setup(arm)
    changed = x.clone()
    changed[:, -1] = 8
    b = torch.tensor([5, 6])
    other = make_donors(m, changed, b, b + 1, 8, 5)
    for key in ["target_norm", "token_norm", "constant", "value", "wrong_value"]:
        torch.testing.assert_close(d[key], other[key], rtol=0, atol=0)
    torch.testing.assert_close(d["relation"].norm(dim=-1), d["wrong_relation"].norm(dim=-1))
    atom = torch.zeros_like(x)
    atom[:, 0] = b
    atom[:, 1] = x[:, -1]
    scale = 2 if arm == "invariant" else 1
    torch.testing.assert_close(d["fresh_atomic"], scale * m._embedded(atom)[:, 1])
    if arm == "rope":
        torch.testing.assert_close(d["fresh_atomic"], d["fresh_site"], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["rope", "invariant"])
def test_forced_bridge_block_reduces_to_receiver_residual_plus_mlp(arm):
    m, x, pos, d = setup(arm)
    old = m._round
    inputs = []
    outputs = []

    def capture(value):
        inputs.append(value.detach().clone())
        out = old(value)
        outputs.append(out.detach().clone())
        return out

    m._round = capture
    try:
        trace(m, x, pos, d, "bridge_scale_add_relation", 4, 6)
    finally:
        m._round = old
    z = inputs[3][:, 5]
    block = m.transformer.h[0]
    post = z[:, None] + block.attn.c_proj(d["value"][:, None])
    expected = post + block.mlp(block.ln_2(post))
    torch.testing.assert_close(outputs[3][:, 5], expected[:, 0])
    old_z = torch.randn_like(z)
    scaled = receiver(old_z, d, "bridge_scale")
    torch.testing.assert_close(scaled.norm(dim=-1), d["target_norm"].expand(len(z)).to(z.dtype))
