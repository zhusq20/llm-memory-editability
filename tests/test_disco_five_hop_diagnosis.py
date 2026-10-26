"""Native replay, single-source value layout and canonical atomic donor contracts."""

import numpy as np
import pytest
import torch
from diagnose_disco_five_hops import CONDITIONS, atomic_donor, trace
from transformers import GPT2Config

from llm_memory_editability.loop_position_invariance import LoopPositionGPT


def setup(position):
    torch.manual_seed(51)
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
    m = LoopPositionGPT(cfg, 4, handoff="additive", position=position).double().eval()
    x = torch.tensor([[3, 8, 9, 10, 11, 12], [4, 8, 10, 9, 12, 11]])
    return m, x, torch.tensor([[5], [5]]), torch.tensor([5, 6])


@pytest.mark.parametrize("position", ["absolute", "rope"])
def test_native_replay_and_patch_causal_scope(position):
    m, x, pos, bridge = setup(position)
    truth = (x + 2) % 23
    old = m.trace(x, pos, 6)[0][:, :, 0].argmax(-1).numpy()
    native = trace(m, x, pos, truth, rounds=6)
    np.testing.assert_array_equal(native["answer"], old)
    np.testing.assert_array_equal(trace(m, x, pos, (truth + 1) % 23, rounds=6)["answer"], old)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    donor = atomic_donor(m, x, bridge)
    for condition in CONDITIONS:
        patched = trace(m, x, pos, rounds=6, condition=condition, at_round=4, donor=donor)
        np.testing.assert_array_equal(patched["answer"][:3], native["answer"][:3])
        for stage in ["raw", "carry"]:
            np.testing.assert_array_equal(
                patched[stage + "_token"][:, :, :-1], native[stage + "_token"][:, :, :-1]
            )
        if condition.endswith("postattention"):
            np.testing.assert_array_equal(patched["answer"][3], donor["raw_token"].numpy())
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
    np.testing.assert_array_equal(m.trace(x, pos, 6)[0][:, :, 0].argmax(-1).numpy(), old)


@pytest.mark.parametrize("position", ["absolute", "rope"])
def test_force_value_head_layout_and_first_round_atomic_embedding(position):
    m, x, pos, bridge = setup(position)
    block = m.transformer.h[0]
    for condition, source in [("force_bridge", 4), ("force_original_head", 0)]:
        qkv, projection = [], []
        handles = [
            block.attn.c_attn.register_forward_hook(
                lambda module, inputs, out, saved=qkv: saved.append(out.detach().clone())
            ),
            block.attn.c_proj.register_forward_hook(
                lambda module, inputs, out, saved=projection: saved.append(
                    inputs[0].detach().clone()
                )
            ),
        ]
        try:
            trace(m, x, pos, rounds=3, condition=condition, at_round=2)
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(
            projection[1][:, 5], qkv[1][:, source, 2 * m.config.n_embd :], rtol=0, atol=0
        )
    donor = atomic_donor(m, x, bridge)
    atomic = torch.zeros_like(x)
    atomic[:, 0], atomic[:, 1] = bridge, x[:, -1]
    raw = m._round(m._embedded(atomic))
    torch.testing.assert_close(donor["raw_state"], raw[:, 1])
    z = donor["postattention"][:, None]
    torch.testing.assert_close(donor["raw_state"], (z + block.mlp(block.ln_2(z)))[:, 0])
    altered = x.clone()
    altered[:, :5] = (altered[:, :5] + 3) % 23
    other = atomic_donor(m, altered, bridge)
    for key in donor:
        torch.testing.assert_close(donor[key], other[key])
