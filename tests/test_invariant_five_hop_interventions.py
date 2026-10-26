"""Causal scope, native replay and atomic donor semantics."""

import numpy as np
import torch
from intervene_invariant_five_hops import CONDITIONS, atomic_donor, intervene
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT
from llm_memory_editability.self_decode_alignment import _loss_dtype


def model_and_input():
    torch.manual_seed(41)
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
    model = LoopNativeHandoffGPT(cfg, 4, handoff="invariant").double().eval()
    tokens = torch.tensor([[3, 8, 9, 10, 11, 12], [4, 8, 10, 9, 12, 11]])
    return model, tokens, torch.tensor([[5], [5]]), torch.tensor([5, 6])


def test_all_patches_preserve_earlier_rounds_and_other_receivers():
    model, tokens, pos, bridge = model_and_input()
    native = intervene(model, tokens, pos, bridge, "native", 0, 8)
    plain = model.trace(tokens, pos, 8)[0][:, :, 0].argmax(-1).numpy()
    np.testing.assert_array_equal(native["answer"], plain)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for condition in CONDITIONS:
        patch = intervene(model, tokens, pos, bridge, condition, 4, 8)
        np.testing.assert_array_equal(
            patch["prefix_token"][:, :, :-1], native["prefix_token"][:, :, :-1]
        )
        np.testing.assert_array_equal(patch["answer"][:3], native["answer"][:3])
        if condition.endswith("postattention"):
            np.testing.assert_array_equal(patch["raw"][3], patch["donor_raw_token"])
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    np.testing.assert_array_equal(model.trace(tokens, pos, 8)[0][:, :, 0].argmax(-1).numpy(), plain)


def test_donor_is_atomic_and_replacement_contains_full_postattention_residual():
    model, tokens, _, bridge = model_and_input()
    donor = atomic_donor(model, tokens, bridge)
    z = donor["postattention"][:, None]
    block = model.transformer.h[0]
    torch.testing.assert_close(donor["raw_state"], (z + block.mlp(block.ln_2(z)))[:, 0])
    atomic = torch.full_like(tokens, 0)
    atomic[:, 0], atomic[:, 1] = bridge, tokens[:, -1]
    qkv = block.attn.c_attn(block.ln_1(2 * _loss_dtype(model._embedded(atomic))))
    torch.testing.assert_close(donor["value"], qkv[:, 0, 2 * model.config.n_embd :])
    changed = tokens.clone()
    changed[:, :5] = (changed[:, :5] + 1) % model.config.vocab_size
    other = atomic_donor(model, changed, bridge)
    for k in donor:
        torch.testing.assert_close(donor[k], other[k])
