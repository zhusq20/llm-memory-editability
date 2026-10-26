"""Check position-only changes preserve checkpoint state, causal inputs and short tasks."""

import pytest
import torch
from evaluate_loop_position_controlled_long_hops import position_intervention, state_hash
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT


def model(handoff="invariant"):
    torch.manual_seed(42)
    config = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=16,
        n_ctx=16,
        embd_pdrop=0,
        resid_pdrop=0,
        attn_pdrop=0,
    )
    return LoopNativeHandoffGPT(config, 3, handoff=handoff).double().eval()


@pytest.mark.parametrize("handoff", ("invariant", "ema", "content", "additive"))
@pytest.mark.parametrize("mode", ("absolute", "clamp4", "interpolate4"))
def test_short_query_exact_and_no_weight_mutation(handoff, mode):
    m = model(handoff)
    x = torch.tensor([[3, 5, 6, 7, 8]])
    before = state_hash(m)
    reference = m(x)
    with position_intervention(m, mode):
        assert torch.equal(m(x), reference)
    assert before == state_hash(m)
    assert torch.equal(m(x), reference)


def test_clamp_preserves_all_trained_slots_and_excludes_unused_entries():
    m = model()
    x = torch.tensor([[3, 5, 6, 7, 8, 9, 10, 11, 12]])
    with position_intervention(m, "clamp4"):
        actual = m._embedded(x)
    expected = m.transformer.wte(x) + m.transformer.wpe(torch.arange(9).clamp(max=4))
    assert torch.equal(actual, expected)


def test_interpolation_maps_endpoints_and_preserves_order():
    m = model()
    with torch.no_grad():
        m.transformer.wpe.weight.copy_(torch.arange(16)[:, None].expand(16, 12))
    x = torch.tensor([[3, 5, 6, 7, 8, 9, 10, 11, 12]])
    with position_intervention(m, "interpolate4"):
        actual = m._embedded(x) - m.transformer.wte(x)
    torch.testing.assert_close(actual[0, :, 0], torch.arange(9).double() / 2)


def test_zero_and_position_only_shift_do_not_insert_tokens():
    m = model()
    x = torch.tensor([[3, 5, 6, 7, 8]])
    with position_intervention(m, "zero"):
        assert torch.equal(m._embedded(x), m.transformer.wte(x))
    expected = m.transformer.wte(x) + m.transformer.wpe(torch.arange(5) + 3)
    with position_intervention(m, "absolute", 3):
        assert torch.equal(m._embedded(x), expected)


def test_hook_restored_after_exception_and_causal_mask_preserved():
    m = model()
    x = torch.tensor([[3, 5, 6, 7, 8, 9, 10]])
    original = m._embedded(x)
    with pytest.raises(RuntimeError), position_intervention(m, "zero"):
        raise RuntimeError("test")
    assert torch.equal(m._embedded(x), original)
    for mode in ("absolute", "zero", "clamp4", "interpolate4"):
        with position_intervention(m, mode):
            changed = x.clone()
            changed[:, -1] = 11
            assert torch.equal(m(x)[:, :-1], m(changed)[:, :-1])
