"""Non-invasive observation and R16-before/after attribution."""

import numpy as np
import torch
from trace_invariant_five_hops import summary, trace_stages
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT


def test_trace_exactly_preserves_native_predictions_and_parameters():
    torch.manual_seed(31)
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
    )
    model = LoopNativeHandoffGPT(cfg, 4, handoff="invariant").double().eval()
    tokens = torch.tensor([[3, 8, 9, 10, 11, 12], [4, 8, 10, 9, 12, 11]])
    positions = torch.tensor([[5], [5]])
    truths = torch.tensor([[3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9]])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    native, _, gamma = model.trace(tokens, positions, 8)
    traced = trace_stages(model, tokens, positions, truths, 8)
    np.testing.assert_array_equal(traced["answer"], native[:, :, 0].argmax(-1).numpy())
    np.testing.assert_array_equal(traced["gamma"], gamma.numpy())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(model.trace(tokens, positions, 8)[0], native)
    changed = tokens.clone()
    changed[:, -1] = 13
    other = trace_stages(model, changed, positions, truths, 8)
    np.testing.assert_array_equal(
        traced["aggregate_token"][:, :, :-1], other["aggregate_token"][:, :, :-1]
    )


def test_r16_loss_does_not_count_future_first_correct_as_prior_overwrite():
    truth = np.ones((3, 6), dtype=int)
    answer = np.zeros((32, 3), dtype=int)
    answer[0, 0] = 1
    answer[15, 1] = 1
    answer[20, 2] = 1
    values = {"answer": answer, "last_attention_weights": np.zeros((32, 3, 2, 6))}
    for stage in ("input", "postattention", "raw", "candidate", "aggregate"):
        values[stage + "_token"] = np.zeros((32, 3, 6), dtype=int)
        values[stage + "_probability"] = np.zeros((32, 3, 6))
    out = summary(values, truth)
    assert out["ever_correct_by_r16"] == 2
    assert out["correct_then_lost_at_r16"] == 1
    assert out["never_correct_by_r16"] == 1
    assert out["first_correct_after_r16"] == 1
