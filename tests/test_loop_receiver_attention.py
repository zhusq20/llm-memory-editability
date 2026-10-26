"""Calibrated attention/residual recombination with original MLP tensor shapes."""

import numpy as np
import pytest
import torch
from diagnose_loop_receiver_attention import CONDITIONS, make_donors, trace
from test_loop_receiver_state import setup


@pytest.mark.parametrize("arm", ["rope", "invariant"])
def test_original_and_atomic_calibration_and_causal_scope(arm):
    model, tokens, pos, _ = setup(arm)
    bridge = torch.tensor([5, 6])
    d = make_donors(model, tokens, bridge, bridge + 1, 8, 5)
    native = trace(model, tokens, pos, d, rounds=6)
    exact = model.trace(tokens, pos, 6)[0][:, :, 0].argmax(-1).numpy()
    np.testing.assert_array_equal(native["answer"], exact)
    same = trace(model, tokens, pos, d, "native_old", 4, 6)
    for key in ["answer", "raw", "candidate", "carry", "raw_prefix", "carry_prefix"]:
        np.testing.assert_array_equal(same[key], native[key])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for condition in CONDITIONS:
        patch = trace(model, tokens, pos, d, condition, 4, 6)
        np.testing.assert_array_equal(patch["answer"][:3], native["answer"][:3])
        for stage in ["raw_prefix", "carry_prefix"]:
            np.testing.assert_array_equal(patch[stage][:, :, :-1], native[stage][:, :, :-1])
        if condition == "atomic_fresh_relation":
            np.testing.assert_array_equal(patch["raw"][3], d["atomic_true_raw_token"].numpy())
        if condition == "wrong_bridge_attention_fresh_relation":
            np.testing.assert_array_equal(
                patch["raw"][3], d["atomic_wrong_bridge_raw_token"].numpy()
            )
    both_wrong = trace(model, tokens, pos, d, "wrong_relation_attention_fresh_wrong_relation", 4, 6)
    np.testing.assert_array_equal(
        both_wrong["raw"][3], d["atomic_wrong_relation_raw_token"].numpy()
    )
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    np.testing.assert_array_equal(model.trace(tokens, pos, 6)[0][:, :, 0].argmax(-1).numpy(), exact)


@pytest.mark.parametrize("arm", ["rope", "invariant"])
def test_native_attention_scale_and_zero_do_not_use_oracle_bridge(arm):
    model, tokens, pos, _ = setup(arm)
    bridge = torch.tensor([5, 6])
    a = make_donors(model, tokens, bridge, bridge + 1, 8, 5)
    b = make_donors(model, tokens, bridge + 2, bridge + 3, 8, 5)
    for condition in ["native_scale", "native_zero"]:
        left = trace(model, tokens, pos, a, condition, 5, 6)
        right = trace(model, tokens, pos, b, condition, 5, 6)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
