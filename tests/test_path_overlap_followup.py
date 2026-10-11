"""Contracts that distinguish causal prefix repair from answer-position damage."""

import copy

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import ReproductionGPT
from llm_memory_editability.path_overlap import Executor, pad_units
from llm_memory_editability.path_overlap_followup import (
    LOOP_MODES,
    STANDARD_MODES,
    Mode,
    PositionExecutor,
    _units_for,
    compare_audit,
    mode_predict,
    prefix_restore,
)


def make_model(unique=4, repeats=2):
    torch.manual_seed(7)
    cfg = GPT2Config(
        vocab_size=31,
        n_embd=16,
        n_layer=unique,
        n_head=4,
        n_positions=8,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    return ReproductionGPT(cfg, repeats).eval().requires_grad_(False)


@pytest.mark.parametrize("unique,repeats", [(4, 2), (8, 1)])
def test_full_transformer_matches_native_and_original(unique, repeats):
    model = make_model(unique, repeats)
    ex, old = PositionExecutor(model), Executor(model)
    tokens = torch.tensor([[3, 7, 11], [8, 2, 21]])
    state = ex.run(ex.embed(tokens), range(ex.executions))
    logits = ex.readout(state)
    assert torch.allclose(logits, model(tokens), atol=1e-6, rtol=1e-5)
    assert torch.equal(state, old.run(old.embed(tokens), range(old.executions)))
    assert torch.equal(logits, old.readout(old.run(old.embed(tokens), range(old.executions))))


@pytest.mark.parametrize("scope", ["all", "round1", "round2"])
def test_all_position_gates_exactly_match_original_executor(scope):
    model = make_model()
    ex, old = PositionExecutor(model), Executor(model)
    tokens = torch.tensor([[3, 7, 11], [8, 2, 21]])
    gates = ex.gates_for(torch.tensor([[0, 65, -1], [64, 129, 193]]), 2)
    kwargs = {"gates": gates, "gate_scope": ex.scope(scope)}
    assert torch.equal(ex.logits(tokens, 2, **kwargs), old.logits(tokens, 2, **kwargs))
    assert torch.equal(
        ex.logits(tokens, 2, gate_positions=(0, 1, 2), **kwargs),
        old.logits(tokens, 2, **kwargs),
    )


def test_gates_act_only_at_the_declared_positions_and_executions():
    model = make_model()
    ex = PositionExecutor(model)
    tokens = torch.tensor([[3, 7, 11], [8, 2, 21]])
    units = torch.tensor([[0], [1]])
    seen = []
    hook = ex.blocks[0].mlp.c_proj.register_forward_pre_hook(
        lambda _module, inputs: seen.append(inputs[0].clone())
    )
    try:
        ex.run(
            ex.embed(tokens),
            (0,),
            gates=ex.gates_for(units, 2),
            gate_scope=(0,),
            gate_positions=(2,),
        )
        ex.run(ex.embed(tokens), (0,))
        ex.run(
            ex.embed(tokens),
            (0,),
            gates=ex.gates_for(units, 2),
            gate_scope=(4,),
            gate_positions=(2,),
        )
    finally:
        hook.remove()
    gated, base, outside_scope = seen
    assert torch.equal(gated[:, :2], base[:, :2])
    assert gated[0, 2, 0] == 0 and gated[1, 2, 1] == 0
    for b, unit in ((0, 0), (1, 1)):
        keep = torch.arange(ex.neurons) != unit
        assert torch.equal(gated[b, 2, keep], base[b, 2, keep])
    assert torch.equal(outside_scope, base)


def test_causal_prefix_donor_cannot_see_alternative_second_relation():
    ex = PositionExecutor(make_model())
    a = torch.tensor([[3, 7, 11], [8, 2, 21]])
    b = torch.tensor([[3, 7, 12], [8, 2, 22]])
    state_a, boundaries_a = ex.run(ex.embed(a), range(4), capture=True)
    state_b, boundaries_b = ex.run(ex.embed(b), range(4), capture=True)
    for x, y in zip(boundaries_a, boundaries_b, strict=True):
        assert torch.equal(x[:, :2], y[:, :2])
    assert torch.equal(state_a[:, :2], state_b[:, :2])
    assert not torch.equal(state_a[:, 2], state_b[:, 2])


def test_last_position_ablation_preserves_both_prefix_positions():
    ex = PositionExecutor(make_model())
    tokens = torch.tensor([[3, 7, 11], [8, 2, 21]])
    gates = ex.gates_for(torch.tensor([[0, 65], [64, 129]]), 2)
    _, base = ex.run(ex.embed(tokens), range(8), capture=True)
    _, changed = ex.run(
        ex.embed(tokens),
        range(8),
        gates=gates,
        gate_scope=ex.scope("round1"),
        gate_positions=(2,),
        capture=True,
    )
    for x, y in zip(base, changed, strict=True):
        assert torch.equal(x[:, :2], y[:, :2])
    assert not torch.equal(base[-1][:, 2], changed[-1][:, 2])


@pytest.mark.parametrize("unique,repeats", [(4, 2), (8, 1)])
def test_same_query_baseline_prefix_restoration_is_exact_identity(unique, repeats):
    ex = PositionExecutor(make_model(unique, repeats))
    tokens = np.asarray([[3, 7, 11], [8, 2, 21]])
    target = np.asarray([13, 14])
    base = mode_predict(ex, tokens, target, Mode("baseline"), batch=1)
    restored = mode_predict(ex, tokens, target, Mode("identity", restore_prefix=True), batch=1)
    assert all(np.array_equal(a, b) for a, b in zip(base, restored, strict=True))


def test_prefix_restore_replaces_both_prefix_positions_and_preserves_last():
    donor = torch.randn(2, 3, 16)
    recipient = torch.randn(2, 3, 16)
    donor_before, recipient_before = donor.clone(), recipient.clone()
    out = prefix_restore(donor)(recipient)
    assert torch.equal(out[:, :2], donor[:, :2])
    assert torch.equal(out[:, 2], recipient[:, 2])
    assert torch.equal(donor, donor_before) and torch.equal(recipient, recipient_before)


def test_boundary_bridge_readout_is_same_query_and_precedes_final_answer_readout():
    ex = PositionExecutor(make_model())
    tokens = np.asarray([[3, 7, 11], [8, 2, 21]])
    final_target, bridge_target = np.asarray([13, 14]), np.asarray([9, 10])
    baseline_capture, ablated_capture, restored_capture = {}, {}, {}
    units = np.asarray([[0, 65], [64, 129]])
    for mode, capture in (
        (Mode("baseline"), baseline_capture),
        (Mode("ablated", "round1"), ablated_capture),
        (Mode("restored", "round1", restore_prefix=True), restored_capture),
    ):
        mode_predict(
            ex,
            tokens,
            final_target,
            mode,
            units,
            batch=1,
            bridge_target=bridge_target,
            bridge_capture=capture,
        )
    assert np.array_equal(ablated_capture["before_margins"], restored_capture["before_margins"])
    assert np.array_equal(baseline_capture["before_margins"], restored_capture["after_margins"])
    assert np.array_equal(
        baseline_capture["before_predictions"], restored_capture["after_predictions"]
    )
    assert not np.array_equal(baseline_capture["before_margins"], ablated_capture["before_margins"])


def test_restore_cannot_change_saved_units_model_parameters_or_config():
    model = make_model()
    ex = PositionExecutor(model)
    records = [
        {"atom": 17, "units": [0, 65], "control_units": [1, 66], "selective": True},
        {"atom": 19, "units": [64], "control_units": [67], "selective": False},
    ]
    config = {"nodes": [{"run": "loop4x2-phi7.2-wd0.1", "step": 1500000}]}
    original_records, original_config = copy.deepcopy(records), copy.deepcopy(config)
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    units = _units_for(records, np.asarray([0, 0]), "units")
    assert np.array_equal(units, pad_units([[0, 65], [0, 65]], 2))
    mode_predict(
        ex,
        np.asarray([[3, 7, 11], [8, 2, 21]]),
        np.asarray([13, 14]),
        Mode("repair", "round1", restore_prefix=True),
        units,
    )
    assert records == original_records and config == original_config
    assert all(torch.equal(weights[key], value) for key, value in model.state_dict().items())
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_invalid_position_masks_fail_instead_of_changing_intervention_scope():
    ex = PositionExecutor(make_model())
    x = ex.embed(torch.tensor([[3, 7, 11]]))
    for positions in ((-1,), (3,), (1, 1)):
        with pytest.raises(ValueError, match="Gate positions"):
            ex.run(x, range(8), gate_positions=positions)


def test_standard_diagnostics_have_no_round_names():
    assert not any("round" in mode.name or mode.scope == "round1" for mode in STANDARD_MODES)
    assert [mode.name for mode in LOOP_MODES] == [
        "baseline",
        "baseline_prefix_identity",
        "round1_all",
        "round2_all",
        "round1_last",
        "round1_prefix",
        "round1_all_prefix_restored",
        "round2_last",
    ]


def test_audit_requires_exact_predictions_and_per_fact_measures():
    summary = {key: {} for key in ("coverage", "reproduction", "modes", "bootstrap", "source")}
    arrays = {
        "primary_predictions": np.asarray([3, 4]),
        "primary_margins": np.asarray([0.5, 1.5], np.float32),
        "fact_delta": np.asarray([0.1, np.nan]),
    }
    replay = {key: value.copy() for key, value in arrays.items()}
    replay["primary_margins"][0] += 1e-5
    assert compare_audit(summary, arrays, summary, replay)["passed"]
    replay["primary_predictions"][0] = 4
    with pytest.raises(ValueError, match="primary_predictions"):
        compare_audit(summary, arrays, summary, replay)
