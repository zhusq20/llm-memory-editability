"""Intervention scope, causal prefixes, response weighting and donor truth."""

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.depth_step import prompt_rows
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.recurrence_use import (
    generate,
    interaction,
    prediction_metrics,
    query_means,
    select_donors,
    trace,
)


@pytest.fixture
def model():
    torch.manual_seed(872)
    value = (
        LoopGPT(
            ModelConfig(vocab_size=14, width=8, layers=1, heads=2, context=9),
            repeats=4,
            dropout=0.0,
        )
        .double()
        .eval()
    )
    value.requires_grad_(False)
    return value


@pytest.mark.parametrize("repeats", [2, 4, 6])
def test_trace_exactly_matches_native_and_never_changes_parameters(model, repeats):
    model.repeats = repeats
    tokens = torch.tensor([[2, 3, 11, 12, 13], [2, 4, 12, 11, 13]])
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    actual, cache = trace(model, tokens, repeats)
    torch.testing.assert_close(actual, model(tokens), rtol=0, atol=0)
    assert cache["full"].shape == (repeats, 2, 5, 8)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("scope", ["all", "r1"])
def test_single_occurrence_scope_not_shared_parameter_ablation(model, scope):
    tokens = torch.tensor([[2, 3, 11, 12, 13]])
    _, normal = trace(model, tokens, 4)
    _, deleted = trace(model, tokens, 4, skip=(scope, 1))
    torch.testing.assert_close(deleted["output"][0], normal["output"][0], rtol=0, atol=0)
    if scope == "all":
        torch.testing.assert_close(deleted["output"][1], deleted["input"][1], rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            deleted["output"][1, :, 2], deleted["input"][1, :, 2], rtol=0, atol=0
        )
        torch.testing.assert_close(
            deleted["output"][1, :, 3:], normal["output"][1, :, 3:], rtol=0, atol=0
        )
    assert not torch.equal(deleted["full"][2], normal["full"][2])


def test_last_local_delete_cannot_change_answer_but_earlier_one_can(model):
    tokens = torch.tensor([[2, 3, 11, 12, 13]])
    normal, _ = trace(model, tokens, 4)
    last, _ = trace(model, tokens, 4, skip=("r1", 3))
    early, _ = trace(model, tokens, 4, skip=("r1", 0))
    torch.testing.assert_close(last[:, -1], normal[:, -1], rtol=0, atol=0)
    assert not torch.equal(early[:, -1], normal[:, -1])


@pytest.mark.parametrize("component", ["full", "mlp"])
def test_pure_prefix_self_patch_has_no_suffix_or_answer_information(model, component):
    rows = np.asarray([[3, 11, 12, 5], [4, 12, 11, 6]])
    self_atoms = np.c_[rows[:, :2], [7, 8]]
    native, _ = generate(model, rows, 13, 4)
    patched, _ = generate(model, rows, 13, 4, donor_atoms=self_atoms, component=component)
    for field in ("generated", "logits"):
        np.testing.assert_array_equal(native[field], patched[field])
    tokens = torch.as_tensor(prompt_rows(rows, 13))
    altered = tokens.clone()
    altered[:, 3:] = 0
    _, a = trace(model, tokens, 4)
    _, b = trace(model, altered, 4)
    torch.testing.assert_close(a["output"][:, :, 2], b["output"][:, :, 2], rtol=0, atol=0)


def test_response_is_ratio_of_means_and_zero_denominator_is_missing():
    value = interaction([1.0, 1.0], [1.0, 3.0])
    assert value["C"] == 0.5
    assert value["C"] != np.mean([1.0, 1 / 3])
    assert interaction([1.0, 1.0], [0.0, 0.0])["C"] is None


def test_multiple_donors_do_not_reweight_queries_and_missing_queries_stay_missing():
    means = query_means(np.asarray([0.0, 1.0, 1.0]), np.asarray([0, 0, 1]), 3)
    np.testing.assert_array_equal(means[:2], [0.5, 1.0])
    assert np.isnan(means[2])
    assert means[:2].mean() == 0.75


def test_generated_eos_uses_generated_answer_instead_of_gold(model):
    rows = np.asarray([[3, 11, 12, 5]])
    pred, _ = generate(model, rows, 13, 4)
    tokens = torch.as_tensor(prompt_rows(rows, 13))
    answer = model(tokens)[:, -1].argmax(-1)
    stop = model(torch.cat((tokens, answer[:, None]), 1))[:, -1].argmax(-1)
    np.testing.assert_array_equal(pred["generated"], torch.stack((answer, stop), 1).numpy())
    scores = prediction_metrics(pred, rows[:, -1])
    assert scores["complete"][0] == int(
        pred["generated"][0, 0] == 5 and pred["generated"][0, 1] == 1
    )


def test_registered_development_donors_preserve_relation_bridge_and_untrained_routes():
    from llm_memory_editability.multihop_scaling import build_world

    low = build_world({"world_seed": 761011, "phi": 1.0})
    high = build_world({"world_seed": 761011, "phi": 4.0})
    donors = select_donors(low, high)
    atoms = low["atomic"]
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    rows = low["strict_2"]
    for label in ("experienced", "strict"):
        recipients, indices = donors[label + "_recipients"], donors[label + "_atoms"]
        for recipient, fact in zip(recipients, indices, strict=True):
            h, r1, bridge = atoms[fact]
            row = rows[recipient]
            assert r1 == row[1] and bridge == lookup[int(row[0]), int(row[1])]
            assert h not in (row[0], row[-1])
        if label == "experienced":
            assert (donors["low_roles"][indices, 0] > 0).all()
            assert (donors["high_roles"][indices, 0] > 0).all()
        else:
            assert not donors["high_roles"][indices].any()
    trained = {tuple(row[:-1]) for row in high["train_2"]}
    for label in ("familiar", "strict"):
        rec = donors[label + "_changed_recipients"]
        facts = donors[label + "_changed_atoms"]
        targets = donors[label + "_changed_targets"]
        for i, fact, target in zip(rec, facts, targets, strict=True):
            row = low[label + "_2"][i]
            atom = atoms[fact]
            assert atom[1] == row[1]
            assert target == lookup[int(atom[2]), int(row[2])]
            assert target != row[-1]
            assert (int(atom[0]), int(row[1]), int(row[2])) not in trained


def test_unknown_occurrence_is_rejected(model):
    tokens = torch.tensor([[2, 3, 11, 12, 13]])
    with pytest.raises(ValueError, match="skip"):
        trace(model, tokens, 4, skip=("all", 4))
