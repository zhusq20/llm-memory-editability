"""Contracts for causal self-decoding, zero intervention, and frozen evaluation."""

import json

import numpy as np
import pytest
import torch

from llm_memory_editability.bridge_reencoding import (
    ReencodingGPT,
    alternative_entity,
    audit,
    reencode_state,
    run,
)
from llm_memory_editability.latent_scaling import build_world, model_digest
from llm_memory_editability.representation_alignment import new_model
from llm_memory_editability.storage_composition import generate_rows

torch.set_num_threads(1)


def small_spec():
    return dict(
        world=771001,
        initialization=772001,
        stream_seed=773001,
        heads_n=32,
        bridges_n=32,
        tails_n=16,
        familiar_n=8,
        strict_n=4,
        anchor_n=4,
        holdout_fraction=0.25,
        low_extra="anchors",
        composition_count=32,
        width=32,
        heads=4,
        layers=1,
        repeats=2,
        dropout=0.0,
    )


def make_model(condition="self_decode", alpha=0.5):
    model = new_model(small_spec(), "cpu")
    model.__class__ = ReencodingGPT
    model.bridge_start, model.bridge_count = 53, 32
    model.configure(condition, alpha)
    return model.eval()


def test_zero_is_exact_identity_and_formula_has_no_outer_normalization():
    torch.manual_seed(11)
    h, e = torch.randn(17, 32), torch.randn(17, 32)
    assert reencode_state(h, e, 0) is h
    for alpha in (0.1, 0.5, 1):
        result = reencode_state(h, e, alpha)
        expected = (1 - alpha) * h + alpha * h.norm(
            dim=-1, keepdim=True
        ) * torch.nn.functional.normalize(e)
        torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        reencode_state(h, e, 1), h.norm(dim=-1, keepdim=True) * torch.nn.functional.normalize(e)
    )
    assert reencode_state(h, -h, 0.5).abs().max() < 1e-6
    torch.testing.assert_close(
        reencode_state(h, e, 0.5, "paper_unit"),
        0.5 * h + 0.5 * torch.nn.functional.normalize(e),
    )
    with pytest.raises(ValueError):
        reencode_state(h, e, 1.1)


def test_alpha_zero_matches_original_logits_and_complete_generation():
    spec = small_spec()
    reference = new_model(spec, "cpu").eval()
    model = make_model(alpha=0)
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4], [2, 22, 3, 14, 3, 18, 4]])
    torch.testing.assert_close(model(tokens), reference(tokens), atol=0, rtol=0)
    rows = build_world(spec)["strict_test"][:8]
    _, original = generate_rows(reference, rows, "cpu")
    for condition in ("baseline", "self_decode", "wrong_entity"):
        model.configure(condition, 0)
        _, actual = generate_rows(model, rows, "cpu")
        for key in original:
            np.testing.assert_array_equal(original[key], actual[key])


def test_decoding_is_original_full_vocabulary_and_cannot_see_future():
    model = make_model()
    a = torch.tensor([[2, 21, 3, 13, 3, 17, 4, 85]])
    b = torch.tensor([[2, 21, 3, 13, 4, 20, 5, 86]])
    _, ia = model(a, return_bridge=True)
    _, ib = model(b, return_bridge=True)
    torch.testing.assert_close(ia["state"], ib["state"], atol=0, rtol=0)
    assert torch.equal(ia["decoded"], ib["decoded"])
    expected = torch.nn.functional.linear(model.ln_final(ia["state"]), model.token.weight).argmax(
        -1
    )
    assert torch.equal(ia["decoded"], expected)
    assert torch.equal(ia["selected"], expected)
    assert model.oracle_table is None


def test_alternative_entity_never_uses_truth_and_is_nonself():
    decoded = torch.arange(101)
    selected = alternative_entity(decoded, 53, 32)
    assert torch.all((selected >= 53) & (selected < 85))
    assert torch.all(selected != decoded)
    model = make_model("wrong_entity")
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4]])
    _, info = model(tokens, return_bridge=True)
    assert torch.equal(info["selected"], alternative_entity(info["decoded"], 53, 32))
    assert model.oracle_table is None


def test_single_execution_changes_only_sender_position_and_keeps_parameters():
    model = make_model("self_decode", 1)
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4], [2, 22, 3, 14, 3, 18, 4]])
    digest = model_digest(model)
    changed = model(tokens, repeats=1)
    model.configure("baseline", 0)
    original = model(tokens, repeats=1)
    unchanged_positions = [0, 1, 2, 4, 5, 6]
    torch.testing.assert_close(
        changed[:, unchanged_positions], original[:, unchanged_positions], atol=0, rtol=0
    )
    assert not torch.equal(changed[:, 3], original[:, 3])
    assert model_digest(model) == digest


def test_oracle_is_explicit_and_does_not_modify_model_parameters():
    model = make_model()
    before = model_digest(model)
    with pytest.raises(ValueError, match="explicit"):
        model.configure("oracle", 1)
    truth = torch.full((101, 101), -1, dtype=torch.long)
    truth[21, 13] = 53
    model.configure("oracle", 1, truth)
    _, info = model(torch.tensor([[2, 21, 3, 13, 3, 17, 4]]), return_bridge=True)
    assert info["selected"].item() == 53
    assert info["oracle_available"].item()
    assert model_digest(model) == before
    model.configure("self_decode", 1)
    assert model.oracle_table is None


def test_full_run_and_reload_preserve_raw_predictions_and_reject_corruption(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    spec = small_spec()
    model = new_model(spec, "cpu")
    torch.save({"spec": spec, "model": model.state_dict()}, parent / "model.pt")
    np.savez_compressed(parent / "world.npz", **build_world(spec))
    out = tmp_path / "evaluation"
    request = {
        "parent_dir": str(parent),
        "alphas": [0, 0.5],
        "variants": ["norm_matched", "paper_unit"],
        "conditions": ["baseline", "self_decode", "wrong_entity", "oracle"],
    }
    run(request, out, "cpu")
    result = audit(out, "cpu")
    assert result["passed"] and result["raw_predictions_recounted"]
    assert json.loads((out / "complete.json").read_text())["independently_reloaded"]
    with pytest.raises(FileExistsError):
        run(request, out, "cpu")
    predictions = dict(np.load(out / "predictions.npz"))
    assert "norm_matched-a0-self_decode__strict_test_first_generated" in predictions
    assert "norm_matched-a0-self_decode__strict_test_common_atoms_and_self_bridge" in predictions
    fixed = predictions["baseline__strict_test_coverage"]
    for name in json.loads((out / "run.json").read_text())["cases"]:
        np.testing.assert_array_equal(
            fixed, predictions[f"{name}__strict_test_baseline_common_atoms"]
        )
    predictions["norm_matched-a0-self_decode__strict_test_generated"][0, 0] += 1
    np.savez_compressed(out / "predictions.npz", **predictions)
    with pytest.raises(AssertionError):
        audit(out, "cpu")
