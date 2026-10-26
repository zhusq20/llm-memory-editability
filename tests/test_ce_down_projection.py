"""Meaningful contracts for down-only CE addition and fixed-feature protection."""

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.ce_down_projection import (
    atomic_ce_write,
    basis_from_covariance,
    choose_learning_rate,
    collect_covariances,
    project_gradient,
)
from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest

META = {"entity_offset": 0, "relation_offset": 10, "end_marker": 28, "pad_token": 27}


@pytest.fixture
def reader():
    torch.set_num_threads(1)
    torch.manual_seed(20261016)
    cfg = GPT2Config(vocab_size=29, n_positions=16, n_embd=16, n_head=2, n_layer=2,
                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    return TTCDReader(SupervisedAlignmentGPT(cfg, repeats=2, discoloop=True).double(),
                      native_causal_attention=True).eval()


def test_projection_orientation_preserves_protected_down_outputs():
    torch.manual_seed(13)
    basis = torch.linalg.qr(torch.randn(7, 3, dtype=torch.double)).Q
    gradient = torch.randn(7, 5, dtype=torch.double)
    projected = project_gradient(gradient, basis)
    torch.testing.assert_close(basis.T @ projected, torch.zeros(3, 5, dtype=torch.double), atol=1e-14, rtol=0)
    old_features = torch.randn(6, 3, dtype=torch.double) @ basis.T
    weights = torch.randn_like(gradient)
    torch.testing.assert_close(old_features @ weights, old_features @ (weights - 0.2 * projected))
    assert projected.norm() <= gradient.norm()


def test_spectral_basis_retains_required_energy_without_arbitrary_rank_cap():
    covariance = torch.diag(torch.tensor([9.0, 0.9, 0.1], dtype=torch.double))
    basis, detail = basis_from_covariance(covariance, 0.95)
    assert detail["rank"] == 2 and detail["retained_energy"] == pytest.approx(0.99)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=torch.double))
    full, detail = basis_from_covariance(covariance, 1.0)
    assert full.shape == (3, 3) and detail["rank"] == 3


@pytest.mark.parametrize("steps", [1, 3])
def test_ce_write_matches_materialized_shared_weights_and_freezes_backbone(reader, steps):
    before = tensor_digest(reader.backbone.state_dict())
    fact, rate = [1, 2, 4], 0.01
    actual, info = atomic_ce_write(reader, fact, META, rate, steps=steps)
    model = copy.deepcopy(reader.backbone)
    weights = [block.mlp.c_proj.weight for block in model.transformer.h]
    for parameter in weights:
        parameter.requires_grad_(True)
    original = [w.detach().clone() for w in weights]
    arrays = encode_rows(np.asarray([fact]), META)
    tokens, positions, labels = (torch.as_tensor(a) for a in arrays)
    for _ in range(steps):
        loss = F.cross_entropy(model(tokens, positions).flatten(0, 1), labels.flatten())
        gradients = torch.autograd.grad(loss, weights)
        with torch.no_grad():
            for weight, gradient in zip(weights, gradients, strict=True):
                weight.sub_(rate * gradient)
    for layer, (weight, old) in enumerate(zip(weights, original, strict=True)):
        torch.testing.assert_close(actual[layer], weight.detach() - old, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(reader(tokens, positions, fast=actual), model(tokens, positions), atol=1e-12, rtol=1e-10)
    assert info["steps"] == steps and len(info["history"]) == steps
    assert tensor_digest(reader.backbone.state_dict()) == before
    assert all(not p.requires_grad and p.grad is None for p in reader.parameters())


def test_projection_is_applied_to_every_actual_sgd_update(reader):
    bases = {k: torch.linalg.qr(torch.randn(v.shape[0], 5, dtype=torch.double)).Q
             for k, v in reader.empty_fast().items()}
    state, info = atomic_ce_write(reader, [1, 2, 4], META, 0.01, bases=bases, steps=3)
    for layer, delta in state.items():
        torch.testing.assert_close(bases[layer].T @ delta, torch.zeros(5, delta.shape[1], dtype=torch.double), atol=1e-14, rtol=0)
    assert all(v["protected_subspace_relative_residual"] < 1e-12
               for step in info["history"] for v in step["layers"].values())


def test_feature_collection_covers_both_rounds_and_excludes_padding(reader):
    before = tensor_digest(reader.backbone.state_dict())
    atoms = np.asarray([[1, 2, 4], [2, 3, 1]])
    chains = np.asarray([[1, 2, 3, 4]])
    covariance, counts = collect_covariances(reader, (atoms, chains), META, batch_size=2)
    assert all(count == {"calls": 4, "positions": 20} for count in counts.values())
    assert all(v.shape == (64, 64) for v in covariance.values())
    assert tensor_digest(reader.backbone.state_dict()) == before


def test_full_protection_cannot_silently_create_unprotected_learning_direction(reader):
    bases = {k: torch.eye(v.shape[0], dtype=torch.double) for k, v in reader.empty_fast().items()}
    state, _ = atomic_ce_write(reader, [1, 2, 4], META, 0.01, bases=bases)
    assert all(torch.count_nonzero(v) == 0 for v in state.values())


def test_scalar_trust_radius_preserves_projection_and_caps_final_update(reader):
    bases = {k: torch.linalg.qr(torch.randn(v.shape[0], 5, dtype=torch.double)).Q
             for k, v in reader.empty_fast().items()}
    state, _ = atomic_ce_write(reader, [1, 2, 4], META, 10.0, bases=bases, steps=3,
                              max_relative_update=0.001)
    for layer, delta in state.items():
        torch.testing.assert_close(bases[layer].T @ delta, torch.zeros(5, delta.shape[1], dtype=torch.double), atol=1e-14, rtol=0)
        assert delta.norm() <= 0.001 * reader.backbone.transformer.h[layer].mlp.c_proj.weight.norm() * (1 + 1e-12)


def test_atomic_stopping_condition_can_make_zero_updates_without_hiding_it(reader):
    state, info = atomic_ce_write(reader, [1, 2, 4], META, 0.01, steps=20, stop_loss=100.0)
    assert info["steps"] == 0 and info["maximum_steps"] == 20
    assert all(torch.count_nonzero(v) == 0 for v in state.values())


def test_development_selection_never_uses_new_composition_answers():
    grid = [{"learning_rate": 0.01, "atomic": 0.5, "old_retention": 0.99},
            {"learning_rate": 0.1, "atomic": 1.0, "old_retention": 0.8}]
    assert choose_learning_rate(grid)["learning_rate"] == 0.01
    with pytest.raises(ValueError):
        choose_learning_rate([{**grid[0], "new_two_hop_accuracy": 1.0}])


def test_trainable_backbone_is_rejected(reader):
    reader.backbone.transformer.wte.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="Freeze"):
        atomic_ce_write(reader, [1, 2, 4], META, 0.01)
