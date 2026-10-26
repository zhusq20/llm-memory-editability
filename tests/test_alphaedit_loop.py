"""Contracts for AlphaEdit's solve, tied execution, atomic-only targets and freeze."""

import copy

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.alphaedit_loop import (
    ExecutionTrace,
    alphaedit_update,
    atomic_arrays,
    collect_old_covariances,
    fit_target,
    null_basis,
    select_l2,
    write_target,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest

META = {"entity_offset": 0, "relation_offset": 10, "end_marker": 28, "pad_token": 27}
SETTINGS = {
    "v_lr": 0.5,
    "v_num_grad_steps": 12,
    "v_weight_decay": 0,
    "kl_factor": 0.0625,
    "clamp_norm_factor": 0.75,
    "stop_loss": 0.05,
}


@pytest.fixture
def reader():
    torch.set_num_threads(1)
    torch.manual_seed(20261016)
    cfg = GPT2Config(
        vocab_size=29,
        n_positions=16,
        n_embd=16,
        n_head=2,
        n_layer=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    return TTCDReader(
        SupervisedAlignmentGPT(cfg, repeats=2, discoloop=True).double(),
        native_causal_attention=True,
    ).eval()


@pytest.mark.parametrize("requests", [1, 3])
def test_reduced_solve_equals_author_solve_in_gpt_orientation(requests):
    torch.manual_seed(7)
    n = torch.linalg.qr(torch.randn(9, 4, dtype=torch.double)).Q
    keys, residual = (
        torch.randn(9, requests, dtype=torch.double),
        torch.randn(5, requests, dtype=torch.double),
    )
    p, l2 = n @ n.T, 10.0
    expected = torch.linalg.solve(p @ (keys @ keys.T) + l2 * torch.eye(9), p @ keys @ residual.T)
    actual = alphaedit_update(keys, residual, n, l2)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    old = torch.randn(6, 9, dtype=torch.double) @ (torch.eye(9) - p)
    torch.testing.assert_close(
        old @ actual, torch.zeros(6, 5, dtype=torch.double), atol=1e-14, rtol=0
    )


def test_numerical_null_space_uses_absolute_author_threshold():
    cov = torch.diag(torch.tensor([100.0, 0.021, 0.019, 0.0], dtype=torch.double))
    basis, detail = null_basis(cov, 0.02)
    assert detail["null_dimension"] == 2 and detail["protected_dimension"] == 2
    torch.testing.assert_close(
        basis @ basis.T, torch.diag(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.double))
    )


def test_trace_matches_native_block_state_and_does_not_change_logits(reader):
    tokens, positions, _ = atomic_arrays([[1, 2, 4]], META, "cpu")
    fast = {i: torch.randn_like(a) * 0.001 for i, a in reader.empty_fast().items()}
    states = []
    hook = reader.backbone.transformer.ln_f.register_forward_pre_hook(
        lambda module, args: states.append(args[0].detach().clone())
    )
    plain = reader(tokens, positions, fast=fast)
    states.clear()
    with ExecutionTrace(reader, fast) as trace:
        observed = reader(tokens, positions, fast=fast)
    hook.remove()
    assert torch.equal(plain, observed)
    assert trace.calls == {0: 2, 1: 2}
    assert torch.equal(trace.states[(1, 1)], states[-1][:, 1])


def test_latent_intervention_selects_one_virtual_execution(reader):
    tokens, positions, _ = atomic_arrays([[1, 2, 4]], META, "cpu")
    with ExecutionTrace(reader) as baseline:
        reader(tokens, positions)
    delta = torch.ones(16, dtype=torch.double) * 0.1
    with ExecutionTrace(reader, delta=delta, target=(1, 1)) as edited:
        reader(tokens, positions)
    for execution in ((0, 0), (0, 1), (1, 0)):
        assert torch.equal(baseline.states[execution], edited.states[execution])
    torch.testing.assert_close(edited.states[(1, 1)] - baseline.states[(1, 1)], delta[None])


def test_shared_down_updates_equal_materialized_native_weights(reader):
    tokens, positions, _ = atomic_arrays([[1, 2, 4]], META, "cpu")
    fast = {i: torch.randn_like(a) * 0.01 for i, a in reader.empty_fast().items()}
    materialized = copy.deepcopy(reader.backbone)
    with torch.no_grad():
        for i, block in enumerate(materialized.transformer.h):
            block.mlp.c_proj.weight.add_(fast[i])
    torch.testing.assert_close(
        reader(tokens, positions, fast=fast),
        materialized(tokens, positions),
        atol=1e-12,
        rtol=1e-10,
    )


def test_latent_fit_uses_atomic_ce_without_backbone_gradients_or_mutation(reader):
    before = tensor_digest(reader.backbone.state_dict())
    target, info = fit_target(reader, [1, 2, 4], [[1, 3, 2]], META, SETTINGS)
    assert info["history"][-1]["nll"] < info["history"][0]["nll"]
    assert info["latent_updates"] <= SETTINGS["v_num_grad_steps"] - 1
    assert info["delta_norm"] <= SETTINGS["clamp_norm_factor"] * info["initial_norm"] + 1e-12
    assert info["new_composition_labels"] == 0 and target.shape == (16,)
    assert before == tensor_digest(reader.backbone.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in reader.parameters())


def test_writer_rejects_composite_training_rows():
    with pytest.raises(ValueError, match="atomic triples"):
        atomic_arrays([[1, 2, 3, 4]], META, "cpu")


def test_no_null_directions_means_no_parameter_edit(reader):
    target, _ = fit_target(reader, [1, 2, 4], [], META, SETTINGS)
    bases = {
        i: torch.empty(a.shape[0], 0, dtype=torch.double) for i, a in reader.empty_fast().items()
    }
    state, info = write_target(reader, [1, 2, 4], META, target, bases, 10)
    assert all(torch.count_nonzero(a) == 0 for a in state.values())
    assert info["down_matrix_updates"] == 2


def test_covariance_covers_both_rounds_and_all_valid_positions(reader):
    cov, counts = collect_old_covariances(
        reader, [np.asarray([[1, 2, 4], [2, 3, 1]]), np.asarray([[1, 2, 3, 4]])], META, 8
    )
    assert all(c == {"calls": 4, "positions": 20} for c in counts.values())
    assert all(c.shape == (64, 64) and torch.isfinite(c).all() for c in cov.values())
    assert all(torch.allclose(c, c.T) for c in cov.values())


def test_selection_prioritizes_retention_without_composition_scores():
    records = [
        {"l2": 10, "atomic_accuracy": 0.75, "minimum_old_retention": 1.0},
        {"l2": 1, "atomic_accuracy": 1.0, "minimum_old_retention": 0.95},
        {"l2": 0.1, "atomic_accuracy": 0.75, "minimum_old_retention": 0.99},
    ]
    assert select_l2(records, 0.99)["l2"] == 10
    assert select_l2(records, 0.99)["selection_new_composition_queries"] == 0


def test_writer_never_changes_non_down_parameters(reader):
    before = tensor_digest(reader.backbone.state_dict())
    target, _ = fit_target(reader, [1, 2, 4], [], META, SETTINGS)
    bases = {i: torch.eye(a.shape[0], dtype=torch.double) for i, a in reader.empty_fast().items()}
    state, info = write_target(reader, [1, 2, 4], META, target, bases, 10)
    assert any(a.norm() > 0 for a in state.values())
    assert len(info["layers"]) == 2 and before == tensor_digest(reader.backbone.state_dict())
    assert all(a["key_execution"][0] == 1 for a in info["layers"])


def test_full_boundary_observer_preserves_native_logits_and_exact_reconstruction(reader):
    from check_alphaedit_edit_rounds import inspect_rows

    reader.float()
    fast = {i: torch.randn_like(a) * 0.01 for i, a in reader.empty_fast().items()}
    before = tensor_digest(reader.backbone.state_dict())
    report = inspect_rows(reader, fast, [[1, 2, 3, 4]], META, [4], kind="second_hop", bridges=[3])
    assert report["hook_logits_exact"]
    assert report["bridge_channel"]["boundary_reconstruction_exact"]
    assert before == tensor_digest(reader.backbone.state_dict())
