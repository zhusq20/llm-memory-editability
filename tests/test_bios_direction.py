"""Contracts that affect the interpretation of the direction experiment."""

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_direction import (
    Suffix,
    batch_setup,
    edit_batch,
    inverse_sketch,
    minimal_task,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("kind", ["selective", "coherent", "independent"])
def test_targets_and_disjoint_information(seed, kind):
    t = minimal_task(seed, 1, kind, "cpu")
    union = sum((t.sets[x] for x in ("E", "R", "V", "U")), [])
    assert len(union) == len(set(union)) == 96
    assert len(t.sets["E"]) == (1 if kind == "selective" else 2)
    assert torch.all(t.labels[t.sets["E"]] != t.old_labels[t.sets["E"]])
    assert torch.equal(t.labels[t.sets["R"]], t.old_labels[t.sets["R"]])
    i, j = t.metadata["pair"]
    assert int(t.labels[j]) == (t.metadata["c"] if kind == "selective" else t.metadata["a"])


def test_cached_suffix_forward_and_parameter_gradient():
    torch.manual_seed(8)
    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(41, width=8, layers=3, heads=2, context=8)).double()
    for p in model.parameters():
        p.requires_grad_(False)
    w = model.blocks[0].mlp.down.weight
    w.requires_grad_(True)
    task = minimal_task(0, 0, "independent", "cpu")
    expected = model(task.tokens)[:, -1:]
    true = torch.autograd.grad(expected.square().sum(), w)[0]
    suffix = Suffix(model, 0, task.tokens, task.positions)
    candidate = w.detach()[None].clone().requires_grad_()
    actual = suffix(candidate)
    computed = torch.autograd.grad(actual.square().sum(), candidate)[0][0]
    torch.testing.assert_close(actual[0], expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(computed, true, rtol=1e-10, atol=1e-12)


def test_woodbury_matches_dense_system():
    torch.manual_seed(9)
    u = torch.randn(2, 3, 8, dtype=torch.float64)
    v = torch.randn(2, 2, 4, dtype=torch.float64)
    actual = inverse_sketch(v, u, 0.1).flatten(1)
    h = 0.1 * torch.eye(8, dtype=torch.float64) + u.transpose(-1, -2) @ u
    expected = torch.linalg.solve(h, v.flatten(1)[..., None])[..., 0]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_softmax_probe_covariance_is_ggn():
    p = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
    b = torch.diag(p.sqrt()) - p[:, None] * p.sqrt()[None, :]
    torch.testing.assert_close(b @ b.T, torch.diag(p) - p[:, None] * p[None, :])


def test_nullspace_protects_all_prefix_positions():
    torch.manual_seed(10)
    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(41, width=8, layers=2, heads=2, context=8)).double()
    for p in model.parameters():
        p.requires_grad_(False)
    task = minimal_task(0, 0, "independent", "cpu")
    suffix, labels, ew, rw, ref, cov, null, ranks, spectrum, w0 = batch_setup(
        model, 0, [task], [{}]
    )
    change = 0.001 * torch.randn_like(w0)[None] @ null
    logits = suffix(w0[None] + change)
    protected = rw.bool()
    torch.testing.assert_close(
        logits.log_softmax(-1)[protected], ref[protected], rtol=1e-7, atol=1e-7
    )


def test_actual_hard_updates_preserve_parent_and_frozen_parameters(tmp_path):
    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(41, width=8, layers=2, heads=2, context=8))
    for p in model.parameters():
        p.requires_grad_(False)
    parent = {k: v.clone() for k, v in model.state_dict().items()}
    task = minimal_task(0, 0, "independent", "cpu")
    cfg = dict(lr=0.001, level=0.0, damping=0.01, rank=2, refresh=2, trust_fraction=0.05)
    edit_batch(model, 0, [task], "repr-hard-adam", [cfg], 4, tmp_path)
    trace = np.load(tmp_path / "steps.npz")["values"]
    assert np.all(trace[:, :, 3] <= 1e-10)
    for key, value in model.state_dict().items():
        assert torch.equal(value, parent[key])
