"""World balancing and genuine learning-rate / freeze interventions."""

import copy

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_direction import minimal_task
from llm_memory_editability.bios_direction_formation import (
    factor_measure,
    optimizer_views,
    world_labels,
)
from llm_memory_editability.bios_model import CausalLM, ModelConfig


def test_new_worlds_have_actual_differences_and_equal_marginals():
    values = []
    for world in range(4000, 4004):
        labels = world_labels(world).numpy().reshape(32, 3) - 38
        assert (labels[:, 0] == labels[:, 1]).sum() == 24
        assert np.all(labels[:3, 0] == labels[:3, 1])
        np.testing.assert_array_equal(np.bincount(labels[:, 0]), np.bincount(labels[:, 1]))
        values.append(labels.tobytes())
    assert len(set(values)) == 4


def test_optimizer_views_match_standard_and_freeze_states():
    torch.manual_seed(71)
    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(41, width=64, layers=2, heads=2, context=8))
    reference = copy.deepcopy(model)
    views, mapping = optimizer_views(model)
    ordinary = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0.1, foreach=True)
    task = minimal_task(0, 0, "independent", "cpu")
    for m in (model, reference):
        F.cross_entropy(m(task.tokens)[:, -1], task.old_labels[:, 0]).backward()
    for _group, p, region, view in mapping:
        view.grad = p.grad[region]
    views.step()
    ordinary.step()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p, q)
    snapshots = [
        (view.detach().clone(), views.state[view]["step"].clone())
        for group, p, region, view in mapping
        if group == "qk"
    ]
    for group, p, region, view in mapping:
        view.grad = None if group == "qk" else p.grad[region]
    views.step()
    frozen = [view for group, p, region, view in mapping if group == "qk"]
    for view, (value, step) in zip(frozen, snapshots, strict=True):
        assert torch.equal(view, value) and torch.equal(views.state[view]["step"], step)


def test_factorization_reconstructs_actual_gradient():
    torch.manual_seed(72)
    torch.set_num_threads(1)
    model = CausalLM(ModelConfig(41, width=64, layers=2, heads=2, context=8))
    rows = factor_measure(model, 0, world_labels(4000), "cpu")
    assert max(r["reconstruction_error"] for r in rows) < 2e-6
