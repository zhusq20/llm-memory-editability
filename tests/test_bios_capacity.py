"""Prevent heldout answer leakage and misleading conditional accuracy."""

import numpy as np

from llm_memory_editability.bios_capacity import qa_split, split_metrics
from llm_memory_editability.bios_data import N_BASE, N_QUERIES, load_world
from llm_memory_editability.bios_organization_train import derived_schedule


def test_split_and_schedule_exclude_heldout_queries():
    world = load_world("data/bios-organization-v1/world-0")
    train, heldout = qa_split(world, "half")
    assert len(train) == len(heldout) == 1024
    assert not np.intersect1d(train, heldout).size
    np.testing.assert_array_equal(np.sort(np.r_[train, heldout]), np.arange(N_BASE, N_QUERIES))
    for company in range(64):
        people = train - N_BASE
        selected = people[world.employers[people] == company]
        assert len(selected) == 16
        assert world.exceptions[selected].sum() == 1
    schedule = derived_schedule(world.seed, 896, train)
    assert np.isin(schedule, train).all()
    assert not np.isin(schedule, heldout).any()
    np.testing.assert_array_equal(schedule, derived_schedule(world.seed, 1792, train)[:896])
    exposure = np.bincount(schedule.ravel(), minlength=N_QUERIES)
    assert exposure[heldout].sum() == 0
    assert np.ptp(exposure[train]) <= 1


def test_component_conditioning_requires_both_facts():
    world = load_world("data/bios-organization-v1/world-0")
    train, heldout = qa_split(world, "half")
    correct = np.ones(N_QUERIES, dtype=bool)
    metrics = split_metrics(
        world, {"correct": correct, "value_nll": np.zeros(N_QUERIES)}, train, heldout
    )
    assert metrics["derived_heldout_known_components_count"] == 1024
    correct[world.relation == 1] = False
    metrics = split_metrics(
        world, {"correct": correct, "value_nll": np.zeros(N_QUERIES)}, train, heldout
    )
    assert metrics["derived_heldout_accuracy"] == 1
    assert metrics["derived_heldout_known_components_count"] == 0
    assert metrics["derived_heldout_known_components_accuracy"] is None
