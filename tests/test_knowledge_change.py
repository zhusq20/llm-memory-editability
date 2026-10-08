import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import interface_editing as editing
from llm_memory_editability.knowledge_change import episode_rows, panel, set_reader_scope
from llm_memory_editability.latent_scaling import build_world
from llm_memory_editability.shared_cache import construct
from llm_memory_editability.storage_composition import pack_sentences


@pytest.fixture
def spec():
    path = Path("docs/development-artifacts/shared-cache-branch-v1/frozen-config.json")
    return copy.deepcopy(
        next(
            s
            for s in json.loads(path.read_text())["specs"]
            if s["world"] == 730011 and s["memory_arm"] == "local" and s["repeats"] == 4
        )
    )


def test_update_addresses_disjoint_and_model_independent(spec):
    p = panel(spec)
    train = {tuple(c["old_fact"][:2]) for c in p["train"]}
    test = {tuple(c["old_fact"][:2]) for c in p["test"]}
    assert len(p["train"]) == 8 and len(p["test"]) == 16
    assert not train & test
    assert p == panel(dict(spec, memory_arm="shared_full", initialization=999))


@pytest.mark.parametrize("world", [107270011, 107270012, 107270013])
def test_registered_new_worlds_qualify(spec, world):
    assert len(panel(dict(spec, world=world))["train"]) == 8


@pytest.mark.parametrize("role", ["first", "second"])
def test_current_truth_and_no_heldout_query_labels(spec, role):
    world = build_world(spec)
    original = {k: v.copy() for k, v in world.items()}
    c = next(c for c in panel(spec)["train"] if c["role"] == role)
    fixed, f = episode_rows(world, c, "fixed")
    changed, d = episode_rows(world, c, "changed_use")
    atom_only, a = episode_rows(world, c, "changed_atomic")
    assert not f["direct_atomic_changed"] and d["direct_atomic_changed"]
    assert d["changed_training_answers"] > 0 and a["changed_training_answers"] == 0
    np.testing.assert_array_equal(fixed[0][0], c["old_fact"])
    np.testing.assert_array_equal(changed[0][0], c["new_fact"])
    np.testing.assert_array_equal(changed[0], atom_only[0])
    # Same queries and prefixes, changed answer supervision only.
    fx, _ = pack_sentences(fixed[2])
    dx, _ = pack_sentences(changed[2])
    np.testing.assert_array_equal(fx[:, :7], dx[:, :7])
    assert not editing.affected(atom_only[2], c["old_fact"], role).any()
    trained_queries = {tuple(r[[0, 1, 3]]) for r in world["train_composite"]}
    heldout = {tuple(r[[0, 1, 3]]) for r in world["familiar_test"]}
    for strata in [fixed, changed, atom_only]:
        for rows in strata[2:]:
            assert {tuple(r[[0, 1, 3]]) for r in rows} <= trained_queries
            assert not {tuple(r[[0, 1, 3]]) for r in rows} & heldout
    for name in world:
        np.testing.assert_array_equal(world[name], original[name])


def test_reader_scope_and_atomic_scope_are_disjoint(spec):
    torch.set_num_threads(1)
    model = construct(spec, "cpu")
    writer = set(editing.editable_parameters(model))
    set_reader_scope(model)
    reader = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not writer & reader
    assert writer | reader == {n for n, _ in model.named_parameters()}


def test_unknown_arm_rejected(spec):
    with pytest.raises(ValueError):
        episode_rows(build_world(spec), panel(spec)["train"][0], "unknown")
