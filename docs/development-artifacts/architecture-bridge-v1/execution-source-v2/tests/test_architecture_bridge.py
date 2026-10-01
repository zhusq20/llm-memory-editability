"""Contracts that distinguish real channel interventions from padding/artifacts."""

import copy
from types import SimpleNamespace

import torch

from llm_memory_editability.architecture_bridge import (
    ART,
    DATA,
    Engine,
    change_cache,
    grade,
    matched_random,
    read,
)


def test_generation_grading_is_not_substring_accuracy():
    assert grade("Paris and Rome", "Paris")["em"] is False
    assert grade("Paris\nRome", "Paris")["em"] is False
    assert grade("The United Kingdom.", "United Kingdom")["em"] is True
    assert grade("UK", "United Kingdom", ["UK"])["alias_em"] is True


def test_state_replacement_is_layer_local_and_norm_matched():
    first = SimpleNamespace(
        conv_states=[torch.zeros(1, 3, 2)], recurrent_states=[torch.ones(1, 2, 2)]
    )
    second = SimpleNamespace(
        conv_states=[torch.ones(1, 3, 2)], recurrent_states=[torch.zeros(1, 2, 2)]
    )
    base = SimpleNamespace(layers=[first, copy.deepcopy(first)])
    donor = SimpleNamespace(layers=[second, copy.deepcopy(second)])
    random = copy.deepcopy(base)
    change_cache(random, donor, 0, 123)
    assert torch.equal(random.layers[1].conv_states[0], first.conv_states[0])
    for field in ["conv_states", "recurrent_states"]:
        a, b, c = [getattr(obj, field)[0] for obj in [first, second, random.layers[0]]]
        torch.testing.assert_close((c - a).norm(), (b - a).norm())
    change_cache(base, donor, 0)
    assert torch.equal(base.layers[0].recurrent_states[0], second.recurrent_states[0])


def test_kv_shapes_cannot_be_silently_broadcast():
    target = SimpleNamespace(
        layers=[SimpleNamespace(keys=torch.zeros(1, 2, 3, 4), values=torch.zeros(1, 2, 3, 4))]
    )
    donor = SimpleNamespace(
        layers=[SimpleNamespace(keys=torch.zeros(1, 2, 1, 4), values=torch.zeros(1, 2, 1, 4))]
    )
    try:
        change_cache(target, donor, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal donor position count must fail")


def test_mlp_patch_changes_only_selected_position_and_removes_hook():
    engine = Engine.__new__(Engine)
    engine.layers = [SimpleNamespace(mlp=torch.nn.Identity())]
    x = torch.randn(1, 5, 3)
    replacement = torch.full((3,), 7.0)
    with engine.mlp_hook(0, replacement):
        changed = engine.layers[0].mlp(x)
    torch.testing.assert_close(changed[:, :-1], x[:, :-1])
    torch.testing.assert_close(changed[0, -1], replacement)
    torch.testing.assert_close(engine.layers[0].mlp(x), x)


def test_zero_random_direction_and_reproducibility():
    assert matched_random(torch.zeros(12), 3).norm() == 0
    d = torch.randn(12)
    torch.testing.assert_close(matched_random(d, 3), matched_random(d, 3))
    torch.testing.assert_close(matched_random(d, 3).norm(), d.norm())


def test_frozen_cases_roles_and_missingness_are_accounted_for():
    if not (ART / "data-lock.json").exists():
        return
    rows, pools = read(DATA / "cases.json"), read(DATA / "learning.json")
    assert len(rows) == 56
    for dataset in ["mquake", "2wiki"]:
        groups = []
        for split, count in [("development", 4), ("evaluation", 24)]:
            part = [r for r in rows if r["dataset"] == dataset and r["split"] == split]
            assert len(part) == count
            groups.append({r["group"] for r in part})
        assert not groups[0] & groups[1]
    roles = [{r["group"] for r in pools[role]} for role in ["E", "R", "U"]]
    assert not roles[0] & roles[1] and not roles[0] & roles[2] and not roles[1] & roles[2]
