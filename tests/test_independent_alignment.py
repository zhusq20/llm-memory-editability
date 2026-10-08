"""Contracts that distinguish ordinary-layer replication from shared computation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grok_depth import SmallGPT
from llm_memory_editability.independent_alignment import diagnose, validate_model
from llm_memory_editability.latent_scaling import build_world, model_digest
from llm_memory_editability.representation_alignment import new_model, objective, pack_training


def small_spec():
    return dict(
        world=107270899,
        initialization=107270898,
        stream_seed=776101,
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
        layers=2,
        repeats=1,
        dropout=0.0,
    )


def test_independent_storage_and_forward_equal_ordinary_small_gpt():
    spec = small_spec()
    model = new_model(spec, "cpu")
    validate_model(spec, model)
    torch.manual_seed(spec["initialization"])
    reference = SmallGPT(model.config, dropout=0.0)
    assert model_digest(model) == model_digest(reference)
    tokens = torch.tensor([[2, 21, 3, 13, 3, 17, 4, 65]])
    torch.testing.assert_close(model(tokens), reference(tokens), atol=0, rtol=0)
    _, before = model(tokens, return_bridge=True)
    with torch.no_grad():
        model.blocks[1].mlp.down.weight.add_(0.1)
    _, after = model(tokens, return_bridge=True)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_shared_model_rejected():
    spec = dict(small_spec(), layers=1, repeats=2)
    with pytest.raises(ValueError):
        validate_model(spec, new_model(spec, "cpu"))


def test_first_layer_state_has_no_future_information():
    model = new_model(small_spec(), "cpu")
    a = torch.tensor([[2, 21, 3, 13, 3, 17, 4, 65]])
    b = torch.tensor([[2, 21, 3, 13, 4, 20, 5, 67]])
    _, x = model(a, return_bridge=True)
    _, y = model(b, return_bridge=True)
    _, prefix = model(a[:, :4], return_bridge=True)
    torch.testing.assert_close(x, y, atol=0, rtol=0)
    torch.testing.assert_close(x, prefix, atol=2e-7, rtol=2e-6)


def test_strict_atoms_have_no_composition_training_and_aux_labels_match():
    world = build_world(small_spec())
    trained = {(int(h), int(r1), int(b)) for h, r1, b, r2, t in world["train_composite"]}
    trained |= {(int(b), int(r2), int(t)) for h, r1, b, r2, t in world["train_composite"]}
    strict = {(int(h), int(r1), int(b)) for h, r1, b, r2, t in world["strict_test"]}
    strict |= {(int(b), int(r2), int(t)) for h, r1, b, r2, t in world["strict_test"]}
    assert not strict & trained
    assert strict <= set(map(tuple, world["common_atomic"]))
    (tokens, labels, targets), sizes = pack_training(world)
    first, last = sizes[0], sizes[0] + sizes[1]
    np.testing.assert_array_equal(targets[first:last], world["train_composite"][:, 2])
    assert not np.any(tokens[first:last] == targets[first:last, None])


def test_alignment_only_changes_the_specified_geometric_term():
    model = new_model(small_spec(), "cpu")
    arrays, _ = pack_training(build_world(small_spec()))
    batch = [torch.as_tensor(a[:8]) for a in arrays]
    ce, parts = objective(model, *batch, 0.3, 0.0)
    aligned, other = objective(model, *batch, 0.3, 0.3)
    torch.testing.assert_close(parts, other, atol=0, rtol=0)
    torch.testing.assert_close(aligned - ce, 0.3 * parts[2], atol=1e-6, rtol=1e-6)


def test_pure_prefix_diagnostic_does_not_use_final_answers():
    model = new_model(small_spec(), "cpu")
    world = build_world(small_spec())
    _, prediction = diagnose(model, world, "cpu")
    other = {k: v.copy() for k, v in world.items()}
    for task in ["familiar_test", "strict_test"]:
        other[task][:, 3:] += 1
    _, altered = diagnose(model, other, "cpu")
    for key, value in prediction.items():
        np.testing.assert_array_equal(value, altered[key])


def test_matrix_is_two_same_label_arms_on_three_fresh_worlds():
    path = Path(__file__).parents[1] / "scripts/run_independent_alignment.py"
    loader = importlib.util.spec_from_file_location("independent_runner", path)
    script = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(script)
    specs = script.specifications()
    assert len(specs) == 14
    confirmation = [s for s in specs if s["phase"] == "confirmation"]
    assert len(confirmation) == 12
    assert len({s["world"] for s in confirmation}) == 3
    assert not {s["world"] for s in confirmation} & {774101, 774102, 774103, 770011}
    for world, initialization in {(s["world"], s["initialization"]) for s in specs}:
        pair = [s for s in specs if (s["world"], s["initialization"]) == (world, initialization)]
        a, b = pair
        for k in a.keys() - {"arm", "name", "alignment_weight"}:
            assert a[k] == b[k], k
        assert a["layers"] == 2 and a["repeats"] == 1
        assert a["steps"] == 16000
        assert a["bridge_weight"] == 0.3
        assert {a["alignment_weight"], b["alignment_weight"]} == {0.0, 0.3}
