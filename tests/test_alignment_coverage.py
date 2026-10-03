"""Contracts that separate supervised coverage, coefficient, and input exposure."""

from collections import Counter

import numpy as np
import pytest
import torch

from llm_memory_editability.alignment_coverage import (
    COMPONENTS,
    COVERAGES,
    _coverage_presentations,
    _draw_indices,
    make_specs,
    new_model,
    objective,
    pack_training,
    validate_spec,
)
from llm_memory_editability.grok_depth import EpochStream
from llm_memory_editability.latent_scaling import build_world, model_digest
from llm_memory_editability.representation_alignment import objective as historical_objective
from llm_memory_editability.representation_alignment import pack_training as historical_pack


def small_spec():
    spec = make_specs()[0]
    spec.update(
        heads_n=32,
        bridges_n=32,
        tails_n=16,
        familiar_n=8,
        strict_n=4,
        anchor_n=4,
        composition_count=32,
        width=32,
        batch_size=6,
        steps=4,
        warmup=1,
        nodes=[0, 2, 4],
    )
    return spec


def mixed_batch(coverage):
    arrays, sizes = pack_training(build_world(small_spec()), coverage)
    offsets = np.cumsum([0, *sizes[:-1]])
    indices = np.concatenate([np.arange(offset, offset + 2) for offset in offsets])
    return tuple(torch.as_tensor(a[indices]) for a in arrays)


def graph_operations(loss):
    queue, visited, kinds = [loss.grad_fn], set(), Counter()
    while queue:
        node = queue.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        kinds[type(node).__name__] += 1
        queue.extend(n for n, _ in node.next_functions)
    return kinds


def test_coverage_changes_only_geometry_mask_not_any_input_or_target():
    world = build_world(small_spec())
    reference, expected_sizes = historical_pack(world)
    for coverage in COVERAGES:
        arrays, sizes = pack_training(world, coverage)
        assert sizes == expected_sizes
        for actual, expected in zip(arrays[:3], reference, strict=True):
            np.testing.assert_array_equal(actual, expected)
        masks = np.split(arrays[-1], np.cumsum(sizes)[:-1])
        expected = {
            "none": [0, 0, 0],
            "composition_only": [0, 1, 0],
            "all_atomic_and_composition": [1, 1, 1],
        }[coverage]
        for mask, value in zip(masks, expected, strict=True):
            assert np.all(mask == value)


@pytest.mark.parametrize("coverage", ["none", "all_atomic_and_composition"])
def test_zero_and_full_coverage_recover_historical_loss_and_all_gradients(coverage):
    model, reference = new_model(small_spec(), "cpu"), new_model(small_spec(), "cpu")
    assert model_digest(model) == model_digest(reference)
    tokens, labels, targets, mask = mixed_batch(coverage)
    loss, _ = objective(model, tokens, labels, targets, mask, normalization="batch_mean")
    legacy, _ = historical_objective(
        reference, tokens, labels, targets, 0.3, 0.0 if coverage == "none" else 0.3
    )
    loss.backward()
    legacy.backward()
    torch.testing.assert_close(loss, legacy, rtol=1e-6, atol=1e-7)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)


def test_selected_and_batch_means_expose_exact_coefficient_difference():
    model = new_model(small_spec(), "cpu")
    batch = mixed_batch("composition_only")
    _, selected = objective(model, *batch, normalization="selected_mean")
    _, full = objective(model, *batch, normalization="batch_mean")
    assert selected[COMPONENTS.index("alignment_selected_count")] == 2
    assert selected[COMPONENTS.index("alignment_denominator")] == 2
    assert full[COMPONENTS.index("alignment_denominator")] == 6
    torch.testing.assert_close(selected[2], 3 * full[2])
    selected_grad = torch.autograd.grad(selected[2], list(model.parameters()), allow_unused=True)
    full_grad = torch.autograd.grad(full[2], list(model.parameters()), allow_unused=True)
    for a, b in zip(selected_grad, full_grad, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, 3 * b, rtol=1e-5, atol=2e-7)


def test_unselected_atomic_targets_cannot_contribute_to_geometry():
    model = new_model(small_spec(), "cpu")
    tokens, labels, targets, mask = mixed_batch("composition_only")
    changed = targets.clone()
    changed[mask == 0] = (changed[mask == 0] + 1) % model.config.vocab_size
    _, before = objective(model, tokens, labels, targets, mask)
    _, after = objective(model, tokens, labels, changed, mask)
    torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    ga = torch.autograd.grad(before[2], list(model.parameters()), allow_unused=True)
    gb = torch.autograd.grad(after[2], list(model.parameters()), allow_unused=True)
    for a, b in zip(ga, gb, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_empty_mask_is_finite_and_all_arms_keep_same_autograd_operations():
    graphs = []
    for coverage in COVERAGES:
        model = new_model(small_spec(), "cpu")
        loss, parts = objective(model, *mixed_batch(coverage))
        assert torch.isfinite(parts).all()
        if coverage == "none":
            assert parts[2] == 0 and parts[6] == 0 and parts[7] == 1
        graphs.append(graph_operations(loss))
    assert graphs[0] == graphs[1] == graphs[2]


def test_stream_and_exposure_are_paired_while_geometry_coverage_is_explicit():
    world = build_world(small_spec())
    outputs, coverages = [], []
    for coverage in COVERAGES:
        arrays, sizes = pack_training(world, coverage)
        streams = [EpochStream(n, 71 + j) for j, n in enumerate(sizes)]
        counts = [np.zeros(n, dtype=np.int64) for n in sizes]
        indices = _draw_indices(streams, np.cumsum([0, *sizes[:-1]]), counts, 5, 6)
        outputs.append(indices)
        coverages.append(_coverage_presentations(counts, arrays[-1], sizes))
        assert [int(c.sum()) for c in counts] == [10, 10, 10]
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[1], outputs[2])
    assert [c["total"] for c in coverages] == [0, 10, 30]
    assert coverages[1] == {
        "common_atomic": 0,
        "train_composite": 10,
        "anchor_atomic": 0,
        "total": 10,
    }


def test_proposed_matrices_preserve_pairing_and_do_not_alias_nodes():
    assert len(make_specs("development")) == 4
    confirmation = make_specs("confirmation")
    assert len(confirmation) == 24
    for start in range(0, len(confirmation), 4):
        group = confirmation[start : start + 4]
        comparable = []
        for spec in group:
            validate_spec(spec)
            comparable.append(
                {
                    k: v
                    for k, v in spec.items()
                    if k not in ("arm", "alignment_coverage", "alignment_normalization")
                }
            )
        assert all(spec == comparable[0] for spec in comparable)
    specs = make_specs()
    specs[0]["nodes"].append(17000)
    assert specs[1]["nodes"][-1] == 16000


def test_invalid_normalization_is_rejected_before_training():
    spec = small_spec()
    spec["alignment_normalization"] = "implicit-default"
    with pytest.raises(ValueError, match="normalization"):
        validate_spec(spec)
