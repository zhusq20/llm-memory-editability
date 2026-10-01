"""Contract tests for causal evaluation, paired exposure and channel patches."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.architecture_bridge_toy import (
    GatedDelta,
    ToyLM,
    diagnostics,
    evaluate,
    exposure,
    initialize,
)
from llm_memory_editability.twohop_depth import (
    ENTITY,
    EOS,
    audit_world,
    make_stream,
    training_tensors,
    world_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/architecture-bridge-toy-v1.json").read_text())
torch.set_num_threads(2)


def model_for(name):
    arch = next(arch for arch in CFG["architectures"] if arch["name"] == name)
    model = ToyLM(arch, CFG)
    initialize(model, 77)
    return model.eval()


@pytest.mark.parametrize("name", ["gqa", "hybrid"])
def test_future_tokens_do_not_change_prefix_logits(name):
    model = model_for(name)
    x = torch.tensor([[1, 5, 36, 37, 9, EOS], [1, 8, 35, 38, 11, EOS]])
    altered = x.clone()
    altered[:, 3:] = torch.tensor([16, 12, 0])
    torch.testing.assert_close(model(x)[:, :3], model(altered)[:, :3], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["gqa", "hybrid"])
def test_cache_continuation_matches_full_sequence(name):
    model = model_for(name)
    x = torch.tensor([[1, 5, 36, 37, 9, EOS], [1, 8, 35, 38, 11, EOS]])
    full = model(x)
    _, cache, _ = model(x[:, :3], return_cache=True)
    continuation, _, _ = model(x[:, 3:], cache=cache, return_cache=True)
    torch.testing.assert_close(full[:, 3:], continuation, rtol=2e-5, atol=2e-6)


def test_delta_state_and_convolution_both_matter_to_continuation():
    model = model_for("hybrid")
    mixer = next(block.mixer for block in model.blocks if isinstance(block.mixer, GatedDelta))
    rng = torch.Generator().manual_seed(192)
    x = torch.randn(2, 5, CFG["width"], generator=rng)
    _, cache = mixer(x[:, :3])
    expected, _ = mixer(x[:, 3:], cache)
    for component in ("state", "conv"):
        modified = {key: value.clone() for key, value in cache.items()}
        modified[component].zero_()
        changed, _ = mixer(x[:, 3:], modified)
        assert not torch.allclose(expected, changed, atol=1e-8, rtol=1e-5)


def test_world_split_donors_balancing_and_stream_pairing():
    for seed in CFG["worlds"]:
        world = world_arrays(seed, CFG)
        audit_world(world, CFG)
        train = {tuple(v) for v in world["comps"][world["train_mask"]]}
        test = {tuple(v) for v in world["comps"][~world["train_mask"]]}
        assert not train & test
        assert len(train) == len(test) == 256
        for init in CFG["initializations"]:
            left, right = make_stream(world, init, CFG), make_stream(world, init, CFG)
            for key in left:
                assert np.array_equal(left[key], right[key])
                assert len(left[key]) == CFG["steps"] * CFG["batch_per_kind"]


def test_common_initial_parameters_are_bit_identical():
    models = [model_for(name) for name in ("gqa", "hybrid")]
    left, right = [dict(model.named_parameters()) for model in models]
    common = [key for key in left if key in right and left[key].shape == right[key].shape]
    assert all(torch.equal(left[key], right[key]) for key in common)
    sizes = [sum(p.numel() for p in model.parameters()) for model in models]
    assert max(sizes) / min(sizes) < 1.1


def test_training_labels_include_answer_then_eos_and_no_test_compositions():
    world = world_arrays(CFG["worlds"][0], CFG)
    tensors = training_tensors(world, "cpu")
    assert len(tensors["composite"][0]) == world["train_mask"].sum()
    for tokens, positions, labels in tensors.values():
        assert torch.all(labels[:, 1] == EOS)
        row = torch.arange(len(tokens))
        assert torch.equal(tokens[row, positions[:, 0] + 1], labels[:, 0])
        assert torch.equal(tokens[row, positions[:, 1] + 1], labels[:, 1])
    counts = exposure(CFG, CFG["steps"])
    assert counts["valid_input_tokens"] == 360448
    assert counts["supervised_answer_eos_tokens"] == 131072


@pytest.mark.parametrize("name", ["gqa", "hybrid"])
def test_identity_patch_and_random_norm_controls(name):
    model = model_for(name)
    world = world_arrays(CFG["worlds"][0], CFG)
    metrics, arrays = diagnostics(model, world, CFG, "cpu")
    assert len(metrics) == 16
    for target in CFG["patch_targets"]:
        np.testing.assert_allclose(
            arrays[f"{target}_identity_logits"], arrays["baseline_logits"], atol=1e-7
        )
        assert np.array_equal(arrays[f"{target}_identity_eos"], arrays["baseline_eos"])
        for donor in ("correct", "wrong"):
            for kind in ("mlp", "mixer") if target == "joint" else (target,):
                np.testing.assert_allclose(
                    arrays[f"{target}_{donor}_{kind}_norm"],
                    arrays[f"{target}_random_{donor}_{kind}_norm"],
                    rtol=1e-5,
                    atol=1e-8,
                )


def test_evaluation_requires_eos_and_uses_autonomous_bridge():
    model = model_for("gqa")
    world = world_arrays(CFG["worlds"][0], CFG)
    metrics, arrays = evaluate(model, world, CFG, "cpu")
    y = world["atomic_y"]
    expected = ((arrays["atomic_pred"] == y) & (arrays["atomic_eos"] == EOS)).mean()
    assert metrics["atomic"]["exact_answer_eos"] == expected
    a, r1, _ = world["comps"][~world["train_mask"]].T
    assert np.array_equal(
        arrays["two_call_bridge"], arrays["atomic_pred"][a * CFG["relations"] + r1]
    )
    assert model.token.num_embeddings == ENTITY + CFG["entities"] + CFG["relations"]
