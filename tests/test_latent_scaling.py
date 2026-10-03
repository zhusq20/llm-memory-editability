"""Contracts for sample scaling, shared parameter control and evaluation truth."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grok_depth import EpochStream, SmallGPT
from llm_memory_editability.latent_scaling import build_world, construct, model_digest, run_name
from llm_memory_editability.storage_composition import audit_world, pack_sentences


@pytest.fixture
def spec():
    return {
        "world": 103,
        "heads_n": 32,
        "bridges_n": 32,
        "tails_n": 16,
        "familiar_n": 8,
        "strict_n": 4,
        "holdout_fraction": 0.25,
        "low_extra": "anchors",
        "anchor_n": 4,
        "composition_count": 16,
        "width": 32,
        "layers": 1,
        "repeats": 2,
        "heads": 4,
        "dropout": 0.0,
        "initialization": 104,
        "stream_seed": 105,
        "steps": 128000,
    }


def test_sample_scale_changes_only_nested_training_compositions(spec):
    a = build_world(spec)
    b = build_world(dict(spec, composition_count="all"))
    audit_world(a)
    audit_world(b)
    np.testing.assert_array_equal(a["train_composite"], b["train_composite"][:16])
    for key in a:
        if key != "train_composite":
            np.testing.assert_array_equal(a[key], b[key])
    for key in ("common_atomic", "anchor_atomic"):
        sa = EpochStream(len(a[key]), 105)
        sb = EpochStream(len(b[key]), 105)
        np.testing.assert_array_equal(a[key][sa.take(1024)], b[key][sb.take(1024)])
    rows = a["train_composite"].copy()
    before = pack_sentences(rows)
    rows[:, 2] += 1
    after = pack_sentences(rows)
    assert all(np.array_equal(x, y) for x, y in zip(before, after, strict=True))


@pytest.mark.parametrize("count", [0, -1, True, 10000])
def test_invalid_sample_counts_fail_without_reroll(spec, count):
    with pytest.raises(ValueError):
        build_world(dict(spec, composition_count=count))


def test_loop_scale_preserves_initial_weights_and_r1_matches_ordinary_gpt(spec):
    a = construct(dict(spec, repeats=1), "cpu")
    b = construct(dict(spec, repeats=8), "cpu")
    assert model_digest(a) == model_digest(b)
    ordinary = SmallGPT(a.config, dropout=0.0)
    ordinary.load_state_dict(a.state_dict())
    tokens = torch.tensor([[2, 21, 3, 13, 4]])
    a.eval()
    b.eval()
    ordinary.eval()
    torch.testing.assert_close(a(tokens), b(tokens, repeats=1), rtol=0, atol=0)
    torch.testing.assert_close(a(tokens), ordinary(tokens), rtol=0, atol=0)


def test_matrix_covers_scaling_axes_without_duplicate_runs():
    path = Path(__file__).parents[1] / "scripts/run_latent_scaling.py"
    module_spec = importlib.util.spec_from_file_location("scaling_runner", path)
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
    specs = runner.specifications()
    assert len(specs) == len({run_name(s) for s in specs}) == 24
    assert {s["width"] for s in specs} == {64, 128, 256}
    assert {s["layers"] for s in specs} == {1, 2, 3}
    assert {s["repeats"] for s in specs} == {1, 2, 3, 4, 8}
    assert {s["composition_count"] for s in specs} == {64, 256, "all"}
    for s in specs:
        assert set(s["repeat_nodes"]) <= set(s["checkpoint_nodes"])
        assert s["nodes"][-1] == s["steps"] == 128000
        assert set(s["checkpoint_nodes"]) <= set(s["nodes"])
