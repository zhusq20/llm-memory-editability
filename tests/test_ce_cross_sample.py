import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from llm_memory_editability.ce_cross_sample import (
    atomic_write,
    compare_readouts,
    query_groups,
    settings_for,
    validate_panel,
)


def panel_world():
    p = {
        "pairs": [
            {
                "split": "evaluation",
                "pair_id": "p",
                "A": {"fact_id": "A", "fact": [0, 0, 1]},
                "B": {"fact_id": "B", "fact": [1, 1, 2]},
                "query": [0, 0, 1],
                "answer": 2,
            }
        ],
        "streams": {"forward": ["A", "B"], "reverse": ["B", "A"]},
        "mixed_queries": [],
        "old_eval_pools": {},
        "metadata": {},
    }
    w = SimpleNamespace(
        entities=3,
        metadata={"relations": 2},
        lookup=np.full((3, 2), -1),
        atoms=np.empty((0, 3), int),
    )
    return p, w


def test_absent_keys_and_truth_are_required():
    p, w = panel_world()
    assert validate_panel(p, w)["new_keys"] == 2
    w.lookup[1, 1] = 0
    with pytest.raises(ValueError, match="absent"):
        validate_panel(p, w)


def test_wrong_chain_and_duplicate_stream_rejected():
    p, w = panel_world()
    bad = copy.deepcopy(p)
    bad["pairs"][0]["answer"] = 0
    with pytest.raises(ValueError, match="truth"):
        validate_panel(bad, w)
    p["streams"]["forward"] = ["A", "A"]
    with pytest.raises(ValueError, match="exactly once"):
        validate_panel(p, w)


def test_queries_keep_atomic_and_composite_labels_separate():
    p, _ = panel_world()
    groups = query_groups(p)
    assert groups["atoms"] == [[0, 0, 1], [1, 1, 2]]
    assert groups["double_new"] == [[0, 0, 1, 2]]


def test_replay_requires_exact_tokens_and_bounded_scores():
    compare_readouts({"tokens": [1], "score": 0.1}, {"tokens": [1], "score": 0.100001})
    with pytest.raises(AssertionError, match="Discrete"):
        compare_readouts({"tokens": [2]}, {"tokens": [1]})
    with pytest.raises(AssertionError, match="Floating"):
        compare_readouts({"score": 0.2}, {"score": 0.1})


def test_settings_use_explicit_global_constraint():
    p = {
        "adaptation": {
            "steps_per_fact": 20,
            "kl_factor": 10,
            "gradient_clip": 1,
            "parameter_constraint": {"maximum_absolute_element_change": 0.0005},
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
        },
        "architectures": {"loop": {"learning_rate": 0.00032}},
    }
    assert settings_for(p, "loop")["steps"] == 20
    assert settings_for(p, "loop", 2)["norm_constraint"] == 0.0005


def test_full_cpu_write_global_anchor_and_frozen_attention():
    from transformers import GPT2Config

    from llm_memory_editability.ce_scope_editing import parameter_names
    from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
    from llm_memory_editability.ttcd_editing import TTCDReader

    config = GPT2Config(
        vocab_size=13,
        n_positions=8,
        n_embd=8,
        n_head=2,
        n_layer=1,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    reader = TTCDReader(SupervisedAlignmentGPT(config, 2), native_causal_attention=True).eval()
    meta = {"entity_offset": 0, "relation_offset": 5, "end_marker": 11, "pad_token": 12}
    names = parameter_names(reader.backbone, "mlp")
    base = {n: p.detach().clone() for n, p in reader.backbone.named_parameters()}
    anchor = {n: base[n] for n in names}
    settings = {
        "steps": 2,
        "learning_rate": 0.1,
        "kl_factor": 10,
        "norm_constraint": 0.0005,
        "gradient_clip": 1,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
    }
    info = atomic_write(reader, [0, 0, 1], meta, settings, [[], []], anchor)
    assert info["steps"] == 2 and info["global_theta0_anchor"]
    # A second fact must remain in the original region, never accumulate two regions.
    atomic_write(reader, [1, 1, 2], meta, settings, [[], []], anchor)
    for n, p in reader.backbone.named_parameters():
        if n in names:
            assert torch.all(p <= base[n] + 0.0005)
            assert torch.all(p >= base[n] - 0.0005)
        else:
            assert torch.equal(p, base[n])
