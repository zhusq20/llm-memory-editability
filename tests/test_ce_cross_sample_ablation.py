import copy
import json
from pathlib import Path

import pytest
import torch

from llm_memory_editability.ce_cross_sample import atomic_write
from llm_memory_editability.ce_cross_sample_ablation import (
    arm_settings,
    gate_summary,
    make_optimizer,
    optimizer_counts,
    optimizer_snapshot,
    tree_digest,
    write_fact,
)
from llm_memory_editability.ce_scope_editing import load_edit, parameter_names, restore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native():
    from transformers import GPT2Config

    from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
    from llm_memory_editability.ttcd_editing import TTCDReader

    torch.set_num_threads(1)
    torch.manual_seed(173)
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
    base = {n: p.detach().clone() for n, p in reader.backbone.state_dict().items()}
    anchor = {n: base[n] for n in parameter_names(reader.backbone, "mlp")}
    meta = {"entity_offset": 0, "relation_offset": 5, "end_marker": 11, "pad_token": 12}
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
    return reader, base, anchor, meta, settings


def test_native_baseline_is_exact_original_loss_and_all_parameters(native):
    reader, base, anchor, meta, settings = native
    old = atomic_write(reader, [0, 0, 1], meta, settings, [[], []], anchor)
    expected = {n: p.detach().clone() for n, p in reader.backbone.state_dict().items()}
    restore(reader, base)
    new, optimizer = write_fact(reader, [0, 0, 1], meta, settings, [[], []], anchor)
    assert new["history"] == old["history"]
    assert optimizer_counts(optimizer) == [2] * len(anchor)
    assert all(torch.equal(p, expected[n]) for n, p in reader.backbone.state_dict().items())


def test_persistent_adam_continues_exactly_after_disk_reload(native, tmp_path):
    reader, base, anchor, meta, settings = native
    _, optimizer = write_fact(
        reader, [0, 0, 1], meta, settings, [[], []], anchor, optimizer_mode="carry_per_stream"
    )
    checkpoint = {
        "values": {n: reader.backbone.get_parameter(n).detach().clone() for n in anchor},
        "optimizer": optimizer_snapshot(optimizer),
    }
    path = tmp_path / "carry.pt"
    torch.save(checkpoint, path)
    uninterrupted, optimizer = write_fact(
        reader,
        [1, 1, 2],
        meta,
        settings,
        [[], []],
        anchor,
        optimizer=optimizer,
        optimizer_mode="carry_per_stream",
    )
    expected = {n: p.detach().clone() for n, p in reader.backbone.state_dict().items()}
    expected_optimizer = tree_digest(optimizer.state_dict())
    restore(reader, base)
    saved = torch.load(path, weights_only=True)
    load_edit(reader, saved["values"], "mlp")
    restored_optimizer = make_optimizer(reader, settings)
    restored_optimizer.load_state_dict(saved["optimizer"])
    replay, restored_optimizer = write_fact(
        reader,
        [1, 1, 2],
        meta,
        settings,
        [[], []],
        anchor,
        optimizer=restored_optimizer,
        optimizer_mode="carry_per_stream",
    )
    assert uninterrupted["history"] == replay["history"]
    assert replay["optimizer_counts_before"] == [2] * len(anchor)
    assert optimizer_counts(restored_optimizer) == [4] * len(anchor)
    assert tree_digest(restored_optimizer.state_dict()) == expected_optimizer
    assert all(torch.equal(p, expected[n]) for n, p in reader.backbone.state_dict().items())


def test_bound_removal_changes_only_region_and_keeps_native_scope(native):
    reader, base, anchor, meta, settings = native
    info, _ = write_fact(
        reader, [0, 0, 1], meta, settings, [[], []], anchor, constraint_enabled=False
    )
    assert info["geometry"]["original_region_outside_fraction"] > 0
    assert info["optimizer_reset_before_fact"]
    assert info["new_composition_supervision"] == 0
    assert info["geometry"]["per_fact_update_rms"] > settings["norm_constraint"]
    assert all(
        torch.equal(p, base[n]) for n, p in reader.backbone.state_dict().items() if n not in anchor
    )
    restore(reader, base)
    bounded, _ = write_fact(reader, [0, 0, 1], meta, settings, [[], []], anchor)
    assert bounded["geometry"]["original_region_outside_fraction"] == 0


def test_fresh_adam_rejects_reusing_optimizer(native):
    reader, _, anchor, meta, settings = native
    _, optimizer = write_fact(reader, [0, 0, 1], meta, settings, [[], []], anchor)
    with pytest.raises(ValueError, match="Fresh Adam"):
        write_fact(reader, [1, 1, 2], meta, settings, [[], []], anchor, optimizer=optimizer)


def test_frozen_plan_alters_exactly_one_factor():
    plan = json.loads((ROOT / "configs/ce-cross-sample-ablation-v1.json").read_text())
    base = json.loads((ROOT / plan["base_plan"]).read_text())
    for arm, item in plan["arms"].items():
        for lr in item["learning_rate_candidates"]:
            settings, mode, bound = arm_settings(plan, base, arm, lr)
            assert settings["steps"] == 20 and settings["kl_factor"] == 10
            assert settings["learning_rate"] == lr
            assert (mode == "carry_per_stream") == (item["variant"] == "carry-adam")
            assert bound == (item["variant"] != "no-bound")
    bad = copy.deepcopy(plan)
    bad["arms"]["loop-no-bound"]["optimizer_mode"] = "carry_per_stream"
    with pytest.raises(ValueError, match="exactly one"):
        arm_settings(bad, base, "loop-no-bound")
    with pytest.raises(ValueError, match="frozen candidate"):
        arm_settings(plan, base, "loop-lower-lr", 1e-9)


def test_write_gate_checks_current_fact_coverage_and_loss():
    gate = {"total_current_readouts_per_candidate": 32, "minimum_correct": 29}
    records = [
        {
            "key": str(i),
            "readout": {"final_correct": [i < 29]},
            "write_info": {"history": [{"new_ce_before": 2.0}, {"new_ce_before": 0.1}]},
        }
        for i in range(32)
    ]
    assert gate_summary(records, gate)["eligible"]
    records[0]["readout"]["final_correct"] = [False]
    assert not gate_summary(records, gate)["eligible"]
    with pytest.raises(ValueError, match="all current-fact"):
        gate_summary(records[:-1], gate)
    for r in records:
        r["readout"]["final_correct"] = [True]
        r["write_info"]["history"][-1]["new_ce_before"] = 2.1
    assert not gate_summary(records, gate)["eligible"]


def test_optimizer_snapshot_does_not_alias_live_moments(native):
    reader, _, anchor, meta, settings = native
    _, optimizer = write_fact(reader, [0, 0, 1], meta, settings, [[], []], anchor)
    snapshot = optimizer_snapshot(optimizer)
    expected = tree_digest(snapshot)
    write_fact(
        reader,
        [1, 1, 2],
        meta,
        settings,
        [[], []],
        anchor,
        optimizer=optimizer,
        optimizer_mode="carry_per_stream",
    )
    assert tree_digest(snapshot) == expected
    assert tree_digest(optimizer.state_dict()) != expected
