"""Answer-only labels, detached KL, held-out partition and weight-freeze checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.depth_step_edit_calibration import (
    PARAMETER,
    answer_eos_batch,
    answer_eos_logits,
    calibrated_edit_one,
    calibration_cases,
    load_prepared_calibration,
    prepare_calibration,
    target_and_replay_loss,
)
from llm_memory_editability.grok_loop_model import LoopGPT


@pytest.fixture
def world():
    n, relations = 64, 4
    atoms = np.asarray(
        [[h + 3, r + n + 3, (h + r + 1) % n + 3] for h in range(n) for r in range(relations)],
        dtype=np.int64,
    )
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    rows = np.asarray(
        [
            [h, r1, r2, lookup[(lookup[(h, r1)], r2)]]
            for h in range(3, n + 3)
            for r1 in range(n + 3, n + relations + 3)
            for r2 in range(n + 3, n + relations + 3)
        ],
        dtype=np.int64,
    )
    return {
        "atomic": atoms,
        "familiar_2": rows[::2],
        "strict_2": rows[1::2],
        "metadata": {"entities": n, "relations": relations, "separator_token": 71},
    }


@pytest.fixture
def model():
    torch.manual_seed(332)
    return (
        LoopGPT(
            ModelConfig(vocab_size=72, width=8, layers=2, heads=2, context=9),
            repeats=2,
            dropout=0.0,
        )
        .double()
        .eval()
    )


def test_answer_and_eos_labels_exclude_every_prefix_token(world):
    rows = world["atomic"][:3]
    tokens, positions, labels = answer_eos_batch(rows, world, "cpu")
    np.testing.assert_array_equal(positions, np.tile([3, 4], (3, 1)))
    np.testing.assert_array_equal(labels, np.c_[rows[:, -1], np.ones(3)])
    assert tokens[:, 3].eq(71).all()
    np.testing.assert_array_equal(tokens[:, 4], rows[:, -1])
    assert labels.shape == (3, 2)


def test_replay_kl_is_detached_reference_and_mean_over_answer_eos(model, world):
    target = answer_eos_batch(world["atomic"][:1], world, "cpu")
    replay = answer_eos_batch(world["atomic"][1:5], world, "cpu")
    with torch.no_grad():
        parent_log = answer_eos_logits(model, replay).log_softmax(-1).detach()
    loss, ce, kl = target_and_replay_loss(model, target, replay, parent_log)
    torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=1e-15)
    expected = F.cross_entropy(answer_eos_logits(model, target).flatten(0, 1), target[2].flatten())
    torch.testing.assert_close(ce, expected, rtol=0, atol=0)
    torch.testing.assert_close(loss, ce + kl, rtol=0, atol=0)
    with pytest.raises(ValueError, match="detached"):
        target_and_replay_loss(model, target, replay, parent_log.requires_grad_())


def test_fixed_keep_set_excludes_replay_target_and_successors(world):
    cases, remaining = calibration_cases(world)
    assert len(cases) == 2 and len(remaining) == 6
    for case in cases:
        e = {case["atomic_index"]}
        r, k, u = map(
            set, (case["replay_indices"], case["keep_dev_indices"], case["unused_atomic_indices"])
        )
        successors = set(map(tuple, case["tasks"]["necessary_successor_atomic"]))
        assert len(r) == len(k) == 32
        assert not (e & r or e & k or e & u or r & k or r & u or k & u)
        assert e | r | k | u == set(range(len(world["atomic"])))
        assert not successors.intersection(map(tuple, case["tasks"]["Kdev_atomic"]))


def test_calibrated_edit_updates_only_selected_weight_keeps_parent_and_raw_nodes(model, world):
    cases, _ = calibration_cases(world)
    before = copy.deepcopy(model.state_dict())
    record, raw = calibrated_edit_one(model, cases[0], world, "cpu", "edit", 0.001, nodes=(0, 1))
    assert record["changed_state_tensors"] == [PARAMETER]
    assert record["parent_model_sha256"] != record["final_model_sha256"]
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    assert "step0_Kdev_atomic_predictions" in raw
    assert "step1_U_atomic_predictions" in raw
    assert "parent_replay_log_probabilities" in raw
    assert set(record["history"][-1]["loss"]) == {"total", "target_ce", "replay_kl"}
    assert "calibration_operation_pass" in record["history"][-1]


def test_prepare_and_reload_preserve_config_then_refuse_mutated_data_or_artifacts(
    model, world, tmp_path
):
    source = {
        "checkpoint_sha256": "test",
        "dataset_sha256": "test-data",
        "step": 32000,
        "spec": {"test": True},
    }
    out = tmp_path / "grid"
    config, cases = prepare_calibration(model, world, source, out)
    before = (out / "calibration-config.json").read_bytes()
    loaded, recovered = load_prepared_calibration(model, world, source, out)
    assert loaded == config and len(recovered) == len(cases)
    assert len(config["matrix"]) == 12
    assert (out / "calibration-config.json").read_bytes() == before
    changed_world = copy.deepcopy(world)
    changed_world["atomic"][-1, -1] = 3
    with pytest.raises(ValueError):
        load_prepared_calibration(model, changed_world, source, out)
    (out / "case00-lr0.0001-edit.json").write_text("{}")
    with pytest.raises(FileExistsError, match="branch artifacts"):
        load_prepared_calibration(model, world, source, out)
