"""Contracts for reused baseline, unchanged data and new objective accounting."""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import (
    MergedBatchStream,
    encode_rows,
    model_digest,
)
from llm_memory_editability.self_decode_alignment import SelfDecodeGPT
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT


def load_runner(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"run_{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_runner_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner("supervised_alignment")
original_runner = load_runner("self_decode_alignment")


def tiny_table():
    world = {
        "atoms": np.array([[0, 1, 2], [2, 0, 1]]),
        "chains": np.array([[0, 1, 0, 1], [2, 0, 1, 2]]),
        "train_order": np.array([1, 0]),
    }
    metadata = {
        "pad_token": 0,
        "entity_offset": 10,
        "relation_offset": 20,
        "end_marker": 30,
        "atomic_id": 1,
    }
    table, selected = runner.make_table(world, metadata, {"phi": 2.0}, "cpu")
    return world, metadata, table, selected


def tiny_models():
    config = GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        attn_pdrop=0.15,
        resid_pdrop=0.15,
        embd_pdrop=0.15,
        use_cache=False,
    )
    torch.manual_seed(61)
    reference = SelfDecodeGPT(config, repeats=2).train()
    candidate = SupervisedAlignmentGPT(config, repeats=2).train()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return reference, candidate


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4, 5, 6, 11])
def test_exposure_preserves_original_masked_epoch_tails(steps):
    counts = np.zeros(10, dtype=np.int64)
    stream = MergedBatchStream(10, 41)
    original_stream = original_runner.MergedBatchStream(10, 41)
    actual = 0
    for _ in range(steps):
        indices, n = stream.batch(4)
        original_indices, original_n = original_stream.batch(4)
        np.testing.assert_array_equal(indices, original_indices)
        assert n == original_n
        actual += n
        counts += np.bincount(indices, minlength=11)[:10]
    summary = runner.exposure_summary(counts, 3, steps, 4)
    assert summary == original_runner.exposure_summary(counts, 3, steps, 4)
    assert summary["examples"] == actual
    assert summary["supervised_tokens"] == actual * 2
    assert summary["executed_input_tokens"] == steps * 16
    assert summary["masked_batch_slots"] == steps * 4 - actual


def test_table_is_exact_original_data_and_answer_eos_supervision():
    world, metadata, table, selected = tiny_table()
    original_table, original_selected = original_runner.make_table(
        world, metadata, {"phi": 2.0}, "cpu"
    )
    np.testing.assert_array_equal(selected, original_selected)
    assert selected.tolist() == [1, 0]
    atoms = encode_rows(world["atoms"], metadata)
    composites = encode_rows(world["chains"][selected], metadata)
    for new, old, a, c in zip(table, original_table, atoms, composites, strict=True):
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        np.testing.assert_array_equal(new[:-1].numpy(), np.concatenate([a, c]))
    assert table[1][:2].tolist() == [[1, 2], [1, 2]]
    assert table[1][2:].tolist() == [[2, 3], [2, 3], [2, 3]]
    assert table[2][-1].tolist() == [-100, -100]
    assert (table[2] != -100).sum().item() == 8


def test_disabled_runner_matches_original_logits_loss_and_all_gradients():
    reference, candidate = tiny_models()
    _, _, table, _ = tiny_table()
    indices = torch.arange(len(table[0]))
    rng = torch.get_rng_state()
    digest_before = model_digest(candidate)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'.*")
        result = runner.baseline_eager_contract(
            reference, candidate, table, indices, 73, original_runner.losses
        )
    assert result["passed"], result
    assert result["parameter_gradient_tensors"] == len(list(candidate.parameters()))
    assert result["max_absolute_gradient_difference"] == 0.0
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert model_digest(candidate) == digest_before
    assert all(parameter.grad is None for parameter in candidate.parameters())


def test_baseline_contract_rejects_different_initial_weights():
    reference, candidate = tiny_models()
    _, _, table, _ = tiny_table()
    with torch.no_grad():
        next(candidate.parameters()).add_(0.01)
    with pytest.raises(AssertionError, match="exactly equal weights"):
        runner.baseline_eager_contract(
            reference, candidate, table, torch.arange(len(table[0])), 73, original_runner.losses
        )


@pytest.mark.parametrize("kind", ["alignment", "label_ce"])
def test_objective_combines_existing_main_loss_and_original_supervision_slots(kind):
    _, model = tiny_models()
    _, _, table, _ = tiny_table()
    arm = {"kind": kind, "coefficient": 0.3, "scope": "supervised"}
    total, main, auxiliary, info = runner.losses(model, table, torch.arange(len(table[0])), arm)
    torch.testing.assert_close(total, main + 0.3 * auxiliary, rtol=0, atol=0)
    assert info["valid_positions"].item() == 8
    assert info["selected_positions"].item() == 8
    assert 0 <= info["confidence_sum"].item() <= 8
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_cost_counts_supervised_projection_dense_masked_kv_and_discoloop():
    spec = {"batch_size": 512, "unique_layers": 4, "repeats": 2}
    width, vocab_size = 768, 2203
    base = spec["batch_size"] * 2 * width * vocab_size
    for kind, factor in (("none", 0), ("alignment", 2), ("label_ce", 6)):
        assert runner.extra_projection_training_flops(spec, width, vocab_size, {"kind": kind}) == (
            factor * base
        )
    kv_arm = {"first_round_kv": True}
    assert runner.extra_attention_training_flops(spec, width, kv_arm) == 301989888
    assert runner.extra_attention_training_flops(spec, width, kv_arm, sequence=8) == 4 * 301989888
    assert runner.extra_attention_training_flops(spec, width, {}) == 0
    disco = runner.extra_discoloop_training_flops(spec, width, vocab_size, {"discoloop": True})
    assert disco == 12 * spec["batch_size"] * 4 * width * vocab_size
    assert runner.extra_discoloop_training_flops(spec, width, vocab_size, {}) == 0
    assert (
        runner.extra_discoloop_training_flops(
            {**spec, "repeats": 3}, width, vocab_size, {"discoloop": True}
        )
        == 2 * disco
    )


@pytest.mark.parametrize(
    "amendment",
    [
        {"kind": "self_ce"},
        {"scope": "all"},
        {"coefficient": -1},
        {"coefficient": float("nan")},
        {"kind": "none", "coefficient": 0.3},
        {"discoloop": True},
        {"first_round_kv": 1},
        {"discoloop_temperature": 0},
    ],
)
def test_arm_validation_rejects_old_objective_and_unapproved_combinations(amendment):
    arm = {"kind": "alignment", "coefficient": 0.3, "scope": "supervised"}
    with pytest.raises(ValueError):
        runner.validate_arm({**arm, **amendment})


def test_architecture_flags_freeze_every_forward_setting():
    flags = runner.architecture_flags({"first_round_kv": True})
    assert flags == {
        "first_round_kv": True,
        "discoloop": False,
        "discoloop_alpha": 1.0,
        "discoloop_temperature": 1.0,
        "discoloop_eps": 1e-6,
    }
    assert (
        runner.architecture_flags({"discoloop": True, "discoloop_temperature": 2.0})[
            "discoloop_temperature"
        ]
        == 2.0
    )
