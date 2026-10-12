"""The new objective must preserve the original sample stream and supervision."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grokking_reproduction import MergedBatchStream, encode_rows
from llm_memory_editability.self_decode_alignment import valid_token_mask

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_self_decode_alignment.py"
SPEC = importlib.util.spec_from_file_location("self_decode_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 4, 5, 6, 11])
def test_exposure_preserves_partial_epoch_batches(steps):
    counts = np.zeros(10, dtype=np.int64)
    stream = MergedBatchStream(10, 41)
    actual = 0
    for _ in range(steps):
        indices, n = stream.batch(4)
        actual += n
        counts += np.bincount(indices, minlength=11)[:10]
    summary = runner.exposure_summary(counts, 3, steps, 4)
    assert summary["examples"] == actual
    assert summary["supervised_tokens"] == actual * 2
    assert summary["executed_input_tokens"] == steps * 16
    assert summary["masked_batch_slots"] == steps * 4 - actual
    assert summary["effective_input_tokens"] == int(counts[:3].sum() * 3 + counts[3:].sum() * 4)


def test_exposure_rejects_padding_counted_as_data():
    with pytest.raises(AssertionError, match="exposure"):
        runner.exposure_summary(np.array([2, 2, 1, 1, 1, 1, 1, 1, 1, 1]), 3, 3, 4)


def test_combined_table_preserves_answers_and_has_zero_aux_padding():
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
    assert selected.tolist() == [1, 0]
    expected_atoms = encode_rows(world["atoms"], metadata)
    expected_composites = encode_rows(world["chains"][selected], metadata)
    for tensor, atoms, composites in zip(table, expected_atoms, expected_composites, strict=True):
        np.testing.assert_array_equal(tensor[:-1].numpy(), np.concatenate([atoms, composites]))
    assert table[2][-1].tolist() == [-100, -100]
    mask = valid_token_mask(table[1], table[2], table[0].shape[1])
    torch.testing.assert_close(mask.sum(-1), torch.tensor([3, 3, 4, 4, 0]))


def test_extra_kv_cost_counts_attention_but_no_parameter_projections():
    spec = {"batch_size": 512, "unique_layers": 4, "repeats": 2}
    assert runner.extra_attention_training_flops(spec, 768, {"first_round_kv": False}) == 0
    # Added QK and AV per later layer: forward two products, multiply+add=2;
    # backward convention=two forward equivalents. Four layers, one later round.
    cost = runner.extra_attention_training_flops(spec, 768, {"first_round_kv": True})
    assert cost == 301989888
    longer = runner.extra_attention_training_flops(spec, 768, {"first_round_kv": True}, sequence=8)
    assert longer == 4 * cost
    assert (
        runner.extra_attention_training_flops({**spec, "repeats": 3}, 768, {"first_round_kv": True})
        == 2 * cost
    )
