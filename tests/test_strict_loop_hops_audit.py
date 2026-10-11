"""Contracts that affect the strict depth/answer interpretation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_loop_model import LoopGPT

spec = importlib.util.spec_from_file_location(
    "strict_hops_audit", Path(__file__).resolve().parents[1] / "scripts/audit_strict_loop_hops.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("layers,repeats", [(1, 1), (1, 2), (2, 1)])
def test_answer_independent_of_hidden_truth_and_counted_depth(layers, repeats):
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=23, width=16, layers=layers, heads=2, context=9)
    model = LoopGPT(config, repeats=repeats, dropout=0.0).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    for hops in (3, 4):
        rows = np.array([[3, *range(10, 10 + hops), 4], [5, *range(10, 10 + hops), 6]])
        changed_truth = rows.copy()
        changed_truth[:, -1] += 2
        first, traces = audit.answers(model, rows, 21, batch_size=1)
        second, _ = audit.answers(model, changed_truth, 21, batch_size=1)
        np.testing.assert_array_equal(first, second)
        assert len(traces) == len(rows)
        for trace in traces:
            assert trace["prompt_length"] == hops + 3
            assert trace["calls"].count("model") == 1
            assert trace["calls"].count("token") == 1
            assert trace["calls"].count("ln_final") == 1
            for i in range(layers):
                for label in ("qkv", "proj", "mlp", "up", "down"):
                    assert trace["calls"].count(f"{i}.{label}") == repeats
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_guess_tables_do_not_use_test_answers():
    train = np.array([[3, 10, 11, 12, 4], [5, 10, 11, 13, 6]])
    test = np.array([[3, 10, 11, 12, 9], [5, 10, 11, 13, 8]])
    changed_truth = test.copy()
    changed_truth[:, -1] = 3
    first, second = audit.guesses(train, test), audit.guesses(train, changed_truth)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_aggregate_weights_worlds_equally_not_queries():
    records = []
    for world, initialization, accuracy in [(1, 1, 0.2), (1, 2, 0.4), (2, 1, 0.9)]:
        records.append(
            {
                "group": "test",
                "width": 16,
                "phi": 4,
                "unique_blocks": 1,
                "repeats": 2,
                "world": world,
                "initialization": initialization,
                "scores": {"familiar_3": {"answer_accuracy": accuracy}},
            }
        )
    result = audit.aggregate(records)
    assert result[0]["scores"]["familiar_3"]["answer_accuracy"] == pytest.approx(0.6)
