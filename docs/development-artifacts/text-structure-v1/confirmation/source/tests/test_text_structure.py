"""Contracts that distinguish structure transfer from exposure or leakage."""

from collections import Counter

import numpy as np
import pytest
import torch

from llm_memory_editability.text_pretrain import composite_sentence, construct
from llm_memory_editability.text_structure import (
    TESTS,
    build_world,
    data_audit,
    evaluate_all,
    training_table,
)


def spec(world=148001):
    return dict(
        world=world,
        initialization=14801,
        heads_n=128,
        bridges_n=32,
        tails_n=32,
        width=16,
        layers=2,
        heads=2,
        repeats=2,
        dropout=0.0,
    )


@pytest.mark.parametrize("world", [148001, 148002])
def test_truth_common_holdout_and_isolated_target(world):
    data = build_world(spec(world))
    lookup = {(h, r): t for h, r, t in data["atomic"]}
    trained = np.concatenate(
        [data["restricted_train"], data["broad_train"], data["positive_train"]]
    )
    train_pairs = {(h, t) for h, _, _, _, t in trained}
    for split in TESTS:
        assert len(data[split])
        for h, r1, b, r2, t in data[split]:
            assert lookup[h, r1] == b and lookup[b, r2] == t
            assert (h, t) not in train_pairs
    strict = data["strict_test"]
    background = np.concatenate([data["restricted_train"], data["broad_train"]])
    assert not set(strict[:, 0]) & set(background[:, [0, 2]].flatten())
    assert not set(strict[:, 2]) & set(background[:, [0, 2]].flatten())
    assert len(data["strict_all"]) == 512


def test_exact_per_fact_role_usage_unique_chains_and_token_marginals():
    data = build_world(spec())
    a, b = data["restricted_train"], data["broad_train"]
    for cols, expected in (([0, 1, 2], 1), ([2, 3, 4], 8)):
        counts_a = Counter(map(tuple, a[:, cols]))
        counts_b = Counter(map(tuple, b[:, cols]))
        assert counts_a == counts_b and set(counts_a.values()) == {expected}
    assert len(a) == len(b) == len(set(map(tuple, a))) == len(set(map(tuple, b))) == 256
    assert Counter(t for row in a for t in composite_sentence(row)) == Counter(
        t for row in b for t in composite_sentence(row)
    )
    assert len(set(map(tuple, a[:, [1, 3]]))) == 8
    assert len(set(map(tuple, b[:, [1, 3]]))) == 16
    assert data_audit(data)["status"] == "passed"


def test_role_strata_match_actual_composition_exposure_in_both_arms():
    data = build_world(spec())
    roles = {
        "id_test": (True, True),
        "first_only": (True, False),
        "second_only": (False, True),
        "background_neither": (False, False),
        "strict_test": (False, False),
    }
    for arm in ("restricted", "broad"):
        rows = data[arm + "_train"]
        first = set(map(tuple, rows[:, :3]))
        second = set(map(tuple, rows[:, 2:]))
        for split, expected in roles.items():
            for row in data[split]:
                assert (tuple(row[:3]) in first, tuple(row[2:]) in second) == expected


def test_training_labels_full_token_budget_and_matching_atom_stream():
    data = build_world(spec())
    a, ba, _ = training_table(data, "restricted")
    b, bb, _ = training_table(data, "broad")
    c, bc, _ = training_table(data, "positive")
    assert ba == bb == [0, 640, 896]
    assert bc == [0, 640, 960]
    for i in range(4):
        np.testing.assert_array_equal(a[i][:640], b[i][:640])
        np.testing.assert_array_equal(a[i][:640], c[i][:640])
    for table, bounds in ((a, ba), (b, bb), (c, bc)):
        assert np.all((table[3][:640] != -100).sum(1) == 7)
        assert np.all((table[3][640:] != -100).sum(1) == 9)
        assert not np.triu(table[2], 1).any()
        assert table[0].shape == (bounds[-1], 25)
        assert not table[2][0, 0, :7, 8:].any()


def test_evaluation_is_read_only_and_two_calls_keep_every_query():
    cfg = spec()
    data = build_world(cfg)
    data = {k: v[:3] if k in ("atomic", "positive_train", *TESTS) else v for k, v in data.items()}
    # Include required atomics so eligibility has a defined denominator.
    data["atomic"] = build_world(cfg)["atomic"]
    model = construct(cfg, "cpu").eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    metrics, predictions = evaluate_all(model, data, data["broad_train"][:3], "cpu")
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
    for split in TESTS:
        assert metrics[split]["n"] == metrics[split + "_two_calls"]["n"] == 3
        assert predictions[split + "_two_calls_first_target"].tolist() == data[split][:, 2].tolist()
        assert len(predictions[split + "_eligible"]) == 3
        assert 0 <= metrics[split + "_conditional"]["coverage"] <= 1
