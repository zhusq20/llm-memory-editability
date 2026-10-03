"""Scientific contracts for paired schedules and nested load comparisons."""

import numpy as np
import pytest

from llm_memory_editability.grok_depth import EpochStream
from llm_memory_editability.storage_frontier import (
    build_world,
    learning_rate,
    run_name,
    strata_for,
)


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
        "architecture": "loop1x2",
        "width": 32,
        "initialization": 104,
        "load": "high",
        "schedule": "cosine",
        "min_lr_ratio": 0.1,
        "lr": 0.001,
        "warmup": 200,
        "steps": 16000,
    }


def test_schedules_share_warmup_and_cosine_reaches_prespecified_floor(spec):
    constant = dict(spec, schedule="constant")
    for step in (1, 100, 200):
        assert learning_rate(spec, step) == learning_rate(constant, step)
    assert learning_rate(spec, 16000) == pytest.approx(0.0001)
    assert learning_rate(constant, 16000) == 0.001
    values = [learning_rate(spec, step) for step in range(200, 16001, 100)]
    assert all(a >= b for a, b in zip(values[:-1], values[1:], strict=True))


def test_nested_loads_preserve_truth_test_queries_and_common_training_streams(spec):
    low, high = dict(spec, extra_count=32), dict(spec, extra_count=128)
    a, b = build_world(low), build_world(high)
    np.testing.assert_array_equal(a["extra_atomic"], b["extra_atomic"][:32])
    for key in ("common_atomic", "train_composite", "familiar_test", "strict_test"):
        np.testing.assert_array_equal(a[key], b[key])
    for i in (0, 1):
        sa, sb = strata_for(low, a)[i], strata_for(high, b)[i]
        ia, ib = EpochStream(len(sa), 123 + i), EpochStream(len(sb), 123 + i)
        np.testing.assert_array_equal(sa[ia.take(1024)], sb[ib.take(1024)])
    common_keys = {(int(h), int(r)) for h, r, _ in a["common_atomic"]}
    assert not common_keys & {(int(h), int(r)) for h, r, _ in b["extra_atomic"]}


def test_run_names_separate_every_manipulated_condition(spec):
    names = {
        run_name(dict(spec, schedule=schedule, load=load, initialization=initialization))
        for schedule in ("constant", "cosine")
        for load in ("low", "high")
        for initialization in (104, 105, 106)
    }
    assert len(names) == 12
    assert run_name(dict(spec, extra_count=32)) != run_name(dict(spec, extra_count=64))
