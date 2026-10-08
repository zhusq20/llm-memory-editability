"""Contracts for shared data exposure and isolated trainer adaptation."""

import copy

import numpy as np
import pytest
from test_sequential_transfer import fixture_data, spec

from llm_memory_editability import loop_learning_train as lt
from llm_memory_editability import sequential_transfer as st


def loop_spec():
    return {**spec(), "stage_b_steps": 4, "replay_source": "full_stage_a"}


def test_replay_preserves_new_stream_and_replays_full_actual_old_stream():
    data, options = fixture_data(), loop_spec()
    records, base = st.training_plan(data, options)
    _, actual = lt.training_plan(data, options)
    np.testing.assert_array_equal(actual[:7], base[:7])
    np.testing.assert_array_equal(actual[7:, 3:], base[7:, 3:])
    np.testing.assert_array_equal(actual[7:, :3].flatten(), base[:7].flatten()[:12])
    old = [records[i] for i in actual[7:, :3].flatten()]
    assert sum(r.get("subset") == "A" for r in old) == 6
    assert sum(r.get("sequential_role") == "AA" for r in old) == 6
    assert not {r["id"] for r in records} & {r["id"] for r in data["evaluation_compositions"]}


def test_all_architectures_get_identical_examples_in_identical_order():
    data, options = fixture_data(), loop_spec()
    plans = [
        lt.training_plan(data, {**options, "arm": arm})[1]
        for arm in ("shared", "untied", "mlp_shared", "shallow")
    ]
    assert all(np.array_equal(plans[0], p) for p in plans[1:])


def test_adapter_restores_all_globals_after_failure(tmp_path):
    keys = ("construct", "training_plan", "save_checkpoint", "update")
    originals = {key: getattr(st, key) for key in keys}
    with pytest.raises(RuntimeError), lt.adapter(loop_spec(), tmp_path):
        assert st.training_plan is lt.training_plan
        raise RuntimeError("injected")
    assert all(getattr(st, key) is value for key, value in originals.items())


def test_diagnostics_never_select_heldout_compositions_or_mutate_data():
    data = fixture_data()
    before = copy.deepcopy(data)
    selected = lt.diagnostic_records(data)
    ids = {r["id"] for rows in selected.values() for r in rows}
    assert not ids & {r["id"] for r in data["evaluation_compositions"]}
    assert data == before
