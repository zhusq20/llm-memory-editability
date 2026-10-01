"""Independent checks of the contracts needed to interpret bridge interventions."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import grok_depth_bridge as bridge
from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import SmallGPT
from llm_memory_editability.grok_depth_data import build_world


def candidate_donors(world, row):
    """Enumerate graph-valid choices without calling the experiment's selector."""
    h, r1, r2, t = map(int, row)
    graph = {(int(hh), int(rr)): int(tt) for hh, rr, tt in world["id_atomic"]}
    train = {tuple(map(int, q[:3])) for q in world["train_composite"]}
    bridge = graph[h, r1]
    different, same = set(), set()
    for (donor_h, donor_r), donor_b in graph.items():
        if (donor_h, donor_r) != (h, r1) and donor_b == bridge and donor_h != t:
            same.add((donor_h, donor_r))
        target = graph.get((donor_b, r2))
        if (
            donor_r == r1
            and donor_h != h
            and donor_b != bridge
            and target is not None
            and target != t
            and donor_h not in (t, target)
            and (donor_h, donor_r, r2) not in train
        ):
            different.add((donor_h, donor_r))
    return different, same


@pytest.fixture
def mixed_world():
    # Both donor families have eligible and missing rows in this world.
    return build_world(
        13,
        entities=12,
        relations=4,
        degree=2,
        phi=0.4,
        id_fraction=0.8,
        id_test_fraction=0.3,
    )


def small_model(layers):
    torch.manual_seed(9103)
    return SmallGPT(
        ModelConfig(vocab_size=20, width=16, layers=layers, heads=2, context=8),
        dropout=0.3,
    ).eval()


def test_donors_have_true_id_paths_and_complete_missing_masks(mixed_world):
    donors = bridge.select_donors(mixed_world, 20260930)
    rows = mixed_world["test_composite"]
    np.testing.assert_array_equal(donors["original_rows"], rows)
    graph = {(int(h), int(r)): int(t) for h, r, t in mixed_world["id_atomic"]}
    held_out = {
        tuple(q) for split in ("test_composite", "unused_composite") for q in mixed_world[split]
    }
    for i, row in enumerate(rows):
        h, r1, r2, tail = map(int, row)
        expected = candidate_donors(mixed_world, row)
        assert donors["original_bridge"][i] == graph[h, r1]
        for family, candidates in zip(("different", "same"), expected, strict=True):
            assert donors[family + "_candidate_count"][i] == len(candidates)
            assert bool(donors[family + "_valid"][i]) == bool(candidates)
            donor = donors[family + "_donor"][i]
            if not candidates:
                np.testing.assert_array_equal(donor, [-1, -1, -1])
                assert donors[family + "_reason"][i] != "eligible"
                continue
            assert tuple(donor[:2]) in candidates
            assert graph[tuple(donor[:2])] == donor[2]
            assert tail not in donor[:2]
            assert donors[family + "_reason"][i] == "eligible"
        if donors["different_valid"][i]:
            dh, dr, db = donors["different_donor"][i]
            cf = donors["counterfactual_rows"][i]
            np.testing.assert_array_equal(cf, [dh, r1, r2, graph[db, r2]])
            assert dr == r1 and dh != h and db != graph[h, r1] and cf[-1] != tail
            assert cf[-1] not in (dh, dr)
            assert tuple(cf) in held_out
        else:
            np.testing.assert_array_equal(donors["counterfactual_rows"][i], [-1] * 4)
    assert 0 < donors["different_valid"].sum() < len(rows)
    assert 0 < donors["same_valid"].sum() < len(rows)


def test_donors_are_shared_deterministically_independent_of_global_rng(mixed_world):
    original = copy.deepcopy(mixed_world)
    first = bridge.select_donors(mixed_world, 20260930)
    np.random.seed(872)
    torch.manual_seed(672)
    np.random.uniform(size=100)
    second = bridge.select_donors(copy.deepcopy(mixed_world), 20260930)
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    for key in original:
        if key == "metadata":
            assert original[key] == mixed_world[key]
        else:
            np.testing.assert_array_equal(original[key], mixed_world[key])


@pytest.mark.parametrize("layers", [1, 2])
def test_trace_and_identity_match_standard_forward_and_state_is_causal(layers):
    model = small_model(layers)
    tokens = torch.tensor([[2, 16, 17, 5], [3, 17, 16, 6]])
    positions = torch.tensor([[2, 3], [2, 3]])
    logits, full_state = bridge.traced_forward(model, tokens, return_state=True)
    torch.testing.assert_close(logits, model(tokens), rtol=0, atol=0)
    torch.testing.assert_close(
        bridge.traced_forward(model, tokens, positions), model(tokens, positions), rtol=0, atol=0
    )
    for patch_positions in [(1,), (0, 1)]:
        identity = bridge.traced_forward(
            model, tokens, patch_positions=patch_positions, identity=True
        )
        torch.testing.assert_close(identity, logits, rtol=0, atol=0)
    _, prefix_state = bridge.traced_forward(model, tokens[:, :2], return_state=True)
    torch.testing.assert_close(prefix_state, full_state[:, :2], rtol=1e-6, atol=1e-7)
    altered = tokens.clone()
    altered[:, 2:] = torch.tensor([[18, 10], [19, 11]])
    _, altered_state = bridge.traced_forward(model, altered, return_state=True)
    torch.testing.assert_close(altered_state[:, :2], full_state[:, :2], rtol=0, atol=0)


@pytest.mark.parametrize("patch_positions", [(0,), (1,), (0, 1)])
def test_one_layer_post_block_prefix_patch_cannot_change_answer_or_generated_eos(patch_positions):
    model = small_model(1)
    rows = np.array([[2, 16, 17, 5], [3, 17, 16, 6]], dtype=np.int64)
    prefixes = np.array([[9, 18], [10, 19]], dtype=np.int64)
    baseline, _ = bridge.evaluate_condition(model, rows, "cpu")
    changed, _ = bridge.evaluate_condition(
        model, rows, "cpu", donor_prefixes=prefixes, patch_positions=patch_positions
    )
    for key in ("answer", "stop", "answer_logits"):
        np.testing.assert_array_equal(changed[key], baseline[key])
    tokens = torch.as_tensor(rows)
    positions = torch.tensor([[2, 3], [2, 3]])
    donor = torch.randn((len(rows), 2, 16)) * 100
    original = bridge.traced_forward(model, tokens, positions)
    patched = bridge.traced_forward(
        model, tokens, positions, donor_state=donor, patch_positions=patch_positions
    )
    torch.testing.assert_close(patched, original, rtol=0, atol=0)


def test_two_layer_patch_has_downstream_effect_and_does_not_mutate_donor():
    model = small_model(2)
    tokens = torch.tensor([[2, 16, 17, 5], [3, 17, 16, 6]])
    donor = torch.randn((2, 2, 16))
    original_donor = donor.clone()
    original_tokens = tokens.clone()
    original = bridge.traced_forward(model, tokens)
    changed = bridge.traced_forward(model, tokens, donor_state=donor, patch_positions=(1,))
    assert (changed[:, 2:] - original[:, 2:]).abs().max() > 1e-6
    torch.testing.assert_close(changed[:, 0], original[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(donor, original_donor, rtol=0, atol=0)
    torch.testing.assert_close(tokens, original_tokens, rtol=0, atol=0)


def test_eos_uses_generated_answer_and_reapplies_identical_prefix_patch(monkeypatch):
    model = small_model(2)
    calls = []
    states = []

    def controlled_forward(
        model,
        tokens,
        positions=None,
        donor_state=None,
        patch_positions=(),
        identity=False,
        return_state=False,
    ):
        calls.append((tokens.clone(), donor_state, patch_positions))
        if return_state:
            state = torch.ones((len(tokens), 2, 16))
            states.append(state)
            return torch.zeros((len(tokens), 2, 20)), state
        logits = torch.full((len(tokens), 2, 20), -10.0)
        logits[:, 0, 7] = 10.0
        generated_stop = (tokens[:, 3] == 7).long()
        logits[torch.arange(len(tokens)), 1, generated_stop] = 10.0
        return logits

    monkeypatch.setattr(bridge, "traced_forward", controlled_forward)
    rows = np.array([[2, 16, 17, 5], [3, 17, 16, 6]], dtype=np.int64)
    prefixes = np.array([[8, 16], [9, 17]], dtype=np.int64)
    original = rows.copy()
    result, _ = bridge.evaluate_condition(
        model, rows, "cpu", donor_prefixes=prefixes, patch_positions=(1,)
    )
    np.testing.assert_array_equal(result["answer"], [7, 7])
    np.testing.assert_array_equal(result["stop"], [1, 1])
    np.testing.assert_array_equal(calls[0][0], prefixes)
    assert calls[0][0].shape[1] == 2
    assert calls[1][1] is calls[2][1] is states[0]
    assert calls[1][2] == calls[2][2] == (1,)
    np.testing.assert_array_equal(calls[2][0][:, 3], [7, 7])
    np.testing.assert_array_equal(rows, original)


def test_run_keeps_all_rows_and_marks_unavailable_conditions(mixed_world):
    model = small_model(1)
    donors = bridge.select_donors(mixed_world, 20260930)
    summary, arrays = bridge.evaluate_run(model, mixed_world, donors, "cpu", batch_size=4)
    n = len(mixed_world["test_composite"])
    for condition in bridge.CONDITIONS:
        valid = arrays[condition + "_valid"]
        expected = (
            donors["same_valid"]
            if condition == "same_bridge_r1"
            else donors["different_valid"]
            if condition.startswith("different_") or condition == "counterfactual_input"
            else np.ones(n, dtype=bool)
        )
        np.testing.assert_array_equal(valid, expected)
        for suffix in ("answer", "stop"):
            assert arrays[condition + "_" + suffix].shape == (n,)
            np.testing.assert_array_equal(arrays[condition + "_" + suffix][~valid], -1)
        saved = summary["scores"]["all_test"]["conditions"][condition]
        assert saved["n"] == valid.sum() and saved["total_test_n"] == n
        assert saved["coverage"] == valid.sum() / n
    np.testing.assert_array_equal(arrays["subset_all_test"], True)


def test_no_available_donors_does_not_drop_test_rows():
    world = build_world(
        91, entities=1, relations=1, degree=1, phi=0, id_fraction=1, id_test_fraction=1
    )
    donors = bridge.select_donors(world, 20260930)
    assert len(donors["original_rows"]) == 1
    assert not donors["different_valid"].any() and not donors["same_valid"].any()
    summary, arrays = bridge.evaluate_run(small_model(1), world, donors, "cpu")
    assert len(arrays["baseline_answer"]) == 1
    primary = summary["scores"]["all_test"]["conditions"]["different_bridge_r1"]
    assert primary["n"] == 0 and primary["total_test_n"] == 1
    assert primary["cf_complete_accuracy"] is None


@pytest.mark.parametrize("first_values", [(0.0, 1.0), (0.0, 0.0, 1.0)])
def test_report_averages_initializations_then_worlds_not_queries_or_repeats(first_values):
    path = Path(__file__).resolve().parents[1] / "scripts/report_grok_depth_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_report_for_test", path)
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    common = {
        "phase": "confirmation",
        "arm": "d2-w128",
        "step": 128000,
        "group": "different_donor_available",
        "condition": "different_bridge_r1",
    }
    records = []
    second_values = (0.8, 1.0)
    for world, denominator, values in [(101, 10, first_values), (102, 1000, second_values)]:
        for initialization, value in enumerate(values):
            records.append(
                {
                    **common,
                    "world": world,
                    "initialization": initialization,
                    "run_id": f"w{world}-s{initialization}",
                    "n": denominator,
                    "cf_complete_accuracy": value,
                }
            )
    world_records, groups = report.aggregate(records)
    assert len(world_records) == 2 and len(groups) == 1
    metric = groups[0]["metrics"]["cf_complete_accuracy"]
    expected = (np.mean(first_values) + np.mean(second_values)) / 2
    assert metric["mean"] == pytest.approx(expected)
    assert metric["min"] == pytest.approx(np.mean(first_values))
    assert metric["max"] == pytest.approx(np.mean(second_values))
    assert metric["n_worlds"] == 2
    query_weighted = np.average(
        [r["cf_complete_accuracy"] for r in records], weights=[r["n"] for r in records]
    )
    assert metric["mean"] != pytest.approx(query_weighted)
    if len(first_values) != len(second_values):
        assert metric["mean"] != pytest.approx(
            np.mean([r["cf_complete_accuracy"] for r in records])
        )
