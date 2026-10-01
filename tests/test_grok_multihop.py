"""Scientific contracts for longer paths: split truth, causality and own answers."""

import copy
import itertools

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import SmallGPT
from llm_memory_editability.grok_depth_data import build_world as build_twohop
from llm_memory_editability.grok_multihop import autonomous_calls, evaluate_rows, pack_rows
from llm_memory_editability.grok_multihop_data import audit_world, build_world


@pytest.mark.parametrize("hops", [2, 3, 4])
def test_all_paths_true_partitioned_and_atomic_graph_reused(hops):
    world = build_world(31, hops, 8, 4, 2, 2.0, 0.8, 0.2, 9)
    old = build_twohop(31, 8, 4, 2, 2.0, 0.8, 0.2)
    for key in ("atomic", "id_atomic", "ood_atomic"):
        np.testing.assert_array_equal(world[key], old[key])
    lookup = {(h, r): t for h, r, t in world["atomic"]}
    id_set = {tuple(row) for row in world["id_atomic"]}
    expected, expected_ood = set(), set()
    for head in range(2, 10):
        for relation_path in itertools.product(range(10, 14), repeat=hops):
            current = head
            ids = []
            for relation in relation_path:
                tail = lookup.get((current, relation))
                if tail is None:
                    break
                ids.append((current, relation, tail) in id_set)
                current = tail
            else:
                row = (head, *relation_path, current)
                if all(ids):
                    expected.add(row)
                if not any(ids):
                    expected_ood.add(row)
    union = set()
    for name in ("train_composite", "test_full_composite", "unused_composite"):
        rows = {tuple(row) for row in world[name]}
        assert not rows & union
        union |= rows
    assert union == expected
    assert {tuple(row) for row in world["ood_composite"]} == expected_ood
    assert len(world["test_composite"]) <= 9
    assert audit_world(world)["complete_query_overlap"] == 0
    again = build_world(31, hops, 8, 4, 2, 2.0, 0.8, 0.2, 9)
    assert audit_world(world)["dataset_sha256"] == audit_world(again)["dataset_sha256"]


def test_audit_rejects_full_query_overlap_and_wrong_answers():
    world = build_world(32, 3, 8, 4, 2, 1.0, 1.0, 0.4, 9)
    broken = copy.deepcopy(world)
    broken["train_composite"][0] = broken["test_full_composite"][0]
    with pytest.raises(ValueError, match="overlap"):
        audit_world(broken)
    broken = copy.deepcopy(world)
    broken["train_composite"][0, -1] = (broken["train_composite"][0, -1] - 1) % 8 + 2
    with pytest.raises(ValueError, match="target"):
        audit_world(broken)


def test_development_phi_changes_do_not_change_reserved_pool_or_fixed_probe():
    low = build_world(32, 3, 8, 4, 2, 1.0, 1.0, 0.4, 9)
    high = build_world(32, 3, 8, 4, 2, 3.0, 1.0, 0.4, 9)
    for name in ("test_composite", "test_full_composite"):
        np.testing.assert_array_equal(low[name], high[name])


@pytest.mark.parametrize("hops", [3, 4])
def test_longer_rows_keep_only_answer_eos_and_no_answer_leakage(hops):
    torch.manual_seed(144)
    model = SmallGPT(ModelConfig(vocab_size=20, width=16, layers=2, heads=2, context=8))
    model.eval()
    rows = [[2, *range(10, 10 + hops), 5]]
    tokens, positions, labels = pack_rows(rows, hops + 2)
    np.testing.assert_array_equal(labels, [[5, 1]])
    np.testing.assert_array_equal(positions, [[hops, hops + 1]])
    x, pos = torch.as_tensor(tokens), torch.as_tensor(positions)
    original = model(x, pos).detach()
    x[:, -1] = 7
    torch.testing.assert_close(model(x, pos)[:, 0], original[:, 0], rtol=0, atol=0)


class OwnAnswerModel(torch.nn.Module):
    def forward(self, x, positions=None):
        logits = torch.full((len(x), 2, 20), -10.0, device=x.device)
        # Chaining must use returned 7 as next head, rather than true bridge 3.
        answer = torch.where(x[:, 0] == 7, 6, 7)
        logits[torch.arange(len(x)), 0, answer] = 10.0
        stop = (x[torch.arange(len(x)), positions[:, 1]] == answer).long()
        logits[torch.arange(len(x)), 1, stop] = 10.0
        return logits


def test_autonomous_calls_carry_generated_entities_and_check_every_eos():
    world = {
        "atomic": np.array([[2, 10, 3], [3, 11, 4], [4, 12, 6]], dtype=np.int64),
        "metadata": {"entities": 8, "relations": 4, "hops": 3},
    }
    rows = np.array([[2, 10, 11, 12, 6]], dtype=np.int64)
    model = OwnAnswerModel()
    result, predictions = autonomous_calls(model, rows, world, "cpu", 5)
    assert predictions["hop1_answer"][0] == 7
    assert predictions["hop2_answer"][0] == 6
    assert predictions["hop3_answer"][0] == 7
    assert result["accuracy"] == 0
    assert result["all_intermediate_answers_and_eos_correct"] == 0
    for j in range(1, 4):
        assert predictions[f"hop{j}_stop"][0] == 1


def test_evaluation_does_not_change_input_rows_or_model_mode():
    model = OwnAnswerModel().eval()
    rows = np.array([[2, 10, 11, 12, 5]], dtype=np.int64)
    original = rows.copy()
    result, prediction = evaluate_rows(model, rows, "cpu", 5)
    np.testing.assert_array_equal(rows, original)
    assert not model.training
    assert prediction["stop"][0] == 1
    assert result["accuracy"] == 0
