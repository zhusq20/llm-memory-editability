"""Scientific contracts for paired iteration-specific atomic writes.

The tiny models use the real ReproductionGPT modules and residual computation.
Graph tests use an all-correct oracle solely to exercise ground-truth selection;
they do not establish that an untrained tiny model knows the graph.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import GPT2Config

from llm_memory_editability.edit_scope_development import (
    ARMS,
    _training_tables,
    case_queries,
    edit_loss,
    evaluate_queries,
    make_edit_model,
    prediction_metrics,
    scope_parameter_names,
    select_case_bank,
)
from llm_memory_editability.grokking_reproduction import (
    ReproductionGPT,
    audit_graph,
    build_graph,
    encode_rows,
)
from llm_memory_editability.path_overlap import World

PROJECT = Path(__file__).resolve().parents[1]
SCOPES = ("joint", "round1", "round2")


@pytest.fixture
def parent():
    torch.set_num_threads(2)
    torch.manual_seed(183)
    config = GPT2Config(
        vocab_size=97,
        n_positions=16,
        n_embd=16,
        n_layer=4,
        n_head=4,
        embd_pdrop=0,
        resid_pdrop=0,
        attn_pdrop=0,
        use_cache=False,
    )
    return ReproductionGPT(config, repeats=2).eval()


@pytest.fixture
def metadata():
    return {"pad_token": 0, "entity_offset": 20, "relation_offset": 60, "end_marker": 90}


@pytest.mark.parametrize("scope", SCOPES)
def test_every_edit_scope_starts_with_identical_native_logits(parent, metadata, scope):
    model = make_edit_model(parent, scope)
    for rows in (np.array([[2, 1, 7], [3, 2, 9]]), np.array([[2, 1, 3, 7], [3, 2, 1, 9]])):
        x, positions, _ = map(torch.as_tensor, encode_rows(rows, metadata))
        with torch.no_grad():
            torch.testing.assert_close(model(x), parent(x), atol=0, rtol=0)
            torch.testing.assert_close(model(x, positions), parent(x, positions), atol=0, rtol=0)
    assert len(parent.transformer.h) == 4 and parent.repeats == 2
    assert len(model.transformer.h) * model.repeats == 8
    assert model.transformer.wte.weight.data_ptr() != parent.transformer.wte.weight.data_ptr()
    if scope != "joint":
        assert model.transformer.h[0].mlp.c_proj.weight.data_ptr() != (
            model.transformer.h[4].mlp.c_proj.weight.data_ptr()
        )


@pytest.mark.parametrize(
    "scope,blocks", [("joint", range(4)), ("round1", range(4)), ("round2", range(4, 8))]
)
def test_only_intended_four_blocks_can_change(parent, metadata, scope, blocks):
    model = make_edit_model(parent, scope)
    names = set(scope_parameter_names(model))
    expected = {
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(f"transformer.h.{index}.") for index in blocks)
    }
    assert names == expected
    assert not any(any(piece in name for piece in ("wte", "wpe", "ln_f")) for name in names)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    original = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
    x, positions, labels = map(
        torch.as_tensor, encode_rows(np.array([[2, 1, 7], [3, 2, 9]]), metadata)
    )
    loss = torch.nn.functional.cross_entropy(model(x, positions).flatten(0, 1), labels.flatten())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss.backward()
    optimizer.step()
    changed = set()
    for name, parameter in model.named_parameters():
        if not torch.equal(before[name], parameter):
            changed.add(name)
        if name not in expected:
            assert parameter.grad is None
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert changed and changed <= expected
    for name, parameter in parent.named_parameters():
        torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)


def test_joint_write_gradient_is_sum_of_both_real_executions(parent, metadata):
    joint = make_edit_model(parent, "joint")
    independent = make_edit_model(parent, "round1")
    for block in independent.transformer.h:
        block.requires_grad_(True)
    target = tuple(map(torch.as_tensor, encode_rows(np.array([[2, 1, 7]]), metadata)))
    replay = tuple(map(torch.as_tensor, encode_rows(np.array([[3, 2, 9], [4, 3, 1]]), metadata)))
    a = edit_loss(joint, target, replay)[0]
    b = edit_loss(independent, target, replay)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    a.backward()
    b.backward()
    for index in range(4):
        first = dict(independent.transformer.h[index].named_parameters())
        second = dict(independent.transformer.h[index + 4].named_parameters())
        assert sum(p.grad.abs().sum().item() for p in first.values()) > 0
        assert sum(p.grad.abs().sum().item() for p in second.values()) > 0
        for name, parameter in joint.transformer.h[index].named_parameters():
            torch.testing.assert_close(
                parameter.grad, first[name].grad + second[name].grad, atol=2e-6, rtol=2e-5
            )


class CorrectOracle:
    def correct(self, kind, indices):
        assert kind in ("atom", "chain")
        return np.ones(len(indices), dtype=bool)


@pytest.fixture
def world(tmp_path):
    graph = build_graph(917, entities=40, relations=12, degree=4)
    audit_graph(graph, 4)
    np.savez(tmp_path / "world.npz", **graph)
    np.savez(tmp_path / "panels.npz")
    (tmp_path / "complete.json").write_text(
        json.dumps(
            {
                "entities": 40,
                "relations": 12,
                "atomic_id": int((~graph["ood"]).sum()),
                "entity_offset": 20,
                "relation_offset": 60,
                "end_marker": 90,
                "pad_token": 0,
            }
        )
    )
    return World(tmp_path)


@pytest.fixture
def selection():
    return {
        "seed": 731,
        "id_cases": 2,
        "ood_cases": 2,
        "affected_chains": 4,
        "min_affected_chains": 1,
        "unrelated_atoms": 3,
        "unrelated_chains": 3,
        "same_answer_atoms": 2,
        "replay_atoms": 3,
        "replay_chains": 3,
    }


@pytest.mark.parametrize("role", ("first", "second"))
def test_case_selection_is_paired_and_preserves_counterfactual_truth(world, selection, role):
    # Selection has no scope input and sees only unedited correctness. All three
    # arms independently execute this same fixed ordering before editing.
    banks = [select_case_bank(world, role, CorrectOracle(), dict(selection)) for _ in SCOPES]
    assert banks[0] == banks[1] == banks[2]
    bank = banks[0]
    assert bank["selected"] == {"id": 2, "ood": 2}
    for case in bank["cases"]:
        target = case["atom"]
        assert bool(world.ood[target]) == (case["group"] == "ood")
        assert case["old_tail"] == int(world.atoms[target, 2])
        assert case["new_tail"] != case["old_tail"]
        assert len(case["D"]) >= selection["min_affected_chains"]
        for index, new_answer in zip(case["D"], case["D_new_answers"], strict=True):
            first, second = world.first[index], world.second[index]
            assert first != second
            if role == "first":
                assert first == target
                new_second = world.lookup[case["new_tail"], world.chains[index, 2]]
                assert new_second >= 0 and new_second != target
                assert int(world.atoms[new_second, 2]) == new_answer
                assert new_second in case["D_prerequisite_atoms"]
                assert second in case["D_prerequisite_atoms"]
            else:
                assert second == target
                assert new_answer == case["new_tail"]
                assert first in case["D_prerequisite_atoms"]
            assert new_answer != int(world.chains[index, 3])
        # Replay and holdouts contain neither the edited edge nor its dependencies.
        dependent = set(case["D_prerequisite_atoms"]) | {target}
        atomic_sets = [
            dependent,
            set(case["U_atoms"]),
            set(case["control_atoms"]),
            set(case["R_atoms"]),
        ]
        for i, left in enumerate(atomic_sets):
            for right in atomic_sets[i + 1 :]:
                assert left.isdisjoint(right)
        chain_sets = [set(case["D"]), set(case["U_chains"]), set(case["R_chains"])]
        for i, left in enumerate(chain_sets):
            for right in chain_sets[i + 1 :]:
                assert left.isdisjoint(right)
        for index in case["U_chains"] + case["R_chains"]:
            assert world.first[index] != target and world.second[index] != target
        unrelated_constituents = set(world.first[case["U_chains"]].tolist()) | set(
            world.second[case["U_chains"]].tolist()
        )
        assert unrelated_constituents.isdisjoint(dependent)
        held_out = (
            dependent | unrelated_constituents | set(case["U_atoms"]) | set(case["control_atoms"])
        )
        assert set(case["R_atoms"]).isdisjoint(held_out)
        replay_constituents = set(world.first[case["R_chains"]].tolist()) | set(
            world.second[case["R_chains"]].tolist()
        )
        assert replay_constituents.isdisjoint(held_out)
        assert case["D_kind"] == world.kind[case["D"]].tolist()
        assert case["D_training_support"] == world.support(7.2)[case["D"]].tolist()
        assert np.all(world.support(7.2)[case["R_chains"]])
        queries = case_queries(world, case)
        assert queries["E"][0].tolist() == world.atoms[[target]].tolist()
        assert queries["E"][1].tolist() == [case["new_tail"]]
        assert queries["D"][1].tolist() == case["D_new_answers"]
        e, replay = _training_tables(world, case, torch.device("cpu"))
        assert e[2].tolist() == [[case["new_tail"] + world.eo, world.metadata["end_marker"]]]
        expected_replay = np.concatenate(
            (world.atoms[case["R_atoms"], -1], world.chains[case["R_chains"], -1])
        )
        assert replay[2][:, 0].tolist() == (expected_replay + world.eo).tolist()
        assert torch.all(replay[2][:, 1] == world.metadata["end_marker"])
        assert len(replay[0]) == len(case["R_atoms"]) + len(case["R_chains"])


def test_native_query_scoring_matches_greedy_full_vocabulary_readout(parent, metadata):
    rows = np.array([[2, 1, 7], [3, 2, 9]])
    new_answers = np.array([11, 12])
    result = evaluate_queries(
        parent, rows, metadata, torch.device("cpu"), new_answers, batch_size=1, precision="float32"
    )
    x, positions, labels = map(torch.as_tensor, encode_rows(rows, metadata))
    with torch.no_grad():
        logits = parent(x, positions)[:, 0]
        answer = logits.argmax(-1)
        x[torch.arange(len(x)), positions[:, 1]] = answer
        stop = parent(x, positions)[:, 1].argmax(-1)
    np.testing.assert_array_equal(result["answer"], answer.numpy())
    np.testing.assert_array_equal(result["stop"], stop.numpy())
    np.testing.assert_array_equal(result["old_answers"], rows[:, -1])
    np.testing.assert_array_equal(result["new_answers"], new_answers)
    for index, (old, new) in enumerate(
        zip(labels[:, 0], new_answers + metadata["entity_offset"], strict=True)
    ):
        assert result["old_rank"][index] == 1 + (logits[index] > logits[index, old]).sum().item()
        assert result["new_rank"][index] == 1 + (logits[index] > logits[index, new]).sum().item()


class KnownLogits(nn.Module):
    def forward(self, tokens, positions):
        logits = torch.zeros((len(tokens), 2, 97))
        logits[:, 0, 0] = 10  # Non-entity token wins the full-vocabulary argmax.
        logits[:, 0, 27] = 6
        logits[:, 0, 31] = 7
        for row in range(len(tokens)):
            token = 90 if tokens[row, positions[row, 1]] == 0 else 17
            logits[row, 1, token] = 10
        return logits


def test_eos_is_conditioned_on_generated_answer_and_wrong_answers_stay_failures(metadata):
    result = evaluate_queries(
        KnownLogits(), np.array([[2, 1, 7]]), metadata, torch.device("cpu"), [11]
    )
    assert result["answer"].tolist() == [0]
    assert result["stop"].tolist() == [90]
    assert result["old_rank"].tolist() == [3]
    assert result["new_rank"].tolist() == [2]
    assert result["old_margin"].tolist() == [-4]
    assert result["new_margin"].tolist() == [-3]
    assert result["new_minus_old"].tolist() == [1]
    metrics = prediction_metrics(result, metadata)
    assert metrics["old_accuracy"] == metrics["new_accuracy"] == 0


def test_development_matrix_is_six_scopes_with_fixed_cost_and_no_confirmation():
    config = json.loads((PROJECT / "configs/edit-scope-development-v1.json").read_text())
    assert {task["name"] for task in config["execution"]["tasks"]} == set(ARMS)
    assert config["execution"]["physical_gpus"] == list(range(2, 8))
    grid = config["development"]
    assert grid["learning_rates"] == [1e-5, 1e-4]
    assert grid["endpoints"] == [10, 50, 100]
    assert grid["save_endpoints"] == [100]
    cases = config["selection"]["id_cases"] + config["selection"]["ood_cases"]
    assert (
        cases * len(grid["learning_rates"]) * max(grid["endpoints"]) == grid["max_updates_per_arm"]
    )
    assert 6 * grid["max_updates_per_arm"] == grid["max_updates_all_arms"]
    assert "never propagation" in grid["selection_rule"]
    assert "no formal confirmation automatically follows" in grid["selection_rule"]
