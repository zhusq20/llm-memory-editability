"""Objective, replay, and recovery contracts for the actual architecture runner."""

import copy
import json
import random

import numpy as np
import pytest
import torch

from llm_memory_editability import parametric_architecture as arch
from llm_memory_editability import parametric_architecture_train as train
from llm_memory_editability import realworld_composition as historical


def config(dropout=0.0):
    return {
        "vocab_size": 31,
        "positions": 32,
        "hidden_size": 8,
        "attention_heads": 2,
        "dropout": dropout,
    }


def spec(architecture="D8", microbatch_size=2):
    return {
        "architecture": architecture,
        "initialization": 42,
        "microbatch_size": microbatch_size,
        "balance_coefficient": 0.01,
    }


def records():
    return [
        {
            "id": str(i),
            "encoded": {
                "prefix": [1, 30, 3],
                "target": [4] * (i % 3 + 1) + [30],
                "input": [1, 30, 3] + [4] * (i % 3 + 1),
            },
        }
        for i in range(5)
    ]


@pytest.fixture(autouse=True)
def small_thread_pool():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_validity_uses_lengths_and_includes_real_eos():
    tokens, _, labels, valid = train.pack(records(), 30, "cpu")
    assert valid[:, 1].all()  # A real EOS-valued token remains a routing exposure.
    assert int(valid.sum()) == sum(len(r["encoded"]["input"]) for r in records())
    assert not valid[:, -1].any()
    assert (labels[:, -1] == -100).any()
    assert tokens.shape == valid.shape


@pytest.mark.parametrize("architecture", ["D8", "M4", "HC8"])
def test_microbatch_objective_and_gradients_match_full_batch(architecture):
    initial = arch.construct(config(), spec(architecture), "cpu")
    micro, full = copy.deepcopy(initial), copy.deepcopy(initial)
    options = spec(architecture)
    micro_result = train.update(
        micro, torch.optim.SGD(micro.parameters(), lr=0.02), records(), options, "cpu", 30
    )
    full_result = train.update(
        full,
        torch.optim.SGD(full.parameters(), lr=0.02),
        records(),
        {**options, "microbatch_size": len(records())},
        "cpu",
        30,
    )
    assert micro_result["answer_ce"] == pytest.approx(full_result["answer_ce"], abs=1e-6)
    assert micro_result["balance_loss"] == pytest.approx(full_result["balance_loss"], abs=1e-6)
    for (name, left), (_, right) in zip(
        micro.named_parameters(), full.named_parameters(), strict=True
    ):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6, msg=name)
    assert micro_result["effective_input_tokens"] == full_result["effective_input_tokens"]
    if architecture == "M4":
        assert micro_result["routing_prepass_effective_input_tokens"] > 0
        assert micro_result["estimated_matmul_training_flops"] == (
            4 * micro_result["compute"]["forward_leading_matmul_flops"]
        )


def test_moe_counting_replay_preserves_dropout_rng_and_routes():
    model = arch.construct(config(dropout=0.1), spec("M4"), "cpu")
    reference = copy.deepcopy(model)
    torch.manual_seed(1234)
    before = train.rng_state("cpu")
    result = train.update(
        model, torch.optim.SGD(model.parameters(), lr=0), records(), spec("M4"), "cpu", 30
    )
    after = torch.get_rng_state().clone()
    train.restore_rng(before, "cpu")
    reference.train()
    stats = []
    for i in range(0, len(records()), 2):
        tokens, positions, _, valid = train.pack(records()[i : i + 2], 30, "cpu")
        _, aux = reference(tokens, positions, valid_mask=valid, return_aux=True)
        stats.append(aux["router_stats"])
    assert torch.equal(after, torch.get_rng_state())
    fractions, total = arch.combine_router_fractions(stats)
    expected_balance = sum(arch.router_balance_loss(s, fractions, total) for s in stats)
    assert result["balance_loss"] == pytest.approx(float(expected_balance), abs=1e-6)
    assert result["router"]["assignment_fractions"] == [f.tolist() for f in fractions]


def test_single_microbatch_matches_previous_two_pass_with_dropout():
    options = spec("M4", microbatch_size=len(records()))
    single = arch.construct(config(dropout=0.1), options, "cpu")
    replayed = copy.deepcopy(single)
    torch.manual_seed(547)
    before = train.rng_state("cpu")
    result = train.update(
        single, torch.optim.SGD(single.parameters(), lr=0.02), records(), options, "cpu", 30
    )
    after = torch.get_rng_state().clone()
    train.restore_rng(before, "cpu")
    replayed.train()
    optimizer = torch.optim.SGD(replayed.parameters(), lr=0.02)
    tokens, positions, labels, valid = train.pack(records(), 30, "cpu")
    with torch.no_grad():
        _, pre_aux = replayed(tokens, positions, valid_mask=valid, return_aux=True)
    fractions, total = arch.combine_router_fractions([pre_aux["router_stats"]])
    train.restore_rng(before, "cpu")
    logits, aux = replayed(tokens, positions, valid_mask=valid, return_aux=True)
    ce = historical.example_losses(logits, labels).sum() / len(records())
    balance = arch.router_balance_loss(aux["router_stats"], fractions, total)
    (ce + options["balance_coefficient"] * balance).backward()
    torch.nn.utils.clip_grad_norm_(replayed.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    assert result["answer_ce"] == float(ce.detach())
    assert result["balance_loss"] == float(balance.detach())
    assert torch.equal(after, torch.get_rng_state())
    for left, right in zip(single.parameters(), replayed.parameters(), strict=True):
        assert torch.equal(left, right)
    assert result["routing_prepass_effective_input_tokens"] == 0
    assert result["routing_prepass_executed_input_tokens"] == 0
    assert result["compute"]["routing_prepass_matmul_flops"] == 0
    assert result["estimated_matmul_training_flops"] == (
        3 * result["compute"]["forward_leading_matmul_flops"]
    )
    assert result["router"]["balance_scope"] == "full_effective_batch_single_forward"
    assert result["router"]["replay_checked"] is False


def test_full_checkpoint_resume_preserves_optimizer_stream_and_rng(tmp_path):
    options = spec("M4")
    model = arch.construct(config(dropout=0.1), options, "cpu")
    optimizer = arch.optimizer_for(model, 1e-3, 0.1)
    stream = historical.BatchStream(5, 9)
    ids = stream.batch(3)
    train.update(model, optimizer, [records()[i] for i in ids], options, "cpu", 30)
    counts = np.bincount(ids, minlength=5)
    path = tmp_path / "latest.pt"
    train.save_checkpoint(path, model, optimizer, stream, counts, 1, {"examples": 3}, "cpu")
    ids_next = stream.batch(3)
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    train.update(model, optimizer, [records()[i] for i in ids_next], options, "cpu", 30)
    restored = arch.construct(config(dropout=0.1), options, "cpu")
    restored_optimizer = arch.optimizer_for(restored, 1e-3, 0.1)
    restored_stream = historical.BatchStream(5, 99)
    saved = train.load_checkpoint(path, restored, restored_optimizer, restored_stream, "cpu")
    np.testing.assert_array_equal(saved["counts"], counts)
    np.testing.assert_array_equal(restored_stream.batch(3), ids_next)
    assert len(ids_next) == 2  # The incomplete epoch final batch is retained.
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    train.update(restored, restored_optimizer, [records()[i] for i in ids_next], options, "cpu", 30)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)
    assert saved["step"] == 1 and saved["counters"] == {"examples": 3}


def test_nonfinite_update_exits_without_an_optimizer_step():
    model = arch.construct(config(), spec(), "cpu")
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        train.update(model, optimizer, records(), spec(), "cpu", 30)
    assert not optimizer.state


def test_selection_uses_only_atomic_and_trained_composition_nll():
    metrics = {
        "atomic": {"nll": 2.0},
        "train_composition": {"nll": 4.0},
        "test_oo": {"nll": -1000.0, "alias_em": 1.0},
    }
    assert train.selection_score(metrics) == 3.0
    metrics["test_oo"] = {"nll": float("nan"), "alias_em": 0.0}
    assert train.selection_score(metrics) == 3.0


def test_cli_freeze_rejects_config_or_data_mutation(tmp_path):
    data = tmp_path / "data.json"
    data.write_text("{}")
    lock = tmp_path / "lock.json"
    configuration = {
        "data_file": str(data),
        "data_sha256": train.sha256(data),
        "execution_lock": str(lock),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    lock.write_text(json.dumps({"config_sha256": train.sha256(path)}))
    train.verify(configuration, path)
    with pytest.raises(ValueError, match="source_root"):
        train.verify({**configuration, "source_root": str(tmp_path)})
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="configuration"):
        train.verify(configuration, path)
    data.write_text("[]")
    with pytest.raises(ValueError, match="data"):
        train.verify(configuration)
