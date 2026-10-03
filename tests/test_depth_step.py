"""Contracts affecting truth, end-to-end text, scoring and paired depth initialization."""

import copy
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.depth_step import (
    BOS,
    EOS,
    EVALUATION_SPLITS,
    TRAIN_SPLITS,
    _validate_output,
    audit_world,
    build_world,
    construct,
    data_digest,
    evaluate,
    generate_rows,
    pack_rows,
    prompt_rows,
    run_name,
    truth_path_details,
)
from llm_memory_editability.grok_depth import EpochStream, SmallGPT
from llm_memory_editability.grok_loop_data import build_world as old_world


@pytest.fixture(scope="module")
def spec():
    return {
        "world_seed": 751011,
        "entities": 64,
        "relations": 4,
        "degree": 4,
        "phi": 4.0,
        "id_fraction": 0.75,
        "id_test_fraction": 0.2,
        "evaluation_size": 512,
        "width": 32,
        "heads": 4,
        "layers": 2,
        "repeats": 1,
        "dropout": 0.0,
        "initialization": 752011,
        "stream_seed": 753011,
        "batch_size": 128,
        "steps": 64000,
    }


@pytest.fixture(scope="module")
def world(spec):
    return build_world(spec)


def test_three_lengths_share_exact_graph_id_mask_and_full_reserved_pools(spec, world):
    report = audit_world(world)
    assert report["counts"] == {
        "atomic": 256,
        "train_2": 444,
        "familiar_2": 124,
        "strict_2": 64,
        "train_3": 768,
        "familiar_3": 329,
        "strict_3": 59,
        "train_4": 768,
        "familiar_4": 1025,
        "strict_4": 53,
    }
    for hops in (2, 3, 4):
        source = old_world(dict(spec, hops=hops))
        np.testing.assert_array_equal(world["atomic"], source["atomic"] + 1)
        np.testing.assert_array_equal(world[f"familiar_{hops}"], source["test_full_composite"] + 1)
        np.testing.assert_array_equal(world[f"strict_{hops}"], source["ood_composite"] + 1)
        source_ids = set(map(tuple, source["id_atomic"]))
        expected = [tuple(row) in source_ids for row in source["atomic"]]
        assert world["metadata"]["id_mask"] == expected
        assert (
            world["metadata"]["source_worlds"][str(hops)]["dataset_sha256"]
            == source["metadata"]["dataset_sha256"]
        )
    assert world["metadata"]["source_token_shift"] == 1
    assert world["metadata"]["source_worlds"]["2"]["train_composite_capped"]


def test_truth_edges_and_strict_facts_never_used_at_any_training_length(world):
    trained = set()
    lookup = {(int(h), int(r)): int(t) for h, r, t in world["atomic"]}
    for hops in (2, 3, 4):
        rows = world[f"train_{hops}"]
        nodes, edges = truth_path_details(world, rows)
        assert nodes.shape == (len(rows), hops + 1)
        assert edges.shape == (len(rows), hops)
        trained.update(edges.ravel())
        for i, row in enumerate(rows):
            head = int(row[0])
            for j, relation in enumerate(row[1:-1]):
                head = lookup[head, int(relation)]
                assert head == nodes[i, j + 1]
            assert head == row[-1]
    for hops in (2, 3, 4):
        strict = truth_path_details(world, world[f"strict_{hops}"])[1]
        assert not set(strict.ravel()) & trained


@pytest.mark.parametrize("change", ["truth", "id_mask", "overlap", "duplicate"])
def test_invalid_data_contracts_are_rejected(world, change):
    damaged = copy.deepcopy(world)
    if change == "truth":
        tail = damaged["familiar_3"][0, -1]
        damaged["familiar_3"][0, -1] = 3 + (tail - 3 + 1) % 64
    elif change == "id_mask":
        edge = truth_path_details(damaged, damaged["train_2"][:1])[1][0, 0]
        damaged["metadata"]["id_mask"][edge] = False
    elif change == "overlap":
        damaged["familiar_2"][0] = damaged["train_2"][0]
    else:
        damaged["train_4"][1] = damaged["train_4"][0]
    with pytest.raises(ValueError):
        audit_world(damaged)


def test_frozen_digest_enforced_and_architecture_does_not_change_world(spec, world):
    digest = data_digest(world)
    for layers, repeats in ((1, 4), (6, 1)):
        actual = build_world(dict(spec, layers=layers, repeats=repeats, frozen_data_sha256=digest))
        assert data_digest(actual) == digest
    with pytest.raises(ValueError, match="frozen"):
        build_world(dict(spec, frozen_data_sha256="invalid"))


def test_compact_prompt_has_separator_full_token_ce_and_no_intermediate_labels(world):
    separator = world["metadata"]["separator_token"]
    assert separator == 71
    assert world["metadata"]["vocab_size"] == 72
    for split in TRAIN_SPLITS:
        rows = world[split][:3]
        prompts = prompt_rows(rows, separator)
        assert np.all(prompts[:, 0] == BOS)
        assert np.all(prompts[:, -1] == separator)
        np.testing.assert_array_equal(prompts[:, 1:-1], rows[:, :-1])
        tokens, labels = pack_rows(rows, separator=separator)
        length = rows.shape[1] + 2
        expected = np.c_[
            rows[:, :-1], np.full(len(rows), separator), rows[:, -1], np.full(len(rows), EOS)
        ]
        np.testing.assert_array_equal(labels[:, :length], expected)
        assert np.all(labels[:, length:] == -100)
        assert np.all(tokens[:, length:] == 0)
        assert not (labels == BOS).any()
    assert len(pack_rows(world["train_4"][:1])[0][0]) == 8
    assert sum(int((pack_rows(world[key][:1])[1] >= 0).sum()) for key in TRAIN_SPLITS) * 32 == 832
    with pytest.raises(ValueError):
        pack_rows(world["train_4"][:1], sequence=7)


def test_paired_initialization_all_architectures_and_effective_depth(spec):
    torch.set_num_threads(1)
    architectures = [(layers, 1) for layers in (1, 2, 3, 4, 6)] + [(1, r) for r in (2, 3, 4, 6)]
    models = {
        (layers, r): construct(dict(spec, layers=layers, repeats=r), "cpu")
        for layers, r in architectures
    }
    reference = models[1, 1]
    for (layers, repeats), model in models.items():
        assert len(model.blocks) == layers and model.effective_depth == layers * repeats
        assert all(p.requires_grad for p in model.parameters())
        for name in ("token.weight", "position.weight", "ln_final.weight", "ln_final.bias"):
            torch.testing.assert_close(
                model.state_dict()[name], reference.state_dict()[name], rtol=0, atol=0
            )
    for depth in (2, 3, 4, 6):
        ordinary, loop = models[depth, 1], models[1, depth]
        for key, value in ordinary.blocks[0].state_dict().items():
            torch.testing.assert_close(value, loop.blocks[0].state_dict()[key], rtol=0, atol=0)
    assert not torch.equal(
        models[2, 1].blocks[0].mlp.up.weight, models[2, 1].blocks[1].mlp.up.weight
    )
    assert len({run_name(dict(spec, layers=layers, repeats=r)) for layers, r in architectures}) == 9


def test_standard_full_forward_causal_and_all_parameters_train(spec, world):
    torch.set_num_threads(1)
    model = construct(spec, "cpu").eval()
    ordinary = SmallGPT(model.config, dropout=0.0).eval()
    ordinary.load_state_dict(model.state_dict())
    tokens = torch.as_tensor(pack_rows(world["train_4"][:2])[0])
    logits = model(tokens)
    torch.testing.assert_close(logits, ordinary(tokens), rtol=0, atol=0)
    assert logits.shape == (2, 8, 72)
    changed = tokens.clone()
    changed[:, 5:] = 3
    torch.testing.assert_close(model(changed)[:, :5], logits[:, :5], rtol=0, atol=0)
    labels = torch.as_tensor(pack_rows(world["train_4"][:2])[1])
    F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100).backward()
    for param in (
        model.token.weight,
        model.position.weight,
        model.blocks[0].attention.qkv.weight,
        model.blocks[0].mlp.up.weight,
        model.blocks[0].mlp.down.weight,
    ):
        assert (
            param.grad is not None
            and torch.isfinite(param.grad).all()
            and param.grad.abs().sum() > 0
        )


class TruthOracle(torch.nn.Module):
    def __init__(self, world, wrong_eos=False):
        super().__init__()
        self.lookup = {(int(h), int(r)): int(t) for h, r, t in world["atomic"]}
        self.separator = world["metadata"]["separator_token"]
        self.vocab = world["metadata"]["vocab_size"]
        self.wrong_eos = wrong_eos

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, self.vocab), -20.0, device=tokens.device)
        for i, row in enumerate(tokens.tolist()):
            if row[-1] == self.separator:
                current = row[1]
                for relation in row[2:-1]:
                    current = self.lookup.get((current, relation), 3)
                result = current
            else:
                result = 0 if self.wrong_eos else EOS
            logits[i, -1, result] = 20.0
        return logits


def test_free_tail_eos_coverage_and_autonomous_truth_scoring(world):
    oracle = TruthOracle(world)
    metrics, predictions = evaluate(oracle, world, "cpu")
    assert oracle.training
    for name in EVALUATION_SPLITS:
        assert metrics[name]["accuracy"] == metrics[name]["answer_accuracy"] == 1.0
        assert predictions[name + "_correct"].all()
        if name != "atomic":
            assert metrics[name]["atomic_correct_coverage"] == 1.0
            assert metrics[name]["conditional_accuracy"] == 1.0
            assert metrics[name]["autonomous_two_calls"] == 1.0
            assert metrics[name]["autonomous_path_accuracy"] == 1.0
    metrics, _ = evaluate(TruthOracle(world, wrong_eos=True), world, "cpu")
    assert metrics["atomic"]["answer_accuracy"] == 1.0
    assert metrics["atomic"]["accuracy"] == 0.0
    assert metrics["familiar_4"]["conditional_accuracy"] is None
    assert metrics["familiar_4"]["autonomous_two_calls"] == 0.0


def test_empty_pools_never_report_nan(world):
    empty = dict(world)
    empty["strict_4"] = np.empty((0, 6), dtype=np.int64)
    metrics, predictions = evaluate(TruthOracle(world), empty, "cpu")
    assert metrics["strict_4"]["n"] == 0
    assert metrics["strict_4"]["accuracy"] is None
    assert metrics["strict_4"]["atomic_correct_coverage"] is None
    assert predictions["strict_4_autonomous_generated"].shape == (0, 4, 2)
    json.dumps(metrics, allow_nan=False)
    task, prediction = generate_rows(TruthOracle(world), empty["strict_4"], "cpu")
    assert task["answer_nll"] is None and prediction["generated"].shape == (0, 2)


def test_epoch_stream_batches_and_saved_continuation_match_across_architectures(world, spec):
    for i, name in enumerate(TRAIN_SPLITS):
        first = EpochStream(len(world[name]), spec["stream_seed"] + i)
        second = EpochStream(len(world[name]), spec["stream_seed"] + i)
        np.testing.assert_array_equal(
            first.take(2048).reshape(64, 32), second.take(2048).reshape(64, 32)
        )
        restored = EpochStream(len(world[name]), 0)
        restored.load_state_dict(first.state_dict())
        np.testing.assert_array_equal(first.take(32), restored.take(32))


@pytest.mark.parametrize(
    "artifact",
    [
        "latest.pt",
        "complete.json",
        "world.npz",
        "spec.json",
        "predictions-000000.npz",
        "exposures.npz",
    ],
)
def test_attempt_artifacts_never_overwritten_but_scheduler_files_allowed(tmp_path, artifact):
    (tmp_path / "input-spec.json").write_text("{}")
    (tmp_path / "train-process.log").write_text("scheduler")
    _validate_output(tmp_path)
    (tmp_path / artifact).write_bytes(b"existing experiment")
    with pytest.raises(FileExistsError):
        _validate_output(tmp_path)
    assert (tmp_path / artifact).read_bytes() == b"existing experiment"
