"""Contracts that change the interpretation of memory-reuse results."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import memory_reuse as experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def spec():
    base = json.loads((ROOT / "configs/memory-reuse-development-v1.json").read_text())["spec"]
    return dict(
        base,
        num_entities=8,
        d_model=8,
        hidden_dim=16,
        junk_length=3,
        junk_vocab_size=3,
        memory_steps=2,
        reader_steps=2,
        memory_nodes=[0, 2],
        reader_nodes=[0, 2],
        eval_repeats=2,
    )


def test_truth_split_and_input_separation(spec):
    world = experiment.build_world(spec, 67)
    assert experiment.data_audit(spec, world)["passed"]
    inputs, keys = experiment.make_inputs(spec, 4, 77)
    assert inputs.shape == (32, 5)
    assert (inputs[:, :-1] < 8).sum(1).tolist() == [1] * 32
    assert np.array_equal(inputs[inputs < 8], keys)
    assert np.all(inputs[:, -1] == 11)
    assert not np.any((inputs[:, :-1] >= 8) & (inputs[:, :-1] >= 11))
    assert np.array_equal(keys.reshape(4, 8), np.tile(np.arange(8), (4, 1)))
    assert world["train_mask"].sum() == 6
    other = experiment.build_world(spec, 68)
    # Sampling the question requires no world or mapping.
    assert np.array_equal(inputs, experiment.make_inputs(spec, 4, 77)[0])
    assert not np.array_equal(other["mapping_B"], world["mapping_B"])


def test_shared_memory_only_qk_learn_and_swap_frozen(spec):
    torch.set_num_threads(1)
    world = experiment.build_world(spec, 67)
    a = experiment.new_memory(spec, 31, "cpu")
    b = experiment.new_memory(spec, 32, "cpu")
    reader = experiment.TwoHopReader(spec, world["embeddings"], a, 31, "cpu")
    assert reader.gpt.transformer.h[0].mlp is reader.gpt.transformer.h[1].mlp
    trainable = [name for name, p in reader.named_parameters() if p.requires_grad]
    assert len(trainable) == 4
    assert all(".attn.c_q." in n or ".attn.c_k." in n for n in trainable)
    before = experiment.hash_state(reader, reader_only=True)
    reader.install(b)
    assert experiment.hash_state(reader, reader_only=True) == before
    for block in reader.gpt.transformer.h:
        assert not block.mlp_residual and not block.attn_residual
        torch.testing.assert_close(block.attn.c_v.weight, torch.eye(8))
        torch.testing.assert_close(block.attn.c_proj.weight, torch.eye(8))
    assert reader.gpt.transformer.wpe is None


def test_reader_loss_does_not_use_heldout_twohop_targets():
    torch.manual_seed(11)
    logits = torch.randn(8, 2, 12, requires_grad=True)
    labels = torch.randint(12, (8, 2))
    mask = torch.tensor([True] * 6 + [False] * 2)
    original = experiment.reader_loss(logits, labels, mask)
    alternate = labels.clone()
    alternate[~mask, 1] = (alternate[~mask, 1] + 1) % 12
    assert torch.equal(original, experiment.reader_loss(logits, alternate, mask))
    gradient = torch.autograd.grad(original, logits)[0]
    assert gradient[~mask, 1].eq(0).all()
    assert gradient[:, 0].abs().sum() > 0


def test_alignment_adds_only_atomic_vector_constraint():
    torch.manual_seed(19)
    embeddings = torch.nn.functional.normalize(torch.randn(7, 4), dim=-1)
    output = torch.randn(7, 4)
    labels = torch.randperm(7)
    ce = experiment.memory_objective(output, embeddings, labels, "ce", 10)
    aligned = experiment.memory_objective(output, embeddings, labels, "aligned", 10)
    expected = 10 * (output - embeddings[labels]).square().sum(-1).mean()
    torch.testing.assert_close(aligned - ce, expected)


def test_scoring_changed_answers_and_actual_successor(spec):
    world = experiment.build_world(spec, 67)
    _, keys = experiment.make_inputs(spec, 2, 71)
    mapping = world["mapping_B"]
    labels = np.stack((mapping[keys], mapping[mapping[keys]]), axis=1)
    logits = np.zeros((len(keys), 2, len(mapping)))
    logits[np.arange(len(keys))[:, None], np.arange(2), labels] = 1
    score = experiment.score_logits(logits, world, keys, "B")
    assert score["one_hop"] == score["two_hop"] == 1
    assert score["both_atoms_correct_coverage"] == 1
    logits[0, 0] = 0
    logits[0, 0, (labels[0, 0] + 1) % 8] = 2
    score = experiment.score_logits(logits, world, keys, "B")
    failed = {0, int(np.where(mapping[:8] == 0)[0][0])}
    assert score["both_atoms_correct_n"] == len(keys) - len(failed)


def test_causal_prefix_and_checkpoint_roundtrip(spec, tmp_path):
    world = experiment.build_world(spec, 67)
    memory = experiment.new_memory(spec, 31, "cpu")
    reader = experiment.TwoHopReader(spec, world["embeddings"], memory, 31, "cpu")
    inputs, _ = experiment.make_inputs(spec, 1, 13)
    tokens = torch.tensor(inputs)
    block = reader.gpt.transformer.h[0]
    embedded = reader.gpt.transformer.wte(tokens)
    other = embedded.clone()
    other[:, 3:] = torch.randn_like(other[:, 3:])
    torch.testing.assert_close(block(embedded)[:, :3], block(other)[:, :3], rtol=0, atol=0)
    state = experiment.cpu_state(reader)
    torch.save(state, tmp_path / "reader.pt")
    fresh = experiment.TwoHopReader(
        spec, world["embeddings"], experiment.new_memory(spec, 19, "cpu"), 19, "cpu"
    )
    fresh.load_state_dict(torch.load(tmp_path / "reader.pt", weights_only=True))
    torch.testing.assert_close(reader(tokens), fresh(tokens), rtol=0, atol=0)


def test_tiny_paired_execution_preserves_initialization_stream_and_freezes(spec, tmp_path):
    a = experiment.run(spec, 67, 31, "ce", tmp_path / "ce", "cpu")
    b = experiment.run(spec, 67, 31, "aligned", tmp_path / "aligned", "cpu")
    assert a["reader"]["initial_hash"] == b["reader"]["initial_hash"]
    assert a["reader"]["stream_hash"] == b["reader"]["stream_hash"]
    hashes = {x["memories"][m]["initial_hash"] for x in (a, b) for m in ("A", "B")}
    assert len(hashes) == 1
    for root, result in ((tmp_path / "ce", a), (tmp_path / "aligned", b)):
        with np.load(root / "endpoints.npz") as arrays:
            world = experiment.build_world(spec, 67)
            for name in ("A", "B"):
                recomputed = experiment.score_logits(
                    arrays[f"logits_{name}"], world, arrays["keys"], name
                )
                assert recomputed == result["endpoints"][name]
        assert all(result["assertions"].values())
