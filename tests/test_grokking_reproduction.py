"""Scientific contracts for the original-task grokking replication."""

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from llm_memory_editability.grok_depth import EpochStream
from llm_memory_editability.grokking_reproduction import (
    MergedBatchStream,
    ReproductionGPT,
    audit_graph,
    build_graph,
    encode_rows,
    learning_rate,
    optimizer_for,
)


def test_graph_matches_author_generator():
    path = Path("results/grokking-reproduction-v1/reference/composition.ipynb")
    if not path.exists():
        pytest.skip("Pinned author reference not downloaded")
    source = "\n\n".join(
        "".join(c.get("source", [])) for c in json.loads(path.read_text())["cells"]
    )
    namespace = {}
    exec(source[: source.index("NUM_ENTITY_IN =")], namespace)
    np.random.seed(917)
    _, _, id_atoms, ood_atoms, train, test, strict = namespace["build_dataset"](
        40, 24, out_degree=4, split_train_inferred=True
    )

    def parse(items):
        return {tuple(map(int, re.findall(r"<[er]_(\d+)>", x["target_text"]))) for x in items}

    world = build_graph(917, entities=40, relations=24, degree=4)
    audit_graph(world, 4)
    assert parse(id_atoms) == set(map(tuple, world["atoms"][~world["ood"]]))
    assert parse(ood_atoms) == set(map(tuple, world["atoms"][world["ood"]]))
    assert parse(train) == set(map(tuple, world["chains"][world["train_order"]]))
    assert parse(test) == set(map(tuple, world["chains"][world["test_ii"]]))
    assert parse(strict) == set(map(tuple, world["chains"][world["kind"] == 3]))


def test_random_edge_roles_and_nested_support():
    w = build_graph(802, entities=100, relations=30, degree=10)
    audit_graph(w, 10)
    first, second = w["ood"][w["first"]], w["ood"][w["second"]]
    for kind in range(4):
        mask = w["kind"] == kind
        assert mask.any()
        assert np.all(first[mask] == bool(kind // 2))
        assert np.all(second[mask] == bool(kind % 2))
    assert set(w["train_order"][:200]) <= set(w["train_order"][:400])


@pytest.mark.parametrize("row", [[2, 1, 7], [2, 1, 3, 7]])
def test_answer_and_end_targets_only(row):
    metadata = {"pad_token": 0, "entity_offset": 20, "relation_offset": 60, "end_marker": 90}
    x, pos, labels = encode_rows(np.array([row]), metadata)
    assert labels.tolist() == [[27, 90]]
    assert pos.tolist() == [[len(row) - 2, len(row) - 1]]
    assert x[0, 0] == 22 and x[0, len(row) - 1] == 27
    assert np.all(x[0, 1 : len(row) - 1] == np.array(row[1:-1]) + 60)


def test_supervised_readout_loss_and_gradients_match_native_gpt2():
    torch.set_num_threads(2)
    cfg = GPT2Config(
        vocab_size=97,
        n_positions=16,
        n_embd=32,
        n_layer=2,
        n_head=4,
        resid_pdrop=0,
        attn_pdrop=0,
        embd_pdrop=0,
        use_cache=False,
    )
    optimized = ReproductionGPT(cfg)
    native = GPT2LMHeadModel(copy.deepcopy(cfg))
    native.transformer.load_state_dict(optimized.transformer.state_dict())
    native.tie_weights()
    metadata = {"pad_token": 0, "entity_offset": 20, "relation_offset": 60, "end_marker": 90}
    losses = []
    for rows in [np.array([[2, 1, 7], [3, 2, 9]]), np.array([[2, 1, 3, 7], [3, 2, 1, 9]])]:
        x, pos, labels = [torch.as_tensor(a) for a in encode_rows(rows, metadata)]
        padded = torch.zeros((2, 10), dtype=torch.long)
        padded[:, :4] = x
        padded[torch.arange(2), pos[:, 1] + 1] = 90
        full_labels = torch.full_like(padded, -100)
        full_labels[torch.arange(2)[:, None], pos + 1] = labels
        a = torch.nn.functional.cross_entropy(optimized(x, pos).flatten(0, 1), labels.flatten())
        b = native(padded, labels=full_labels).loss
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-6)
        losses.append((a, b))
    sum(a for a, _ in losses).backward()
    sum(b for _, b in losses).backward()
    for (_, a), (_, b) in zip(
        optimized.transformer.named_parameters(), native.transformer.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_shared_blocks_are_reused_without_reembedding():
    cfg = GPT2Config(
        vocab_size=50,
        n_positions=16,
        n_embd=16,
        n_layer=2,
        n_head=4,
        resid_pdrop=0,
        attn_pdrop=0,
        embd_pdrop=0,
    )
    model = ReproductionGPT(cfg, repeats=2)
    counts = {"embedding": 0, "block": 0}

    def embedding_hook(*_):
        counts["embedding"] += 1

    def block_hook(*_):
        counts["block"] += 1

    handles = [model.transformer.wte.register_forward_hook(embedding_hook)]
    handles.extend(b.mlp.register_forward_hook(block_hook) for b in model.transformer.h)
    model(torch.tensor([[1, 2, 3, 4]]))
    for h in handles:
        h.remove()
    assert counts == {"embedding": 1, "block": 4}
    assert len(model.transformer.h) == 2


def test_optimizer_excludes_author_bias_and_norm_names():
    cfg = GPT2Config(vocab_size=50, n_positions=16, n_embd=16, n_layer=2, n_head=4)
    model = ReproductionGPT(cfg)
    optimizer = optimizer_for(model, torch.tensor(1e-4), 0.3)
    no_decay = {id(p) for p in optimizer.param_groups[1]["params"]}
    for name, p in model.named_parameters():
        assert (id(p) in no_decay) == any(s in name for s in ["bias", "ln"])


def test_lr_matches_hf_optimizer_then_scheduler_order():
    spec = {"lr": 1e-4, "warmup": 2000}
    assert learning_rate(spec, 1) == 0
    assert learning_rate(spec, 2001) == 1e-4
    assert learning_rate(spec, 1500000) == 1e-4


def test_merged_sampling_and_exact_stream_restore():
    a = EpochStream(97, 123)
    initial = a.take(153)
    saved = a.state_dict()
    expected = a.take(300)
    b = EpochStream(97, 991)
    b.load_state_dict(saved)
    assert np.array_equal(expected, b.take(300))
    counts = np.bincount(np.concatenate([initial, expected]), minlength=97)
    assert counts.max() - counts.min() <= 1


def test_epoch_boundary_matches_partial_batch_without_cross_epoch_filling():
    stream = MergedBatchStream(10, 44)
    batches = [stream.batch(4) for _ in range(3)]
    assert [n for _, n in batches] == [4, 4, 2]
    assert sorted(np.concatenate([a[:n] for a, n in batches])) == list(range(10))
    assert batches[-1][0][-2:].tolist() == [10, 10]
    assert stream.batch(4)[1] == 4


def test_graph_padding_does_not_change_partial_batch_loss_or_gradient():
    cfg = GPT2Config(
        vocab_size=101,
        n_positions=16,
        n_embd=16,
        n_layer=2,
        n_head=4,
        resid_pdrop=0,
        attn_pdrop=0,
        embd_pdrop=0,
    )
    a = ReproductionGPT(cfg)
    b = copy.deepcopy(a)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    pos = torch.tensor([[2, 3], [2, 3]])
    targets = torch.tensor([[4, 90], [8, 90]])
    xpad = torch.cat([tokens, torch.zeros((2, 4), dtype=torch.long)])
    ppad = torch.cat([pos, pos])
    ypad = torch.cat([targets, torch.full((2, 2), -100)])
    loss_a = torch.nn.functional.cross_entropy(a(tokens, pos).flatten(0, 1), targets.flatten())
    loss_b = torch.nn.functional.cross_entropy(b(xpad, ppad).flatten(0, 1), ypad.flatten())
    torch.testing.assert_close(loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, atol=2e-6, rtol=2e-5)


def test_frozen_matrix_has_original_size_and_fixed_budget():
    c = json.loads(Path("configs/grokking-reproduction-development-v1.json").read_text())
    assert len(c["runs"]) == 8 and c["optimization_updates"] == 12000000
    assert c["evaluation_nodes"][0] == 0 and c["evaluation_nodes"][-1] == 1500000
    assert set(c["checkpoint_nodes"]) <= set(c["evaluation_nodes"])
    for s in c["runs"]:
        assert s["unique_layers"] * s["repeats"] == 8
        assert s["batch_size"] == 512 and s["steps"] == 1500000
