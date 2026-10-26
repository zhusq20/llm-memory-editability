"""Scientific contracts: graph truth, heldout scope, exact depth and causal readout."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import MergedBatchStream, ReproductionGPT
from llm_memory_editability.loop_hop_data import (
    audit_data,
    exposure_summary,
    load,
    make_table,
    pack_rows,
    path_details,
    prepare,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
torch.set_num_threads(1)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RUNNER = module("loop_hop_runner", "run_loop_hop.py")
LAUNCHER = module("loop_hop_launcher", "launch_loop_hop.py")


@pytest.fixture
def dataset(tmp_path):
    config = {
        "data": {
            "world_seed": 41,
            "entities_per_graph": 8,
            "relations": 3,
            "train_compositions_per_hop": 12,
            "evaluation_templates": 2,
            "training_panel_size": 7,
        }
    }
    prepare(config, tmp_path / "data")
    return load(tmp_path / "data")


def tiny(layers=1, repeats=2, disco=False, alpha=1.0):
    config = GPT2Config(
        vocab_size=24,
        n_embd=12,
        n_head=3,
        n_layer=layers,
        n_positions=16,
        attn_pdrop=0,
        resid_pdrop=0,
        embd_pdrop=0,
        use_cache=False,
    )
    torch.manual_seed(71)
    return SupervisedAlignmentGPT(config, repeats, discoloop=disco, discoloop_alpha=alpha)


def test_truth_separation_and_balanced_suffix(dataset):
    world, panels, metadata = dataset
    assert audit_data(world, panels, metadata)["passed"]
    assert len(world["atoms"]) == 48
    for k in (2, 3, 4):
        nodes, edges = path_details(panels[f"ood_{k}"], metadata, world["permutations"])
        assert np.all(nodes >= 8)
        assert np.all(edges >= 24)
        assert np.all(world[f"train_{k}"][:, 0] < 8)


def test_full_frozen_dataset_when_mounted():
    root = Path("/inputs/data")
    if not root.exists():
        pytest.skip("Full registered dataset is checked in the frozen execution container")
    world, panels, metadata = load(root)
    assert audit_data(world, panels, metadata)["passed"]
    assert metadata["training_examples"] == 40000
    assert metadata["vocab_size"] == 1013
    assert sum(map(len, panels.values())) == 43000


def test_reject_training_query_leak(dataset):
    world, panels, metadata = copy.deepcopy(dataset)
    world["train_3"][0] = panels["id_3"][0]
    with pytest.raises(ValueError, match="leaked"):
        audit_data(world, panels, metadata)


def test_reject_nonpermutation(dataset):
    world, panels, metadata = copy.deepcopy(dataset)
    world["permutations"][0, 0, 0] = world["permutations"][0, 0, 1]
    with pytest.raises(ValueError, match="not a permutation"):
        audit_data(world, panels, metadata)


@pytest.mark.parametrize("hop", [1, 2, 3, 4])
def test_only_answer_and_eos_targets(dataset, hop):
    world, panels, metadata = dataset
    rows = world["atoms"] if hop == 1 else panels[f"ood_{hop}"]
    x, positions, labels = pack_rows(rows, metadata)
    assert x.shape == (len(rows), 6)
    assert np.all(positions == [hop, hop + 1])
    assert np.all(labels[:, 0] == rows[:, -1] + metadata["entity_offset"])
    assert np.all(labels[:, 1] == metadata["end_marker"])
    assert np.all(x[:, hop + 1] == labels[:, 0])
    assert np.all(x[:, hop + 2 :] == metadata["pad_token"])


def test_stream_tails_masking_and_actual_exposure(dataset):
    world, _, metadata = dataset
    table = make_table(world, metadata, "cpu")
    assert table[0].shape == (85, 6)
    assert torch.all(table[2][-1] == -100)
    stream = MergedBatchStream(84, 91)
    counts = np.zeros(84, dtype=np.int64)
    for _ in range(18):
        index, count = stream.batch(13)
        assert np.count_nonzero(index != 84) == count
        counts += np.bincount(index, minlength=85)[:-1]
    summary = exposure_summary(counts, world, metadata, 18, 13)
    assert summary["examples"] == counts.sum() == 220
    assert summary["masked_batch_slots"] == 14
    assert summary["supervised_tokens"] == 440
    assert summary["executed_input_tokens"] == 18 * 13 * 6


@pytest.mark.parametrize("layers,repeats", [(1, 2), (1, 3), (1, 4), (2, 2)])
@pytest.mark.parametrize("disco", [False, True])
def test_actual_primitive_depth_and_causal_labels(layers, repeats, disco):
    model = tiny(layers, repeats, disco).eval()
    counts = {"qkv": 0, "mlp": 0}
    handles = []

    def count(name):
        def hook(module, args, output):
            counts[name] += 1

        return hook

    for block in model.transformer.h:
        handles += [
            block.attn.c_attn.register_forward_hook(count("qkv")),
            block.mlp.register_forward_hook(count("mlp")),
        ]
    x = torch.tensor([[3, 19, 20, 21, 22, 6]])
    positions = torch.tensor([[4, 5]])
    first = model(x, positions)
    assert counts == {"qkv": layers * repeats, "mlp": layers * repeats}
    for handle in handles:
        handle.remove()
    changed = x.clone()
    changed[:, 5] = 7
    second = model(changed, positions)
    assert torch.equal(first[:, 0], second[:, 0])
    assert not torch.equal(first[:, 1], second[:, 1])


def test_disabled_channel_forward_and_every_gradient_match():
    candidate = tiny(2, 2, True, 0).train()
    reference = ReproductionGPT(candidate.config, 2).train()
    reference.load_state_dict(candidate.state_dict())
    x, pos = torch.tensor([[3, 19, 20, 6, 0, 0]]), torch.tensor([[2, 3]])
    labels = torch.tensor([[6, 2]])
    logits = []
    grads = []
    for model in (reference, candidate):
        output = model(x, pos)
        F.cross_entropy(output.flatten(0, 1), labels.flatten()).backward()
        logits.append(output)
        grads.append({n: p.grad for n, p in model.named_parameters()})
    assert torch.equal(*logits)
    assert grads[0].keys() == grads[1].keys()
    assert all(torch.equal(grads[0][n], grads[1][n]) for n in grads[0])


def test_paired_initialization_and_independent_layers():
    loop, disco = tiny(2, 2), tiny(2, 2, True)
    assert all(torch.equal(v, disco.state_dict()[k]) for k, v in loop.state_dict().items())
    assert not torch.equal(
        loop.transformer.h[0].attn.c_attn.weight, loop.transformer.h[1].attn.c_attn.weight
    )
    a, b = tiny(1, 2), tiny(1, 4)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


@pytest.mark.parametrize("disco", [False, True])
def test_trace_observes_actual_forward_without_modification(disco):
    model = tiny(2, 2, disco).eval()
    x, pos = torch.tensor([[3, 19, 20, 6, 0, 0]]), torch.tensor([[2, 3]])
    before = model(x, pos)
    traced, states = RUNNER.traced_states(model, x, pos)
    assert torch.equal(before, traced)
    assert len([s for s in states if s[0] == "after_layer"]) == 4
    assert len([s for s in states if s[0] == "after_boundary"]) == int(disco)
    assert all(not block.mlp._forward_hooks for block in model.transformer.h)


def test_premise_coverage_includes_failures(dataset):
    world, panels, metadata = dataset
    rows = panels["ood_3"]
    correct = np.ones(len(world["atoms"]), dtype=bool)
    _, edges = path_details(rows, metadata, world["permutations"])
    correct[edges[0, 0]] = False
    prediction = {"answer": rows[:, -1] + metadata["entity_offset"]}
    summary = RUNNER.premise_metrics(rows, prediction, correct, world, metadata)
    assert 0 < summary["premise_correct_n"] < len(rows)
    assert summary["answer_accuracy_given_premises"] == 1
    assert 0 < summary["premise_coverage"] < 1


def test_reserved_idle_gpu_is_not_admitted():
    config = json.loads((ROOT / "configs/loop-hop-development-v1.json").read_text())
    row = {
        "process_present": False,
        "explicit_allocations": [{"container": "other-project"}],
        "used_mib": 0,
        "utilization": 0,
    }
    assert not LAUNCHER.admissible(row, config["runtime"])
    row["explicit_allocations"] = []
    assert LAUNCHER.admissible(row, config["runtime"])
    row["process_present"] = True
    assert not LAUNCHER.admissible(row, config["runtime"])


def test_registered_matrix_and_budget(monkeypatch):
    config = json.loads((ROOT / "configs/loop-hop-development-v1.json").read_text())
    # The test container is deliberately pinned to only two utility CPUs;
    # validate the planned host matrix independently of that runtime affinity.
    monkeypatch.setattr(LAUNCHER.os, "sched_getaffinity", lambda _: set(range(96)))
    LAUNCHER.validate_config(config)
    assert sum(config["training"]["steps"] for _ in config["arms"]) == 2400000
    config["arms"].pop("loop-l1-r2")
    with pytest.raises(ValueError, match="eight-condition"):
        LAUNCHER.validate_config(config)
