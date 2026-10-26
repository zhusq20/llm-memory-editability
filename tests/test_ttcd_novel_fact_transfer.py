"""Contracts that distinguish absent-key addition from answer replacement."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, load_fast, save_fast, tensor_digest
from transformers import GPT2Config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "novel_transfer", SCRIPTS / "run_ttcd_novel_fact_transfer.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)

META = {
    "entities": 8,
    "relations": 4,
    "entity_offset": 3,
    "relation_offset": 11,
    "end_marker": 20,
    "pad_token": 0,
}
WRITER = {"gap": 17, "teacher_window": 32, "student_window": 16}


def test_missing_key_packet_and_existing_key_rejection():
    packet = experiment.addition_packet([1, 1, 2], META, WRITER)
    assert packet == [4, 12, 5, 20] + [0] * 17 + [4, 12]
    assert len(packet) - 1 - 3 >= 16 and len(packet) - 1 < 32
    world = SimpleNamespace(lookup=np.full((8, 4), -1), entities=8)
    fact = {
        "novel_fact": [1, 1, 2],
        "preexisting_truth_for_novel_key": None,
        "first_hop": [],
        "second_hop": [],
    }
    experiment.validate_fact(fact, world)
    world.lookup[1, 1] = 0
    with pytest.raises(ValueError, match="never overwrites"):
        experiment.validate_fact(fact, world)


def test_nonentity_bridge_never_replaced_by_oracle(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No second query is allowed for a nonentity first answer")

    monkeypatch.setattr(experiment, "greedy_rows", forbidden)
    queries = [{"query": [0, 1, 2], "bridge": 1, "answer": 2}]
    result = experiment.two_step(None, None, queries, {"answer": [20], "stop": [20]}, META)
    assert result["valid_entity_head"] == [False]
    assert result["second_rows"] == [] and result["final_correct"] == [False]


def test_actual_local_write_freezes_backbone_and_reloads_exactly(tmp_path):
    torch.manual_seed(20261016)
    torch.set_num_threads(1)
    config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_embd=8,
        n_head=2,
        n_layer=4,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    reader = TTCDReader(
        SupervisedAlignmentGPT(config, repeats=2, discoloop=True), native_causal_attention=True
    ).eval()
    before = tensor_digest(reader.backbone.state_dict())
    baseline = reader.empty_fast()
    baseline_hash = tensor_digest(baseline)
    packet = torch.tensor([experiment.addition_packet([1, 1, 2], META, WRITER)])
    fast, _ = reader.write(
        packet, baseline, teacher_window=32, student_window=16, learning_rate=0.001
    )
    assert tensor_digest(reader.backbone.state_dict()) == before
    assert tensor_digest(baseline) == baseline_hash and tensor_digest(fast) != baseline_hash
    assert all(not p.requires_grad for p in reader.parameters())
    path = tmp_path / "fast.pt"
    save_fast(path, reader, fast, {"fact": [1, 1, 2]})
    loaded, provenance = load_fast(path, reader)
    assert provenance == {"fact": [1, 1, 2]}
    assert tensor_digest(loaded) == tensor_digest(fast)
    with torch.no_grad():
        query = torch.tensor([[4, 12]])
        assert torch.equal(reader(query, fast=fast), reader(query, fast=loaded))
