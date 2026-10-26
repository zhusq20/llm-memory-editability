"""Observation must preserve the model, causality, and round/position identity."""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest

SPEC = importlib.util.spec_from_file_location(
    "round_diagnostic",
    Path(__file__).resolve().parents[1] / "scripts/check_discoloop_edit_rounds.py",
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def tiny_reader():
    torch.manual_seed(20261016)
    torch.set_num_threads(1)
    config = GPT2Config(
        vocab_size=32,
        n_positions=12,
        n_embd=8,
        n_head=2,
        n_layer=4,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    return TTCDReader(
        SupervisedAlignmentGPT(config, repeats=2, discoloop=True),
        native_causal_attention=True,
    ).eval()


def test_trace_does_not_change_logits_weights_or_fast_state():
    reader = tiny_reader()
    fast = {k: torch.randn_like(v) * 0.001 for k, v in reader.empty_fast().items()}
    model_before = tensor_digest(reader.backbone.state_dict())
    fast_before = tensor_digest(fast)
    tokens = torch.tensor([[3, 12, 14, 5], [4, 13, 15, 6]])
    positions = torch.tensor([[2, 3], [2, 3]])
    with torch.no_grad():
        expected = reader(tokens, positions, fast=fast)
        with diagnostic.RoundTrace(reader) as trace:
            observed = reader(tokens, positions, fast=fast)
        states = trace.states()
        assert torch.equal(expected, observed)
        assert torch.equal(
            reader.backbone._discoloop_boundary(states["round1_end"]),
            states["round2_input"],
        )
        final = torch.nn.functional.linear(
            reader.backbone.transformer.ln_f(states["round2_layer4"]),
            reader.backbone.transformer.wte.weight,
        )
        assert torch.equal(final[:, 2].argmax(-1), observed[:, 0].argmax(-1))
    assert model_before == tensor_digest(reader.backbone.state_dict())
    assert fast_before == tensor_digest(fast)
    assert all(not block.ln_1._forward_pre_hooks for block in reader.backbone.transformer.h)


def test_bridge_readout_cannot_see_later_relation_or_answer():
    reader = tiny_reader()
    fast = reader.empty_fast()
    first = torch.tensor([[3, 12, 14, 5]])
    second = torch.tensor([[3, 12, 17, 9]])
    with torch.no_grad():
        with diagnostic.RoundTrace(reader) as a:
            reader(first, fast=fast)
        with diagnostic.RoundTrace(reader) as b:
            reader(second, fast=fast)
    for stage in a.states():
        assert torch.equal(a.states()[stage][:, :2], b.states()[stage][:, :2])


def test_inspection_uses_correct_atomic_and_two_hop_positions():
    reader = tiny_reader()
    fast = reader.empty_fast()
    metadata = {
        "pad_token": 0,
        "entity_offset": 2,
        "relation_offset": 12,
        "end_marker": 31,
    }
    atom = diagnostic.inspect_rows(reader, fast, np.array([[1, 0, 4]]), metadata, [5], kind="atom")
    chain = diagnostic.inspect_rows(
        reader,
        fast,
        np.array([[1, 0, 1, 4]]),
        metadata,
        [5],
        kind="two_hop",
        bridges=[3],
    )
    assert atom["answer_position"] == [1]
    assert chain["answer_position"] == [2]
    assert chain["stages"]["round1_end"]["bridge"]["expected"] == [5]
    assert chain["bridge_channel"]["boundary_reconstruction_exact"]
    assert not torch.is_grad_enabled() or all(not v.requires_grad for v in fast.values())
