"""Readout, causal isolation, and donor scope for the checkpoint diagnosis."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOOP_DIAGNOSTIC_SOURCE", str(ROOT))
spec = importlib.util.spec_from_file_location(
    "state_regression_diagnosis", ROOT / "scripts/diagnose_loop_state_regression.py"
)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def model():
    torch.manual_seed(72)
    config = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    result = LoopNativeHandoffGPT(config, 4, handoff="state-mlp").eval()
    with torch.no_grad():
        result.handoff_score_out.fill_(0.2)
    return result


def inputs():
    return (
        torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5]]),
        torch.tensor([[3, 4], [3, 4]]),
        torch.tensor([[7, 2], [5, 2]]),
    )


def test_observer_matches_native_recurrence_and_causal_prefix_without_parameter_change():
    current = model()
    result = diag.engineering(current, *inputs(), torch.device("cpu"))
    assert result["native_answer_gamma_and_change_exact"]
    assert result["future_answer_causal_prefix_exact"]
    assert result["parameters_unchanged"]


def test_truth_labels_only_change_diagnostic_nll_and_margin_never_execution():
    current = model()
    tokens, positions, labels = inputs()
    original = diag.trace(current, tokens, positions, labels, 8)
    changed = diag.trace(current, tokens, positions, (labels + 1) % 23, 8)
    for key in ("answer", "aggregate_token", "candidate_token", "score", "gamma"):
        np.testing.assert_array_equal(original[key], changed[key])
    assert not np.array_equal(original["nll"], changed["nll"])
    assert all(p.grad is None for p in current.parameters())


def test_score_observer_is_restored_even_if_readout_fails(monkeypatch):
    current = model()
    original = current.score

    def fail(*args, **kwargs):
        raise ValueError("readout failure")

    monkeypatch.setattr(diag, "_readout", fail)
    try:
        diag.trace(current, *inputs(), 8)
    except ValueError:
        pass
    assert current.score == original


def test_scorer_exchange_changes_all_and_only_the_documented_handoff_parameters():
    current = model()
    state = {k: v.clone() for k, v in current.state_dict().items()}
    snapshots = {
        step: {"model": state, "ema": {k: torch.full_like(v, value) for k, v in state.items()}}
        for step, value in ((1, 0.5), (2, 0.75))
    }
    diag.install(current, snapshots, 1, 2)
    for key, parameter in current.named_parameters():
        expected = 0.75 if key in diag.SCORE_KEYS else 0.5
        assert torch.all(parameter == expected), key
        assert not parameter.requires_grad


def test_summary_distinguishes_final_accuracy_from_ever_correct_and_prefix_retention():
    rows = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1]])
    tokens = np.tile(np.array([5, 6])[None, :, None], (8, 1, 6))
    aggregate = tokens.copy()
    aggregate[7, 0, 2] = 6
    answer = np.tile(np.array([5, 6])[None], (8, 1))
    answer[7, 0] = 6
    values = {"answer": answer, "answer_margin": np.ones((8, 2)), "nll": np.ones((8, 2, 2))}
    for key in ("raw_token", "candidate_token", "aggregate_token"):
        values[key] = aggregate if key == "aggregate_token" else tokens
    for key in (
        "score",
        "gamma",
        "denominator",
        "previous_norm",
        "state_norm",
        "candidate_norm",
        "relative_change",
        "candidate_previous_cosine",
    ):
        values[key] = np.ones((8, 2, 6))
    permutations = np.broadcast_to(np.arange(2), (1, 1, 2)).copy()
    result = diag.summarize(
        values,
        rows,
        {"entity_offset": 5, "entities_per_graph": 2, "relations": 1},
        {"permutations": permutations},
    )
    assert result["answer_accuracy"][-1] == 0.5
    assert result["ever_correct_through_r8"] == 1
    assert result["correct_before_r8_but_wrong_at_r8"] == 0.5
    assert result["candidate_token_prefix_accuracy"][-1] == [1, 1, 1, 1]
    assert result["aggregate_token_prefix_accuracy"][-1] == [1, 0.5, 1, 1]
    assert result["prefix_lost_at_r8"] == [0, 0.5, 0, 0]


def test_raw_control_uses_original_parameters_instead_of_ema():
    current = model()
    state = {k: v.clone() for k, v in current.state_dict().items()}
    snapshots = {1: {"model": state, "ema": {k: torch.zeros_like(v) for k, v in state.items()}}}
    diag.install(current, snapshots, 1, 1, "raw")
    for key, value in current.state_dict().items():
        assert torch.equal(value, state[key])
