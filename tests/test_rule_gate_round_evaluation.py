import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.round_evaluation import (
    correctness_summary,
    inference_matmul_flops,
    paired_summary,
    round_logits,
)
from llm_memory_editability.rule_gate import RuleGateGPT
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_rule_gate_rounds import prerequisite_state  # noqa: E402


def test_correctness_keeps_transient_and_never_correct_cases():
    values = np.array([[1, 0, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1]], bool)
    s = correctness_summary(values, reference_round=2)
    assert s["reference_correct_n"] == 2
    assert s["gained_vs_reference_n"] == [1, 0, 1, 1]
    assert s["lost_vs_reference_n"] == [2, 0, 1, 1]
    assert s["retention_vs_reference"] == [0, 1, 0.5, 0.5]
    assert s["first_correct_round_histogram"] == {"0": 1, "1": 1, "2": 2, "3": 0, "4": 0}
    assert s["first_correct_through_final_round_histogram"] == {
        "0": 2,
        "1": 0,
        "2": 1,
        "3": 1,
        "4": 0,
    }
    assert s["right_to_wrong_n"] == [0, 1, 1, 0]


def test_empty_reference_is_undefined_and_paired_gains_keep_denominator():
    s = correctness_summary([[0, 1], [0, 0]], reference_round=1)
    assert s["retention_vs_reference"] == [None, None]
    got = paired_summary([[1, 0], [0, 1]], [[0, 0], [0, 1]])
    assert got["difference_pp"] == [50, 0]
    assert got["candidate_only_correct_n"] == [1, 0]
    assert got["both_correct_n"] == [0, 1]
    assert got["baseline_never_correct_n"] == 1
    assert got["correct_when_baseline_never_correct_n"] == [1, 0]
    with pytest.raises(ValueError):
        paired_summary([[1, 0]], [[1]])


@pytest.mark.parametrize("rule", [None, "stability", "readiness"])
def test_single_trajectory_equals_every_native_final_round(rule):
    torch.set_num_threads(2)
    torch.manual_seed(51)
    hf = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=16,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    model = (
        RuleGateGPT(hf, 4, rule=rule) if rule else SupervisedAlignmentGPT(hf, 4, discoloop=True)
    ).eval()
    tokens = torch.tensor([[3, 7, 9, 5, 1, 0], [8, 6, 11, 2, 4, 0]])
    positions = torch.tensor([[2, 3], [3, 4]])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        for r, actual in enumerate(round_logits(model, tokens, positions, 16), 1):
            model.repeats = r
            assert torch.equal(actual, model(tokens, positions))
            generated = tokens.clone()
            generated[torch.arange(len(tokens)), positions[:, 1]] = actual[:, 0].argmax(-1)
            # Generated future answer cannot alter answer-position causal logits.
            assert torch.equal(actual[:, 0], model(generated, positions)[:, 0])
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_cost_includes_both_forwards_and_rule_work_only_at_boundaries():
    costs = [
        inference_matmul_flops(12, 23, 6, 1, rule) for rule in [None, "stability", "readiness"]
    ]
    assert costs[0] == costs[1] == costs[2]
    base = inference_matmul_flops(12, 23, 6, 4)
    stability = inference_matmul_flops(12, 23, 6, 4, "stability")
    readiness = inference_matmul_flops(12, 23, 6, 4, "readiness")
    assert stability - base == 2 * 3 * (2 * 6 * 12 * 23)
    assert readiness > stability > base


def test_scheduler_requires_train_and_grid_success(tmp_path):
    assert prerequisite_state(tmp_path, "stability-u") == "waiting"
    for stage in ("train", "grid"):
        folder = tmp_path / "execution" / "stability-u" / stage
        folder.mkdir(parents=True)
        (folder / "exit.json").write_text(json.dumps({"code": 0}))
        assert prerequisite_state(tmp_path, "stability-u") == (
            "waiting" if stage == "train" else "ready"
        )
    (folder / "exit.json").write_text(json.dumps({"code": 1}))
    assert prerequisite_state(tmp_path, "stability-u") == "failed"


def test_protocol_has_no_new_training_or_best_horizon_selection():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/rule-gate-round-evaluation-v1.json").read_text())
    assert config["training_updates"] == 0
    assert config["checkpoint_step"] == 1500000
    assert config["rounds"] == list(range(1, 17))
    assert config["primary"]["round"] == 8
    assert config["tracking"]["entity"] == "zsqzz"
    assert {lane["gpu"] for lane in config["lanes"]} == {0, 1}
    assert sorted(a for lane in config["lanes"] for a in lane["arms"]) == sorted(config["arms"])
