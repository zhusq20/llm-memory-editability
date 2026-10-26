"""Contracts for the rule-gate configuration and unchanged training-runner delegation."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_rule_gate as runner  # noqa: E402


def documents():
    config = json.loads((ROOT / "configs/rule-gate-training-v1.json").read_text())
    reference = json.loads((ROOT / config["paired_recipe"]["config"]).read_text())
    return config, reference


def test_exact_two_arm_recipe_and_reject_unpaired_changes():
    config, reference = documents()
    runner.validate_config(config, reference)
    for section, key, value in (
        ("training", "steps", 300000),
        ("training", "stream_seed", 71),
        ("model", "n_embd", 256),
        ("budget", "new_scientific_updates", 1500000),
    ):
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed, reference)
    for arm, key, value in (
        ("stability-u", "gate_rule", "readiness"),
        ("readiness-u", "tau", 0.05),
    ):
        changed = copy.deepcopy(config)
        changed["arms"][arm][key] = value
        with pytest.raises(ValueError):
            runner.validate_config(changed, reference)


def test_initialization_matches_fixed_q_plus_u_backbone_and_rng():
    config, _ = documents()
    config["model"].update(n_embd=12, n_head=3, n_positions=12, n_ctx=12)
    metadata = {"vocab_size": 23, "pad_token": 0, "end_marker": 2, "sequence_length": 6}
    outputs = []
    for name in config["arms"]:
        model, spec, arm, digest = runner.initialize(config, name, metadata, torch.device("cpu"))
        assert model.rule == config["arms"][name]["gate_rule"] and model.tau == 0.1
        assert arm["discoloop"] and not arm["first_round_kv"] and arm["kind"] == "none"
        assert not any(n.startswith("gate_") for n in model.state_dict())
        cost = runner.accounting(model, spec, arm, metadata)
        assert cost["parameters"] == sum(p.numel() for p in model.parameters())
        assert cost["update_gate_parameters"] == 0 and cost["channel_calls_per_forward"] == 3
        assert cost["rule_forward_only_matmul_flops_per_step"] > 0
        outputs.append((model.state_dict(), digest, torch.get_rng_state()))
    assert outputs[0][1] == outputs[1][1]
    assert torch.equal(outputs[0][2], outputs[1][2])
    assert all(torch.equal(v, outputs[1][0][n]) for n, v in outputs[0][0].items())


def test_adapter_restores_globals_even_after_failure_and_preserves_training_functions():
    replaced = ("initialize", "accounting", "optimizer_for", "preflight")
    unchanged = ("train", "AlignmentStep", "MergedBatchStream", "measure_panels")
    before = {name: getattr(runner.native, name) for name in (*replaced, *unchanged)}
    with pytest.raises(RuntimeError, match="fixture"):
        with runner.installed():
            assert runner.native.initialize is runner.initialize
            assert runner.native.optimizer_for is runner.original.optimizer_for
            assert all(getattr(runner.native, n) is before[n] for n in unchanged)
            raise RuntimeError("fixture")
    assert all(getattr(runner.native, name) is value for name, value in before.items())
