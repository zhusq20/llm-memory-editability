"""Minimal-model semantics, error alignment, and independent trajectory resets."""

import json

import numpy as np
import torch

from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_path_minimal import (
    gradients,
    probe_data,
    toy_data,
    trajectory,
)


def test_same_logit_common_error_and_competition():
    ea, eb = np.eye(3)[:2]
    for p in (np.array([0.01, 0.01, 0.98]), np.array([0.5, 0.5, 0.0])):
        actual = np.dot(p - ea, p - eb)
        expected = np.dot(p, p) - p[0] - p[1]
        np.testing.assert_allclose(actual, expected)
    assert np.dot(np.array([0.01, 0.01, 0.98]) - ea, np.array([0.01, 0.01, 0.98]) - eb) > 0
    assert np.dot(np.array([0.5, 0.5, 0.0]) - ea, np.array([0.5, 0.5, 0.0]) - eb) < 0
    # b-directed logit SGD can improve CE(a) while worsening a-versus-b margin.
    logits = torch.tensor([-4.0, -4.0, 0.0], dtype=torch.float64, requires_grad=True)
    old = -logits.log_softmax(-1)[0]
    loss_b = -logits.log_softmax(-1)[1]
    changed = logits - 0.001 * torch.autograd.grad(loss_b, logits)[0]
    assert -changed.log_softmax(-1)[0] < old
    assert changed[0] - changed[1] < logits[0] - logits[1]


def test_minimal_truth_and_update_pairs():
    config = {"people": 32}
    tokens, old, start = toy_data(config, "cpu")
    assert tokens.shape == (96, 4)
    for person in (0, 1, 2):
        _, coherent, abc = probe_data(config, person, "coherent", "cpu")
        _, conflict, _ = probe_data(config, person, "conflict", "cpu")
        assert len(set(abc)) == 3
        assert coherent[0] == coherent[1] == abc[0]
        assert conflict[0] == abc[1] and conflict[1] == abc[0]
        assert conflict[2] == old[person * 3 + 2]
        assert old[person * 3] == old[person * 3 + 1] == start + person % 3


def test_minimal_counterfactuals_preserve_forward_and_update_only_target(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = {
        "people": 4,
        "target_parameter": "blocks.0.mlp.down.weight",
        "edit_steps": 2,
        "edit_lr": 0.01,
        "measure_every": 1,
    }
    model = CausalLM(ModelConfig(13, width=8, layers=2, heads=2, context=8))
    parent = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == config["target_parameter"])
    tokens, labels, _ = probe_data(config, 0, "conflict", "cpu")
    reference = gradients(model, tokens, labels, "full")
    for arm in ("full", "no_cross", "no_mlp", "fixed_qk"):
        result = gradients(model, tokens, labels, arm)
        assert torch.equal(result["logits"], reference["logits"])
    for arm in ("full", "no_cross"):
        trajectory(model, parent, config, 0, "conflict", arm, "cpu", tmp_path / arm)
        record = json.loads((tmp_path / arm / "trajectory.json").read_text())
        assert len(record["timeline"]) == 3
        assert record["timeline"][0]["loss"] == reference["loss"].tolist()
        for step in record["steps"]:
            np.testing.assert_allclose(
                step["direction_norm"], step["source_gradient_norm"], rtol=1e-6
            )
        for name, value in model.state_dict().items():
            if name != config["target_parameter"]:
                assert torch.equal(value, parent[name])
