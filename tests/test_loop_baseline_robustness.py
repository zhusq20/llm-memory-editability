"""Ordinary Loop must remain the ungated shared transformer, with no feedback."""

import json
from pathlib import Path

import pytest
import torch
from test_loop_native_handoff import POSITIONS, TOKENS, build, config, runner

from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT


def test_plain_matches_existing_loop_values_all_gradients_and_no_extra_parameters(monkeypatch):
    current = build("plain", repeats=4)
    torch.manual_seed(71)
    reference = SupervisedAlignmentGPT(config(), 4, discoloop=False).double().eval()

    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary Loop must never decode and re-encode between rounds")

    monkeypatch.setattr(current, "_discoloop_boundary", forbidden)
    monkeypatch.setattr(current, "_decoded_feedback", forbidden)
    assert list(current.state_dict()) == list(reference.state_dict())
    actual, expected = current(TOKENS, POSITIONS), reference(TOKENS, POSITIONS)
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter.grad, current.get_parameter(name).grad), name


def test_plain_compute_excludes_every_feedback_channel():
    plain, disco = build("plain"), build("additive")
    metadata = {"vocab_size": 23, "sequence_length": 5}
    channel = 3 * 4 * 2 * 5 * 12 * 23
    assert (
        runner().step_flops(disco, metadata, 2, [2, 3])
        - runner().step_flops(plain, metadata, 2, [2, 3])
        == 4 * channel
    )


@pytest.mark.parametrize("condition", ("init-1", "init-2", "new-world", "many-to-one"))
def test_new_baselines_pair_exactly_with_invariant_data_and_training(condition):
    root = Path(__file__).resolve().parents[1]
    original = json.loads(
        (root / f"configs/loop-invariant-robustness-v1/{condition}.json").read_text()
    )
    new = json.loads((root / f"configs/loop-baseline-robustness-v1/{condition}.json").read_text())
    for key in ("training", "model", "data", "input_root", "input_origin", "dropout_seed"):
        assert new[key] == original[key], key
    assert (
        new["paired_recipe"]["initial_backbone_sha256"]
        == original["paired_recipe"]["initial_backbone_sha256"]
    )
    assert new["arms"] == {"loop": {"handoff": "plain"}, "discoloop": {"handoff": "additive"}}


def test_budget_reuses_original_discoloop_and_adds_only_missing_runs():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "configs/loop-baseline-robustness-v1.json").read_text())
    new = {k for k, v in plan["jobs"].items() if v["kind"] == "new"}
    assert new == {"original-loop"} | {
        f"{condition}-{arm}"
        for condition in ("init-1", "init-2", "new-world", "many-to-one")
        for arm in ("loop", "discoloop")
    }
    assert plan["budget"]["new_updates"] == 9 * 1500000


def test_controller_passes_selected_baseline_plan_to_each_container(tmp_path):
    from launch_loop_robustness import Controller

    root = Path(__file__).resolve().parents[1]
    path = root / "configs/loop-baseline-robustness-v1.json"
    controller = Controller(tmp_path, tmp_path / "unused-credential", path)
    label, command = controller.command("original-loop", "smoke", "GPU-test", 0)
    assert label == "lbr-v1-original-loop-smoke"
    assert command[command.index("--plan") + 1] == "configs/loop-baseline-robustness-v1.json"
