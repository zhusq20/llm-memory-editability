"""Controlled recipe interventions, exact baseline replay and bounded training."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT
from llm_memory_editability.loop_recipe_ablation import recipe_loss, split_boundary
from llm_memory_editability.loop_terminal_training import sample_depths, terminal_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_loop_recipe_ablation as runner  # noqa: E402

TOKENS = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5], [5, 2, 6, 8, 3], [4, 8, 1, 5, 2]])
POSITIONS = torch.tensor([[1, 3, 4]] * 4)
LABELS = torch.tensor([[7, -100, -100], [5, 4, 2], [8, 3, -100], [-100, -100, -100]])


@pytest.fixture(autouse=True)
def cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return json.loads((ROOT / "configs/loop-recipe-ablation-20261025.json").read_text())


def model(dropout=0):
    torch.manual_seed(71)
    hf = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
    )
    result = LoopNativeHandoffGPT(
        hf, 8, handoff="invariant", first_round_input="embedding"
    ).double()
    with torch.no_grad():
        result.handoff_score_out.add_(0.1)
    return result.train() if dropout else result.eval()


def assert_gradients_equal(a, b, *, exact=False):
    for name, parameter in a.named_parameters():
        other = b.get_parameter(name)
        assert parameter.grad is not None and other.grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            other.grad,
            rtol=0 if exact else 1e-7,
            atol=0 if exact else 1e-9,
            msg=name,
        )


def reference(model, depths, boundary, supervision, truncate):
    total = 0
    for row, depth in enumerate(depths):
        x, p, y = TOKENS[row : row + 1], POSITIONS[row : row + 1], LABELS[row]
        if depth <= boundary:
            logits = model.run(x, p, int(depth), carry=False)[0]
            total += F.cross_entropy(logits.flatten(0, 1), y, reduction="sum")
            continue
        first, state = model.run(x, p, boundary)
        if truncate:
            state = tuple(value.detach() for value in state[:4]) + (state[4],)
        last, _ = model.run(x, p, int(depth) - boundary, state, carry=False)
        end = F.cross_entropy(last.flatten(0, 1), y, reduction="sum")
        if supervision:
            end = 0.5 * (end + F.cross_entropy(first.flatten(0, 1), y, reduction="sum"))
        total += end
    return total / LABELS.ne(-100).sum()


@pytest.mark.parametrize("depths", [[1, 4, 8, 16], [1, 1, 1, 1], [16, 16, 16, 16]])
def test_disabled_switches_exact_baseline_loss_gradient_and_dropout_rng(depths):
    actual = model(dropout=0.2)
    expected = copy.deepcopy(actual)
    rng = torch.get_rng_state()
    loss = terminal_loss(expected, TOKENS, POSITIONS, LABELS, depths)
    loss.backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    observed = recipe_loss(actual, TOKENS, POSITIONS, LABELS, depths, 5)
    observed.backward()
    torch.testing.assert_close(observed, loss, rtol=0, atol=0)
    assert_gradients_equal(actual, expected, exact=True)
    assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize("supervision,truncate", [(True, False), (False, True), (True, True)])
def test_mixed_depth_and_masked_target_gradients_match_independent_segments(supervision, truncate):
    actual, expected = model(), model()
    depths = [1, 5, 8, 16]
    observed = recipe_loss(
        actual,
        TOKENS,
        POSITIONS,
        LABELS,
        depths,
        5,
        supervision=supervision,
        truncate=truncate,
    )
    loss = reference(expected, depths, 5, supervision, truncate)
    observed.backward()
    loss.backward()
    torch.testing.assert_close(observed, loss, rtol=1e-9, atol=1e-11)
    assert_gradients_equal(actual, expected)


@pytest.mark.parametrize("supervision,truncate", [(True, False), (False, True), (True, True)])
def test_endpoints_at_or_before_boundary_are_noop(supervision, truncate):
    actual, expected = model(), model()
    depths = [1, 2, 4, 5]
    loss = terminal_loss(expected, TOKENS, POSITIONS, LABELS, depths)
    observed = recipe_loss(
        actual,
        TOKENS,
        POSITIONS,
        LABELS,
        depths,
        5,
        supervision=supervision,
        truncate=truncate,
    )
    loss.backward()
    observed.backward()
    torch.testing.assert_close(observed, loss, rtol=0, atol=0)
    assert_gradients_equal(actual, expected, exact=True)


def test_truncation_blocks_carry_but_keeps_original_injection_gradient(monkeypatch):
    actual = model()
    saved_states, injections = [], []
    original_round, original_embed = actual._trajectory_round, actual._embedded

    def embed(tokens):
        value = original_embed(tokens)
        value.retain_grad()
        injections.append(value)
        return value

    def step(*args, **kwargs):
        state, gamma = original_round(*args, **kwargs)
        for value in state[:4]:
            value.retain_grad()
        saved_states.append(state)
        return state, gamma

    monkeypatch.setattr(actual, "_embedded", embed)
    monkeypatch.setattr(actual, "_trajectory_round", step)
    loss = recipe_loss(actual, TOKENS, POSITIONS, LABELS, [8] * 4, 4, truncate=True)
    loss.backward()
    assert len(injections) == 1
    assert injections[0].grad is not None and injections[0].grad.abs().sum() > 0
    for value in saved_states[3][:4]:
        assert value.grad is None or torch.count_nonzero(value.grad) == 0
    assert saved_states[4][0].grad.abs().sum() > 0


def test_segment_loss_restores_first_segment_gradient_under_truncation(monkeypatch):
    actual = model()
    states = []
    original = actual._trajectory_round

    def step(*args, **kwargs):
        state, gamma = original(*args, **kwargs)
        state[0].retain_grad()
        states.append(state)
        return state, gamma

    monkeypatch.setattr(actual, "_trajectory_round", step)
    recipe_loss(
        actual, TOKENS, POSITIONS, LABELS, [8] * 4, 4, supervision=True, truncate=True
    ).backward()
    assert states[0][0].grad.abs().sum() > 0
    assert states[3][0].grad.abs().sum() > 0


def test_boundary_is_uniform_and_reconstructs_without_consuming_depth_rng():
    rng = np.random.default_rng(123)
    before = copy.deepcopy(rng.bit_generator.state)
    draws = [split_boundary(2026102501, step) for step in range(1, 12001)]
    assert before == rng.bit_generator.state
    np.testing.assert_allclose(np.bincount(draws)[4:7] / len(draws), [1 / 3] * 3, atol=0.02)
    checkpoint = {"step": 12000, "split_seed": 2026102501}
    assert split_boundary(checkpoint["split_seed"], checkpoint["step"] + 1) == split_boundary(
        2026102501, 12001
    )
    assert set(draws) == {4, 5, 6}


@pytest.mark.parametrize("arm", list(runner.ARMS))
def test_runner_depth_stream_is_unchanged_and_floor_only_affects_shallow(arm):
    actual = model()
    actual.recipe_flags = runner.ARMS[arm]
    actual.recipe_step = 100
    training = config()["training"]
    rng, reference_rng = np.random.default_rng(9), np.random.default_rng(9)
    original = sample_depths(reference_rng, len(TOKENS))
    _, parts, depths = runner.backward_batch(
        actual, TOKENS, POSITIONS, LABELS, training, rng, torch.device("cpu")
    )
    np.testing.assert_array_equal(depths, np.maximum(original, runner.ARMS[arm]["floor_depth"]))
    assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert parts == [] and actual.recipe_step == 101
    assert actual.recipe_boundary == split_boundary(training["split_sampler"]["seed"], 101)


def test_floor4_changes_exactly_the_original_shallow_draws():
    class FixedDraws:
        def poisson(self, mean, size):
            assert mean == 8 and size == 4
            return np.array([1, 2, 3, 8])

    actual = model()
    actual.recipe_flags = runner.ARMS["floor4"]
    actual.recipe_step = 0
    _, _, depths = runner.backward_batch(
        actual,
        TOKENS,
        POSITIONS,
        LABELS,
        config()["training"],
        FixedDraws(),
        torch.device("cpu"),
    )
    np.testing.assert_array_equal(depths, [4, 4, 4, 8])
    assert actual.recipe_changed_rows == 3


def test_model_initialization_and_state_dict_equal_reused_baseline():
    c = config()
    c["model"].update(n_embd=12, n_head=3, n_positions=12, n_ctx=12)
    metadata = {"vocab_size": 23, "pad_token": 0, "end_marker": 1}
    all_models = [runner.initialize(c, arm, metadata, torch.device("cpu")) for arm in runner.ARMS]
    view = copy.deepcopy(c)
    view["arms"] = {"invariant": {"handoff": "invariant"}}
    baseline = runner.fourway.initialize(view, "invariant", metadata, torch.device("cpu"))
    assert {runner.native.model_digest(x) for x in all_models} == {
        runner.native.model_digest(baseline)
    }


def test_scientific_budget_ends_at_256k_but_lr_uses_1500k():
    c = config()
    runner.validate_config(c)
    t = c["training"]
    assert t["steps"] == c["evaluation_nodes"][-1] == 256000
    assert c["budget"]["new_scientific_updates_authorized"] == 4 * 256000
    for step in [1, 2000, 128000, 256000]:
        assert runner.learning_rate(t, step) == runner._LEARNING_RATE({**t, "steps": 1500000}, step)
    assert runner.learning_rate(t, 256000) > 8e-5


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "steps", 1500000),
        ("training", "lr_schedule_steps", 256000),
        ("training", "stream_seed", 7),
        ("handoff", "first_round_input", "state_plus_embedding"),
        ("model", "embd_pdrop", 0.0),
        ("data", "world_seed", 7),
    ],
)
def test_frozen_recipe_guards_reject_unplanned_changes(section, key, value):
    c = config()
    c[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_config(c)


def test_scoped_patches_restore_and_clip_fraction_counts_actual_steps(tmp_path):
    before_lr = runner.native.learning_rate
    before_clip = torch.nn.utils.clip_grad_norm_
    with runner.implementation():
        p = torch.nn.Parameter(torch.tensor([1.0]))
        p.grad = torch.tensor([2.0])
        torch.nn.utils.clip_grad_norm_([p], 1.0)
        p.grad = torch.tensor([0.25])
        torch.nn.utils.clip_grad_norm_([p], 1.0)
        path = tmp_path / "status.json"
        runner.native.write_json(path, {"step": 2})
        saved = json.loads(path.read_text())
        assert saved["gradient_clip_fraction"] == 0.5
        assert saved["gradient_clip_count"] == 1
        assert saved["ablation_optimizer_steps"] == 2
    assert runner.native.learning_rate is before_lr
    assert torch.nn.utils.clip_grad_norm_ is before_clip
    assert runner.telemetry() == {}
