"""Terminal token weighting, per-sequence depth, and uninterrupted carry gradients."""

import copy
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT, rms_normalize
from llm_memory_editability.loop_terminal_training import (
    backward_terminal_loss,
    run_terminal_batch,
    sample_depths,
    terminal_loss_groups,
)
from llm_memory_editability.self_decode_alignment import _readout

ROOT = Path(__file__).resolve().parents[1]
TOKENS = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5], [5, 2, 6, 8, 3], [4, 8, 1, 5, 2]])
POSITIONS = torch.tensor([[1, 3, 4]] * 4)
# Unequal target counts, including more than one answer token in a sequence.
LABELS = torch.tensor([[7, -100, -100], [5, 4, 2], [8, 3, -100], [2, -100, -100]])


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def build(embedding_dropout=0):
    torch.manual_seed(71)
    config = GPT2Config(
        vocab_size=23,
        n_embd=12,
        n_head=3,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        resid_pdrop=0,
        embd_pdrop=embedding_dropout,
        attn_pdrop=0,
    )
    model = (
        LoopNativeHandoffGPT(config, 8, handoff="invariant", first_round_input="embedding")
        .double()
        .eval()
    )
    with torch.no_grad():
        model.handoff_score_out.add_(0.1)  # Exercise feedback through the score features, too.
    return model


def runner():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("run_loop_native_handoff")


def new_config():
    return json.loads((ROOT / "configs/loop-invariant-poisson-terminal-v1.json").read_text())


def reference_loss(model, depths, labels=LABELS):
    total = 0
    for row, depth in enumerate(depths):
        logits = model.run(
            TOKENS[row : row + 1], POSITIONS[row : row + 1], int(depth), carry=False
        )[0]
        total = total + F.cross_entropy(
            logits.flatten(0, 1), labels[row].flatten(), reduction="sum"
        )
    return total / labels.ne(-100).sum()


def assert_gradients_match(actual, expected):
    for name, parameter in actual.named_parameters():
        other = expected.get_parameter(name)
        assert parameter.grad is not None and other.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-7, atol=1e-9, msg=name)


def test_conditional_poisson_distribution_and_rng_restore():
    rng = np.random.default_rng(42)
    values = sample_depths(rng, 100000)
    weights = np.array([8**t / math.factorial(t) for t in range(1, 17)])
    probabilities = weights / weights.sum()
    assert set(values) == set(range(1, 17))
    np.testing.assert_allclose(
        np.bincount(values, minlength=17)[1:] / len(values), probabilities, atol=0.003, rtol=0
    )
    assert values.mean() == pytest.approx(7.96644, abs=0.03)
    state = copy.deepcopy(rng.bit_generator.state)
    expected = sample_depths(rng, 512)
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    np.testing.assert_array_equal(sample_depths(restored, 512), expected)


def test_poisson_rejects_outliers_instead_of_clamping_or_shifting():
    class Draws:
        calls = 0

        def poisson(self, mean, size):
            self.calls += 1
            assert mean == 8
            if self.calls == 1:
                assert size == 4
                return np.array([0, 1, 17, 16])
            assert size == 2
            return np.array([3, 8])

    np.testing.assert_array_equal(sample_depths(Draws(), 4), [3, 1, 8, 16])


@pytest.mark.parametrize(
    "options",
    [{"mean": 0}, {"mean": float("nan")}, {"minimum": 0}, {"maximum": 0}, {"minimum": 1.5}],
)
def test_invalid_sampling_rejected(options):
    with pytest.raises(ValueError):
        sample_depths(np.random.default_rng(0), 4, **options)


def test_first_block_input_is_e_and_first_merge_keeps_the_original_prior():
    model = build()
    e = model._embedded(TOKENS)
    H = model._round(e)
    p, q = model._decoded_feedback(H)
    c = rms_normalize(H + q, model.eps)
    p0 = model._decoded_feedback(e)[0]
    weight = model.score(c, e, p, p0).clamp(-8, 8).exp().unsqueeze(-1)
    expected_h = (model.beta * e + weight * c) / (model.beta + weight)
    logits, state = model.run(TOKENS, POSITIONS, 1)
    torch.testing.assert_close(state[0], expected_h)
    torch.testing.assert_close(state[3], p)
    torch.testing.assert_close(state[4], e)
    torch.testing.assert_close(logits, _readout(model, expected_h, POSITIONS))
    assert not torch.allclose(logits, _readout(model, H, POSITIONS))


def test_continuation_keeps_one_embedding_dropout_draw_and_only_one_first_round(monkeypatch):
    model = build(embedding_dropout=0.4).train()
    rng = torch.get_rng_state()
    expected = model.run(TOKENS, POSITIONS, 8, carry=False)[0]
    torch.set_rng_state(rng)
    original = model._embedded
    calls = []

    def embed(tokens):
        calls.append(1)
        return original(tokens)

    monkeypatch.setattr(model, "_embedded", embed)
    _, state = model.run(TOKENS, POSITIONS, 3)
    actual, _ = model.run(TOKENS, POSITIONS, 5, state, carry=False)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_trace_forward_and_streaming_agree_at_all_depths():
    model = build()
    traced, _, _ = model.trace(TOKENS, POSITIONS, 16)
    state = None
    for depth in range(1, 17):
        streamed, state = model.run(TOKENS, POSITIONS, 1, state)
        torch.testing.assert_close(traced[depth - 1], streamed, rtol=0, atol=0)
    for depth in (1, 8, 16):
        torch.testing.assert_close(traced[depth - 1], model.run(TOKENS, POSITIONS, depth)[0])


def test_historical_two_argument_diagnostic_hooks_remain_usable(monkeypatch):
    model = build()
    model.first_round_input = "state_plus_embedding"
    expected = model.run(TOKENS, POSITIONS, 3)[0]
    original = model.aggregate_round

    def historical_hook(state, injection):
        return original(state, injection)

    monkeypatch.setattr(model, "aggregate_round", historical_hook)
    observed = model.trace(TOKENS, POSITIONS, 3)[0][-1]
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_masked_terminal_gradient_is_global_token_mean_with_unequal_masks():
    actual, expected = build(), build()
    depths = np.array([1, 4, 4, 16])
    original = {name: parameter.detach().clone() for name, parameter in actual.named_parameters()}
    observed = backward_terminal_loss(actual, TOKENS, POSITIONS, LABELS, depths)
    loss = reference_loss(expected, depths)
    loss.backward()
    torch.testing.assert_close(observed, loss)
    assert_gradients_match(actual, expected)
    for name, parameter in actual.named_parameters():
        assert torch.equal(parameter, original[name])  # Helper never performs optimizer updates.
    # Equal weight per sample/group is a different objective for these masks.
    with torch.no_grad():
        per_sample = [
            F.cross_entropy(
                actual.run(TOKENS[i : i + 1], POSITIONS[i : i + 1], int(t))[0].flatten(0, 1),
                LABELS[i],
            )
            for i, t in enumerate(depths)
        ]
    assert not torch.isclose(observed, torch.stack(per_sample).mean(), atol=1e-7, rtol=1e-7)


def test_zero_target_group_is_finite_and_an_empty_target_batch_is_rejected():
    actual, expected = build(), build()
    labels = LABELS.clone()
    labels[0] = -100
    depths = [1, 4, 4, 8]
    observed = backward_terminal_loss(actual, TOKENS, POSITIONS, labels, depths)
    loss = reference_loss(expected, depths, labels)
    loss.backward()
    torch.testing.assert_close(observed, loss)
    assert_gradients_match(actual, expected)
    with pytest.raises(ValueError, match="at least one supervised token"):
        list(terminal_loss_groups(actual, TOKENS, POSITIONS, labels.fill_(-100), depths))
    with pytest.raises(ValueError, match="at least one supervised token"):
        backward_terminal_loss(actual, TOKENS, POSITIONS, labels, depths)


@pytest.mark.parametrize("depths", [[1, 4, 4, 16], [1] * 4, [16] * 4, [2, 3, 5, 8]])
def test_masked_matches_original_grouped_loss_and_every_parameter_gradient(depths):
    actual, expected = build(), build()
    observed = backward_terminal_loss(actual, TOKENS, POSITIONS, LABELS, depths)
    reference = torch.zeros_like(observed)
    for loss in terminal_loss_groups(expected, TOKENS, POSITIONS, LABELS, depths):
        loss.backward()
        reference += loss.detach()
    torch.testing.assert_close(observed, reference, rtol=1e-9, atol=1e-11)
    assert_gradients_match(actual, expected)


def test_full_batch_carry_matches_each_samples_endpoint_and_stays_frozen(monkeypatch):
    model = build()
    depths = [1, 4, 4, 16]
    expected = [
        model.run(TOKENS[i : i + 1], POSITIONS[i : i + 1], depth) for i, depth in enumerate(depths)
    ]
    original = model.aggregate_round
    inputs = []

    def capture(state, injection, *, first_round):
        inputs.append((state, first_round))
        return original(state, injection, first_round=first_round)

    monkeypatch.setattr(model, "aggregate_round", capture)
    logits, carry = run_terminal_batch(model, TOKENS, POSITIONS, depths)
    assert len(inputs) == max(depths)
    assert [first for _, first in inputs] == [True] + [False] * 15
    assert all(len(state[0]) == len(TOKENS) for state, _ in inputs)
    for row, (reference_logits, reference_carry) in enumerate(expected):
        torch.testing.assert_close(logits[row : row + 1], reference_logits)
        for part, reference in zip(carry, reference_carry, strict=True):
            torch.testing.assert_close(part[row : row + 1], reference)
        for state, _ in inputs[depths[row] :]:
            for part, reference in zip(state, reference_carry, strict=True):
                torch.testing.assert_close(part[row : row + 1], reference)


def test_uniform_depth_preserves_dropout_draws_and_uses_one_embedding(monkeypatch):
    actual = build(embedding_dropout=0.4).train()
    actual.config.attn_pdrop = actual.config.resid_pdrop = 0.2
    for module in actual.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.2
    expected = copy.deepcopy(actual)
    rng = torch.get_rng_state()
    logits = expected.run(TOKENS, POSITIONS, 8, carry=False)[0]
    reference = F.cross_entropy(logits.flatten(0, 1), LABELS.flatten())
    reference.backward()
    torch.set_rng_state(rng)
    original = actual._embedded
    calls = []

    def embed(tokens):
        calls.append(1)
        return original(tokens)

    monkeypatch.setattr(actual, "_embedded", embed)
    observed = backward_terminal_loss(actual, TOKENS, POSITIONS, LABELS, [8] * 4)
    assert calls == [1]
    torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert_gradients_match(actual, expected)


@pytest.mark.parametrize("depths", [[1, 2], [1, 0, 2, 3], [1, -1, 2, 3], [1.0] * 4])
def test_masked_rejects_invalid_per_sample_depths(depths):
    with pytest.raises(ValueError, match="one positive integer depth"):
        run_terminal_batch(build(), TOKENS, POSITIONS, depths)


def test_terminal_loss_reaches_first_round_history_and_probability(monkeypatch):
    model = build()
    original = model.aggregate_round
    states = []

    def capture(*args, **kwargs):
        state, gamma = original(*args, **kwargs)
        for value in state[:4]:
            value.retain_grad()
        states.append(state)
        return state, gamma

    monkeypatch.setattr(model, "aggregate_round", capture)
    backward_terminal_loss(model, TOKENS, POSITIONS, LABELS, [8] * 4)
    assert len(states) == 8
    for value in states[0][:4]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


def test_runner_uses_sampled_depths_and_preserves_rng_for_replay():
    module = runner()
    actual, expected = build(), build()
    training = new_config()["training"]
    rng = np.random.default_rng(123)
    state = copy.deepcopy(rng.bit_generator.state)
    loss, parts, depths = module.backward_batch(
        actual, TOKENS, POSITIONS, LABELS, training, rng, torch.device("cpu")
    )
    assert parts == [] and len(depths) == len(TOKENS)
    reference = reference_loss(expected, depths)
    reference.backward()
    torch.testing.assert_close(loss, reference)
    assert_gradients_match(actual, expected)
    rng.bit_generator.state = state
    np.testing.assert_array_equal(sample_depths(rng, len(TOKENS)), depths)


def test_new_recipe_validates_without_changing_legacy_config():
    module = runner()
    config = new_config()
    module.validate_config(config)
    legacy = json.loads((ROOT / "configs/loop-native-handoff-development-v1.json").read_text())
    module.validate_config(legacy)
    assert legacy["training"]["segments"] == 2
    assert "first_round_input" not in legacy["handoff"]
    for section, key, value in [
        ("training", "unique_layers", 2),
        ("training", "backpropagation", "last_four"),
        ("training", "execution", "grouped"),
        ("handoff", "first_round_input", "state_plus_embedding"),
    ]:
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError):
            module.validate_config(changed)


def test_new_config_reconstructs_the_same_forward_after_reload():
    module = runner()
    config = new_config()
    config["model"].update(
        n_embd=12, n_head=3, n_positions=12, n_ctx=12, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
    )
    metadata = {"vocab_size": 23, "pad_token": 0, "end_marker": 1}
    model = module.initialize(config, "invariant", metadata, torch.device("cpu")).eval()
    restored = module.initialize(config, "invariant", metadata, torch.device("cpu")).eval()
    restored.load_state_dict(model.state_dict())
    assert len(restored.transformer.h) == 1 and restored.first_round_input == "embedding"
    torch.testing.assert_close(
        model(TOKENS, POSITIONS), restored(TOKENS, POSITIONS), rtol=0, atol=0
    )


def test_compute_counts_expired_rows_separately_from_sampled_depths():
    module = runner()
    model = build()
    metadata = {"vocab_size": 23, "sequence_length": 5}
    depths = [1, 8, 8, 16]
    together = module.terminal_step_flops(model, metadata, depths, 3)
    separate = sum(module.terminal_step_flops(model, metadata, [depth], 3) for depth in depths)
    assert (
        module.terminal_step_flops(model, metadata, depths, 3, include_inactive=False) == separate
    )
    assert together > separate
    assert together == module.terminal_step_flops(model, metadata, [16] * 4, 3)
    assert together - module.terminal_step_flops(model, metadata, depths, 2) == 3 * 2 * 4 * 12 * 23
