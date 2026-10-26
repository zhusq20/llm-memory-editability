"""Raw-state handoff contracts without changing the existing aggregation recipe."""

import io

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import (
    HANDOFFS,
    SUPPORTED_HANDOFFS,
    LoopNativeHandoffGPT,
    deep_supervision_loss,
    detach_state,
    handoff_parameter_count,
    rms_normalize,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

TOKENS = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5]])
POSITIONS = torch.tensor([[3, 4], [3, 4]])
LABELS = torch.tensor([[7, 2], [5, -100]])


def config(dropout=0.0):
    return GPT2Config(
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


def build(handoff="state-mlp", *, score_seed=11, score_hidden=16, dropout=0.0):
    torch.manual_seed(71)
    return (
        LoopNativeHandoffGPT(
            config(dropout), 3, handoff=handoff, score_seed=score_seed, score_hidden=score_hidden
        )
        .double()
        .eval()
    )


def activate_score(model):
    with torch.no_grad():
        model.handoff_score_out.copy_(
            torch.linspace(-0.3, 0.4, model.handoff_score_out.numel(), dtype=torch.double)
        )
        model.handoff_score_out_bias.fill_(0.2)


def ce(model):
    return F.cross_entropy(model(TOKENS, POSITIONS).flatten(0, 1), LABELS.flatten())


def test_state_mlp_is_opt_in_and_does_not_expand_the_historical_matrix():
    assert HANDOFFS == ("additive", "ema", "content", "invariant", "plain")
    assert SUPPORTED_HANDOFFS == (*HANDOFFS, "state-mlp")


@pytest.mark.parametrize("hidden", (1, 16, 23))
def test_parameter_count_matches_the_actual_two_state_mlp(hidden):
    model = build(score_hidden=hidden)
    actual = sum(p.numel() for n, p in model.named_parameters() if n.startswith("handoff_"))
    assert actual == handoff_parameter_count("state-mlp", 12, hidden)
    assert actual == (2 * 12 + 2) * hidden + 2
    assert model.handoff_score_in.shape == (hidden, 24)
    assert handoff_parameter_count("state-mlp", 768, 16) == 24610


def test_backbone_and_global_rng_match_discoloop_for_any_score_seed():
    current = build(score_seed=11)
    current_rng = torch.get_rng_state().clone()
    torch.manual_seed(71)
    reference = SupervisedAlignmentGPT(config(), 3, discoloop=True).double().eval()
    assert torch.equal(current_rng, torch.get_rng_state())
    for name, value in reference.state_dict().items():
        assert torch.equal(current.state_dict()[name], value), name
    again = build(score_seed=12)
    assert torch.equal(current_rng, torch.get_rng_state())
    assert not torch.equal(current.handoff_score_in, again.handoff_score_in)
    for name, value in reference.state_dict().items():
        assert torch.equal(again.state_dict()[name], value), name


@pytest.mark.parametrize("dropout", (0.0, 0.2))
def test_zero_score_matches_ema_forward_all_shared_gradients_and_dropout_rng(dropout):
    current = build(dropout=dropout).train()
    ema = build("ema", dropout=dropout).train()
    initial_rng = torch.get_rng_state().clone()
    current_logits = current(TOKENS, POSITIONS)
    after_current = torch.get_rng_state().clone()
    current_loss = F.cross_entropy(current_logits.flatten(0, 1), LABELS.flatten())
    current_loss.backward()
    torch.set_rng_state(initial_rng)
    ema_logits = ema(TOKENS, POSITIONS)
    assert torch.equal(after_current, torch.get_rng_state())
    ema_loss = F.cross_entropy(ema_logits.flatten(0, 1), LABELS.flatten())
    ema_loss.backward()
    assert torch.equal(current_logits, ema_logits)
    assert torch.equal(current_loss, ema_loss)
    for name, parameter in ema.named_parameters():
        assert parameter.grad is not None, name
        assert torch.equal(current.get_parameter(name).grad, parameter.grad), name


def test_nonzero_score_reads_both_raw_states_and_ignores_decode_probabilities():
    model = build()
    # A deterministic score reads different coordinates from each half of [previous; candidate].
    with torch.no_grad():
        model.handoff_score_in.zero_()
        model.handoff_score_in_bias.zero_()
        model.handoff_score_out.zero_()
        model.handoff_score_out_bias.zero_()
        model.handoff_score_in[0, 0] = 0.5
        model.handoff_score_in[0, 12] = 0.25
        model.handoff_score_out[0] = 1.0
    previous = torch.ones(2, 5, 12, dtype=torch.double, requires_grad=True)
    candidate = torch.full((2, 5, 12), 2.0, dtype=torch.double, requires_grad=True)
    score = model.score(candidate, previous, None, None)
    expected = F.gelu(0.5 * previous[..., 0] + 0.25 * candidate[..., 0])
    assert torch.equal(score, expected)
    # Poisoned probabilities would propagate NaNs if they were inspected.
    poison = torch.full((2, 5, 23), float("nan"), dtype=torch.double)
    assert torch.equal(score, model.score(candidate, previous, poison, poison))
    changed_previous = model.score(candidate, previous + 1, None, None)
    changed_candidate = model.score(candidate + 1, previous, None, None)
    assert not torch.equal(score, changed_previous)
    assert not torch.equal(score, changed_candidate)
    score.sum().backward()
    assert previous.grad[..., 0].abs().min() > 0
    assert candidate.grad[..., 0].abs().min() > 0


def test_zero_output_first_layer_gradient_becomes_nonzero_after_one_optimizer_update():
    model = build().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    assert torch.count_nonzero(model.handoff_score_out) == 0
    assert model.handoff_score_out_bias == 0
    ce(model).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert torch.count_nonzero(model.handoff_score_in.grad) == 0
    assert torch.count_nonzero(model.handoff_score_in_bias.grad) == 0
    assert torch.count_nonzero(model.handoff_score_out.grad) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ce(model).backward()
    assert torch.isfinite(model.handoff_score_in.grad).all()
    assert torch.count_nonzero(model.handoff_score_in.grad) > 0
    assert torch.count_nonzero(model.handoff_score_in_bias.grad) > 0


def test_streaming_aggregation_uses_the_original_candidate_beta_and_normalizer():
    model = build()
    activate_score(model)
    injection = model._embedded(TOKENS)
    state = model.initial_state(injection)
    candidates = [injection]
    weights = [torch.ones_like(injection[..., 0])]
    beta = model.beta
    for _ in range(4):
        previous = state[0]
        output = model._round(previous + injection)
        probability, feedback = model._decoded_feedback(output)
        candidate = rms_normalize(output + feedback, model.eps)
        score = model.score(candidate, previous, probability, None)
        weights = [beta * value for value in weights]
        weights.append(score.clamp(-model.score_clip, model.score_clip).exp())
        candidates.append(candidate)
        numerator = sum(w.unsqueeze(-1) * c for w, c in zip(weights, candidates, strict=True))
        denominator = sum(weights).unsqueeze(-1)
        state, gamma = model.aggregate_round(state, injection)
        torch.testing.assert_close(state[1], numerator, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(state[2], denominator, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(state[0], numerator / denominator, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gamma, weights[-1].unsqueeze(-1) / denominator)
        assert state[3] is None


def test_segment_continuation_strict_reload_and_no_probability_carry():
    model = build()
    activate_score(model)
    whole = model.run(TOKENS, POSITIONS, rounds=5, carry=False)[0]
    _, carry = model.run(TOKENS, POSITIONS, rounds=2)
    assert len(carry) == 4 and carry[3] is None
    assert model.initial_state(model._embedded(TOKENS))[3] is None
    detached = detach_state(carry)
    assert all(value is None or not value.requires_grad for value in detached)
    checkpoint = io.BytesIO()
    torch.save({"model": model.state_dict(), "carry": detached}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    reloaded = build(score_seed=99)
    result = reloaded.load_state_dict(saved["model"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    continued = reloaded.run(TOKENS, POSITIONS, rounds=3, state=saved["carry"], carry=False)[0]
    assert torch.equal(whole, continued)
    assert torch.equal(model(TOKENS, POSITIONS), reloaded(TOKENS, POSITIONS))


def test_deep_supervision_keeps_only_answer_eos_ce_and_detaches_segments():
    model = build().train()
    activate_score(model)
    loss, parts = deep_supervision_loss(model, TOKENS, POSITIONS, LABELS, [2, 3])
    first, carry = model.run(TOKENS, POSITIONS, rounds=2)
    second = model.run(TOKENS, POSITIONS, rounds=3, state=detach_state(carry), carry=False)[0]
    expected_parts = [F.cross_entropy(x.flatten(0, 1), LABELS.flatten()) for x in (first, second)]
    assert torch.equal(loss, torch.stack(expected_parts).mean())
    for actual, expected in zip(parts, expected_parts, strict=True):
        assert torch.equal(actual, expected.detach())
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("score_bias", (-1e6, 1e6))
def test_clipped_extreme_scores_keep_gates_and_states_finite(score_bias):
    model = build()
    with torch.no_grad():
        model.handoff_score_out_bias.fill_(score_bias)
    logits, change, gamma = model.trace(TOKENS, POSITIONS, 6)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(change).all()
    assert torch.isfinite(gamma).all()
    assert ((gamma > 0) & (gamma <= 1)).all()
    for rounds in (1, 3, 6):
        assert torch.equal(logits[rounds - 1], model.run(TOKENS, POSITIONS, rounds, carry=False)[0])
