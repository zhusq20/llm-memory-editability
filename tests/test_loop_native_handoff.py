"""Contracts for the loop-native handoff: backbone pairing, exact DiscoLoop reuse,
streaming recurrence, segment continuation and relabeling invariance of the score."""

import math

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.loop_native_handoff import (
    FEATURES,
    HANDOFFS,
    LoopNativeHandoffGPT,
    deep_supervision_loss,
    detach_state,
    handoff_parameter_count,
    rms_normalize,
    sample_schedule,
)
from llm_memory_editability.self_decode_alignment import _readout
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT

TOKENS = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5]])
POSITIONS = torch.tensor([[3, 4], [3, 4]])


def config():
    return GPT2Config(
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


def build(handoff, repeats=3):
    torch.manual_seed(71)
    return LoopNativeHandoffGPT(config(), repeats, handoff=handoff).double().eval()


def perturb_handoff(model, seed=5):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("handoff_"):
                parameter.add_(
                    torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                )


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_backbone_matches_discoloop_and_consumes_no_global_rng(handoff):
    current = build(handoff)
    after = torch.get_rng_state().clone()
    torch.manual_seed(71)
    reference = SupervisedAlignmentGPT(config(), 3, discoloop=True).double()
    assert torch.equal(after, torch.get_rng_state())
    names = [n for n in current.state_dict() if not n.startswith("handoff_")]
    assert names == list(reference.state_dict())
    assert all(torch.equal(current.state_dict()[n], reference.state_dict()[n]) for n in names)
    extra = sum(p.numel() for n, p in current.named_parameters() if n.startswith("handoff_"))
    assert extra == handoff_parameter_count(handoff, 12)


def test_additive_handoff_is_the_existing_discoloop_forward_and_gradient():
    current = build("additive", repeats=4)
    torch.manual_seed(71)
    reference = SupervisedAlignmentGPT(config(), 4, discoloop=True).double().eval()
    a, b = reference(TOKENS, POSITIONS), current(TOKENS, POSITIONS)
    assert torch.equal(a, b)
    a.square().mean().backward()
    b.square().mean().backward()
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter.grad, current.get_parameter(name).grad), name


@pytest.mark.parametrize("handoff", ("content", "invariant"))
def test_scored_handoffs_start_as_the_same_exponential_average(handoff):
    ema, current = build("ema"), build(handoff)
    current.load_state_dict(
        {
            **{n: p for n, p in ema.state_dict().items()},
            **{n: p for n, p in current.state_dict().items() if n != "handoff_beta_logit"},
        }
    )
    assert torch.equal(ema(TOKENS, POSITIONS), current(TOKENS, POSITIONS))


@pytest.mark.parametrize("handoff", ("ema", "content", "invariant"))
def test_streaming_recurrence_equals_normalized_attention_over_history(handoff):
    model = build(handoff)
    perturb_handoff(model)
    injection = model._embedded(TOKENS)
    state = model.initial_state(injection)
    candidates, scores = [injection], [torch.zeros(injection.shape[:-1], dtype=injection.dtype)]
    for _ in range(4):
        h, _, _, previous_probability = state
        output = model._round(h + injection)
        probability, feedback = model._decoded_feedback(output)
        candidate = rms_normalize(output + feedback, model.eps)
        raw = model.score(candidate, h, probability, previous_probability)
        candidates.append(candidate)
        scores.append(raw.clamp(-model.score_clip, model.score_clip))
        state, _ = model.aggregate_round(state, injection)
    beta = float(model.beta.detach())
    t = len(candidates) - 1
    weights = [scores[i].exp() * beta ** (t - i) for i in range(t + 1)]
    explicit = sum(w.unsqueeze(-1) * c for w, c in zip(weights, candidates, strict=True))
    explicit = explicit / sum(weights).unsqueeze(-1)
    torch.testing.assert_close(state[0], explicit, rtol=1e-10, atol=1e-12)
    norms = torch.stack([c.norm(dim=-1) for c in candidates]).amax(0)
    assert (state[0].norm(dim=-1) <= norms + 1e-9).all()


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_two_segments_continue_one_call_exactly(handoff):
    model = build(handoff)
    perturb_handoff(model)
    once = model.run(TOKENS, POSITIONS, rounds=5, carry=False)[0]
    _, carry = model.run(TOKENS, POSITIONS, rounds=2)
    twice = model.run(TOKENS, POSITIONS, rounds=3, state=carry, carry=False)[0]
    torch.testing.assert_close(once, twice, rtol=0, atol=0)


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_trace_readout_equals_a_fresh_call_at_every_round(handoff):
    model = build(handoff)
    perturb_handoff(model)
    logits, change, gamma = model.trace(TOKENS, POSITIONS, 5)
    for r in range(1, 6):
        torch.testing.assert_close(logits[r - 1], model.run(TOKENS, POSITIONS, r, carry=False)[0])
    assert torch.isfinite(change).all()
    if handoff in ("additive", "plain"):
        assert torch.isnan(gamma).all()
    else:
        assert ((gamma > 0) & (gamma <= 1)).all()


def test_invariant_score_ignores_direction_and_vocabulary_labels():
    model = build("invariant")
    perturb_handoff(model)
    generator = torch.Generator().manual_seed(3)
    candidate = rms_normalize(torch.randn(2, 5, 12, generator=generator, dtype=torch.double), 1e-6)
    previous = rms_normalize(torch.randn(2, 5, 12, generator=generator, dtype=torch.double), 1e-6)
    p = torch.randn(2, 5, 23, generator=generator, dtype=torch.double).softmax(-1)
    q = torch.randn(2, 5, 23, generator=generator, dtype=torch.double).softmax(-1)
    rotation = torch.linalg.qr(torch.randn(12, 12, generator=generator, dtype=torch.double))[0]
    relabel = torch.randperm(23, generator=generator)
    base = model.score(candidate, previous, p, q)
    moved = model.score(candidate @ rotation, previous @ rotation, p[..., relabel], q[..., relabel])
    torch.testing.assert_close(base, moved, rtol=1e-10, atol=1e-12)
    assert model.features(candidate, previous, p, q).shape[-1] == len(FEATURES)
    content = build("content")
    perturb_handoff(content)
    assert not torch.allclose(
        content.score(candidate, previous, p, q),
        content.score(candidate @ rotation, previous @ rotation, p[..., relabel], q[..., relabel]),
    )


@pytest.mark.parametrize("handoff", HANDOFFS)
def test_deep_supervision_detaches_between_segments_and_trains_handoff(handoff):
    model = build(handoff).train()
    perturb_handoff(model)
    labels = torch.tensor([[7, 2], [5, -100]])
    loss, parts = deep_supervision_loss(model, TOKENS, POSITIONS, labels, [2, 3])
    assert len(parts) == 2 and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # The second segment's loss equals a call from the detached first-segment carry.
    model.zero_grad(set_to_none=True)
    _, carry = model.run(TOKENS, POSITIONS, 2)
    assert carry[0].requires_grad
    detached = detach_state(carry)
    assert all(value is None or not value.requires_grad for value in detached)
    second = model.run(TOKENS, POSITIONS, 3, detached, carry=False)[0]
    expected = torch.nn.functional.cross_entropy(second.flatten(0, 1), labels.flatten())
    torch.testing.assert_close(parts[1], expected.detach())


def test_schedule_uses_only_configured_ranges():
    rng = np.random.default_rng(4)
    draws = np.array([sample_schedule(rng, (4, 6), (1, 4), 2) for _ in range(2000)])
    assert set(draws[:, 0]) == {4, 5, 6} and set(draws[:, 1]) == {1, 2, 3, 4}
    again = np.random.default_rng(4)
    assert sample_schedule(again, (4, 6), (1, 4), 2) == list(draws[0])
    with pytest.raises(ValueError):
        sample_schedule(rng, (4, 6), (1, 4), 0)


def test_readout_reads_the_aggregated_state():
    model = build("ema")
    logits, state = model.run(TOKENS, POSITIONS, 3)
    torch.testing.assert_close(logits, _readout(model, state[0], POSITIONS))
    assert math.isclose(float(model.beta.detach()), 0.55, rel_tol=1e-6)


def runner():
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_loop_native_handoff")


def test_config_validates_and_tracks_to_the_team_project():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs/loop-native-handoff-development-v1.json"
    config = json.loads(path.read_text())
    runner().validate_config(config)
    assert {a["handoff"] for a in config["arms"].values()} == set(HANDOFFS) - {"plain"}
    assert config["training"]["first_segment_rounds"][0] >= 4  # covers the longest trained hop


def test_fixed_four_round_additive_flops_equal_the_old_discoloop_accounting():
    module = runner()
    torch.manual_seed(71)
    hf = GPT2Config(n_embd=768, n_head=12, n_layer=1, n_positions=1024, n_ctx=1024, vocab_size=1013)
    model = LoopNativeHandoffGPT(hf, 4, handoff="additive")
    metadata = {"vocab_size": 1013, "sequence_length": 6}
    # Recorded by the reused ungated DiscoLoop: base 527297937408 + channel 86038806528.
    assert module.step_flops(model, metadata, 512, [4]) == 613336743936


def test_learning_rate_warms_up_then_decays_to_the_floor():
    module = runner()
    training = {"lr": 1e-4, "warmup": 10, "steps": 110, "final_lr_fraction": 0.1}
    assert module.learning_rate(training, 1) == 0
    assert math.isclose(module.learning_rate(training, 10), 0.9e-4)
    assert math.isclose(module.learning_rate(training, 11), 1e-4, rel_tol=1e-3)
    assert math.isclose(module.learning_rate(training, 110), 1e-5)


def test_parameter_average_is_applied_and_then_restored():
    module = runner()
    model = build("invariant")
    average = module.ParameterAverage(model, 0.5)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    perturb_handoff(model)
    average.update(model)
    perturbed = {n: p.detach().clone() for n, p in model.named_parameters()}
    with average.applied(model):
        for name, parameter in model.named_parameters():
            expected = 0.5 * original[name] + 0.5 * perturbed[name]
            torch.testing.assert_close(parameter, expected.to(parameter.dtype))
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, perturbed[name])


def test_initial_readout_is_not_a_copy_of_the_input_token():
    """The aggregated state keeps the native embedding scale, like the additive chain."""
    torch.manual_seed(71)
    hf = GPT2Config(
        n_embd=64,
        n_head=4,
        n_layer=1,
        n_positions=12,
        n_ctx=12,
        vocab_size=301,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    tokens = torch.randint(3, 301, (64, 6))
    for handoff in HANDOFFS:
        torch.manual_seed(71)
        model = LoopNativeHandoffGPT(hf, 4, handoff=handoff).eval()
        logits = model(tokens)
        copied = (logits.argmax(-1) == tokens).float().mean()
        assert copied < 0.5, handoff
