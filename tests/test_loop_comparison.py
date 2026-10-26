"""Contracts for the four-method comparison and its two feedback controls."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_loop_feedback_controls as controls  # noqa: E402
import run_loop_fourway as runner  # noqa: E402

from llm_memory_editability.loop_comparison import endpoint_loss, terminal_logits  # noqa: E402
from llm_memory_editability.loop_native_handoff import rms_normalize  # noqa: E402

ALL_ARMS = [*runner.ARMS, *controls.ARMS]


def config():
    return json.loads((ROOT / "configs/loop-fourway-terminal-20261024.json").read_text())


def model(arm):
    c = config()
    c["arms"].update(controls.ARMS)
    c["model"].update(
        n_embd=12, n_head=3, n_positions=12, n_ctx=12, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0
    )
    return runner.initialize(
        c, arm, {"vocab_size": 23, "pad_token": 0, "end_marker": 1}, torch.device("cpu")
    ).eval()


@pytest.fixture(autouse=True)
def cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_masked_endpoints_and_gradients_match_independent_trajectories(arm):
    a = model(arm)
    if a.handoff == "infiloop":
        with torch.no_grad():
            a.handoff_query.uniform_(-0.1, 0.1)
    elif a.handoff == "invariant":
        with torch.no_grad():
            a.handoff_score_out.add_(0.1)
    b = copy.deepcopy(a)
    tokens = torch.tensor([[2, 4, 7, 8, 1], [3, 2, 9, 1, 5], [5, 2, 6, 8, 3]])
    positions = torch.tensor([[1, 3], [1, 2], [2, 4]])
    labels = torch.tensor([[7, -100], [9, 1], [8, 3]])
    depths = [1, 4, 16]
    loss = endpoint_loss(a, tokens, positions, labels, depths)
    loss.backward()
    expected = (
        sum(
            F.cross_entropy(
                b.run(tokens[i : i + 1], positions[i : i + 1], d, carry=False)[0].flatten(0, 1),
                labels[i],
                reduction="sum",
            )
            for i, d in enumerate(depths)
        )
        / labels.ne(-100).sum()
    )
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    for name, parameter in a.named_parameters():
        other = b.get_parameter(name)
        assert parameter.grad is not None and other.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_same_backbone_and_one_block_for_every_arm():
    models = [model(arm) for arm in ALL_ARMS]
    assert len({runner.native.backbone_digest(m) for m in models}) == 1
    assert all(len(m.transformer.h) == 1 for m in models)
    runner.validate_config(config())


def test_infiloop_raw_candidate_and_connection_match_pinned_author_source(monkeypatch):
    source = ROOT / "results/loop-fourway-terminal-20261024/references/infiloop-model.py"
    spec = importlib.util.spec_from_file_location("pinned_infiloop", source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    actual = model("infiloop")
    connection = reference.Connection(reference.Config(hidden_size=12, num_heads=3, layers=1))
    with torch.no_grad():
        actual.handoff_query.uniform_(-0.1, 0.1)
        connection.weight.copy_(actual.handoff_query[None])
        connection.beta_logit.copy_(actual.handoff_beta_logit)
    e = torch.randn(2, 5, 12)
    carry = actual.initial_state(e)
    assert torch.equal(carry[0][0, 0], carry[0][1, 4])
    assert not torch.equal(carry[0], e)
    assert not carry[1].requires_grad and not carry[2].requires_grad
    candidate = torch.randn_like(e, requires_grad=True)
    clone = candidate.detach().clone().requires_grad_(True)
    seen = []

    def block(x):
        seen.append(x)
        return candidate

    monkeypatch.setattr(actual, "_round", block)
    result, _ = actual.aggregate_round(carry, e, first_round=True)
    with torch.no_grad():
        memory = connection.start(carry[0])
    expected, _ = connection(clone, memory)
    torch.testing.assert_close(seen[0], carry[0] + e)
    torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
    result[0].square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(candidate.grad, clone.grad)
    torch.testing.assert_close(actual.handoff_query.grad, connection.weight.grad[0])
    torch.testing.assert_close(actual.handoff_beta_logit.grad, connection.beta_logit.grad[0])


@pytest.mark.parametrize("arm", ALL_ARMS)
def test_future_answer_cannot_change_answer_position(arm):
    m = model(arm)
    x = torch.tensor([[2, 4, 7, 8, 1], [2, 4, 7, 3, 5]])
    pos = torch.tensor([[2], [2]])
    logits = terminal_logits(m, x, pos, [8, 8])
    torch.testing.assert_close(logits[0], logits[1], rtol=0, atol=0)


def test_infiloop_feedback_changes_only_candidate_before_original_connection(monkeypatch):
    actual, baseline = model("infiloop-feedback"), model("infiloop")
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, baseline.state_dict()[key], rtol=0, atol=0)
    with torch.no_grad():
        actual.handoff_query.uniform_(-0.1, 0.1)
    baseline.load_state_dict(actual.state_dict())
    e = torch.randn(2, 5, 12)
    output = torch.randn_like(e, requires_grad=True)
    feedback = torch.randn_like(e, requires_grad=True)
    monkeypatch.setattr(actual, "_round", lambda x: output)
    monkeypatch.setattr(actual, "_decoded_feedback", lambda x: (None, feedback))
    clone = (output + feedback).detach().clone().requires_grad_(True)
    monkeypatch.setattr(baseline, "_round", lambda x: clone)
    state, gamma = actual.aggregate_round(actual.initial_state(e), e)
    expected, expected_gamma = baseline.aggregate_round(baseline.initial_state(e), e)
    for a, b in zip(state[:3], expected[:3], strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(gamma, expected_gamma, rtol=0, atol=0)
    state[0].square().sum().backward()
    expected[0].square().sum().backward()
    torch.testing.assert_close(output.grad, clone.grad)
    torch.testing.assert_close(feedback.grad, clone.grad)


def test_invariant_removes_embedding_projection_but_keeps_probability_score(monkeypatch):
    actual = model("invariant-no-feedback")
    with torch.no_grad():
        actual.handoff_score_out.fill_(0.5)

    def forbidden(*args):
        raise AssertionError("The removed embedding-feedback branch was evaluated")

    monkeypatch.setattr(actual, "_decoded_feedback", forbidden)
    monkeypatch.setattr(actual, "_discoloop_boundary", forbidden)
    e = torch.randn(2, 5, 12)
    old = actual.initial_state(e)
    output = torch.randn_like(e, requires_grad=True)
    probability = actual._decoded_probabilities(output)
    probability.retain_grad()
    monkeypatch.setattr(actual, "_round", lambda x: output)
    monkeypatch.setattr(actual, "_decoded_probabilities", lambda x: probability)
    state, gamma = actual.aggregate_round(old, e, first_round=True)
    candidate = rms_normalize(output, actual.eps)
    expected_weight = (
        actual.score(candidate, old[0], probability, old[3]).clamp(-8, 8).exp()[..., None]
    )
    denominator = actual.beta * old[2] + expected_weight
    expected = (actual.beta * old[1] + expected_weight * candidate) / denominator
    torch.testing.assert_close(state[0], expected)
    torch.testing.assert_close(gamma, expected_weight / denominator)
    assert state[3] is probability
    state[0].square().sum().backward()
    assert probability.grad is not None and probability.grad.abs().sum() > 0
    assert output.grad is not None and output.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", list(controls.ARMS))
def test_control_continuation_trace_and_reload_retain_the_same_endpoint(arm):
    actual = model(arm)
    tokens = torch.tensor([[2, 4, 7, 8, 1]])
    positions = torch.tensor([[1, 3]])
    _, state = actual.run(tokens, positions, rounds=2)
    continued, _ = actual.run(tokens, positions, rounds=3, state=state, carry=False)
    expected = terminal_logits(actual, tokens, positions, [5])
    trace = actual.trace(tokens, positions, 5)[0][-1]
    reloaded = model(arm)
    reloaded.load_state_dict(actual.state_dict())
    torch.testing.assert_close(continued, expected, rtol=0, atol=0)
    torch.testing.assert_close(trace, expected, rtol=0, atol=0)
    torch.testing.assert_close(reloaded.run(tokens, positions, 5)[0], expected, rtol=0, atol=0)


def test_feedback_control_recipe_is_paired_and_separate_from_fourway():
    c = json.loads((ROOT / "configs" / f"{controls.BATCH}.json").read_text())
    controls.validate_config(c)
    with pytest.raises(ValueError):
        runner.validate_config(c)
    for change in ("candidate", "seed", "budget"):
        altered = copy.deepcopy(c)
        if change == "candidate":
            altered["arms"]["invariant-no-feedback"]["feedback_to_candidate"] = True
        elif change == "seed":
            altered["training"]["initialization"] += 1
        else:
            altered["budget"]["arms"] += 1
        with pytest.raises(ValueError):
            controls.validate_config(altered)
