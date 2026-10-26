import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.self_decode_alignment import SelfDecodeGPT
from llm_memory_editability.whole_loop_gate import WholeLoopGateGPT


def config():
    return GPT2Config(
        vocab_size=31,
        n_positions=10,
        n_layer=2,
        n_embd=16,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        use_cache=False,
    )


def model(mode="full", beta=0):
    torch.manual_seed(11)
    return WholeLoopGateGPT(config(), gate_mode=mode, feedback_beta=beta).eval()


def tokens():
    return torch.tensor([[3, 5, 9, 10, 1], [4, 8, 7, 10, 1]])


def test_native_full_forward_and_gradients():
    ours = model()
    native = SelfDecodeGPT(config(), repeats=5).eval()
    native.transformer.load_state_dict(ours.transformer.state_dict())
    x = tokens()
    pos = torch.full((2, 1), 4)
    a, b = ours(x, pos), native(x, pos).squeeze(1)
    assert torch.equal(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(ours.transformer.parameters(), native.transformer.parameters(), strict=True):
        assert torch.equal(p.grad, q.grad)


@pytest.mark.parametrize("beta", [0, 1])
def test_initial_fixed_token_exact_values_and_backbone_gradients(beta):
    fixed, learned = model("fixed", beta), model("token", beta)
    assert torch.equal(fixed(tokens()), learned(tokens()))
    fixed(tokens()).square().mean().backward()
    learned(tokens()).square().mean().backward()
    for p, q in zip(fixed.transformer.parameters(), learned.transformer.parameters(), strict=True):
        assert torch.equal(p.grad, q.grad)
    assert learned.gate_weight.grad.norm() > 0
    assert learned.gate_bias.grad.abs() > 0
    assert sum(p.numel() for n, p in learned.named_parameters() if n.startswith("gate_")) == 17


@pytest.mark.parametrize("beta", [0, 1])
@pytest.mark.parametrize("mode", ["full", "fixed", "token"])
def test_prefix_shared_weights_no_mutation_and_finite(beta, mode):
    current = model(mode, beta)
    ids = [id(p) for p in current.parameters()]
    before = {n: p.detach().clone() for n, p in current.state_dict().items()}
    with torch.no_grad():
        short = list(current.trajectory(tokens(), 5))
        long = list(current.trajectory(tokens(), 32))
    assert all(torch.equal(short[i][1], long[i][1]) for i in range(5))
    assert all(torch.isfinite(state).all() for _, state, _ in long)
    assert all(torch.equal(p, before[n]) for n, p in current.state_dict().items())
    assert ids == [id(p) for p in current.parameters()]
    assert len(current.transformer.h) == 2


def test_endpoint_copy_and_candidate():
    current = model("token", 1)
    x = current._embedded(tokens())
    zero, _ = current.step(x, gate_override=torch.zeros_like)
    one, info = current.step(x, gate_override=torch.ones_like)
    assert torch.equal(zero, x)
    assert torch.equal(one, info["candidate"])


def test_beta_zero_never_calls_feedback(monkeypatch):
    current = model("token")

    def fail(*args):
        raise AssertionError("Unnecessary vocabulary softmax")

    monkeypatch.setattr(current, "feedback", fail)
    assert current(tokens()).shape == (2, 31)


def test_causal_prefix_and_post_loop_gate():
    current = model("token", 1)
    x, y = tokens(), tokens().clone()
    y[:, 3:] = 18
    with torch.no_grad():
        for (_, a, _), (_, b, _) in zip(
            current.trajectory(x, 5), current.trajectory(y, 5), strict=True
        ):
            assert torch.equal(a[:, :3], b[:, :3])


def test_final_feedback_is_active():
    current = model("full", 1)
    x = current._embedded(tokens())
    h = current._round(x)
    state, _ = current.step(x)
    assert torch.equal(state, h + current.feedback(h))
    assert not torch.equal(state, h)
