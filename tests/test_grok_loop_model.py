"""Weight sharing and causal/compute contracts for the loop comparison."""

import copy
import io
import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_model import ModelConfig, matmul_flops
from llm_memory_editability.grok_depth import SmallGPT
from llm_memory_editability.grok_loop_model import LoopGPT, flops


@pytest.fixture
def config():
    return ModelConfig(vocab_size=23, width=16, layers=2, heads=2, context=8)


@pytest.mark.parametrize("training", [False, True])
def test_one_loop_legacy_matches_smallgpt_initialization_logits_and_gradients(config, training):
    torch.manual_seed(81)
    baseline = SmallGPT(config, dropout=0.1).train(training)
    torch.manual_seed(81)
    loop = LoopGPT(config, repeats=1, dropout=0.1, initialization="legacy_unique").train(training)
    assert baseline.state_dict().keys() == loop.state_dict().keys()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, loop.state_dict()[key])
    x = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    positions = torch.tensor([[2, 3], [2, 3]])
    rng = torch.get_rng_state()
    expected = baseline(x, positions)
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = loop(x, positions)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_rng, torch.get_rng_state())
    actual.sum().backward()
    expected.sum().backward()
    for left, right in zip(baseline.parameters(), loop.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)


def test_executed_blocks_share_all_parameters_and_storage(config):
    loop = LoopGPT(config, repeats=3)
    executed = list(loop.iter_blocks())
    assert len(executed) == loop.effective_depth == 6
    assert len(loop.blocks) == config.layers
    assert len(list(loop.parameters())) == len(list(SmallGPT(config).parameters()))
    for index, block in enumerate(executed):
        original = loop.blocks[index % config.layers]
        assert block is original
        for name, parameter in block.named_parameters():
            assert parameter is dict(original.named_parameters())[name]
            assert parameter.data_ptr() == dict(original.named_parameters())[name].data_ptr()
    assert len({id(block) for block in executed}) == config.layers
    assert len(list(loop.iter_blocks(repeats=2))) == 4


@pytest.mark.parametrize("training", [False, True])
def test_shared_gradients_equal_sum_of_unrolled_copy_gradients(config, training):
    torch.manual_seed(45)
    loop = LoopGPT(config, repeats=3, dropout=0.1).double().train(training)
    unrolled = SmallGPT(replace(config, layers=loop.effective_depth), dropout=0.1).double()
    unrolled.train(training)
    unrolled.token.load_state_dict(loop.token.state_dict())
    unrolled.position.load_state_dict(loop.position.state_dict())
    unrolled.ln_final.load_state_dict(loop.ln_final.state_dict())
    for block, copied in zip(loop.iter_blocks(), unrolled.blocks, strict=True):
        copied.load_state_dict(block.state_dict())
    x = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    positions = torch.tensor([[2, 3], [2, 3]])
    labels = torch.tensor([[5, 1], [9, 1]])
    rng = torch.get_rng_state()
    actual = loop(x, positions)
    torch.set_rng_state(rng)
    expected = unrolled(x, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    F.cross_entropy(actual.flatten(0, 1), labels.flatten()).backward()
    F.cross_entropy(expected.flatten(0, 1), labels.flatten()).backward()
    for index, block in enumerate(loop.blocks):
        for name, parameter in block.named_parameters():
            summed = sum(
                dict(unrolled.blocks[r * config.layers + index].named_parameters())[name].grad
                for r in range(loop.repeats)
            )
            torch.testing.assert_close(parameter.grad, summed, rtol=1e-12, atol=1e-12)
    for name in ("token", "position", "ln_final"):
        for left, right in zip(
            getattr(loop, name).parameters(), getattr(unrolled, name).parameters(), strict=True
        ):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


@pytest.mark.parametrize("hops", [1, 2, 3, 4])
def test_answer_and_eos_are_causal_across_all_loops(config, hops):
    torch.manual_seed(91)
    model = LoopGPT(config, repeats=4).eval()
    x = torch.tensor([[2, *range(10, 10 + hops), 5, 0]])
    positions = torch.tensor([[hops, hops + 1]])
    original = model(x, positions).detach()
    changed_answer = x.clone()
    changed_answer[:, hops + 1] = 21
    torch.testing.assert_close(
        model(changed_answer, positions)[:, 0], original[:, 0], rtol=0, atol=0
    )
    changed_padding = x.clone()
    changed_padding[:, -1] = 22
    torch.testing.assert_close(model(changed_padding, positions), original, rtol=0, atol=0)
    torch.testing.assert_close(model(x[:, :-1], positions), original, rtol=1e-6, atol=1e-7)


def test_residual_initialization_changes_only_output_matrices(config):
    models = {}
    for initialization in ("legacy_unique", "scaled_effective", "zero_residual"):
        torch.manual_seed(74)
        models[initialization] = LoopGPT(config, repeats=4, initialization=initialization)
    legacy = models["legacy_unique"].state_dict()
    for name, value in legacy.items():
        scaled = models["scaled_effective"].state_dict()[name]
        zeroed = models["zero_residual"].state_dict()[name]
        if name.endswith(("attention.proj.weight", "mlp.down.weight")):
            torch.testing.assert_close(scaled, value / math.sqrt(4), rtol=0, atol=0)
            assert torch.count_nonzero(zeroed) == 0
        else:
            assert torch.equal(scaled, value)
            assert torch.equal(zeroed, value)


def test_zero_residual_starts_as_identity_hidden_computation(config):
    model = LoopGPT(config, repeats=4, dropout=0, initialization="zero_residual").eval()
    x = torch.tensor([[2, 3, 4, 5]])
    embedded = model.token(x) + model.position(torch.arange(x.shape[1]))
    expected = F.linear(model.ln_final(embedded), model.token.weight)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_parameter_count_and_flops_follow_unique_and_executed_depth(config):
    model = LoopGPT(config, repeats=3)
    baseline = SmallGPT(config)
    deep = SmallGPT(replace(config, layers=6))

    def parameters(m):
        return sum(p.numel() for p in m.parameters())

    assert parameters(model) == parameters(baseline)
    assert parameters(model) < parameters(deep)
    for backward in (True, False):
        expected = matmul_flops(replace(config, layers=6), 7, 6, 2, backward)
        assert flops(config, 3, 7, 6, backward=backward) == expected
    assert flops(config, 1, 7, 6) == matmul_flops(config, 7, 6)
    readout = 2 * 7 * 2 * config.width * config.vocab_size * 3
    assert flops(config, 3, 7, 6) == 3 * (flops(config, 1, 7, 6) - readout) + readout


def test_repeat_override_is_pure_computation_change(config):
    model = LoopGPT(config, repeats=3).eval()
    single = SmallGPT(config).eval()
    single.load_state_dict(model.state_dict())
    before = copy.deepcopy(model.state_dict())
    x = torch.tensor([[2, 3, 4, 5]])
    torch.testing.assert_close(model(x, repeats=1), single(x), rtol=0, atol=0)
    assert model.repeats == 3 and model.effective_depth == 6
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


def test_checkpoint_roundtrip_preserves_sharing_and_predictions(config):
    model = LoopGPT(config, repeats=3, dropout=0.2, initialization="zero_residual").eval()
    with torch.no_grad():
        model.blocks[0].mlp.down.weight.add_(0.03)
    saved = io.BytesIO()
    torch.save(
        {
            "config": model.config_dict(),
            "repeats": model.repeats,
            "dropout": model.dropout,
            "initialization": model.initialization,
            "model": model.state_dict(),
        },
        saved,
    )
    saved.seek(0)
    checkpoint = torch.load(saved, weights_only=True)
    restored = LoopGPT(
        ModelConfig(**checkpoint["config"]),
        repeats=checkpoint["repeats"],
        dropout=checkpoint["dropout"],
        initialization=checkpoint["initialization"],
    ).eval()
    restored.load_state_dict(checkpoint["model"])
    x = torch.tensor([[2, 3, 4, 5]])
    assert torch.equal(model(x), restored(x))
    executed = list(restored.iter_blocks())
    assert executed[0] is executed[2] is executed[4]


@pytest.mark.parametrize("repeats", [0, -1, 1.5, True, "2"])
def test_invalid_repeats_rejected_everywhere(config, repeats):
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        LoopGPT(config, repeats=repeats)
    model = LoopGPT(config)
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        list(model.iter_blocks(repeats=repeats))
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        model(torch.tensor([[2, 3]]), repeats=repeats)
    with pytest.raises(ValueError, match="repeats must be a positive integer"):
        flops(config, repeats, 2, 4)


def test_unknown_initialization_and_no_unique_blocks_rejected(config):
    with pytest.raises(ValueError, match="initialization must"):
        LoopGPT(config, initialization="typo")
    with pytest.raises(ValueError, match="config.layers must"):
        LoopGPT(replace(config, layers=0))
