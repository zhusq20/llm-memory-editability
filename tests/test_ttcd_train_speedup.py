"""Numerical-only checks: no optimizer steps, short training, or CUDA work."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability import ttcd_train_speedup as speedup
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_continual import write_states
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def reader_for(name, dtype=torch.float32):
    torch.manual_seed(17)
    config = GPT2Config(
        vocab_size=32,
        n_positions=40,
        n_embd=8,
        n_head=2,
        n_layer=8 if name == "d8" else 4,
        n_inner=16,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    model = SupervisedAlignmentGPT(
        config, repeats=1 if name == "d8" else 2, discoloop=name == "disco"
    ).to(dtype=dtype)
    return TTCDReader(model, trainable_backbone=True, native_causal_attention=True).eval()


def fixed_inputs():
    # Eight distinct atomic correction packets, original23-token layout.
    packets = torch.tensor(
        [[2 + i, 18 + i % 4, 10 + i, 1] + [0] * 17 + [2 + i, 18 + i % 4] for i in range(8)]
    )
    atomic = torch.tensor([[2 + i, 18 + i % 4, 10 + i] for i in range(8)])
    retained = torch.tensor(
        [[2 + (i + 1) % 8, 18 + (i + 2) % 4, 10 + (i + 3) % 8] for i in range(8)]
    )
    return packets, atomic, retained


def objective(reader, length, precision):
    packets, atomic, retained = fixed_inputs()
    writer = {"steps": 4, "teacher_window": 32, "student_window": 16, "learning_rate": 0.001}
    mode = torch.autocast("cpu", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with mode:
        fast = write_states(reader, packets, writer, length, "single" if length == 1 else "stream")
        positions = torch.tensor([[1, 2]]).expand(8, -1)
        labels = torch.stack((atomic[:, -1], torch.ones(8, dtype=torch.long)), dim=1)
        old = torch.stack((retained[:, -1], torch.ones(8, dtype=torch.long)), dim=1)
        loss = F.cross_entropy(reader(atomic, positions, fast=fast).flatten(0, 1), labels.flatten())
        loss = loss + F.cross_entropy(
            reader(retained, positions, fast=fast).flatten(0, 1), old.flatten()
        )
        loss = loss + F.cross_entropy(reader(atomic, positions).flatten(0, 1), labels.flatten())
    loss.backward()
    return (
        fast,
        loss.detach(),
        {
            name: None if value.grad is None else value.grad.clone()
            for name, value in reader.named_parameters()
        },
    )


@pytest.mark.parametrize("name", ["d8", "loop", "disco"])
@pytest.mark.parametrize("length", [1, 8])
@pytest.mark.parametrize("precision", ["fp64", "fp32", "bf16"])
def test_exact_fast_loss_all_gradients_and_rng_without_optimizer(name, length, precision):
    original = reader_for(name, torch.float64 if precision == "fp64" else torch.float32)
    optimized = copy.deepcopy(original)
    before = tensor_digest(original.backbone.state_dict())
    rng = torch.get_rng_state().clone()
    old_fast, old_loss, old_grad = objective(original, length, precision)
    old_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    with speedup.enabled() as provenance:
        new_fast, new_loss, new_grad = objective(optimized, length, precision)
    assert provenance["pruned_calls"] == 4 * length
    assert torch.equal(old_loss, new_loss)
    assert old_fast.keys() == new_fast.keys()
    assert all(torch.equal(old_fast[key], new_fast[key]) for key in old_fast)
    assert old_grad.keys() == new_grad.keys()
    for key in old_grad:
        if old_grad[key] is None:
            assert new_grad[key] is None
        else:
            assert torch.equal(old_grad[key], new_grad[key]), key
    assert torch.equal(old_rng, torch.get_rng_state())
    assert (
        before
        == tensor_digest(original.backbone.state_dict())
        == tensor_digest(optimized.backbone.state_dict())
    )
    assert original.compute.keys() == optimized.compute.keys()
    assert original.compute["forward_calls"] == optimized.compute["forward_calls"]
    assert original.compute["executed_input_tokens"] == optimized.compute["executed_input_tokens"]
    assert (
        original.compute["estimated_dense_forward_matmul_flops"]
        > optimized.compute["estimated_dense_forward_matmul_flops"]
    )
    assert (
        original.compute["estimated_backward_matmul_flops_proxy"]
        > optimized.compute["estimated_backward_matmul_flops_proxy"]
    )


def test_native_and_inference_capture_outputs_remain_original_and_undo_restores_function():
    reader = reader_for("disco")
    packets, atomic, _ = fixed_inputs()
    fast = reader.empty_fast()
    baseline = reader(atomic).detach()
    expected, expected_info = reader.write(
        packets[:1], fast, teacher_window=32, student_window=16, learning_rate=0.001
    )
    function = TTCDReader.forward
    undo = speedup.install()
    try:
        assert torch.equal(reader(atomic).detach(), baseline)
        observed, info = reader.write(
            packets[:1], fast, teacher_window=32, student_window=16, learning_rate=0.001
        )
        assert all(torch.equal(expected[k], observed[k]) for k in fast)
        assert info == expected_info
        assert undo.provenance["pruned_calls"] == 0
        with pytest.raises(RuntimeError, match="already installed"):
            speedup.install()
    finally:
        undo()
    undo()
    assert TTCDReader.forward is function


def test_capture_prefix_returns_before_any_disco_boundary(monkeypatch):
    reader = reader_for("disco")
    packets, _, _ = fixed_inputs()
    capture = {"student_window": 16, "positions": torch.full((8,), 22), "outer_grad": True}
    fast = {k: v.unsqueeze(0).expand(8, -1, -1).clone() for k, v in reader.empty_fast().items()}

    def forbidden(*args):
        raise AssertionError("The discarded Disco boundary must not execute")

    monkeypatch.setattr(reader.backbone, "_discoloop_boundary", forbidden)
    with speedup.enabled():
        assert (
            reader(packets, capture["positions"][:, None], fast=fast, window=32, capture=capture)
            is None
        )
    assert set(capture["features"]) == set(range(4))


def test_source_pin_rejects_unreviewed_reader_without_mutating_runtime(monkeypatch):
    original = TTCDReader.forward
    monkeypatch.setattr(speedup, "ORIGINAL_FILE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="exact audited"):
        speedup.install()
    assert TTCDReader.forward is original
