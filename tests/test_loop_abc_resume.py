"""Mixed-autocast regression, complete recovery, and snapshot isolation."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grokking_reproduction import MergedBatchStream
from llm_memory_editability.loop_abc_resume import SnapshotWriter, autocast, restore
from llm_memory_editability.loop_abc_training import abc_loss, auxiliary_depth

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_loop_abc_training import P, X, Y, build  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_loop_native_handoff as native  # noqa: E402


def test_old_b_autocast_cache_regression_is_detected():
    model = build().float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = abc_loss(model, X, P, Y, "B")[0]
    loss.backward()
    missing = [name for name, value in model.named_parameters() if value.grad is None]
    assert "transformer.h.0.attn.c_attn.weight" in missing


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_safe_bfloat16_all_parameters_receive_gradients(arm):
    model = build().float()
    with autocast(torch.device("cpu")):
        loss = abc_loss(model, X, P, Y, arm, 3)[0]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_checkpoint_resume_keeps_next_update_rng_adam_stream_and_auxiliary(arm):
    training = {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}
    model = build(0.1).float()
    optimizer = native.optimizer_for(model, training, fused=False)
    average = native.ParameterAverage(model, 0.999)
    stream = MergedBatchStream(7, 19)
    rng = np.random.default_rng(23)
    with autocast(torch.device("cpu")):
        abc_loss(model, X, P, Y, arm, 3)[0].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    average.update(model)
    stream.batch(2)
    saved = copy.deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": average.state_dict(),
            "stream": stream.state_dict(),
            "schedule_rng": rng.bit_generator.state,
            "counts": np.ones(7),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": None,
        }
    )
    expected_indices = stream.batch(2)[0]
    k = auxiliary_depth(31, 151)
    with autocast(torch.device("cpu")):
        expected_loss = abc_loss(model, X, P, Y, arm, k)[0]
    expected_loss.backward()
    optimizer.step()
    resumed = build(0.1).float()
    other = native.optimizer_for(resumed, training, fused=False)
    restored_average = native.ParameterAverage(resumed, 0.999)
    restored_stream = MergedBatchStream(7, 19)
    counts = restore(
        saved,
        resumed,
        other,
        restored_average,
        restored_stream,
        np.random.default_rng(23),
        torch.device("cpu"),
    )
    np.testing.assert_array_equal(counts, saved["counts"])
    np.testing.assert_array_equal(expected_indices, restored_stream.batch(2)[0])
    with autocast(torch.device("cpu")):
        actual_loss = abc_loss(resumed, X, P, Y, arm, k)[0]
    actual_loss.backward()
    other.step()
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, resumed.get_parameter(name), rtol=0, atol=0)


def test_cpu_snapshot_owns_its_tensors_and_reloads_full_state(tmp_path):
    original = {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "step": 151,
        "optimizer": {"state": {}},
        "rng": np.random.default_rng(3).bit_generator.state,
    }
    writer = SnapshotWriter(rate=2**30)
    path = tmp_path / "latest.pt"
    writer.save(path, original)
    original["model"]["weight"].add_(10)
    writer.wait()
    actual = torch.load(path, weights_only=False)
    torch.testing.assert_close(actual["model"]["weight"], torch.tensor([1.0, 2.0]))
    writer.save(path, original)
    writer.close()
    assert (tmp_path / "previous.pt").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA recovery contract")
def test_cpu_ema_snapshot_can_update_the_restored_cuda_model():
    torch.cuda.set_per_process_memory_fraction(0.16)
    model = build().float()
    training = {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1}
    optimizer = native.optimizer_for(model, training, fused=False)
    average = native.ParameterAverage(model, 0.999)
    stream = MergedBatchStream(7, 19)
    rng = np.random.default_rng(23)
    saved = copy.deepcopy(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": average.state_dict(),
            "stream": stream.state_dict(),
            "schedule_rng": rng.bit_generator.state,
            "counts": np.ones(7),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
    )
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = native.optimizer_for(model, training, fused=False)
    restore(saved, model, optimizer, average, stream, rng, device)
    assert all(value.device == device for value in average.values.values())
    average.update(model)
