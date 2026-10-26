import importlib.util
import random
import threading
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "checkpoint_io", Path(__file__).resolve().parents[1] / "scripts/knowledge_checkpoint_io.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_background_snapshot_owns_weights_optimizer_and_rng(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def writer(value, path, rate):
        entered.set()
        assert release.wait(10)
        M.paced_save(value, path, rate)

    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(4, 3)

    def update():
        optimizer.zero_grad()
        model(x).square().mean().backward()
        optimizer.step()

    update()
    initial = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
    }
    expected = M.snapshot(initial)
    transport = M.CheckpointTransport(torch.save, 1024**3, writer)
    try:
        transport.save(initial, tmp_path / "resume.pt")
        assert entered.wait(10)
        update()
        release.set()
        transport.drain()
        restored = torch.load(tmp_path / "resume.pt", weights_only=False)
        for name, tensor in expected["model"].items():
            assert torch.equal(tensor, restored["model"][name])
        for key, state in expected["optimizer"]["state"].items():
            for name, tensor in state.items():
                assert torch.equal(tensor, restored["optimizer"]["state"][key][name])
        assert torch.equal(expected["rng"], restored["rng"])
        # Replay the same next optimizer update from the asynchronously saved state.
        continuation = M.snapshot(model.state_dict())
        model.load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["rng"])
        update()
        for name, tensor in continuation.items():
            assert torch.equal(tensor, model.state_dict()[name])
    finally:
        release.set()
        transport.close()


def test_failed_write_preserves_old_checkpoint_and_prevents_final_save(tmp_path):
    path = tmp_path / "resume.pt"
    torch.save({"step": 1}, path)

    def fail(*args):
        raise OSError("storage unavailable")

    transport = M.CheckpointTransport(torch.save, writer=fail)
    transport.save({"step": 2}, path)
    with pytest.raises(OSError, match="storage unavailable"):
        transport.save({}, tmp_path / "weights.pt")
    assert torch.load(path, weights_only=False) == {"step": 1}
    assert not (tmp_path / "weights.pt").exists()
    with pytest.raises(OSError, match="storage unavailable"):
        transport.close()


def test_transport_does_not_consume_rng_and_drains_before_final(tmp_path):
    torch.manual_seed(41)
    before = torch.get_rng_state().clone()
    random_before = random.getstate()
    transport = M.CheckpointTransport(torch.save, 1024**3)
    transport.save({"x": torch.ones(5)}, tmp_path / "resume.pt")
    transport.save({"final": True}, tmp_path / "weights.pt")
    transport.close()
    assert torch.equal(before, torch.get_rng_state())
    assert random_before == random.getstate()
    assert torch.load(tmp_path / "resume.pt", weights_only=False)["x"].eq(1).all()
    assert torch.load(tmp_path / "weights.pt", weights_only=False)["final"]
