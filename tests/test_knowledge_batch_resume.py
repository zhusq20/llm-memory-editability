import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from knowledge_batch_resume import (  # noqa: E402
    choose_batch,
    lr_scale,
    memory_allowance,
    next_indices,
    validate_checkpoint,
)


def test_switching_batch_preserves_exact_sample_stream_and_epoch_boundaries():
    n, epochs, seed, offset = 28, 5, 1234, 12
    expected = []
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        expected += order
    actual = []
    cursor = offset
    while cursor < n * epochs:
        chunk = next_indices(n, epochs, seed, cursor, 16)
        actual += chunk
        cursor += len(chunk)
    assert actual == expected[offset:]
    assert cursor == n * epochs
    assert next_indices(n, epochs, seed, cursor, 16) == []


def test_learning_rate_continues_at_old_sample_position():
    cfg = dict(batch_size=4, expected_training_rows=297840, epochs=5, warmup_steps=100)
    for step in [0, 50, 100, 25000, 372299]:
        old = step / 100 if step < 100 else (372300 - step) / (372300 - 100)
        assert lr_scale(step * 4, cfg) == pytest.approx(old)
    assert lr_scale(1489200, cfg) == 0


def test_resume_rejects_wrong_data_configuration_and_cursor():
    cfg = dict(batch_size=4, expected_training_rows=100, epochs=5)
    saved = dict(config=cfg, arm="pd_full", data_sha256="abc", step=5, exposure=20)
    validate_checkpoint(saved, cfg, "pd_full", "abc")
    for patch in [dict(exposure=21), dict(arm="sft_full"), dict(data_sha256="wrong")]:
        with pytest.raises(ValueError):
            validate_checkpoint({**saved, **patch}, cfg, "pd_full", "abc")


def test_selection_uses_throughput_and_pressure_not_largest_feasible_batch():
    rows = [
        dict(batch_size=b, samples_per_second=s, passed=p)
        for b, s, p in [
            (4, 10, True),
            (8, 18, True),
            (16, 17.9, True),
            (32, 15, True),
            (64, 22, False),
        ]
    ]
    assert choose_batch(rows) == 16
    assert memory_allowance(98000, 65000, 40000) == 43664


def test_adam_moments_weights_and_counter_survive_probe_discard(tmp_path):
    torch = pytest.importorskip("torch")
    from run_knowledge_batch_resume import verify_restoration

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(2, 3)).square().sum().backward()
        optimizer.step()
    path = tmp_path / "checkpoint.pt"
    torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    source = torch.load(path, weights_only=False, mmap=True)
    optimizer.load_state_dict(source["optimizer"])
    for _ in range(4):
        optimizer.zero_grad()
        model(torch.full((8, 3), 2.0)).square().sum().backward()
        optimizer.step()
    pristine = torch.load(path, weights_only=False, mmap=True)
    model.load_state_dict(pristine["weights"])
    optimizer.load_state_dict(pristine["optimizer"])
    verify_restoration(model, optimizer, pristine)
    assert all(int(s["step"]) == 3 for s in optimizer.state.values())
    with torch.no_grad():
        model.weight.add_(1)
    with pytest.raises(AssertionError, match="weight differs"):
        verify_restoration(model, optimizer, pristine)
