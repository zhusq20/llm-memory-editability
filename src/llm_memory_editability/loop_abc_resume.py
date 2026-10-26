"""Exact ABC recovery and CPU snapshots; the original 150-step recipe is retained."""

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


def autocast(device):
    # The first four no_grad rounds of B must not seed detached cached casts
    # that would subsequently be reused in its differentiable rounds.
    return torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def restore(saved, model, optimizer, average, stream, schedule_rng, device):
    for key in ["model", "optimizer", "ema", "stream", "counts", "cpu_rng", "cuda_rng"]:
        if key not in saved:
            raise ValueError(f"Incomplete recovery state: {key}")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    average.load_state_dict({name: value.to(device) for name, value in saved["ema"].items()})
    if any(value.device != device for value in average.values.values()):
        raise ValueError("EMA recovery tensors are on the wrong device")
    stream.load_state_dict(saved["stream"])
    schedule_rng.bit_generator.state = copy.deepcopy(saved["schedule_rng"])
    restore_rng(saved, device)
    return np.array(saved["counts"], copy=True)


def restore_rng(saved, device):
    torch.set_rng_state(saved["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda_rng"], device)


class ThrottledWriter:
    def __init__(self, path, rate):
        self.file = Path(path).open("wb")
        self.rate, self.started, self.written = rate, time.perf_counter(), 0

    def write(self, data):
        view = memoryview(data)
        for begin in range(0, len(view), 1024 * 1024):
            piece = view[begin : begin + 1024 * 1024]
            self.file.write(piece)
            self.written += len(piece)
            time.sleep(max(0, self.written / self.rate - (time.perf_counter() - self.started)))
        return len(view)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class SnapshotWriter:
    def __init__(self, rate=32 * 2**20):
        self.rate = rate
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.seconds = 0.0

    def wait(self):
        if self.pending is not None:
            self.seconds += self.pending.result()
            self.pending = None

    def save(self, path, payload, retain_previous=True):
        self.wait()
        copied = cpu_copy(payload)
        self.pending = self.pool.submit(self._save, Path(path), copied, retain_previous)

    def _save(self, path, payload, retain_previous):
        started = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.pt")
        stream = ThrottledWriter(temporary, self.rate)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.file.fileno())
        finally:
            stream.close()
        if retain_previous and path.exists():
            path.replace(path.with_name("previous.pt"))
        temporary.replace(path)
        return time.perf_counter() - started

    def close(self):
        self.wait()
        self.pool.shutdown()
