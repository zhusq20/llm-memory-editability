"""Bounded asynchronous checkpoint transport; training arithmetic stays untouched."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch


def snapshot(value):
    """Own every tensor before the optimizer is allowed to mutate its next state."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot(item) for item in value)
    return copy.deepcopy(value)


class PacedWriter:
    """Limit checkpoint write bursts without changing the torch archive format."""

    def __init__(self, handle, bytes_per_second):
        self.handle = handle
        self.rate = bytes_per_second
        self.deadline = time.monotonic()

    def write(self, data):
        view = memoryview(data)
        for offset in range(0, len(view), 1024 * 1024):
            part = view[offset : offset + 1024 * 1024]
            delay = self.deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            written = self.handle.write(part)
            if written != len(part):
                raise OSError("Incomplete checkpoint write")
            self.deadline = max(self.deadline, time.monotonic()) + len(part) / self.rate
        return len(view)

    def flush(self):
        self.handle.flush()


def paced_save(value, path, bytes_per_second):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, PacedWriter(handle, bytes_per_second))
        handle.flush()
        # NFS close/rename retain the original persistence contract. A failed
        # write leaves the preceding complete checkpoint in place.
    temporary.replace(path)


class CheckpointTransport:
    """At most one owned snapshot; propagate errors before another save or exit."""

    def __init__(self, original_save, bytes_per_second=32 * 1024 * 1024, writer=paced_save):
        if bytes_per_second <= 0:
            raise ValueError("Checkpoint bandwidth must be positive")
        self.original_save = original_save
        self.rate = bytes_per_second
        self.writer = writer
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-io")
        self.pending = None

    def drain(self):
        if self.pending is not None:
            self.pending.result()
            self.pending = None

    def save(self, value, path):
        self.drain()
        if Path(path).name != "resume.pt":
            return self.original_save(value, path)
        owned = snapshot(value)
        self.pending = self.pool.submit(self.writer, owned, path, self.rate)
        return None

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True)
