"""Controls separating knowledge load, derived supervision, and serial recall."""

from __future__ import annotations

import numpy as np

from .capacity_scaling import DIGIT, EOS, TYPE_B


def training_indices(world, spec, epoch):
    """Keep the historical mixed stream exactly, or replace derived examples.

    Replay preserves the mixed stream's batch sizes/updates and each original
    atomic slot. Extra slots carry balanced atomic replay, never held-out labels.
    """
    atoms = len(world["atomic"])
    mixed = atoms + len(world["train_composition"])
    mode = spec.get("training_mode", "mixed")
    if mode not in {"mixed", "atomic", "atomic_replay"}:
        raise ValueError(mode)
    n = atoms if mode == "atomic" else mixed
    rng = np.random.default_rng(np.random.SeedSequence([spec["stream_seed"], epoch]))
    order = rng.permutation(n)
    if mode == "atomic_replay":
        replay_rng = np.random.default_rng(
            np.random.SeedSequence([spec["stream_seed"], epoch, 771])
        )
        replacements = np.resize(replay_rng.permutation(atoms), mixed - atoms)
        source = np.concatenate((np.arange(atoms), replacements))
        return source[order]
    return order


def decode_bridge(predictions):
    """Reject malformed/out-of-domain predictions instead of oracle repairing."""
    p = np.asarray(predictions)
    d = p[:, 1:5] - DIGIT
    valid = (
        (p[:, 0] == TYPE_B)
        & (p[:, 5] == EOS)
        & ((d >= 0) & (d < 16)).all(1)
        & (d[:, :2] == 0).all(1)
    )
    values = (np.clip(d, 0, 15) * np.array([4096, 256, 16, 1])).sum(1)
    return np.where(valid, values, 0), valid


def first_queries(compositions):
    rows = np.zeros_like(compositions)
    rows[:, 0] = 0
    rows[:, 1:3] = compositions[:, 1:3]
    rows[:, 3] = -1
    rows[:, 5:] = -1
    return rows


def second_queries(compositions, bridges):
    rows = np.zeros_like(compositions)
    rows[:, 0] = 1
    rows[:, 1] = bridges
    rows[:, 2] = compositions[:, 3]
    rows[:, 3] = -1
    rows[:, 5:] = -1
    return rows
