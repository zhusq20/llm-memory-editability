"""Fixed-depth terminal, truncated, and auxiliary language-model losses."""

from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from .self_decode_alignment import _loss_dtype, _readout


def auxiliary_depth(seed, step):
    rng = np.random.Generator(np.random.Philox(seed=seed, counter=step))
    return int(rng.integers(1, 8))


def abc_loss(model, tokens, positions, labels, arm, depth=None, audit=False):
    """Eight rounds, one embedding draw, and a valid-target-token mean.

    B differentiates only rounds 5--8 of the carry, preserving the live original
    embedding injection. C uses 0.9 terminal CE plus 0.1 CE at an independent
    intermediate depth. Arbitrary target positions and -100 masks are supported.
    """
    if arm not in ("A", "B", "C"):
        raise ValueError("Expected A, B, or C")
    if arm == "C" and depth not in range(1, 8):
        raise ValueError("C needs an intermediate depth in 1..7")
    count = labels.ne(-100).sum()
    if int(count) == 0 or labels.shape != positions.shape:
        raise ValueError("Expected nonempty supervised targets with matching positions")
    e = _loss_dtype(model._embedded(tokens))
    if audit:
        e.retain_grad()
    state = model.initial_state(e)
    first = None
    middle = None
    for index in range(8):
        context = torch.no_grad() if arm == "B" and index < 4 else nullcontext()
        with context:
            state, _ = model._trajectory_round(state, e, first_round=index == 0)
        if index == 0 and audit:
            first = state[0]
            if first.requires_grad:
                first.retain_grad()
        if arm == "B" and index == 3:
            state = tuple(v.detach() for v in state[:4]) + (e,)
        if arm == "C" and index + 1 == depth:
            middle = _readout(model, state[0], positions)
    terminal = _readout(model, state[0], positions)

    def ce(logits):
        return (
            F.cross_entropy(
                _loss_dtype(logits).flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
            / count
        )

    end_loss = ce(terminal)
    auxiliary = ce(middle) if middle is not None else None
    loss = end_loss if auxiliary is None else 0.9 * end_loss + 0.1 * auxiliary
    return loss, end_loss, auxiliary, terminal, {"embedding": e, "first_state": first}
