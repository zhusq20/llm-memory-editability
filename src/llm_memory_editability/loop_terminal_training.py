"""Per-sequence stochastic depth and terminal, supervised-token-mean cross entropy.

The whole batch advances together. Samples at their sampled endpoint retain their
complete carry through differentiable identity paths. One terminal token-mean loss
backpropagates through every active round; no trajectory is detached or truncated.
The original grouped loss remains available as a numerical reference.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F

from .self_decode_alignment import _loss_dtype, _readout


def sample_depths(rng, batch_size, *, mean=8.0, minimum=1, maximum=16):
    """Independent Poisson draws conditioned on inclusive bounds, by rejection.

    The caller owns/checkpoints this RNG independently of data and dropout RNGs.
    Depths depend on neither labels nor the number of supervised tokens.
    """
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError("Poisson mean must be finite and positive")
    if any(type(n) is not int for n in (batch_size, minimum, maximum)):
        raise ValueError("Batch size and depth bounds must be integers")
    if batch_size < 1 or not 1 <= minimum <= maximum:
        raise ValueError("Require a nonempty batch and 1 <= minimum <= maximum")
    depths = rng.poisson(mean, size=batch_size)
    invalid = (depths < minimum) | (depths > maximum)
    while invalid.any():
        depths[invalid] = rng.poisson(mean, size=int(invalid.sum()))
        invalid = (depths < minimum) | (depths > maximum)
    return depths


def checked_depths(depths, batch_size):
    """Validate the caller's independently sampled CPU depth vector."""
    depths = np.asarray(depths)
    if (
        batch_size < 1
        or depths.shape != (batch_size,)
        or not np.issubdtype(depths.dtype, np.integer)
        or np.any(depths < 1)
    ):
        raise ValueError("Provide one positive integer depth per sequence")
    return depths


def depth_groups(depths, batch_size):
    """Group already sampled depths without changing the sampling law."""
    depths = checked_depths(depths, batch_size)
    for depth in np.unique(depths):
        yield int(depth), np.flatnonzero(depths == depth)


def run_terminal_batch(model, tokens, positions, depths):
    """Advance a fixed batch, freezing h/N/D/p after each sample's endpoint.

    Masking does not skip candidate computation for expired rows. It preserves the
    old carry's graph instead of detaching it. The original embedding/dropout draw
    is shared by all rounds, and only the trajectory's first block receives e.
    Returns each sample's terminal logits and full, still-connected carry.
    """
    depths = checked_depths(depths, len(tokens))
    if model.handoff != "invariant" or model.first_round_input != "embedding":
        raise ValueError("Masked terminal execution requires the embedding-first invariant recipe")
    injection = _loss_dtype(model._embedded(tokens))
    state = model.initial_state(injection)
    depth_tensor = torch.as_tensor(depths, device=tokens.device)
    for index in range(int(depths.max())):
        proposed, _ = model._trajectory_round(state, injection, first_round=index == 0)
        active = (depth_tensor > index)[:, None, None]
        state = tuple(
            torch.where(active, new, old) for new, old in zip(proposed[:4], state[:4], strict=True)
        ) + (injection,)
    return _readout(model, state[0], positions), state


def supervised_token_count(tokens, positions, labels, ignore_index):
    if labels.ndim != 2 or len(labels) != len(tokens) or positions.shape != labels.shape:
        raise ValueError("Labels and readout positions must have matching [batch, slots] shapes")
    count = int(labels.ne(ignore_index).sum().item())
    if count == 0:
        raise ValueError("A logical batch needs at least one supervised token")
    return count


def terminal_loss(
    model, tokens, positions, labels, depths, *, ignore_index=-100, forward_context=nullcontext
):
    """One loss on the frozen endpoints, normalized over all valid batch tokens."""
    valid_tokens = supervised_token_count(tokens, positions, labels, ignore_index)
    with forward_context():
        logits, _ = run_terminal_batch(model, tokens, positions, depths)
        return (
            F.cross_entropy(
                _loss_dtype(logits).flatten(0, 1),
                labels.flatten(),
                ignore_index=ignore_index,
                reduction="sum",
            )
            / valid_tokens
        )


def terminal_loss_groups(
    model, tokens, positions, labels, depths, *, ignore_index=-100, forward_context=nullcontext
):
    """Yield each depth group's contribution to the batch's token-mean loss.

    Labels/positions contain only supervised slots; ignore_index masks absent slots.
    Prompt/padding positions are excluded, and no answer/EOS or per-example weighting
    is applied. Zero-target groups contribute zero and still execute their rollout.
    """
    valid_tokens = supervised_token_count(tokens, positions, labels, ignore_index)
    groups = list(depth_groups(depths, len(tokens)))
    for depth, rows in groups:
        index = torch.as_tensor(rows, device=tokens.device)
        with forward_context():
            logits, _ = model.run(tokens[index], positions[index], rounds=depth, carry=False)
            loss = (
                F.cross_entropy(
                    _loss_dtype(logits).flatten(0, 1),
                    labels[index].flatten(),
                    ignore_index=ignore_index,
                    reduction="sum",
                )
                / valid_tokens
            )
        yield loss


def backward_terminal_loss(model, tokens, positions, labels, depths, **kwargs):
    """Backpropagate the full batch's endpoint loss, returning detached logging loss.

    Caller must zero gradients at logical-batch boundaries and clip/update once,
    after this backward. Backward runs outside the forward autocast context.
    """
    loss = terminal_loss(model, tokens, positions, labels, depths, **kwargs)
    loss.backward()
    return loss.detach()
