"""Independent supervision, carry-gradient and shallow-depth interventions.

The forward recurrence and original embedding injection are unchanged. A split is
effective only for examples whose sampled endpoint lies strictly after it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .loop_terminal_training import checked_depths, supervised_token_count, terminal_loss
from .self_decode_alignment import _loss_dtype, _readout


def split_boundary(seed, step):
    """Uniform K in {4,5,6}, from a separate counter-based Philox stream.

    Frozen seed and optimizer step completely specify the draw. Both are already
    available in the saved config/checkpoint; no mutable second RNG can be lost.
    Drawing K consumes neither the original depth RNG nor any Torch/dropout RNG.
    """
    if type(seed) is not int or type(step) is not int or seed < 0 or step < 1:
        raise ValueError("Split seed must be nonnegative and optimizer step positive")
    generator = np.random.Generator(np.random.Philox(seed=seed, counter=step))
    return int(generator.integers(4, 7))


def recipe_loss(
    model,
    tokens,
    positions,
    labels,
    depths,
    boundary,
    *,
    supervision=False,
    truncate=False,
    ignore_index=-100,
):
    """Return token-mean loss with independently controlled S and G switches.

    For T>K and S, each target token has weight 1/2 at K and 1/2 at T.
    For T<=K, the endpoint has weight one regardless of either switch. With G,
    h/N/D/p are detached just after K only on continuing rows; e remains attached.
    The routine performs no parameter update, backward, RNG sampling or dropout
    resampling beyond the common masked recurrence.
    """
    if type(supervision) is not bool or type(truncate) is not bool:
        raise ValueError("The supervision and truncation switches must be booleans")
    if type(boundary) is not int or boundary not in (4, 5, 6):
        raise ValueError("The common boundary must be 4, 5, or 6")
    depths = checked_depths(depths, len(tokens))
    if model.handoff != "invariant" or model.first_round_input != "embedding":
        raise ValueError("Recipe ablations require embedding-first Invariant")
    if not supervision and not truncate:
        return terminal_loss(model, tokens, positions, labels, depths, ignore_index=ignore_index)
    count = supervised_token_count(tokens, positions, labels, ignore_index)
    injection = _loss_dtype(model._embedded(tokens))
    state = model.initial_state(injection)
    limits = torch.as_tensor(depths, device=tokens.device)
    continuing = limits > boundary
    crosses_boundary = bool(np.any(depths > boundary))
    split_logits = None
    for index in range(int(depths.max())):
        proposed, _ = model._trajectory_round(state, injection, first_round=index == 0)
        active = (limits > index)[:, None, None]
        state = tuple(
            torch.where(active, new, old) for new, old in zip(proposed[:4], state[:4], strict=True)
        ) + (injection,)
        if index + 1 == boundary and crosses_boundary:
            if supervision:
                split_logits = _readout(model, state[0], positions)
            if truncate:
                # Do not detach state[4]: the original embedding still enters later rounds.
                state = tuple(
                    torch.where(continuing[:, None, None], value.detach(), value)
                    for value in state[:4]
                ) + (injection,)
    endpoint = _readout(model, state[0], positions)
    if split_logits is None:
        return (
            F.cross_entropy(
                _loss_dtype(endpoint).flatten(0, 1),
                labels.flatten(),
                ignore_index=ignore_index,
                reduction="sum",
            )
            / count
        )
    terminal_ce = F.cross_entropy(
        _loss_dtype(endpoint).flatten(0, 1),
        labels.flatten(),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(labels)
    split_ce = F.cross_entropy(
        _loss_dtype(split_logits).flatten(0, 1),
        labels.flatten(),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(labels)
    split_weight = continuing[:, None].to(terminal_ce.dtype) * 0.5
    return ((1 - split_weight) * terminal_ce + split_weight * split_ce).sum() / count
