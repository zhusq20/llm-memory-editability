"""Old-knowledge protection drawn from the written fact's address neighbours.

A neighbour of a written fact (h, r, t) shares its subject or its relation. Atoms: old facts
(h, *) and (*, r). Trained chains: those whose first entity or bridge is h, or whose first or
second relation is r. Evaluation rows never enter a protection set.
"""

from __future__ import annotations

import numpy as np


def eligibility(world, excluded, support):
    """Masks of protectable atoms and trained chains, plus each chain's bridge entity."""
    keep_atom = np.array([tuple(row) not in excluded for row in world.atoms.tolist()])
    keep_chain = support.copy()
    for index in np.flatnonzero(support):
        if tuple(world.chains[index].tolist()) in excluded:
            keep_chain[index] = False
    return keep_atom, keep_chain, world.atoms[world.first, 2]


def neighbor_pools(world, fact, keep_atom, keep_chain, bridge):
    """Index arrays of subject- and relation-neighbour atoms and trained chains."""
    h, r, _ = fact
    atoms, chains = world.atoms, world.chains
    return {
        "atoms": {
            "subject": np.flatnonzero(keep_atom & (atoms[:, 0] == h)),
            "relation": np.flatnonzero(keep_atom & (atoms[:, 1] == r) & (atoms[:, 0] != h)),
        },
        "chains": {
            "subject": np.flatnonzero(keep_chain & ((chains[:, 0] == h) | (bridge == h))),
            "relation": np.flatnonzero(
                keep_chain
                & ((chains[:, 1] == r) | (chains[:, 2] == r))
                & (chains[:, 0] != h)
                & (bridge != h)
            ),
        },
    }


class Cycler:
    """Draw without replacement until exhausted, then reshuffle."""

    def __init__(self, values, rng):
        self.values, self.rng, self.queue = np.asarray(values), rng, []

    def take(self, count):
        out = []
        while len(out) < count:
            if not self.queue:
                self.queue = self.rng.permutation(self.values).tolist()
            out.append(self.queue.pop())
        return out


def neighbor_schedule(pools, steps, count, rng):
    """Per step and kind, `count` neighbour indices: half subject, half relation."""
    schedule = []
    cyclers = {
        kind: {side: Cycler(values, rng) for side, values in sides.items() if len(values)}
        for kind, sides in pools.items()
    }
    for _ in range(steps):
        step = {}
        for kind, sides in cyclers.items():
            if not sides:
                raise ValueError(f"No {kind} neighbour is available")
            if len(sides) == 1:
                step[kind] = next(iter(sides.values())).take(count)
            else:
                half = count // 2
                step[kind] = sides["subject"].take(half) + sides["relation"].take(count - half)
        schedule.append(step)
    return schedule
