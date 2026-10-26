"""Add held-out first-hop queries without changing the frozen facts or second-hop panel.

Selection uses graph truth alone. Both old and replacement successors must be
unmodified keys, so a selected query changes only its first hop even after all
facts in the batch have been written.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .path_overlap import World


def _chain_parts(world: World, query: int) -> tuple[int, int]:
    """Check that the stored chain, atom indices and sparse lookup agree."""
    if not 0 <= query < len(world.chains):
        raise ValueError("Query index is outside the original world")
    h, r1, r2, answer = map(int, world.chains[query])
    first, second = int(world.first[query]), int(world.second[query])
    if not (0 <= first < len(world.atoms) and 0 <= second < len(world.atoms)):
        raise ValueError("Chain atom index is outside the original world")
    a, b = world.atoms[first], world.atoms[second]
    if (
        (int(a[0]), int(a[1])) != (h, r1)
        or (int(b[0]), int(b[1]), int(b[2])) != (int(a[2]), r2, answer)
        or int(world.lookup[h, r1]) != first
        or int(world.lookup[int(a[2]), r2]) != second
    ):
        raise ValueError("Original chain truth does not agree with atom indices/lookup")
    return first, second


def build_panel(parent_panel: dict, world: World, settings: dict, phi: float) -> dict:
    """Deep-copy the parent and add ``first_hop`` and ``first_hop_coverage``.

    ``first_hop`` has one entry per unchanged parent fact, including empty ones.
    Its parallel arrays ``D``, ``new_answers``, ``successor_atoms`` and
    ``original_successor_atoms`` contain original chain indices, updated final
    entities, replacement successor atom indices and original successor atom
    indices. Parent ``facts[*].D`` remains the second-hop query panel.

    ``first_hop_coverage`` records candidate/selected counts per fact, empty
    fact indices, and total/covered fact and query counts. No model outcomes
    enter selection. ``settings['first_hop_panel']`` fixes seed 20261015 and may
    specify ``max_queries_per_fact`` (default 8, maximum 8).
    """
    selection = settings["first_hop_panel"]
    seed = selection["seed"]
    limit = selection.get("max_queries_per_fact", 8)
    if seed != 20261015 or not isinstance(limit, int) or not 1 <= limit <= 8:
        raise ValueError("First-hop selection requires seed 20261015 and 1–8 queries per fact")
    facts = parent_panel["facts"]
    edited = set(map(int, parent_panel["edited_atoms"]))
    fact_atoms = [int(fact["atom"]) for fact in facts]
    if len(edited) != len(facts) or len(set(fact_atoms)) != len(facts):
        raise ValueError("Every fixed fact must have a distinct edited atom")
    if set(fact_atoms) != edited:
        raise ValueError("Fixed facts and the complete edited-atom set differ")
    support = np.asarray(world.support(phi), dtype=bool)
    if support.shape != (len(world.chains),):
        raise ValueError("Training support must contain one flag per original chain")
    # Validate the entire original panel before adding any new selection.
    for fact in facts:
        atom = int(fact["atom"])
        if not 0 <= atom < len(world.atoms):
            raise ValueError("Fixed fact atom is outside the original world")
        old, new = int(fact["old_tail"]), int(fact["new_tail"])
        if old != int(world.atoms[atom, 2]) or not 0 <= new < world.entities or new == old:
            raise ValueError("Fixed replacement must differ from the unchanged original truth")
        for query in fact["D"]:
            query = int(query)
            first, second = _chain_parts(world, query)
            if second != atom or first in edited or support[query]:
                raise ValueError("Parent D must edit only the second hop of an untrained chain")

    rng = np.random.RandomState(seed)
    entries, counts = [], []
    for index, fact in enumerate(facts):
        atom, new_bridge = int(fact["atom"]), int(fact["new_tail"])
        candidates = []
        for query in np.flatnonzero((world.first == atom) & ~support):
            query = int(query)
            _, old_successor = _chain_parts(world, query)
            relation = int(world.chains[query, 2])
            new_successor = int(world.lookup[new_bridge, relation])
            # Sparse graphs need not define a successor for the fixed replacement.
            if new_successor < 0 or old_successor in edited or new_successor in edited:
                continue
            if new_successor >= len(world.atoms):
                raise ValueError("Replacement successor lookup points outside the world")
            successor = world.atoms[new_successor]
            if tuple(map(int, successor[:2])) != (new_bridge, relation):
                raise ValueError("Replacement successor does not agree with sparse lookup")
            new_answer = int(successor[2])
            if new_answer == int(world.chains[query, 3]):
                continue
            candidates.append((query, new_answer, new_successor, old_successor))
        selected = [candidates[int(i)] for i in rng.permutation(len(candidates))[:limit]]
        entries.append(
            {
                "fact_index": index,
                "atom": atom,
                "D": [row[0] for row in selected],
                "new_answers": [row[1] for row in selected],
                "successor_atoms": [row[2] for row in selected],
                "original_successor_atoms": [row[3] for row in selected],
            }
        )
        counts.append(
            {
                "fact_index": index,
                "atom": atom,
                "candidates": len(candidates),
                "selected": len(selected),
            }
        )
    result = copy.deepcopy(parent_panel)
    result["first_hop"] = entries
    result["first_hop_coverage"] = {
        "selection": "Graph-only; fixed facts and replacements; exactly one changed hop",
        "seed": seed,
        "max_queries_per_fact": limit,
        "per_fact": counts,
        "empty_fact_indices": [row["fact_index"] for row in counts if row["selected"] == 0],
        "facts_total": len(facts),
        "facts_covered": sum(row["selected"] > 0 for row in counts),
        "candidate_queries_total": sum(row["candidates"] for row in counts),
        "queries_total": sum(row["selected"] for row in counts),
    }
    return result
