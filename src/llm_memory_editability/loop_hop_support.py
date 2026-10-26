"""Nested composition support with the original per-batch category schedule."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from .loop_hop_data import audit_data, load, sha256, traverse


def expand_support(origin, destination, *, multiplier=3, seed=2026101702):
    """Append new A queries; preserve every original fact, query and test byte."""
    world, panels, metadata = load(origin)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=False)
    base_support = metadata["train_compositions_per_hop"]
    n, relations = metadata["entities_per_graph"], metadata["relations"]
    rng = np.random.default_rng(seed)
    for hop in (2, 3, 4):
        original = world[f"train_{hop}"]
        reserved = {tuple(row[:-1]) for row in panels[f"id_{hop}"]}
        seen = {tuple(row[:-1]) for row in original}
        requested = base_support * multiplier
        if requested > n * relations**hop - len(reserved):
            raise ValueError("Expanded support exceeds the nonreserved query space")
        additions = []
        while len(additions) < requested - len(original):
            query = (int(rng.integers(n)), *map(int, rng.integers(relations, size=hop)))
            if query in reserved or query in seen:
                continue
            additions.append(query)
            seen.add(query)
        queries = np.asarray(additions, dtype=np.int64).reshape(-1, hop + 1)
        endpoints = traverse(queries[:, 0], queries[:, 1:], world["permutations"][0])[:, -1]
        world[f"train_{hop}"] = np.concatenate((original, np.column_stack((queries, endpoints))))
    np.savez_compressed(target / "world.npz", **world)
    shutil.copy2(Path(origin) / "panels.npz", target / "panels.npz")
    metadata = {
        **metadata,
        "base_train_compositions_per_hop": base_support,
        "support_multiplier": multiplier,
        "support_expansion_seed": seed,
        "train_compositions_per_hop": base_support * multiplier,
        "training_examples": len(world["atoms"]) + 3 * base_support * multiplier,
        "base_world_sha256": metadata["world_sha256"],
        "world_sha256": sha256(target / "world.npz"),
        "sampling_policy": "Original merged category/query-slot schedule; cycle support blocks "
        "each template epoch; identical atomic, category, token and masking exposure",
    }
    (target / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    audit = audit_data(world, panels, metadata)
    audit["original_panels_byte_identical"] = (
        sha256(target / "panels.npz") == metadata["panels_sha256"]
    )
    (target / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return metadata


class MatchedSupportStream:
    """Keep the low-support stream, replacing composition slots with more queries.

    Atoms and dummy slots are identical in every batch. A composition slot visits
    block0, block1, block2 on consecutive template epochs. Block0 contains the
    original query. Each expanded composition appears uniformly over three
    template epochs, without changing the relative weight of atoms or hop lengths.
    """

    def __init__(self, size, seed, *, atoms, base_support, multiplier):
        from .grokking_reproduction import MergedBatchStream

        self.size, self.atoms = size, atoms
        self.base_support, self.multiplier = base_support, multiplier
        self.template_size = atoms + 3 * base_support
        if size != atoms + 3 * base_support * multiplier or multiplier < 2:
            raise ValueError("Support stream dimensions disagree")
        self.template = MergedBatchStream(self.template_size, seed)
        self.template_epochs = 0

    def batch(self, batch_size):
        indices, count = self.template.batch(batch_size)
        result = np.full(batch_size, self.size, dtype=np.int64)
        valid = indices[:count]
        result[:count] = valid
        mask = valid >= self.atoms
        slots = valid[mask] - self.atoms
        category, offset = np.divmod(slots, self.base_support)
        result[np.flatnonzero(mask)] = (
            self.atoms
            + category * (self.base_support * self.multiplier)
            + (self.template_epochs % self.multiplier) * self.base_support
            + offset
        )
        if not len(self.template.remaining):
            self.template_epochs += 1
        return result, count

    def state_dict(self):
        return {
            "size": self.size,
            "atoms": self.atoms,
            "base_support": self.base_support,
            "multiplier": self.multiplier,
            "template_epochs": self.template_epochs,
            "template": self.template.state_dict(),
        }

    def load_state_dict(self, state):
        for key in ("size", "atoms", "base_support", "multiplier"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Support stream restoration differs: {key}")
        self.template_epochs = state["template_epochs"]
        self.template.load_state_dict(state["template"])


def matched_exposure_summary(counts, world, metadata, steps, batch_size):
    atoms = len(world["atoms"])
    support = metadata["base_train_compositions_per_hop"]
    template_size = atoms + 3 * support
    epochs, remaining = divmod(steps, math.ceil(template_size / batch_size))
    if counts.sum() != epochs * template_size + remaining * batch_size:
        raise AssertionError("Expanded support changed the baseline effective exposure")
    groups, end = {}, 0
    for name in ("atoms", "train_2", "train_3", "train_4"):
        begin, end = end, end + len(world[name])
        selected = counts[begin:end]
        if selected.max() - selected.min() > 1:
            raise AssertionError("Support sampling is not uniform within a category")
        groups[name] = int(selected.sum())
    actual = int(counts.sum())
    return {
        "examples": actual,
        "counts": groups,
        "atomic_epochs": groups["atoms"] / atoms,
        "composition_epochs": (actual - groups["atoms"]) / (len(counts) - atoms),
        "effective_input_tokens": groups["atoms"] * 3
        + sum(groups[f"train_{k}"] * (k + 2) for k in (2, 3, 4)),
        "supervised_tokens": actual * 2,
        "executed_input_tokens": steps * batch_size * metadata["sequence_length"],
        "masked_batch_slots": steps * batch_size - actual,
        "template_epochs": epochs,
        "unique_training_queries": len(counts),
    }
