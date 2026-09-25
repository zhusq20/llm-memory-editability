"""Balanced fact-table interventions, defined independently of model training."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticWorld:
    """All entity/attribute queries and two interventions on identical support."""

    x: np.ndarray
    original: np.ndarray
    rule: np.ndarray
    exception: np.ndarray
    edit_mask: np.ndarray
    group_ids: np.ndarray
    n_entities: int
    n_attributes: int
    n_groups: int
    n_answers: int
    seed: int

    def task_exceptions(self, labels: np.ndarray) -> int:
        """Minimum overrides of a group+attribute table; not neural model degrees of freedom."""
        labels = np.asarray(labels)
        if labels.shape != self.original.shape:
            raise ValueError("labels must contain one answer per fact")
        exceptions = 0
        for group in range(self.n_groups):
            for attribute in range(self.n_attributes):
                cell = labels[(self.group_ids[self.x[:, 0]] == group) & (self.x[:, 1] == attribute)]
                exceptions += cell.size - int(np.bincount(cell, minlength=self.n_answers).max())
        return exceptions

    def to_metadata(self) -> dict[str, Any]:
        """JSON-compatible task diagnostics, available before training or editing."""
        return {
            "n_entities": self.n_entities,
            "n_attributes": self.n_attributes,
            "n_groups": self.n_groups,
            "n_answers": self.n_answers,
            "seed": self.seed,
            "n_facts": len(self.x),
            "n_edited": int(self.edit_mask.sum()),
            "edited_entities": np.unique(self.x[self.edit_mask, 0]).tolist(),
            "edited_attributes": np.unique(self.x[self.edit_mask, 1]).tolist(),
            "matched_controls": [
                "edit support",
                "edited fact count",
                "entity and attribute coverage",
                "per-entity target-answer histograms",
            ],
            "task_side_group_attribute_exceptions": {
                name: self.task_exceptions(getattr(self, name))
                for name in ("original", "rule", "exception")
            },
            "structure_metric_scope": (
                "Minimum label overrides of a deterministic group+attribute answer table. "
                "This is a task-side generator property, not neural degrees of freedom "
                "or evidence that a trained network learned these groups."
            ),
        }


def make_world(
    n_entities: int = 24,
    n_attributes: int = 8,
    n_groups: int = 3,
    n_answers: int = 4,
    seed: int = 0,
) -> SyntheticWorld:
    """Generate balanced labels and conflicting exception patterns in edited group 0.

    Entity targets are independently shuffled, conditioned on changing every edited
    fact. The final row is resampled if all entities happen to share one pattern.
    Rejection sampling is bounded, including for long attribute tables.
    """
    for name, value in {
        "n_entities": n_entities,
        "n_attributes": n_attributes,
        "n_groups": n_groups,
        "n_answers": n_answers,
        "seed": seed,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} is outside its valid range")
    n_entities, n_attributes, n_groups, n_answers, seed = map(
        int,
        (n_entities, n_attributes, n_groups, n_answers, seed),
    )
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2 to leave retained facts outside the edit support")
    if n_entities % n_groups or n_entities // n_groups < 2:
        raise ValueError("n_entities must be divisible by n_groups, with >= 2 entities per group")
    if n_answers < 3 or n_attributes % n_answers:
        raise ValueError("n_answers must be >= 3 and divide n_attributes")

    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), n_entities // n_groups).astype(np.int64)
    x = np.column_stack(
        (
            np.repeat(np.arange(n_entities), n_attributes),
            np.tile(np.arange(n_attributes), n_entities),
        )
    ).astype(np.int64)
    original = (group_ids[x[:, 0]] + x[:, 1]) % n_answers
    edit_mask = group_ids[x[:, 0]] == 0
    rule = original.copy()
    rule[edit_mask] = (rule[edit_mask] + 1) % n_answers
    exception = original.copy()
    # Every edited entity has the same balanced original labels and rule targets.
    old_row = original[:n_attributes]
    edited_entities = np.flatnonzero(group_ids == 0)
    sampled_rows = []
    for row_index, entity in enumerate(edited_entities):
        for _ in range(10_000):
            candidate = rng.permutation(old_row)
            shared_pattern = row_index == len(edited_entities) - 1 and all(
                np.array_equal(row, candidate) for row in sampled_rows
            )
            if np.all(candidate != old_row) and not shared_pattern:
                sampled_rows.append(candidate)
                start = entity * n_attributes
                exception[start : start + n_attributes] = candidate
                break
        else:
            raise ValueError(
                "Could not sample balanced exception patterns in 10,000 attempts; "
                "use more answers or fewer attributes to make rejection sampling practical."
            )
    return SyntheticWorld(
        x,
        original,
        rule,
        exception,
        edit_mask,
        group_ids,
        n_entities,
        n_attributes,
        n_groups,
        n_answers,
        seed,
    )
