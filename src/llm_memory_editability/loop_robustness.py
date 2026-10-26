"""Data-distribution and depth diagnostics for the unchanged invariant model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import loop_hop_data as original


def rewrite_function_world(source, destination, seed):
    """Keep every query/split; replace permutation relations by deterministic maps."""
    world, panels, metadata = original.load(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    n, r = metadata["entities_per_graph"], metadata["relations"]
    functions = np.random.default_rng(seed).integers(n, size=(2, r, n))

    def relabel(rows):
        rows = rows.copy()
        graph = rows[:, 0] // n
        head = rows[:, 0] % n
        for column in range(1, rows.shape[1] - 1):
            head = functions[graph, rows[:, column], head]
        rows[:, -1] = graph * n + head
        return rows

    world = {k: relabel(v) for k, v in world.items() if k != "permutations"}
    world["permutations"] = functions  # Historical key denotes the relation lookup table.
    panels = {k: relabel(v) for k, v in panels.items()}
    np.savez_compressed(destination / "world.npz", **world)
    np.savez_compressed(destination / "panels.npz", **panels)
    metadata.update(
        relation_family="random_deterministic_function",
        function_seed=seed,
        query_source_world_sha256=metadata["world_sha256"],
        world_sha256=original.sha256(destination / "world.npz"),
        panels_sha256=original.sha256(destination / "panels.npz"),
        task_status="Controlled nonbijective relation extension; query splits unchanged",
    )
    audit = audit_function_world(world, panels, metadata)
    (destination / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return metadata


def audit_function_world(world, panels, metadata):
    n, r = metadata["entities_per_graph"], metadata["relations"]
    relation = world["permutations"]
    if relation.shape != (2, r, n) or np.any((relation < 0) | (relation >= n)):
        raise ValueError("Invalid deterministic relation table")
    if set(panels) != set(original.panel_names()):
        raise ValueError("Incomplete panel matrix")
    expected_atoms = np.array(
        [
            (g * n + h, t, g * n + relation[g, t, h])
            for g in range(2)
            for h in range(n)
            for t in range(r)
        ]
    )
    if not np.array_equal(world["atoms"], expected_atoms):
        raise ValueError("Atomic facts do not match relation table")
    for k in (2, 3, 4):
        train = world[f"train_{k}"]
        keys = {tuple(row[:-1]) for row in train}
        if len(keys) != len(train) or np.any(train[:, [0, -1]] >= n):
            raise ValueError("Duplicate query or graph-B composition in training")
        if keys & {tuple(row[:-1]) for row in panels[f"id_{k}"]}:
            raise ValueError("Held-out query leaked into training")
        if not {tuple(row) for row in panels[f"train_{k}"]} <= {tuple(row) for row in train}:
            raise ValueError("Training panel is not drawn from training rows")
        for split, graph in (("id", 0), ("ood", 1)):
            rows = panels[f"{split}_{k}"]
            if np.any(rows[:, 0] // n != graph):
                raise ValueError("Incorrect graph membership")
            _, group = np.unique(rows[:, 1:-1], axis=0, return_inverse=True)
            for i in np.unique(group):
                if not np.array_equal(np.sort(rows[group == i, 0]), np.arange(n) + graph * n):
                    raise ValueError("Every relation template must cover every head")
    diagnostics = {}
    for name, rows in {**panels, **{k: v for k, v in world.items() if k != "permutations"}}.items():
        nodes, edges = original.path_details(rows, metadata, relation)
        if not np.array_equal(world["atoms"][edges, -1], nodes[:, 1:]):
            raise ValueError("Composition is not supported by trained atomic facts")
        _, group = np.unique(rows[:, 1:-1], axis=0, return_inverse=True)
        best = sum(
            np.bincount(rows[group == i, -1], minlength=2 * n).max() for i in np.unique(group)
        )
        diagnostics[name] = {
            "n": len(rows),
            "constant_oracle_accuracy": float(np.bincount(rows[:, -1]).max() / len(rows)),
            "relation_sequence_only_oracle_accuracy": float(best / len(rows)),
        }
    return {"passed": True, "panels": diagnostics, "graph_b_composition_training_examples": 0}


def nested_indices(name, size, count, seed):
    """Deterministic, model-independent nested subsets; no process-randomized hash."""
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(seed ^ salt).permutation(size)[: min(count, size)]


def summarize_predictions(prediction, truth, reference_round=8, entity_range=None):
    """Per-round outcomes, including transient loss that endpoint accuracy hides."""
    correct = prediction == truth[None]
    if not 1 <= reference_round <= len(prediction):
        raise ValueError("Reference round not present")
    reference = correct[reference_round - 1]
    ever = np.maximum.accumulate(correct, axis=0)
    lost_after_solved = np.maximum.accumulate(ever & ~correct, axis=0)
    ref_lost = np.maximum.accumulate(~correct[reference_round - 1 :], axis=0) & reference
    result = {
        "n": len(truth),
        "rounds": len(prediction),
        "reference_round": reference_round,
        "answer_accuracy": correct.mean(1).tolist(),
        "ever_correct": ever.mean(1).tolist(),
        "ever_lost_after_solved": lost_after_solved.mean(1).tolist(),
        "reference_correct": int(reference.sum()),
        "reference_wrong": int((~reference).sum()),
        "reference_correct_now_wrong": ((~correct) & reference).sum(1).tolist(),
        "reference_wrong_now_correct": (correct & ~reference).sum(1).tolist(),
        "reference_correct_ever_lost_after_reference": ref_lost.sum(1).tolist(),
        "prediction_changed": [0.0, *(prediction[1:] != prediction[:-1]).mean(1).tolist()],
        "first_correct_round": np.where(ever[-1], correct.argmax(0) + 1, -1).tolist(),
        "eos_fraction": (prediction == 2).mean(1).tolist(),
    }
    if entity_range is not None:
        low, high = entity_range
        result["nonentity_fraction"] = ((prediction < low) | (prediction >= high)).mean(1).tolist()
    return result


def stream_predictions(model, tokens, positions, rounds):
    """Yield one round's argmax at a time, with constant device-state storage."""
    import torch

    with torch.no_grad():
        state = None
        for _ in range(rounds):
            logits, state = model.run(tokens, positions, rounds=1, state=state, carry=True)
            yield logits[:, 0].argmax(-1)
