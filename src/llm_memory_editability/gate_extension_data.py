"""Frozen data adapters for the registered Learning Gate extension.

Confirmation delegates to the existing permutation generator unchanged. Wang
reuses its archived graph and seven evaluation pools byte for byte. The length
extension trains only atoms and two-hop queries, and shares test relation
templates across the two graphs. Model-side NoPE is a separate required control.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from . import loop_hop_data as permutation

KINDS = ("permutation_confirmation", "wang", "depth_extrapolation")
WANG_PANELS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")
DEPTH_PANELS = (
    "atomic_id",
    "atomic_ood",
    "train_2",
    "id_2",
    "ood_2",
    "id_3",
    "ood_3",
    "id_4",
    "ood_4",
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _normalized(metadata):
    result = dict(metadata)
    kind = result.setdefault("kind", "permutation_confirmation")
    if kind not in KINDS:
        raise ValueError(f"Unknown extension data kind: {kind}")
    hops = (2, 3, 4) if kind == "permutation_confirmation" else (2,)
    result["training_groups"] = ["atoms", *(f"train_{hop}" for hop in hops)]
    result["training_hops"] = [1, *hops]
    result["max_training_hops"] = max(hops)
    result["panel_names"] = list(result["panels"])
    result["primary_score"] = "accuracy" if kind == "wang" else "answer_accuracy"
    result["sequence_length"] = 4 if kind == "wang" else 6
    return result


def prepare(config, destination):
    """Create a new immutable data directory; never overwrite a previous batch."""
    kind = config["data"]["kind"]
    destination = Path(destination)
    if kind == "permutation_confirmation":
        metadata = permutation.prepare(config, destination)
        metadata = _normalized(metadata)
        _write_json(destination / "complete.json", metadata)
    elif kind == "wang":
        _prepare_wang(config["data"], destination)
    elif kind == "depth_extrapolation":
        _prepare_depth(config["data"], destination)
    else:
        raise ValueError(f"Unknown extension data kind: {kind}")
    world, panels, metadata = load(destination)
    _write_json(destination / "audit.json", audit_data(world, panels, metadata))
    return metadata


def _prepare_wang(spec, destination):
    source = Path(spec["source_directory"])
    original = json.loads((source / "complete.json").read_text())
    expected = {
        "entities": 2000,
        "relations": 200,
        "degree": 20,
        "base_vocabulary": 50257,
        "vocab_size": 52463,
        "entity_offset": 50257,
        "relation_offset": 52257,
        "pad_token": 50256,
        "end_marker": 52460,
        "atomic_id": 38000,
        "atomic_ood": 2000,
    }
    for key, value in expected.items():
        if original.get(key) != value:
            raise ValueError(f"Archived Wang contract mismatch for {key}")
    if float(spec.get("phi", 7.2)) != 7.2:
        raise ValueError("The registered Wang support is phi=7.2")
    for name in ("world", "panels"):
        if permutation.sha256(source / f"{name}.npz") != original[f"{name}_sha256"]:
            raise ValueError(f"Archived Wang {name} hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("world.npz", "panels.npz"):
        shutil.copyfile(source / name, destination / name)
    shutil.copyfile(source / "complete.json", destination / "source-complete.json")
    metadata = _normalized(
        {
            **original,
            "kind": "wang",
            "phi": 7.2,
            "atomic_examples": 40000,
            "train_compositions": 273600,
            "training_examples": 313600,
            "source_directory": str(source),
            "source_complete_sha256": permutation.sha256(source / "complete.json"),
            "primary": "full-vocabulary answer plus generated EOS; answer accuracy also reported",
            "task_status": "original archived Wang world and seven pools, phi=7.2 support",
        }
    )
    _write_json(destination / "complete.json", metadata)


def _prepare_depth(spec, destination):
    n, r = spec["entities_per_graph"], spec["relations"]
    count, target = spec["evaluation_templates"], spec["train_compositions_per_hop"]
    if min(n, r, count, target, spec["training_panel_size"]) <= 0:
        raise ValueError("Graph and support sizes must be positive")
    if count >= r**2 or target > n * (r**2 - count):
        raise ValueError("Requested support exhausts the reserved two-hop query space")
    rng = np.random.default_rng(spec["world_seed"])
    permutations = np.stack([[rng.permutation(n) for _ in range(r)] for _ in range(2)])
    atoms = np.array(
        [
            (graph * n + head, rel, graph * n + permutations[graph, rel, head])
            for graph in range(2)
            for head in range(n)
            for rel in range(r)
        ],
        dtype=np.int64,
    )
    world = {"atoms": atoms, "permutations": permutations}
    panels = {"atomic_id": atoms[: n * r], "atomic_ood": atoms[n * r :]}
    for hop in (2, 3, 4):
        codes = rng.choice(r**hop, count, replace=False)
        templates = np.array([np.unravel_index(int(code), (r,) * hop) for code in codes])
        relations = np.repeat(templates, n, axis=0)
        heads = np.tile(np.arange(n), count)
        for graph, split in ((0, "id"), (1, "ood")):
            nodes = permutation.traverse(heads, relations, permutations[graph])
            panels[f"{split}_{hop}"] = np.column_stack(
                (heads + graph * n, relations, nodes[:, -1] + graph * n)
            )
        if hop != 2:
            continue
        forbidden = {tuple(row[:-1]) for row in panels["id_2"]}
        selected, seen = [], set()
        while len(selected) < target:
            key = (int(rng.integers(n)), *map(int, rng.integers(r, size=2)))
            if key in forbidden or key in seen:
                continue
            selected.append(key)
            seen.add(key)
        queries = np.asarray(selected, dtype=np.int64)
        nodes = permutation.traverse(queries[:, 0], queries[:, 1:], permutations[0])
        world["train_2"] = np.column_stack((queries, nodes[:, -1]))
        take = rng.choice(target, min(target, spec["training_panel_size"]), replace=False)
        panels["train_2"] = world["train_2"][take]
    panels = {name: panels[name] for name in DEPTH_PANELS}
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "world.npz", **world)
    np.savez_compressed(destination / "panels.npz", **panels)
    metadata = _normalized(
        {
            **spec,
            "entity_offset": 3,
            "relation_offset": 3 + 2 * n,
            "pad_token": 0,
            "end_marker": 2,
            "vocab_size": 3 + 2 * n + r,
            "atomic_examples": len(atoms),
            "training_examples": len(atoms) + target,
            "world_sha256": permutation.sha256(destination / "world.npz"),
            "panels_sha256": permutation.sha256(destination / "panels.npz"),
            "panels": {name: len(rows) for name, rows in panels.items()},
            "shared_evaluation_templates": True,
            "required_position_encoding": "none",
            "primary": "full-vocabulary answer argmax; answer plus generated EOS also reported",
            "task_status": "two-hop-only training; heldout length three and four; NoPE required",
        }
    )
    _write_json(destination / "complete.json", metadata)


def load(destination):
    """Load verified arrays; derive Wang support without altering archived bytes."""
    root = Path(destination)
    metadata = _normalized(json.loads((root / "complete.json").read_text()))
    values = []
    for name in ("world", "panels"):
        path = root / f"{name}.npz"
        if permutation.sha256(path) != metadata[f"{name}_sha256"]:
            raise ValueError(f"Frozen {name} hash mismatch")
        with np.load(path, allow_pickle=False) as packed:
            values.append({key: packed[key] for key in packed.files})
    world, panels = values
    if metadata["kind"] == "wang":
        support = metadata["train_compositions"]
        world["train_2"] = world["chains"][world["train_order"][:support]]
        lookup = np.full((metadata["entities"], metadata["relations"]), -1, dtype=np.int64)
        lookup[world["atoms"][:, 0], world["atoms"][:, 1]] = np.arange(len(world["atoms"]))
        world["edge_lookup"] = lookup
    if set(panels) != set(metadata["panel_names"]):
        raise ValueError("Panel names disagree with frozen metadata")
    return world, panels, metadata


def pack_rows(rows, metadata):
    rows = np.asarray(rows)
    if rows.ndim != 2 or not 3 <= rows.shape[1] <= metadata["sequence_length"]:
        raise ValueError("Rows must fit the frozen sequence length")
    return permutation.pack_rows(rows, metadata)


def path_details(rows, metadata, world):
    """Return entity trajectories and indices into the complete atomic table."""
    rows = np.asarray(rows)
    if metadata["kind"] != "wang":
        return permutation.path_details(rows, metadata, world["permutations"])
    atoms, lookup = world["atoms"], world["edge_lookup"]
    hops = rows.shape[1] - 2
    nodes = np.empty((len(rows), hops + 1), dtype=np.int64)
    edges = np.empty((len(rows), hops), dtype=np.int64)
    nodes[:, 0] = rows[:, 0]
    for hop in range(hops):
        edges[:, hop] = lookup[nodes[:, hop], rows[:, hop + 1]]
        if np.any(edges[:, hop] < 0):
            raise ValueError("Query uses a missing Wang atomic fact")
        nodes[:, hop + 1] = atoms[edges[:, hop], -1]
    if not np.array_equal(nodes[:, -1], rows[:, -1]):
        raise ValueError("Path target disagrees with its atomic facts")
    return nodes, edges


def premise_edges(rows, metadata, world):
    return path_details(rows, metadata, world)[1]


def audit_data(world, panels, metadata):
    """Validate truth, test isolation, training scope and premise index coverage."""
    kind = metadata["kind"]
    if kind == "permutation_confirmation":
        result = permutation.audit_data(world, panels, metadata)
    elif kind == "depth_extrapolation":
        result = _audit_depth(world, panels, metadata)
    elif kind == "wang":
        result = _audit_wang(world, panels, metadata)
    else:
        raise ValueError(f"Unknown extension data kind: {kind}")
    if (
        sum(len(world[name]) for name in metadata["training_groups"])
        != metadata["training_examples"]
    ):
        raise ValueError("Training group sizes disagree with frozen metadata")
    coverage = {}
    for name, rows in panels.items():
        nodes, edges = path_details(rows, metadata, world)
        if not np.array_equal(world["atoms"][edges, -1], nodes[:, 1:]):
            raise ValueError("Premise indices disagree with trajectories")
        coverage[name] = {
            "queries": len(rows),
            "required_atomic_facts": int(np.unique(edges).size),
            "premise_index_coverage": 1.0,
        }
    result["premise_coverage"] = coverage
    result["conditional_evaluation_requires_all_atoms"] = len(world["atoms"])
    return result


def _audit_depth(world, panels, metadata):
    n, r = metadata["entities_per_graph"], metadata["relations"]
    if set(world) != {"atoms", "permutations", "train_2"}:
        raise ValueError("Depth extrapolation may train only atoms and two-hop queries")
    if set(panels) != set(DEPTH_PANELS):
        raise ValueError("Incorrect depth-extrapolation panel matrix")
    perms = world["permutations"]
    if perms.shape != (2, r, n) or not np.all(np.sort(perms, axis=-1) == np.arange(n)):
        raise ValueError("A relation is not a permutation")
    atoms = world["atoms"]
    if len(atoms) != 2 * n * r:
        raise ValueError("Incomplete atomic training coverage")
    if not np.array_equal(atoms[:, 0] * r + atoms[:, 1], np.arange(len(atoms))):
        raise ValueError("Atomic table order disagrees with premise indexing")
    train = world["train_2"]
    if train.shape[1] != 4 or np.any(train[:, [0, -1]] >= n):
        raise ValueError("Graph B composition leaked into training")
    support = {tuple(row[:-1]) for row in train}
    if len(support) != len(train):
        raise ValueError("Training queries are not unique")
    if support & {tuple(row[:-1]) for row in panels["id_2"]}:
        raise ValueError("ID test query leaked into training")
    if not {tuple(row[:-1]) for row in panels["train_2"]} <= support:
        raise ValueError("Training probe is outside the training support")
    for hop in (2, 3, 4):
        if not np.array_equal(panels[f"id_{hop}"][:, 1:-1], panels[f"ood_{hop}"][:, 1:-1]):
            raise ValueError("Graph A/B evaluation templates must match")
        for split, graph in (("id", 0), ("ood", 1)):
            rows = panels[f"{split}_{hop}"]
            _, groups = np.unique(rows[:, 1:-1], axis=0, return_inverse=True)
            if len(np.unique(groups)) != metadata["evaluation_templates"]:
                raise ValueError("Incorrect evaluation template count")
            for group in np.unique(groups):
                selected = rows[groups == group]
                expected = np.arange(n) + graph * n
                if not np.array_equal(np.sort(selected[:, 0]), expected):
                    raise ValueError("Evaluation heads are not balanced")
                if not np.array_equal(np.sort(selected[:, -1]), expected):
                    raise ValueError("Suffix-only answer shortcut detected")
    for rows in (atoms, train):
        path_details(rows, metadata, world)
    for name, expected in (("atomic_id", atoms[: n * r]), ("atomic_ood", atoms[n * r :])):
        if not np.array_equal(panels[name], expected):
            raise ValueError("Atomic panels must retain complete graph coverage")
    return {
        "passed": True,
        "balanced_suffix_only_best_accuracy": 1 / n,
        "graph_b_composition_training_examples": 0,
        "three_or_four_hop_training_examples": 0,
        "shared_evaluation_templates": True,
    }


def _audit_wang(world, panels, metadata):
    if set(panels) != set(WANG_PANELS):
        raise ValueError("Wang must retain the original seven evaluation pools")
    atoms, chains = world["atoms"], world["chains"]
    first, second, ood = world["first"], world["second"], world["ood"]
    if len(atoms) != 40000 or len(np.unique(atoms[:, :2], axis=0)) != len(atoms):
        raise ValueError("Incorrect Wang atomic graph")
    if not np.all(np.bincount(atoms[:, 0], minlength=2000) == 20):
        raise ValueError("Wang degree differs from the original recipe")
    if not np.array_equal(atoms[first, 2], atoms[second, 0]):
        raise ValueError("Disconnected Wang chains")
    expected_chains = np.column_stack(
        (atoms[first, 0], atoms[first, 1], atoms[second, 1], atoms[second, 2])
    )
    if not np.array_equal(chains, expected_chains):
        raise ValueError("Wang chains disagree with atomic truth")
    if not np.array_equal(world["kind"], ood[first] * 2 + ood[second]):
        raise ValueError("Wang II/IO/OI/OO classification mismatch")
    selected = world["train_order"][: metadata["train_compositions"]]
    if len(selected) != 273600 or not np.array_equal(world["train_2"], chains[selected]):
        raise ValueError("Incorrect phi=7.2 Wang training support")
    if np.any(world["kind"][selected] != 0):
        raise ValueError("OOD composition leaked into Wang training")
    if np.intersect1d(selected, world["test_ii"]).size:
        raise ValueError("Heldout II leaked into training")
    train_keys = {tuple(row[:-1]) for row in world["train_2"]}
    if len(train_keys) != len(selected):
        raise ValueError("Wang training queries are not unique")
    for name, rows in panels.items():
        _, edges = path_details(rows, metadata, world)
        if name.startswith("atomic_"):
            expected_ood = name == "atomic_ood"
            if not np.all(ood[edges[:, 0]] == expected_ood):
                raise ValueError("Wang atomic panel split mismatch")
        else:
            actual_kind = ood[edges[:, 0]] * 2 + ood[edges[:, 1]]
            expected_kind = {"train_ii": 0, "test_ii": 0, "test_io": 1, "test_oi": 2, "test_oo": 3}[
                name
            ]
            if np.any(actual_kind != expected_kind):
                raise ValueError("Wang evaluation pool split mismatch")
            query_keys = {tuple(row[:-1]) for row in rows}
            if name == "train_ii" and not query_keys <= train_keys:
                raise ValueError("Wang training probe is outside phi=7.2 support")
            if name != "train_ii" and query_keys & train_keys:
                raise ValueError("Wang test query leaked into training")
    return {
        "passed": True,
        "ood_composition_training_examples": 0,
        "archived_pool_rows_preserved": True,
    }


def make_table(world, metadata, device):
    import torch

    packed = [pack_rows(world[name], metadata) for name in metadata["training_groups"]]
    padding = (
        np.full((1, metadata["sequence_length"]), metadata["pad_token"], dtype=np.int64),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[-100, -100]], dtype=np.int64),
    )
    return tuple(
        torch.as_tensor(np.concatenate(parts), device=device)
        for parts in zip(*packed, padding, strict=True)
    )


def exposure_summary(counts, world, metadata, steps, batch_size):
    counts = np.asarray(counts)
    if counts.shape != (metadata["training_examples"],) or np.any(counts < 0):
        raise AssertionError("Invalid training exposure counts")
    epochs, remaining = divmod(steps, math.ceil(len(counts) / batch_size))
    if counts.sum() != epochs * len(counts) + remaining * batch_size:
        raise AssertionError("Exposure does not match the merged shuffled sample stream")
    if counts.max() - counts.min() > 1:
        raise AssertionError("Merged epoch stream exposure became imbalanced")
    end, groups, tokens = 0, {}, 0
    for name in metadata["training_groups"]:
        begin, end = end, end + len(world[name])
        groups[name] = int(counts[begin:end].sum())
        tokens += groups[name] * world[name].shape[1]
    actual = int(counts.sum())
    return {
        "examples": actual,
        "counts": groups,
        "atomic_epochs": groups["atoms"] / len(world["atoms"]),
        "composition_epochs": (actual - groups["atoms"]) / (len(counts) - len(world["atoms"])),
        "effective_input_tokens": tokens,
        "supervised_tokens": actual * 2,
        "executed_input_tokens": steps * batch_size * metadata["sequence_length"],
        "masked_batch_slots": steps * batch_size - actual,
    }
