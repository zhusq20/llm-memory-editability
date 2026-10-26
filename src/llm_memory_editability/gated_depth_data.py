"""Fixed-slot permutation queries; all atoms, A-only short compositions.

Reuses the project's permutation traversal. Query identity ignores NOOP layout.
One-hop tests intentionally repeat trained facts; only k>=2 are held-out queries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .loop_hop_data import sha256, traverse


def pack(records, metadata, rng=None, slots=None, layout="random"):
    slots = metadata["slots"] if slots is None else slots
    rng = np.random.default_rng(metadata["layout_seed"]) if rng is None else rng
    tokens = np.full((len(records), metadata["slots"] + 2), metadata["pad_token"], np.int64)
    tokens[:, 1 : slots + 1] = metadata["noop_token"]
    tokens[:, slots + 1] = metadata["query_token"]
    for i, row in enumerate(records):
        k = row["hop_count"]
        if k > slots:
            raise ValueError("Too few relation slots")
        if layout == "left":
            positions = np.arange(k)
        elif layout == "right":
            positions = np.arange(slots - k, slots)
        elif layout == "random":
            positions = np.sort(rng.choice(slots, k, replace=False))
        else:
            raise ValueError("Unknown layout")
        tokens[i, 0] = row["start_entity"] + metadata["entity_offset"]
        tokens[i, positions + 1] = np.asarray(row["relation_ids"]) + metadata["relation_offset"]
    positions = np.full((len(records), 1), slots + 1, np.int64)
    labels = np.array([r["answer_entity"] + metadata["entity_offset"] for r in records])
    return tokens, positions, labels


def query_key(row):
    return row["start_entity"], tuple(row["relation_ids"])


def generate(config):
    spec = config["data"]
    n, relations = spec["entities_per_graph"], spec["relations"]
    rng = np.random.default_rng(spec["world_seed"])
    permutations = np.stack([[rng.permutation(n) for _ in range(relations)] for _ in range(2)])
    pools = {
        name: []
        for name in (
            "facts_train",
            "comp_A_train_k1to5",
            "comp_A_val_k1to5",
            "eval_A_short",
            "eval_A_long",
            "eval_B_short",
            "eval_B_long",
            "train_probe",
        )
    }

    def records(graph, sequences, split):
        sequence = np.repeat(np.asarray(sequences), n, axis=0)
        heads = np.tile(np.arange(n), len(sequences))
        nodes = traverse(heads, sequence, permutations[graph]) + graph * n
        return [
            {
                "example_id": f"{split}:{graph}:{int(h)}:" + ".".join(map(str, rel)),
                "graph_id": "AB"[graph],
                "start_entity": int(h),
                "relation_ids": rel.tolist(),
                "hop_count": len(rel),
                "answer_entity": int(path[-1]),
                "intermediate_entities": path.tolist(),
                "split": split,
                "rng_seed": spec["world_seed"],
            }
            for h, rel, path in zip(nodes[:, 0], sequence, nodes, strict=True)
        ]

    for graph in (0, 1):
        rows = records(graph, [[r] for r in range(relations)], "facts_train")
        pools["facts_train"].extend(rows)
        pools[f"eval_{'AB'[graph]}_short"].extend(
            records(graph, [[r] for r in range(relations)], f"eval_{'AB'[graph]}_short")
        )
    # k=1 is fully covered fact supervision, explicitly not unseen-query validation.
    pools["comp_A_train_k1to5"].extend(
        records(0, [[r] for r in range(relations)], "comp_A_train_k1to5")
    )
    pools["comp_A_val_k1to5"].extend(
        records(0, [[r] for r in range(relations)], "comp_A_val_k1to5")
    )
    for k in range(2, 6):
        train_count = spec["train_templates"][str(k)]
        count = spec["evaluation_templates"]
        if train_count + 2 * count > relations**k:
            raise ValueError("Insufficient disjoint sequence space")
        codes = rng.choice(relations**k, train_count + 2 * count, replace=False)
        sequences = [list(np.unravel_index(int(c), (relations,) * k)) for c in codes]
        # Convert numpy scalars before JSON serialization.
        sequences = np.asarray(sequences, np.int64)
        train = records(0, sequences[:train_count], "comp_A_train_k1to5")
        pools["comp_A_train_k1to5"].extend(train)
        pools["train_probe"].extend(train[: count * n])
        pools["comp_A_val_k1to5"].extend(
            records(0, sequences[train_count : train_count + count], "comp_A_val_k1to5")
        )
        pools["eval_A_short"].extend(records(0, sequences[-count:], "eval_A_short"))
        pools["eval_B_short"].extend(records(1, sequences[-count:], "eval_B_short"))
    for k in spec["long_hops"]:
        count = spec["evaluation_templates"]
        unique = set()
        while len(unique) < count:
            unique.add(tuple(map(int, rng.integers(relations, size=k))))
        for graph in (0, 1):
            name = f"eval_{'AB'[graph]}_long"
            pools[name].extend(records(graph, sorted(unique), name))
    metadata = {
        **spec,
        "entity_offset": 3,
        "relation_offset": 3 + 2 * n,
        "noop_token": 3 + 2 * n + relations,
        "vocab_size": 4 + 2 * n + relations,
        "pad_token": 0,
        "query_token": 1,
        "unused_eos_token": 2,
        "one_hop_overlap": "intentional trained-fact recall, excluded from composition novelty",
        "checkpoint_selection": (
            "macro A validation k=1..5 R=5 accuracy; lower CE then earliest tie"
        ),
    }
    audit = audit_data(pools, permutations, metadata)
    return pools, permutations, metadata, audit


def audit_data(pools, permutations, metadata):
    n, r = metadata["entities_per_graph"], metadata["relations"]
    for graph in permutations:
        for mapping in graph:
            assert np.array_equal(np.sort(mapping), np.arange(n))
    assert len(pools["facts_train"]) == 2 * n * r
    facts = {query_key(row): row["answer_entity"] for row in pools["facts_train"]}
    assert len(facts) == 2 * n * r
    distributions = {}
    for name, rows in pools.items():
        keys = [query_key(row) for row in rows]
        assert len(keys) == len(set(keys))
        counts = {}
        for row in rows:
            current = row["start_entity"]
            path = [current]
            for relation in row["relation_ids"]:
                current = facts[current, (relation,)]
                path.append(current)
            assert path == row["intermediate_entities"] and current == row["answer_entity"]
            assert row["hop_count"] == len(row["relation_ids"])
            assert all(p // n == row["start_entity"] // n for p in path)
            key = f"{row['graph_id']}-k{row['hop_count']}"
            counts.setdefault(key, np.zeros(n, dtype=int))[current % n] += 1
        distributions[name] = {
            key: {"min": int(v.min()), "max": int(v.max())} for key, v in counts.items()
        }
        assert all(v.min() == v.max() for v in counts.values())
    train = {query_key(row) for row in pools["comp_A_train_k1to5"] if row["hop_count"] > 1}
    val = {query_key(row) for row in pools["comp_A_val_k1to5"] if row["hop_count"] > 1}
    test = {query_key(row) for row in pools["eval_A_short"] if row["hop_count"] > 1}
    assert not (train & val or train & test or val & test)
    assert all(
        row["graph_id"] == "A" and row["hop_count"] <= 5 for row in pools["comp_A_train_k1to5"]
    )
    return {
        "passed": True,
        "composition_query_overlaps": 0,
        "atom_coverage": len(facts),
        "answer_frequency": distributions,
        "counts": {k: len(v) for k, v in pools.items()},
    }


def prepare(config, destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    pools, permutations, metadata, audit = generate(config)
    for name, rows in pools.items():
        with (root / f"{name}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    np.save(root / "permutations.npy", permutations)
    arrays = {}
    for name, rows in pools.items():
        for key, values in zip(
            ("tokens", "positions", "labels"), pack(rows, metadata), strict=True
        ):
            arrays[f"{name}__{key}"] = values
    controls = [row for row in pools["eval_A_short"] if row["hop_count"] == 3]
    for slots in (8, 12, 20):
        for layout in ("left", "right", "random"):
            name = f"noop_k3_slots{slots}_{layout}"
            for key, values in zip(
                ("tokens", "positions", "labels"),
                pack(controls, metadata, slots=slots, layout=layout),
                strict=True,
            ):
                arrays[f"{name}__{key}"] = values
    np.savez_compressed(root / "panels.npz", **arrays)
    metadata["sha256"] = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return metadata


def load(destination):
    root = Path(destination)
    metadata = json.loads((root / "metadata.json").read_text())
    for name, expected in metadata["sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Data changed: {name}")
    pools = {
        p.stem: [json.loads(line) for line in p.read_text().splitlines()]
        for p in root.glob("*.jsonl")
    }
    return pools, dict(np.load(root / "panels.npz")), metadata


class BalancedStream:
    """Uniform hop sampling; k1 half A/half B, k2..5 only A; random NOOPs."""

    def __init__(self, pools, metadata, seed):
        self.rng = np.random.default_rng(seed)
        self.metadata = metadata
        self.groups = {1: pools["facts_train"]}
        for k in range(2, 6):
            self.groups[k] = [row for row in pools["comp_A_train_k1to5"] if row["hop_count"] == k]

    def batch(self, size):
        hops = self.rng.integers(1, 6, size=size)
        rows = [self.groups[int(k)][int(self.rng.integers(len(self.groups[int(k)])))] for k in hops]
        return pack(rows, self.metadata, self.rng), rows
