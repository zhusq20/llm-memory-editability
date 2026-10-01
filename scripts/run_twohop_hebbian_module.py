#!/usr/bin/env python3
"""Two continuous Hebbian-style MLP calls on a public benchmark's fact graph.

Synthetic entity geometry and engineered composition are explicit. No natural language
or pretrained Transformer is evaluated. Only atomic facts fit the memory weights.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/twohop-hebbian-v1"
OUT = ROOT / "docs/development-artifacts/twohop-hebbian-v1"
RAW = ROOT / "results/twohop-hebbian-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def graph():
    cases = json.loads((DATA / "mquake-candidates.json").read_text())
    mappings = defaultdict(set)
    for row in cases:
        for s, r, o in row["original_triples"]:
            mappings[(s, r)].add(o)
    ambiguous = {key for key, values in mappings.items() if len(values) > 1}
    eligible = [
        row
        for row in cases
        if all((edge[0], edge[1]) not in ambiguous for edge in row["original_triples"])
    ]
    chains = sorted({tuple(tuple(edge) for edge in row["original_triples"]) for row in eligible})
    facts = sorted({edge for chain in chains for edge in chain})
    entities = sorted({value for s, _, o in facts for value in [s, o]})
    relations = sorted({r for _, r, _ in facts})
    audit = dict(
        source_cases=len(cases),
        ambiguous_keys=len(ambiguous),
        excluded_cases=len(cases) - len(eligible),
        retained_cases=len(eligible),
        unique_chains=len(chains),
        unique_atomic_facts=len(facts),
        entities=len(entities),
        relations=len(relations),
        no_compositional_labels_used_for_fit=True,
        note="Atomic facts include evaluation-chain constituent facts by design. "
        "This is transductive fact composition, not unseen atomic knowledge.",
    )
    return facts, chains, entities, relations, audit


def run(cfg, seed, width, facts, chains, entities, relations):
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    e = normalize(rng.normal(size=(len(entities), cfg["entity_dimension"])))
    r = normalize(rng.normal(size=(len(relations), cfg["relation_dimension"])))
    eids = {value: i for i, value in enumerate(entities)}
    rids = {value: i for i, value in enumerate(relations)}

    def key(entity_vectors, relation_ids):
        return np.concatenate([entity_vectors, r[relation_ids]], axis=-1) / np.sqrt(2)

    subjects = np.array([eids[s] for s, _, _ in facts])
    predicates = np.array([rids[p] for _, p, _ in facts])
    objects = np.array([eids[o] for _, _, o in facts])
    first = np.array([eids[chain[0][0]] for chain in chains])
    bridge = np.array([eids[chain[0][2]] for chain in chains])
    target = np.array([eids[chain[1][2]] for chain in chains])
    r1 = np.array([rids[chain[0][1]] for chain in chains])
    r2 = np.array([rids[chain[1][1]] for chain in chains])
    dimension = cfg["entity_dimension"] + cfg["relation_dimension"]
    # Separate seeds make geometry identical across widths and keep all widths fixed.
    features_rng = np.random.default_rng(seed + 100000)
    matrices = features_rng.normal(size=(width, 2, dimension))
    up, gate = matrices[:, 0], matrices[:, 1]

    def phi(q):
        return ((q @ up.T) * (q @ gate.T)) / np.sqrt(width)

    features = phi(key(e[subjects], predicates))
    if width <= len(facts):
        weight = np.linalg.solve(
            features.T @ features + cfg["ridge"] * np.eye(width), features.T @ e[objects]
        )
    else:
        weight = features.T @ np.linalg.solve(
            features @ features.T + cfg["ridge"] * np.eye(len(facts)), e[objects]
        )
    atomic_output = features @ weight
    atom_correct = np.argmax(atomic_output @ e.T, axis=1) == objects
    u = phi(key(e[first], r1)) @ weight
    first_correct = (u @ e.T).argmax(axis=1) == bridge
    ideal_q = key(e[bridge], r2)
    ideal_features = phi(ideal_q)
    ideal_output = ideal_features @ weight
    clean_logits = ideal_output @ e.T
    second_correct = clean_logits.argmax(axis=1) == target
    target_index = np.arange(len(chains))
    clean_gaps = clean_logits[target_index, target, None] - clean_logits
    # For a fixed B and all candidate value vectors: w_j = B^T(v_c-v_j).
    all_readouts = e @ weight.T
    squared = np.sum(all_readouts**2, axis=1)
    wnorm = np.sqrt(
        np.maximum(
            squared[target, None] + squared[None, :] - 2 * all_readouts[target] @ all_readouts.T, 0
        )
    )
    statistics = []
    for interface in cfg["interfaces"]:
        q = key(u if interface == "raw" else normalize(u), r2)
        query_features = phi(q)
        output = query_features @ weight
        logits = output @ e.T
        correct = logits.argmax(axis=1) == target
        delta = q - ideal_q
        delta_features = query_features - ideal_features
        lower_gaps = clean_gaps - wnorm * np.linalg.norm(delta_features, axis=1)[:, None]
        lower_gaps[target_index, target] = np.inf
        certified = lower_gaps.min(axis=1) > 1e-10
        false_certificates = int(np.sum(certified & ~correct))
        if false_certificates:
            raise AssertionError("A sufficient feature certificate certified an incorrect case")
        # Exact bilinear finite-perturbation expansion, evaluated before candidate decoding.
        exact_delta_features = (
            (ideal_q @ up.T) * (delta @ gate.T)
            + (delta @ up.T) * (ideal_q @ gate.T)
            + (delta @ up.T) * (delta @ gate.T)
        ) / np.sqrt(width)
        predicted_delta_logits = (exact_delta_features @ weight) @ e.T
        residual = float(np.max(np.abs(logits - clean_logits - predicted_delta_logits)))
        if residual > 1e-8:
            raise AssertionError(f"Finite-perturbation identity failed: {residual}")
        mastered = first_correct & second_correct
        statistics.append(
            dict(
                seed=seed,
                hidden_width=width,
                interface=interface,
                atoms=len(facts),
                chains=len(chains),
                atomic_accuracy=float(atom_correct.mean()),
                first_hop_accuracy=float(first_correct.mean()),
                clean_second_hop_accuracy=float(second_correct.mean()),
                twohop_accuracy=float(correct.mean()),
                mastered_chains=int(mastered.sum()),
                twohop_given_both_atoms=float(correct[mastered].mean()) if mastered.any() else None,
                certified_chains=int(certified.sum()),
                false_certificates=false_certificates,
                exact_expansion_max_error=residual,
                mean_query_error=float(np.linalg.norm(delta, axis=1).mean()),
                mean_feature_error=float(np.linalg.norm(delta_features, axis=1).mean()),
                fit_and_evaluation_seconds=time.perf_counter() - started,
            )
        )
        RAW.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            RAW / f"seed-{seed}-width-{width}-{interface}.npz",
            first=first,
            bridge=bridge,
            target=target,
            predicted=logits.argmax(axis=1),
            correct=correct,
            mastered=mastered,
            certified=certified,
            delta=delta,
            lower_margin=lower_gaps.min(axis=1),
        )
    np.savez_compressed(
        RAW / f"seed-{seed}-width-{width}-memory.npz",
        up=up,
        gate=gate,
        weight=weight,
        entity_codes=e,
        relation_codes=r,
    )
    return statistics


def main():
    cfg_path = ROOT / "configs/twohop-hebbian-v1.json"
    cfg = json.loads(cfg_path.read_text())
    facts, chains, entities, relations, audit = graph()
    lock = dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        config=cfg,
        config_sha256=digest(cfg_path),
        script_sha256=digest(Path(__file__)),
        source_manifest_sha256=digest(DATA / "mquake-candidates.json"),
        graph=audit,
        numpy_version=np.__version__,
        dtype="float64",
        results_scope=cfg["scope"],
    )
    write(OUT / "module-lock.json", lock)
    write(
        RAW / "graph.json", dict(facts=facts, chains=chains, entities=entities, relations=relations)
    )
    statistics = []
    for seed in cfg["seeds"]:
        for width in cfg["hidden_widths"]:
            rows = run(cfg, seed, width, facts, chains, entities, relations)
            statistics.extend(rows)
            print(
                json.dumps(
                    {
                        key: rows[0][key]
                        for key in ["seed", "hidden_width", "atomic_accuracy", "twohop_accuracy"]
                    }
                ),
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "module-results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    write(
        OUT / "module-results.json",
        dict(
            state="complete",
            completed_utc=datetime.now(timezone.utc).isoformat(),
            scope=cfg["scope"],
            graph=audit,
            rows=statistics,
            raw_artifacts={p.name: digest(p) for p in sorted(RAW.glob("*.npz"))},
        ),
    )


if __name__ == "__main__":
    main()
