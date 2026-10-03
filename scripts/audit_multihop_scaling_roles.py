"""Describe composition roles and audit paired exposures without loading a model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc
from llm_memory_editability.multihop_scaling import (
    EVALUATION_SPLITS,
    TRAIN_SPLITS,
    build_world,
    run_name,
)
from llm_memory_editability.storage_composition import file_hash


def independent_paths(rows, atoms):
    lookup = {
        (int(head), int(relation)): (index, int(tail))
        for index, (head, relation, tail) in enumerate(atoms)
    }
    paths = np.empty((len(rows), rows.shape[1] - 1), dtype=np.int64)
    facts = np.empty((len(rows), rows.shape[1] - 2), dtype=np.int64)
    for i, row in enumerate(rows):
        paths[i, 0] = current = int(row[0])
        for j, relation in enumerate(row[1:-1]):
            facts[i, j], current = lookup[current, int(relation)]
            paths[i, j + 1] = current
        if current != int(row[-1]):
            raise AssertionError("Graph traversal disagrees with saved answer")
    return paths, facts


def counts_summary(counts, id_mask):
    return {
        "counts_by_fact": counts.tolist(),
        "total_fact_occurrences": int(counts.sum()),
        "id_facts_covered": int((counts[id_mask] > 0).sum()),
        "id_facts_total": int(id_mask.sum()),
        "ood_facts_covered": int((counts[~id_mask] > 0).sum()),
        "ood_facts_total": int((~id_mask).sum()),
    }


def summarize(config_path, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    config = json.loads(Path(config_path).read_text())
    datasets, exposures, failures = {}, [], []
    paired_atomic = defaultdict(list)
    for spec in config["specs"]:
        data_key = f"w{spec['world_seed']}-phi{spec['phi']:g}"
        world = build_world(spec)
        atoms = world["atomic"]
        id_mask = np.asarray(world["metadata"]["id_mask"], dtype=bool)
        fact_edges = {}
        for name in EVALUATION_SPLITS:
            fact_edges[name] = independent_paths(world[name], atoms)[1]
        if data_key not in datasets:
            trained = np.zeros(len(atoms), dtype=np.int64)
            pools = {}
            for name in EVALUATION_SPLITS:
                rows = world[name]
                paths, facts = independent_paths(rows, atoms)
                counts = np.bincount(facts.ravel(), minlength=len(atoms))
                if name.startswith("train_"):
                    trained += counts
                pools[name] = {
                    "n": len(rows),
                    "facts": counts_summary(counts, id_mask),
                    "fact_roles": [
                        counts_summary(np.bincount(facts[:, j], minlength=len(atoms)), id_mask)
                        for j in range(facts.shape[1])
                    ],
                    "entity_roles_covered": [
                        int(len(np.unique(paths[:, j]))) for j in range(paths.shape[1])
                    ],
                    "repeated_entity_queries": sum(len(set(path)) < len(path) for path in paths),
                }
            for hop in (2, 3, 4):
                familiar = fact_edges[f"familiar_{hop}"]
                strict = fact_edges[f"strict_{hop}"]
                pools[f"familiar_{hop}"]["all_facts_with_composition_role"] = int(
                    (trained[familiar] > 0).all(axis=1).sum()
                )
                pools[f"strict_{hop}"]["any_fact_with_composition_role"] = int(
                    (trained[strict] > 0).any(axis=1).sum()
                )
                if (trained[strict] > 0).any():
                    raise AssertionError("Strict facts acquired composition training role")
            datasets[data_key] = {
                "dataset_sha256": world["metadata"]["dataset_sha256"],
                "common_evaluation_sha256": world["metadata"]["common_evaluation_sha256"],
                "composition_fact_occurrences": counts_summary(trained, id_mask),
                "pools": pools,
            }
        folder = Path(config["results"]) / run_name(spec)
        for stage in ("train", "audit"):
            status_path = folder / f"{stage}-process-status.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status["returncode"] != 0:
                    failures.append({"run": folder.name, **status})
        for node in spec["nodes"]:
            path = folder / f"exposures-{node:06d}.npz"
            if not path.exists():
                continue
            with np.load(path) as saved:
                if set(saved.files) != set(TRAIN_SPLITS):
                    raise AssertionError("Saved exposure strata differ")
                total_roles = np.zeros(len(atoms), dtype=np.int64)
                for name in TRAIN_SPLITS:
                    counts = saved[name]
                    if (
                        counts.shape != (len(world[name]),)
                        or counts.dtype != np.int64
                        or counts.sum() != node * 32
                        or np.ptp(counts) > 1
                    ):
                        raise AssertionError("Unbalanced or mismatched sampled exposures")
                    np.add.at(
                        total_roles,
                        fact_edges[name].ravel(),
                        np.repeat(counts, fact_edges[name].shape[1]),
                    )
                key = spec["world_seed"], spec["initialization"], node
                paired_atomic[key].append((folder.name, saved["atomic"].copy()))
                exposures.append(
                    {
                        "run": folder.name,
                        "step": node,
                        "facts_by_actual_atomic_and_composite_presentations": counts_summary(
                            total_roles, id_mask
                        ),
                    }
                )
    paired = []
    for (world, initialization, node), values in paired_atomic.items():
        first = values[0][1]
        if any(not np.array_equal(first, value) for _, value in values):
            raise AssertionError("Atomic exposure sequence differs between paired conditions")
        paired.append(
            {
                "world": world,
                "initialization": initialization,
                "step": node,
                "available_conditions": len(values),
                "identical_atomic_counts": True,
            }
        )
    record = {
        "utc": utc(),
        "scope": "Descriptive roles and exposure checks; no performance selection",
        "config_sha256": file_hash(config_path),
        "source_sha256": file_hash(__file__),
        "datasets": datasets,
        "actual_exposure_nodes": exposures,
        "paired_atomic_checks": paired,
        "process_failures": failures,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "datasets": len(datasets),
                "exposure_nodes": len(exposures),
                "failures": failures,
                "out": str(out),
            }
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.config, args.out)


if __name__ == "__main__":
    main()
