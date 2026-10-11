"""World-paired analysis and independent W&B endpoint verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import netrc
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def world_scores(path, kind, phrasing):
    grouped = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["kind"] == kind and row["phrasing"] == phrasing:
            grouped.setdefault(row["world_id"], []).append(row["correct"])
    return {world: float(np.mean(scores)) for world, scores in grouped.items()}


def paired_difference(first, second, replicates=10000):
    if first.keys() != second.keys() or not first:
        raise ValueError("Paired worlds differ or are empty")
    worlds = sorted(first)
    difference = np.array([first[w] - second[w] for w in worlds])
    rng = np.random.default_rng(110317)
    values = [
        float(difference[rng.integers(0, len(worlds), len(worlds))].mean())
        for _ in range(replicates)
    ]
    return {
        "worlds": len(worlds),
        "difference_pp": float(difference.mean() * 100),
        "world_bootstrap_95_pp": (np.quantile(values, [0.025, 0.975]) * 100).tolist(),
    }


def paired_material_stream(first_path, second_path):
    def stream(path):
        return [
            (r["outer_updates"], r["material_id"], r["material_sha256"])
            for r in (json.loads(line) for line in Path(path).read_text().splitlines())
        ]

    first, second = stream(first_path), stream(second_path)
    if first != second or not first:
        raise ValueError("Actual paired training material streams differ")
    return len(first)


def common_atomic_subset(first_path, second_path):
    def records(path):
        return {
            (r["world_id"], r["query_id"]): r
            for r in (json.loads(line) for line in Path(path).read_text().splitlines())
        }

    first, second = records(first_path), records(second_path)
    if first.keys() != second.keys():
        raise ValueError("Conditional comparison query pools differ")
    kept, total, differences = 0, 0, {}
    for key, row in first.items():
        if row["kind"] != "compositional" or row["phrasing"] != "standard":
            continue
        total += 1
        prerequisites = [(row["world_id"], query) for query in row["prerequisites"]]
        if all(first[p]["correct"] and second[p]["correct"] for p in prerequisites):
            kept += 1
            differences.setdefault(row["world_id"], []).append(
                int(row["correct"]) - int(second[key]["correct"])
            )
    return {
        "covered_queries": kept,
        "all_queries": total,
        "coverage": kept / total,
        "covered_worlds": len(differences),
        "fact_prerequisites_correct_in_both_arms": True,
        "difference_pp_world_mean": float(np.mean([np.mean(v) for v in differences.values()]) * 100)
        if differences
        else None,
        "replaces_primary": False,
    }


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    args = parser.parse_args()
    base = args.base
    plan = read(base / "frozen/plan.json")
    analyses, tasks, subsets, baseline_effects = {}, {}, {}, {}
    paired_streams = {}
    seed_world_scores = []
    for seed in plan["training_seeds"]:
        paired_streams[str(seed)] = paired_material_stream(
            base / f"runs/compositional-seed{seed}/history.jsonl",
            base / f"runs/atomic-seed{seed}/history.jsonl",
        )
        subsets[str(seed)] = common_atomic_subset(
            base / f"runs/compositional-seed{seed}/evaluation/predictions.jsonl",
            base / f"runs/atomic-seed{seed}/evaluation/predictions.jsonl",
        )
        for kind in ("atomic_location", "atomic_owner", "compositional"):
            for phrasing in ("standard", "alternate"):
                first = world_scores(
                    base / f"runs/compositional-seed{seed}/evaluation/predictions.jsonl",
                    kind,
                    phrasing,
                )
                second = world_scores(
                    base / f"runs/atomic-seed{seed}/evaluation/predictions.jsonl", kind, phrasing
                )
                analyses[f"seed{seed}/{kind}/{phrasing}"] = paired_difference(
                    first, second, plan["bootstrap_replicates"]
                )
                if kind == "compositional" and phrasing == "standard":
                    seed_world_scores.append({w: first[w] - second[w] for w in first})
                fixed = world_scores(
                    base
                    / f"runs/fixed-seed{plan['training_seeds'][0]}/evaluation/predictions.jsonl",
                    kind,
                    phrasing,
                )
                for condition, scores in (("atomic", second), ("compositional", first)):
                    baseline_effects[f"seed{seed}/{condition}/{kind}/{phrasing}"] = (
                        paired_difference(scores, fixed, plan["bootstrap_replicates"])
                    )
    combined = {
        world: float(np.mean([scores[world] for scores in seed_world_scores]))
        for world in seed_world_scores[0]
    }
    primary = paired_difference(combined, dict.fromkeys(combined, 0), plan["bootstrap_replicates"])
    key = netrc.netrc("/root/.netrc").authenticators("api.wandb.ai")[2]
    api = wandb.Api(api_key=key, timeout=60)
    rows = [
        (condition, seed)
        for seed in plan["training_seeds"]
        for condition in ("atomic", "compositional")
    ]
    rows.append(("fixed", plan["training_seeds"][0]))
    for condition, seed in rows:
        name = f"{condition}-seed{seed}"
        root = base / "runs" / name
        local, audit = read(root / "science-complete.json"), read(root / "audit.json")
        if not local["passed"] or not audit["passed"]:
            raise ValueError("Scientific endpoint or replay is missing")
        cloud = api.run(f"zhusq20/llm-memory-editability/{local['wandb_run_id']}")
        if (
            cloud.state != "finished"
            or cloud.summary.get("outer_updates") != local["outer_updates"]
        ):
            raise ValueError("Cloud endpoint or update budget differs")
        if (
            cloud.config.get("initial_reader_sha256")
            != local["provenance"]["initial_reader_sha256"]
        ):
            raise ValueError("Cloud reader provenance differs")
        for group, metric in local["evaluation"]["metrics"].items():
            value = cloud.summary.get(f"eval/{group}/accuracy")
            if value is None or not math.isclose(
                value, metric["accuracy"], abs_tol=1e-12, rel_tol=1e-12
            ):
                raise ValueError("Cloud evaluation metric differs")
        tasks[name] = {
            "science_sha256": sha(root / "science-complete.json"),
            "audit_sha256": sha(root / "audit.json"),
            "wandb_url": local["wandb_url"],
            "cloud_finished": True,
            "outer_updates": local["outer_updates"],
        }
    result = {
        "passed": True,
        "primary": "compositional minus atomic on standard composed queries",
        "analysis_unit": "world; initialization seeds are not extra worlds",
        "paired_effects": analyses,
        "primary_effect": primary,
        "baseline_effects": baseline_effects,
        "common_atomic_prerequisite_subsets": subsets,
        "paired_material_stream_updates": paired_streams,
        "tasks": tasks,
        "scientific_training": True,
        "independent_replays": True,
        "cloud_verified": True,
        "new_outer_updates": sum(t["outer_updates"] for t in tasks.values()),
        "persistent_editing_tested": False,
    }
    (base / "completion-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": True, "tasks": len(tasks), "new_outer_updates": result["new_outer_updates"]}
        )
    )


if __name__ == "__main__":
    main()
