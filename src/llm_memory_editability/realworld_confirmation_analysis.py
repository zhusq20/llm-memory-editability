"""Registered relation/component and paired-knowledge trajectory descriptions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .realworld_composition_data import write_json


def threshold_intervals(history, group, threshold):
    passed = [r["metrics"][group]["alias_em"] >= threshold for r in history]

    def interval(index):
        if index is None:
            return None
        return [history[index - 1]["step"] if index else None, history[index]["step"]]

    first = next((i for i, value in enumerate(passed) if value), None)
    sustained = next((i for i, value in enumerate(passed) if value and all(passed[i:])), None)
    return {
        "first_observed_interval": interval(first),
        "sustained_observed_interval": interval(sustained),
    }


def analyze_confirmation(config):
    root = Path(config["results_root"]) / config["phase"]
    data = json.loads(Path(config["data_file"]).read_text())
    held = {r["id"]: r for r in data["evaluation_compositions"]}
    endpoint_groups, thresholds, nodes = {}, {}, {}
    for spec in config["runs"]:
        name = spec["name"]
        out = root / name
        raw = json.loads((out / "endpoint-predictions.json").read_text())["test_all"]
        by_relation, by_component = defaultdict(list), defaultdict(list)
        for prediction in raw:
            row = held[prediction["id"]]
            by_relation[" -> ".join(e[1] for e in row["edges"])].append(prediction["alias_em"])
            by_component[row["group"]].append(prediction["alias_em"])
        endpoint_groups[name] = {
            key: {k: {"n": len(v), "alias_em": sum(v) / len(v)} for k, v in groups.items()}
            for key, groups in [
                ("relation_pair", by_relation),
                ("source_head_bridge_component", by_component),
            ]
        }
        history = json.loads((out / "learning.json").read_text())
        thresholds[name] = {
            f"{group}_{value}": threshold_intervals(history, group, value)
            for group, value in [("train_composition", 0.99), ("test_all", 0.5), ("test_all", 0.9)]
        }
        nodes[name] = {}
        for step in config["evaluation_nodes"]:
            node = json.loads((out / f"predictions-{step:07d}.json").read_text())
            nodes[name][step] = {
                group: {r["id"]: r["alias_em"] for r in node[group]}
                for group in ("atomic", "test_all")
            }
    trajectories = []
    for seed in sorted({s["initialization"] for s in config["runs"]}):
        specs = {s["architecture"]: s for s in config["runs"] if s["initialization"] == seed}
        loop = specs["loop4x2"]["name"]
        for baseline in ("standard8", "standard4"):
            base = specs[baseline]["name"]
            for step in config["evaluation_nodes"]:
                left, right = nodes[loop][step], nodes[base][step]
                assert set(left["test_all"]) == set(right["test_all"])
                for pool in ("all", "OO"):
                    selected = [
                        held[i]
                        for i in left["test_all"]
                        if pool == "all" or held[i]["role"] == pool
                    ]
                    common = [
                        r
                        for r in selected
                        if all(n["atomic"][a] for n in (left, right) for a in r["atom_ids"])
                    ]
                    trajectories.append(
                        {
                            "initialization": seed,
                            "baseline": baseline,
                            "step": step,
                            "pool": pool,
                            "panel_n": len(selected),
                            "both_models_necessary_atoms_correct_n": len(common),
                            "both_models_necessary_atoms_correct_coverage": len(common)
                            / len(selected),
                            "full_panel_paired_difference": sum(
                                left["test_all"][r["id"]] - right["test_all"][r["id"]]
                                for r in selected
                            )
                            / len(selected),
                            "conditional_paired_difference": sum(
                                left["test_all"][r["id"]] - right["test_all"][r["id"]]
                                for r in common
                            )
                            / len(common)
                            if common
                            else None,
                        }
                    )
    result = {
        "endpoint_groups": endpoint_groups,
        "threshold_intervals": thresholds,
        "paired_knowledge_trajectories": trajectories,
        "trajectory_scope": (
            "Fixed panels at registered nodes; threshold brackets are descriptive and do not "
            "identify exact crossing times or establish grokking."
        ),
        "endpoint_scope": (
            "Full registered evaluation pool; source components are shared by questions, "
            "not independent worlds."
        ),
    }
    write_json(Path(config["artifact_root"]) / "confirmation-analysis.json", result)
    return result
