"""Zero-training re-reads of archived results for the 2026-10-10 roadmap audit.

Reads archived report files from the Hugging Face dataset at a pinned revision and
computes no model outputs. All three analyses are descriptive and post hoc; none
replaces a frozen contract.

1. write-target-confirmation-v1: per edit, whether the first-execution readout flipped
   to the new entity, derived-answer propagation D, and loss of parent-correct strict U
   compositions, with the same case's sham as baseline.
2. bridge-workspace-v1 comparison on grokking-reproduction-v1 weights: swap of the
   bridge input-embedding coordinates (x1, x2) and erasure of the bridge component at
   the r1 position after layer index 3 (end of the first loop; output of layer 4).
3. realworld-loop-addressability-v1: II/IO/OI/OO endpoint accuracy.

Usage: python scripts/reanalyze_history_20261010.py --cache <dir> [--output <json>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "zsqzz/llm-memory-editability"
REVISION = "6c580455ce1667274ea592c1c88f5d0fb1e5a190"
WRITE_TARGET = "results/write-target-confirmation-v1/report/summary.json"
BRIDGE = "docs/development-artifacts/bridge-workspace-v1/comparison/report/endpoints.csv"
ADDRESS = "results/realworld-loop-addressability-v1/development/{run}/endpoint.json"
ADDRESS_RUNS = ["standard8-shared", "standard8-unique", "loop4x2-shared", "loop4x2-unique"]
OBJECTIVES = ["final_ce", "early_ce", "early_ce_align"]
PANELS = ["test_ii", "test_io", "test_oi", "test_oo"]


def fetch(name, cache):
    path = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=cache)
    return Path(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)


def spearman(x, y, permutations=20000, seed=20261010):
    rho = pearson(ranks(x), ranks(y))
    if rho is None:
        return {"rho": None, "permutation_p": None, "n": len(x)}
    rng, shuffled, extreme = random.Random(seed), list(y), 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        value = pearson(ranks(x), ranks(shuffled))
        extreme += value is not None and abs(value) >= abs(rho) - 1e-12
    return {"rho": rho, "permutation_p": (extreme + 1) / (permutations + 1), "n": len(x)}


def write_target(path):
    records = json.loads(path.read_text())["case_records"]
    key = lambda r: (r["objective"], r["world"], r["initialization"], r["case_id"])  # noqa: E731
    sham = {key(r): r for r in records if r["arm"] == "sham"}
    edits = []
    for r in records:
        if r["arm"] != "edit":
            continue
        u, s = r["tasks"]["U_strict"], sham[key(r)]["tasks"]["U_strict"]
        edits.append(
            {
                "objective": r["objective"],
                "case": key(r)[1:],
                "flipped": r["early"]["new_entity_accuracy"],
                "early_probability": r["early"]["new_entity_probability"],
                "d_new": r["tasks"]["D_first_strict"]["correct"],
                "d_n": r["tasks"]["D_first_strict"]["n"],
                "sham_d_new": sham[key(r)]["tasks"]["D_first_strict"]["correct"],
                "u_parent_correct": u["parent_correct_n"],
                "damage": u["parent_correct_n"] - u["retained_parent_correct"],
                "sham_damage": s["parent_correct_n"] - s["retained_parent_correct"],
                "delta_l2": r["delta_l2"],
            }
        )
    out = {"flip": {}, "transfer_interference": {}}
    for group, rows in [("flipped", [e for e in edits if e["flipped"]])] + [
        ("not_flipped", [e for e in edits if not e["flipped"]])
    ]:
        out["flip"][group] = {
            "edits": len(rows),
            "objectives": sorted({e["objective"] for e in rows}),
            "d_new": sum(e["d_new"] for e in rows),
            "d_queries": sum(e["d_n"] for e in rows),
            "sham_d_new": sum(e["sham_d_new"] for e in rows),
            "edits_with_any_d": sum(e["d_new"] > 0 for e in rows),
            "edits_with_all_d": sum(e["d_new"] == e["d_n"] for e in rows),
            "early_probability_range": [
                min(e["early_probability"] for e in rows),
                max(e["early_probability"] for e in rows),
            ],
        }
    for objective in OBJECTIVES:
        rows = [e for e in edits if e["objective"] == objective]
        out["transfer_interference"][objective] = {
            "d_new": sum(e["d_new"] for e in rows),
            "d_queries": sum(e["d_n"] for e in rows),
            "damage": sum(e["damage"] for e in rows),
            "sham_damage": sum(e["sham_damage"] for e in rows),
            "u_parent_correct": sum(e["u_parent_correct"] for e in rows),
            "spearman_d_damage": spearman([e["d_new"] for e in rows], [e["damage"] for e in rows]),
            "spearman_l2_damage": spearman(
                [e["delta_l2"] for e in rows], [e["damage"] for e in rows]
            ),
            "spearman_l2_d": spearman([e["delta_l2"] for e in rows], [e["d_new"] for e in rows]),
        }
    centered_d, centered_damage = [], []
    for objective in OBJECTIVES:
        rows = [e for e in edits if e["objective"] == objective]
        md = sum(e["d_new"] for e in rows) / len(rows)
        mu = sum(e["damage"] for e in rows) / len(rows)
        centered_d += [e["d_new"] - md for e in rows]
        centered_damage += [e["damage"] - mu for e in rows]
    by_case = {}
    for e in edits:
        by_case.setdefault(e["case"], {})[e["objective"]] = e
    same, opposite, ties = 0, 0, 0
    for arms in by_case.values():
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            lo, hi = arms[OBJECTIVES[a]], arms[OBJECTIVES[b]]
            dd, du = hi["d_new"] - lo["d_new"], hi["damage"] - lo["damage"]
            if dd == 0 or du == 0:
                ties += 1
            elif (dd > 0) == (du > 0):
                same += 1
            else:
                opposite += 1
    out["transfer_interference"]["pooled"] = {
        "spearman_d_damage": spearman([e["d_new"] for e in edits], [e["damage"] for e in edits]),
        "spearman_d_damage_objective_centered": spearman(centered_d, centered_damage),
        "spearman_l2_damage": spearman(
            [e["delta_l2"] for e in edits], [e["damage"] for e in edits]
        ),
        "same_case_objective_pairs": {"same_direction": same, "opposite": opposite, "ties": ties},
    }
    return out


def bridge(path):
    rows = list(csv.DictReader(path.open()))
    cells = {
        "native_ii": ("native", "test_ii", "answer_accuracy", None),
        "native_io": ("native", "test_io", "answer_accuracy", None),
        "native_oi": ("native", "test_oi", "answer_accuracy", None),
        "swap_x1_ood_first": ("swap_embedding", "ood_id", "cf_answer_accuracy", "3"),
        "swap_x2_ood_first": ("swap_x2_embedding", "ood_id", "cf_answer_accuracy", "3"),
        "random_swap_ood_first": ("random_swap_embedding", "ood_id", "cf_answer_accuracy", "3"),
        "full_prefix_ood_first": ("full_prefix_embedding", "ood_id", "cf_answer_accuracy", "3"),
        "erase_bridge_ii": ("erase_state_embedding", "test_ii", "answer_accuracy", "3"),
        "erase_bridge_oi": ("erase_state_embedding", "test_oi", "answer_accuracy", "3"),
    }
    out = {}
    for run in sorted({r["run"] for r in rows if r["family"] == "reproduction"}):
        entry = {}
        for name, (condition, stratum, metric, layer) in cells.items():
            match = [
                r
                for r in rows
                if r["run"] == run
                and r["stratum"] == stratum
                and (
                    r["condition"] == "baseline"
                    if condition == "native"
                    else r["condition"] == condition and r["layer"] == layer
                )
            ]
            assert len(match) == 1, (run, name, len(match))
            entry[name] = {"accuracy": float(match[0][metric]), "n": int(match[0]["n_queries"])}
        out[run] = entry
    return out


def address(paths):
    return {
        run: {p: json.loads(path.read_text())[p]["answer_accuracy"] for p in PANELS}
        | {f"{p}_n": json.loads(path.read_text())[p]["n"] for p in PANELS}
        for run, path in paths.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sources = {}
    wt_path, sources[WRITE_TARGET] = fetch(WRITE_TARGET, args.cache)
    bw_path, sources[BRIDGE] = fetch(BRIDGE, args.cache)
    address_paths = {}
    for run in ADDRESS_RUNS:
        name = ADDRESS.format(run=run)
        address_paths[run], sources[name] = fetch(name, args.cache)
    result = {
        "repository": REPO,
        "revision": REVISION,
        "sources_sha256": sources,
        "write_target": write_target(wt_path),
        "bridge_workspace_layer3": bridge(bw_path),
        "addressability": address(address_paths),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
