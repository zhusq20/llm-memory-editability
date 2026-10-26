"""Summarize neighbor-protection-development-v1 against its reused random-protection controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_sequential_write_interference import PANEL, PERSISTENT, WORLD, analyze_arm

ROOT = Path(__file__).resolve().parents[1]
BATCH = "neighbor-protection-development-v1"
ARMS = ("discoloop-neighbor", "discoloop-mixed", "loop-neighbor", "loop-mixed")
CONTROLS = {
    "discoloop-random-no-bound": "discoloop-no-bound",
    "loop-random-no-bound": "loop-no-bound",
    "discoloop-random-bounded": "discoloop-control",
    "loop-random-bounded": "loop-control",
}


def pooled_old(summary, category):
    values = [
        summary["streams"][s]["endpoint"]["old"]["categories"][category][
            "originally_correct_retained"
        ]
        for s in ("forward", "reverse")
    ]
    correct, n = sum(v["correct"] for v in values), sum(v["n"] for v in values)
    return {"correct": correct, "n": n, "accuracy": correct / n if n else None}


def table_row(summary):
    primary = summary["primary_endpoint"]
    return {
        "new_atoms": primary["all_written_atoms"],
        "double_new": primary["double_new"],
        "old_atoms": pooled_old(summary, "atoms"),
        "old_compositions": pooled_old(summary, "compositions"),
    }


def pair_double_new(summary):
    return np.array(
        [
            np.mean([order["double_new"] for order in pair["orders"].values()])
            for pair in summary["pair_endpoint_values"]
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / BATCH)
    args = parser.parse_args()
    table, results = {}, {}
    comparison = json.loads(
        (ROOT / "results/ce-cross-sample-ablation-v1/comparison.json").read_text()
    )
    for name, key in CONTROLS.items():
        row = comparison["table"][key]
        table[name] = {
            k: row[k] for k in ("new_atoms", "double_new", "old_atoms", "old_compositions")
        }
    for arm in ARMS:
        results[arm] = json.loads((PERSISTENT / BATCH / arm / "result.json").read_text())
        table[arm] = table_row(results[arm])
    canonical = {
        name: f"ce-cross-sample-ablation-v1/{name}/output/science/result.json"
        for name in ("discoloop-no-bound", "loop-no-bound")
    }
    rng = np.random.default_rng(2026102012)
    differences = {}
    for arm in ARMS:
        control = "discoloop-no-bound" if arm.startswith("discoloop") else "loop-no-bound"
        base = json.loads((PERSISTENT / canonical[control]).read_text())
        diff = pair_double_new(results[arm]) - pair_double_new(base)
        boot = rng.choice(diff, size=(10000, diff.size)).mean(axis=1)
        differences[arm] = {
            "control": control,
            "double_new_difference": float(diff.mean()),
            "ci95_pair_bootstrap": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
        }
    panel = json.loads((ROOT / PANEL).read_text())
    facts, partners = {}, {}
    for pair in panel["pairs"]:
        a, b = pair["A"], pair["B"]
        facts[a["fact_id"]], facts[b["fact_id"]] = tuple(a["fact"]), tuple(b["fact"])
        partners[a["fact_id"]], partners[b["fact_id"]] = tuple(b["fact"]), tuple(a["fact"])
    world = np.load(WORLD)
    bridges = {(int(h), int(r)): int(t) for h, r, t in world["atoms"]}
    interference = {
        arm: analyze_arm(
            PERSISTENT / BATCH / arm / "records",
            facts,
            partners,
            bridges,
            np.random.default_rng(2026102013),
        )
        for arm in ARMS
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = {
        "schema": "neighbor-protection-summary-v1",
        "table": table,
        "double_new_vs_random_no_bound": differences,
        "interference": interference,
        "elapsed_seconds": {arm: results[arm]["elapsed_seconds"] for arm in ARMS},
        "protection_pool_sizes": {
            arm: results[arm]["protection_schedules"] for arm in ("discoloop-neighbor",)
        },
    }
    (args.out / "summary.json").write_text(json.dumps(out, indent=1))
    for name, row in table.items():
        print(
            f"{name:28s}",
            *(f"{k} {row[k]['correct']}/{row[k]['n']}" for k in row),
        )
    print(json.dumps(differences, indent=1))


if __name__ == "__main__":
    main()
