"""Summarize the write-position decomposition and the composition-position readout."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from analyze_sequential_write_interference import (
    PANEL,
    PERSISTENT,
    WORLD,
    atomic_group,
    chain_group,
    new_group,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = PERSISTENT / "write-position-decomposition-20261020"
ARMS = ("transformer", "loop", "discoloop", "loop-no-bound", "discoloop-no-bound")


def regions_of(block):
    return [k for k in block if k != "rows"]


def summarize_arm(path, facts, partners, bridges):
    current = defaultdict(list)
    acc = defaultdict(lambda: defaultdict(lambda: {"d": [], "lost": [], "ok": []}))
    fact_ids = {v: k for k, v in facts.items()}
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        fact = tuple(entry["fact"])
        block = entry["current"]
        for region in regions_of(block):
            current[region].append(block[region][1][0])
        for kind in ("old_atomic", "chain", "earlier_new"):
            block = entry[kind]
            if not block:
                continue
            none_nll, none_ok = block["none"]
            for i, row in enumerate(block["rows"]):
                row = tuple(row)
                if kind == "old_atomic":
                    group = atomic_group(row, fact)
                elif kind == "chain":
                    group = chain_group(row, bridges[row[:2]], fact)
                else:
                    group = new_group(row, fact, partners[fact_ids[fact]])
                if group is None:
                    continue
                for region in regions_of(block):
                    if region == "none":
                        continue
                    nll, ok = block[region]
                    slot = acc[kind][(group, region)]
                    slot["d"].append(nll[i] - none_nll[i])
                    slot["ok"].append(bool(none_ok[i]))
                    slot["lost"].append(bool(none_ok[i] and not ok[i]))
    out = {"current_fact_correct": {r: float(np.mean(v)) for r, v in current.items()}}
    for kind, groups in acc.items():
        out[kind] = {}
        for (group, region), s in groups.items():
            d, ok = np.array(s["d"]), int(np.sum(s["ok"]))
            out[kind].setdefault(group, {})[region] = {
                "pairs": int(d.size),
                "mean_delta_nll": float(d.mean()),
                "loss_rate": float(np.sum(s["lost"]) / ok) if ok else None,
                "lost": int(np.sum(s["lost"])),
                "correct_before": ok,
            }
    return out


def summarize_composition(path):
    rows = json.loads(path.read_text())
    out = {}
    for order in ("A_then_B", "B_then_A"):
        part = [r for r in rows if r["order"] == order]
        out[order] = {}
        for name in ("atom", "double_new", "mixed"):
            regions = [k for k in part[0][name] if k != "rows"]
            out[order][name] = {
                region: {
                    "correct": int(sum(sum(r[name][region]["correct"]) for r in part)),
                    "n": int(sum(len(r[name][region]["correct"]) for r in part)),
                    "mean_nll": float(np.mean([x for r in part for x in r[name][region]["nll"]])),
                }
                for region in regions
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/write-position-20261020")
    args = parser.parse_args()
    panel = json.loads((ROOT / PANEL).read_text())
    facts, partners = {}, {}
    for pair in panel["pairs"]:
        a, b = pair["A"], pair["B"]
        facts[a["fact_id"]], facts[b["fact_id"]] = tuple(a["fact"]), tuple(b["fact"])
        partners[a["fact_id"]], partners[b["fact_id"]] = tuple(b["fact"]), tuple(a["fact"])
    world = np.load(WORLD)
    bridges = {(int(h), int(r)): int(t) for h, r, t in world["atoms"]}
    result = {
        "schema": "write-position-summary-v1",
        "zero_training": True,
        "regions": "none/full; subject = position 0; answer = answer position; "
        "middle = position 1 of two-hop queries",
        "decomposition": {},
        "composition": {},
        "checks": {},
    }
    for arm in ARMS:
        path = RUNS / "runs" / f"{arm}.jsonl"
        if (RUNS / "runs" / f"{arm}-check.json").exists():
            result["decomposition"][arm] = summarize_arm(path, facts, partners, bridges)
            result["checks"][arm] = json.loads((RUNS / "runs" / f"{arm}-check.json").read_text())
    for architecture in ("discoloop", "loop", "transformer"):
        path = RUNS / "composition" / f"{architecture}.json"
        if path.exists():
            result["composition"][architecture] = summarize_composition(path)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
