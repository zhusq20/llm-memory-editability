"""Sequential-write interference grouped by address overlap (zero training).

Re-reads the per-step readouts of ce-cross-sample-double-new-v1 (three architectures) and
ce-cross-sample-ablation-v1 (six recipe variants). Each stream writes 64 new facts one at a time;
after every write all fixed old items and all new atoms were re-scored. For write k with fact
f_k = (h_k, r_k, t_k), the damage to item i is the change of its answer NLL from step k-1 to k.

Items are grouped by what they share with f_k:

atomic old items (h, r, t), from the two atomic pools and the mixed-query old companions:
  same_subject   h == h_k: identical state at the subject position (position 0)
  same_relation  r == r_k: same token at the answer position
  same_answer    t == t_k
  unrelated      none of the above, and h, t not in {h_k, t_k}
two-hop old items (h, r1, r2, t) with bridge b:
  first_subject  h == h_k
  bridge_subject b == h_k: the second hop reads the written subject
  first_relation r1 == r_k
  second_relation r2 == r_k
  unrelated      no shared entity or relation
earlier new atoms j < k: same_relation, same_subject, pair_partner, unrelated.

An answer-supervised write addressed by the question's last-position state should move
same-relation items most; a write addressed by the entity should move same-subject items most.
The within-item contrast compares the same item under related and unrelated writes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PERSISTENT = Path(
    "/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability"
)
WORLD = PERSISTENT / "ttcd-successful-base-v1/input-mount/data/ttcd-successful-base-v1/world.npz"
PANEL = Path("docs/development-artifacts/ce-cross-sample-double-new-v1/panel.json")
DOUBLE_NEW = "ce-cross-sample-double-new-v1/{}/output/evaluate/records"
ABLATION = "ce-cross-sample-ablation-v1/{}/output/science/records"
ARMS = {
    **{arm: DOUBLE_NEW.format(arm) for arm in ("transformer", "loop", "discoloop")},
    **{
        f"{arch}-{variant}": ABLATION.format(f"{arch}-{variant}")
        for arch in ("loop", "discoloop")
        for variant in ("lower-lr", "carry-adam", "no-bound")
    },
}
ATOMIC_POOLS = ("atomic_id", "atomic_ood")
CHAIN_POOLS = ("train_ii", "test_ii", "test_io", "test_oi", "test_oo")
ATOMIC_GROUPS = ("same_subject", "same_relation", "same_answer", "unrelated")
CHAIN_GROUPS = ("first_subject", "bridge_subject", "first_relation", "second_relation", "unrelated")
NEW_GROUPS = ("same_subject", "same_relation", "pair_partner", "unrelated")
BOOTSTRAP = 10000


def readout_arrays(record):
    """Return {kind: (rows, nll, correct)} with old atomic items deduplicated by fact."""
    out = {}
    old = record["readout"]["old"]
    companions = record["readout"]["mixed"]["old_companion"]
    seen, rows, nll, ok = set(), [], [], []
    for block in [old[p] for p in ATOMIC_POOLS] + [companions]:
        for row, x, c in zip(
            block["rows"], block["answer_nll"], block["final_correct"], strict=True
        ):
            key = tuple(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
            nll.append(x)
            ok.append(c)
    out["atomic"] = (rows, np.array(nll), np.array(ok, bool))
    rows, nll, ok = [], [], []
    for p in CHAIN_POOLS:
        block = old[p]
        rows += [tuple(r) for r in block["rows"]]
        nll += block["answer_nll"]
        ok += block["final_correct"]
    out["chain"] = (rows, np.array(nll), np.array(ok, bool))
    atoms = record["readout"]["atoms"]
    out["new"] = (
        [tuple(r) for r in atoms["rows"]],
        np.array(atoms["answer_nll"]),
        np.array(atoms["final_correct"], bool),
    )
    return out


def atomic_group(item, fact):
    h, r, t = item
    hk, rk, tk = fact
    if h == hk:
        return "same_subject"
    if r == rk:
        return "same_relation"
    if t == tk:
        return "same_answer"
    if {h, t} & {hk, tk}:
        return None
    return "unrelated"


def chain_group(item, bridge, fact):
    h, r1, r2, t = item
    hk, rk, tk = fact
    if h == hk:
        return "first_subject"
    if bridge == hk:
        return "bridge_subject"
    if r1 == rk:
        return "first_relation"
    if r2 == rk:
        return "second_relation"
    if {h, bridge, t} & {hk, tk}:
        return None
    return "unrelated"


def new_group(item, fact, partner):
    if item == fact:
        return None
    if item == partner:
        return "pair_partner"
    if item[0] == fact[0]:
        return "same_subject"
    if item[1] == fact[1]:
        return "same_relation"
    return "unrelated"


def group_stats(deltas, lost, eligible):
    d = np.array(deltas)
    n_ok = int(np.sum(eligible))
    return {
        "pairs": int(d.size),
        "mean_delta_nll": float(d.mean()) if d.size else None,
        "median_delta_nll": float(np.median(d)) if d.size else None,
        "fraction_delta_nll_above_1": float(np.mean(d > 1.0)) if d.size else None,
        "correct_before": n_ok,
        "lost": int(np.sum(lost)),
        "loss_rate": float(np.sum(lost) / n_ok) if n_ok else None,
    }


def within_item(per_item, group, rng):
    """Mean over items of [mean delta under `group` writes - mean delta under unrelated writes]."""
    diffs = []
    for groups in per_item.values():
        if groups.get(group) and groups.get("unrelated"):
            diffs.append(np.mean(groups[group]) - np.mean(groups["unrelated"]))
    if not diffs:
        return {"items": 0}
    diffs = np.array(diffs)
    boot = rng.choice(diffs, size=(BOOTSTRAP, diffs.size)).mean(axis=1)
    return {
        "items": int(diffs.size),
        "mean_difference": float(diffs.mean()),
        "ci95_item_bootstrap": [float(q) for q in np.quantile(boot, [0.025, 0.975])],
        "fraction_items_positive": float(np.mean(diffs > 0)),
    }


def endpoint_concentration(baseline, endpoint, new_facts):
    """Old atomic items lost between theta0 and the 64th write, by overlap with new facts."""
    subjects = {f[0] for f in new_facts}
    relations = {f[1] for f in new_facts}
    rows, _, ok0 = baseline["atomic"]
    _, _, ok1 = endpoint["atomic"]
    out = {
        g: {"correct_before": 0, "lost": 0}
        for g in ("shares_subject", "shares_relation", "neither")
    }
    for i, (h, r, _) in enumerate(rows):
        if not ok0[i]:
            continue
        g = (
            "shares_subject"
            if h in subjects
            else "shares_relation"
            if r in relations
            else "neither"
        )
        out[g]["correct_before"] += 1
        out[g]["lost"] += int(not ok1[i])
    for g in out.values():
        g["loss_rate"] = g["lost"] / g["correct_before"] if g["correct_before"] else None
    return out


def analyze_arm(record_dir, facts, partners, bridges, rng):
    acc = {
        kind: {g: {"d": [], "lost": [], "ok": []} for g in groups}
        for kind, groups in (
            ("atomic", ATOMIC_GROUPS),
            ("chain", CHAIN_GROUPS),
            ("new", NEW_GROUPS),
        )
    }
    per_item = {kind: defaultdict(lambda: defaultdict(list)) for kind in acc}
    baseline = readout_arrays(json.loads((record_dir / "baseline.json").read_text()))
    endpoints = {}
    for stream in ("forward", "reverse"):
        previous = baseline
        for step in range(1, 65):
            record = json.loads((record_dir / f"{stream}-{step:03d}.json").read_text())
            assert record["step"] == step and record["stream"] == stream
            fid = record["written"][-1]
            fact = facts[fid]
            written = {facts[w] for w in record["written"][:-1]}
            current = readout_arrays(record)
            for kind in acc:
                rows, nll, ok = current[kind]
                prev_rows, prev_nll, prev_ok = previous[kind]
                assert rows == prev_rows, (record_dir, stream, step, kind)
                for i, item in enumerate(rows):
                    if kind == "atomic":
                        g = atomic_group(item, fact)
                    elif kind == "chain":
                        g = chain_group(item, bridges[item[:2]], fact)
                    else:
                        if item not in written:
                            continue
                        g = new_group(item, fact, partners[fid])
                    if g is None:
                        continue
                    delta = float(nll[i] - prev_nll[i])
                    slot = acc[kind][g]
                    slot["d"].append(delta)
                    slot["ok"].append(bool(prev_ok[i]))
                    slot["lost"].append(bool(prev_ok[i] and not ok[i]))
                    per_item[kind][item][g].append(delta)
            previous = current
        endpoints[stream] = endpoint_concentration(baseline, previous, list(facts.values()))
    out = {"endpoint_old_atomic_loss": endpoints}
    for kind, groups in acc.items():
        out[kind] = {
            "groups": {g: group_stats(s["d"], s["lost"], s["ok"]) for g, s in groups.items()},
            "within_item_vs_unrelated": {
                g: within_item(per_item[kind], g, rng) for g in groups if g != "unrelated"
            },
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = json.loads(PANEL.read_text())
    facts, partners = {}, {}
    for pair in panel["pairs"]:
        a, b = pair["A"], pair["B"]
        facts[a["fact_id"]] = tuple(a["fact"])
        facts[b["fact_id"]] = tuple(b["fact"])
        partners[a["fact_id"]] = tuple(b["fact"])
        partners[b["fact_id"]] = tuple(a["fact"])
    world = np.load(WORLD)
    bridges = {(int(h), int(r)): int(t) for h, r, t in world["atoms"]}
    rng = np.random.default_rng(2026102001)
    result = {
        "schema": "sequential-write-interference-v1",
        "zero_training": True,
        "delta": "answer NLL at step k minus step k-1; positive = damage by write k",
        "unit": "(item, write) pair; within-item contrasts bootstrap over items",
        "sources": {arm: str(PERSISTENT / path) for arm, path in ARMS.items()},
        "arms": {},
    }
    for arm, path in ARMS.items():
        result["arms"][arm] = analyze_arm(PERSISTENT / path, facts, partners, bridges, rng)
        print(arm, "done", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
