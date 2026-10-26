"""Summarize use-adapter-development-v1 with the pre-registered comparisons.

Facts are the unit: rates are fact means of template means; paired comparisons bootstrap facts
(the E4 rule: an improvement is a paired 95% lower bound above 0). The E4 matrix keeps, as E4
did, facts whose U0 is answered correctly in that arm and, for non-country operations, values
that passed the operation screen. Here facts are further limited to the 50 E3-written facts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from summarize_read_interface_v2 import (
    GRADERS,
    answer_types,
    boot_diff,
    classify_op,
    e4_judgment,
    fact_mean,
    mean,
    per_fact,
)

MODELS = ("qwen3-1.7b", "qwen3.5-2b")
SETS = ("dev", "second")
ARMS = ("p0", "pu", "pp")
OPS = ("continent", "language", "country", "currency")
SPLITS = ("adapt", "test", "test_shared_subject", "test_own_subject")
PRIMARY = [f"test/{op}/{mode}" for op in ("country", "currency") for mode in ("direct", "cot")]


def read(path):
    return json.loads(Path(path).read_text())


def matrix(facts, order, practiced_subjects, passes, ops, uses):
    """E4 cells over the written subset of `order`: answer types, fact-mean correctness, and
    per-fact correct / named-v rates for paired comparisons."""
    written = {r["fid"] for r in uses if r["use"] == "U0" and r["strict"] and r["fid"] in order}
    rows = [r for r in ops if r["fid"] in written]
    rows += [{**r, "op": "country", "golds": [r["gold"]]} for r in uses
             if r["fid"] in written and r["use"].startswith("U2")]  # fmt: skip
    cells, correct, named_v = {}, {}, {}
    for split in SPLITS:
        for op in OPS:
            for mode in ("direct", "cot"):
                sel = []
                for r in rows:
                    f = facts[r["fid"]]
                    if r.get("op") != op or r["mode"] != mode:
                        continue
                    if op != "country" and not passes[f["v"]].get(op):
                        continue
                    if split == "adapt" and f["split"] != "adapt":
                        continue
                    if split.startswith("test") and f["split"] != "test":
                        continue
                    if split == "test_shared_subject" and f["h"] not in practiced_subjects:
                        continue
                    if split == "test_own_subject" and f["h"] in practiced_subjects:
                        continue
                    sel.append(r)
                if not sel:
                    continue
                key = f"{split}/{op}/{mode}"
                kinds = Counter(classify_op(r, facts[r["fid"]], op) for r in sel)
                n = sum(kinds.values())
                cells[key] = {"facts": len({r["fid"] for r in sel}),
                              **{k: v / n for k, v in sorted(kinds.items())}}  # fmt: skip
                cells[key]["correct"] = fact_mean(sel, lambda r: bool(r["strict"]))[0]
                correct[key] = per_fact(sel, lambda r: bool(r["strict"]))
                named_v[key] = per_fact(
                    sel, lambda r, op=op: classify_op(r, facts[r["fid"]], op) == "named_v"
                )
    return {"written": len(written), "cells": cells}, correct, named_v


def uses_summary(uses, facts, order):
    rows = [r for r in uses if r["fid"] in order]
    out = {}
    out["U0"] = fact_mean([r for r in rows if r["use"] == "U0"], lambda r: r["strict"])[0]
    out["U1"] = fact_mean([r for r in rows if r["use"].startswith("U1")], lambda r: r["strict"])[0]
    for use in ("U2", "U3"):
        for mode in ("direct", "cot"):
            sel = [r for r in rows if r["use"].startswith(use) and r["mode"] == mode]
            if use == "U2":
                out[f"U2_{mode}"] = answer_types(sel, facts)
            else:
                out[f"U3_{mode}"] = fact_mean(sel, lambda r: r["strict"])[0]
                out[f"U3_{mode}_types"] = answer_types(sel, facts)
    short = [r for r in rows if r["use"] == "shortcut"]
    out["shortcut"] = fact_mean(short, lambda r: r["strict"])[0]
    out["facts"] = len({r["fid"] for r in rows})
    return out


def natural_retention(natural, base_natural):
    """Kept share of the released model's correct natural answers, by group."""
    base = {r["prompt"]: r["strict"] for r in base_natural}
    groups = defaultdict(list)
    for r in natural:
        if not base.get(r["prompt"]):
            continue
        qid = r.get("qid", "")
        group = "two_hop" if qid.startswith("nat2") else r.get("group", "other")
        groups[group].append(r["strict"])
    return {g: {"kept": mean(v), "n": len(v)} for g, v in sorted(groups.items())}


def model_set(root, v2_root, model, fact_set):
    run = root / fact_set / model
    v2 = v2_root / fact_set / model
    facts = {f["fid"]: f for f in read(run / "facts.json")["facts"]}
    e3 = read(v2 / "e3" / "structural.json")
    order = set(e3["order"])
    practiced = {fid for fid in order if facts[fid]["split"] == "adapt"}
    practiced_subjects = {facts[fid]["h"] for fid in practiced}
    passes = defaultdict(dict)
    for r in read(run / "e4" / "screen.json")["rows"]:
        passes[r["v"]][r["op"]] = r["strict"]
    base_natural = read(run / "eval" / "base.json")["natural"]
    status = read(run / "status.json")
    out = {
        "e3_lr": e3["lr"],
        "e3_trainable_parameters": e3["trainable_parameters"],
        "e3_facts": len(order),
        "practiced_facts": len(practiced),
        "test_facts": sum(facts[fid]["split"] == "test" for fid in order),
        "test_facts_shared_subject": sum(
            facts[fid]["split"] == "test" and facts[fid]["h"] in practiced_subjects for fid in order
        ),  # fmt: skip
        "p0_vs_saved_e3": status.get("p0_vs_saved_e3"),
        "arms": {},
    }
    per_c, per_v = {}, {}
    for arm in ARMS:
        if not (run / "ops" / f"{arm}.json").exists():
            continue
        ev = read(run / "eval" / f"{arm}.json")
        uses = ev["uses"]
        ops = read(run / "ops" / f"{arm}.json")["records"]
        m, per_c[arm], per_v[arm] = matrix(facts, order, practiced_subjects, passes, ops, uses)
        m["uses"] = uses_summary(uses, facts, order)
        m["retention"] = {
            "unrelated_300": mean(r["strict"] for r in read(run / "unrelated" / f"{arm}.json")
                                  ["records"]),
            "natural": natural_retention(ev["natural"], base_natural),
        }  # fmt: skip
        if (run / "train" / f"{arm}.json").exists():
            t = read(run / "train" / f"{arm}.json")
            m["cost"] = {k: t[k] for k in t if k != "history"}
            m["cost"]["u0_teacher_forced_last"] = t["history"][-1].get("u0_teacher_forced")
        out["arms"][arm] = m
    # E4 references from knowledge-read-interface-v2 (E4 practiced every F_adapt fact)
    adapt_subjects = {f["h"] for f in facts.values() if f["split"] == "adapt"}
    v2_passes = defaultdict(dict)
    for r in read(v2 / "e4" / "screen.json")["rows"]:
        v2_passes[r["v"]][r["op"]] = r["strict"]
    all_main = {fid for fid, f in facts.items() if f["split"] != "calibration"}
    refs = {}
    for tag in ("e4u", "e4p"):
        ops = read(v2 / "e4" / f"ops-{tag}.json")["records"]
        uses = read(v2 / "eval" / f"{tag}.json")["uses"]
        same, per_c[f"ref_{tag}"], per_v[f"ref_{tag}"] = matrix(
            facts, order, adapt_subjects, v2_passes, ops, uses
        )
        full, _, _ = matrix(facts, all_main, adapt_subjects, v2_passes, ops, uses)
        refs[tag] = {"e3_facts": same, "all_facts": full}
    out["e4_references"] = refs
    comp = {}
    if "pu" in per_c and "pp" in per_c:
        paired = {k: boot_diff(per_c["pu"][k], per_c["pp"].get(k, {}))
                  for k in per_c["pu"] if k in per_c["pp"]}  # fmt: skip
        comp["pu_minus_pp"] = paired
        comp["primary"] = {
            k: {**(paired.get(k) or {}),
                "improvement": bool((paired.get(k) or {}).get("lo", 0) > 0)}
            for k in PRIMARY
        }  # fmt: skip
        comp["e4_table_judgment"] = {mode: e4_judgment(paired, mode) for mode in ("direct", "cot")}
    for other in ("p0", "ref_e4u"):
        if "pu" in per_c and other in per_c:
            comp[f"pu_minus_{other}"] = {
                k: boot_diff(per_c["pu"][k], per_c[other][k])
                for k in per_c["pu"] if k in per_c[other]
            }  # fmt: skip
    if "pu" in per_c and "p0" in per_c:
        key = "test/country/direct"
        pu, p0 = out["arms"]["pu"]["cells"], out["arms"]["p0"]["cells"]
        e4u = refs["e4u"]["e3_facts"]["cells"]
        comp["direct_test_country"] = {
            "pu": pu.get(key, {}).get("correct"),
            "p0": p0.get(key, {}).get("correct"),
            "e4u_same_facts": e4u.get(key, {}).get("correct"),
            "e4u_all_facts": refs["e4u"]["all_facts"]["cells"].get(key, {}).get("correct"),
        }
        c = comp["direct_test_country"]
        c["pu_exceeds_both"] = bool(c["pu"] is not None and c["pu"] > (c["p0"] or 0)
                                    and c["pu"] > (c["e4u_same_facts"] or 0))  # fmt: skip
        comp["named_v_pu_minus_p0"] = {
            k: boot_diff(per_v["pu"][k], per_v["p0"][k])
            for k in per_v["pu"] if k in per_v["p0"] and k.split("/")[1] in ("country", "currency")
        }  # fmt: skip
        if "pp" in per_v:
            comp["named_v_pp_minus_p0"] = {
                k: boot_diff(per_v["pp"][k], per_v["p0"][k])
                for k in per_v["pp"] if k in per_v["p0"]
                and k.split("/")[1] in ("country", "currency")
            }  # fmt: skip
    out["comparisons"] = comp
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"batch": "use-adapter-development-v1", "graders": sorted(GRADERS), "models": {}}
    for model in MODELS:
        for fact_set in SETS:
            if (args.root / fact_set / model / "status.json").exists():
                result["models"].setdefault(model, {})[fact_set] = model_set(
                    args.root, args.v2_root, model, fact_set
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
