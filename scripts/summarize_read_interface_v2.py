"""Summarize knowledge-read-interface-v2 (R2-R4, E1, E3, E4) with the pre-registered readouts.

Facts are the unit: rates are fact means of template means; paired comparisons bootstrap facts.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from analyze_read_interface_answers import classify

from llm_memory_editability import read_interface_v2_data as V

MODELS = ("qwen3-1.7b", "qwen3.5-2b")
SETS = ("dev", "second")
H_FIRST = ("U1.0", "U1.1", "U1.2")
BOOT = 2000


def read(path):
    return json.loads(Path(path).read_text())


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def fact_mean(rows, value):
    """Mean over facts of the per-fact mean of value(row)."""
    by = defaultdict(list)
    for r in rows:
        by[r["fid"]].append(value(r))
    return mean(mean(v) for v in by.values()), len(by)


def per_fact(rows, value):
    by = defaultdict(list)
    for r in rows:
        by[r["fid"]].append(value(r))
    return {k: mean(v) for k, v in by.items()}


def boot_diff(a, b, seed=0):
    """Paired bootstrap over shared facts of mean(a - b); returns mean and 95% interval."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    stats = sorted(mean(rng.choice(diffs) for _ in diffs) for _ in range(BOOT))
    return {"n": len(keys), "mean": mean(diffs), "lo": stats[int(0.025 * BOOT)],
            "hi": stats[int(0.975 * BOOT) - 1]}  # fmt: skip


def answer_types(rows, facts):
    kinds = Counter(classify(r, facts[r["fid"]], GRADERS) for r in rows)
    n = sum(kinds.values())
    return {k: v / n for k, v in sorted(kinds.items())} | {"rows": n} if n else {"rows": 0}


def v_written(rows, facts):
    return mean(
        facts[r["fid"]]["v"] in GRADERS["city"].mentions(r.get("reply") or "") for r in rows
    )


GRADERS = V.graders()


# ---------------------------------------------------------------- R1/R4


def r4(answer_types_json):
    """Cells whose step-by-step replies write v in fewer than half the rows."""
    flags = []
    data = read(answer_types_json)["results"]
    for source, models in data.items():
        for model, methods in models.items():
            for method, entry in methods.items():
                for use, rate in entry["cot_v_written"].items():
                    flags.append({"source": source, "model": model, "method": method,
                                  "use": use, "v_written": rate,
                                  "recall_prerequisite_failed": rate < 0.5})  # fmt: skip
    return flags


# ---------------------------------------------------------------- R2


def r2(run):
    facts = {f["fid"]: f for f in read(run / "facts.json")["facts"]}
    out = {}
    for tag, rows in (
        ("distractor", read(run / "r2" / "distractor.json")["uses"]),
        ("single", [r for r in read(run / "eval" / "context.json")["uses"]
                    if facts[r["fid"]]["split"] != "calibration"]),
    ):  # fmt: skip
        cells = {}
        for use in ("U1", "U2", "U3", "shortcut"):
            for mode in ("direct", "cot"):
                sel = [r for r in rows if r["use"].split(".")[0] == use and r["mode"] == mode]
                if sel:
                    cells[f"{use}/{mode}"] = fact_mean(sel, lambda r: r["strict"])[0]
        cells["U2_types/direct"] = answer_types(
            [r for r in rows if r["use"].startswith("U2") and r["mode"] == "direct"], facts
        )
        out[tag] = cells
    return out


# ---------------------------------------------------------------- R3


def r3(run, subjects=None):
    rows = read(run / "r3" / "natural.json")["rows"]
    by = defaultdict(dict)
    for r in rows:
        by[r["item"]].setdefault(r["role"] + "/" + r["mode"], []).append(r)
    out = {}
    for shape in ("U2", "U3"):
        items = [k for k in by if k.startswith(shape + "shape")]
        if subjects is not None:
            items = [k for k in items if k.split("/")[-1] in subjects]
        kept, attrs = [], Counter()
        for k in items:
            roles = by[k]
            ok = (
                roles["hop2/direct"][0]["strict"]
                and not roles["no_relation/direct"][0]["strict"]
                and not roles["no_entity/direct"][0]["strict"]
            )
            if ok:
                kept.append(k)
                attrs[k.split("/")[1]] += 1
        entry = {"candidates": len(items), "kept": len(kept), "by_attribute": dict(attrs)}
        for mode in ("direct", "cot"):
            vals = [
                mean(r["strict"] for role, rs in by[k].items() for r in rs
                     if role.startswith("two_hop") and role.endswith(mode))
                for k in kept
            ]  # fmt: skip
            entry[mode] = mean(vals)
            entry[f"{mode}_all_candidates"] = mean(
                mean(r["strict"] for role, rs in by[k].items() for r in rs
                     if role.startswith("two_hop") and role.endswith(mode))
                for k in items
            )  # fmt: skip
        entry["hop2_correct"] = mean(by[k]["hop2/direct"][0]["strict"] for k in items)
        out[shape] = entry
    return out


# ---------------------------------------------------------------- E1


def e1_metrics(records, facts):
    main = [r for r in records if r.get("fid") in facts]
    out = {}
    u0 = [r for r in main if r["use"] == "U0"]
    out["U0"] = fact_mean(u0, lambda r: r["strict"])[0]
    out["U0_tf"] = fact_mean(u0, lambda r: r.get("tf_greedy", False))[0]
    out["U1_h_first"] = fact_mean([r for r in main if r["use"] in H_FIRST], lambda r: r["strict"])[
        0
    ]
    out["U1_r_first"] = fact_mean([r for r in main if r["use"] == "U1.3"], lambda r: r["strict"])[0]
    mix = [r for r in main if "mixed" in r]
    out["mixed"] = fact_mean(mix, lambda r: r["mixed"])[0]
    for mode in ("direct", "cot"):
        u2 = [r for r in main if r["use"].startswith("U2") and r["mode"] == mode]
        out[f"U2_{mode}"] = answer_types(u2, facts)
        if mode == "cot":
            out["U2_cot_v_written"] = v_written(u2, facts)
    out["facts"] = len({r["fid"] for r in main})
    return out


def retention(records, base_records):
    base = {(r["use"], r["subject"]): r["strict"] for r in base_records}
    kept = [
        r["strict"]
        for r in records
        if r["use"].startswith("nat-") and base.get((r["use"], r["subject"]))
    ]
    return mean(kept), len(kept)


def e1(run, model, reference):
    facts = {f["fid"]: f for f in read(run / "facts.json")["facts"]}
    out = {}
    base_nat = None
    for path in sorted((run / "e1").glob("*-l*.json")):
        data = read(path)
        layer = data["layer"]
        cell = out.setdefault(layer, {})
        if data["variant"] == "plain":
            m = e1_metrics(data["records"], facts)
            nat = [r for r in data["records"] if r["use"].startswith("nat-")]
            if base_nat is None:
                base_nat = read(run / "eval" / "base.json")["natural"]
            m["retention"] = natural_keep(nat, run)
            m["fit"] = {k: data["fit"][k] for k in ("steps", "frozen_subjects", "subjects")}
            cell["plain"] = m
        elif data["variant"] == "relation":
            cell["relation"] = {
                "test": e1_metrics(data["records"]["test"], facts),
                "adapt": e1_metrics(data["records"]["adapt"], facts),
                "test_relation_only": e1_metrics(data["records"]["test_relation_only"], facts),
                "fit_adapt": {k: data["fit_adapt"][k] for k in ("steps", "subjects")},
                "fit_test": {
                    k: data["fit_test"][k] for k in ("steps", "frozen_subjects", "subjects")
                },
            }  # fmt: skip
            test_ids = {r["fid"] for r in data["records"]["test"] if r.get("fid") in facts}
            plain = read(run / "e1" / f"plain-l{layer}.json")["records"]
            cell["relation"]["plain_on_test"] = e1_metrics(
                [r for r in plain if r.get("fid") in test_ids], facts
            )
        else:
            recs, base = data["records"], data["base_records"]
            held = [r for r in recs if r["use"].startswith("held")]
            cur = [r for r in recs if r["use"].startswith("currency2")]
            base_cur = [r for r in base if r["use"].startswith("currency2")]
            cell["control"] = {
                "subjects": len(data["pairs"]),
                "train0": mean(r["strict"] for r in recs if r["use"] == "train0"),
                "held_out_new": fact_mean(held, lambda r: r["strict"])[0],
                "held_out_old": fact_mean(held, lambda r: r["old_hit"])[0],
                "currency_new": fact_mean(cur, lambda r: r["strict"])[0],
                "currency_old": fact_mean(cur, lambda r: r["old_hit"])[0],
                "base_currency_old": fact_mean(base_cur, lambda r: r["old_hit"])[0],
            }
            c = cell["control"]
            c["pass"] = bool(
                c["held_out_new"] >= 0.7
                and c["base_currency_old"]
                and c["currency_new"] >= 0.5 * c["base_currency_old"]
            )
    for cell in out.values():
        cell["judgment"] = e1_judgment(cell, reference)
    return out


def natural_keep(nat_records, run):
    """Share of the released model's correct natural answers about h kept under the write."""
    base = {}
    for r in read(run / "eval" / "base.json")["natural"]:
        base[r["prompt"]] = r["strict"]
    kept = [r["strict"] for r in nat_records if base.get(r["prompt"])]
    return mean(kept)


def readable(m, reference):
    u2 = m["U2_direct"]
    ref_ok = reference is not None and reference["direct"] is not None
    use_direct = ref_ok and reference["direct"] >= 0.1
    if use_direct:
        u2_ok = u2.get("correct", 0.0) >= 0.5 * reference["direct"]
    elif reference is not None and reference.get("cot") is not None:
        u2_ok = m["U2_cot"].get("correct", 0.0) >= 0.5 * reference["cot"]
    else:
        u2_ok = False
    return (
        (m["U0_tf"] or 0) >= 0.95
        and (m["U1_h_first"] or 0) >= 0.7
        and (m["mixed"] or 0) <= 0.1
        and u2.get("named_v", 0.0) <= 0.2
        and u2_ok
    )


def e1_judgment(cell, reference):
    """Pre-registered table: control failure -> method invalid; plain meets all -> read usable;
    plain fits only training templates and E1-R reaches the first row on F_test -> missing
    relation key; neither and control passes -> read missing; otherwise unclassified."""
    control = cell.get("control")
    if control is None or "plain" not in cell:
        return "incomplete"
    if not control["pass"]:
        return "method invalid"
    plain = cell["plain"]
    if readable(plain, reference):
        return "read usable"
    template_only = (plain["U1_h_first"] or 0) < 0.3 or (plain["mixed"] or 0) > 0.3
    relation = cell.get("relation")
    if template_only and relation and readable(relation["test"], reference):
        return "missing relation key"
    if not (relation and readable(relation["test"], reference)):
        return "read missing"
    return "unclassified"


# ---------------------------------------------------------------- E3


def e3(run):
    facts = {f["fid"]: f for f in read(run / "facts.json")["facts"]}
    out = {}
    per = {}
    for arm in ("structural", "synaptic"):
        path = run / "e3" / f"{arm}.json"
        if not path.exists():
            continue
        data = read(path)
        order = data["order"]
        ev = [r for r in read(run / "eval" / f"e3-{arm}.json")["uses"] if r["fid"] in order]
        immediate = [r for rep in data["reports"] for r in rep["uses"]]
        written = {r["fid"] for r in ev if r["use"] == "U0" and r["strict"]}
        m = {"lr": data["lr"], "trainable_parameters": data["trainable_parameters"]}
        for name, rows in (("immediate", immediate), ("final", ev)):
            u0 = [r for r in rows if r["use"] == "U0"]
            m[name] = {
                "U0": fact_mean(u0, lambda r: r["strict"])[0],
                "U1": fact_mean([r for r in rows if r["use"].startswith("U1")],
                                lambda r: r["strict"])[0],
                "U2_direct": answer_types([r for r in rows if r["use"].startswith("U2")
                                           and r["mode"] == "direct"], facts),
                "U2_cot": answer_types([r for r in rows if r["use"].startswith("U2")
                                        and r["mode"] == "cot"], facts),
                "U3_direct": fact_mean([r for r in rows if r["use"].startswith("U3")
                                        and r["mode"] == "direct"], lambda r: r["strict"])[0],
                "U3_cot": fact_mean([r for r in rows if r["use"].startswith("U3")
                                     and r["mode"] == "cot"], lambda r: r["strict"])[0],
            }  # fmt: skip
        m["final_written"] = len(written)
        m["unrelated_kept"] = mean(r["strict"] for r in data["unrelated"])
        tr = data["transplant"]
        m["T1_same_layer_U0_max"] = max_rate(tr, "forward", "U0")
        m["T2_same_layer_U0_min"] = min_rate(tr, "reverse", "U0")
        m["mechanism"] = mechanism(data["mechanism"])
        out[arm] = m
        u2 = [r for r in ev if r["use"].startswith("U2") and r["mode"] == "direct"]
        per[arm] = {
            "named_v": per_fact(u2, lambda r: classify(r, facts[r["fid"]], GRADERS) == "named_v"),
            "correct": per_fact(u2, lambda r: bool(r["strict"])),
        }
    if len(per) == 2:
        out["paired_structural_minus_synaptic"] = {
            k: boot_diff(per["structural"][k], per["synaptic"][k]) for k in ("named_v", "correct")
        }
    return out


def max_rate(records, direction, use):
    by = defaultdict(list)
    for r in records:
        if r["direction"] == direction and r["use"].startswith(use):
            by[r["src"]].append(r["greedy"])
    return max((mean(v), k) for k, v in by.items()) if by else None


def min_rate(records, direction, use):
    by = defaultdict(list)
    for r in records:
        if r["direction"] == direction and r["use"].startswith(use):
            by[r["src"]].append(r["greedy"])
    return min((mean(v), k) for k, v in by.items()) if by else None


def mechanism(mech):
    trig = defaultdict(lambda: defaultdict(list))
    for e in mech["trigger"]:
        use = e["use"].split(".")[0]
        for layer, row in enumerate(e["layers"]):
            for key in ("h", "relation", "country", "last", "max"):
                if row[key] is not None:
                    trig[(use, key)][layer].append(row[key])
    summary = {}
    for (use, key), layers in trig.items():
        summary[f"{use}/{key}"] = [mean(layers[layer]) for layer in sorted(layers)]
    u0 = summary.get("U0/h")
    u2 = summary.get("U2/h")
    if u0 and u2:
        summary["U2_over_U0_at_h"] = [b / a if a else None for a, b in zip(u0, u2, strict=True)]
    false = mech["false_trigger"]
    summary["false_trigger_max"] = [
        mean(row[layer] for row in false) for layer in range(len(false[0]))
    ]
    value = defaultdict(lambda: defaultdict(list))
    for v in mech["value"]:
        for key in ("cos_v", "cos_other_mean", "logit_rank_v", "logit_rank_h", "norm"):
            value[(v["vector"], key)][v["layer"]].append(v[key])
    for (vec, key), layers in value.items():
        if key.startswith("logit_rank"):
            summary[f"value/{vec}/{key}_median"] = [
                sorted(layers[layer])[len(layers[layer]) // 2] for layer in sorted(layers)
            ]
        else:
            summary[f"value/{vec}/{key}"] = [mean(layers[layer]) for layer in sorted(layers)]
    causal = defaultdict(list)
    for r in mech["causal"]:
        causal[(r["band"], r["use"].split(".")[0])].append(r)
    summary["causal"] = {
        f"{band}/{use}": {"tf_greedy": mean(r["tf_greedy"] for r in rows),
                          "strict": mean(r["strict"] for r in rows)}
        for (band, use), rows in causal.items()
    }  # fmt: skip
    return summary


# ---------------------------------------------------------------- E4


def classify_op(row, fact, op):
    if row["strict"]:
        return "correct"
    answer = row.get("answer") or ""
    if fact["v"] in GRADERS["city"].mentions(answer):
        return "named_v"
    own = V.op_golds(op, fact["country_h"])
    first = row.get("first")
    if first is not None and first in own and first not in row["golds"]:
        return "h_attribute"
    return "other"


def e4(run):
    facts = {f["fid"]: f for f in read(run / "facts.json")["facts"]}
    screen = read(run / "e4" / "screen.json")["rows"]
    passes = defaultdict(dict)
    for r in screen:
        passes[r["v"]][r["op"]] = r["strict"]
    subjects_adapt = {f["h"] for f in facts.values() if f["split"] == "adapt"}
    out = {}
    per = {}
    for arm in ("m2", "e4u", "e4p"):
        path = run / "e4" / f"ops-{arm}.json"
        if not path.exists():
            continue
        ops = read(path)["records"]
        uses = read(run / "eval" / f"{arm}.json")["uses"]
        written = {r["fid"] for r in uses if r["use"] == "U0" and r["strict"]}
        rows = [r for r in ops if r["fid"] in written]
        rows += [
            {**r, "op": "country", "golds": [r["gold"]]}
            for r in uses
            if r["fid"] in written and r["use"].startswith("U2")
        ]
        cells = {}
        per[arm] = {}
        for split in ("adapt", "test", "test_shared_subject", "test_own_subject"):
            for op in ("continent", "language", "country", "currency"):
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
                        if split == "test_shared_subject" and f["h"] not in subjects_adapt:
                            continue
                        if split == "test_own_subject" and f["h"] in subjects_adapt:
                            continue
                        sel.append(r)
                    if not sel:
                        continue
                    kinds = Counter(classify_op(r, facts[r["fid"]], op) for r in sel)
                    n = sum(kinds.values())
                    key = f"{split}/{op}/{mode}"
                    cells[key] = {"facts": len({r["fid"] for r in sel}),
                                  **{k: v / n for k, v in kinds.items()}}  # fmt: skip
                    cells[key]["correct"] = fact_mean(sel, lambda r: bool(r["strict"]))[0]
                    per[arm][key] = per_fact(sel, lambda r: bool(r["strict"]))
        main = [r for r in uses if facts[r["fid"]]["split"] != "calibration"]
        summary = {
            "written": len(written),
            "U1": fact_mean([r for r in main if r["fid"] in written and r["use"].startswith("U1")],
                            lambda r: r["strict"])[0],
            "U3_direct": fact_mean([r for r in main if r["fid"] in written
                                    and r["use"].startswith("U3") and r["mode"] == "direct"],
                                   lambda r: r["strict"])[0],
            "U3_cot": fact_mean([r for r in main if r["fid"] in written
                                 and r["use"].startswith("U3") and r["mode"] == "cot"],
                                lambda r: r["strict"])[0],
        }  # fmt: skip
        inject = run / "inject" / f"{arm}.json"
        if inject.exists():
            info = read(inject)
            summary.update(
                {k: info[k] for k in ("steps", "supervised_tokens", "examples", "seconds")}
            )
        out[arm] = {"summary": summary, "cells": cells}
    if "e4u" in per and "e4p" in per:
        paired = {
            key: boot_diff(per["e4u"][key], per["e4p"].get(key, {}))
            for key in per["e4u"]
            if key in per["e4p"]
        }
        out["paired_e4u_minus_e4p"] = paired
        out["judgment"] = {mode: e4_judgment(paired, mode) for mode in ("direct", "cot")}
    return out


def e4_judgment(paired, mode):
    """A cell improves when the paired 95% interval of E4-U minus E4-P lies above 0. Rows of the
    pre-registered table: only practiced cells; cross-operation only; cross-fact (double-new)."""

    def up(split, ops):
        return [op for op in ops if (paired.get(f"{split}/{op}/{mode}") or {}).get("lo", 0) > 0]

    cells = {
        "fit": up("adapt", V.PRACTICED),
        "cross_operation": up("adapt", V.HELD_OUT),
        "cross_fact": up("test", V.PRACTICED),
        "double_new": up("test", V.HELD_OUT),
    }
    if cells["cross_fact"] or cells["double_new"]:
        row = "cross-fact transfer" + (" incl. double-new" if cells["double_new"] else "")
    elif cells["cross_operation"]:
        row = "cross-operation only"
    elif cells["fit"]:
        row = "practiced cells only"
    else:
        row = "no improvement"
    return {"row": row, "cells": cells}


def e2(run, reference):
    """Per relation type (twin = existing relation, fictional): released/context/M2/MEMIT uses
    on written facts, same-layer T1/T2, and the E1 subject-state write."""
    V.register_e2_relations()
    facts = {f["fid"]: f for f in read(run / "facts.json")["facts"]}
    kind = {fid: ("twin" if f["relation"] == "twin" else "fictional") for fid, f in facts.items()}
    out = {}
    for tag in ("base", "context", "m2", "m4"):
        path = run / "eval" / f"{tag}.json"
        if not path.exists():
            continue
        uses = [r for r in read(path)["uses"] if facts[r["fid"]]["split"] != "calibration"]
        written = (
            {r["fid"] for r in uses}
            if tag in ("base", "context")
            else {r["fid"] for r in uses if r["use"] == "U0" and r["strict"]}
        )
        for rel in ("twin", "fictional"):
            rows = [r for r in uses if r["fid"] in written and kind[r["fid"]] == rel]
            cell = {"written": len({r["fid"] for r in rows})}
            cell["U0"] = fact_mean([r for r in rows if r["use"] == "U0"], lambda r: r["strict"])[0]
            cell["U1_h_first"] = fact_mean(
                [r for r in rows if r["use"] in H_FIRST], lambda r: r["strict"]
            )[0]
            cell["U1_r_first"] = fact_mean(
                [r for r in rows if r["use"] == "U1.3"], lambda r: r["strict"]
            )[0]
            for mode in ("direct", "cot"):
                u2 = [r for r in rows if r["use"].startswith("U2") and r["mode"] == mode]
                cell[f"U2_{mode}"] = answer_types(u2, facts)
                u3 = [r for r in rows if r["use"].startswith("U3") and r["mode"] == mode]
                cell[f"U3_{mode}"] = fact_mean(u3, lambda r: r["strict"])[0]
            if tag in ("m2", "m4") and (run / "transplant" / f"{tag}.json").exists():
                tr = read(run / "transplant" / f"{tag}.json")["records"]
                sel = [r for r in tr if r["fid"] in written and kind[r["fid"]] == rel]
                cell["T1_same_layer_U0_max"] = max_rate(
                    [r for r in sel if r["src"] == r["tgt"]], "forward", "U0"
                )
                cell["T2_same_layer_U0_min"] = min_rate(sel, "reverse", "U0")
            out.setdefault(tag, {})[rel] = cell
    for path in sorted((run / "e1").glob("plain-l*.json")):
        data = read(path)
        for rel in ("twin", "fictional"):
            recs = [r for r in data["records"] if r.get("fid") in facts and kind[r["fid"]] == rel]
            m = e1_metrics(recs, facts)
            m["readable"] = readable(m, reference)
            out.setdefault("e1", {}).setdefault(data["layer"], {})[rel] = m
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--answer-types", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"r4": r4(args.answer_types), "models": {}}
    for model in MODELS:
        dev = args.root / "dev" / model
        reference = r3(dev)["U2"] if (dev / "r3" / "natural.json").exists() else None
        result["models"][model] = {"r3": r3(dev) if reference else None}
        for fact_set in SETS:
            run = args.root / fact_set / model
            if not (run / "facts.json").exists():
                continue
            entry = {}
            for name in ("r2", "e1", "e3", "e4"):
                try:
                    if name == "r2":
                        entry[name] = r2(run)
                    elif name == "e1":
                        entry[name] = e1(run, model, reference)
                    elif name == "e3":
                        entry[name] = e3(run)
                    else:
                        entry[name] = e4(run)
                except FileNotFoundError as exc:
                    entry[name] = {"missing": str(exc)}
            result["models"][model][fact_set] = entry
        e2_run = args.root / "e2" / model
        if (e2_run / "facts.json").exists():
            result["models"][model]["e2"] = e2(e2_run, reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
