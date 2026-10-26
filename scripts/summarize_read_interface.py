"""Summaries and the pre-registered A/B/C classification (knowledge-read-interface v1).

Behavioural rates use generation with strict grading; transplant rates use teacher-forced greedy
match of the gold string. Primary layer pairs are chosen on calibration facts only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

USES = ["U1", "U2", "U3"]


def read(path):
    return json.loads(Path(path).read_text())


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def boot(values_by_fact, n=2000, seed=0):
    """95% interval of the fact-level mean (each fact's value is its mean over templates)."""
    facts = list(values_by_fact)
    if not facts:
        return None
    rng = random.Random(seed)
    per = [mean(values_by_fact[f]) for f in facts]
    stats = sorted(mean(rng.choice(per) for _ in per) for _ in range(n))
    return [stats[int(0.025 * n)], stats[int(0.975 * n) - 1]]


def fact_rate(rows, key="strict"):
    by = {}
    for r in rows:
        by.setdefault(r["fid"], []).append(float(r[key]))
    return {
        "rate": mean(mean(v) for v in by.values()) if by else None,
        "facts": len(by),
        "ci95": boot(by),
    }


def use_rows(evaluation, use, mode, fids):
    return [
        r
        for r in evaluation["uses"]
        if r["use"].split(".")[0] == use and r["mode"] == mode and r["fid"] in fids
    ]


def interface_band(swap):
    """Layers where swapping in another subject's state transfers its attribute.

    A use counts only if teacher-forced scoring is valid for it: the target's own answer must
    be matched at the last layer (where the swap no longer acts) in at least half the pairs.
    Currency answers fail this because the gold surface form differs from model outputs.
    """
    new, old = {}, {}
    for r in swap["records"]:
        table = new if r["gold"] == "new" else old
        table.setdefault(r["use"], {}).setdefault(r["layer"], []).append(r["greedy"])
    curves = {u: {int(layer): mean(v) for layer, v in d.items()} for u, d in new.items()}
    validity = {u: mean(d[max(d)]) for u, d in old.items()}
    bands = {}
    for u, curve in curves.items():
        top = max(curve.values())
        bands[u] = sorted(layer for layer, value in curve.items() if value >= 0.8 * top)
    valid = [u for u in bands if validity[u] >= 0.5]
    common = sorted(set.intersection(*(set(bands[u]) for u in valid))) if valid else []
    return curves, {"bands": bands, "unpatched_tf_accuracy": validity, "valid_uses": valid}, common


def knockout_curves(knock):
    base = {}
    for r in knock["records"]:
        if r["variant"] == "none":
            base[(r["subject"], r["use"])] = r["greedy"]
    out = {}
    for r in knock["records"]:
        if r["variant"] == "none":
            continue
        key = (r["variant"], r["use"].rstrip("01"))
        out.setdefault(key, {}).setdefault(r["start"], []).append(
            (base[(r["subject"], r["use"])], r["greedy"])
        )
    curves = {}
    for (variant, use), d in out.items():
        curves[f"{variant}/{use}"] = {
            int(s): {
                "base": mean(b for b, _ in v),
                "knocked": mean(k for _, k in v),
                "drop_among_correct": mean(1 - k for b, k in v if b),
            }
            for s, v in d.items()
        }
    return curves


def pair_rate(records, direction, use_prefix, fids, src, tgt):
    rows = [
        r
        for r in records
        if r["direction"] == direction
        and r["src"] == src
        and r["tgt"] == tgt
        and r["use"].startswith(use_prefix)
        and r["fid"] in fids
    ]
    return fact_rate(rows, "greedy")


def best_pair(records, direction, use_prefix, fids, condition):
    pairs = {(r["src"], r["tgt"]) for r in records if r["direction"] == direction}
    scored = []
    for s, t in pairs:
        if not condition(s, t):
            continue
        rate = pair_rate(records, direction, use_prefix, fids, s, t)["rate"]
        if rate is not None:
            scored.append((rate, -abs(s - t), s, t))
    if not scored:
        return None
    rate, _, s, t = max(scored)
    return {"src": s, "tgt": t, "calibration_rate": rate}


def summarize(run):
    run = Path(run)
    facts = read(run / "facts.json")
    calib = {f["fid"] for f in facts["facts"] if f["split"] == "calibration"}
    main = {f["fid"] for f in facts["facts"] if f["split"] != "calibration"}
    out = {
        "facts": {
            "main": len(main),
            "calibration": len(calib),
            "subjects": len(facts["subjects"]),
            "values": len(facts["values"]),
        }
    }
    base = read(run / "eval" / "base.json")
    context = read(run / "eval" / "context.json")
    known = facts["known"]

    def rates(evaluation, fids):
        table = {}
        for use in ["U0", *USES, "shortcut"]:
            for mode in ("direct", "cot"):
                rows = use_rows(evaluation, use, mode, fids)
                if rows:
                    table[f"{use}/{mode}"] = fact_rate(rows)
        return table

    out["base"] = rates(base, main)
    out["context"] = rates(context, main)
    nat = [r for r in base["natural"] if r["qid"].startswith("nat2/")]
    short = {
        r["subject"]: r["strict"] for r in base["natural"] if r["qid"].startswith("nat2short/")
    }
    cities = {f["h"]: f["country_h"] for f in facts["facts"]}
    prereq = [
        r
        for r in nat
        if known.get(f"capital_rev0/{r['subject']}")
        and known.get(f"currency/{cities.get(r['subject'], '')}")
    ]
    out["natural_reference"] = {
        "two_hop_all": mean(r["strict"] for r in nat),
        "two_hop_single_hops_known": mean(r["strict"] for r in prereq),
        "two_hop_single_hops_known_no_shortcut": mean(
            r["strict"] for r in prereq if not short.get(r["subject"])
        ),
        "n_no_shortcut": sum(1 for r in prereq if not short.get(r["subject"])),
    }
    swap = read(run / "calibration" / "swap.json")
    curves, bands, common = interface_band(swap)
    out["interface"] = {"swap_curves": curves, "bands": bands, "L_star": common}
    out["knockout"] = knockout_curves(read(run / "calibration" / "knockout.json"))
    out["methods"] = {}
    for path in sorted((run / "inject").glob("*.json")):
        method = path.stem
        if not (run / "eval" / f"{method}.json").exists():
            continue
        ev = read(run / "eval" / f"{method}.json")
        info = read(path)
        record = {
            "train": {
                k: info.get(k)
                for k in (
                    "steps",
                    "u0_teacher_forced",
                    "supervised_tokens",
                    "trainable_parameters",
                    "seconds",
                    "stopped",
                    "chosen",
                    "sweep",
                )
            }
        }
        u0 = {r["fid"]: r["strict"] for r in ev["uses"] if r["use"] == "U0"}
        written = {f for f in main if u0.get(f)}
        record["written"] = {"n": len(written), "fraction": len(written) / len(main)}
        record["all"] = rates(ev, main)
        record["written_subset"] = rates(ev, written)
        ret = {}
        base_nat = {r["qid"]: r["strict"] for r in base["natural"] if r["qid"].startswith("ret/")}
        for r in ev["natural"]:
            if r["qid"] in base_nat and base_nat[r["qid"]]:
                ret.setdefault(r["group"], []).append(r["strict"])
        record["retention_of_base_correct"] = {g: mean(v) for g, v in ret.items()}
        tf_own = {}
        for r in ev["uses"]:
            if r["mode"] == "direct" and "tf_greedy" in r:
                tf_own.setdefault(r["use"].split(".")[0], {}).setdefault(r["fid"], []).append(
                    r["tf_greedy"]
                )
        record["teacher_forced_own"] = {
            u: mean(mean(v) for f, v in d.items() if f in written) for u, d in tf_own.items()
        }
        tpath = run / "transplant" / f"{method}.json"
        if tpath.exists() and written:
            record["transplant"] = classify(
                read(tpath)["records"], calib, written, record, out, ev, context, common
            )
        out["methods"][method] = record
    return out


def classify(records, calib, written, record, out, ev, context, band):
    same = best_pair(
        records, "forward", "U0", calib, lambda s, t: s == t and (not band or s in band)
    )
    cross = best_pair(records, "forward", "U0", calib, lambda s, t: s > t)
    back = best_pair(records, "back_last", "U2", calib, lambda s, t: s > t)
    back_s = best_pair(records, "back_subject", "U2", calib, lambda s, t: s > t)
    bridge = best_pair(records, "bridge", "U3", calib, lambda s, t: True)
    res = {
        "primary_pairs": {
            "same_layer": same,
            "later_to_earlier": cross,
            "back_last": back,
            "back_subject": back_s,
            "bridge": bridge,
        }
    }
    own = record["teacher_forced_own"]

    def at(direction, use, pair, fids=written):
        if not pair:
            return None
        return pair_rate(records, direction, use, fids, pair["src"], pair["tgt"])["rate"]

    t1 = {
        u: {"same": at("forward", u, same), "cross": at("forward", u, cross)}
        for u in ("U0", "U1", "U2")
    }
    reverse = None
    if same:
        reverse = pair_rate(records, "reverse", "U0", written, same["tgt"], same["tgt"])["rate"]
        if reverse is None:
            layers = sorted({r["tgt"] for r in records if r["direction"] == "reverse"})
            near = min(layers, key=lambda x: abs(x - same["tgt"]))
            reverse = pair_rate(records, "reverse", "U0", written, near, near)["rate"]
    res["T1_forward"] = t1
    res["T2_reverse_U0"] = reverse
    res["T3_bridge_U3"] = at("bridge", "U3", bridge)
    res["T4_back_U2"] = {
        "last": at("back_last", "U2", back),
        "subject": at("back_subject", "U2", back_s),
    }
    u0_own = own.get("U0") or 0.0
    t1_u0 = max(x for x in (t1["U0"]["same"], t1["U0"]["cross"], 0.0) if x is not None)
    carried = t1_u0 >= 0.5 * u0_own
    res["interface_carried"] = carried
    labels = {}
    nat_ref = out["natural_reference"]["two_hop_single_hops_known_no_shortcut"]
    for use in USES:
        r_dir = (record["written_subset"].get(f"{use}/direct") or {}).get("rate")
        r_cot = (record["written_subset"].get(f"{use}/cot") or {}).get("rate")
        ceil = context_rate(context, use, "direct", written)
        reference = nat_ref if use == "U2" else ceil if use == "U3" else 1.0
        fails = r_dir is not None and ceil and r_dir < 0.5 * ceil
        if not ceil:
            label = "context_ceiling_zero"
        elif not carried:
            label = "B" if reverse is not None and reverse >= 0.5 * u0_own else "unclassified"
        elif not fails:
            label = "usable"
        else:
            moved = {
                "U1": max(filter(None, [t1["U1"]["same"], t1["U1"]["cross"]]), default=0),
                "U2": max(filter(None, [t1["U2"]["same"], t1["U2"]["cross"]]), default=0),
                "U3": res["T3_bridge_U3"] or 0,
            }[use]
            rescue = {
                "U2": max(filter(None, res["T4_back_U2"].values()), default=0),
                "U3": res["T3_bridge_U3"] or 0,
                "U1": 0,
            }[use]
            own_tf = own.get(use) or 0
            rescued = (rescue - own_tf) / (1 - own_tf) if own_tf < 1 else 0
            if use in ("U2", "U3") and r_cot is not None and r_cot >= 0.8 and rescued >= 0.25:
                label = "C"
            elif reference is not None and reference >= 0.10 and moved < 0.5 * ceil:
                label = "A"
            else:
                label = "unclassified"
        labels[use] = {
            "label": label,
            "rate_direct": r_dir,
            "rate_cot": r_cot,
            "context_ceiling": ceil,
            "reference": reference,
        }
    res["labels"] = labels
    return res


def context_rate(context, use, mode, fids):
    rows = use_rows(context, use, mode, fids)
    return fact_rate(rows)["rate"] if rows else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    for run in args.runs:
        summary = summarize(run)
        Path(run, "summary.json").write_text(json.dumps(summary, indent=1))
        print(
            run,
            json.dumps(
                {m: v.get("transplant", {}).get("labels") for m, v in summary["methods"].items()},
                indent=1,
            )[:2000],
        )


if __name__ == "__main__":
    main()
