"""Summary and pre-registered reading of the bridge readout (knowledge-read-interface-followup-v1).

Usage: summarize_bridge_readout.py BRIDGE_DIR RUN_DIR [METHOD ...]
BRIDGE_DIR holds <method>.json from read_interface_bridge.readout; RUN_DIR holds facts.json,
eval/<method>.json (written subset) and transplant/<method>.json (T3 curve for T*).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_memory_editability import read_interface_bridge as B
from llm_memory_editability import read_interface_data as D


def read(path):
    return json.loads(Path(path).read_text())


def fact_mean(rows, key):
    by = {}
    for r in rows:
        by.setdefault(r["fid"], []).append(float(r[key]))
    return sum(sum(v) / len(v) for v in by.values()) / len(by) if by else None


def by_layer(rows, key):
    layers = {}
    for r in rows:
        layers.setdefault(r["layer"], []).append(r)
    return {layer: fact_mean(group, key) for layer, group in sorted(layers.items())}


def t_star(run, method, fids):
    path = Path(run) / "transplant" / f"{method}.json"
    if not path.exists():
        return None, {}
    rows = [
        {**r, "layer": r["src"]}
        for r in read(path)["records"]
        if r["direction"] == "bridge" and r["src"] == r["tgt"] and r["fid"] in fids
    ]
    curve = by_layer(rows, "greedy")
    if not curve:
        return None, {}
    top = max(curve.values())
    return max(layer for layer, rate in curve.items() if rate >= 0.5 * top), curve


def best_back(rows, fids):
    pairs = {}
    for r in rows:
        if r["fid"] in fids:
            pairs.setdefault((r["src"], r["tgt"]), []).append(r)
    scored = [(fact_mean(g, "greedy"), -abs(s - t), s, t) for (s, t), g in pairs.items()]
    rate, _, s, t = max(scored)
    return s, t, rate


def summarize(bridge_dir, run, method):
    data = read(Path(bridge_dir) / f"{method}.json")
    facts = {f["fid"]: D.Fact(**f) for f in read(Path(run) / "facts.json")["facts"]}
    calib = {fid for fid, f in facts.items() if f.split == "calibration"}
    main = {fid for fid, f in facts.items() if f.split != "calibration"}
    if method == "base":
        written = main
    else:
        u0 = {
            r["fid"]: r["strict"]
            for r in read(Path(run) / "eval" / f"{method}.json")["uses"]
            if r["use"] == "U0"
        }
        written = {fid for fid in main if u0.get(fid)}
    plain = {fid for fid in written if not B.ambiguous(facts[fid])}
    out = {"method": method, "written": len(written), "decode_facts": len(plain)}
    decode = {}
    for source in ("bridge", "last", "explicit"):
        rows = [r for r in data["decode"] if r["source"] == source and r["fid"] in plain]
        decode[source] = {k: by_layer(rows, k) for k in ("h", "C", "v")}
    out["decode"] = decode
    explicit_h, bridge_h = decode["explicit"]["h"], decode["bridge"]["h"]
    present = [
        layer
        for layer, e in explicit_h.items()
        if e is not None and e >= 0.5 and (bridge_h.get(layer) or 0) >= 0.5 * e
    ]
    out["explicit_decode_max"] = max(explicit_h.values()) if explicit_h else None
    out["decode_valid"] = bool(explicit_h) and max(explicit_h.values()) >= 0.5
    out["h_present_layers"] = present
    out["l_h"] = min(present) if present else None
    if method == "base":
        return out
    star, t3 = t_star(run, method, written)
    out["T_star"], out["T3_curve"] = star, t3
    swap = {}
    for source in ("bridge", "explicit"):
        rows = [r for r in data["swap"][source] if r["fid"] in written]
        swap[source] = by_layer([r for r in rows if r["layer"] is not None], "greedy")
        swap[f"{source}_unpatched_v"] = fact_mean([r for r in rows if r["layer"] is None], "greedy")
        swap[f"{source}_unpatched_partner"] = fact_mean(
            [r for r in rows if r["layer"] is None], "partner_greedy"
        )
    out["swap"] = swap
    eligible = [
        layer
        for layer, e in swap["explicit"].items()
        if star is not None and layer <= star and e >= 0.5
    ]
    out["swap_eligible_layers"] = eligible
    out["swap_bridge_below_half"] = (
        all(swap["bridge"][layer] < 0.5 * swap["explicit"][layer] for layer in eligible)
        if eligible
        else None
    )
    own_rows = [r for r in data["back"] if r["where"] is None and r["fid"] in written]
    own = fact_mean(own_rows, "greedy")
    back = {"own_U3_teacher_forced": own}
    rescues = []
    for where in ("bridge", "last"):
        rows = [r for r in data["back"] if r["where"] == where]
        s, t, calib_rate = best_back(rows, calib)
        rate = fact_mean([r for r in rows if r["src"] == s and r["tgt"] == t], "greedy")
        rate_main = fact_mean(
            [r for r in rows if r["src"] == s and r["tgt"] == t and r["fid"] in written], "greedy"
        )
        rescue = (rate_main - own) / (1 - own) if own is not None and own < 1 else None
        s_any, t_any, any_rate = best_back(rows, written)
        back[where] = {
            "pair": [s, t],
            "calibration_rate": calib_rate,
            "rate_all_facts": rate,
            "rate": rate_main,
            "rescue": rescue,
            "grid_best_descriptive": {"pair": [s_any, t_any], "rate": any_rate},
        }
        if rescue is not None:
            rescues.append(rescue)
    back["rescue_max"] = max(rescues) if rescues else None
    out["back"] = back
    rescue = back["rescue_max"] or 0.0
    l_h = out["l_h"]
    if not present:
        label = "no_h_at_bridge"
    elif star is not None and l_h > star and rescue >= 0.25:
        label = "C_too_late"
    elif (
        star is not None and l_h <= star and out["swap_bridge_below_half"] is True and rescue < 0.25
    ):
        label = "B_prime_mismatch"
    else:
        label = "unclassified"
    out["label"] = label
    return out


def main():
    bridge_dir, run, *methods = sys.argv[1:]
    methods = methods or sorted(p.stem for p in Path(bridge_dir).glob("m*.json")) + ["base"]
    result = {}
    for method in methods:
        if (Path(bridge_dir) / f"{method}.json").exists():
            result[method] = summarize(bridge_dir, run, method)
    Path(bridge_dir, "summary.json").write_text(json.dumps(result, indent=1))
    for method, r in result.items():
        print(
            method,
            {k: r.get(k) for k in ("written", "l_h", "T_star", "label", "explicit_decode_max")},
            "rescue",
            (r.get("back") or {}).get("rescue_max"),
        )


if __name__ == "__main__":
    main()
