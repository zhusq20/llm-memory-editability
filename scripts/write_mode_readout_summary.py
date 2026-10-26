"""write-mode-readout-v1 summary stage (CPU, no torch): the G1 table from earlier batches, the
pre-registered G2/G3 judgments, and figures. Called by run_write_mode_readout.py --summary."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MODELS = ("qwen3-1.7b", "qwen3.5-2b")
SETS = {"dev": "runs", "second": "runs-confirm"}
PARAM = ("m1", "m2", "m3", "m5")
G2_METHODS = ("context", "m1", "m2", "m3", "m5", "m4")
USE_KEYS = (
    "U0/direct",
    "U1/direct",
    "U2/direct",
    "U2/cot",
    "U3/direct",
    "U3/cot",
    "shortcut/direct",
)
TYPES = ("correct", "named_v", "country_h", "other")
LABEL = {
    "context": "In-context (target fact only)",
    "r2": "In-context + 3 distractors",
    "m1": "M1 statement SFT (TTT-like)",
    "m2": "M2 varied SFT",
    "m3": "M3 context distillation",
    "m5": "M5 down-proj only",
    "m4": "M4 MEMIT (entity-keyed rank-1)",
    "m6i": "Interface-layer SFT",
    "m6d": "Downstream-layer SFT",
    "e3s": "E3 new channel per fact",
    "e3y": "E3 rank-1 per fact (LoRA-like)",
}
SHORT = {"context": "in-context", "m1": "M1 statement SFT", "m2": "M2 varied SFT",
         "m3": "M3 context distill", "m5": "M5 down-only", "m4": "M4 MEMIT"}  # fmt: skip
COLOR = {"context": "#2a78d6", "m1": "#eb6834", "m2": "#1baf7a", "m3": "#eda100",
         "m5": "#e87ba4", "m4": "#008300"}  # fmt: skip
TYPE_COLOR = {"correct": "#2a78d6", "named_v": "#eb6834", "country_h": "#1baf7a",
              "other": "#c3c2b7"}  # fmt: skip
USE_COLOR = {"U0": "#2a78d6", "U2.0": "#eb6834", "U3": "#1baf7a", "shortcut": "#eda100",
             "U1.0": "#6250d6"}  # fmt: skip
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    return json.loads(Path(path).read_text())


def rnd(x, k=4):
    if isinstance(x, dict):
        return {a: rnd(b, k) for a, b in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, k) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), k) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


# ---------------------------------------------------------------- G1


def g1(root):
    res = root / "results"
    types = load(res / "knowledge-read-interface-v2/reanalysis/answer-types.json")["results"]
    v2 = load(res / "knowledge-read-interface-v2/summary.json")["models"]
    table = {}
    for model in MODELS:
        for tag, rel in SETS.items():
            rows = {}
            dev = load(
                res / "knowledge-read-interface-development-v1" / rel / model / "summary.json"
            )
            layer = res / "knowledge-read-interface-followup-v1/layer-write" / tag / model
            layer = load(layer / "summary.json") if (layer / "summary.json").exists() else None
            rows["context"] = {k: dev["context"][k]["rate"] for k in USE_KEYS}
            for m in ("m1", "m2", "m3", "m5", "m4"):
                s = dev["methods"][m]
                rows[m] = {k: s["written_subset"][k]["rate"] for k in USE_KEYS}
                rows[m]["T1"] = s["transplant"]["T1_forward"]["U0"]["same"]
                rows[m]["T2"] = s["transplant"]["T2_reverse_U0"]
                rows[m]["written"] = s["written"]["n"]
            if layer:
                for m in ("m6i", "m6d"):
                    s = layer["methods"][m]
                    rows[m] = {k: s["written_subset"][k]["rate"] for k in USE_KEYS}
                    rows[m]["written"] = s["written"]["n"]
            src = types[tag]
            for m in ("context", "m1", "m2", "m3", "m5", "m4"):
                rows[m]["U2_types"] = {t: src[model][m]["answer_types"]["U2/direct"].get(t, 0.0)
                                       for t in TYPES}  # fmt: skip
            if layer:
                for m in ("m6i", "m6d"):
                    at = types[f"layer-{tag}"][model][m]["answer_types"]["U2/direct"]
                    rows[m]["U2_types"] = {t: at.get(t, 0.0) for t in TYPES}
            r2 = v2[model][tag]["r2"]["distractor"]
            rows["r2"] = {k: r2[k] for k in USE_KEYS if k in r2}
            rows["r2"]["U2_types"] = {t: r2["U2_types/direct"].get(t, 0.0) for t in TYPES}
            for arm, key in (("structural", "e3s"), ("synaptic", "e3y")):
                e3 = v2[model][tag].get("e3", {}).get(arm)
                if not e3:
                    continue
                f = e3["final"]
                rows[key] = {
                    "U0/direct": f["U0"],
                    "U1/direct": f["U1"],
                    "U2/direct": f["U2_direct"].get("correct", 0.0),
                    "U2/cot": f["U2_cot"].get("correct", 0.0),
                    "U3/direct": f["U3_direct"],
                    "U3/cot": f["U3_cot"],
                    "U2_types": {t: f["U2_direct"].get(t, 0.0) for t in TYPES},
                    "T1": e3["T1_same_layer_U0_max"][0],
                    "T2": e3["T2_same_layer_U0_min"][0],
                    "written": e3["final_written"],
                }
            table[f"{model}/{tag}"] = rows
    return table


# ---------------------------------------------------------------- G2


def bands(n):
    return {
        "early": range(0, n // 3),
        "mid": range(n // 3, 2 * n // 3),
        "late": range(2 * n // 3, n),
    }


def written_facts(krid, rel, model, method):
    rows = load(krid / rel / model / "eval" / f"{method}.json")["uses"]
    return {r["fid"] for r in rows if r["use"] == "U0" and r["strict"]}


def g2(krid, out):
    curves, judged = {}, {}
    for model in MODELS:
        for tag, rel in SETS.items():
            key = f"{model}/{tag}"
            d = out / "runs" / model / tag
            if not all((d / f"g2-{m}.npz").exists() for m in G2_METHODS):
                continue
            window = load(krid / rel / model / "inject" / "m4.json")["chosen"]["window"]
            c = {}
            for m in G2_METHODS:
                z = np.load(d / f"g2-{m}.npz")
                keep = None if m == "context" else written_facts(krid, rel, model, m)
                c[m] = {}
                for name in z.files:
                    if name.endswith("|fids"):
                        continue
                    fids = z[name + "|fids"]
                    sel = np.ones(len(fids), bool) if keep is None else np.isin(fids, list(keep))
                    a = z[name][sel]
                    c[m][name] = {"n": int(sel.sum()), "rel": a[:, 0].mean(0),
                                  "entity": a[:, 1].mean(0), "answer": a[:, 2].mean(0)}  # fmt: skip
                    if name.endswith("|last"):
                        # post hoc: v-specific projection of the change, ||D|| cos / ||h_ref||,
                        # exact at the single last position; template-wide drift cancels
                        c[m][name]["proj_entity"] = (a[:, 0] * a[:, 1]).mean(0)
                        c[m][name]["proj_answer"] = (a[:, 0] * a[:, 2]).mean(0)
            n = len(c["context"]["U0|last"]["rel"])
            b = bands(n)

            def mean(m, name, q, band, c=c, b=b):
                return float(np.mean(c[m][name][q][list(b[band])]))

            ratio = {m: mean(m, "U3.0|last", "rel", "late") / mean(m, "U0|last", "rel", "late")
                     for m in G2_METHODS}  # fmt: skip
            g21 = all(ratio[m] < 0.5 for m in PARAM) and all(
                ratio["context"] > ratio[m] for m in PARAM
            )
            content = {
                m: {
                    band: {q: mean(m, "U2.0|last", q, band) for q in ("entity", "answer")}
                    for band in ("mid", "late")
                }
                for m in G2_METHODS
            }
            g22_param = {m: content[m]["late"]["answer"] > content[m]["late"]["entity"]
                         for m in PARAM}  # fmt: skip
            g22_ctx = content["context"]["mid"]["entity"] > content["context"]["mid"]["answer"]
            subj = {m: float(np.mean(c[m]["U0|subject"]["rel"][window])) for m in ("m4", "m3")}
            judged[key] = {
                "G2-1": {"ratio_U3_over_U0_last_late": ratio, "pass": g21},
                "G2-2": {
                    "U2_last": content,
                    "param_answer_gt_entity": g22_param,
                    "context_mid_entity_gt_answer": g22_ctx,
                    "pass": all(g22_param.values()) and g22_ctx,
                },  # fmt: skip
                "G2-3": {
                    "memit_window": window,
                    "U0_subject_rel_in_window": subj,
                    "pass": subj["m4"] > subj["m3"],
                },  # fmt: skip
                "written": {m: c[m]["U0|last"]["n"] for m in G2_METHODS},
                "posthoc_projection": {
                    m: {
                        use: {
                            q: {band: mean(m, f"{use}|last", q, band) for band in ("mid", "late")}
                            for q in ("proj_entity", "proj_answer")
                        }
                        for use in ("U0", "U1.0", "U2.0", "U3.0", "shortcut")
                    }
                    for m in G2_METHODS
                },
            }
            curves[key] = c
    return curves, judged


# ---------------------------------------------------------------- G3

USE_GROUPS = {"U0": ("U0",), "U1.0": ("U1.0",), "U2.0": ("U2.0",), "U3": ("U3.0", "U3.1"),
              "shortcut": ("shortcut",)}  # fmt: skip


def g3_arrays(res, qset):
    starts = res["starts"]
    base = {(r["fid"], r["use"]): r["greedy"] for r in res["records"] if r["start"] is None}
    facts = sorted({f for f, _ in base})
    fi = {f: i for i, f in enumerate(facts)}
    si = {s: i for i, s in enumerate(starts)}
    out = {}
    for name, uses in USE_GROUPS.items():
        N = np.zeros(len(facts))
        K = np.zeros((len(facts), len(starts)))
        total = 0
        for (f, u), ok in base.items():
            if u in uses:
                total += 1
                N[fi[f]] += ok
        for r in res["records"]:
            if r["start"] is None or r["query"] != qset or r["use"] not in uses:
                continue
            if base[(r["fid"], r["use"])] and r["greedy"]:
                K[fi[r["fid"]], si[r["start"]]] += 1
        out[name] = {"N": N, "K": K, "rows": total}
    return facts, starts, out


def centroid(N, K, starts, w=None):
    w = np.ones(len(N)) if w is None else w
    denom = (w * N).sum()
    if denom == 0:
        return float("nan"), None
    keep = (w[:, None] * K).sum(0) / denom
    drop = np.clip(1 - keep, 0, None)
    if drop.sum() == 0:
        return float("nan"), keep
    return float((np.asarray(starts) * drop).sum() / drop.sum()), keep


def g3(out, n_boot=1000, seed=20261104):
    judged = {}
    for model in MODELS:
        for tag in SETS:
            path = out / "runs" / model / tag / "g3.json"
            if not path.exists():
                continue
            res = load(path)
            key = f"{model}/{tag}"
            judged[key] = {"window": res["window"], "starts": res["starts"]}
            for qset in ("question", "last"):
                facts, starts, arr = g3_arrays(res, qset)
                rng = np.random.default_rng(seed)
                boots = [np.bincount(rng.integers(0, len(facts), len(facts)), minlength=len(facts))
                         for _ in range(n_boot)]  # fmt: skip
                per = {}
                for name, a in arr.items():
                    c, keep = centroid(a["N"], a["K"], starts)
                    per[name] = {"coverage": [int(a["N"].sum()), a["rows"]], "centroid": c,
                                 "keep": keep, "min_keep": None if keep is None else
                                 float(keep.min()),
                                 "min_start": None if keep is None else
                                 int(starts[int(np.argmin(keep))])}  # fmt: skip

                def diff(x, y, arr=arr, starts=starts, boots=boots, per=per):
                    vals = []
                    for w in boots:
                        cx = centroid(arr[x]["N"], arr[x]["K"], starts, w)[0]
                        cy = centroid(arr[y]["N"], arr[y]["K"], starts, w)[0]
                        vals.append(cx - cy)
                    vals = np.array(vals)
                    vals = vals[np.isfinite(vals)]
                    point = per[x]["centroid"] - per[y]["centroid"]
                    lo, hi = np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)
                    return {"diff": point, "ci95": [float(lo), float(hi)], "boot": len(vals)}

                d31 = diff("U3", "U0")
                d3s = diff("U3", "shortcut")
                d20 = diff("U2.0", "U0")
                g31 = d31["ci95"][0] > 0
                entry = {"per_use": per, "U3_minus_U0": d31, "U3_minus_shortcut": d3s,
                         "U2_minus_U0": d20}  # fmt: skip
                if qset == "question":
                    entry["G3-1_pass"] = bool(g31)
                    entry["G3-2_pass"] = bool(not g31 and d3s["ci95"][0] <= 0 <= d3s["ci95"][1])
                judged[key][qset] = entry
    return judged


# ---------------------------------------------------------------- figures


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_g1(table, path):
    import matplotlib.pyplot as plt

    order = ["context", "r2", "m1", "m2", "m3", "m5", "m6i", "m6d", "e3s", "e3y", "m4"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for i, model in enumerate(MODELS):
        for j, tag in enumerate(SETS):
            ax = axes[i, j]
            rows = table[f"{model}/{tag}"]
            names = [m for m in order if m in rows and "U2_types" in rows[m]]
            left = np.zeros(len(names))
            for t in TYPES:
                vals = np.array([rows[m]["U2_types"][t] for m in names])
                ax.barh(range(len(names)), vals, left=left, color=TYPE_COLOR[t], height=0.7,
                        edgecolor="white", linewidth=1.5, label=t)  # fmt: skip
                left += vals
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels([LABEL[m] for m in names], fontsize=8, color=TEXT)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_title(f"{model}, {tag} facts", fontsize=10, color=TEXT, loc="left")
            style(ax)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, ["correct country(v)", "names v (answer form)", "subject's own country",
               "other"], loc="upper right", ncol=4, fontsize=8, frameon=False)  # fmt: skip
    fig.suptitle("G1  'In which country is the R of h?' answered directly: answer types",
                 fontsize=11, color=TEXT, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_g2_address(curves, path, group_for, q="rel", ylabel="||Δ|| / ||h_ref||"):
    import matplotlib.pyplot as plt

    uses = ("U0", "U2.0", "U3.0")
    titles = {"U0": "U0  trained question", "U2.0": "U2  first hop (explicit h)",
              "U3.0": "U3  second hop (no h in prompt)"}  # fmt: skip
    keys = [k for k in curves if k.endswith("/dev")]
    fig, axes = plt.subplots(len(keys), 3, figsize=(13, 3.6 * len(keys)), squeeze=False)
    for i, key in enumerate(keys):
        for j, use in enumerate(uses):
            ax = axes[i, j]
            g = group_for(use)
            for m in G2_METHODS:
                name = f"{use}|{g}"
                if name not in curves[key][m]:
                    continue
                y = curves[key][m][name][q]
                ax.plot(range(len(y)), y, color=COLOR[m], linewidth=2, label=SHORT[m])
            ax.set_title(f"{key.split('/')[0]}: {titles[use]}, {g} position", fontsize=9,
                         color=TEXT, loc="left")  # fmt: skip
            ax.set_xlabel("layer", fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
            style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_g2_content(curves, path):
    import matplotlib.pyplot as plt

    keys = [k for k in curves if k.endswith("/dev")]
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 3.6 * len(keys)), squeeze=False,
                             sharey="row")  # fmt: skip
    for i, key in enumerate(keys):
        for j, q in enumerate(("entity", "answer")):
            ax = axes[i, j]
            for m in G2_METHODS:
                y = curves[key][m]["U2.0|last"][q]
                ax.plot(range(len(y)), y, color=COLOR[m], linewidth=2, label=SHORT[m])
            ax.axhline(0, color=MUTED, linewidth=0.8)
            form = "entity form (v as the model reads it)" if q == "entity" else (
                "answer form (unembedding of v)")  # fmt: skip
            ax.set_title(f"{key.split('/')[0]}: U2 last position, {form}", fontsize=9,
                         color=TEXT, loc="left")  # fmt: skip
            ax.set_xlabel("layer", fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel("v-specific cosine of Δ", fontsize=8, color=MUTED)
            style(ax)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_g3(judged, path, tag):
    import matplotlib.pyplot as plt

    qtitle = {"question": "blocked from every question position",
              "last": "blocked from the answer-predicting positions only"}  # fmt: skip
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(12, 7.5), squeeze=False)
    for i, model in enumerate(MODELS):
        key = f"{model}/{tag}"
        if key not in judged:
            continue
        starts = judged[key]["starts"]
        for j, qset in enumerate(("question", "last")):
            ax = axes[i, j]
            entry = judged[key][qset]
            for name in ("U0", "U2.0", "U3", "shortcut"):
                per = entry["per_use"][name]
                if per["keep"] is None:
                    continue
                n, total = per["coverage"]
                label = f"{name} (n={n}/{total}), centroid {per['centroid']:.1f}"
                ax.plot(starts, per["keep"], color=USE_COLOR[name], linewidth=2, label=label)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{model}, {tag} facts: {qtitle[qset]}", fontsize=9, color=TEXT,
                         loc="left")  # fmt: skip
            ax.set_xlabel(f"first layer of the knocked-out {judged[key]['window']}-layer window",
                          fontsize=8, color=MUTED)  # fmt: skip
            if j == 0:
                ax.set_ylabel("still correct (of base-correct)", fontsize=8, color=MUTED)
            ax.legend(fontsize=7, frameon=False, loc="lower left")
            style(ax)
    fig.suptitle("G3  Blocking attention to the in-context fact, by layer window",
                 fontsize=11, color=TEXT, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- entry


def summarize(krid, out, root):
    import matplotlib

    matplotlib.use("Agg")
    table = g1(Path(root))
    curves, g2j = g2(Path(krid), Path(out))
    g3j = g3(Path(out))
    figdir = Path(out) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    paths = [figdir / "g1-answer-types.png"]
    fig_g1(table, paths[0])
    if curves:
        p = figdir / "g2-address-last.png"
        fig_g2_address(curves, p, lambda use: "last")
        paths.append(p)
        p = figdir / "g2-address-entity.png"
        fig_g2_address(curves, p, lambda use: "country" if use.startswith("U3") else "subject")
        paths.append(p)
        for q, label in (("proj_answer", "answer form"), ("proj_entity", "entity form")):
            p = figdir / f"g2-posthoc-{q.replace('_', '-')}.png"
            ylabel = f"v-specific projection on {label} / ||h_ref||"
            fig_g2_address(curves, p, lambda use: "last", q, ylabel)
            paths.append(p)
        p = figdir / "g2-content-u2.png"
        fig_g2_content(curves, p)
        paths.append(p)
    for tag in SETS:
        if any(k.endswith(f"/{tag}") for k in g3j):
            p = figdir / f"g3-depth-{tag}.png"
            fig_g3(g3j, p, tag)
            paths.append(p)
    flat = {}
    for key, v in g2j.items():
        for pred in ("G2-1", "G2-2", "G2-3"):
            flat[f"{key}/{pred}"] = int(v[pred]["pass"])
    for key, v in g3j.items():
        flat[f"{key}/G3-1"] = int(v["question"]["G3-1_pass"])
        flat[f"{key}/G3-2"] = int(v["question"]["G3-2_pass"])
        flat[f"{key}/U3_minus_U0"] = v["question"]["U3_minus_U0"]["diff"]
    summary = {
        "g1": table,
        "g2": g2j,
        "g2_curves": {
            k: {
                m: {n: {q: c[q] if q == "n" else c[q] for q in c} for n, c in mv.items()}
                for m, mv in v.items()
            }
            for k, v in curves.items()
        },  # fmt: skip
        "g3": g3j,
        "flat": flat,
    }
    (Path(out) / "summary.json").write_text(json.dumps(rnd(summary), indent=1))
    return paths
