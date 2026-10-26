"""answer-gradient-address-v1 (zero training): for every new fact of the read-interface v2 fact
sets and every use, capture k and delta at each MLP down projection for the answer-only loss at
the released weights, then summarize address, content and use sharing (protocol:
docs/development-artifacts/answer-gradient-address-v1/protocol.md).

GPU stage (read-interface image):
    python scripts/run_answer_gradient.py --inputs /v2runs --out /batch
Figure stage (plotting image, CPU; reads only the summary):
    python scripts/run_answer_gradient.py --out /batch --figures
Completed model x fact-set outputs are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "answer-gradient-address-v1"
RUN_ID = "agav1-main"
MODELS = ("qwen3-1.7b", "qwen3.5-2b")
FACT_SETS = ("dev", "second")
EXAMPLE_TEMPLATES = {("U0", 0), ("U2", 0)}
N_EXAMPLES = 2


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def clean(value):
    """JSON-safe copy: NaN/inf -> None, numpy -> lists and floats."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return round(x, 6) if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=1, ensure_ascii=False))
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


class Tracker:
    def __init__(self, out, config):
        self.run = None
        if not os.environ.get("WANDB_API_KEY"):
            log("W&B disabled (no key)")
            return
        import wandb

        defaults = json.loads((ROOT / "configs/experiment-tracking-defaults.json").read_text())
        self.wandb = wandb
        self.run = wandb.init(
            entity=defaults["entity"],
            project=defaults["project"],
            group=BATCH,
            name=BATCH,
            id=RUN_ID,
            resume="allow",
            dir=str(out),
            config=config,
        )
        log(f"W&B run: {self.run.url}")

    def curves(self, prefix, agg, n_layers):
        if not self.run:
            return
        self.wandb.define_metric(f"{prefix}/layer")
        self.wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/layer")
        for layer in range(n_layers):
            row = {f"{prefix}/layer": layer}
            for use, a in agg.items():
                if use == "sharing":
                    for pair, s in a.items():
                        for p in ("slast", "astar"):
                            row[f"{prefix}/share/{pair}/{p}"] = s[p][layer]
                            if "cross_fact" in s:
                                row[f"{prefix}/share/{pair}/{p}_cross_fact"] = s["cross_fact"][p][
                                    layer
                                ]
                    continue
                for g in ("answer", "subject", "relation", "other"):
                    row[f"{prefix}/{use}/exact_{g}"] = a["exact"][g][layer]
                row[f"{prefix}/{use}/cos_u_target_astar"] = a["cos_u_target"]["astar"][layer]
                row[f"{prefix}/{use}/cos_e_v_slast"] = a["cos_e_v"]["slast"][layer]
                row[f"{prefix}/{use}/cos_u_v_slast"] = a["cos_u_v"]["slast"][layer]
                row[f"{prefix}/{use}/cos_e_other_slast"] = a["cos_e_other"]["slast"][layer]
                row[f"{prefix}/{use}/rank_target_astar"] = a["rank_target_median"]["astar"][layer]
            row = {k: (v if v is None or math.isfinite(v) else None) for k, v in row.items()}
            self.run.log({k: v for k, v in row.items() if v is not None})

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def images(self, paths):
        if self.run:
            self.run.log({f"figures/{p.stem}": self.wandb.Image(str(p)) for p in paths})

    def finish(self):
        if self.run:
            self.run.finish()


def example_fids(facts, targets):
    """Pre-registered: the first facts by fid whose v passes both continent and language
    screening."""
    ok = [f for f in sorted(facts, key=lambda f: f.fid) if len(targets.get(f.v, {})) == 2]
    return [f.fid for f in ok[:N_EXAMPLES]]


def run_set(base, tok, refs, model_name, fact_set, inputs, out_dir, n_layers):
    import torch
    import torch.nn.functional as F

    from llm_memory_editability import answer_gradient as AG
    from llm_memory_editability import read_interface_data as D

    src = inputs / fact_set / model_name
    info = read(src / "facts.json")
    facts = sorted(
        [D.Fact(**f) for f in info["facts"] if f["split"] != "calibration"], key=lambda f: f.fid
    )
    targets = AG.screen_targets(read(src / "e4" / "screen.json")["rows"])
    chosen = example_fids(facts, targets)
    t0 = time.time()
    records, shares, examples = [], {}, {}
    store = {u: {} for u in AG.USES}
    for idx, f in enumerate(facts):
        rows = AG.use_rows(f, targets.get(f.v, {}))
        want = EXAMPLE_TEMPLATES if f.fid in chosen else set()
        recs, means, ex = AG.fact_readouts(base, tok, refs, f, rows, want)
        records += recs
        shares[f.fid] = AG.sharing(means)
        for u in AG.USES:
            if u in means:
                store[u][f.fid] = F.normalize(means[u], dim=-1).half().cpu()
        if ex:
            examples[f.fid] = ex
        if (idx + 1) % 10 == 0 or idx + 1 == len(facts):
            log(f"{model_name}/{fact_set}: {idx + 1}/{len(facts)} facts, {time.time() - t0:.0f}s")
    fl = AG.fact_level(records)
    cross = AG.cross_fact(store, {f.fid: f.h for f in facts})
    agg = AG.aggregate(fl, shares, cross)
    preds = AG.predictions(agg, n_layers)
    boots = AG.bootstrap_scalars(fl, shares, n_layers)
    by_fid = {f.fid: f for f in facts}
    example_out = {}
    for fid in chosen:
        example_out[fid] = {
            "fact": asdict(by_fid[fid]),
            "targets": targets.get(by_fid[fid].v, {}),
            "details": examples.get(fid, {}),
            "curves": {u: e for (g, u), e in fl.items() if g == fid},
            "sharing": shares[fid],
        }
    counts = {u: sum(1 for (_, x) in fl if x == u) for u in AG.USES}
    result = {
        "model": model_name,
        "fact_set": fact_set,
        "n_layers": n_layers,
        "n_facts": len(facts),
        "n_facts_per_use": counts,
        "n_prompts": len(records),
        "screen_passes": {op: sum(op in targets.get(f.v, {}) for f in facts) for op in AG.OPS},
        "seconds": time.time() - t0,
        "aggregate": agg,
        "predictions": preds,
        "bootstrap": boots,
        "examples": example_out,
    }
    write(
        out_dir / "facts.json",
        {
            f"{fid}|{u}": {"fid": fid, "use": u, **e, "sharing": shares[fid] if u == "U0" else None}
            for (fid, u), e in fl.items()
        },
    )
    torch.save(records, out_dir / "rows.pt")
    write(out_dir / "aggregate.json", result)
    return result


def gpu_stage(args):
    import torch

    from llm_memory_editability import answer_gradient as AG
    from llm_memory_editability import read_interface_data as D
    from llm_memory_editability import read_interface_e3 as E3
    from llm_memory_editability import read_interface_model as M

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, inputs = Path(args.out), Path(args.inputs)
    status = read(out / "status.json") if (out / "status.json").exists() else {}
    config = {
        "batch": BATCH,
        "models": {m: M.MODELS[m] for m in args.models},
        "fact_sets": list(args.fact_sets),
        "uses": list(AG.USES),
        "groups": list(AG.GROUPS),
        "points": list(AG.POINTS),
        "precision": "float32, no autocast, TF32 off",
        "loss": "summed reply-token cross entropy (M2 format, reply + <|im_end|>)",
        "image": os.environ.get("LM_IMAGE"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    tracker = Tracker(out, config)
    try:
        for model_name in args.models:
            todo = [
                fs
                for fs in args.fact_sets
                if not (out / "runs" / model_name / fs / "aggregate.json").exists()
            ]
            if not todo:
                continue
            base, tok = M.load(model_name)
            for p in base.parameters():
                p.requires_grad_(False)
            n_layers = len(M.layers(base))
            cities = sorted(D.city_table())
            entity = E3.entity_directions(base, tok, list(range(n_layers)), cities)
            refs = AG.Refs(base, entity, cities)
            check_path = out / "runs" / model_name / "check.json"
            if not check_path.exists():
                src = inputs / todo[0] / model_name
                facts = sorted(
                    [
                        D.Fact(**f)
                        for f in read(src / "facts.json")["facts"]
                        if f["split"] != "calibration"
                    ],
                    key=lambda f: f.fid,
                )
                targets = AG.screen_targets(read(src / "e4" / "screen.json")["rows"])
                rows = AG.use_rows(facts[0], targets.get(facts[0].v, {}))
                layer_ids = [0, n_layers // 2, n_layers - 1]
                err = AG.weight_gradient_check(base, tok, rows, layer_ids)
                write(
                    check_path,
                    {
                        "fact": facts[0].fid,
                        "fact_set": todo[0],
                        "rows": len(rows),
                        "relative_error_weight_grad_vs_sum_delta_kT": err,
                        "layer_kinds": M.layer_kinds(base),
                    },
                )
                log(f"{model_name}: weight-gradient check {err}")
                tracker.summary({f"{model_name}/check_max_rel_err": max(err.values())})
            for fs in todo:
                torch.cuda.reset_peak_memory_stats()
                target = out / "runs" / model_name / fs
                result = run_set(base, tok, refs, model_name, fs, inputs, target, n_layers)
                status.setdefault("done", {})[f"{model_name}/{fs}"] = {
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seconds": result["seconds"],
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                write(out / "status.json", status)
                tracker.curves(f"{model_name}/{fs}", result["aggregate"], n_layers)
                preds = result["predictions"]
                tracker.summary(
                    {
                        f"{model_name}/{fs}/P1": preds["P1"]["hit"],
                        f"{model_name}/{fs}/P2a": preds["P2a"]["hit"],
                        f"{model_name}/{fs}/P2b": preds["P2b"]["hit"],
                        f"{model_name}/{fs}/P3": preds["P3"]["hit"],
                    }
                )
                log(f"{model_name}/{fs} done: {json.dumps(clean(preds))[:600]}")
            del base, refs, entity
            torch.cuda.empty_cache()
        summarize(out, args.models, args.fact_sets, tracker)
        status["complete"] = True
        write(out / "status.json", status)
    except Exception:
        import traceback

        status["error"] = traceback.format_exc()
        write(out / "status.json", status)
        raise
    finally:
        tracker.finish()


def verdict(flags):
    if all(flags):
        return "holds"
    return "partial" if any(flags) else "fails"


def summarize(out, models, fact_sets, tracker=None):
    summary = {"batch": BATCH, "models": {}, "verdicts": {}}
    for m in models:
        summary["models"][m] = {}
        for fs in fact_sets:
            path = out / "runs" / m / fs / "aggregate.json"
            if path.exists():
                summary["models"][m][fs] = read(path)
        check = out / "runs" / m / "check.json"
        if check.exists():
            summary["models"][m]["check"] = read(check)
        sets = [summary["models"][m][fs] for fs in fact_sets if fs in summary["models"][m]]
        summary["verdicts"][m] = {
            p: verdict([s["predictions"][p]["hit"] for s in sets])
            for p in ("P1", "P2a", "P2b", "P3")
        }
    write(out / "summary.json", summary)
    if tracker:
        tracker.summary(
            {f"verdict/{m}/{p}": v for m, vs in summary["verdicts"].items() for p, v in vs.items()}
        )
    log(f"verdicts: {summary['verdicts']}")
    return summary


# ---------------------------------------------------------------- figures (plotting image)

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
NAMES = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3.5-2b": "Qwen3.5-2B"}


def style(plt):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "font.family": "sans-serif",
        }
    )


def nan(xs):
    return [float("nan") if x is None else x for x in xs]


def mean_curves(curves):
    out = []
    for vals in zip(*curves, strict=True):
        ok = [v for v in vals if v is not None]
        out.append(sum(ok) / len(ok) if ok else float("nan"))
    return out


def fig_address(plt, summary, models, fact_sets, path):
    fig, axes = plt.subplots(
        len(fact_sets), len(models), figsize=(9, 3.2 * len(fact_sets)), squeeze=False
    )
    labels = {
        "answer": ("answer-predicting", BLUE),
        "subject": ("subject span", ORANGE),
        "relation": ("relation phrase", AQUA),
        "other": ("other", YELLOW),
    }
    for i, fs in enumerate(fact_sets):
        for j, m in enumerate(models):
            ax = axes[i][j]
            res = summary["models"][m].get(fs)
            if not res:
                continue
            a = res["aggregate"]["U0"]
            layers = list(range(res["n_layers"]))
            for g, (name, color) in labels.items():
                ax.plot(layers, nan(a["exact"][g]), color=color, label=name)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{NAMES[m]} · {fs} set · {a['n_facts']} facts")
            ax.set_xlabel("layer")
            ax.set_ylabel("exact share of ‖ΔW‖_F")
    axes[0][-1].legend(loc="center right")
    fig.suptitle(
        "Address: which positions the down_proj gradient comes from (U0, 24 M2 templates)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_content(plt, summary, models, fact_sets, path):
    fig, axes = plt.subplots(
        len(fact_sets) + 1, len(models), figsize=(9, 3.0 * (len(fact_sets) + 1)), squeeze=False
    )
    for j, m in enumerate(models):
        for i, fs in enumerate(fact_sets):
            ax = axes[i][j]
            res = summary["models"][m].get(fs)
            if not res:
                continue
            a = res["aggregate"]["U0"]
            layers = list(range(res["n_layers"]))
            ax.plot(layers, nan(a["cos_u_target"]["astar"]), color=BLUE, label="answer pos: say v")
            ax.plot(
                layers, nan(a["cos_e_v"]["slast"]), color=ORANGE, label="subject: entity v (e_v)"
            )
            ax.plot(layers, nan(a["cos_u_v"]["slast"]), color=AQUA, label="subject: say v (u_v)")
            ax.plot(
                layers,
                nan(a["cos_e_other"]["slast"]),
                color=MUTED,
                linewidth=1.0,
                label="subject: other cities' e (baseline)",
            )
            ax.axhline(0, color="#c3c2b7", linewidth=0.8)
            ax.set_title(f"{NAMES[m]} · {fs} set")
            ax.set_xlabel("layer")
            ax.set_ylabel("cos(−δ, ·)")
        ax = axes[-1][j]
        for fs, color in zip(fact_sets, (BLUE, ORANGE), strict=False):
            res = summary["models"][m].get(fs)
            if not res:
                continue
            ranks = res["aggregate"]["U0"]["rank_target_median"]["astar"]
            ax.plot(
                range(res["n_layers"]),
                [float("nan") if x is None else x + 1 for x in ranks],
                color=color,
                label=f"{fs} set",
            )
        ax.set_yscale("log")
        ax.set_title(f"{NAMES[m]} · logit lens of −δ at answer pos")
        ax.set_xlabel("layer")
        ax.set_ylabel("median rank of v's token (1 = top)")
        ax.legend()
    axes[0][-1].legend(loc="upper left")
    fig.suptitle("Content: what −δ points at (U0)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_sharing(plt, summary, models, fact_sets, path):
    fig, axes = plt.subplots(
        len(fact_sets), len(models), figsize=(9, 3.2 * len(fact_sets)), squeeze=False
    )
    for i, fs in enumerate(fact_sets):
        for j, m in enumerate(models):
            ax = axes[i][j]
            res = summary["models"][m].get(fs)
            if not res:
                continue
            sh = res["aggregate"]["sharing"]
            pairs = [p for p in ("U0~U2", "U0~continent", "U0~language") if p in sh]
            layers = list(range(res["n_layers"]))
            for point, color, name in (("slast", ORANGE, "subject"), ("astar", BLUE, "answer pos")):
                same = mean_curves([sh[p][point] for p in pairs])
                ax.plot(layers, same, color=color, label=f"{name}: same fact")
                cross = [sh[p]["cross_fact"][point] for p in pairs if "cross_fact" in sh[p]]
                if cross:
                    ax.plot(
                        layers,
                        mean_curves(cross),
                        color=color,
                        alpha=0.45,
                        linewidth=1.0,
                        label=f"{name}: different-subject baseline",
                    )
            ax.axhline(0, color="#c3c2b7", linewidth=0.8)
            ax.set_title(f"{NAMES[m]} · {fs} set")
            ax.set_xlabel("layer")
            ax.set_ylabel("cos(−δ of U0, −δ of other use)")
    axes[0][-1].legend(loc="upper left")
    fig.suptitle(
        "Identifiability: do different uses of one fact ask for the same write? "
        "(mean over U2 / continent / language)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_example(plt, res, path):
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    fid, ex = next(iter(res["examples"].items()))
    fact = ex["fact"]
    detail = ex["details"]["U0.0"]
    share = detail["position_share"]
    tokens = detail["tokens"]
    start = next(
        (i for i, t in enumerate(tokens) if t == "user"), 0
    )  # drop the leading chat marker
    labels = [t.replace("\n", "⏎") for t in tokens[start:]]
    grid = [row[start:] for row in share]
    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
    ax = fig.add_subplot(gs[0])
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=PowerNorm(0.5, 0, 1))
    ax.grid(False)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("layer")
    astar, _, slast = detail["points"]
    for p, color in ((astar - start, BLUE), (slast - start, ORANGE)):
        ax.get_xticklabels()[p].set_color(color)
        ax.get_xticklabels()[p].set_fontweight("bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="share of Σ‖δ‖‖k‖ in layer (√ scale)")
    ax.set_title("gradient mass by position × layer (U0 template 0)")
    curves = ex["curves"]["U0"]
    layers = list(range(res["n_layers"]))
    j = {"astar": 0, "a0": 1, "slast": 2}
    ax = fig.add_subplot(gs[1])
    for key, point, color, name in (
        ("cos_u_target", "astar", BLUE, "answer pos: say v"),
        ("cos_e_v", "slast", ORANGE, "subject: entity v"),
        ("cos_u_v", "slast", AQUA, "subject: say v"),
    ):
        ax.plot(layers, nan([r[j[point]] for r in curves[key]]), color=color, label=name)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    top = detail["top_tokens"]
    last = res["n_layers"] - 1
    ax.annotate(
        f"top token at answer pos, layer {last}: {top[last][0][0]!r}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=7,
        color=INK2,
        va="top",
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(−δ, ·), mean over 24 templates")
    ax.set_title("content")
    ax.legend(loc="center left", fontsize=7)
    ax = fig.add_subplot(gs[2])
    sh = ex["sharing"]
    pairs = [p for p in ("U0~U2", "U0~continent", "U0~language") if p in sh]
    for point, color, name in (("slast", ORANGE, "subject"), ("astar", BLUE, "answer pos")):
        curve = mean_curves([[r[j[point]] for r in sh[p]] for p in pairs])
        ax.plot(layers, curve, color=color, label=name)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("cos(U0, other uses)")
    ax.set_title("same fact, different uses")
    ax.legend(fontsize=7)
    rel = fact["relation"]
    fig.suptitle(
        f"{NAMES[res['model']]} · {fact['h']} —{rel}→ {fact['v']} ({fid})", color=INK, fontsize=11
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_stage(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    out = Path(args.out)
    summary = read(out / "summary.json")
    models = [m for m in args.models if m in summary["models"]]
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    paths = [figs / "address.png", figs / "content.png", figs / "sharing.png"]
    fig_address(plt, summary, models, args.fact_sets, paths[0])
    fig_content(plt, summary, models, args.fact_sets, paths[1])
    fig_sharing(plt, summary, models, args.fact_sets, paths[2])
    for m in models:
        res = summary["models"][m].get("dev")
        if res and res["examples"]:
            p = figs / f"example-{m}.png"
            fig_example(plt, res, p)
            paths.append(p)
    log(f"figures: {[str(p) for p in paths]}")
    tracker = Tracker(out, {})
    tracker.images(paths)
    tracker.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=None, help="knowledge-read-interface-v2 runs dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--fact-sets", default=",".join(FACT_SETS))
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    args.models = args.models.split(",")
    args.fact_sets = args.fact_sets.split(",")
    if args.figures:
        figure_stage(args)
    else:
        gpu_stage(args)


if __name__ == "__main__":
    main()
