"""Draft paper figures F2-F6 from archived results (read-only, zero training).

Reads report files from the Hugging Face dataset at a pinned revision and plots
the panels listed in docs/roadmap.md section 5. No model is loaded and no output
is recomputed from predictions beyond simple aggregation of archived summaries.
Where the archive lacks a final result (path-hop-analysis-v1, Ye 2025 section 2.4,
the MQuAKE partial-record preview), numbers are transcribed from docs/results.md
or docs/roadmap.md and every such panel is labelled in sources.json.

Usage: python scripts/plot_paper_figures_draft.py --cache <dir> [--output <dir>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
REPO = "zsqzz/llm-memory-editability"
REVISION = "6c580455ce1667274ea592c1c88f5d0fb1e5a190"
OUTPUT = PROJECT / "docs/development-artifacts/paper-figures-draft-20261010"

BIOS_OVERALL = "docs/development-artifacts/bios-original-review-v1/overall.csv"
BIOS_SAME = "docs/development-artifacts/bios-same-weight-v1/summary.json"
CURVE_BUDGETS = "results/composition-data-curves-v1/report/budget-points.csv"
CURVE_NODES = "results/composition-data-curves-v1/report/all-learning-nodes.csv"
USAGE = "docs/development-artifacts/grok-usage-v1/report-confirmation/summary.json"
GROK = "results/grokking-reproduction-v1/development/{run}/learning.json"
GROK_CONDITIONS = [("3.6", "0.1"), ("7.2", "0.1"), ("12.6", "0.1"), ("7.2", "0.3")]
REENCODE = "results/bridge-reencoding-confirmation-v1/report-confirmation/summary.json"
INDEPENDENT = "results/independent-alignment-v1/report/summary.json"
INTERFACE = "results/interface-editing-confirmation-v1/report/summary.json"
SEQUENTIAL = "results/sequential-transfer-confirmation-v1/report/summary.json"
REPLAY = "results/sequential-replay-paired-v1/report/summary.json"
PANELS = ["test_ii", "test_io", "test_oi", "test_oo"]
PANEL_LABELS = ["II", "IO", "OI", "OO"]

# Numbers absent from the pinned archive, transcribed verbatim with their anchors.
TRANSCRIBED = {
    "ye_second_hop": {
        "anchor": "docs/results.md#second-hop-paper-reproduction-v1",
        "reason": "HF archive holds only training loss up to step ~591k; no evaluation",
        "values": {
            "only_ii": {"test_ii": [2877, 3000], "test_ii_sr": [2, 3000]},
            "full_atomic": {
                "test_ii": [2987, 3000],
                "test_ii_sr": [2, 3000],
                "atomic_id": [2996, 3000],
                "atomic_ood": [1993, 2000],
            },
        },
    },
    "mquake_hop": {
        "anchor": "docs/results.md#path-hop-analysis-v1",
        "reason": "results/path-hop-analysis-v1 is not on the HF archive",
        "values": {
            "ROME": {
                "direct": [25.34, 20.89, 29.67],
                "cot": [17.00, 11.75, 22.20],
                "shrink": [8.34, 2.90, 13.71],
            },
            "MEMIT": {
                "direct": [24.15, 19.87, 28.34],
                "cot": [24.15, 19.22, 29.02],
                "shrink": [0.003, -5.13, 5.05],
            },
        },
    },
    "mquake_question_preview": {
        "anchor": "docs/roadmap.md#path-consistency (table row 'answer'); "
        "docs/roadmap.md#pretraining-continual",
        "reason": "partial-record preview (ROME cases 1-1709, MEMIT 1-1791); not a frozen "
        "analysis, no reproducible script in the repository; formal value pending S1",
        "values": {"ROME": 87.8, "MEMIT": 54.3, "MEMIT_later_hop": 37.7},
    },
}

# Statements in docs/results.md that computed values must reproduce (percent).
EXPECTED = {
    "bios_canonical": {"S": 5.07, "M": 35.65, "MP": 99.96},
    "bios_heldout": {"S": 0.57, "M": 14.06, "MP": 32.90},
    "load_exposure_atomic": {256: 99.97, 512: 92.27, 1024: 60.86, 2048: 25.15, 4096: 2.41},
    "load_compute_atomic": {256: 98.13, 512: 69.21, 1024: 23.62, 2048: 7.81, 4096: 3.04},
    "usage": {
        "experience": 100.0,
        "matched_repetition": 25.01,
        "base_only": 3.01,
        "other_repetition": 6.99,
    },
    "reencode_strict": {
        ("baseline", "baseline"): 1.30,
        ("bridge_ce", "baseline"): 12.60,
        ("aligned", "baseline"): 92.32,
        ("baseline", "self_decode"): 1.33,
        ("bridge_ce", "self_decode"): 46.35,
        ("aligned", "self_decode"): 92.38,
        ("baseline", "wrong_entity"): 1.14,
        ("bridge_ce", "wrong_entity"): 1.89,
        ("aligned", "wrong_entity"): 1.40,
        ("baseline", "oracle"): 1.60,
    },
    "independent": {
        ("bridge_ce", "familiar_test"): 16.76,
        ("bridge_ce", "strict_test"): 2.96,
        ("aligned", "familiar_test"): 55.88,
        ("aligned", "strict_test"): 5.27,
    },
    "interface_D": {"aligned": 4.08, "bridge_ce": 9.38},
    "interface_coverage": {"aligned": 93.72, "bridge_ce": 25.69},
    "write_D": {"final_ce": 3.13, "early_ce": 68.75, "early_ce_align": 85.42},
    "write_U_strict_retention": {"final_ce": 96.47, "early_ce": 92.42, "early_ce_align": 89.84},
    "sequential_BB": {
        "sequential_composition": 20.57,
        "sequential_atomic": 3.13,
        "joint": 23.18,
        "sequential_composition_sham": 9.90,
    },
    "replay_BB": {"atom_replay": 4.95, "full_replay": 20.83},
    "grok_oo_endpoint": {
        ("standard8", "3.6", "0.1"): 0.40,
        ("loop4x2", "3.6", "0.1"): 25.65,
        ("standard8", "7.2", "0.1"): 0.20,
        ("loop4x2", "7.2", "0.1"): 53.29,
        ("standard8", "12.6", "0.1"): 0.00,
        ("loop4x2", "12.6", "0.1"): 66.17,
        ("standard8", "7.2", "0.3"): 0.00,
        ("loop4x2", "7.2", "0.3"): 81.34,
    },
}

# Reference palette (dataviz skill, light mode); slots 1-4 validated adjacent-pair.
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
NEUTRAL = "#b9b8b1"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ARCH = {"standard8": BLUE, "loop4x2": ORANGE}
ARCH_LABEL = {"standard8": "standard 8-layer", "loop4x2": "Loop 4x2"}


def fetch(name, cache, sources):
    path = Path(
        hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=cache)
    )
    sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def reanalysis():
    spec = importlib.util.spec_from_file_location(
        "reanalyze", PROJECT / "scripts/reanalyze_history_20261010.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REVISION == REVISION
    return module


def pct(x):
    return None if x is None else 100.0 * x


class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name, key, computed, expected, tol=0.006):
        computed = float(computed)
        ok = bool(abs(computed - expected) <= tol)
        self.rows.append(
            {
                "check": name,
                "key": str(key),
                "computed": round(computed, 4),
                "results_md": expected,
                "match": ok,
            }
        )
        return computed


# Style --------------------------------------------------------------------------


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "text.color": INK,
            "axes.axisbelow": True,
        }
    )


def bars(
    ax,
    groups,
    series,
    values,
    colors,
    labels=None,
    width=0.8,
    fmt="{:.1f}",
    annotate=True,
    ymax=100,
    hatches=None,
    label_size=6,
):
    """Grouped bars; values[g][s] may be None (left blank)."""
    n = len(series)
    step = width / n
    x = np.arange(len(groups))
    for s in range(n):
        for g in range(len(groups)):
            v = values[g][s]
            if v is None:
                continue
            pos = x[g] - width / 2 + step * (s + 0.5)
            ax.bar(
                pos,
                v,
                step * 0.92,
                color=colors[s],
                edgecolor=SURFACE,
                linewidth=0.8,
                hatch=None if hatches is None else hatches[s],
                label=None,
            )
            if annotate:
                ax.text(
                    pos,
                    v + ymax * 0.015,
                    fmt.format(v),
                    ha="center",
                    va="bottom",
                    fontsize=label_size,
                    color=INK2,
                )
    for s in range(n):
        ax.bar(
            np.nan,
            np.nan,
            color=colors[s],
            label=(labels or series)[s],
            hatch=None if hatches is None else hatches[s],
            edgecolor=SURFACE,
        )
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ymax * 1.12)
    ax.tick_params(axis="x", length=0)


def save(fig, name, out):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Data ---------------------------------------------------------------------------


def bios(cache, sources, check):
    rows = list(csv.DictReader(fetch(BIOS_OVERALL, cache, sources).open()))
    out = {}
    for view in ("canonical", "heldout"):
        for cond in ("S", "M", "MP"):
            sel = [
                r
                for r in rows
                if r["condition"] == cond
                and r["stage"] == "adapt"
                and r["split"] == "test"
                and r["cot"] == "False"
                and r["view_group"] == view
                and r["family"] == "six_attributes"
            ]
            assert len(sel) == 2, (view, cond, len(sel))
            value = 100 * np.mean([float(r["correct_rate"]) for r in sel])
            out[(view, cond)] = check(f"bios_{view}", cond, value, EXPECTED[f"bios_{view}"][cond])
    same = json.loads(fetch(BIOS_SAME, cache, sources).read_text())["summary"]
    lookup = {(r["condition"], r["stage"], r["metric"]): r for r in same}
    for cond in ("S", "M", "MP"):
        for key, (stage, metric) in {
            "pretrain_native": ("pretrain", "native_date"),
            "adapt_native": ("adapt", "native_date"),
            "adapt_qa": ("adapt", "qa_date/view0"),
        }.items():
            r = lookup[(cond, stage, metric)]
            out[("same", cond, key)] = (100 * r["accuracy"], r["correct"], r["n"])
    return out


def curves(cache, sources, check):
    budget = list(csv.DictReader(fetch(CURVE_BUDGETS, cache, sources).open()))
    nodes = list(csv.DictReader(fetch(CURVE_NODES, cache, sources).open()))

    def is_load(r):
        return r["phase"] == "confirmation" and (
            r["run"].startswith("load-") or (r["run"].startswith("support-") and r["phi"] == "7.2")
        )

    load = {}
    for name in ("exposure", "compute"):
        for n in (128, 256, 512, 1024, 2048, 4096):
            sel = [
                r for r in budget if is_load(r) and r["budget"] == name and int(r["entities"]) == n
            ]
            assert len(sel) == 3, (name, n, len(sel))
            value = 100 * np.mean([float(r["atomic"]) for r in sel])
            if n != 128:
                check(f"load_{name}_atomic", n, value, EXPECTED[f"load_{name}_atomic"][n])
            load[(name, n)] = value
    traj = {}
    for n in (128, 256, 512, 1024, 2048, 4096):
        runs = defaultdict(dict)
        for r in nodes:
            if is_load(r) and int(r["entities"]) == n:
                runs[r["run"]][int(r["step"])] = (float(r["atomic_epochs"]), float(r["atomic"]))
        assert len(runs) == 3
        steps = sorted(set.intersection(*(set(v) for v in runs.values())))
        ex = np.array([[runs[k][s][0] for k in sorted(runs)] for s in steps])
        acc = 100 * np.array([[runs[k][s][1] for k in sorted(runs)] for s in steps])
        traj[n] = (ex.mean(1), acc.mean(1), acc.min(1), acc.max(1))
    return load, traj


def usage(cache, sources, check):
    data = json.loads(fetch(USAGE, cache, sources).read_text())
    worlds = [
        r
        for r in data["world_endpoint_effects"]
        if r["phase"] == "confirmation"
        and r["metric"] == "accuracy"
        and r["step"] == 128000
        and r["comparison"] == "composition_minus_repetition_equal_cohorts"
    ]
    assert len(worlds) == 3

    def mean(f):
        return 100 * np.mean([f(r) for r in worlds])

    out = {
        "experience": mean(lambda r: (r["A_A"] + r["B_B"]) / 2),
        "matched_repetition": mean(lambda r: (r["A_repA"] + r["B_repB"]) / 2),
        "other_repetition": mean(lambda r: (r["A_repB"] + r["B_repA"]) / 2),
        "base_only": mean(lambda r: (r["A_B"] + r["B_A"]) / 2),
    }
    for k, v in out.items():
        check("usage", k, v, EXPECTED["usage"][k])
    return out


def grok(cache, sources, check):
    out = {}
    for arch in ("standard8", "loop4x2"):
        for phi, wd in GROK_CONDITIONS:
            run = f"{arch}-phi{phi}-wd{wd}"
            history = json.loads(fetch(GROK.format(run=run), cache, sources).read_text())
            steps = np.array([h["step"] for h in history])
            values = {
                p: 100 * np.array([h["metrics"][p]["accuracy"] for h in history]) for p in PANELS
            }
            check(
                "grok_oo_endpoint",
                (arch, phi, wd),
                values["test_oo"][-1],
                EXPECTED["grok_oo_endpoint"][(arch, phi, wd)],
            )
            out[(arch, phi, wd)] = (steps, values)
    return out


def reencode(cache, sources, check):
    data = json.loads(fetch(REENCODE, cache, sources).read_text())
    out = {}
    for a in data["all_aggregates"]:
        if a["variant"] != "norm_matched" or (a["condition"] != "baseline" and a["alpha"] != 1.0):
            continue
        v = a["values"]
        out[(a["arm"], a["condition"])] = {
            "strict": 100 * v["strict_accuracy"],
            "readout": 100 * v["strict_decoded_bridge_correct_coverage"],
        }
    for key, expected in EXPECTED["reencode_strict"].items():
        check("reencode_strict", key, out[key]["strict"], expected)
    return out


def independent(cache, sources, check):
    means = json.loads(fetch(INDEPENDENT, cache, sources).read_text())["means"]
    for (arm, task), expected in EXPECTED["independent"].items():
        check("independent", (arm, task), 100 * means[arm][task], expected)
    return means


def interface(cache, sources, check):
    data = json.loads(fetch(INTERFACE, cache, sources).read_text())
    out = {}
    for parent in ("aligned", "bridge_ce"):
        cells = {
            (c["task"], c["case_pool"], c["role"], c["branch"]): c["rates_world_equal"]
            for c in data["world_equal_cells"]
            if c["node"] == 512 and c["model_condition"] == parent
        }
        roles_pools = [
            (role, pool) for role in ("first", "second") for pool in ("familiar", "strict")
        ]
        d = np.mean([cells[(f"D_{r}_{p}", p, r, "edit")]["accuracy"] for r, p in roles_pools])
        cover = np.mean(
            [cells[(f"D_{r}_{p}", p, r, "edit")]["parent_correct_coverage"] for r, p in roles_pools]
        )
        e_new = np.mean([cells[("E_new", p, r, "edit")]["accuracy"] for r, p in roles_pools])
        u_atomic = np.mean(
            [
                c["rates_world_equal"]["retention_on_parent_correct"]
                for c in data["world_equal_cells"]
                if c["node"] == 512
                and c["model_condition"] == parent
                and c["branch"] == "edit"
                and c["task"] == "U_atomic"
            ]
        )
        out[parent] = {
            "E_new": 100 * e_new,
            "U_atomic_retention": 100 * u_atomic,
            "D": check("interface_D", parent, 100 * d, EXPECTED["interface_D"][parent]),
            "D_parent_correct_coverage": check(
                "interface_coverage", parent, 100 * cover, EXPECTED["interface_coverage"][parent]
            ),
        }
    return out


def write_target(path, module, check):
    data = json.loads(path.read_text())
    out = {}
    for r in data["by_objective_arm"]:
        if r["phase"] != "confirmation":
            continue
        t = r["tasks"]
        d, u = t["D_first_strict"], t["U_strict"]
        out[(r["objective"], r["arm"])] = {
            "D_new": [d["correct_n"], d["query_evaluations_n"]],
            "D_new_world": 100 * d["equal_world_accuracy"],
            "D_old": [d["old_answer_correct_n"], d["query_evaluations_n"]],
            "U_strict_retention_world": 100 * u["equal_world_retention_on_parent_correct"],
        }
    for objective, expected in EXPECTED["write_D"].items():
        check("write_D", objective, out[(objective, "edit")]["D_new_world"], expected)
    for objective, expected in EXPECTED["write_U_strict_retention"].items():
        check(
            "write_U_strict_retention",
            objective,
            out[(objective, "edit")]["U_strict_retention_world"],
            expected,
        )
    per_edit = module.write_target(path)
    flips = {}
    for objective in module.OBJECTIVES:
        rows = [
            rec
            for rec in data["case_records"]
            if rec["phase"] == "confirmation"
            and rec["arm"] == "edit"
            and rec["objective"] == objective
        ]
        flips[objective] = [
            sum(bool(rec["early"]["new_entity_accuracy"]) for rec in rows),
            len(rows),
        ]
    return out, per_edit["flip"], flips


def sequential(cache, sources, check):
    seq = json.loads(fetch(SEQUENTIAL, cache, sources).read_text())["history_aggregates"]
    rep = json.loads(fetch(REPLAY, cache, sources).read_text())["arm_aggregates"]
    out = {}
    for name, block in list(seq.items()) + list(rep.items()):
        e, s = block["endpoint"], block["stage_a"]
        out[name] = {
            "BB": 100 * e["BB"]["mean"],
            "AA": 100 * e["AA"]["mean"],
            "BB_counts": [e["BB"].get("pooled_correct"), e["BB"].get("pooled_prediction_n")],
            "AA_counts": [e["AA"].get("pooled_correct"), e["AA"].get("pooled_prediction_n")],
            "BB_before": 100 * s["BB"]["mean"],
            "AA_before": 100 * s["AA"]["mean"],
        }
    for k, v in EXPECTED["sequential_BB"].items():
        check("sequential_BB", k, out[k]["BB"], v)
    for k, v in EXPECTED["replay_BB"].items():
        check("replay_BB", k, out[k]["BB"], v)
    return out


# Figures ------------------------------------------------------------------------


def header(fig, title, subtitle=None):
    fig.text(0.01, 1.0, title, ha="left", va="bottom", fontsize=9.5, weight="bold")
    if subtitle:
        fig.text(0.01, 0.985, subtitle, ha="left", va="top", fontsize=7, color=INK2)


def legend_below(ax, ncol=2, offset=-0.22):
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, offset),
        ncol=ncol,
        fontsize=6.3,
        handlelength=1.2,
        columnspacing=1.0,
    )


def end_labels(ax, xs, ys, labels, gap=5.5):
    """Right-end labels nudged apart vertically, with leader lines when moved."""
    order = np.argsort(ys)[::-1]
    placed = []
    for i in order:
        y = ys[i]
        if placed and placed[-1] - y < gap:
            y = placed[-1] - gap
        placed.append(y)
        ax.annotate(
            labels[i],
            (xs[i], ys[i]),
            xytext=(xs[i] * 1.6, y),
            fontsize=6,
            color=INK2,
            va="center",
            arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6},
        )


def figure_f2(data, out):
    b, (load, traj) = data["bios"], data["curves"]
    fig = plt.figure(figsize=(7.2, 5.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.62, top=0.93)
    top = outer[0].subgridspec(1, 3, wspace=0.38)
    bottom = outer[1].subgridspec(1, 2, wspace=0.28)

    ax = fig.add_subplot(top[0])
    bars(
        ax,
        ["S", "M", "MP"],
        ["canonical", "held-out"],
        [[b[("canonical", c)], b[("heldout", c)]] for c in ("S", "M", "MP")],
        [BLUE, ORANGE],
        labels=["canonical QA", "5 held-out phrasings"],
        fmt="{:.2f}",
    )
    ax.set_ylabel("accuracy (%)")
    ax.set_title("a  bioS organisation")
    legend_below(ax, ncol=1)

    ax = fig.add_subplot(top[1])
    keys = ["pretrain_native", "adapt_native", "adapt_qa"]
    bars(
        ax,
        ["S", "M", "MP"],
        keys,
        [[b[("same", c, k)][0] for k in keys] for c in ("S", "M", "MP")],
        [NEUTRAL, BLUE, ORANGE],
        fmt="{:.0f}",
        labels=[
            "native continuation, pretrain end",
            "native continuation, QA-adapted",
            "QA, same QA-adapted weights",
        ],
    )
    ax.set_title("b  same weights, birth date")
    legend_below(ax, ncol=1)

    ax = fig.add_subplot(top[2])
    p = TRANSCRIBED["mquake_question_preview"]["values"]
    bars(
        ax,
        ["ROME", "MEMIT"],
        ["preview"],
        [[p["ROME"]], [p["MEMIT"]]],
        [NEUTRAL],
        hatches=["////"],
        width=0.5,
    )
    ax.set_title("c  MQuAKE cloze edit, asked")
    ax.text(
        0.5,
        1.0,
        "PREVIEW: partial records,\nnot frozen; formal value = S1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color=INK,
        bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": ORANGE, "lw": 0.8},
    )
    ax.text(
        0.5,
        -0.2,
        "successful cloze edits whose fact\nis answered as a question (%)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
        color=INK2,
    )

    ax = fig.add_subplot(bottom[0])
    ns = [128, 256, 512, 1024, 2048, 4096]
    for name, color, label in (
        ("exposure", BLUE, "equal 4000 exposures / fact"),
        ("compute", ORANGE, "equal 128k updates"),
    ):
        ax.plot(ns, [load[(name, n)] for n in ns], color=color, marker="o", ms=4, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("entities N (16N facts; each N a different closed graph)")
    ax.set_ylabel("atomic accuracy (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("d  knowledge load (3 worlds)")
    ax.legend(loc="lower left")

    ax = fig.add_subplot(bottom[1])
    xs, ys, labels = [], [], []
    for color, n in zip(SEQ, ns, strict=True):
        ex, mean, lo, hi = traj[n]
        keep = ex >= 1
        ax.plot(ex[keep], mean[keep], color=color, lw=1.4)
        ax.fill_between(ex[keep], lo[keep], hi[keep], color=color, alpha=0.12, lw=0)
        xs.append(ex[keep][-1])
        ys.append(mean[keep][-1])
        labels.append(f"N={n}")
    end_labels(ax, xs, ys, labels)
    ax.set_xscale("log")
    ax.set_xlim(1, 4000 * 6)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("exposures per atomic fact")
    ax.set_title("e  atomic learning by load (band = world range)")
    ax.grid(True, axis="both", color=GRID, lw=0.6)
    header(fig, "F2  Answering: how a fact is written decides which queries retrieve it")
    save(fig, "F2-answering", out)


def figure_f3(data, out):
    use, grok_runs, address = data["usage"], data["grok"], data["address"]
    ye = TRANSCRIBED["ye_second_hop"]["values"]
    fig = plt.figure(figsize=(7.2, 7.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.9], hspace=0.42, top=0.94)
    top = outer[0].subgridspec(1, 2, wspace=0.25)
    lower = outer[1].subgridspec(2, 4, hspace=0.18, wspace=0.12)

    ax = fig.add_subplot(top[0])
    keys = ["experience", "matched_repetition", "other_repetition", "base_only"]
    labels = [
        "composition\npractice",
        "matched atom\nrepetition",
        "other-group\nrepetition",
        "base atoms\nonly",
    ]
    bars(
        ax,
        labels,
        ["held-out same-group"],
        [[use[k]] for k in keys],
        [BLUE],
        width=0.55,
        fmt="{:.2f}",
    )
    ax.set_ylabel("held-out composition (%)")
    ax.set_title("a  usage experience (3 worlds, 128k)")
    ax.tick_params(axis="x", labelsize=6.5)

    ax = fig.add_subplot(top[1])
    vals = [
        [
            100 * ye[a]["test_ii"][0] / ye[a]["test_ii"][1],
            100 * ye[a]["test_ii_sr"][0] / ye[a]["test_ii_sr"][1],
        ]
        for a in ("only_ii", "full_atomic")
    ]
    bars(
        ax,
        ["Train-II only", "+ all atomic supervision"],
        ["Test-II", "Test-II-SR"],
        vals,
        [BLUE, ORANGE],
        fmt="{:.2f}",
        labels=["Test-II", "restricted second hop (2/3000)"],
    )
    ax.text(
        1.03,
        14,
        "single hop of\nrestricted facts:\n2996/3000 ID",
        ha="left",
        fontsize=6,
        color=INK2,
    )
    ax.set_title("b  Ye 2025 Sec. 2.4 reproduction (seed 42)")
    ax.tick_params(axis="x", labelsize=6.5)
    legend_below(ax, ncol=2, offset=-0.14)

    colors = [BLUE, ORANGE, AQUA, YELLOW]
    first = None
    for row, arch in enumerate(("standard8", "loop4x2")):
        for col, (phi, wd) in enumerate(GROK_CONDITIONS):
            ax = fig.add_subplot(lower[row, col], sharey=first)
            first = first or ax
            steps, values = grok_runs[(arch, phi, wd)]
            keep = steps >= 1000
            for p, label, color in zip(PANELS, PANEL_LABELS, colors, strict=True):
                ax.plot(steps[keep], values[p][keep], color=color, lw=1.2, label=label)
            ax.set_xscale("log")
            ax.set_ylim(-3, 103)
            ax.set_xticks([1e3, 1e4, 1e5, 1e6], ["1k", "10k", "100k", "1M"])
            ax.minorticks_off()
            ax.grid(True, axis="both", color=GRID, lw=0.6)
            if col == 0:
                ax.set_ylabel(f"{ARCH_LABEL[arch]}\naccuracy (%)")
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(
                    ("c  " if col == 0 else "") + f"phi={phi}, wd={wd}",
                    fontweight="bold" if col == 0 else "normal",
                    loc="left",
                )
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("training step")
            end = {p: values[p][-1] for p in PANELS}
            ax.text(
                0.04,
                0.95 if row == 0 else 0.6,
                f"1.5M: IO {end['test_io']:.1f}, OO {end['test_oo']:.1f}",
                transform=ax.transAxes,
                fontsize=5.8,
                color=INK2,
                va="top",
            )
            if row == 0 and col == 0:
                ax.legend(
                    loc="center left",
                    fontsize=6,
                    handlelength=1.2,
                    bbox_to_anchor=(0.0, 0.55),
                    ncol=2,
                )
    fig.text(
        0.01,
        0.6,
        "Wang 2024 graph (2000 entities, 8 layers, width 768), four "
        "test panels (first/second hop ID or OOD) over 1.5M steps; one world, one init",
        fontsize=7,
        color=INK2,
    )
    header(
        fig,
        "F3  Composition: a fact is usable in the roles it was written in; sharing "
        "makes the single-hop path the second-hop path",
    )
    save(fig, "F3-composition", out)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, arch in zip(axes, ("standard8", "loop4x2"), strict=True):
        vals = [
            [100 * address[f"{arch}-shared"][p], 100 * address[f"{arch}-unique"][p]] for p in PANELS
        ]
        bars(
            ax,
            PANEL_LABELS,
            ["shared", "unique"],
            vals,
            [NEUTRAL, ARCH[arch]],
            labels=[
                "shared head address (out-degree ~8)",
                f"unique head address (out-degree 1), {ARCH_LABEL[arch]}",
            ],
            fmt="{:.1f}",
        )
        ax.set_title(ARCH_LABEL[arch], loc="center", fontweight="normal")
        legend_below(ax, ncol=1, offset=-0.14)
    axes[0].set_ylabel("accuracy at 128k (%)")
    fig.subplots_adjust(top=0.84)
    n = {p: address["standard8-shared"][f"{p}_n"] for p in PANELS}
    header(
        fig,
        "F3d  Address experiment (small synthetic graph, width 128, one init, 128k): "
        "first-hop head-entity out-degree",
        f"panel n = {'/'.join(str(n[p]) for p in PANELS)} (II/IO/OI/OO); unique-address "
        "OOD first-hop heads never appear in training compositions",
    )
    save(fig, "F3d-address", out)


def figure_f4(data, out):
    re_, ind, bridge = data["reencode"], data["independent"], data["bridge"]
    fig = plt.figure(figsize=(7.2, 5.8))
    outer = fig.add_gridspec(2, 1, hspace=0.55, top=0.93)
    top = outer[0].subgridspec(1, 2, width_ratios=[3, 2], wspace=0.22)
    ax = fig.add_subplot(top[0])
    arms = ["baseline", "bridge_ce", "aligned"]
    conds = ["baseline", "self_decode", "oracle", "wrong_entity"]
    names = ["full-text CE", "+ first-hop CE", "+ first-hop CE\n+ alignment"]
    ticks = [
        f"{n}\nslot readout {re_[(a, 'baseline')]['readout']:.0f}%"
        for n, a in zip(names, arms, strict=True)
    ]
    bars(
        ax,
        ticks,
        conds,
        [[re_[(a, c)]["strict"] for c in conds] for a in arms],
        [NEUTRAL, BLUE, AQUA, ORANGE],
        fmt="{:.1f}",
        width=0.88,
        label_size=5.4,
        labels=[
            "native",
            "re-encode self-decoded bridge (alpha=1)",
            "inject true bridge",
            "inject wrong bridge",
        ],
    )
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_ylabel("strict composition (%)")
    ax.set_title("a  shared block x2, 3 worlds x 2 inits")
    ax.legend(loc="upper left", fontsize=6)

    ax = fig.add_subplot(top[1])
    vals = [
        [100 * ind[a]["familiar_test"], 100 * ind[a]["strict_test"]]
        for a in ("bridge_ce", "aligned")
    ]
    ticks = [
        f"first-hop CE\ncos {ind['bridge_ce']['strict_test_cosine']:.4f}",
        f"+ alignment\ncos {ind['aligned']['strict_test_cosine']:.4f}",
    ]
    bars(
        ax,
        ticks,
        ["familiar", "strict"],
        vals,
        [BLUE, ORANGE],
        fmt="{:.2f}",
        labels=["familiar new chains", "strict"],
    )
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_title("b  untied 2-layer, same targets")
    ax.legend(loc="upper left", fontsize=6)

    ax = fig.add_subplot(outer[1])
    cats = [
        ("native_oi", "native OI"),
        ("erase_bridge_oi", "OI, bridge\ncomponent erased"),
        ("native_ii", "native II"),
        ("erase_bridge_ii", "II, bridge\ncomponent erased"),
        ("swap_x2_ood_first", "OOD first hop,\nx2 swap to other\nbridge: converted"),
        ("random_swap_ood_first", "same swap,\nrandom rotation"),
    ]
    for ci, (key, _) in enumerate(cats):
        for ai, arch in enumerate(("standard8", "loop4x2")):
            runs = sorted(r for r in bridge if r.startswith(arch))
            ys = [100 * bridge[r][key]["accuracy"] for r in runs]
            low = ["phi3.6" in r for r in runs]
            xs = ci + (ai - 0.5) * 0.34 + np.linspace(-0.09, 0.09, len(ys))
            for x, y, lw in zip(xs, ys, low, strict=True):
                ax.scatter(
                    x,
                    y,
                    s=16,
                    facecolor=SURFACE if lw else ARCH[arch],
                    edgecolor=ARCH[arch],
                    linewidth=0.9,
                    zorder=3,
                )
    for arch in ("standard8", "loop4x2"):
        ax.scatter([], [], s=16, color=ARCH[arch], label=ARCH_LABEL[arch])
    ax.scatter([], [], s=16, facecolor=SURFACE, edgecolor=MUTED, label="phi=3.6 (hollow)")
    ax.set_xticks(range(len(cats)), [c[1] for c in cats], fontsize=6.3)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(-3, 105)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_ylabel("accuracy or converted (%)")
    ax.set_title(
        "c  Wang graph: r1 slot after layer 4 (Loop: end of pass 1), input-embedding "
        "direction; 12 endpoints per architecture"
    )
    ax.legend(loc="center right", fontsize=6.3, bbox_to_anchor=(1.0, 0.55))
    header(
        fig, "F4  Readable is not usable: format repair works only where the lookup path is shared"
    )
    save(fig, "F4-readable-vs-usable", out)


def figure_f5(data, out):
    inter, (wt, flip, flips) = data["interface"], data["write_target"]
    mq = TRANSCRIBED["mquake_hop"]["values"]
    fig = plt.figure(figsize=(7.2, 5.9))
    outer = fig.add_gridspec(2, 1, hspace=0.75, top=0.93)
    top = outer[0].subgridspec(1, 2, wspace=0.22)
    bottom = outer[1].subgridspec(1, 3, wspace=0.42)
    ax = fig.add_subplot(top[0])
    keys = [
        ("E_new", "edited atom:\nnew answer"),
        ("U_atomic_retention", "unrelated atoms\nkept"),
        ("D", "derived answers\nupdated (D)"),
    ]
    bars(
        ax,
        [k[1] for k in keys],
        ["aligned", "bridge_ce"],
        [[inter["aligned"][k], inter["bridge_ce"][k]] for k, _ in keys],
        [BLUE, ORANGE],
        labels=["aligned parent", "first-hop CE parent"],
    )
    ax.set_title("a  strong parent, local MLP edit")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(loc="center right", fontsize=6, bbox_to_anchor=(1.0, 0.45))

    ax = fig.add_subplot(top[1])
    objectives = ["final_ce", "early_ce", "early_ce_align"]
    vals = [
        [
            100 * flips[o][0] / flips[o][1],
            100 * wt[(o, "edit")]["D_new"][0] / wt[(o, "edit")]["D_new"][1],
            100 * wt[(o, "edit")]["D_old"][0] / wt[(o, "edit")]["D_old"][1],
        ]
        for o in objectives
    ]
    bars(
        ax,
        ["final CE", "+ early CE", "+ early CE\n+ alignment"],
        ["flip", "D_new", "D_old"],
        vals,
        [NEUTRAL, BLUE, ORANGE],
        fmt="{:.0f}",
        labels=[
            "pass-1 readout flipped (/24)",
            "derived new answer (/96)",
            "old derived answer kept (/96)",
        ],
    )
    ax.set_title("b  same new fact, three write targets")
    ax.tick_params(axis="x", labelsize=6.5)
    legend_below(ax, ncol=2, offset=-0.2)

    ax = fig.add_subplot(bottom[0])
    vals = [
        [
            100 * flip["flipped"]["d_new"] / flip["flipped"]["d_queries"],
            100 * flip["flipped"]["sham_d_new"] / flip["flipped"]["d_queries"],
        ],
        [
            100 * flip["not_flipped"]["d_new"] / flip["not_flipped"]["d_queries"],
            100 * flip["not_flipped"]["sham_d_new"] / flip["not_flipped"]["d_queries"],
        ],
    ]
    bars(
        ax,
        ["flipped\n(48 edits)", "not flipped\n(24 edits)"],
        ["edit", "sham"],
        vals,
        [BLUE, NEUTRAL],
        labels=["edit", "same-case sham"],
    )
    ax.set_title("c  D by readout flip")
    ax.set_ylabel("derived new answer (%)")
    ax.legend(loc="upper right", fontsize=6)

    ax = fig.add_subplot(bottom[1])
    for i, o in enumerate(objectives):
        edit = wt[(o, "edit")]["U_strict_retention_world"]
        sham = wt[(o, "sham")]["U_strict_retention_world"]
        ax.plot([i, i], [sham, edit], color=AXIS, lw=1, zorder=1)
        ax.scatter(i, sham, s=22, color=NEUTRAL, zorder=2, label="sham" if i == 0 else None)
        ax.scatter(i, edit, s=22, color=BLUE, zorder=3, label="edit" if i == 0 else None)
        ax.text(i + 0.14, edit, f"{edit:.1f}", fontsize=6, va="center", color=INK2)
    ax.set_xticks(range(3), ["final\nCE", "+early\nCE", "+align"])
    ax.set_xlim(-0.5, 2.8)
    ax.set_ylim(86, 100.5)
    ax.set_ylabel("parent-correct U strict kept (%)")
    ax.set_title("d  interference")
    ax.legend(loc="lower left", fontsize=6)

    ax = fig.add_subplot(bottom[2])
    for i, editor in enumerate(("ROME", "MEMIT")):
        for j, (key, color, label) in enumerate(
            (("direct", BLUE, "direct answer"), ("cot", ORANGE, "CoT"))
        ):
            est, lo, hi = mq[editor][key]
            x = i + (j - 0.5) * 0.3
            ax.errorbar(
                x,
                est,
                yerr=[[est - lo], [hi - est]],
                fmt="o",
                color=color,
                ms=4,
                capsize=2,
                lw=1,
                label=label if i == 0 else None,
            )
        shrink = mq[editor]["shrink"][0]
        ax.text(
            i,
            3,
            f"CoT shrink\n{shrink:.2f} pp"
            if abs(shrink) >= 0.01
            else f"CoT shrink\n{shrink:.3f} pp",
            ha="center",
            fontsize=6,
            color=INK2,
        )
    ax.set_xticks([0, 1], ["ROME", "MEMIT"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 38)
    ax.set_ylabel("first-hop minus later-hop (pp)")
    ax.set_title("e  MQuAKE, GPT-J")
    ax.legend(loc="upper center", fontsize=6, ncol=2)
    header(
        fig,
        "F5  Updates: an edit reaches only the uses whose read path passes the written location",
    )
    save(fig, "F5-updates", out)


def figure_f6(data, out):
    s = data["sequential"]
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [4, 4, 2.6], "wspace": 0.3}
    )
    hist = ["sequential_composition", "joint", "sequential_composition_sham", "sequential_atomic"]
    labels = ["A comp.\n-> B atoms", "joint\nshuffle", "A-only\nsham", "A atoms\n-> B atoms"]
    for ax, key, title in (
        (axes[0], "BB", "a  new+new composition BB"),
        (axes[1], "AA", "b  old held-out composition AA"),
    ):
        vals = [[s[h][f"{key}_before"] if h != "joint" else None, s[h][key]] for h in hist]
        bars(
            ax,
            labels,
            ["before", "end"],
            vals,
            [NEUTRAL, BLUE],
            labels=["before stage B", "end"],
            ymax=50,
        )
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=6.3)
        ax.legend(loc="upper right", fontsize=6)
    axes[0].set_ylabel("accuracy (%), 384 evaluations")
    ax = axes[2]
    vals = [[s[a]["BB"], s[a]["AA"]] for a in ("atom_replay", "full_replay")]
    bars(
        ax,
        ["atom\nreplay", "full\nreplay"],
        ["BB", "AA"],
        vals,
        [BLUE, ORANGE],
        ymax=50,
        labels=["BB", "AA"],
    )
    ax.axhline(s["full_replay"]["BB_before"], color=MUTED, lw=0.8)
    ax.text(-0.45, s["full_replay"]["BB_before"] + 1, "BB before", fontsize=6, color=MUTED)
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_title("c  replay, same parents")
    ax.legend(loc="upper left", fontsize=6)
    header(
        fig,
        "F6  New use follows the write context; interference follows timing and replay",
        "2Wiki, three splits of one real graph x 2 inits; B facts only ever trained as atoms",
    )
    save(fig, "F6-continual", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    sources, check = {}, Checks()
    module = reanalysis()
    wt_path = fetch(module.WRITE_TARGET, args.cache, sources)
    bw_path = fetch(module.BRIDGE, args.cache, sources)
    address_paths = {
        run: fetch(module.ADDRESS.format(run=run), args.cache, sources)
        for run in module.ADDRESS_RUNS
    }
    data = {
        "bios": bios(args.cache, sources, check),
        "curves": curves(args.cache, sources, check),
        "usage": usage(args.cache, sources, check),
        "grok": grok(args.cache, sources, check),
        "address": module.address(address_paths),
        "reencode": reencode(args.cache, sources, check),
        "independent": independent(args.cache, sources, check),
        "bridge": module.bridge(bw_path),
        "interface": interface(args.cache, sources, check),
        "write_target": write_target(wt_path, module, check),
        "sequential": sequential(args.cache, sources, check),
    }
    figure_f2(data, args.output)
    figure_f3(data, args.output)
    figure_f4(data, args.output)
    figure_f5(data, args.output)
    figure_f6(data, args.output)
    grok_files = [
        GROK.format(run=f"{a}-phi{p}-wd{w}")
        for a in ("standard8", "loop4x2")
        for p, w in GROK_CONDITIONS
    ]
    panels = {
        "F2a": {
            "source": [BIOS_OVERALL],
            "computed": "adapt/test/no-CoT six_attributes, mean of 2 worlds",
        },
        "F2b": {"source": [BIOS_SAME], "computed": "birth date; pretrain/adapt endpoints"},
        "F2c": {"transcribed": "mquake_question_preview"},
        "F2d": {
            "source": [CURVE_BUDGETS],
            "computed": "confirmation load runs + support "
            "phi=7.2 (N=128), mean of 3 worlds at 'exposure'/'compute' budgets",
        },
        "F2e": {"source": [CURVE_NODES], "computed": "all nodes, same runs; band = world range"},
        "F3a": {"source": [USAGE], "computed": "128k endpoint, equal cohorts, 3 worlds"},
        "F3b": {"transcribed": "ye_second_hop"},
        "F3c": {"source": grok_files, "computed": "every evaluation node (3000-query panels)"},
        "F3d": {
            "source": [module.ADDRESS.format(run=r) for r in module.ADDRESS_RUNS],
            "computed": "endpoint answer_accuracy via reanalyze_history_20261010.address",
        },
        "F4a": {"source": [REENCODE], "computed": "norm_matched, alpha=1, world-equal"},
        "F4b": {"source": [INDEPENDENT], "computed": "confirmation means"},
        "F4c": {"source": [module.BRIDGE], "computed": "reanalyze_history_20261010.bridge"},
        "F5a": {
            "source": [INTERFACE],
            "computed": "node 512, mean of 4 D cells (first/second x familiar/strict), world-equal",
        },
        "F5b": {"source": [module.WRITE_TARGET], "computed": "pooled counts per objective"},
        "F5c": {
            "source": [module.WRITE_TARGET],
            "computed": "reanalyze_history_20261010.write_target flip groups",
        },
        "F5d": {"source": [module.WRITE_TARGET], "computed": "world-equal U_strict retention"},
        "F5e": {"transcribed": "mquake_hop"},
        "F6a-c": {
            "source": [SEQUENTIAL, REPLAY],
            "computed": "split-equal means of "
            "initialization means; joint has no A-only stage, so no 'before' bar",
        },
    }
    record = {
        "repository": REPO,
        "revision": REVISION,
        "script": "scripts/plot_paper_figures_draft.py",
        "sources_sha256": dict(sorted(sources.items())),
        "panels": panels,
        "transcribed": TRANSCRIBED,
        "consistency_checks": check.rows,
        "all_checks_match": all(r["match"] for r in check.rows),
    }
    (args.output / "sources.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {"checks": len(check.rows), "mismatches": [r for r in check.rows if not r["match"]]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
