#!/usr/bin/env python3
"""Read-only, reproducible atlas of the October gate comparisons.

Snapshots JSON outputs only; never loads models or changes experiment state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "lm-memory-persistent/workspace/llm-memory-editability"
EXT = BASE / "gate-generalization-extension-v1/execution-v2/cases"
INK, MUTED = "#183247", "#586d80"
COLORS = {
    "loop": "#788694",
    "disco": "#a47743",
    "paper": "#273f59",
    "fixed90": "#73a8cb",
    "learn90": "#007da5",
    "half90": "#8753a8",
    "fixed95": "#d7a32b",
    "learn95": "#d66428",
    "joint": "#008c78",
    "candidate": "#b74669",
    "scalar": "#5275b8",
    "dual": "#9e6a99",
    "vector-injection": "#7c8744",
    "compact": "#8c54b7",
}
LABELS = {
    "loop": "普通 Loop",
    "disco": "无学习门 DiscoLoop",
    "paper": "原论文门 P",
    "fixed90": "固定更新门 0.90",
    "learn90": "学习更新门 0.90",
    "half90": "学习门 0.90 · 门学习率减半",
    "fixed95": "固定更新门 0.95",
    "learn95": "学习更新门 0.95",
    "joint": "共同门 0.95",
    "candidate": "融合后候选门 0.95",
    "scalar": "G：标量注入门",
    "dual": "D：逐维连续门＋注入",
    "vector-injection": "V：逐维注入门",
    "compact": "紧凑共同门 · 一层×四轮",
}
MULTI = [
    "atomic_id",
    "atomic_ood",
    "train_2",
    "id_2",
    "ood_2",
    "train_3",
    "id_3",
    "ood_3",
    "train_4",
    "id_4",
    "ood_4",
]
WANG = [
    "atomic_id",
    "atomic_ood",
    "atomic_all",
    "train_ii",
    "test_ii",
    "test_oi",
    "test_io",
    "test_oo",
]
PN = {
    "atomic_id": "图A原子",
    "atomic_ood": "图B原子",
    "atomic_all": "全部原子",
    "train_2": "训练两跳",
    "id_2": "图A未见两跳",
    "ood_2": "图B两跳",
    "train_3": "训练三跳",
    "id_3": "图A未见三跳",
    "ood_3": "图B三跳",
    "train_4": "训练四跳",
    "id_4": "图A未见四跳",
    "ood_4": "图B四跳",
    "train_ii": "Train-II",
    "test_ii": "Test-II",
    "test_oi": "OI",
    "test_io": "IO",
    "test_oo": "OO",
}
DEV = [
    "loop",
    "disco",
    "paper",
    "fixed90",
    "learn90",
    "half90",
    "fixed95",
    "learn95",
    "joint",
    "candidate",
]
ARM = {"paper": "paper", "fixed-update": "fixed95", "token-update": "learn95"}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def specs():
    cont = BASE / "paper-update-gate-continuation-v1/execution/runs"
    old90 = ROOT / "results/paper-update-gate-multihop-v1/execution/runs"
    old95 = BASE / "paper-update-gate-multihop-u95-v1/runs"
    old = BASE / "loop-hop-unified-v1/runs"
    dual = BASE / "dual-gate-multihop-v1/attempts/preflight-repair-v2/runs"
    result = []

    def add(key, family, variant, path, budget=1500000, parent=None, repeats=4):
        result.append(
            dict(
                key=key,
                family=family,
                variant=variant,
                path=str(path),
                budget=budget,
                parent=str(parent) if parent else None,
                repeats=repeats,
            )
        )

    for key, name in [("loop", "loop-l1-r4"), ("disco", "disco-l1-r4")]:
        add(
            key,
            "development",
            key,
            BASE / "ungated-loop-continuation-v1/runs" / name,
            parent=old / name,
        )
    for key, name, parent in [
        ("paper", "paper", dual / "paper"),
        ("fixed90", "fixed-update", old90 / "fixed-update"),
        ("learn90", "token-update", old90 / "token-update"),
        ("half90", "token-update-half-lr", old90 / "token-update"),
    ]:
        add(key, "development", key, cont / name, parent=parent)
    for key, name in [("fixed95", "fixed-update"), ("learn95", "token-update")]:
        add(
            key,
            "development",
            key,
            EXT / "continuation/runs" / name,
            parent=old95 / name,
        )
    add(
        "joint",
        "development",
        "joint",
        BASE / "joint-update-gate-multihop-v1/runs/joint-token",
    )
    add(
        "candidate",
        "development",
        "candidate",
        BASE / "candidate-update-gate-gpu1-v3/runs/candidate-token",
    )
    for name in ["scalar", "dual", "vector-injection"]:
        add(name, "historical", name, dual / name, budget=300000)
    for world in range(1, 4):
        for arm, variant in ARM.items():
            add(
                f"new{world}-{variant}",
                f"new{world}",
                variant,
                EXT / f"confirmation-{world}/runs/{arm}",
                budget=300000,
            )
            path = (
                BASE / "gate-depth-long-gpu4-v1/execution/cases/depth-1/runs" / arm
                if world == 1
                else EXT / f"depth-{world}/runs/{arm}"
            )
            add(
                f"depth{world}-{variant}",
                f"depth{world}",
                variant,
                path,
                budget=1500000 if world == 1 else 300000,
            )
    for arm, variant in ARM.items():
        add(f"wang-{variant}", "wang", variant, EXT / f"wang/runs/{arm}", repeats=2)
    add(
        "wang-joint",
        "wang",
        "joint",
        BASE / "joint-update-gate-wang-v1/runs/joint-token",
        repeats=2,
    )
    add(
        "compact",
        "compact",
        "compact",
        BASE / "compact-wang-joint-gate-v1/runs/joint-token",
    )
    return result


def collect(out):
    sources, runs, checks = [], {}, []

    def read(path, key, name):
        raw = path.read_bytes()
        value = json.loads(raw)
        dest = out / "sources" / key / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        sources.append(
            dict(
                path=str(path.resolve()),
                snapshot=str(dest.relative_to(out)),
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
                read_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        return value

    for spec in specs():
        key, path = spec["key"], Path(spec["path"])
        run = dict(spec, nodes=[], grid=None, status=dict(state="pending", step=0))
        runs[key] = run
        if not (path / "status.json").exists():
            continue
        for filename, field in [
            ("status.json", "status"),
            ("run.json", "meta"),
            ("learning.json", "nodes"),
            ("grid.json", "grid"),
        ]:
            if (path / filename).exists():
                run[field] = read(path / filename, key, filename)
        assert not run["meta"].get("engineering_only"), key
        nodes = run["nodes"]
        assert [n["step"] for n in nodes] == sorted({n["step"] for n in nodes}), key
        if spec["parent"]:
            parent = Path(spec["parent"])
            pm = read(parent / "run.json", key, "parent-run.json")
            pn = read(parent / "learning.json", key, "parent-learning.json")
            for field in ["world_sha256", "panels_sha256"]:
                assert pm[field] == run["meta"][field], (key, field)
            previous = {n["step"]: n for n in pn if n["step"] <= 300000}
            for n in nodes:
                if n["step"] in previous:
                    for panel, metric in n["metrics"].items():
                        for score in ["answer_accuracy", "accuracy"]:
                            assert metric[score] == previous[n["step"]]["metrics"][panel][score], (
                                key
                            )
                previous[n["step"]] = n
            run["nodes"] = [previous[s] for s in sorted(previous)]
            checks.append(f"{key}: parent data identity and overlapping scores agree")
        for n in run["nodes"]:
            for metrics in n["metrics"].values():
                for score in ["answer_accuracy", "accuracy"]:
                    assert math.isfinite(metrics[score]) and 0 <= metrics[score] <= 1, key
        if run["status"]["state"] == "complete":
            st, grid = run["status"], run["grid"]
            assert st["step"] == spec["budget"], key
            assert grid is not None and grid["step"] == st["step"], key
            assert grid["model_sha256"] == st["model_sha256"], key
            assert grid["training_repeats"] == spec["repeats"], key
            last = run["nodes"][-1]
            assert last["step"] == st["step"], key
            for panel, metrics in last["metrics"].items():
                for score in ["answer_accuracy", "accuracy"]:
                    assert metrics[score] == st["endpoint"]["metrics"][panel][score], key
                    assert metrics[score] == grid["metrics"][str(spec["repeats"])][panel][score], (
                        key
                    )
            checks.append(f"{key}: endpoint/status/training-repeat grid and model hashes agree")
    for keys in [DEV] + [[f"new{i}-{v}" for v in ARM.values()] for i in range(1, 4)]:
        for field in ["world_sha256", "panels_sha256"]:
            assert len({runs[k]["meta"][field] for k in keys}) == 1, (keys, field)
    checks.append("Development and each new-graph comparison use identical evaluation panels")
    return dict(
        captured_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs,
        sources=sources,
        checks=checks,
    )


def score_name(run):
    return "accuracy" if run["family"] in ["wang", "compact"] else "answer_accuracy"


def value(run, panel, step, score=None):
    # Missing or unfinished endpoints remain NaN, never zero or carried forward.
    if step == run["budget"] and run["status"]["state"] != "complete":
        return np.nan
    for n in run["nodes"]:
        if n["step"] == step and panel in n["metrics"]:
            return 100 * n["metrics"][panel][score or score_name(run)]
    return np.nan


def label(run, status=False):
    name = LABELS[run["variant"]]
    if status and run["status"]["state"] != "complete":
        last = run["nodes"][-1]["step"] / 10000 if run["nodes"] else 0
        name += f"（评测至{last:g}万步）" if last else "（待运行）"
    return name


def style():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#c5d1da",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


class Atlas:
    def __init__(self, out, data):
        self.out, self.data, self.runs = out, data, data["runs"]
        dt = datetime.fromisoformat(data["captured_utc"]).astimezone(ZoneInfo("America/Chicago"))
        self.stamp = dt.strftime("快照 %Y-%m-%d %H:%M CDT")
        self.pages = []
        self.pdf = PdfPages(out / "complete-atlas.pdf")
        self.main_pdf = PdfPages(out / "comparison-summary.pdf")

    def save(self, fig, slug, title, subtitle, main=False, note=None, legend=None):
        fig.text(0.045, 0.967, title, va="top", fontsize=22, weight="bold")
        fig.text(0.045, 0.914, subtitle, va="top", fontsize=11, color=MUTED)
        if legend:
            fig.legend(
                handles=legend,
                loc="upper center",
                bbox_to_anchor=(0.51, 0.867),
                frameon=False,
                ncol=min(3, len(legend)),
                fontsize=11,
            )
        note = (
            note or "点为全部已保存评测；直线仅连接相邻记录，无平滑。单图/初始化不能证明普遍机制。"
        )
        fig.text(0.045, 0.055, note, va="top", fontsize=10, color=MUTED)
        fig.text(0.97, 0.015, self.stamp, ha="right", fontsize=9, color=MUTED)
        fig.savefig(self.out / f"{slug}.png", dpi=180)
        fig.savefig(self.out / f"{slug}.pdf")
        self.pdf.savefig(fig)
        if main:
            self.main_pdf.savefig(fig)
        self.pages.append(dict(slug=slug, title=title, subtitle=subtitle, main=main, note=note))
        plt.close(fig)

    def curves(
        self,
        keys,
        panels,
        title,
        slug,
        subtitle,
        main=False,
        grid=False,
        cost=False,
        zoom=False,
    ):
        rows = math.ceil(len(panels) / 3)
        fig, axes = plt.subplots(rows, 3, figsize=(16, 5.8 if rows == 1 else 9.2), squeeze=False)
        fig.subplots_adjust(
            left=0.065,
            right=0.97,
            bottom=0.18 if rows == 1 else 0.13,
            top=0.72 if rows == 1 else 0.77,
            hspace=0.50,
            wspace=0.25,
        )
        legend = []
        for key in keys:
            r = self.runs[key]
            color = COLORS[r["variant"]]
            linestyle = "--" if r["status"]["state"] != "complete" else "-"
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    lw=2.4,
                    ls=linestyle,
                    marker="o",
                    ms=4,
                    label=(
                        label(r) + "（无终点，未绘制）"
                        if grid and not r["grid"]
                        else label(r, True)
                    ),
                )
            )
        for ax, panel in zip(axes.flat, panels, strict=False):
            for key in keys:
                r, sc = self.runs[key], score_name(self.runs[key])
                if grid:
                    if not r["grid"]:
                        continue
                    points = sorted(
                        (int(n), m[panel][sc] * 100)
                        for n, m in r["grid"]["metrics"].items()
                        if panel in m
                    )
                else:
                    points = [
                        (
                            n["estimated_matmul_training_flops"] / 1e18
                            if cost
                            else n["step"] / 10000,
                            n["metrics"][panel][sc] * 100,
                        )
                        for n in r["nodes"]
                        if panel in n["metrics"]
                    ]
                if not points:
                    continue
                xs, ys = zip(*points, strict=True)
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[r["variant"]],
                    lw=2,
                    marker="o",
                    ms=3.5,
                    ls="--" if r["status"]["state"] != "complete" else "-",
                )
                if r["status"]["state"] != "complete":
                    ax.scatter(
                        xs[-1],
                        ys[-1],
                        s=65,
                        facecolor="white",
                        edgecolor=COLORS[r["variant"]],
                        zorder=5,
                    )
            panel_title = PN[panel]
            if self.runs[keys[0]]["family"] in ["wang", "compact"]:
                panel_title = {"atomic_id": "ID原子", "atomic_ood": "OOD原子"}.get(
                    panel, panel_title
                )
            ax.set_title(panel_title, loc="left", pad=12)
            ax.set_ylim(-2, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            if zoom:
                ax.set_ylim(-0.01, 0.5)
                ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax.set_ylabel("准确率（%）")
            ax.grid(alpha=0.18)
            if grid:
                ticks = sorted(
                    {
                        int(n)
                        for k in keys
                        if self.runs[k]["grid"]
                        for n in self.runs[k]["grid"]["metrics"]
                    }
                )
                ax.set_xticks(ticks)
                ax.set_xlabel("推理轮数 R")
            elif cost:
                ax.set_xlabel("累计估算矩阵训练 FLOPs（×10^18）")
                ax.set_xlim(left=0)
            else:
                limit = max(self.runs[k]["budget"] for k in keys) / 10000
                ax.set_xlim(0, limit)
                ax.set_xticks([0, 30, 60, 90, 120, 150] if limit == 150 else [0, 10, 20, 30])
                ax.set_xlabel("累计训练步数（万）")
                if limit > 30:
                    ax.axvline(30, color="#aab7c3", lw=1, ls=":")
        for ax in list(axes.flat)[len(panels) :]:
            ax.axis("off")
        note = (
            "固定训练终点；改变推理轮数会改变计算量。预定主读数仍为训练轮数，不能选最好轮数替代。"
            if grid
            else "保留全部评测节点，无平滑；空心末点/虚线为未完成训练，未绘制未来结果。"
        )
        if cost:
            note = (
                "矩阵 FLOPs 为估算，不含 softmax、归一化等；不等于实际耗时。"
                "续训包含原30万步累计成本。"
            )
        if zoom:
            note = "局部放大：纵轴为0–0.5%，不是0–100%。完整量程图同时保留；这些分数仍接近零。"
        self.save(fig, slug, title, subtitle, main, note=note, legend=legend)

    def heatmap(self, keys, panels, step, title, slug, subtitle, main=False):
        vals = np.array([[value(self.runs[k], p, step) for p in panels] for k in keys])
        width = max(12, 5 + len(panels) * 0.88)
        height = max(6, 2.8 + len(keys) * 0.46)
        fig, ax = plt.subplots(figsize=(width, height))
        fig.subplots_adjust(left=0.27, right=0.94, bottom=0.22, top=0.82)
        cmap = LinearSegmentedColormap.from_list("accuracy", ["#f3f6f9", "#a5d5df", "#006b80"])
        cmap.set_bad("#e3e7ec")
        ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(panels)), [PN[p] for p in panels], rotation=25, ha="right")
        ax.set_yticks(range(len(keys)), [label(self.runs[k]) for k in keys])
        ax.tick_params(length=0, pad=9)
        for (i, j), v in np.ndenumerate(vals):
            ax.text(
                j,
                i,
                "待完成" if np.isnan(v) else f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=11.5,
                color="white" if v >= 70 else INK,
            )
        for side in ax.spines.values():
            side.set_visible(False)
        self.save(
            fig,
            slug,
            title,
            subtitle,
            main,
            note="单元格为准确率（%），统一0–100色阶；灰格为无该预算终点，不作零分。\n"
            "近期批次未另做独立重载/云端终点审计；本图仅核对已保存结果的一致性。",
        )

    def budget_compare(self, panel):
        fig, ax = plt.subplots(figsize=(13, 8))
        fig.subplots_adjust(left=0.29, right=0.74, bottom=0.17, top=0.82)
        for i, k in enumerate(DEV):
            a, b = (value(self.runs[k], panel, s) for s in [300000, 1500000])
            if i % 2 == 0:
                ax.axhspan(i - 0.44, i + 0.44, color="#f0f4f7")
            ax.plot([a, b], [i, i], color="#b3c1ce", lw=3)
            ax.scatter(a, i, s=72, facecolor="white", edgecolor="#71869b", zorder=3)
            ax.scatter(b, i, s=68, marker="s", color="#007a88", zorder=4)
            for x, v in [(1.16, a), (1.44, b)]:
                ax.text(
                    x,
                    i,
                    "—" if np.isnan(v) else f"{v:.2f}%",
                    ha="center",
                    va="center",
                    transform=ax.get_yaxis_transform(),
                    fontsize=12,
                )
        ax.set_yticks(range(len(DEV)), [label(self.runs[k]) for k in DEV])
        ax.set_ylim(len(DEV) - 0.5, -0.5)
        ax.set_xlim(-2, 103)
        ax.set_xlabel("答案准确率（%）")
        ax.grid(axis="x", alpha=0.18)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        for x, text in [(1.16, "30万步 ○"), (1.44, "150万步 ■")]:
            ax.text(x, 1.04, text, transform=ax.transAxes, ha="center", weight="bold")
        self.save(
            fig,
            f"budget-{panel}",
            f"{PN[panel]}：延长训练后的变化",
            "同一开发图 · 固定推理4轮 · 按预定预算比较，不选最高节点",
            True,
            note="线段仅连接30万和150万两个预算，完整训练曲线另页给出。\n"
            "学习门标注的是初值；门学习率减半分支从共同30万步状态开始。",
        )

    def world_deltas(self):
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.subplots_adjust(left=0.075, right=0.96, bottom=0.20, top=0.77, wspace=0.35)
        rows = []
        for ax, panel in zip(axes, ["ood_2", "ood_3", "ood_4"], strict=True):
            for j, ref in enumerate(["fixed95", "paper"]):
                diffs = [
                    value(self.runs[f"new{i}-learn95"], panel, 300000)
                    - value(self.runs[f"new{i}-{ref}"], panel, 300000)
                    for i in range(1, 4)
                ]
                y = np.arange(3) + (j - 0.5) * 0.18
                color = COLORS[ref]
                ax.scatter(diffs, y, color=color, s=65, label=f"学习门 − {LABELS[ref]}")
                for d, yy in zip(diffs, y, strict=True):
                    ax.annotate(
                        f"{d:+.2f}",
                        (d, yy),
                        xytext=(5, 3),
                        textcoords="offset points",
                        fontsize=10,
                    )
                rows.append(
                    dict(
                        panel=panel,
                        comparison=f"learn95-{ref}",
                        paired_deltas_pp=diffs,
                        mean_pp=float(np.mean(diffs)),
                        sample_sd_pp=float(np.std(diffs, ddof=1)),
                        n_worlds=3,
                    )
                )
            ax.axvline(0, color=MUTED, lw=1)
            ax.set(
                xlim=(-30, 55),
                ylim=(2.5, -0.5),
                yticks=[0, 1, 2],
                yticklabels=["新图1", "新图2", "新图3"],
                xlabel="配对准确率差（百分点）",
            )
            ax.set_title(PN[panel], loc="left")
            ax.grid(axis="x", alpha=0.18)
        handles, _ = axes[0].get_legend_handles_labels()
        self.save(
            fig,
            "new-world-paired-differences",
            "学习门收益在三个新图上是否一致？",
            "固定30万步 / 推理4轮 · 每个点为同一图和初始化内的配对差",
            True,
            note="右侧表示学习门更好，左侧表示更差。三个图与初始化共同改变，不能分离两者影响。\n"
            "均值、样本标准差及逐图差值另存 paired-world-differences.json；"
            "不把查询数当重复实验数。",
            legend=handles,
        )
        dump(self.out / "paired-world-differences.json", rows)

    def inventory(self):
        keys = list(self.runs)
        for page, start in enumerate(range(0, len(keys), 18), 1):
            subset = keys[start : start + 18]
            fig, ax = plt.subplots(figsize=(15, 10))
            fig.subplots_adjust(left=0.05, right=0.96, bottom=0.14, top=0.83)
            ax.axis("off")
            rows = []
            for k in subset:
                r = self.runs[k]
                status = r["status"]
                state = (
                    "已完成"
                    if status["state"] == "complete"
                    else ("待运行" if not r["nodes"] else "运行中快照")
                )
                rows.append(
                    [
                        k,
                        label(r),
                        f"{status.get('step', 0) / 10000:g} / {r['budget'] / 10000:g}",
                        f"{r['nodes'][-1]['step'] / 10000:g}" if r["nodes"] else "—",
                        state,
                    ]
                )
            table = ax.table(
                cellText=rows,
                colLabels=[
                    "实验标识",
                    "方法",
                    "训练步数（万）",
                    "末次评测（万）",
                    "状态",
                ],
                colWidths=[0.26, 0.30, 0.17, 0.13, 0.14],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.65)
            for (row, _col), cell in table.get_celld().items():
                cell.set_edgecolor("white")
                cell.set_facecolor("#dce9ef" if row == 0 else "#f1f5f8" if row % 2 else "white")
            self.save(
                fig,
                f"inventory-{page}",
                f"完整实验清单 · {page}",
                "训练步数与末次完整评测分开；未运行任务显式保留",
                False,
                note="本表来自各run实际输出，不采用可能落后的父控制器状态。原始文件、读取时间与SHA256均随图册保存。",
            )


def export_data(out, data):
    learning, grid, status = [], [], []
    for key, r in data["runs"].items():
        for node in r["nodes"]:
            for panel, metrics in node["metrics"].items():
                row = dict(
                    run=key,
                    family=r["family"],
                    step=node["step"],
                    panel=panel,
                    training_repeats=r["repeats"],
                    primary_score=score_name(r),
                    **metrics,
                )
                for field in [
                    "examples",
                    "effective_input_tokens",
                    "supervised_tokens",
                    "estimated_matmul_training_flops",
                    "training_seconds",
                    "wall_seconds",
                ]:
                    row[field] = node.get(field)
                learning.append(row)
        if r["grid"]:
            for repeat, panels in r["grid"]["metrics"].items():
                for panel, metrics in panels.items():
                    grid.append(
                        dict(
                            run=key,
                            step=r["grid"]["step"],
                            inference_repeats=int(repeat),
                            panel=panel,
                            **metrics,
                        )
                    )
        status.append(
            dict(
                run=key,
                family=r["family"],
                method=label(r),
                state=r["status"]["state"],
                step=r["status"].get("step", 0),
                budget=r["budget"],
                last_evaluation_step=r["nodes"][-1]["step"] if r["nodes"] else None,
                source=r["path"],
            )
        )
    historical_grid = out / "sources/paper/parent-grid.json"
    if historical_grid.exists():
        parent_grid = json.loads(historical_grid.read_text())
        for repeat, panels in parent_grid["metrics"].items():
            for panel, metrics in panels.items():
                grid.append(
                    dict(
                        run="paper",
                        step=parent_grid["step"],
                        inference_repeats=int(repeat),
                        panel=panel,
                        **metrics,
                    )
                )
    for name, rows in [
        ("all-learning-data.csv", learning),
        ("all-inference-data.csv", grid),
        ("training-status.csv", status),
    ]:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return dict(learning_rows=len(learning), inference_rows=len(grid), runs=len(status))


def render(out, data):
    style()
    a = Atlas(out, data)
    a.heatmap(
        DEV,
        ["ood_2", "ood_3", "ood_4", "id_2", "id_3", "id_4"],
        1500000,
        "150万步：所有开发模型的终点对比",
        "development-endpoints",
        "同一开发图 · 固定推理4轮 · 左侧图B组合迁移，右侧图A未见组合",
        True,
    )
    for p in ["ood_2", "ood_3", "ood_4"]:
        a.budget_compare(p)
    groups = [
        (
            "architecture",
            "架构对照",
            ["loop", "disco", "paper"],
            "同一开发图 · 原子及2–4跳混合训练 · 推理4轮",
        ),
        (
            "update90",
            "更新门0.90与门学习率",
            ["paper", "fixed90", "learn90", "half90"],
            "同一开发图 · 半学习率分支从30万步分叉",
        ),
        (
            "update95",
            "更新门0.95与共同门",
            ["paper", "fixed95", "learn95", "joint"],
            "同一开发图 · 固定推理4轮 · 门数值为初值或固定值",
        ),
        (
            "candidate",
            "共同门与融合后候选门",
            ["paper", "joint", "candidate"],
            "同一开发图 · 对比门读取的位置 · 固定推理4轮",
        ),
        (
            "historical",
            "原注入门参数化：30万步历史对照",
            ["paper", "scalar", "dual", "vector-injection"],
            "P仅显示其0–30万步段；G/D/V没有150万步结果",
        ),
    ]
    for slug, title, keys, sub in groups[:4]:
        a.curves(keys, ["ood_2", "ood_3", "ood_4"], title, f"{slug}-main-curves", sub, True)
    a.world_deltas()
    for i in range(1, 4):
        keys = [f"new{i}-{v}" for v in ARM.values()]
        groups.append(
            (
                f"new{i}",
                f"新图{i}：三方法确认",
                keys,
                "固定30万步 · 同图内配对 · 推理4轮",
            )
        )
        a.heatmap(
            keys,
            ["ood_2", "ood_3", "ood_4", "id_3", "id_4"],
            300000,
            f"新图{i}：固定30万步对比",
            f"new{i}-endpoints",
            "答案准确率 · 每图独立列出，不隐藏负结果",
            True,
        )
    wk = [f"wang-{v}" for v in ["paper", "fixed95", "learn95", "joint"]]
    a.curves(
        wk,
        ["test_oi", "test_io", "test_oo"],
        "原Wang任务：相同大模型的门控比较",
        "wang-main-curves",
        "四层×两轮 · 52,463词表 · 主评分为答案＋EOS联合正确率",
        True,
    )
    a.heatmap(
        wk + ["compact"],
        ["test_oi", "test_io", "test_oo"],
        1500000,
        "Wang任务：150万步终点与待完成项",
        "wang-endpoints",
        "紧凑模型为一层×四轮 / 2,203词表，同时改变多个条件，不能单变量归因",
        True,
    )
    a.curves(
        ["wang-joint", "compact"],
        ["test_oi", "test_io", "test_oo"],
        "共同门：原模型与紧凑模型",
        "compact-main-curves",
        "同原始图与查询语义；词表、层数、轮数同时改变，曲线仅作配置比较",
        True,
    )
    groups += [
        ("wang", "原Wang大模型", wk, "四层×两轮 · 答案＋EOS联合正确率"),
        (
            "compact",
            "紧凑模型与原共同门",
            ["wang-joint", "compact"],
            "多条件配置比较 · 答案＋EOS联合正确率",
        ),
    ]
    dk = [f"depth1-{v}" for v in ARM.values()]
    a.heatmap(
        dk,
        ["id_2", "ood_2", "id_3", "ood_3", "id_4", "ood_4"],
        1500000,
        "长度外推：只训练原子与两跳",
        "depth1-endpoints",
        "新图1 · 150万步 / 推理4轮 / NoPE · 三跳和四跳完全不进训练",
        True,
    )
    a.curves(
        dk,
        ["id_3", "ood_3", "id_4", "ood_4"],
        "长度外推：增加推理轮数能否改善？",
        "depth1-inference-main",
        "固定150万步 · 原定R4与额外R6/R8同时保留",
        True,
        grid=True,
    )
    a.curves(
        dk,
        ["id_3", "ood_3", "id_4", "ood_4"],
        "长度外推：低准确率区域的放大图",
        "depth1-inference-zoom",
        "固定150万步 · 纵轴仅0–0.5% · 与完整量程图配套阅读",
        True,
        grid=True,
        zoom=True,
    )
    groups.append(
        (
            "depth1",
            "长度外推·新图1",
            dk,
            "原子与两跳训练 / NoPE · 不与混合训练直接归因比较",
        )
    )
    a.curves(
        ["loop", "disco", "paper", "joint", "candidate"],
        ["ood_2", "ood_3", "ood_4"],
        "开发模型：按估算训练计算量比较",
        "development-compute",
        "相同步数不等于相同计算；仅比较同一开发任务",
        True,
        cost=True,
    )
    a.curves(
        wk + ["compact"],
        ["test_oi", "test_io", "test_oo"],
        "Wang配置：按估算训练计算量比较",
        "wang-compute",
        "紧凑模型改变多个条件；实测速度及完整成本保留在CSV",
        True,
        cost=True,
    )
    # Complete appendix: every available evaluation pool and every recorded R.
    original_paper = a.runs["paper"]
    for slug, title, keys, sub in groups:
        if slug == "historical":
            # Restrict P to the actual historical 300k checkpoint/grid.
            p = dict(
                original_paper,
                budget=300000,
                nodes=[n for n in original_paper["nodes"] if n["step"] <= 300000],
            )
            parent = out / "sources/paper"
            raw_path = Path(original_paper["parent"]) / "grid.json"
            cached = parent / "parent-grid.json"
            if cached.exists():
                raw = cached.read_bytes()
            else:
                raw = raw_path.read_bytes()
                cached.write_bytes(raw)
            p["grid"] = json.loads(raw)
            if not any(s["snapshot"] == "sources/paper/parent-grid.json" for s in data["sources"]):
                data["sources"].append(
                    dict(
                        path=str(raw_path.resolve()),
                        snapshot="sources/paper/parent-grid.json",
                        sha256=hashlib.sha256(raw).hexdigest(),
                        bytes=len(raw),
                        read_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
            a.runs["paper"] = p
        panels = [
            p
            for p in (WANG if slug in ["wang", "compact"] else MULTI)
            if any(p in n["metrics"] for k in keys for n in a.runs[k]["nodes"])
        ]
        for start in range(0, len(panels), 6):
            page = start // 6 + 1
            selected = panels[start : start + 6]
            a.curves(
                keys,
                selected,
                f"{title} · 全评测池 {page}",
                f"{slug}-all-curves-{page}",
                sub,
            )
            a.curves(
                keys,
                selected,
                f"{title} · 推理轮数 {page}",
                f"{slug}-all-inference-{page}",
                sub + "；仅显示已有终点评测",
                grid=True,
            )
        if slug == "historical":
            a.runs["paper"] = original_paper
    a.heatmap(
        DEV,
        MULTI,
        300000,
        "开发模型：30万步全11池",
        "development-300k-all-pools",
        "同一预算 / R4 · 训练池、原子池与未见组合池全部保留",
    )
    a.heatmap(
        DEV,
        MULTI,
        1500000,
        "开发模型：150万步全11池",
        "development-1500k-all-pools",
        "同一预算 / R4 · 不以条件筛选结果替代全池分数",
    )
    a.inventory()
    a.pdf.close()
    a.main_pdf.close()
    dump(out / "figure-index.json", a.pages)
    return a.pages


def report(out, data, pages, counts):
    dump(out / "plot-data.json", data)
    dump(out / "provenance.json", data["sources"])
    dump(
        out / "verification.json",
        dict(
            checks=data["checks"],
            **counts,
            figures=len(pages),
            training_or_model_execution=False,
            independent_audit_performed=False,
        ),
    )
    cards = []
    for page in pages:
        s, title, sub = (html.escape(page[k]) for k in ["slug", "title", "subtitle"])
        cards.append(
            f'<article id="{s}"><h2>{title}</h2><p>{sub}</p>'
            f'<a href="{s}.png"><img loading="lazy" src="{s}.png" alt="{title}"></a>'
            f'<p><a href="{s}.png">高清PNG</a> · <a href="{s}.pdf">矢量PDF</a></p></article>'
        )
    main = [p for p in pages if p["main"]]
    nav = "".join(f'<li><a href="#{p["slug"]}">{html.escape(p["title"])}</a></li>' for p in main)
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>门控实验完整图册</title>
<style>
body{{max-width:1400px;margin:40px auto;padding:0 24px;font:17px/1.7 sans-serif;
color:#183247;background:#f4f7fa}}
h1{{font-size:34px}}a{{color:#007580}}
article{{background:white;margin:28px 0;padding:24px;border-radius:12px}}
img{{width:100%;height:auto}}nav{{columns:2}}h2{{font-size:23px}}
.note{{background:#e3edf3;padding:20px;border-radius:10px}}
</style>
<h1>门控实验完整图册</h1><p>共{len(pages)}张图；{counts["runs"]}个实验条目（含待运行项）。
快照：{html.escape(data["captured_utc"])}。后续运行进度不自动覆盖本快照。</p>
<p><a href="comparison-summary.pdf">主要比较 PDF</a> ·
<a href="complete-atlas.pdf">完整图册 PDF</a> ·
<a href="all-learning-data.csv">全部训练评测 CSV</a> ·
<a href="all-inference-data.csv">全部推理评测 CSV</a> ·
<a href="training-status.csv">运行清单 CSV</a></p>
<div class="note">多跳图使用答案准确率，Wang图使用答案＋EOS联合正确率。
所有保存的评测节点均保留，无平滑；缺失终点留空。
新图逐图配对。长度外推仅训练原子和两跳，并使用NoPE；不能与混合训练作单变量比较。
紧凑Wang模型同时改变词表、层数和轮数。步数、数据曝光与计算量不是同一预算。
近期批次沿用独立重载/云端终点审计豁免，本次仅做记录一致性验证。单个开发图和初始化不能确立普遍机制。</div>
<h2>主要比较导航</h2><nav><ol>{nav}</ol></nav>{"".join(cards)}</html>"""
    (out / "index.html").write_text(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.snapshot is not None)
    if args.snapshot:
        data = json.loads(args.snapshot.read_text())
        if args.snapshot.parent.resolve() != args.output.resolve():
            shutil.copytree(args.snapshot.parent / "sources", args.output / "sources")
    else:
        data = collect(args.output)
    pages = render(args.output, data)
    counts = export_data(args.output, data)
    report(args.output, data, pages, counts)
    shutil.copy2(Path(__file__), args.output / "generator.py")
    with zipfile.ZipFile(args.output / "charts-and-data.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path.suffix != ".zip":
                archive.write(path, path.relative_to(args.output))
    print(
        json.dumps(
            dict(output=str(args.output), figures=len(pages), **counts),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
