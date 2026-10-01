#!/usr/bin/env python3
"""Render the complete audited hop/depth comparison without training models."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plot_hop_depth_history as history
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/development-artifacts/hop-depth-comparison-v1"
COLORS = {1: "#B86636", 2: "#3273AF", 3: "#178777", 4: "#8060AF", 6: "#BE4665"}
WIDE = "#9D8B70"
INK = "#172C40"
MUTED = "#596C7B"
CMAP = LinearSegmentedColormap.from_list("accuracy", ["#F2F5F7", "#A9CFDA", "#197B91", "#114B66"])
CONDITIONS = [
    ("two_confirm", "两跳 · 独立世界及配对扩展", "φ=6；3世界 × 2初始化"),
    ("two_dev", "两跳 · 开发", "φ=6；1世界 × 1初始化"),
    ("three_dev", "三跳 · 开发", "φ=24；1世界 × 1初始化"),
    ("three_confirm", "三跳 · 独立世界", "φ=24；2世界 × 1初始化"),
    ("three_low", "三跳 · 首轮开发", "φ=6；1世界 × 1初始化"),
    ("four_low", "四跳 · 首轮开发", "φ=6；1世界 × 1初始化"),
    ("four_high", "四跳 · 扩大组合训练", "φ=24；1世界 × 1初始化"),
]
ARCHS = [(1, 128), (2, 128), (3, 128), (4, 128), (1, 180), (6, 128)]


def read_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, val in row.items():
            if val == "":
                row[key] = None
                continue
            try:
                row[key] = float(val)
            except ValueError:
                pass
    return rows


def condition(row):
    if row["hops"] == 2:
        return "two_dev" if row["phase"] == "development" else "two_confirm"
    if row["hops"] == 3:
        if row["phi"] == 6:
            return "three_low"
        return "three_confirm" if row["phase"] == "confirmation" else "three_dev"
    return "four_low" if row["phi"] == 6 else "four_high"


def select(rows, cond, arch=None):
    return [
        r
        for r in rows
        if condition(r) == cond and (arch is None or (r["layers"], r["width"]) == arch)
    ]


def world_values(rows, metric):
    worlds = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            worlds[row["world"]].append(row[metric])
    return {w: float(np.mean(v)) for w, v in sorted(worlds.items())}


def world_mean(rows, metric):
    vals = list(world_values(rows, metric).values())
    return float(np.mean(vals)) if vals else None


def style():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#CED8DF",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def heading(fig, title, subtitle):
    fig.text(0.045, 0.972, title, fontsize=24, weight="bold", va="top")
    fig.text(0.045, 0.936, subtitle, fontsize=11, color=MUTED, va="top")


def finish_axes(ax, title, flops=False):
    ax.set_title(title, loc="left", pad=12)
    ax.set_ylim(-2, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("留出组合正确率（%）")
    ax.set_xlabel("估算训练计算量（10¹⁴ FLOPs）" if flops else "训练步数（千步）")
    if not flops:
        ax.set_xlim(0, 131)
        ax.set_xticks([0, 32, 64, 96, 128])
    ax.grid(axis="y", color="#E3E9ED", lw=0.8)
    ax.axhline(90, color="#A3ADB5", lw=0.8, ls="--", zorder=0)
    ax.set_axisbelow(True)


def curves(ax, endpoints, learning, cond, title, flops=False, include_wide=True):
    finish_axes(ax, title, flops)
    members = select(endpoints, cond)
    for arch in ARCHS:
        if arch[1] != 128 and not include_wide:
            continue
        runs = select(members, cond, arch)
        if not runs:
            continue
        color = WIDE if arch[1] != 128 else COLORS[arch[0]]
        linestyle = "--" if arch[1] != 128 else "-"
        trajectories = [r for r in learning if r["run_id"] in {e["run_id"] for e in runs}]
        nodes = sorted({r["step"] for r in trajectories})
        xx, yy = [], []
        for step in nodes:
            here = [r for r in trajectories if r["step"] == step]
            xx.append(world_mean(here, "estimated_training_flops") / 1e14 if flops else step / 1000)
            yy.append(world_mean(here, "heldout_accuracy") * 100)
        for world in sorted({r["world"] for r in trajectories}):
            subset = [r for r in trajectories if r["world"] == world]
            wy = [
                np.mean([r["heldout_accuracy"] for r in subset if r["step"] == step]) * 100
                for step in nodes
            ]
            ax.plot(xx, wy, color=color, lw=0.8, alpha=0.32)
        label = f"{arch[0]}层" + ("（宽180）" if arch[1] != 128 else "")
        ax.plot(xx, yy, color=color, lw=2.4, ls=linestyle, label=label)
        if runs[0]["hops"] > 2:
            ax.scatter(
                xx[-1],
                world_mean(runs, "heldout_full_accuracy") * 100,
                marker="x",
                color=color,
                s=66,
                lw=2,
                zorder=5,
            )
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#E4E9ED", fontsize=9)


def overview(endpoints, learning):
    fig = plt.figure(figsize=(18.8, 13.1))
    heading(
        fig,
        "两跳、三跳、四跳：Transformer 深度实验全景",
        "43条当前训练轨迹 · 每条128,000步 · 主指标为完整留出集的“答案与EOS均正确” · 灰格表示未运行",
    )
    gs = fig.add_gridspec(
        2,
        3,
        left=0.07,
        right=0.975,
        top=0.875,
        bottom=0.16,
        height_ratios=[1.15, 1],
        hspace=0.44,
        wspace=0.27,
    )
    ax = fig.add_subplot(gs[0, :])
    values = np.full((len(CONDITIONS), len(ARCHS)), np.nan)
    for i, (cond, _, _) in enumerate(CONDITIONS):
        for j, arch in enumerate(ARCHS):
            runs = select(endpoints, cond, arch)
            if runs:
                values[i, j] = world_mean(runs, "heldout_full_accuracy") * 100
    ax.imshow(
        np.ma.masked_invalid(values),
        cmap=CMAP,
        vmin=0,
        vmax=100,
        aspect="auto",
        extent=(0, 6, 7, 0),
    )
    ax.set_facecolor("#ECEFF2")
    for i in range(7):
        for j in range(6):
            val = values[i, j]
            ax.text(
                j + 0.5,
                i + 0.5,
                "未运行" if np.isnan(val) else f"{val:.2f}%",
                ha="center",
                va="center",
                fontsize=11 if np.isnan(val) else 16,
                color="#8B969F" if np.isnan(val) else ("white" if val > 60 else INK),
                weight="normal" if np.isnan(val) else "bold",
            )
    ax.set_xticks(
        np.arange(6) + 0.5, ["1层", "2层", "3层", "4层", "1层 · 宽180对照", "6层 · 四跳参照"]
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=12)
    ax.set_yticks(
        np.arange(7) + 0.5, [f"{name}\n{desc}" for _, name, desc in CONDITIONS], fontsize=10
    )
    ax.set_xticks(np.arange(7), minor=True)
    ax.set_yticks(np.arange(8), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_position([0.25, ax.get_position().y0, 0.725, ax.get_position().height])
    for spine in ax.spines.values():
        spine.set_visible(False)
    curves(fig.add_subplot(gs[1, 0]), endpoints, learning, "two_confirm", "两跳：1–4层与宽一层")
    curves(fig.add_subplot(gs[1, 1]), endpoints, learning, "three_dev", "三跳：同一开发世界，1–4层")
    curves(
        fig.add_subplot(gs[1, 2]), endpoints, learning, "three_confirm", "三跳：两个新世界，3/4层"
    )
    fig.text(
        0.07,
        0.105,
        "曲线：两跳使用全量留出集；三跳使用固定1024题评测子集（probe），×为全量终点。粗线为世界等权均值，浅线为各世界均值。",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.077,
        "条件：除宽一层外，宽度均为128。φ=6/24对应5,838/23,352条训练组合；所有原子事实均参与训练。两跳与三/四跳的训练数据构成不同。",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.049,
        "覆盖边界：四跳尚无1–4层数据；三跳1/2层仅开发世界。三跳3层已在一个新世界超过90%，现有结果不能证明“每跳必须一层”。",
        fontsize=10,
        color=MUTED,
    )
    return fig


def efficiency(endpoints, learning):
    fig = plt.figure(figsize=(18.8, 12.5))
    heading(
        fig,
        "学习过程与计算成本",
        "每个面板使用同一评测口径 · 全部69个保存节点均保留 · FLOPs为训练矩阵计算估算，不是硬件能耗",
    )
    gs = fig.add_gridspec(
        2, 3, left=0.065, right=0.975, top=0.86, bottom=0.14, hspace=0.50, wspace=0.28
    )
    for i, (cond, title) in enumerate(
        [
            ("two_confirm", "两跳：随计算量的变化"),
            ("three_dev", "三跳开发：随计算量的变化"),
            ("three_confirm", "三跳独立世界：随计算量的变化"),
        ]
    ):
        curves(fig.add_subplot(gs[0, i]), endpoints, learning, cond, title, flops=True)
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("两跳：连续三节点达到90%的时间", loc="left", pad=12)
    for d in [2, 3, 4]:
        runs = select(endpoints, "two_confirm", (d, 128))
        assert all(r["t90_step"] is not None for r in runs)
        vals = [r["t90_step"] / 1000 for r in runs]
        mean = world_mean(runs, "t90_step") / 1000
        ax.scatter(
            np.linspace(d - 0.10, d + 0.10, len(vals)), vals, s=34, color=COLORS[d], alpha=0.65
        )
        ax.plot([d - 0.24, d + 0.24], [mean, mean], color=COLORS[d], lw=3)
        ax.text(
            d,
            max(vals) + 4,
            f"均值 {mean:.2f}k\n6/6 达到",
            ha="center",
            color=COLORS[d],
            fontsize=10,
        )
    ax.set_xticks([2, 3, 4], ["2层", "3层", "4层"])
    ax.set_xlim(1.5, 4.5)
    ax.set_ylim(0, 80)
    ax.set_ylabel("T90起点（千步）")
    ax.grid(axis="y", color="#E3E9ED")
    ax.text(
        0.02,
        0.97,
        "两种1层：各6/6均未在128k内达到",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        color=MUTED,
    )
    for i, hops in enumerate([3, 4]):
        ax = fig.add_subplot(gs[1, i + 1])
        finish_axes(ax, f"{hops}跳：组合训练数据量对比")
        conds = ["three_low", "three_dev"] if hops == 3 else ["four_low", "four_high"]
        for cond, phi, color, ls in zip(
            conds,
            [6, 24],
            ["#8D9DAA", COLORS[4] if hops == 3 else COLORS[6]],
            ["--", "-"],
            strict=True,
        ):
            runs = [r for r in select(endpoints, cond) if r["layers"] == (4 if hops == 3 else 6)]
            assert len(runs) == 1
            rows = sorted(
                [r for r in learning if r["run_id"] == runs[0]["run_id"]], key=lambda r: r["step"]
            )
            ax.plot(
                [r["step"] / 1000 for r in rows],
                [r["heldout_accuracy"] * 100 for r in rows],
                color=color,
                ls=ls,
                lw=2.3,
                label=f"φ={phi}；{int(runs[0]['train_n']):,}条组合",
            )
            final = runs[0]["heldout_full_accuracy"] * 100
            ax.scatter(128, final, marker="x", color=color, s=75, lw=2)
            ax.annotate(
                f"全量 {final:.2f}%",
                (128, final),
                xytext=(-8, 11 if phi == 24 else -19),
                textcoords="offset points",
                ha="right",
                fontsize=10,
                color=color,
            )
        ax.legend(loc="center left" if hops == 3 else "upper left", fontsize=9, frameon=False)
    fig.text(
        0.065,
        0.079,
        "T90为连续三个已登记节点≥90%的首节点，不要求后续永不回落。未达到的轨迹不赋值为128k，也不用于计算精确加速倍数。",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.065,
        0.050,
        "数据量对比使用同一开发世界，三跳固定4层、四跳固定6层。扩大组合集同时改变训练构成与每记录曝光，不能单独归因于覆盖率。",
        fontsize=10,
        color=MUTED,
    )
    return fig


def group_statistics(endpoints):
    result = []
    for cond, name, desc in CONDITIONS:
        for arch in ARCHS:
            rows = select(endpoints, cond, arch)
            if not rows:
                continue
            obj = {
                "condition": cond,
                "condition_label": name,
                "design": desc,
                "layers": arch[0],
                "width": arch[1],
                "runs": len(rows),
                "worlds": len({r["world"] for r in rows}),
            }
            for metric in [
                "atomic_accuracy",
                "train_accuracy",
                "heldout_full_accuracy",
                "heldout_probe_accuracy",
                "calls_accuracy",
                "ood_accuracy",
                "estimated_training_flops",
                "atomic_exposures",
            ]:
                obj[metric] = world_mean(rows, metric)
            worlds = list(world_values(rows, "heldout_full_accuracy").values())
            obj["heldout_world_min"] = min(worlds)
            obj["heldout_world_max"] = max(worlds)
            obj["heldout_full_n_each_world"] = ";".join(
                f"{int(w)}:{int(next(r['heldout_full_n'] for r in rows if r['world'] == w))}"
                for w in sorted({r["world"] for r in rows})
            )
            obj["t90_reached_runs"] = sum(r["t90_step"] is not None for r in rows)
            obj["t90_mean_step"] = (
                world_mean(rows, "t90_step") if obj["t90_reached_runs"] == len(rows) else None
            )
            obj["t90_basis"] = rows[0]["t90_basis"]
            result.append(obj)
    return result


def diagnostics(endpoints, groups):
    fig = plt.figure(figsize=(18.8, 12.5))
    heading(
        fig,
        "终点诊断：事实记住了，组合是否可用",
        "固定128k步 · 全量结果与1024题评测子集分别显示 · 每行先平均同世界初始化，再对世界等权",
    )
    ax = fig.add_axes([0.25, 0.17, 0.51, 0.69])
    metrics = [
        "atomic_accuracy",
        "train_accuracy",
        "heldout_full_accuracy",
        "heldout_probe_accuracy",
        "calls_accuracy",
        "ood_accuracy",
    ]
    values = (
        np.array([[r.get(m) if r.get(m) is not None else np.nan for m in metrics] for r in groups])
        * 100
    )
    ax.imshow(np.ma.masked_invalid(values), cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_facecolor("#ECEFF2")
    labels = []
    for r in groups:
        base = r["condition_label"].replace("独立世界及配对扩展", "配对比较")
        labels.append(f"{base} · {r['layers']}层" + ("宽180" if r["width"] == 180 else ""))
    ax.set_yticks(range(len(groups)), labels, fontsize=10)
    ax.set_xticks(
        range(6),
        [
            "单跳事实",
            "训练组合",
            "全量留出\n主指标",
            "留出子集\nprobe",
            "外部逐跳\n调用",
            "OOD组合\n小样本",
        ],
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=10)
    for i in range(len(groups)):
        for j in range(6):
            v = values[i, j]
            ax.text(
                j,
                i,
                "—" if np.isnan(v) else f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v > 60 else INK,
                fontsize=11,
            )
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    note = fig.add_axes([0.79, 0.17, 0.19, 0.69])
    note.axis("off")
    note.text(0, 1.055, "全量留出题数 / 每个世界", fontsize=11, weight="bold")
    for i, row in enumerate(groups):
        counts = [v.split(":")[1] for v in row["heldout_full_n_each_world"].split(";")]
        note.text(0, 1 - (i + 0.5) / len(groups), " / ".join(counts), va="center", fontsize=10)
    fig.text(
        0.07,
        0.105,
        "外部逐跳调用提供了关系分解并增加调用次数；两跳在全量留出集评价，三/四跳只在1024题probe评价，不能等同于单次回答。",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.077,
        "OOD链的原子事实已训练，但未参与组合训练；每世界仅1–23题，单列保留，不能与ID主指标合并。缺失子集列表示未另设probe。",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.049,
        "三跳1层的单跳与训练组合尚未完全拟合；两跳3层有一条训练组合错1/5838题。低留出分数需要结合这些训练前提解释。",
        fontsize=10,
        color=MUTED,
    )
    return fig


def endpoint_pages(endpoints):
    rank = {c[0]: i for i, c in enumerate(CONDITIONS)}
    ordered = sorted(
        endpoints,
        key=lambda r: (
            rank[condition(r)],
            r["layers"],
            r["width"],
            r["world"],
            r["initialization"],
        ),
    )
    pages = []
    for start in range(0, len(ordered), 23):
        rows = ordered[start : start + 23]
        fig = plt.figure(figsize=(18.8, 12.5))
        heading(
            fig,
            f"逐运行完整终点表 · {start + 1}–{start + len(rows)} / {len(ordered)}",
            "全部128,000步；准确率单位为%；计算量单位为10¹⁴ FLOPs；"
            "完整精度、样本数、来源与全部节点见配套CSV",
        )
        table_data = []
        for r in rows:
            phase = (
                "开发"
                if condition(r) in ["two_dev", "three_dev", "three_low", "four_low", "four_high"]
                else ("扩展" if r["hops"] == 2 and r["layers"] > 2 else "确认")
            )
            vals = [
                f"{int(r['hops'])}跳 φ{int(r['phi'])}",
                phase,
                str(int(r["world"])),
                str(int(r["initialization"])),
                f"{int(r['layers'])}层/{int(r['width'])}",
            ]
            vals.extend(
                "—" if r[m] is None else f"{100 * r[m]:.2f}"
                for m in [
                    "atomic_accuracy",
                    "train_accuracy",
                    "heldout_full_accuracy",
                    "heldout_probe_accuracy",
                    "calls_accuracy",
                    "ood_accuracy",
                ]
            )
            vals.extend(
                [
                    "未达到" if r["t90_step"] is None else f"{r['t90_step'] / 1000:g}k",
                    f"{r['parameters'] / 1000:.1f}",
                    f"{r['estimated_training_flops'] / 1e14:.3f}",
                ]
            )
            table_data.append(vals)
        ax = fig.add_axes([0.035, 0.16, 0.93, 0.69])
        ax.axis("off")
        tbl = ax.table(
            cellText=table_data,
            colLabels=[
                "任务",
                "阶段",
                "世界",
                "初始化",
                "层/宽",
                "单跳",
                "训练",
                "全量留出",
                "probe",
                "逐跳调用",
                "OOD",
                "T90",
                "参数/千",
                "FLOPs",
            ],
            cellLoc="center",
            colWidths=[
                0.085,
                0.05,
                0.08,
                0.07,
                0.07,
                0.061,
                0.061,
                0.075,
                0.066,
                0.073,
                0.058,
                0.07,
                0.065,
                0.072,
            ],
            bbox=[0, 0, 1, 1],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor("white")
            if i == 0:
                cell.set_facecolor(INK)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#F0F5F7" if i % 2 else "#FFFFFF")
                if j == 7:
                    cell.set_text_props(weight="bold", color="#126A7E")
        fig.text(
            0.05,
            0.098,
            "T90：两跳按全量留出集，三/四跳按固定probe，均要求连续三个保存节点≥90%；“未达到”表示128k预算内未满足此规则。",
            fontsize=10,
            color=MUTED,
        )
        fig.text(
            0.05,
            0.066,
            "开发世界不计入独立世界均值；两跳3/4层扩展沿用原三个确认世界。每个世界内的初始化重复不当作额外独立世界。",
            fontsize=10,
            color=MUTED,
        )
        pages.append(fig)
    return pages


def all_trajectories(endpoints, learning, groups):
    fig, axes = plt.subplots(4, 4, figsize=(18.8, 15.6))
    heading(
        fig,
        "全部43条训练轨迹：按条件展开",
        "每条细线对应一次完整训练，不合并初始化；每个面板展示该条件的全部69个保存节点",
    )
    fig.subplots_adjust(left=0.06, right=0.975, top=0.87, bottom=0.10, hspace=0.57, wspace=0.26)
    for ax, group in zip(axes.flat, groups, strict=True):
        members = select(endpoints, group["condition"], (group["layers"], group["width"]))
        color = WIDE if group["width"] == 180 else COLORS[group["layers"]]
        title = group["condition_label"].replace("独立世界及配对扩展", "配对比较")
        title = title.replace("扩大组合训练", "扩大组合")
        arch = f"{group['layers']}层" + (" / 宽180" if group["width"] == 180 else "")
        ax.set_title(
            f"{title} · {arch}\nφ={int(members[0]['phi'])}；{len(members)}次运行",
            loc="left",
            fontsize=10,
            pad=7,
        )
        for endpoint in members:
            rows = sorted(
                [r for r in learning if r["run_id"] == endpoint["run_id"]],
                key=lambda r: r["step"],
            )
            ax.plot(
                [r["step"] / 1000 for r in rows],
                [r["heldout_accuracy"] * 100 for r in rows],
                color=color,
                lw=1.1,
                alpha=0.75,
            )
            ax.scatter(
                128,
                endpoint["heldout_full_accuracy"] * 100,
                color=color,
                marker="x",
                s=22,
                lw=1.1,
            )
        ax.set_xlim(0, 132)
        ax.set_ylim(-2, 104)
        ax.set_xticks([0, 64, 128])
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel("千步", fontsize=9)
        ax.set_ylabel("正确率（%）", fontsize=9)
        ax.grid(axis="y", color="#E3E9ED")
        ax.axhline(90, color="#B5BFC7", ls="--", lw=0.7)
    fig.text(
        0.06,
        0.045,
        "两跳曲线按全量留出集；三/四跳曲线按固定1024题probe；×均为全量终点。"
        "灰色虚线是90%参照，所有面板纵轴保持0–100%。",
        fontsize=10,
        color=MUTED,
    )
    return fig


def main():
    style()
    endpoints = read_csv(OUT / "endpoints.csv")
    learning = read_csv(OUT / "learning.csv")
    assert len(endpoints) == 43 and len(learning) == 43 * 69
    assert len({r["run_id"] for r in endpoints}) == 43
    groups = group_statistics(endpoints)
    with (OUT / "group-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    figures = [
        ("overview", overview(endpoints, learning)),
        ("learning-and-cost", efficiency(endpoints, learning)),
        ("endpoint-diagnostics", diagnostics(endpoints, groups)),
        ("all-trajectories", all_trajectories(endpoints, learning, groups)),
    ]
    for name, fig in figures:
        fig.savefig(OUT / f"{name}.png", dpi=190)
        fig.savefig(OUT / f"{name}.pdf")
    with PdfPages(OUT / "complete-comparison.pdf") as pdf:
        for _, fig in figures:
            pdf.savefig(fig)
        for fig in endpoint_pages(endpoints):
            pdf.savefig(fig)
            plt.close(fig)
        history.setup_style()
        history_rows = history.read_csv(history.SOURCE / "learning.csv")
        history_cfg = history.read_json(history.SOURCE / "lock.json")["config"]
        for fig in (
            history.plot_overview(history_rows, history_cfg),
            history.plot_learning(history_rows, history_cfg),
        ):
            pdf.savefig(fig)
            plt.close(fig)
    for _, fig in figures:
        plt.close(fig)
    (OUT / "figure-notes.json").write_text(
        json.dumps(
            {
                "scope": "43 unique current grok trajectories, 2967 evaluation nodes; "
                "plus 28 historical runs and 196 nodes as separate PDF appendix and history/",
                "pdf_pages": 8,
                "aggregation": "equal weight to worlds, after averaging initializations "
                "within world; development kept separate",
                "primary_metric": "full heldout ID answer-plus-EOS accuracy at step 128000",
                "curves": "two-hop full heldout; three/four-hop fixed 1024-case probe; "
                "crosses show full endpoint",
                "missing": "unrun configurations shown grey; censored T90 left missing, "
                "never replaced with budget",
                "not_comparable": "different hops/phi use different training composition; "
                "depth also changes parameters; external calls use extra computation",
                "reproduce": [
                    ".venv/bin/python scripts/export_hop_depth_comparison.py",
                    ".venv/bin/python scripts/plot_hop_depth_comparison.py",
                    ".venv/bin/python scripts/plot_hop_depth_history.py",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "runs": len(endpoints),
                "nodes": len(learning),
                "groups": len(groups),
                "output": str(OUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
