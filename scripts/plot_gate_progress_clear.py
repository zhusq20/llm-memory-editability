#!/usr/bin/env python3
"""Make larger, separated figures from an existing immutable experiment snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

LABELS = {
    "loop": "普通 Loop",
    "disco": "无学习门控 DiscoLoop",
    "paper": "原论文门 P",
    "fixed90": "固定门 0.90",
    "learn90": "学习门 0.90",
    "half90": "学习门 0.90 · 门学习率减半",
    "fixed95": "固定门 0.95",
    "learn95": "学习门 0.95",
    "joint": "共同门控 0.95",
}
SHORT = {**LABELS, "disco": "无学习门 DiscoLoop", "half90": "学习门 0.90 · 半学习率"}
COLORS = {
    "loop": "#7E8C9A",
    "disco": "#A47540",
    "paper": "#344657",
    "fixed90": "#5B9CC5",
    "learn90": "#087FA3",
    "half90": "#8257AD",
    "fixed95": "#D6A34C",
    "learn95": "#D26B29",
    "joint": "#008C78",
}
GROUPS = {
    "architecture": ("架构对照", ["loop", "disco", "paper"]),
    "gate90": ("更新门 0.90", ["paper", "fixed90", "learn90", "half90"]),
    "gate95": ("更新门 0.95", ["paper", "fixed95", "learn95", "joint"]),
}
INK, MUTED = "#173047", "#607386"
HOPS = {2: "两跳", 3: "三跳", 4: "四跳"}


def style():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 14,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#BAC8D3",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def title(fig, main, subtitle):
    fig.text(0.055, 0.96, main, fontsize=24, weight="bold", va="top")
    fig.text(0.055, 0.895, subtitle, fontsize=12.5, color=MUTED, va="top")


def footnote(fig, text, stamp):
    fig.text(0.055, 0.065, text, fontsize=10.5, va="top", color=MUTED)
    fig.text(0.96, 0.018, stamp, fontsize=9, ha="right", color=MUTED)


def value(run, step, hops):
    if step == 1500000 and run["status"]["state"] != "complete":
        return None
    for node in run["nodes"]:
        if node["step"] == step:
            return 100 * node["metrics"][f"ood_{hops}"]["answer_accuracy"]
    return None


def endpoint_overview(runs, stamp):
    fig = plt.figure(figsize=(11.8, 7.9))
    ax = fig.add_axes((0.09, 0.17, 0.87, 0.59))
    names = [n for n in LABELS if runs[n]["status"]["state"] == "complete"]
    assert names == ["learn90", "half90", "learn95"]
    xs, width = np.arange(3), 0.23
    for i, name in enumerate(names):
        vals = [value(runs[name], 1500000, hops) for hops in HOPS]
        assert all(v is not None for v in vals)
        bars = ax.bar(xs + (i - 1) * width, vals, width, color=COLORS[name], label=LABELS[name])
        ax.bar_label(
            bars,
            labels=[f"{v:.2f}%" for v in vals],
            padding=6,
            fontsize=14,
            fontweight="bold",
            color=COLORS[name],
        )
    ax.set(
        ylim=(0, 110),
        yticks=[0, 20, 40, 60, 80, 100],
        xticks=xs,
        xticklabels=["两跳", "三跳", "四跳"],
        ylabel="答案准确率（%）",
    )
    ax.tick_params(axis="x", labelsize=20, pad=12, length=0)
    ax.grid(axis="y", alpha=0.17)
    ax.set_axisbelow(True)
    fig.legend(
        loc="upper left",
        bbox_to_anchor=(0.07, 0.84),
        ncol=3,
        frameon=False,
        fontsize=12,
        handlelength=1.3,
        columnspacing=1.4,
    )
    title(
        fig, "150万步终点：两跳、三跳、四跳", "图B组合泛化 · 固定推理4轮 · 只展示已经完成的三条训练"
    )
    footnote(
        fig,
        "学习门数值为初值；半学习率分支从共同的30万步状态分叉。\n"
        "单一开发图 / 初始化；当前终点未另做独立重载与云端审计。",
        stamp,
    )
    return fig


def budget_figure(runs, hops, stamp):
    fig = plt.figure(figsize=(12.6, 9.1))
    ax = fig.add_axes((0.305, 0.16, 0.435, 0.61))
    names = list(LABELS)
    for i, name in enumerate(names):
        if i % 2 == 0:
            ax.axhspan(i - 0.45, i + 0.45, color="#F1F5F8", zorder=0)
        a, b = value(runs[name], 300000, hops), value(runs[name], 1500000, hops)
        assert a is not None
        if b is not None:
            ax.plot([a, b], [i, i], color="#C3D0D9", lw=3, zorder=1)
            ax.scatter([b], [i], s=85, marker="s", color="#0A7398", zorder=4)
        ax.scatter([a], [i], s=85, facecolors="white", edgecolors="#6C8298", linewidths=2, zorder=3)
        ax.text(
            1.20,
            i,
            f"{a:.2f}%",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=14,
            color="#526A81",
        )
        ax.text(
            1.49,
            i,
            "—" if b is None else f"{b:.2f}%",
            transform=ax.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=14,
            weight="bold" if b is not None else "normal",
            color="#0A7398" if b is not None else "#95A2AF",
        )
    ax.set(
        xlim=(-2, 103),
        ylim=(len(names) - 0.55, -0.55),
        xticks=[0, 20, 40, 60, 80, 100],
        xlabel="图B答案准确率（%）",
    )
    ax.set_yticks(range(len(names)), list(LABELS.values()))
    ax.tick_params(axis="y", length=0, pad=13, labelsize=13.5)
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for x, text in [(1.20, "30万步"), (1.49, "150万步")]:
        ax.text(x, 1.055, text, transform=ax.transAxes, ha="center", fontsize=14, weight="bold")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markeredgecolor="#6C8298",
            markerfacecolor="white",
            markersize=9,
            label="30万步",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#0A7398",
            markersize=9,
            label="150万步（已完成）",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.06, 0.845),
        frameon=False,
        ncol=2,
        fontsize=12,
    )
    title(
        fig,
        f"{HOPS[hops]}：30万步与150万步对照",
        "同一开发图 · 固定推理4轮 · 每行一种模型，右侧给出精确数值",
    )
    footnote(
        fig,
        "— 表示该快照中还没有150万步终点；连接线只比较两个指定预算，不代表中间曲线。\n"
        "学习门0.90的两条分支共享30万步检查点；不同架构的相同步数不等于相同计算量。",
        stamp,
    )
    return fig


def separated_labels(endpoints):
    """Separate labels in axes coordinates without changing measured coordinates."""
    order = sorted(endpoints, key=lambda item: item[2])
    positions = np.array([item[2] / 100 for item in order])
    positions = np.clip(positions, 0.07, 0.93)
    for i in range(1, len(positions)):
        positions[i] = max(positions[i], positions[i - 1] + 0.16)
    if len(positions) and positions[-1] > 0.93:
        positions -= positions[-1] - 0.93
    return [(item, float(y)) for item, y in zip(order, positions, strict=True)]


def learning_figure(runs, hops, group, stamp):
    group_title, names = GROUPS[group]
    fig = plt.figure(figsize=(12.2, 7.5))
    ax = fig.add_axes((0.085, 0.19, 0.60, 0.60))
    endpoints = []
    for name in names:
        nodes = [n for n in runs[name]["nodes"] if name != "half90" or n["step"] >= 300000]
        xs = [n["step"] / 10000 for n in nodes]
        ys = [n["metrics"][f"ood_{hops}"]["answer_accuracy"] * 100 for n in nodes]
        ax.plot(
            xs,
            ys,
            color=COLORS[name],
            lw=2.8,
            ls="--" if name.startswith("fixed") or name == "half90" else "-",
            marker="o",
            markersize=4,
        )
        done = runs[name]["status"]["state"] == "complete"
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            color=COLORS[name] if done else "white",
            edgecolor=COLORS[name],
            s=85,
            marker="s" if done else "o",
            zorder=4,
        )
        endpoints.append((name, xs[-1], ys[-1], done))
    for (name, x, y, done), label_y in separated_labels(endpoints):
        ax.text(
            1.045,
            label_y + 0.022,
            SHORT[name],
            transform=ax.transAxes,
            fontsize=12.5,
            weight="bold",
            color=COLORS[name],
            va="bottom",
        )
        ax.text(
            1.045,
            label_y - 0.005,
            f"{y:.2f}%  |  {x:g}万步" + (" · 终点" if done else " · 中途"),
            transform=ax.transAxes,
            fontsize=11.5,
            color=COLORS[name],
            va="top",
        )
    ax.axvline(30, color="#B5C2CE", lw=1.1, ls=":")
    ax.set(
        xlim=(0, 153),
        ylim=(0, 102),
        xticks=[0, 30, 60, 90, 120, 150],
        yticks=[0, 20, 40, 60, 80, 100],
        xlabel="累计训练步数（万步）",
        ylabel="图B答案准确率（%）",
    )
    ax.grid(axis="y", alpha=0.17)
    title(
        fig,
        f"{HOPS[hops]}学习曲线 · {group_title}",
        "每张只放一组对照 · 固定推理4轮 · 右侧直接标注模型、最新评测及步数",
    )
    footnote(
        fig,
        "点为实际评测，连线不平滑、不外推；■ 已完成终点，○ 训练中最新评测。\n"
        "各曲线结束步数不同，不能把中途值与150万步值视为同预算比较。",
        stamp,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.snapshot.resolve(), args.output.resolve()
    runs = json.loads((source / "plot-data.json").read_text())
    provenance = json.loads((source / "provenance.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    stamp = provenance["display_timestamp"] + "（沿用原数据）"
    style()
    figures = [("completed-overview", endpoint_overview(runs, stamp), None)]
    for hops in HOPS:
        figures.append((f"{hops}-hop-comparison", budget_figure(runs, hops, stamp), hops))
        for group in GROUPS:
            figures.append((f"{hops}-hop-{group}", learning_figure(runs, hops, group, stamp), hops))
    with PdfPages(output / "all-charts.pdf") as all_pdf:
        hop_pdfs = {h: PdfPages(output / f"{h}-hop-charts.pdf") for h in HOPS}
        try:
            for name, fig, hops in figures:
                fig.savefig(output / f"{name}.png", dpi=180)
                all_pdf.savefig(fig)
                if hops is not None:
                    hop_pdfs[hops].savefig(fig)
                plt.close(fig)
        finally:
            for pdf in hop_pdfs.values():
                pdf.close()
    for name in ("plot-data.json", "all-learning-data.csv", "training-status.csv"):
        shutil.copyfile(source / name, output / name)
    shutil.copyfile(__file__, output / "generator.py")
    evidence = {
        "source_snapshot": str(source),
        "source_plot_data_sha256": hashlib.sha256(
            (source / "plot-data.json").read_bytes()
        ).hexdigest(),
        "snapshot_timestamp": stamp,
        "figure_count": len(figures),
        "all_pdf_pages": len(figures),
        "per_hop_pdf_pages": 4,
        "new_training_updates": 0,
        "new_model_evaluations": 0,
        "data_values_changed": False,
        "display_changes": [
            "one hop and comparison group per chart",
            "larger text",
            "direct curve labels outside axes",
            "exact values in separate columns",
            "all accuracy axes start at zero",
            "missing endpoints remain missing",
        ],
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )
    with zipfile.ZipFile(
        output / "clear-charts-and-data.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(output.iterdir()):
            if path.suffix != ".zip":
                archive.write(path, arcname=path.name)
    print(json.dumps({"output": str(output), "figures": len(figures), "data_changed": False}))


if __name__ == "__main__":
    main()
