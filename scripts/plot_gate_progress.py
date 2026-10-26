#!/usr/bin/env python3
"""Snapshot existing gate experiments and render figures without model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "lm-memory-persistent/workspace/llm-memory-editability"
EXT = BASE / "gate-generalization-extension-v1/execution-v2/cases"
PANELS = [
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
PANEL_NAMES = [
    "图A\n原子",
    "图B\n原子",
    "训练\n两跳",
    "图A\n两跳",
    "图B\n两跳",
    "训练\n三跳",
    "图A\n三跳",
    "图B\n三跳",
    "训练\n四跳",
    "图A\n四跳",
    "图B\n四跳",
]
LABELS = {
    "loop": "普通 Loop",
    "disco": "无学习门控 DiscoLoop",
    "paper": "原论文门 P",
    "fixed90": "固定更新门 0.90",
    "learn90": "学习更新门，初值 0.90",
    "half90": "初值 0.90，门学习率减半",
    "fixed95": "固定更新门 0.95",
    "learn95": "学习更新门，初值 0.95",
    "joint": "共同门控，初值 0.95",
}
COLORS = {
    "loop": "#8793A3",
    "disco": "#A47540",
    "paper": "#263D54",
    "fixed90": "#6195C5",
    "learn90": "#126B9B",
    "half90": "#785BB0",
    "fixed95": "#D2A156",
    "learn95": "#CF6C30",
    "joint": "#008A7D",
}
INK = "#173047"
MUTED = "#5D6E80"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def snapshot_json(path, slug, name, output, sources):
    data = path.read_bytes()
    parsed = json.loads(data)
    dest = output / "sources" / slug / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    sources.append(
        {
            "path": str(path.resolve()),
            "snapshot": str(dest.relative_to(output)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "read_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return parsed


def collect(output):
    continuation = BASE / "paper-update-gate-continuation-v1/execution/runs"
    old90 = ROOT / "results/paper-update-gate-multihop-v1/execution/runs"
    old95 = BASE / "paper-update-gate-multihop-u95-v1/runs"
    paper_parent = BASE / "dual-gate-multihop-v1/attempts/preflight-repair-v2/runs/paper"
    ungated = BASE / "ungated-loop-continuation-v1/runs"
    old_ungated = BASE / "loop-hop-unified-v1/runs"
    specs = [
        ("loop", ungated / "loop-l1-r4", old_ungated / "loop-l1-r4"),
        ("disco", ungated / "disco-l1-r4", old_ungated / "disco-l1-r4"),
        ("paper", continuation / "paper", paper_parent),
        ("fixed90", continuation / "fixed-update", old90 / "fixed-update"),
        ("learn90", continuation / "token-update", old90 / "token-update"),
        ("half90", continuation / "token-update-half-lr", old90 / "token-update"),
        ("fixed95", EXT / "continuation/runs/fixed-update", old95 / "fixed-update"),
        ("learn95", EXT / "continuation/runs/token-update", old95 / "token-update"),
        ("joint", BASE / "joint-update-gate-multihop-v1/runs/joint-token", None),
    ]
    for world in range(1, 4):
        for arm in ("paper", "fixed-update", "token-update"):
            path = EXT / f"confirmation-{world}/runs/{arm}"
            specs.append((f"new{world}-{arm}", path, None))

    sources, runs, checks = [], {}, []
    for slug, path, parent in specs:
        if not (path / "status.json").exists():
            runs[slug] = {"status": {"state": "pending", "step": 0, "budget": 300000}}
            continue
        read = lambda name, p=path, s=slug: snapshot_json(  # noqa: E731
            p / name, s, name, output, sources
        )
        status = read("status.json")
        meta = read("run.json")
        nodes = read("learning.json")
        assert nodes and not meta.get("engineering_only"), slug
        assert len({n["step"] for n in nodes}) == len(nodes), slug
        grid = read("grid.json") if (path / "grid.json").exists() else None
        if parent is not None:
            parent_meta = snapshot_json(
                parent / "run.json", slug, "parent-run.json", output, sources
            )
            earlier = snapshot_json(
                parent / "learning.json", slug, "parent-learning.json", output, sources
            )
            for key in ("world_sha256", "panels_sha256"):
                assert parent_meta[key] == meta[key], (slug, key)
            previous = {n["step"]: n for n in earlier if n["step"] <= 300000}
            for node in nodes:
                if node["step"] in previous:
                    for panel in PANELS:
                        assert (
                            previous[node["step"]]["metrics"][panel]["answer_accuracy"]
                            == node["metrics"][panel]["answer_accuracy"]
                        ), (slug, panel, "parent/continuation mismatch")
                previous[node["step"]] = node
            nodes = [previous[step] for step in sorted(previous)]
            checks.append(f"{slug}: parent world, panels and shared checkpoint scores match")
        for node in nodes:
            assert set(PANELS).issubset(node["metrics"]), slug
            for panel in PANELS:
                assert 0 <= node["metrics"][panel]["answer_accuracy"] <= 1
        if status["state"] == "complete":
            assert grid is not None and grid["step"] == status["step"], slug
            assert grid["model_sha256"] == status["model_sha256"], slug
            final = next(n for n in nodes if n["step"] == status["step"])
            for panel in PANELS:
                value = final["metrics"][panel]["answer_accuracy"]
                assert value == grid["metrics"]["4"][panel]["answer_accuracy"], slug
                assert value == status["endpoint"]["metrics"][panel]["answer_accuracy"]
            checks.append(f"{slug}: endpoint/status/R4 grid scores and model hashes match")
        runs[slug] = {"status": status, "meta": meta, "nodes": nodes, "grid": grid}

    assert len({runs[k]["meta"]["world_sha256"] for k in LABELS}) == 1
    assert len({runs[k]["meta"]["panels_sha256"] for k in LABELS}) == 1
    checks.append("All nine development conditions use the same world and evaluation panels")
    return runs, sources, checks


def accuracy(node, panel):
    return node["metrics"][panel]["answer_accuracy"] * 100


def style():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#CAD4DE",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def decorate(fig, title, subtitle, stamp, footer):
    fig.text(0.05, 0.965, title, fontsize=21, fontweight="bold", va="top")
    fig.text(0.05, 0.917, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(0.05, 0.045, footer, fontsize=9, color=MUTED, va="bottom")
    fig.text(0.95, 0.018, stamp, fontsize=8, color=MUTED, ha="right")


def curve(ax, run, panel, color, label, linestyle="-", start=0):
    nodes = [n for n in run.get("nodes", []) if n["step"] >= start]
    if not nodes:
        return
    x = [n["step"] / 10000 for n in nodes]
    y = [accuracy(n, panel) for n in nodes]
    ax.plot(x, y, label=label, color=color, lw=1.9, ls=linestyle, marker=".", ms=4)
    complete = run["status"]["state"] == "complete"
    ax.scatter(
        [x[-1]],
        [y[-1]],
        s=45,
        marker="s" if complete else "o",
        edgecolor=color,
        facecolor=color if complete else "white",
        zorder=5,
    )


def learning_figure(runs, stamp):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    groups = [
        ("架构对照", ["loop", "disco", "paper"]),
        ("更新门：初值 / 固定值 0.90", ["paper", "fixed90", "learn90", "half90"]),
        ("更新门：初值 / 固定值 0.95", ["paper", "fixed95", "learn95", "joint"]),
    ]
    for col, (title, names) in enumerate(groups):
        for row, panel in enumerate(["ood_3", "ood_4"]):
            ax = axes[row, col]
            for name in names:
                curve(
                    ax,
                    runs[name],
                    panel,
                    COLORS[name],
                    LABELS[name],
                    "--" if name.startswith("fixed") or name == "half90" else "-",
                    300000 if name == "half90" else 0,
                )
            ax.axvline(30, color="#AFBAC6", lw=1, ls=":")
            ax.set(xlim=(0, 153), ylim=(-2, 104), xticks=[0, 30, 60, 90, 120, 150])
            ax.grid(axis="y", alpha=0.18)
            if row == 0:
                ax.set_title(title, pad=13)
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0, 1.0),
                    fontsize=8.4,
                    framealpha=0.88,
                    labelspacing=0.35,
                )
            else:
                ax.set_xlabel("累计训练步数（万步）")
            if col == 0:
                ax.set_ylabel(f"图B{'三' if row == 0 else '四'}跳准确率（%）")
    decorate(
        fig,
        "门控与多跳能力：已有完整学习曲线",
        "固定推理 4 轮 · 同一开发图 / 初始化 · 原论文门 P 在三列重复展示以便比较",
        stamp,
        "点为实际评测；连线仅连接观测，不平滑、不外推。■ 已完成终点；○ 仍在训练的最新评测。\n"
        "30万步后半学习率分支与原学习率分支共享同一父检查点；相同步数不代表相同计算成本。",
    )
    fig.subplots_adjust(left=0.065, right=0.975, top=0.83, bottom=0.15, hspace=0.20, wspace=0.13)
    return fig


def fixed_budget_figure(runs, stamp):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    names = list(LABELS)
    y = np.arange(len(names))
    for ax, panel, title in zip(
        axes, ["ood_3", "ood_4"], ["图B三跳 · 主指标", "图B四跳"], strict=True
    ):
        for i, name in enumerate(names):
            nodes = {n["step"]: n for n in runs[name]["nodes"]}
            for step, offset, color in [
                (300000, -0.18, "#ABC6DB"),
                (1500000, 0.18, "#176F91"),
            ]:
                if step not in nodes or (
                    step == 1500000 and runs[name]["status"]["state"] != "complete"
                ):
                    ax.text(
                        1,
                        i + offset,
                        "未到终点",
                        fontsize=8,
                        va="center",
                        color="#7C8895",
                    )
                    continue
                value = accuracy(nodes[step], panel)
                ax.barh(i + offset, value, height=0.29, color=color)
                ax.text(
                    value + 1,
                    i + offset,
                    f"{value:.2f}",
                    va="center",
                    fontsize=9,
                    color=INK,
                )
        ax.set_xlim(0, 108)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("全词表答案准确率（%）")
        ax.set_title(title, pad=15)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [LABELS[n] for n in names])
    axes[0].invert_yaxis()
    fig.legend(
        handles=[
            Patch(color="#ABC6DB", label="30万步"),
            Patch(color="#176F91", label="150万步"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.97, 0.91),
        ncol=2,
        frameon=False,
    )
    decorate(
        fig,
        "固定预算比较：早期排名是否保持？",
        "只比较指定的 30万 / 150万步节点 · 固定推理 4 轮 · 不使用峰值替代终点",
        stamp,
        "“未到终点”表示暂无150万步完成结果，并非0分。半学习率分支的30万步值复用原学习率父检查点。\n"
        "这是单一开发图 / 初始化的观察；当前长训未另做独立重载及云端终点审计。",
    )
    fig.subplots_adjust(left=0.22, right=0.97, top=0.81, bottom=0.15, wspace=0.13)
    return fig


def confirmation_figure(runs, stamp):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    methods = [
        ("paper", "paper"),
        ("fixed-update", "fixed95"),
        ("token-update", "learn95"),
    ]
    for world in range(1, 4):
        for row, panel in enumerate(["ood_3", "ood_4"]):
            ax = axes[row, world - 1]
            has_data = False
            for arm, name in methods:
                run = runs[f"new{world}-{arm}"]
                has_data |= bool(run.get("nodes"))
                curve(
                    ax,
                    run,
                    panel,
                    COLORS[name],
                    LABELS[name],
                    "--" if name == "fixed95" else "-",
                )
            if not has_data:
                ax.text(
                    0.5,
                    0.5,
                    "尚未开始\n保留为缺失，不计为0分",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=MUTED,
                    fontsize=12,
                )
            ax.set(xlim=(0, 30.8), ylim=(-2, 104), xticks=[0, 10, 20, 30])
            ax.grid(axis="y", alpha=0.15)
            if row == 0:
                statuses = []
                for arm, name in methods:
                    s = runs[f"new{world}-{arm}"]["status"]
                    short = {"paper": "P", "fixed95": "固定", "learn95": "学习"}[name]
                    statuses.append(f"{short} {s['step'] / 10000:g}万")
                ax.set_title(f"新图 {world}  /  新初始化", pad=23)
                ax.text(
                    0.5,
                    1.025,
                    " · ".join(statuses),
                    ha="center",
                    transform=ax.transAxes,
                    fontsize=8.5,
                    color=MUTED,
                )
            else:
                ax.set_xlabel("累计训练步数（万步）")
            if world == 1:
                ax.set_ylabel(f"图B{'三' if row == 0 else '四'}跳准确率（%）")
    fig.legend(
        handles=[Line2D([0], [0], color=COLORS[n], lw=2, label=LABELS[n]) for _, n in methods],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.91),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    decorate(
        fig,
        "新图验证：逐图保留已有观测",
        "每组计划三种方法各30万步 · 固定推理4轮 · 未凑齐配对结果前不汇总跨图均值",
        stamp,
        "■ 已完成30万步及推理轮数评测；○ 仍在训练的最新评测。"
        "曲线末点可能早于状态中的实时训练步数。\n"
        "第一组新图按接收端实际产物读取；旧调度器的目录冲突标记不视为训练失败。",
    )
    fig.subplots_adjust(left=0.065, right=0.97, top=0.78, bottom=0.15, hspace=0.20, wspace=0.12)
    return fig


def all_pools_figure(runs, stamp):
    fig, axes = plt.subplots(2, 1, figsize=(15, 13))
    cmap = LinearSegmentedColormap.from_list("accuracy", ["#F4F6F9", "#90C8D0", "#07566F"])
    cmap.set_bad("#E7E9EC")
    names = list(LABELS)
    for ax, step in zip(axes, [300000, 1500000], strict=True):
        values = np.full((len(names), len(PANELS)), np.nan)
        for i, name in enumerate(names):
            node = next((n for n in runs[name]["nodes"] if n["step"] == step), None)
            if node and (step == 300000 or runs[name]["status"]["state"] == "complete"):
                values[i] = [accuracy(node, p) for p in PANELS]
        ax.imshow(values, vmin=0, vmax=100, cmap=cmap, aspect="auto")
        for i in range(len(names)):
            for j in range(len(PANELS)):
                value = values[i, j]
                ax.text(
                    j,
                    i,
                    "—" if np.isnan(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if value >= 70 else INK,
                )
        ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
        ax.set_xticks(range(len(PANELS)), PANEL_NAMES)
        ax.tick_params(length=0)
        ax.set_title(f"{step // 10000}万步 · 固定推理4轮", loc="left", pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
    decorate(
        fig,
        "完整11个评测池：固定预算结果",
        "数值为全词表答案准确率（%）；灰格为尚未完成的终点",
        stamp,
        "训练池与未见题池分别保留；图B原子事实在训练中出现，图B组合用于检验组合泛化。\n"
        "相同开发图与初始化；没有跨图误差条。来源及完整计分字段见随附CSV和快照。",
    )
    fig.subplots_adjust(left=0.235, right=0.97, top=0.85, bottom=0.12, hspace=0.29)
    return fig


def export_data(runs, output):
    rows, statuses = [], []
    for slug, run in runs.items():
        s = run["status"]
        statuses.append(
            {
                "condition": slug,
                "label": LABELS.get(slug, slug),
                "state": s["state"],
                "step": s["step"],
                "budget": s["budget"],
                "last_evaluation_step": max((n["step"] for n in run.get("nodes", [])), default=""),
                "updated_utc": s.get("updated_utc", ""),
            }
        )
        for node in run.get("nodes", []):
            for panel, metrics in node["metrics"].items():
                rows.append(
                    {
                        "condition": slug,
                        "step": node["step"],
                        "panel": panel,
                        "inference_repeats": 4,
                        "world_sha256": run["meta"]["world_sha256"],
                        "state_at_snapshot": s["state"],
                        "shared_pre300k_with": "learn90"
                        if slug == "half90" and node["step"] <= 300000
                        else "",
                        **{
                            k: node.get(k, "")
                            for k in [
                                "examples",
                                "effective_input_tokens",
                                "executed_input_tokens",
                                "supervised_tokens",
                                "estimated_matmul_training_flops",
                                "training_seconds",
                            ]
                        },
                        **metrics,
                    }
                )
    for filename, data in [
        ("all-learning-data.csv", rows),
        ("training-status.csv", statuses),
    ]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    runs, sources, checks = collect(output)
    ended = datetime.now(timezone.utc)
    stamp = "数据快照：" + ended.astimezone(ZoneInfo("America/Chicago")).strftime(
        "%Y-%m-%d %H:%M CDT"
    )
    n_rows = export_data(runs, output)
    write_json(output / "plot-data.json", runs)
    style()
    figures = [
        ("learning-curves", learning_figure(runs, stamp)),
        ("fixed-budget-comparison", fixed_budget_figure(runs, stamp)),
        ("new-graph-validation", confirmation_figure(runs, stamp)),
        ("all-evaluation-pools", all_pools_figure(runs, stamp)),
    ]
    with PdfPages(output / "experiment-charts.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(output / f"{name}.png", dpi=160)
            fig.savefig(output / f"{name}.pdf")
            pdf.savefig(fig)
            plt.close(fig)
    write_json(
        output / "provenance.json",
        {
            "snapshot_start_utc": started.isoformat(),
            "snapshot_end_utc": ended.isoformat(),
            "display_timestamp": stamp,
            "sources": sources,
            "checks": checks,
            "csv_rows": n_rows,
            "new_training_updates": 0,
            "model_evaluations": 0,
            "metric": "full-vocabulary answer_accuracy",
            "inference_repeats": 4,
            "independent_audit": "not performed; existing user waiver retained",
            "limits": [
                "single development world/initialization",
                "cross-environment differences retained",
                "half90 shares learn90 history through 300k",
                "no smoothing or extrapolation",
                "missing endpoints are not zero",
                "no partial-world aggregate",
            ],
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "timestamp": stamp,
                "csv_rows": n_rows,
                "source_files": len(sources),
                "checks": len(checks),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
