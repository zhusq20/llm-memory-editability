#!/usr/bin/env python3
"""Export the complete historical two-hop batch without pooling newer protocols."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/development-artifacts/twohop-depth-v1"
OUTPUT = ROOT / "docs/development-artifacts/hop-depth-comparison-v1/history"
METRICS = ("atomic_exact", "train_exact", "test_exact", "two_call_exact")
METRIC_LABELS = ("单跳事实", "训练两跳", "留出两跳", "外部两次调用")
WORLD_COLORS = ("#087E8B", "#AA4499")


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style():
    for path in (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        raise RuntimeError("A Chinese font is required to render this appendix.")
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlelocation": "left",
            "axes.titlesize": 14,
            "axes.labelcolor": "#263746",
            "text.color": "#263746",
            "xtick.color": "#263746",
            "ytick.color": "#263746",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def selected_rows(rows, arch, step, world=None):
    return [
        row
        for row in rows
        if row["arch"] == arch
        and int(row["step"]) == step
        and (world is None or int(row["world"]) == world)
    ]


def world_average(rows, metric, worlds):
    return float(
        np.mean(
            [
                np.mean([float(row[metric]) for row in rows if int(row["world"]) == world])
                for world in worlds
            ]
        )
    )


def architecture_label(arch):
    return f"{arch['layers']} 层\n宽 {arch['width']}"


def plot_overview(rows, cfg):
    architectures = cfg["architectures"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.3), gridspec_kw={"height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.075, right=0.975, top=0.83, bottom=0.15, hspace=0.55, wspace=0.25)
    fig.suptitle("历史两跳实验：深度、宽度与训练预算", x=0.075, y=0.965, ha="left", fontsize=23)
    fig.text(
        0.075,
        0.915,
        "twohop-depth-v1  ·  7 种架构 × 2 个世界 × 2 个初始化 = 28 次训练"
        "  ·  全部 196 个评价节点保留",
        fontsize=12,
    )
    fig.text(
        0.075,
        0.88,
        "这是较早的独立协议；图生成、组合覆盖、采样、优化、格式和预算均与后续实验不同，不合并计算均值。",
        fontsize=10.5,
        color="#636B74",
    )
    for col, step in enumerate((cfg["primary_step"], cfg["steps"])):
        ax = axes[0, col]
        for index, arch in enumerate(architectures):
            group = selected_rows(rows, arch["name"], step)
            mean = world_average(group, "test_exact", cfg["worlds"]) * 100
            color = "#78B8C5" if index < 5 else "#E7B469"
            ax.bar(index, mean, width=0.69, color=color, edgecolor="white", zorder=2)
            for world_index, world in enumerate(cfg["worlds"]):
                world_rows = sorted(
                    selected_rows(rows, arch["name"], step, world), key=lambda row: row["seed"]
                )
                offset = (-0.16, 0.16)[world_index]
                vals = [float(row["test_exact"]) * 100 for row in world_rows]
                ax.scatter(
                    index + offset + np.array([-0.047, 0.047]),
                    vals,
                    s=20,
                    c=WORLD_COLORS[world_index],
                    alpha=0.63,
                    zorder=4,
                )
                ax.scatter(
                    index + offset,
                    np.mean(vals),
                    s=29,
                    marker="D",
                    facecolors="white",
                    edgecolors=WORLD_COLORS[world_index],
                    linewidths=1.2,
                    zorder=5,
                )
            max_value = max(float(row["test_exact"]) * 100 for row in group)
            ax.text(index, max(mean, max_value) + 0.17, f"{mean:.2f}", ha="center", fontsize=10)
        ax.axhline(100 / cfg["entities"], color="#8B949E", ls=":", lw=1.2)
        ax.axvline(4.5, color="#BAC1C7", lw=1, ls="--")
        ax.set(
            title=("A  主预算：4,096 步" if col == 0 else "B  加长预算：8,192 步"),
            ylabel="留出两跳准确率（%，答案及 EOS 均正确）",
            xticks=range(7),
            xticklabels=[architecture_label(arch) for arch in architectures],
            ylim=(0, 5.85),
            yticks=range(6),
        )
        ax.grid(axis="y", alpha=0.16, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
        ax.text(2, -0.27, "同宽深度比较", transform=ax.get_xaxis_transform(), ha="center")
        ax.text(5.5, -0.27, "近等参数宽度对照", transform=ax.get_xaxis_transform(), ha="center")

        ax = axes[1, col]
        values = np.array(
            [
                [
                    world_average(selected_rows(rows, arch["name"], step), metric, cfg["worlds"])
                    * 100
                    for metric in METRICS
                ]
                for arch in architectures
            ]
        )
        ax.imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
        for row_index in range(len(architectures)):
            for metric_index in range(len(METRICS)):
                value = values[row_index, metric_index]
                ax.text(
                    metric_index,
                    row_index,
                    f"{value:.2f}",
                    color="white" if value > 55 else "#263746",
                    ha="center",
                    va="center",
                    fontsize=11,
                )
        ax.set(
            title="C  各项任务前提（%）" if col == 0 else "D  各项任务前提（%）",
            xticks=range(len(METRICS)),
            xticklabels=METRIC_LABELS,
            yticks=range(7),
            yticklabels=[f"{a['layers']} 层 / 宽 {a['width']}" for a in architectures],
        )
        ax.tick_params(length=0)
        ax.spines[["left", "bottom"]].set_visible(False)

    handles = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=c, label=f"世界 {i + 1} 的单次运行"
        )
        for i, c in enumerate(WORLD_COLORS)
    ] + [
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markeredgecolor="#455A64",
            label="空心菱形：世界均值",
        ),
        Line2D([0], [0], color="#8B949E", ls=":", label="均匀猜实体参考：0.78%"),
    ]
    handles[2].set_markerfacecolor("white")
    fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.07, 0.089), ncol=4, frameon=False
    )
    fig.text(
        0.075,
        0.065,
        "柱及格内数值：先对同一世界的两个初始化求均值，再平均两个世界；点不代表四个独立世界。",
        fontsize=10,
    )
    fig.text(
        0.075,
        0.037,
        "8,192 步宽模型出现事实和训练准确率回退，CPU FP32 复核仍存在。"
        "外部两次调用提供关系分解并增加计算。",
        fontsize=10,
    )
    return fig


def plot_learning(rows, cfg):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.067, right=0.975, top=0.81, bottom=0.13, hspace=0.4, wspace=0.2)
    fig.suptitle("历史两跳实验：28 条完整学习轨迹", x=0.067, y=0.965, ha="left", fontsize=23)
    fig.text(
        0.067,
        0.906,
        "每条曲线保留 0、128、512、1,024、2,048、4,096、8,192 步全部节点；不选择最佳检查点。",
        fontsize=12,
    )
    fig.text(
        0.067,
        0.868,
        "横轴按评价节点等距排列；纵轴为完整留出集上的答案及 EOS 准确率。所有面板使用相同刻度。",
        fontsize=10.5,
        color="#636B74",
    )
    for ax, arch in zip(axes.flat, cfg["architectures"], strict=False):
        for wi, world in enumerate(cfg["worlds"]):
            for si, seed in enumerate(cfg["initializations"]):
                vals = [
                    100
                    * float(
                        next(
                            row["test_exact"]
                            for row in selected_rows(rows, arch["name"], step, world)
                            if int(row["seed"]) == seed
                        )
                    )
                    for step in cfg["nodes"]
                ]
                ax.plot(
                    range(len(cfg["nodes"])),
                    vals,
                    color=WORLD_COLORS[wi],
                    ls=("-", "--")[si],
                    marker=("o", "s")[si],
                    markersize=3,
                    lw=1.15,
                    alpha=0.7,
                )
        mean = [
            world_average(selected_rows(rows, arch["name"], step), "test_exact", cfg["worlds"])
            * 100
            for step in cfg["nodes"]
        ]
        ax.plot(range(len(cfg["nodes"])), mean, color="#23374D", lw=2.5, zorder=5)
        ax.axhline(100 / cfg["entities"], color="#8B949E", ls=":", lw=1)
        ax.axvline(5, color="#BAC1C7", ls="--", lw=1)
        ax.set(
            title=f"{arch['layers']} 层 · 宽 {arch['width']}",
            xticks=range(len(cfg["nodes"])),
            xticklabels=["0", "128", "512", "1,024", "2,048", "4,096", "8,192"],
            ylim=(0, 6),
            yticks=[0, 2, 4, 6],
        )
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.18)
    axes[1, 3].set_axis_off()
    handles = [
        Line2D([0], [0], color=WORLD_COLORS[0], label="世界 1（142801）"),
        Line2D([0], [0], color=WORLD_COLORS[1], label="世界 2（142802）"),
        Line2D([0], [0], color="#66737E", marker="o", label="实线：初始化 14281"),
        Line2D([0], [0], color="#66737E", marker="s", ls="--", label="虚线：初始化 14282"),
        Line2D([0], [0], color="#23374D", lw=2.5, label="粗深线：两个世界的均值"),
        Line2D([0], [0], color="#8B949E", ls=":", label="横虚线：均匀猜实体 0.78%"),
        Line2D([0], [0], color="#BAC1C7", ls="--", label="竖虚线：主预算 4,096 步"),
    ]
    axes[1, 3].legend(handles=handles, loc="upper left", frameon=False, labelspacing=1.1)
    fig.supylabel("留出两跳准确率（%）", x=0.014, fontsize=12)
    fig.text(
        0.067,
        0.035,
        "图中全部运行在两个固定预算下保留。相同训练步数不等于相同计算量；本批恒定学习率未逐架构调优。",
        fontsize=10.5,
    )
    return fig


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = read_json(SOURCE / "lock.json")["config"]
    audit = read_json(SOURCE / "audit.json")
    rows = read_csv(SOURCE / "learning.csv")
    assert audit["complete"] and audit["runs"] == 28 and audit["nodes"] == 196
    assert len(rows) == 196
    runs = sorted({row["run"] for row in rows})
    assert len(runs) == 28
    assert len({(row["run"], row["step"]) for row in rows}) == 196
    for run in runs:
        assert sorted(int(row["step"]) for row in rows if row["run"] == run) == cfg["nodes"]
    endpoints = [row for row in rows if int(row["step"]) in (cfg["primary_step"], cfg["steps"])]
    assert len(endpoints) == 56
    summaries = []
    originals = read_json(SOURCE / "summary.json")["summary"]
    for step in (cfg["primary_step"], cfg["steps"]):
        for arch in cfg["architectures"]:
            group = selected_rows(rows, arch["name"], step)
            assert len(group) == 4
            original = next(v for v in originals if v["step"] == step and v["arch"] == arch["name"])
            summary = {
                "step": step,
                "arch": arch["name"],
                "layers": arch["layers"],
                "width": arch["width"],
                "parameters": int(group[0]["parameters"]),
                "worlds": len(cfg["worlds"]),
                "runs": len(group),
            }
            for metric in METRICS:
                value = world_average(group, metric, cfg["worlds"])
                assert np.isclose(value, original[metric], atol=1e-12, rtol=0)
                summary[f"{metric}_pct"] = value * 100
            summaries.append(summary)
    shutil.copyfile(SOURCE / "learning.csv", OUTPUT / "learning-all-196-nodes.csv")
    shutil.copyfile(SOURCE / "world-means.csv", OUTPUT / "world-means.csv")
    save_csv(OUTPUT / "endpoints-all-56.csv", endpoints)
    save_csv(OUTPUT / "architecture-summary.csv", summaries)
    run_manifest = []
    for run in runs:
        group = sorted((row for row in rows if row["run"] == run), key=lambda row: int(row["step"]))
        row = group[-1]
        run_manifest.append(
            {
                "run": run,
                "world": row["world"],
                "seed": row["seed"],
                "arch": row["arch"],
                "parameters": row["parameters"],
                "evaluation_nodes": len(group),
                "final_step": row["step"],
                "raw_run_directory": f"results/twohop-depth-v1/{run}",
                "source_learning_csv": str((SOURCE / "learning.csv").relative_to(ROOT)),
            }
        )
    save_csv(OUTPUT / "runs-all-28.csv", run_manifest)
    setup_style()
    for name, fig in (
        ("history-overview", plot_overview(rows, cfg)),
        ("history-learning-curves", plot_learning(rows, cfg)),
    ):
        for extension in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{name}.{extension}", dpi=190)
        plt.close(fig)
    sources = [
        SOURCE / filename
        for filename in (
            "learning.csv",
            "summary.json",
            "world-means.csv",
            "lock.json",
            "audit.json",
            "supplemental-audit.json",
        )
    ]
    manifest = {
        "batch": "twohop-depth-v1",
        "analysis": "Post-hoc descriptive visualization only; no new training or model selection.",
        "runs": 28,
        "worlds": cfg["worlds"],
        "initializations": cfg["initializations"],
        "evaluation_rows": 196,
        "endpoint_rows": 56,
        "configuration": cfg,
        "task": {
            "entities": 128,
            "relations": 4,
            "relation_maps": "independent random permutations of entities",
            "atomic_facts_per_world": 512,
            "train_compositions_per_world": 1024,
            "heldout_compositions_per_world": 1024,
            "answer_marginal": "exactly balanced in each composition split",
            "coverage": "Every atom appears in two training compositions in each hop role.",
            "metric": "Full held-out set: both answer and EOS must be correct.",
            "sampling": "128 atomic + 128 composite examples per update",
            "optimizer": "AdamW; 128-step warmup; then constant lr=0.001; weight_decay=0.1",
        },
        "aggregation": "Average seeds within each world, then equally average the two worlds.",
        "plot_captions_zh": {
            "history-overview": (
                "历史两跳批次的全部七架构在 4096 / 8192 步的比较。柱及热图为世界均值的均值，"
                "圆点为单次运行、空心菱形为世界均值；无置信区间或显著性检验。"
                "主预算全部架构单跳、训练组合及外部两次调用均 100%，"
                "留出两跳仅 0.88%–3.34%。8192 步浅宽模型出现前提回退。"
            ),
            "history-learning-curves": (
                "28 条运行、196 节点全量曲线；横轴为等距评价节点而非线性训练时间。"
                "粗线按世界等权平均，细线区分世界和初始化。"
            ),
        },
        "caveats_zh": [
            "历史协议与后续 grok 系列同时改变图生成、组合覆盖、采样、优化、格式和预算；"
            "不可混合平均。",
            "同世界不同初始化、架构和节点不视为独立世界；两个世界不足以给出普遍层数下界。",
            "近等参数对照为 d6w128=1208192、d1w312=1217424、d2w220=1199220 个参数。",
            "等步数不等于等 FLOPs；逐节点计算估计与训练耗时保留在 CSV 中。",
            "8192 步浅宽模型存在训练及单跳回退，CPU FP32 复核仍存在；学习率未按架构调优。",
            "外部两次调用使用模型自己预测的中间实体，但提供已知关系分解并增加计算。",
            "0.78% 线仅为 1/128 均匀猜实体参考，不是随机全词表生成答案及 EOS 的测量值。",
            "结果是端到端行为观察，不证明一层执行一跳或架构的普遍能力上限。",
        ],
        "validation": {
            "source_audit_complete": True,
            "all_28_runs_have_all_7_nodes": True,
            "all_56_endpoints_retained": True,
            "four_metrics_match_original_summary_at_14_architecture_budgets": True,
            "copied_learning_sha256_matches_source": (
                digest(OUTPUT / "learning-all-196-nodes.csv") == digest(SOURCE / "learning.csv")
            ),
        },
        "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "output_files": {
            str(path.relative_to(ROOT)): digest(path)
            for path in sorted(OUTPUT.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(f"Historical appendix: {len(runs)} runs, {len(rows)} nodes, {len(endpoints)} endpoints.")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
