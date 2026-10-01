#!/usr/bin/env python3
"""Create a Chinese overview and PDF atlas from audited loop-study summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/development-artifacts/grok-loop-v1"
OUT = BASE / "visual-summary"
ARCHS = ("c1", "c2", "cd", "l1", "l2")
INK, MUTED = "#18334A", "#617487"
COLORS = {"attention": "#2878B5", "mlp": "#D77A20", "both": "#26966A", "full": "#9B5AA4"}
SOURCES = {}


def read(relative):
    path = BASE / relative
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def heading(fig, title, subtitle, page):
    fig.text(0.055, 0.955, title, fontsize=23, weight="bold", color=INK, va="top")
    fig.text(0.055, 0.91, subtitle, fontsize=11, color=MUTED, va="top")
    fig.text(0.945, 0.025, f"2026-09-30 · {page}/8", ha="right", fontsize=9, color=MUTED)


def metric(groups, kind, hops, arch, split, key, **filters):
    rows = [
        r
        for r in groups
        if r["phase"] == "confirmation"
        and r["kind"] == kind
        and r["hops"] == hops
        and r["architecture"] == arch
        and r["split"] == split
        and r["group"] == "all_rows"
        and all(r[k] == v for k, v in filters.items())
    ]
    if len(rows) != 1 or not rows[0]["registration_complete"]:
        raise ValueError(f"Incomplete or ambiguous group: {kind}, {hops}, {arch}, {filters}")
    return rows[0]["metrics"][key]


def recurrence(groups, hops, arch, split):
    return np.array(
        [
            metric(groups, "recurrence", hops, arch, split, "accuracy", evaluated_repeats=r)["mean"]
            * 100
            for r in range(1, 9)
        ]
    )


def heatmap(ax, main, key, title, vmax):
    values = np.array(
        [
            [
                next(
                    r[key]
                    for r in main["groups"]
                    if r["phase"] == "confirmation" and r["architecture"] == a and r["hops"] == h
                )
                * 100
                for h in (2, 3, 4)
            ]
            for a in ARCHS
        ]
    )
    ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3), ["两跳", "三跳", "四跳"])
    ax.set_yticks(range(5), [a.upper() for a in ARCHS])
    ax.set_title(title, loc="left", fontsize=14, pad=12)
    for i in range(5):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}%",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if values[i, j] > vmax * 0.6 else INK,
            )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def overview(main, groups):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.83, bottom=0.17, hspace=0.55, wspace=0.24)
    heading(
        fig,
        "Loop Transformer：正式实验总览",
        "3个独立世界 × 每世界2个初始化 · 90条正式训练 · 128k步固定终点 · 答案与EOS均正确",
        1,
    )
    heatmap(axes[0, 0], main, "test_full_composite", "① ID：有组合训练经验的事实，其未见组合", 100)
    heatmap(axes[0, 1], main, "ood_composite", "② OOD：仅接受单跳训练的事实链（色标0–8%）", 8)
    ax = axes[1, 0]
    for i, (component, label) in enumerate(
        [("attention", "仅注意力"), ("mlp", "仅MLP"), ("full", "完整残差")]
    ):
        values = [
            metric(
                groups,
                "intervention",
                h,
                "l1",
                "test_composite",
                "target_complete_accuracy",
                condition=f"e000_different_{component}",
            )["mean"]
            * 100
            for h in (2, 3, 4)
        ]
        bars = ax.bar(
            np.arange(3) + (i - 1) * 0.24, values, 0.23, color=COLORS[component], label=label
        )
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], fontsize=9, padding=3)
    ax.set_xticks(range(3), ["两跳", "三跳", "四跳"])
    ax.set_ylim(0, 116)
    ax.set_ylabel("新路径正确率（%）")
    ax.set_title("③ L1第一执行层：纯首跳供体状态置换", loc="left", fontsize=14, pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax = axes[1, 1]
    for split, color, label in [
        ("atomic", "#8995A3", "单跳"),
        ("test_composite", "#26966A", "ID组合probe"),
        ("ood_composite", "#9B5AA4", "OOD组合"),
    ]:
        ax.plot(
            range(1, 9), recurrence(groups, 2, "l2", split), "o-", color=color, label=label, lw=2
        )
    ax.axvline(2, ls="--", color=MUTED, lw=1)
    ax.set_xticks(range(1, 9))
    ax.set_ylim(-2, 108)
    ax.set_xlabel("测试循环次数R（虚线为训练R=2）")
    ax.set_ylabel("正确率（%）")
    ax.set_title("④ 两跳L2：固定权重，扫描全部R=1…8", loc="left", fontsize=14, pad=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.15)
    fig.text(
        0.055,
        0.11,
        "架构：C1/C2=普通1/2层；CD=普通D层；L1=1块循环D次；L2=2块循环D/2次。两/三跳D=4，四跳D=6。",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.078,
        "置换成绩仅针对合格ID供体子集：两/三/四跳覆盖82.92%/87.60%/69.66%。MLP增量依赖供体注意力，不证明知识独占于MLP。",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.046,
        "各值先平均世界内初始化，再世界等权；100.00可能由四舍五入得到。额外测试循环改变执行过程，主成绩仍取训练R。",
        fontsize=10,
        color=MUTED,
    )
    return fig


def image_page(relative, title, subtitle, note, page):
    path = BASE / relative
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    portrait = page in (4, 5, 6)
    fig = plt.figure(figsize=(11, 16) if portrait else (16, 11))
    if portrait:
        subtitle = subtitle.replace(" · 供体", "\n供体")
        note = note.replace("虚线为", "\n虚线为")
    heading(fig, title, subtitle, page)
    ax = fig.add_axes((0.045, 0.105, 0.91, 0.765))
    ax.imshow(plt.imread(path))
    ax.axis("off")
    fig.text(0.055, 0.06, note, fontsize=10, color=MUTED)
    return fig


def scan_page(groups):
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.83, bottom=0.14, hspace=0.42, wspace=0.25)
    heading(
        fig,
        "全部正式循环扫描：增加计算的收益与损伤",
        "36个固定权重模型 · 所有R=1…8 · 折线为世界均值，阴影为三个世界范围",
        7,
    )
    for i, arch in enumerate(("l1", "l2")):
        for j, hops in enumerate((2, 3, 4)):
            ax = axes[i, j]
            for split, color, label in [
                ("atomic", "#8995A3", "单跳"),
                ("test_composite", "#26966A", "ID组合probe"),
                ("ood_composite", "#9B5AA4", "OOD组合"),
            ]:
                stats = [
                    metric(groups, "recurrence", hops, arch, split, "accuracy", evaluated_repeats=r)
                    for r in range(1, 9)
                ]
                ax.plot(
                    range(1, 9),
                    [s["mean"] * 100 for s in stats],
                    "o-",
                    color=color,
                    label=label,
                    ms=4,
                )
                ax.fill_between(
                    range(1, 9),
                    [s["min"] * 100 for s in stats],
                    [s["max"] * 100 for s in stats],
                    color=color,
                    alpha=0.12,
                )
            depth = 6 if hops == 4 else 4
            native = depth if arch == "l1" else depth // 2
            ax.axvline(native, color=MUTED, ls="--", lw=1)
            ax.set_title(f"{hops}跳 · {arch.upper()} · 训练R={native}", loc="left", fontsize=13)
            ax.set_ylim(-2, 103)
            ax.set_xticks(range(1, 9))
            ax.set_xlabel("测试循环次数R")
            ax.set_ylabel("答案+EOS正确率（%）")
            ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.text(
        0.055,
        0.07,
        "虚线标明训练循环数；不按测试结果选择最佳R。阴影是世界范围，不是置信区间。",
        color=MUTED,
        fontsize=11,
    )
    return fig


def audit_page(receipt):
    fig = plt.figure(figsize=(16, 11))
    heading(
        fig,
        "完成范围与证据边界",
        "完成核验为complete；缺项=0，错误=0 · 本图册仅整理已完成结果，未新增训练或模型评价",
        8,
    )
    ax = fig.add_axes((0.07, 0.54, 0.86, 0.3))
    ax.axis("off")
    v = receipt["verified"]
    rows = [
        ["训练", "开发15 + 敏感性6 + 正式90", str(v["training_runs"])],
        ["学习节点", "所有预定节点与采样/FLOPs记录", f"{v['evaluation_nodes']:,}"],
        ["GPU终点审计", "开发15 + 敏感性6 + 正式90", str(v["gpu_audits"])],
        ["机制评价", "开发30 + 正式180；所有执行层/组件", str(v["mechanism_splits"])],
        ["循环扫描", "开发6 + 正式36；每模型R=1…8", str(v["recurrence_runs"])],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["核验对象", "范围", "完成数量"],
        colWidths=[0.22, 0.6, 0.18],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.1)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DBE4EB")
        cell.set_facecolor("#EAF1F6" if r == 0 else "white")
    paragraphs = [
        "① ID与OOD：两类事实都接受单跳训练，只有ID事实参与组合训练；混合链未纳入主评分。",
        "② 架构比较：CD/L1/L2等展开深度；等参数浅层与循环比较还改变了残差初始化尺度。",
        "③ 机制边界：纯前缀供体不含后续关系或答案；MLP增量仍依赖注意力；混合后不重算当前MLP。",
        "④ 覆盖边界：无合格供体案例仍保留；四跳OOD反事实供体覆盖仅1.81%–3.53%。",
        "⑤ 统计边界：3个独立世界；初始化、查询、执行层和循环数不当成独立世界。",
        "⑥ 解释边界：支持受控任务中的组合调用，不证明自然大模型机制、知识独占MLP或“一轮一跳”。",
    ]
    for i, text in enumerate(paragraphs):
        fig.text(0.075, 0.46 - i * 0.055, text, fontsize=12)
    fig.text(
        0.075,
        0.09,
        "数据来源：main-report/summary.json、mechanism-report/summary.json、completion-manifest.json。",
        fontsize=10,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.055,
        "正式90端点159,930项审计全部通过，GPU重载最大NLL差为0；冻结契约与所有失败/重试记录保留。",
        fontsize=10,
        color=MUTED,
    )
    return fig


def main():
    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "font.size": 11,
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK,
        }
    )
    data = read("main-report/summary.json")
    mechanisms = read("mechanism-report/summary.json")
    receipt = read("completion-manifest.json")
    assert data["complete_runs"] == 111 and not data["missing"]
    assert receipt["state"] == "complete" and not receipt["missing"] and not receipt["errors"]
    assert mechanisms["completeness"]["all_registered_endpoints_complete"]
    groups = mechanisms["groups"]
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    with PdfPages(OUT / "loop-results-atlas.pdf") as pdf:
        figures = [overview(data, groups)]
        for page, (suffix, title) in enumerate(
            [("steps", "正式学习曲线：训练步数"), ("flops", "正式学习曲线：估算计算量")], 2
        ):
            figures.append(
                image_page(
                    f"main-report/grok-loop-confirmation-v1-learning-{suffix}.png",
                    title,
                    "五种架构 × 两/三/四跳 · 上排ID固定probe，下排全量OOD · "
                    "世界内平均初始化后世界等权",
                    "主终点使用全量ID留出池；曲线使用固定probe。阴影为世界范围；并行墙钟不作为架构速度证据。",
                    page,
                )
            )
        for h in (2, 3, 4):
            paths = list(
                (BASE / "mechanism-report").glob(f"mechanism-different-confirmation-h{h}-*.png")
            )
            assert len(paths) == 1, paths
            figures.append(
                image_page(
                    str(paths[0].relative_to(BASE)),
                    f"{h}跳正式机制评价：遍历全部执行层",
                    "各架构第一关系r1位置 · 纯首跳供体 · 左列ID、右列OOD · "
                    "供体合格子集的新路径答案+EOS正确率",
                    "attention=注意力增量；mlp=MLP增量；both=两者；full=完整残差；虚线为完整反事实输入。小覆盖OOD须按子集解释。",
                    h + 2,
                )
            )
        figures.extend([scan_page(groups), audit_page(receipt)])
        for page, fig in enumerate(figures, 1):
            path = OUT / ("overview.png" if page == 1 else f"page-{page:02d}.png")
            fig.savefig(path, dpi=180)
            pdf.savefig(fig)
            images.append(str(path.relative_to(ROOT)))
            plt.close(fig)
    source = Path(__file__)
    manifest = {
        "pages": 8,
        "pdf": str((OUT / "loop-results-atlas.pdf").relative_to(ROOT)),
        "images": images,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "input_sha256": SOURCES,
        "new_training_steps": 0,
        "new_model_evaluations": 0,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"pages": 8, "pdf": manifest["pdf"], "overview": images[0]}))


if __name__ == "__main__":
    main()
