"""Render all frozen gradient endpoints; no model fitting or endpoint selection."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/development-artifacts/context-gradient-v1"


def average(values):
    return float(np.mean(values))


def main():
    rows = json.loads((OUT / "metrics.json").read_text())
    modules = json.loads((OUT / "module-metrics.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())
    with (OUT / "gradient-comparisons.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for step in (0, 32, 128):
        for kind in ("answer", "eos", "combined"):
            for predictor in ("position", "token", "shuffled"):
                subset = [
                    r
                    for r in rows
                    if r["mask"] == "open"
                    and r["step"] == step
                    and r["kind"] == kind
                    and r["predictor"] == predictor
                    and r["second"] == "neither"
                ]
                summaries.append(
                    {
                        "step": step,
                        "kind": kind,
                        "predictor": predictor,
                        "paired_comparisons": 8,
                        "mean_cosine": average([r["cosine"] for r in subset]),
                        "min_cosine": min(r["cosine"] for r in subset),
                        "max_cosine": max(r["cosine"] for r in subset),
                        "mean_relative_error": average([r["relative_error"] for r in subset]),
                        "mean_norm_ratio": average([r["norm_ratio"] for r in subset]),
                        "world_cosines": {
                            str(w): average([r["cosine"] for r in subset if r["world"] == w])
                            for w in (0, 1)
                        },
                    }
                )
    (OUT / "descriptive-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"position": "#2463a5", "token": "#b3711b", "shuffled": "#89909b"}
    labels = {
        "position": "Position-aware statistic",
        "token": "Token-only statistic",
        "shuffled": "Shuffled target control",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    for ax, kind, title in zip(
        axes,
        ("answer", "eos", "combined"),
        ("Answer gradient", "EOS gradient", "Combined objective"),
        strict=True,
    ):
        for predictor in colors:
            subset = [s for s in summaries if s["kind"] == kind and s["predictor"] == predictor]
            ax.plot(
                [s["step"] for s in subset],
                [s["mean_cosine"] for s in subset],
                "o-",
                color=colors[predictor],
                label=labels[predictor],
                linewidth=2,
            )
            for world in (0, 1):
                ax.plot(
                    [s["step"] for s in subset],
                    [s["world_cosines"][str(world)] for s in subset],
                    color=colors[predictor],
                    alpha=0.35,
                    linewidth=0.8,
                )
        ax.set(
            title=title,
            xlabel="Common reference training step",
            ylabel="Gradient contrast cosine",
            xticks=[0, 32, 128],
            ylim=(-1.0, 1.02),
        )
        ax.axhline(0, color="#b0b0b0", linewidth=0.7)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Can corpus context statistics predict organization-induced gradient changes?", fontsize=13
    )
    fig.savefig(OUT / "gradient-directions.png", dpi=180)
    fig.savefig(OUT / "gradient-directions.svg")
    plt.close(fig)

    lines = [
        "# 上下文统计与真实早期梯度：完整开发结果",
        "",
        "固定两开发世界×两初始化；4条共同中性短轨迹，0/32/128步，72个测量臂。"
        "所有统计均为开发描述；没有拟合预测比例、选取最佳状态或做显著性检验。",
        "",
        "每个测量臂使用完整2048篇文档×10位置轮换。方向表先平均两个组织相对中性组织的差值，"
        "再平均初始化和世界；共8个配对对比，但只有两个开发世界。",
        "",
        "![梯度方向预测](gradient-directions.png)",
        "",
        "## 全部预定状态与监督部分",
        "",
        "| 步数 | 监督部分 | 预测器 | 平均余弦 | 最小–最大 | 平均相对误差 | 平均范数比 |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['step']} | {row['kind']} | {row['predictor']} | "
            f"{row['mean_cosine']:.4f} | {row['min_cosine']:.4f}–{row['max_cosine']:.4f} | "
            f"{row['mean_relative_error']:.4f} | {row['mean_norm_ratio']:.4f} |"
        )
    maximum_residue = max(r["relative_cancellation_residue"] for r in audit["isolated_checks"])
    lines += [
        "",
        "相对误差为‖真实差值−预测差值‖/‖真实差值‖；范数比为‖预测差值‖/‖真实差值‖。"
        "没有用真实梯度估计最佳缩放。EOS目标恒定，shuffled在该部分与position相同，"
        "因此EOS的两者相等不算负对照失败，也不算关联预测证据。",
        "",
        "## 主终点的全部配对块",
        "",
        "| 世界 | 初始化 | 组织差值 | position余弦 | token余弦 | shuffled余弦 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for world in (0, 1):
        for seed in (0, 1):
            for condition in ("company", "project"):
                selected = {
                    r["predictor"]: r
                    for r in rows
                    if r["step"] == 0
                    and r["mask"] == "open"
                    and r["kind"] == "answer"
                    and r["world"] == world
                    and r["seed"] == seed
                    and r["first"] == condition
                    and r["second"] == "neither"
                }
                lines.append(
                    f"| {world} | {seed} | {condition}−neither | "
                    f"{selected['position']['cosine']:.4f} | "
                    f"{selected['token']['cosine']:.4f} | "
                    f"{selected['shuffled']['cosine']:.4f} |"
                )
    lines += [
        "",
        "## 数据与数值控制",
        "",
        f"72/72臂完成，原始梯度与源码哈希复核通过。阻断跨事实注意力后，主投影梯度的"
        f"最大相对消减残差为{maximum_residue:.3g}，"
        "低于运行前固定的1e-4阈值；不解释这些数值残差的余弦。",
        "",
        "监督bigram计数在三组织间相同；城市×人物上下文统计在开放时不同，"
        "阻断后相同。这是对共同曝光和掩码的审计，不是表示或行为机制证明。",
        "",
        "## 各模块的真实梯度差异",
        "",
        "下表为开放注意力、答案部分、两个组织相对中性组织的差值；"
        "报告模块内范数，不能把不同参数块的欧氏范数解释为因果重要性。",
        "",
        "| 步数 | 模块 | 平均差值范数 | 相对两臂平均范数 |",
        "| --- | --- | ---: | ---: |",
    ]
    selected_modules = (
        "token",
        "position",
        "blocks.0.attention.query",
        "blocks.0.attention.key",
        "blocks.0.attention.value",
        "blocks.0.attention",
        "blocks.0.mlp",
        "blocks.3.mlp",
        "blocks.7.mlp",
    )
    for step in (0, 32, 128):
        selected = [
            r
            for r in modules
            if r["step"] == step
            and r["mask"] == "open"
            and r["kind"] == "answer"
            and r["second"] == "neither"
        ]
        for name in selected_modules:
            values = [r["modules"][name] for r in selected]
            lines.append(
                f"| {step} | {name} | "
                f"{average([v['difference_norm'] for v in values]):.6g} | "
                f"{average([v['relative_to_mean_arm_norm'] for v in values]):.6g} |"
            )
    lines += [
        "",
        "attention无后缀的行只包含输出投影及其bias；Q/K/V分别列出。"
        "token是绑定的输入embedding/输出head，不能只解释为输出层。所有其他层、"
        "EOS、combined及company−project均保存在完整JSON/CSV中，不只保留上述示例行。",
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/run_bios_context_gradient.py summarize",
        "PYTHONPATH=src python scripts/report_bios_context_gradient.py",
        "```",
        "",
        "运行前定义与全部固定指标见[快照](preregistration.md)和[锁](preregistration-lock.json)。"
        "原始产物位于results/bios-context-gradient-v1，包含12个共同参数/优化器状态、"
        "72臂的答案/EOS全模型梯度和三个预测器的主投影矩阵。首次GPU0加载触发ECC错误，"
        "无梯度产出；随后使用GPU6完成，设计及预算不变。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
