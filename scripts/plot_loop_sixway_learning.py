#!/usr/bin/env python3
"""Plot captured training logs and saved atomic evaluations without model execution."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ARMS = {
    "loop": ("Loop", "#64748b", "-", "o"),
    "discoloop": ("DiscoLoop", "#a855f7", "-", "s"),
    "infiloop": ("InfiLoop", "#2563eb", "-", "^"),
    "infiloop-feedback": ("InfiLoop + feedback", "#06b6d4", "--", "v"),
    "invariant": ("Invariant", "#15803d", "-", "D"),
    "invariant-no-feedback": ("Invariant − feedback", "#d97706", "--", "P"),
}
BASE = Path("/ossfs/workspace/llm-memory-editability")


def dump_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture(output):
    runs, configs = {}, {}
    for batch in ("loop-fourway-terminal-20261024", "loop-feedback-controls-20261024"):
        config = json.loads((BASE / batch / "config.json").read_text())
        configs[batch] = config
        for arm in config["arms"]:
            root = BASE / batch / "runs" / arm
            raw = (root / "progress.jsonl").read_bytes()
            lines = raw.splitlines(keepends=True)
            complete_lines = [line for line in lines if line.endswith(b"\n")]
            progress = [json.loads(line) for line in complete_lines]
            assert all(
                b["step"] - a["step"] == 100 for a, b in zip(progress, progress[1:], strict=False)
            )
            assert progress[0]["step"] == 100
            learning_bytes = (root / "learning.json").read_bytes()
            runs[arm] = {
                "root": str(root),
                "progress": progress,
                "learning": json.loads(learning_bytes),
                "initial_backbone_sha256": json.loads((root / "run.json").read_text())[
                    "initial_backbone_sha256"
                ],
                "source_sha256": {
                    "progress.jsonl": hashlib.sha256(raw).hexdigest(),
                    "learning.json": hashlib.sha256(learning_bytes).hexdigest(),
                },
                "ignored_incomplete_log_lines": len(lines) - len(complete_lines),
            }
    first, second = configs.values()
    for key in ("training", "model", "data", "dropout_seed", "input_origin", "paired_recipe"):
        assert first[key] == second[key], key
    assert len({run["initial_backbone_sha256"] for run in runs.values()}) == 1
    common_eval = sorted(
        set.intersection(*(set(row["step"] for row in run["learning"]) for run in runs.values()))
    )
    common_progress = min(run["progress"][-1]["step"] for run in runs.values())
    reference = None
    for run in runs.values():
        exposure = [
            [row[k] for k in ("step", "supervised_tokens", "sampled_sample_rounds", "lr")]
            for row in run["progress"]
            if row["step"] <= common_progress
        ]
        if reference is None:
            reference = exposure
        else:
            assert exposure == reference, "Steps, supervision, depths or learning rate differ"
    now = datetime.now(timezone.utc)
    result = {
        "captured_utc": now.isoformat(),
        "captured_local": now.astimezone(ZoneInfo("America/Chicago")).isoformat(),
        "shared_evaluation_step": max(common_eval),
        "shared_progress_step": common_progress,
        "comparison_verified": "Same model backbone, recipe, exposure, depth RNG and learning rate",
        "new_training_updates": 0,
        "new_model_forwards": 0,
        "configs": configs,
        "runs": runs,
    }
    (output / "snapshot.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def derive(snapshot, output):
    progress_rows, smooth_rows, eval_rows = [], [], []
    for arm, run in snapshot["runs"].items():
        previous = 0
        arm_rows = []
        for row in run["progress"]:
            count = row["supervised_tokens"] - previous
            previous = row["supervised_tokens"]
            assert count > 0 and math.isfinite(row["mean_loss"]) and row["mean_loss"] > 0
            record = {
                "arm": arm,
                "step": row["step"],
                "loss": row["mean_loss"],
                "supervised_tokens_in_window": count,
                "gradient_norm_before_clip": row["mean_grad_norm"],
                "beta": row.get("beta"),
            }
            progress_rows.append(record)
            arm_rows.append(record)
        for begin in range(0, len(arm_rows) - 9, 10):
            window = arm_rows[begin : begin + 10]
            smooth_rows.append(
                {
                    "arm": arm,
                    "step": window[-1]["step"],
                    "loss_1000_steps": sum(
                        r["loss"] * r["supervised_tokens_in_window"] for r in window
                    )
                    / sum(r["supervised_tokens_in_window"] for r in window),
                    "gradient_1000_steps": float(
                        np.mean([r["gradient_norm_before_clip"] for r in window])
                    ),
                    "beta": window[-1]["beta"],
                }
            )
        for row in run["learning"]:
            for weights, rounds in row["metrics"].items():
                for depth, metrics in rounds.items():
                    for panel in ("atomic_id", "atomic_ood"):
                        m = metrics[panel]
                        eval_rows.append(
                            {
                                "arm": arm,
                                "step": row["step"],
                                "weights": weights,
                                "rounds": int(depth),
                                "panel": panel,
                                "accuracy_percent": 100 * m["answer_accuracy"],
                                "answer_nll": m["answer_nll"],
                                "answer_and_eos_mean_nll": m["nll"],
                            }
                        )
    dump_csv(output / "training-100-step-windows.csv", progress_rows)
    dump_csv(output / "training-1000-step-windows.csv", smooth_rows)
    dump_csv(output / "atomic-evaluations.csv", eval_rows)
    return progress_rows, smooth_rows, eval_rows


def style():
    path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=path).get_name(),
            "font.size": 11,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#334155",
            "text.color": "#0f172a",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def main_plot(snapshot, raw, smooth, evaluations, output):
    common = snapshot["shared_evaluation_step"]
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.1))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.12, top=0.79, hspace=0.46, wspace=0.23)
    handles = []
    for arm, (label, color, linestyle, marker) in ARMS.items():
        progress = [r for r in raw if r["arm"] == arm]
        averaged = [r for r in smooth if r["arm"] == arm]
        atomic = [
            r
            for r in evaluations
            if r["arm"] == arm
            and r["weights"] == "ema"
            and r["rounds"] == 8
            and r["panel"] == "atomic_ood"
        ]
        for i, limit in enumerate((common, 64000)):
            pr = [r for r in progress if r["step"] <= limit]
            av = [r for r in averaged if r["step"] <= limit]
            at = [r for r in atomic if r["step"] <= limit]
            axes[i, 0].plot(
                [r["step"] / 1000 for r in pr],
                [r["loss"] for r in pr],
                color=color,
                alpha=0.13,
                linewidth=0.65,
            )
            (line,) = axes[i, 0].plot(
                [r["step"] / 1000 for r in av],
                [r["loss_1000_steps"] for r in av],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                label=label,
            )
            axes[i, 1].plot(
                [r["step"] / 1000 for r in at],
                [r["accuracy_percent"] for r in at],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.9,
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.4,
                clip_on=False,
            )
            if i == 0:
                handles.append(line)
    titles = [
        f"A  训练 loss：六组共同评测区间（0–{common // 1000}k）",
        "B  B图原子准确率：EMA，R8",
        "C  训练 loss：后续轨迹（至64k）",
        "D  B图原子准确率：后续已保存节点",
    ]
    for i, row in enumerate(axes):
        limit = common / 1000 if i == 0 else 64
        for j, ax in enumerate(row):
            ax.set_title(titles[2 * i + j], loc="left", fontsize=12, pad=12)
            ax.set_xlabel("训练更新数（千步）")
            ax.set_xlim(0, limit)
            ax.grid(True, color="#e2e8f0", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.set_xticks([0, 4, 8, 12, 16] if i == 0 else [0, 8, 16, 32, 48, 64])
            if j == 0:
                ax.set_yscale("log")
                ax.set_ylim(0.003, 12)
                ax.set_ylabel("监督token平均CE（对数轴）")
                ax.axhline(math.log(500) / 2, color="#94a3b8", linewidth=1, linestyle=":")
            else:
                ax.set_ylim(-3, 104)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.set_ylabel("答案准确率（%）")
            if i == 1:
                ax.axvline(common / 1000, color="#cbd5e1", linestyle=":", linewidth=1.1)
    axes[0, 0].text(
        0.98,
        math.log(500) / 2 * 1.13,
        "½ ln(500) ≈ 3.11",
        ha="right",
        fontsize=9,
        color="#64748b",
        transform=axes[0, 0].get_yaxis_transform(),
    )
    axes[1, 1].annotate(
        "原 InfiLoop：32k 66.56% → 64k 100%",
        xy=(32, 66.56),
        xytext=(20, 36),
        fontsize=10,
        color="#2563eb",
        arrowprops={"arrowstyle": "-", "color": "#2563eb"},
    )
    fig.suptitle(
        "六组学习曲线：按训练步数对齐", x=0.075, y=0.975, ha="left", fontsize=21, weight="bold"
    )
    fig.text(
        0.075,
        0.927,
        "同一主干初值、数据顺序、学习率与逐样本随机深度；每个B原子评测池5,000题。",
        fontsize=11,
    )
    fig.legend(
        handles,
        [ARMS[a][0] for a in ARMS],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.895),
        ncol=3,
        frameon=False,
        columnspacing=2.5,
        handlelength=3,
    )
    fig.text(
        0.075,
        0.055,
        "loss粗线：每1,000步按监督token数加权；细线：原始100步记录。准确率只在实际评测节点标点，连线仅作指引。",
        fontsize=9.5,
    )
    fig.text(
        0.075,
        0.025,
        "后两组在已有数据处截断，不延长到未完成步数。训练loss使用随机深度/开启dropout；准确率使用EMA/R8/关闭dropout。",
        fontsize=9.5,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"sixway-loss-and-b-atomic.{suffix}", dpi=170)
    plt.close(fig)


def overview_plot(snapshot, smooth, evaluations, output):
    """Retain the complete captured trajectories alongside the early-learning zoom."""
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.5))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.2, top=0.7, wspace=0.22)
    maximum = max(run["progress"][-1]["step"] for run in snapshot["runs"].values())
    for arm, (label, color, linestyle, marker) in ARMS.items():
        rows = [r for r in smooth if r["arm"] == arm]
        axes[0].plot(
            [r["step"] / 1000 for r in rows],
            [r["loss_1000_steps"] for r in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
        )
        rows = [
            r
            for r in evaluations
            if r["arm"] == arm
            and r["weights"] == "ema"
            and r["rounds"] == 8
            and r["panel"] == "atomic_ood"
        ]
        axes[1].plot(
            [r["step"] / 1000 for r in rows],
            [r["accuracy_percent"] for r in rows],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white",
            linewidth=1.9,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("训练loss：每1,000步监督token加权均值")
    axes[1].set_ylabel("B图原子准确率（%，EMA/R8）")
    axes[1].set_ylim(-3, 104)
    for ax in axes:
        ax.set_xlim(0, maximum / 1000)
        ax.set_xlabel("训练更新数（千步）")
        ax.grid(True, color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("六组完整已记录轨迹", x=0.075, y=0.97, ha="left", fontsize=21)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 0.9), frameon=False
    )
    fig.text(
        0.075,
        0.06,
        "同一横轴；各组截止于自己的最新记录。准确率仅绘制实际保存节点，线段仅作指引。",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"sixway-full-history.{suffix}", dpi=170)
    plt.close(fig)


def diagnostic_plot(smooth, evaluations, output):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.78, bottom=0.19, wspace=0.28)
    for arm, (label, color, linestyle, _) in ARMS.items():
        rows = [r for r in smooth if r["arm"] == arm and r["step"] <= 64000]
        axes[0].plot(
            [r["step"] / 1000 for r in rows],
            [r["gradient_1000_steps"] for r in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
        )
        if rows[0]["beta"] is not None:
            axes[1].plot(
                [r["step"] / 1000 for r in rows],
                [r["beta"] for r in rows],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
            )
    for arm in ("infiloop", "infiloop-feedback"):
        for weights, line, mark in (("ema", "-", "o"), ("raw", "--", "x")):
            rows = [
                r
                for r in evaluations
                if r["arm"] == arm
                and r["weights"] == weights
                and r["rounds"] == 4
                and r["panel"] == "atomic_ood"
                and r["step"] <= 64000
            ]
            axes[2].plot(
                [r["step"] / 1000 for r in rows],
                [r["accuracy_percent"] for r in rows],
                color=ARMS[arm][1],
                linestyle=line,
                marker=mark,
                label=f"{ARMS[arm][0]} / {weights.upper()}",
                linewidth=1.7,
            )
    axes[0].set_yscale("log")
    axes[0].axhline(1, color="#94a3b8", linestyle=":")
    axes[0].set_ylabel("裁剪前全模型梯度范数：1,000步均值")
    axes[1].set_ylabel("历史衰减 β（不是实际接收率 γ）")
    axes[1].set_ylim(0, 1)
    axes[2].set_ylabel("B原子准确率（%，固定R4）")
    axes[2].set_ylim(-3, 104)
    axes[2].legend(fontsize=8, loc="lower right", frameon=False)
    for ax, title in zip(axes, ("梯度记录", "聚合衰减的变化", "同轮数排查EMA滞后"), strict=True):
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xlabel("训练更新数（千步）")
        ax.set_xlim(0, 64)
        ax.set_xticks([0, 16, 32, 48, 64])
        ax.grid(True, alpha=0.2)
    fig.suptitle("辅助观察：梯度、β与RAW/EMA", x=0.055, ha="left", fontsize=18)
    fig.text(
        0.055,
        0.86,
        "颜色同主图。日志记录的是梯度范数均值，不能据此计算逐步裁剪比例；β的变化本身不能证明因果。",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"optimization-diagnostics.{suffix}", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "snapshot.json"
    data = json.loads(path.read_text()) if path.exists() else capture(args.output)
    raw, smooth, evaluations = derive(data, args.output)
    style()
    main_plot(data, raw, smooth, evaluations, args.output)
    overview_plot(data, smooth, evaluations, args.output)
    diagnostic_plot(smooth, evaluations, args.output)
    matched = [r for r in smooth if r["step"] in (8000, 16000, 24000, 32000, 64000)]
    (args.output / "matched-loss-summary.json").write_text(json.dumps(matched, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "captured_local": data["captured_local"],
                "shared_evaluation_step": data["shared_evaluation_step"],
                "shared_training_step": data["shared_progress_step"],
                "figures": 7,
                "training_windows": len(raw),
                "atomic_records": len(evaluations),
            }
        )
    )


if __name__ == "__main__":
    main()
