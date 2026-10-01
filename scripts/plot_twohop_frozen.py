#!/usr/bin/env python3
"""Standalone scientific figure from the frozen experiment's complete summaries."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare_twohop_frozen import ART, read


def main():
    summary = read(ART / "summary.json")
    precision = read(ART / "full-precision-summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    conditions = ["direct", "self_bridge", "scaffold_chain", "oracle_bridge", "cot"]
    labels = ["Direct", "Self bridge", "Given decomposition", "Gold bridge", "Brief CoT"]
    colors = ["#4361A8", "#16897C", "#C49731", "#BB6152", "#8265A7"]
    for col, dataset in enumerate(["mquake", "2wiki"]):
        ax = axes[0, col]
        for i, (condition, label, color) in enumerate(zip(conditions, labels, colors, strict=True)):
            values = [
                next(
                    r["em"]
                    for r in summary["behavior"]
                    if r["model"] == model
                    and r["dataset"] == dataset
                    and r["split"] == "evaluation"
                    and r["condition"] == condition
                )
                * 100
                for model in ["small", "main"]
            ]
            ax.bar(np.arange(2) + (i - 2) * 0.15, values, 0.145, label=label, color=color)
        ax.set_xticks([0, 1], ["0.6B Base", "4B Instruct"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Canonical exact match (%)")
        ax.set_title("MQuAKE: closed book" if dataset == "mquake" else "2Wiki: full context")
        ax.grid(axis="y", alpha=0.2)
        if col == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax = axes[1, col]
        for i, (metric, label, color) in enumerate(
            zip(
                ["zero_mae", "input_mae", "feature_mae"],
                ["Predict no change", "Input gradient", "Finite MLP feature"],
                colors,
                strict=False,
            )
        ):
            values = [
                next(
                    r[metric]
                    for r in precision["prediction"]
                    if r["model"] == model
                    and r["dataset"] == dataset
                    and r["condition"] == "all_nonzero"
                )
                for model in ["small", "main"]
            ]
            ax.bar(np.arange(2) + (i - 1) * 0.23, values, 0.225, label=label, color=color)
        ax.set_xticks([0, 1], ["0.6B Base", "4B Instruct"])
        ax.set_ylabel("FP32 score-change MAE (lower is better)")
        ax.set_title(
            "MQuAKE: closed-book interventions"
            if dataset == "mquake"
            else "2Wiki: gold-support interventions"
        )
        ax.grid(axis="y", alpha=0.2)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(
        "Frozen models: behavior (128 cases/dataset) and local prediction (24 cases/dataset)\n"
        "Model size and instruction tuning differ; no pure scaling comparison",
        fontsize=12,
    )
    fig.savefig(ART / "overview.png", dpi=180)
    fig.savefig(ART / "overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
