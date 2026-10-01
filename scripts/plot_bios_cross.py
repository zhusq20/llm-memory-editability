"""Standalone crossover figure from audited tabular results."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    rows = list(csv.DictReader((root / "learning.csv").open()))
    conditions = ("company", "project", "neither")
    colors = ("#2166ac", "#b35806", "#666666")
    names = ("Link company", "Link project", "Neither linked")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, task in enumerate(("company", "project")):
        ax = axes[0, column]
        for condition, color, label in zip(conditions, colors, names, strict=True):
            selected = [r for r in rows if r["condition"] == condition]
            steps = sorted({int(r["step"]) for r in selected})
            values = [
                [float(r[f"{task}_heldout"]) * 100 for r in selected if int(r["step"]) == step]
                for step in steps
            ]
            ax.plot(steps, [np.mean(v) for v in values], color=color, label=label, marker=".")
            ax.fill_between(
                steps, [min(v) for v in values], [max(v) for v in values], color=color, alpha=0.12
            )
        ax.set(
            title=f"Held-out {task} queries",
            xlabel="Training steps",
            ylabel="Full-answer accuracy (%)",
            ylim=(0, 102),
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    final = max(int(r["step"]) for r in rows)
    matrix = np.array(
        [
            [
                np.mean(
                    [
                        float(r[f"{task}_heldout"])
                        for r in rows
                        if r["condition"] == c and int(r["step"]) == final
                    ]
                )
                * 100
                for task in ("company", "project")
            ]
            for c in conditions
        ]
    )
    ax = axes[1, 0]
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set(
        xticks=[0, 1],
        xticklabels=["Company test", "Project test"],
        yticks=[0, 1, 2],
        yticklabels=names,
        title=f"Crossed evaluation at step {final:,}",
    )
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            f"{value:.2f}%",
            ha="center",
            va="center",
            color="white" if value > 60 else "black",
            fontsize=13,
        )
    fig.colorbar(im, ax=ax, label="Accuracy (%)", fraction=0.05)
    paired = list(csv.DictReader((root / "paired-learning.csv").open()))
    ax = axes[1, 1]
    for metric, color, label in zip(
        ("matching_effect", "matched_vs_neither", "mismatched_vs_neither"),
        colors,
        ("Matching interaction", "Matched minus neutral", "Unmatched minus neutral"),
        strict=True,
    ):
        steps = sorted({int(r["step"]) for r in paired})
        values = [[float(r[metric]) * 100 for r in paired if int(r["step"]) == s] for s in steps]
        ax.plot(steps, [np.mean(v) for v in values], label=label, color=color, marker=".")
        ax.fill_between(
            steps, [min(v) for v in values], [max(v) for v in values], color=color, alpha=0.1
        )
    ax.axhline(0, color="black", lw=0.7)
    ax.set(title="Paired contrasts", xlabel="Training steps", ylabel="Percentage points")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.suptitle(
        "Organization x relation crossover | 2 worlds x 2 initializations\n"
        "Shading: observed block range, not confidence intervals",
        fontsize=13,
    )
    fig.savefig(root / "crossover-learning.png", dpi=180)
    fig.savefig(root / "crossover-learning.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for condition, color, label in zip(conditions, colors, names, strict=True):
        selected = [r for r in rows if r["condition"] == condition]
        steps = sorted({int(r["step"]) for r in selected})
        values = [
            [float(r["base_accuracy"]) * 100 for r in selected if int(r["step"]) == s]
            for s in steps
        ]
        ax.plot(steps, [np.mean(v) for v in values], label=label, color=color, marker=".")
        ax.fill_between(
            steps, [min(v) for v in values], [max(v) for v in values], color=color, alpha=0.1
        )
    ax.set(
        xlabel="Training steps",
        ylabel="Full-answer accuracy (%)",
        ylim=(0, 102),
        title="Learning the same base facts under three organizations\n"
        "Means and observed ranges over four paired blocks",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(root / "crossover-base-learning.png", dpi=180)
    fig.savefig(root / "crossover-base-learning.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bios-cross-dev-v1")
    args = parser.parse_args()
    plot(Path(args.output))
