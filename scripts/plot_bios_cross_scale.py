"""Export size-by-relation learning curves and paired effects from audited tables."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("company", "project", "neither")
COLORS = ("#2166ac", "#b35806", "#666666")
LABELS = ("Link company", "Link project", "Neither linked")


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, root, name):
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def plot(root):
    rows = read(root / "learning.csv")
    widths = (64, 128, 256, 768)
    params = [
        np.mean([float(r["parameters"]) for r in rows if int(r["width"]) == w]) / 1e6
        for w in widths
    ]
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), constrained_layout=True, sharex=True)
    for i, width in enumerate(widths):
        for j, (metric, title) in enumerate(
            (
                ("base_accuracy", "Base facts"),
                ("company_heldout", "Company held-out QA"),
                ("project_heldout", "Project held-out QA"),
            )
        ):
            ax = axes[i, j]
            for condition, color, label in zip(CONDITIONS, COLORS, LABELS, strict=True):
                selected = [
                    r for r in rows if int(r["width"]) == width and r["condition"] == condition
                ]
                steps = sorted({int(r["step"]) for r in selected})
                values = [
                    [float(r[metric]) * 100 for r in selected if int(r["step"]) == s] for s in steps
                ]
                ax.plot(steps, [np.mean(v) for v in values], color=color, label=label, marker=".")
                ax.fill_between(
                    steps,
                    [min(v) for v in values],
                    [max(v) for v in values],
                    color=color,
                    alpha=0.12,
                )
            ax.set(title=f"{params[i]:.3f}M: {title}", ylim=(-1, 101), ylabel="Accuracy (%)")
            ax.grid(alpha=0.2)
            if i == 3:
                ax.set_xlabel("Training steps")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "All trained sizes: equal exposure and optimization steps\n"
        "Lines: four paired-block means; shading: observed range, not confidence intervals"
    )
    save(fig, root, "scale-learning-curves")

    final = [r for r in rows if int(r["step"]) == 15360]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, width, param in zip(axes, widths, params, strict=True):
        matrix = np.array(
            [
                [
                    np.mean(
                        [
                            float(r[f"{task}_heldout"]) * 100
                            for r in final
                            if int(r["width"]) == width and r["condition"] == condition
                        ]
                    )
                    for task in ("company", "project")
                ]
                for condition in CONDITIONS
            ]
        )
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if value > 60 else "black",
            )
        ax.set(
            title=f"{param:.3f}M parameters",
            xticks=[0, 1],
            xticklabels=["Company", "Project"],
            yticks=[0, 1, 2],
            yticklabels=LABELS,
            xlabel="Test relation",
        )
    fig.colorbar(im, ax=axes, label="Held-out full-answer accuracy (%)", shrink=0.8)
    fig.suptitle("Training organization x test relation at 15,360 steps")
    save(fig, root, "scale-crossover-matrices")

    paired = [r for r in read(root / "paired-learning.csv") if int(r["step"]) == 15360]
    edits = read(root / "paired-editing.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    panels = [
        (
            axes[0, 0],
            paired,
            "matching_effect",
            "Learning: relation matching effect",
            [(None, None, "Matching", COLORS[0])],
        ),
        (
            axes[0, 1],
            paired,
            None,
            "Learning: two-task average gain vs neither",
            [
                ("company_mean_vs_neither", None, LABELS[0], COLORS[0]),
                ("project_mean_vs_neither", None, LABELS[1], COLORS[1]),
            ],
        ),
    ]
    for ax, source, metric, title, series in panels:
        for override, _, label, color in series:
            key = override or metric
            values = [[float(r[key]) * 100 for r in source if int(r["width"]) == w] for w in widths]
            ax.plot(params, [np.mean(v) for v in values], marker="o", color=color, label=label)
            for x, vs in zip(params, values, strict=True):
                ax.scatter([x] * len(vs), vs, color=color, alpha=0.35, s=18)
        ax.set_title(title)
    for ax, scope in zip(axes[1], ("mlp", "all"), strict=True):
        for kind, color in (("coherent", COLORS[0]), ("exception", COLORS[1])):
            values = [
                [
                    float(r["matching_effect"]) * 100
                    for r in edits
                    if int(r["width"]) == w
                    and r["kind"] == kind
                    and r["scope"] == scope
                    and r["metric"] == "D_heldout"
                ]
                for w in widths
            ]
            ax.plot(params, [np.mean(v) for v in values], marker="o", color=color, label=kind)
            for x, vs in zip(params, values, strict=True):
                ax.scatter([x] * len(vs), vs, color=color, alpha=0.35, s=18)
        ax.set_title(f"Editing ({scope}): held-out propagation matching effect")
    for ax in axes.flat:
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xscale("log")
        ax.set(
            xticks=params,
            xticklabels=[f"{p:.2f}" for p in params],
            xlabel="Parameters (millions, log scale)",
            ylabel="Difference (percentage points)",
        )
        ax.minorticks_off()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Paired effects across model sizes\nDots: four world/initialization blocks; lines: means"
    )
    save(fig, root, "scale-paired-effects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bios-cross-scale-dev-v1")
    args = parser.parse_args()
    plot(Path(args.output))
