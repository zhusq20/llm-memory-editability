"""Plot encoding and addressability as separate, explicitly scoped contrasts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
ART = PROJECT / "docs/development-artifacts/realworld-loop-diagnosis-v1"


def main():
    real = json.loads((ART / "summary.json").read_text())["runs"]
    real = [
        r
        for r in real
        if r["batch"] in ["realworld-loop-entity-v1", "realworld-loop-entity-replica-v1"]
    ]
    graph = json.loads(
        (
            PROJECT / "docs/development-artifacts/realworld-loop-addressability-v1" / "summary.json"
        ).read_text()
    )
    step = "128000" if len(graph["128000"]["runs"]) == 4 else "32000"
    graph = graph[step]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    colors = {"standard8": "#3574ad", "loop4x2": "#e18437"}
    for column, (conditions, labels) in enumerate(
        [
            (["natural", "entity"], ["Natural names", "Single-token entities"]),
            (["shared", "unique"], ["Many facts per head", "One fact per head"]),
        ]
    ):
        ax = axes[column]
        for index, arch in enumerate(["standard8", "loop4x2"]):
            positions = np.arange(2) + (index - 0.5) * 0.34
            values = []
            for condition in conditions:
                scores = (
                    [r["test_oo"] for r in real if r["name"] == f"{arch}-{condition}"]
                    if column == 0
                    else [graph[f"{arch}-{condition}"]["test_oo"]]
                )
                values.append(scores)
            means = [float(np.mean(scores)) for scores in values]
            ax.bar(
                positions,
                means,
                width=0.31,
                color=colors[arch],
                label="Standard 8" if index == 0 else "Loop 4 x 2",
            )
            for x, mean, scores in zip(positions, means, values, strict=True):
                if column == 0:
                    ax.scatter([x - 0.035, x + 0.035], scores, s=14, color="black", zorder=4)
                ax.text(x, max(scores) + 2.5, f"{mean:.1f}", ha="center", fontsize=10)
        ax.set_xticks(np.arange(2), labels)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("OO exact-match accuracy (%)")
    axes[0].set_title(
        "Same real graph: entity encoding\nwidth 256; 32k steps; 2 initializations", fontsize=11
    )
    axes[1].set_title(
        f"Synthetic causal control: fact addresses\nwidth 128; {int(step) // 1000}k steps; 1 world",
        fontsize=11,
    )
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Left dots show individual initializations, not confidence intervals. "
        "Compare architectures within each panel.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ART / "diagnosis-overview.png", dpi=180)
    fig.savefig(ART / "diagnosis-overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
