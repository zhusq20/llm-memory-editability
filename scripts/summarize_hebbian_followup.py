#!/usr/bin/env python3
"""Post-run descriptive tables and figures; frozen primary scoring is unchanged."""

import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/hebbian-future-v2"
os.environ.setdefault("MPLCONFIGDIR", str(ART / "matplotlib-cache"))


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    chain = json.loads((ART / "chain-summary.json").read_text())
    real = json.loads((ART / "real-summary.json").read_text())
    with (ART / "chain-learning.csv").open() as f:
        rows = list(csv.DictReader(f))
    architectures = ["d2w96", "d2w135", "d4w96"]
    rhos = [0.25, 0.5, 0.75]
    curves = []
    for arch in architectures:
        for rho in rhos:
            for step in (0, 128, 512, 1024, 2048):
                part = [
                    r
                    for r in rows
                    if r["architecture"] == arch
                    and float(r["rho"]) == rho
                    and int(r["step"]) == step
                ]
                curves.append(
                    {
                        "architecture": arch,
                        "rho": rho,
                        "step": step,
                        "runs": len(part),
                        **{
                            key: float(np.mean([float(r[key]) for r in part]))
                            for key in (
                                "held_composite",
                                "held_common_conflict_composite",
                                "held_common_conflict_home_copy",
                            )
                        },
                        "minimum_basic_accuracy": min(
                            float(r[key])
                            for r in part
                            for key in ("member_accuracy", "root_accuracy", "home_accuracy")
                        ),
                    }
                )
    details = {
        "note": "Descriptive follow-up after all results; no primary scoring changes.",
        "evaluation_nodes": len(rows),
        "supervised_tokens": 108 * 2048 * 252 * 2,
        "curves": curves,
    }
    (ART / "descriptive-checks.json").write_text(json.dumps(details, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for arch in architectures:
        part = [r for r in chain["aggregates"] if r["architecture"] == arch]
        for ax, metric in zip(
            axes[:2], ["held_composite", "held_common_conflict_composite"], strict=True
        ):
            ax.plot([r["rho"] for r in part], [r[metric] for r in part], "o-", label=arch)
            ax.set(xticks=rhos, ylim=(0.5, 1.02), xlabel="Training home/HQ agreement")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(fontsize=9)
            ax.grid(alpha=0.2)
    axes[0].set_title("All 48 held-out people")
    axes[1].set_title("12 fixed conflicting people")
    selection = [
        f"{kind}-L{layer}-ridge{ridge}"
        for layer in (22, 27)
        for kind in ("x", "phi")
        for ridge in (0.01, 0.1)
    ]
    for i, method in enumerate(selection):
        r = next(
            r
            for r in real["predictions"]
            if r["method"] == method
            and r["style"] == "plain"
            and r["outcome"] == "candidate_mean_error"
        )
        axes[2].errorbar(
            r["brier_gain"] * 1000,
            i,
            xerr=np.array([[r["brier_gain"] - r["ci_low"]], [r["ci_high"] - r["brier_gain"]]])
            * 1000,
            fmt="o",
            color="tab:blue" if method.startswith("x-") else "tab:orange",
            capsize=3,
        )
    axes[2].set(
        yticks=range(len(selection)),
        yticklabels=selection,
        xlabel="Brier improvement × 1000 (positive is better)",
        title="Full-candidate errors: late-layer variants",
    )
    axes[2].invert_yaxis()
    axes[2].axvline(0, color="gray", lw=1)
    axes[2].xaxis.set_major_locator(MaxNLocator(5))
    axes[2].grid(axis="x", alpha=0.2)
    fig.suptitle("108 training runs; Qwen intervals: 320-subject bootstrap, unadjusted")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"results-overview.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, arch in zip(axes, architectures, strict=True):
        for rho in rhos:
            part = [r for r in curves if r["architecture"] == arch and r["rho"] == rho]
            ax.plot(
                [r["step"] for r in part],
                [r["held_common_conflict_composite"] for r in part],
                "o-",
                label=f"rho={rho}",
            )
        ax.set(title=arch, xlabel="Training steps", ylim=(0, 1), xticks=[0, 512, 1024, 2048])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Fixed conflicting people: accuracy")
    fig.suptitle("All scheduled checkpoints; 12 runs per curve; no early stopping")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"learning-curves.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
