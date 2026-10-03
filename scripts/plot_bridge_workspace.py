"""Plot audited bridge interventions; worlds remain the independent comparison unit."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_memory_editability.grok_depth import write_json


def plot(table, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(Path(table).open()))
    arms = ("baseline", "bridge_ce", "aligned")
    labels = ("Full CE", "+ bridge CE", "+ alignment")
    colors = ("#6b7280", "#c67c28", "#237caa", "#4b9472", "#a85779")
    panels = [
        (
            "Causal counterfactual answers",
            "swap",
            "cf_accuracy",
            [
                ("swap", "Prefix J frame"),
                ("swap_embedding", "Input entity frame"),
                ("random_swap_embedding", "Rotated frame, same norm"),
                ("full_prefix", "Full prefix state"),
            ],
        ),
        (
            "Local component deletion",
            "native",
            "accuracy",
            [
                ("baseline", "Native"),
                ("erase_state_embedding", "Remove entity state"),
                ("erase_mlp_embedding", "Remove MLP entity component"),
                ("random_mlp_embedding", "Rotated MLP component"),
            ],
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    statistics = []
    for ax, (title, family, metric, conditions) in zip(axes, panels, strict=True):
        for j, (condition, caption) in enumerate(conditions):
            means, lows, highs = [], [], []
            for arm in arms:
                worlds = defaultdict(list)
                for row in rows:
                    if (
                        row["family"] != "alignment"
                        or row["stratum"] != "strict_test"
                        or row["intervention"] != family
                        or row["condition"] != condition
                        or int(row["layer"]) != (-1 if condition == "baseline" else 0)
                    ):
                        continue
                    match = re.search(r"w(\d+)-i(\d+)-(.*)$", row["run"])
                    if match and match[3] == arm:
                        worlds[match[1]].append(float(row[metric]))
                values = np.asarray([np.mean(a) for a in worlds.values()])
                means.append(float(values.mean()) if len(values) else np.nan)
                lows.append(float(values.min()) if len(values) else np.nan)
                highs.append(float(values.max()) if len(values) else np.nan)
                statistics.append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "metric": metric,
                        "worlds": len(values),
                        "world_means": values.tolist(),
                    }
                )
            means, lows, highs = (100 * np.asarray(a) for a in (means, lows, highs))
            ax.errorbar(
                np.arange(3) + (j - 1.5) * 0.055,
                means,
                yerr=np.stack((means - lows, highs - means)),
                label=caption,
                color=colors[j],
                marker="o",
                capsize=3,
                linewidth=1.5,
            )
        ax.set_xticks(np.arange(3), labels)
        ax.set_title(title)
        ax.set_ylim(-2, 103)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Strict composition accuracy (%)")
    figure.suptitle(
        "Shared one-block GPT x2: 3 existing worlds, 2 initializations\n"
        "Ranges show world means; relation queries are repeated measurements",
        fontsize=11,
    )
    for extension in ("png", "pdf", "svg"):
        figure.savefig(out / f"representation-interface.{extension}", dpi=180)
    plt.close(figure)
    write_json(out / "representation-interface-data.json", statistics)

    for step in sorted(
        {int(row["checkpoint_step"]) for row in rows if row["family"] == "reproduction"}
    ):
        figure, axes = plt.subplots(
            2, 2, figsize=(12, 8), sharex=True, sharey=True, layout="constrained"
        )
        for ax, (phi, wd) in zip(
            axes.flat, ((3.6, 0.1), (7.2, 0.1), (12.6, 0.1), (7.2, 0.3)), strict=True
        ):
            for arch, color in (("standard8", "#bd6464"), ("loop4x2", "#267da7")):
                name = f"{arch}-phi{phi}-wd{wd}-s{step}"
                for condition, style, caption in (
                    ("full_prefix", "-", "full prefix"),
                    ("swap_embedding", "--", "entity frame"),
                    ("swap", ":", "prefix J frame"),
                ):
                    by_layer = defaultdict(list)
                    for row in rows:
                        if (
                            row["run"] == name
                            and row["intervention"] == "swap"
                            and row["condition"] == condition
                            and row["stratum"] in {"ood_id", "ood_ood"}
                        ):
                            by_layer[int(row["layer"])].append(row)
                    layers = sorted(by_layer)
                    if not layers:
                        continue
                    values = [
                        sum(float(r["cf_accuracy"]) * int(r["n_queries"]) for r in by_layer[layer])
                        / sum(int(r["n_queries"]) for r in by_layer[layer])
                        for layer in layers
                    ]
                    ax.plot(
                        np.asarray(layers) + 1,
                        100 * np.asarray(values),
                        style,
                        color=color,
                        label=f"{arch}: {caption}",
                        linewidth=1.6,
                    )
            ax.set_title(f"phi={phi}, weight decay={wd}")
            ax.set_xticks(np.arange(1, 9))
            ax.grid(alpha=0.2)
        for ax in axes[-1]:
            ax.set_xlabel("Executed layer after which r1 is changed")
        for ax in axes[:, 0]:
            ax.set_ylabel("Counterfactual full generation (%)")
        axes[0, 0].legend(fontsize=7, loc="upper left")
        figure.suptitle(
            f"Matched checkpoint: {step:,} updates, one graph / initialization\n"
            "OOD first facts, all graph-shared r2; second-fact roles can differ",
            fontsize=11,
        )
        for extension in ("png", "pdf", "svg"):
            figure.savefig(out / f"architecture-s{step}.{extension}", dpi=180)
        plt.close(figure)
    print(f"Plots saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.table, args.out)
