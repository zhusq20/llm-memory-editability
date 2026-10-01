#!/usr/bin/env python3
"""Summarize independent worlds without counting nuisance repeats as worlds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/hebbian-interface-v1"


def main():
    config = json.loads((ROOT / "configs/hebbian-interface-v1.json").read_text())
    worlds = []
    manifest = {}
    for seed in config["confirmation_seeds"]:
        path = ROOT / f"results/hebbian-interface-v1/confirmation/{seed}/summary.json"
        worlds.append(json.loads(path.read_text()))
        manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for world in worlds:
        for objective, reader in world["readers"].items():
            for endpoint, measurement in reader["endpoints"].items():
                rows.append(
                    {
                        "seed": world["seed"],
                        "reader_objective": objective,
                        "endpoint": endpoint,
                        **{
                            k: measurement[k]
                            for k in [
                                "accuracy_all",
                                "accuracy_changed",
                                "n_all",
                                "n_changed",
                            ]
                        },
                    }
                )
    with (ART / "world-results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for objective in ["ce", "mse"]:
        for endpoint in ["A", "wrong_A_on_B", "B_ce", "B_mse"]:
            selected = [
                row
                for row in rows
                if row["reader_objective"] == objective and row["endpoint"] == endpoint
            ]
            aggregates.append(
                {
                    "reader_objective": objective,
                    "endpoint": endpoint,
                    "world_mean_accuracy": float(np.mean([x["accuracy_all"] for x in selected])),
                    "world_accuracies": [x["accuracy_all"] for x in selected],
                    "changed_world_mean_accuracy": float(
                        np.mean([x["accuracy_changed"] for x in selected])
                    ),
                }
            )
    primary_differences = [
        w["readers"]["mse"]["endpoints"]["B_mse"]["accuracy_all"]
        - w["readers"]["ce"]["endpoints"]["B_ce"]["accuracy_all"]
        for w in worlds
    ]
    summary = {
        "experiment": config["experiment"],
        "phase": "confirmation",
        "seeds": config["confirmation_seeds"],
        "independent_worlds": len(worlds),
        "claim_level": config["claim_level"],
        "aggregates": aggregates,
        "primary_paired_mse_minus_ce": primary_differences,
        "primary_mean_mse_minus_ce": float(np.mean(primary_differences)),
        "standalone": [{"seed": w["seed"], "memories": w["memories"]} for w in worlds],
        "audits": [{"seed": w["seed"], "audits": w["audits"]} for w in worlds],
        "timing_seconds": [w["timing_seconds"] for w in worlds],
        "cost": [w["cost"] for w in worlds],
        "source_results_sha256": manifest,
        "uncertainty": "Three independent synthetic worlds; all per-world results reported. "
        "No query-level confidence interval or general architecture ranking.",
    }
    (ART / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, objective in zip(axes, ["ce", "mse"], strict=True):
        selected = [x for x in aggregates if x["reader_objective"] == objective]
        values = [100 * x["world_mean_accuracy"] for x in selected]
        ax.bar(range(4), values, color=["#7c8fa8", "#c9b1a6", "#619db5", "#6cad85"])
        for i, record in enumerate(selected):
            dots = np.array(record["world_accuracies"]) * 100
            ax.scatter(i + np.linspace(-0.09, 0.09, len(dots)), dots, color="#222222", s=16)
        ax.set_xticks(range(4), ["A task", "Wrong A\non B", "CE B", "MSE B"])
        ax.set_ylim(0, 105)
        ax.set_title(f"Reader trained with {objective.upper()} memory A")
        ax.set_ylabel("Full-vocabulary accuracy (%)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fresh fact mapping: frozen readers, swapped memories\nDots: 3 independent worlds")
    fig.savefig(ART / "summary.png", dpi=180)
    plt.close(fig)
    print(
        json.dumps({"aggregates": aggregates, "paired_difference": primary_differences}, indent=2)
    )


if __name__ == "__main__":
    main()
