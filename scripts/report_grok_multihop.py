#!/usr/bin/env python3
"""Summarize registered longer-path runs without changing scientific outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def first_sustained_node(rows, key, threshold=0.9, count=3):
    for i in range(len(rows) - count + 1):
        if all(row[key]["accuracy"] >= threshold for row in rows[i : i + count]):
            return rows[i]["step"]
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-multihop-development-v1.json")
    parser.add_argument("--out", default="docs/development-artifacts/grok-multihop-v1")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    endpoints, learning = [], []
    for run_id, item in cfg["runs"].items():
        spec = {**cfg["base"], **item}
        directory = ROOT / cfg["output_root"] / spec["phase"] / run_id
        if not (directory / "complete.json").exists():
            continue
        complete = json.loads((directory / "complete.json").read_text())
        rows = json.loads((directory / "learning.json").read_text())
        world = json.loads((directory / "world-metadata.json").read_text())
        endpoint = complete["endpoint"]
        record = {
            "run_id": run_id,
            "phase": spec["phase"],
            "world": spec["world_seed"],
            "initialization": spec["initialization"],
            "hops": spec["hops"],
            "layers": spec["layers"],
            "width": spec["width"],
            "phi": spec["phi"],
            "steps": endpoint["step"],
            "parameters": complete["parameters"],
            "atomic": endpoint["atomic"]["accuracy"],
            "train_composite": endpoint["train_composite"]["accuracy"],
            "test_probe": endpoint["test_composite"]["accuracy"],
            "test_full": endpoint["test_full_composite"]["accuracy"],
            "test_full_n": endpoint["test_full_composite"]["n"],
            "autonomous_calls_probe": endpoint["autonomous_calls"]["accuracy"],
            "ood_composite": endpoint["ood_composite"]["accuracy"],
            "ood_composite_n": endpoint["ood_composite"]["n"],
            "T90_probe": first_sustained_node(rows, "test_composite"),
            "atomic_mean_exposure": endpoint["counts"]["atomic"] / len_atomic(world),
            "estimated_training_flops": endpoint["estimated_training_flops"],
            "examples": endpoint["examples"],
            "effective_input_tokens": endpoint["effective_input_tokens"],
            "training_seconds": complete["training_seconds"],
            "evaluation_seconds": complete["evaluation_seconds"],
            "dataset_sha256": world["dataset_sha256"],
            "training_fraction_of_id_paths": world["training_fraction_of_id_paths"],
        }
        endpoints.append(record)
        for row in rows:
            learning.append(
                {
                    "run_id": run_id,
                    "hops": spec["hops"],
                    "layers": spec["layers"],
                    "phi": spec["phi"],
                    "step": row["step"],
                    "atomic_mean_exposure": row["counts"]["atomic"] / len_atomic(world),
                    "estimated_training_flops": row["estimated_training_flops"],
                    "atomic": row["atomic"]["accuracy"],
                    "train_composite": row["train_composite"]["accuracy"],
                    "test_probe": row["test_composite"]["accuracy"],
                    "autonomous_calls_probe": row["autonomous_calls"]["accuracy"],
                }
            )
    for filename, values in (("endpoint-table.csv", endpoints), ("learning.csv", learning)):
        if values:
            with (out / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    worlds = sorted({record["world"] for record in endpoints})
    is_confirmation = bool(endpoints) and all(
        record["phase"] == "confirmation" for record in endpoints
    )
    batch_label = "Independent confirmation" if is_confirmation else "Development"
    figure_filename = "learning-confirmation.png" if is_confirmation else "learning-development.png"
    summary = {
        "config": args.config,
        "completed_runs": len(endpoints),
        "registered_runs": len(cfg["runs"]),
        "endpoints": endpoints,
        "scope": (
            "Independent confirmation; endpoint rows remain per world"
            if is_confirmation
            else "Separate fixed-length tasks; development is not independent confirmation"
        ),
        "worlds": worlds,
        "figure": figure_filename,
        "T90_definition": "First of three consecutive probe nodes >=90%; null is censored",
        "primary_endpoint": "Complete answer plus EOS on the entire reserved ID test pool",
        "learning_curve": "Fixed random probe at all nodes, not whole pool until endpoint",
        "totals": {
            key: sum(record[key] for record in endpoints)
            for key in (
                "steps",
                "examples",
                "effective_input_tokens",
                "estimated_training_flops",
                "training_seconds",
                "evaluation_seconds",
            )
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if learning:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        for record in endpoints:
            rows = [row for row in learning if row["run_id"] == record["run_id"]]
            layer_label = "layer" if record["layers"] == 1 else "layers"
            label = (
                f"{record['hops']} hops / {record['layers']} {layer_label} / phi={record['phi']:g}"
            )
            if len(worlds) > 1:
                label += f" / world={record['world']}"
            for ax, key in zip(axes, ("step", "estimated_training_flops"), strict=True):
                line = ax.plot(
                    [r[key] for r in rows], [r["test_probe"] for r in rows], label=label
                )[0]
                ax.scatter(
                    [rows[-1][key]],
                    [record["test_full"]],
                    color=line.get_color(),
                    marker="x",
                    s=45,
                    zorder=3,
                )
                ax.set_ylim(0, 1.02)
                ax.set_ylabel("Held-out probe answer + EOS accuracy")
                ax.grid(alpha=0.2)
        axes[0].set_xlabel("Training updates")
        axes[1].set_xlabel("Estimated training matrix FLOPs")
        axes[0].legend(fontsize=7)
        fig.suptitle(
            f"{batch_label}: {len(worlds)} world(s); fixed probe curves; x = full endpoint",
            fontsize=10,
        )
        fig.savefig(out / figure_filename, dpi=180)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


def len_atomic(world):
    return world["counts"]["atomic"]


if __name__ == "__main__":
    main()
