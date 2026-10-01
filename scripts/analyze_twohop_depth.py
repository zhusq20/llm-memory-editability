#!/usr/bin/env python3
"""Descriptive follow-up, specified after interim results; never selects models."""

from __future__ import annotations

import csv
import os

import numpy as np

from llm_memory_editability.twohop_depth import ART, DATA, RESULTS, digest, read, write


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save_csv(path, values):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    audit = read(ART / "audit.json")
    lock = read(ART / "lock.json")
    cfg = lock["config"]
    assert audit["complete"] and audit["runs"] == 28 and audit["nodes"] == 196
    learning = rows(ART / "learning.csv")
    flow = rows(ART / "interventions.csv")
    pairs = []
    for step in (cfg["primary_step"], cfg["steps"]):
        selected = [row for row in learning if int(row["step"]) == step]
        for world in cfg["worlds"]:
            for seed in cfg["initializations"]:
                unit = {
                    row["arch"]: row
                    for row in selected
                    if int(row["world"]) == world and int(row["seed"]) == seed
                }
                for other in ("d1w128", "d1w312", "d2w220"):
                    pairs.append(
                        {
                            "step": step,
                            "world": world,
                            "seed": seed,
                            "comparison": "d6w128-minus-" + other,
                            "test_difference_pp": 100
                            * (
                                float(unit["d6w128"]["test_exact"])
                                - float(unit[other]["test_exact"])
                            ),
                        }
                    )
    save_csv(ART / "paired-differences.csv", pairs)

    strata = []
    for job in lock["jobs"]:
        w = dict(np.load(DATA / f"world-{job['world']}.npz"))
        cases, r = w["cases"], cfg["relations"]
        alt_index = cases[:, 6] * r * r + cases[:, 2] * r + cases[:, 3]
        alt_trained = w["train_mask"][alt_index]
        for step in (cfg["primary_step"], cfg["steps"]):
            d = np.load(RESULTS / job["name"] / f"diagnostics-{step}.npz")
            baseline = d["clean_pred"] == d["new_y"]
            for layer in range(job["arch"]["layers"] + 1):
                for kind in ("bridge", "same_bridge", "random", "source", "prefix", "identity"):
                    alt = d[f"{kind}_{layer}_pred"] == d["new_y"]
                    for label, mask in (("seen", alt_trained), ("held_out", ~alt_trained)):
                        strata.append(
                            {
                                "run": job["name"],
                                "arch": job["arch"]["name"],
                                "world": job["world"],
                                "seed": job["seed"],
                                "step": step,
                                "layer": layer,
                                "kind": kind,
                                "donor_query_split": label,
                                "n": int(mask.sum()),
                                "counterfactual_correct": int(alt[mask].sum()),
                                "baseline_counterfactual_correct": int(baseline[mask].sum()),
                                "old_correct": int(
                                    (d[f"{kind}_{layer}_pred"][mask] == d["old_y"][mask]).sum()
                                ),
                            }
                        )
    save_csv(ART / "donor-query-strata.csv", strata)
    final = [row for row in learning if int(row["step"]) == cfg["steps"]]
    world_means = []
    for step in (cfg["primary_step"], cfg["steps"]):
        for arch in cfg["architectures"]:
            for world in cfg["worlds"]:
                values = [
                    float(row["test_exact"])
                    for row in learning
                    if int(row["step"]) == step
                    and row["arch"] == arch["name"]
                    and int(row["world"]) == world
                ]
                assert len(values) == 2
                world_means.append(
                    {
                        "step": step,
                        "arch": arch["name"],
                        "world": world,
                        "test_mean": float(np.mean(values)),
                    }
                )
    save_csv(ART / "world-means.csv", world_means)
    write(
        ART / "descriptive-analysis.json",
        {
            "status": (
                "post-hoc descriptive analysis; "
                "no tuning, checkpoint selection, or significance testing"
            ),
            "script_sha256": digest(__file__),
            "audit_sha256": digest(ART / "audit.json"),
            "learning_rows": len(learning),
            "total_training_seconds": sum(float(row["training_seconds"]) for row in final),
            "total_training_flops_estimate": sum(
                float(row["training_flops_estimate"]) for row in final
            ),
            "final_atomic_min": min(float(row["atomic_exact"]) for row in final),
            "final_two_call_min": min(float(row["two_call_exact"]) for row in final),
            "final_training_min": min(float(row["train_exact"]) for row in final),
            "world_means": world_means,
            "paired_differences": pairs,
        },
    )
    plot(learning, flow, cfg)


def plot(learning, flow, cfg):
    os.environ.setdefault("MPLCONFIGDIR", str(ART / "mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [arch["name"] for arch in cfg["architectures"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, step in zip(axes, (cfg["primary_step"], cfg["steps"]), strict=True):
        for i, arch in enumerate(labels):
            vals = [
                100 * float(row["test_exact"])
                for row in learning
                if row["arch"] == arch and int(row["step"]) == step
            ]
            ax.bar(i, np.mean(vals), color="#377eb8" if i < 5 else "#ff7f00", alpha=0.65)
            ax.scatter(i + np.linspace(-0.12, 0.12, len(vals)), vals, s=20, color="black", zorder=3)
        ax.axhline(
            100 / cfg["entities"], color="gray", linestyle=":", label="Uniform entity guessing"
        )
        ax.set(
            title=f"{step} training steps; dots = 2 worlds x 2 seeds",
            xticks=range(len(labels)),
            xticklabels=labels,
            ylabel="Held-out exact answer + EOS (%)",
        )
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"heldout-depth-width.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for layer in range(7):
        for key, ax in (("bridge_lens_pos2", axes[0]), ("counterfactual_gain", axes[1])):
            vals = []
            for step in cfg["nodes"]:
                data = [
                    float(row[key]) * 100
                    for row in flow
                    if row["arch"] == "d6w128"
                    and row["kind"] == "bridge"
                    and int(row["layer"]) == layer
                    and int(row["step"]) == step
                ]
                vals.append(np.mean(data))
            ax.plot(range(len(cfg["nodes"])), vals, marker=".", label=f"After {layer}")
    for ax in axes:
        ax.set_xticks(range(len(cfg["nodes"])), cfg["nodes"], rotation=45)
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.2)
    axes[0].set(
        title="6 blocks: bridge readout at first-relation token",
        ylabel="Bridge logit-lens top-1 (%)",
        ylim=(-2, 102),
    )
    axes[1].set(title="Same runs: bridge-state swap", ylabel="Counterfactual answer gain (pp)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"bridge-readout-and-use.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
