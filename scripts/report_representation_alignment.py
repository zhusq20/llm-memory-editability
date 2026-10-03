"""Recount complete generation, verify pairing, and plot all intervention arms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json

ROOT = Path("results/representation-alignment-v1")
ARTIFACTS = Path("docs/development-artifacts/representation-alignment-v1")
TASKS = ("common_atomic", "train_composite", "familiar_test", "strict_test")


def recount(path, spec):
    world = np.load(path / "world.npz")
    pred = np.load(path / f"predictions-{spec['steps']:06d}.npz")
    metrics = {}
    for task in TASKS:
        generated = pred[task + "_generated"]
        target = world[task][:, -1]
        correct = (generated[:, 0] == target) & (generated[:, 1] == 5) & (generated[:, 2] == 1)
        np.testing.assert_array_equal(correct, pred[task + "_correct"])
        metrics[task] = float(correct.mean())
    return metrics


def report(phase):
    rows = []
    for path in sorted((ROOT / "runs").glob(phase + "-w*")):
        if not (path / "complete.json").exists():
            continue
        payload = json.loads((path / "complete.json").read_text())
        spec = payload["spec"]
        assert (path / "audit.json").exists(), f"Missing independent reload: {path}"
        scores = recount(path, spec)
        for task, score in scores.items():
            assert score == payload["metrics"][task]["accuracy"]
        rows.append(
            {
                "run": path.name,
                "world": spec["world"],
                "initialization": spec["initialization"],
                "arm": spec["arm"],
                "steps": spec["steps"],
                **scores,
                "training_seconds": payload["training_seconds"],
                "initial_model_sha256": payload["initial_model_sha256"],
                "sample_stream_sha256": payload["sample_stream_sha256"],
                "world_sha256": payload["world_sha256"],
            }
        )
    if not rows:
        raise ValueError("No completed, reloaded runs")
    paired = {}
    for row in rows:
        paired.setdefault((row["world"], row["initialization"]), []).append(row)
    for block in paired.values():
        for field in ("initial_model_sha256", "sample_stream_sha256", "world_sha256"):
            assert len({row[field] for row in block}) == 1, f"Pair mismatch: {field}"
        ref = np.load(ROOT / "runs" / block[0]["run"] / "exposures.npz")
        for row in block[1:]:
            other = np.load(ROOT / "runs" / row["run"] / "exposures.npz")
            for key in ref.files:
                np.testing.assert_array_equal(ref[key], other[key])
    arms = sorted({row["arm"] for row in rows})
    world_means = {}
    for world in sorted({row["world"] for row in rows}):
        world_means[str(world)] = {
            arm: {
                task: float(
                    np.mean(
                        [row[task] for row in rows if row["world"] == world and row["arm"] == arm]
                    )
                )
                for task in TASKS
            }
            for arm in arms
        }
    means = {
        arm: {
            task: float(np.mean([world_means[w][arm][task] for w in world_means])) for task in TASKS
        }
        for arm in arms
    }
    contrasts = []
    for (world, initialization), block in paired.items():
        lookup = {row["arm"]: row for row in block}
        for arm in arms:
            if arm == "baseline":
                continue
            for reference in ("baseline", "bridge_ce"):
                if reference == arm:
                    continue
                contrasts.append(
                    {
                        "world": world,
                        "initialization": initialization,
                        "arm": arm,
                        "reference": reference,
                        **{
                            task: 100 * (lookup[arm][task] - lookup[reference][task])
                            for task in TASKS
                        },
                    }
                )
    target = ARTIFACTS / phase
    target.mkdir(parents=True, exist_ok=True)
    for name, records in (("endpoints.csv", rows), ("contrasts.csv", contrasts)):
        with (target / name).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = {
        "phase": phase,
        "runs": len(rows),
        "worlds": len(world_means),
        "means": means,
        "per_world": world_means,
        "contrasts_pp": contrasts,
        "pairing_and_independent_generation_recount": "passed",
    }
    write_json(target / "summary.json", summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {"baseline": "#64748b", "bridge_ce": "#d97706"}
    for arm in arms:
        curves = [
            json.loads((ROOT / "runs" / row["run"] / "learning.json").read_text())
            for row in rows
            if row["arm"] == arm
        ]
        nodes = [point["step"] for point in curves[0]]
        for ax, task in zip(axes, ("familiar_test", "strict_test"), strict=True):
            scores = np.array(
                [[100 * point["metrics"][task]["accuracy"] for point in curve] for curve in curves]
            )
            ax.plot(nodes, scores.mean(0), marker="o", ms=3, label=arm, color=colors.get(arm))
            if len(curves) > 1:
                ax.fill_between(
                    nodes, scores.min(0), scores.max(0), alpha=0.12, color=colors.get(arm)
                )
            ax.set(
                xlabel="Optimizer updates",
                ylabel="Full generation accuracy (%)",
                title="Familiar facts, held-out combinations"
                if task == "familiar_test"
                else "Facts with atomic training only",
                ylim=(0, 100),
            )
            ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(target / "learning.png", dpi=180)
    fig.savefig(target / "learning.pdf")
    plt.close(fig)
    print(json.dumps({"runs": len(rows), "means": means, "report": str(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    report(parser.parse_args().phase)
