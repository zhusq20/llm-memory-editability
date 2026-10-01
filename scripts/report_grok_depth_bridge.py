#!/usr/bin/env python3
"""Aggregate fixed bridge interventions with worlds, not queries, as units."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def aggregate(records):
    worlds = defaultdict(list)
    for record in records:
        key = tuple(record[k] for k in ("phase", "arm", "step", "group", "condition", "world"))
        worlds[key].append(record)
    world_records = []
    identifiers = {
        "phase",
        "arm",
        "step",
        "group",
        "condition",
        "world",
        "initialization",
        "run_id",
    }
    for key, repeats in sorted(worlds.items()):
        result = dict(
            zip(("phase", "arm", "step", "group", "condition", "world"), key, strict=True)
        )
        result["initializations"] = sorted(r["initialization"] for r in repeats)
        for metric in sorted(repeats[0].keys() - identifiers):
            values = [r[metric] for r in repeats if r[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        world_records.append(result)
    groups = defaultdict(list)
    for record in world_records:
        key = tuple(record[k] for k in ("phase", "arm", "step", "group", "condition"))
        groups[key].append(record)
    grouped = []
    for key, members in sorted(groups.items()):
        result = dict(zip(("phase", "arm", "step", "group", "condition"), key, strict=True))
        result["worlds"] = [r["world"] for r in members]
        result["metrics"] = {}
        for metric in sorted(members[0].keys() - identifiers - {"initializations"}):
            values = [r[metric] for r in members if r[metric] is not None]
            result["metrics"][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "min": float(min(values)) if values else None,
                "max": float(max(values)) if values else None,
                "n_worlds": len(values),
            }
        grouped.append(result)
    return world_records, grouped


def plot(groups, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = {
        (r["arm"], r["step"], r["group"], r["condition"]): r
        for r in groups
        if r["phase"] == "confirmation"
    }

    def get(arm, step, condition, metric, group="different_donor_available"):
        return index[arm, step, group, condition]["metrics"][metric]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    arms = ["d1-w128", "d1-w180", "d2-w128"]
    colors = ["#667085", "#D99033", "#247CA8"]
    conditions = ["different_bridge_r1", "different_bridge_h", "different_bridge_h_r1"]
    x = np.arange(len(conditions))
    for offset, (arm, color) in enumerate(zip(arms, colors, strict=True)):
        stats = [get(arm, 128000, c, "cf_complete_accuracy") for c in conditions]
        y = np.array([s["mean"] * 100 for s in stats])
        err = (
            np.array([[s["mean"] - s["min"] for s in stats], [s["max"] - s["mean"] for s in stats]])
            * 100
        )
        axes[0].bar(x + (offset - 1) * 0.24, y, 0.23, color=color, label=arm, yerr=err, capsize=3)
    axes[0].set_xticks(x, ["Replace r1", "Replace h", "Replace h + r1"])
    axes[0].set_ylabel("Counterfactual answer + EOS (%)")
    axes[0].set_title("Frozen endpoints: prefix-state replacement")
    axes[0].legend(frameon=False)
    steps = [4000, 32000, 64000, 128000]
    for condition, metric, label, color in [
        ("baseline", "original_complete_accuracy", "Original question", "#667085"),
        ("counterfactual_input", "cf_complete_accuracy", "Full counterfactual question", "#D99033"),
        ("different_bridge_r1", "cf_complete_accuracy", "Replace r1: new answer", "#247CA8"),
        ("different_bridge_h", "cf_complete_accuracy", "Replace h: new answer", "#A95385"),
    ]:
        stats = [get("d2-w128", step, condition, metric) for step in steps]
        axes[1].plot(
            np.array(steps) / 1000,
            [s["mean"] * 100 for s in stats],
            "o-",
            label=label,
            color=color,
            markersize=4,
        )
        axes[1].fill_between(
            np.array(steps) / 1000,
            [s["min"] * 100 for s in stats],
            [s["max"] * 100 for s in stats],
            color=color,
            alpha=0.12,
        )
    axes[1].set_xlabel("Historical training steps (thousands)")
    axes[1].set_ylabel("Answer + EOS accuracy (%)")
    axes[1].set_title("Two layers: fixed checkpoint follow-up")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes:
        axis.set_ylim(0, 105)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    fig.suptitle(
        "3 previously studied worlds; 2 initializations per world\n"
        "Bars/lines: world means; whiskers/bands: world range",
        fontsize=10,
    )
    fig.savefig(output / "bridge-intervention.png", dpi=180)
    fig.savefig(output / "bridge-intervention.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-depth-bridge-v1.json")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    output = ROOT / "docs/development-artifacts/grok-depth-bridge-v1"
    output.mkdir(parents=True, exist_ok=True)
    records, provenance = [], []
    for phase in ("development", "confirmation"):
        for run in config[phase + "_runs"]:
            for step in run["steps"]:
                path = (
                    ROOT
                    / config["output_root"]
                    / phase
                    / run["run_id"]
                    / f"step-{step:07d}"
                    / "summary.json"
                )
                summary = read_json(path)
                assert summary["step"] == step
                assert summary["state"] == "complete"
                spec = summary["spec"]
                common = {
                    "phase": phase,
                    "run_id": run["run_id"],
                    "step": step,
                    "world": spec["world_seed"],
                    "initialization": spec["initialization"],
                    "arm": f"d{spec['layers']}-w{spec['width']}",
                }
                provenance.append(
                    {
                        **common,
                        "summary": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "donor_coverage": summary["donor_coverage"],
                        "engineering_checks": summary["engineering_checks"],
                        "atomic_preconditions": summary["atomic_preconditions"],
                        "evaluation_seconds": summary["evaluation_seconds"],
                        "wall_seconds": summary["wall_seconds"],
                    }
                )
                for group, group_scores in summary["scores"].items():
                    for condition, metrics in group_scores["conditions"].items():
                        records.append(
                            {**common, "group": group, "condition": condition, **metrics}
                        )
                for condition, metrics in summary["atomic_preconditions"].items():
                    records.append(
                        {
                            **common,
                            "group": "atomic_preconditions",
                            "condition": condition,
                            **metrics,
                        }
                    )
    world_records, groups = aggregate(records)
    summary = {
        "experiment": config["experiment"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "statistical_unit": "World; mean over initializations within world, then equal world mean.",
        "scope": "Predefined mechanistic follow-up on previously studied worlds; no new training.",
        "model_states": len(provenance),
        "evaluation_seconds": sum(r["evaluation_seconds"] for r in provenance),
        "state_wall_seconds_sum": sum(r["wall_seconds"] for r in provenance),
        "new_training_steps": 0,
        "groups": groups,
        "world_records": world_records,
        "runs": provenance,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, rows in (("state-results.csv", records), ("world-results.csv", world_records)):
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with (output / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    plot(groups, output)
    print(
        json.dumps({"status": "complete", "model_states": len(provenance), "output": str(output)})
    )


if __name__ == "__main__":
    main()
