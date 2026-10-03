"""Score the fixed calibrated editor on data-reserved facts and parent-known chains."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from llm_memory_editability.grok_depth import write_json

ARTIFACTS = Path("docs/development-artifacts/depth-step-v1/edit-followup")
RESULTS = Path("results/depth-step-v1/edit-followup")


def main():
    config = json.loads((ARTIFACTS / "config.json").read_text())
    states, rows = [], []
    for state in config["states"]:
        source = json.loads((RESULTS / state["name"] / "summary.json").read_text())
        arms = {}
        for arm in ("edit", "review"):
            records = [record for record in source["records"] if record["arm"] == arm]
            tasks = records[0]["history"][-1]["metrics"]
            means = {}
            for task in tasks:
                values = [record["history"][-1]["metrics"][task] for record in records]
                present = [value["accuracy"] for value in values if value["accuracy"] is not None]
                means[task] = float(np.mean(present)) if present else None
            known_n = new_n = old_n = 0
            full_n = full_new_n = 0
            for record in records:
                index = record["original_case_index"]
                raw = np.load(RESULTS / state["name"] / f"case{index:02d}-{arm}-raw.npz")
                task = "D_first_familiar_2"
                original = raw[task + "_original_rows"]
                target = raw[f"step0_{task}_rows"]
                before = raw[f"step0_{task}_predictions"]
                after = raw[f"step200_{task}_predictions"]
                known = (before[:, 0] == original[:, -1]) & (before[:, 1] == 1)
                following = (after[:, 0] == target[:, -1]) & (after[:, 1] == 1)
                unchanged = (after[:, 0] == original[:, -1]) & (after[:, 1] == 1)
                known_n += int(known.sum())
                new_n += int((following & known).sum())
                old_n += int((unchanged & known).sum())
                full_n += len(target)
                full_new_n += int(following.sum())
                for node in record["history"]:
                    for task, metric in node["metrics"].items():
                        rows.append(
                            {
                                "model": state["model"],
                                "parent_step": state["step"],
                                "edit_step": node["step"],
                                "original_case_index": index,
                                "arm": arm,
                                "task": task,
                                **metric,
                            }
                        )
            arms[arm] = {
                "case_mean_endpoint": means,
                "full_first_hop_D_queries": full_n,
                "full_first_hop_D_accuracy": full_new_n / full_n,
                "parent_original_correct_D_n": known_n,
                "parent_original_correct_D_coverage": known_n / full_n,
                "new_D_following_on_parent_correct": new_n / known_n if known_n else None,
                "old_D_retained_on_parent_correct": old_n / known_n if known_n else None,
            }
        states.append({**state, "arms": arms})
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ARTIFACTS / "learning.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        ARTIFACTS / "summary.json",
        {
            "phase": config["phase"],
            "selected_lr": config["lr"],
            "selection": config["selection"],
            "states": states,
            "limits": config["limits"],
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for model, color in (("standard2", "#2467a2"), ("loop2", "#d56a22")):
        selected = sorted(
            [state for state in states if state["model"] == model], key=lambda state: state["step"]
        )
        for task, marker in (("E_new", "o"), ("U_atomic", "s")):
            axes[0].plot(
                [state["step"] for state in selected],
                [100 * state["arms"]["edit"]["case_mean_endpoint"][task] for state in selected],
                color=color,
                marker=marker,
                label=f"{model} {task}",
            )
        for key, marker in (
            ("new_D_following_on_parent_correct", "o"),
            ("old_D_retained_on_parent_correct", "s"),
        ):
            axes[1].plot(
                [state["step"] for state in selected],
                [100 * state["arms"]["edit"][key] for state in selected],
                color=color,
                marker=marker,
                linestyle="-" if key.startswith("new") else "--",
                label=f"{model} {'new route' if key.startswith('new') else 'old route'}",
            )
    axes[0].set_title("Target update and unreplayed atomic retention")
    axes[1].set_title("Only parent-correct affected chains")
    for ax in axes:
        ax.set_xticks([8000, 32000, 64000], ["8k", "32k", "64k"])
        ax.set_ylim(-3, 103)
        ax.set_xlabel("Parent checkpoint training updates")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle("Fixed calibrated MLP editor, six data-reserved facts (one development world)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"comparison.{suffix}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "states": len(states),
                "branch_task_nodes": len(rows),
                "states_summary": [
                    {"name": state["name"], **state["arms"]["edit"]} for state in states
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
