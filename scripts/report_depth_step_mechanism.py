"""Keep descriptive readouts, causal patches and parameter edits separate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from llm_memory_editability.grok_depth import write_json

ARTIFACTS = Path("docs/development-artifacts/depth-step-v1")
RESULTS = Path("results/depth-step-v1/mechanism")


def analyze():
    config = json.loads((ARTIFACTS / "mechanism-config.json").read_text())
    entity_rows, patch_rows, edit_rows, summaries = [], [], [], []
    for state in config["states"]:
        name = state["name"]
        trace = json.loads((RESULTS / name / "trace" / "trace-summary.json").read_text())
        edited = json.loads((RESULTS / name / "edit" / "edit-summary.json").read_text())
        raw = np.load(RESULTS / name / "trace" / "trace-raw.npz")
        for layer in range(state["layers"] * state["repeats"]):
            for position in range(raw["bridge_residual_entity_rank"].shape[-1]):
                rank = raw["bridge_residual_entity_rank"][layer, :, position]
                entity_rows.append(
                    {
                        "model": state["model"],
                        "step": state["step"],
                        "execution": layer,
                        "position": position,
                        "n": len(rank),
                        "bridge_mrr": float((1 / rank).mean()),
                        "bridge_top1": float((rank == 1).mean()),
                        "bridge_mean_mlp_cosine": float(
                            raw["bridge_mlp_embedding_cosine"][layer, :, position].mean()
                        ),
                        "bridge_mean_mlp_cosine_rank": float(
                            raw["bridge_mlp_embedding_cosine_rank"][layer, :, position].mean()
                        ),
                    }
                )
        for record in trace["interventions"]:
            patch_rows.append({"model": state["model"], "step": state["step"], **record})
        tasks = list(edited["records"][0]["history"][0]["metrics"])
        for record in edited["records"]:
            for node in record["history"]:
                for task in tasks:
                    edit_rows.append(
                        {
                            "model": state["model"],
                            "checkpoint_step": state["step"],
                            "edit_step": node["step"],
                            "arm": record["arm"],
                            "atomic_index": record["atomic_index"],
                            "task": task,
                            **node["metrics"][task],
                        }
                    )
        endpoint = {}
        for arm in ("edit", "review"):
            values = [record for record in edited["records"] if record["arm"] == arm]
            endpoint[arm] = {}
            for task in tasks:
                stats = [record["history"][-1]["metrics"][task] for record in values]
                present = [item["accuracy"] for item in stats if item["accuracy"] is not None]
                denominator = sum(item["n"] for item in stats)
                endpoint[arm][task] = {
                    "case_mean_accuracy": float(np.mean(present)) if present else None,
                    "query_weighted_accuracy": sum(
                        item["n"] * (item["accuracy"] or 0) for item in stats
                    )
                    / denominator
                    if denominator
                    else None,
                    "query_instances": denominator,
                    "nonempty_cases": len(present),
                }
        summaries.append(
            {
                **state,
                "baseline": trace["baseline"],
                "forward_max_difference": trace["forward_max_absolute_difference"],
                "edit_endpoint": endpoint,
            }
        )
    for filename, values in (
        ("entity-readouts", entity_rows),
        ("patches", patch_rows),
        ("edits", edit_rows),
    ):
        keys = list(dict.fromkeys(key for item in values for key in item))
        with (ARTIFACTS / f"mechanism-{filename}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(values)
    write_json(
        ARTIFACTS / "mechanism-summary.json",
        {
            "phase": "single-world development",
            "states": summaries,
            "limits": [
                "Readouts and attention maps are descriptive",
                "Query instances from different edits can repeat",
                "MLP scope differs for independent and shared models",
                "Report all target edit failures and U drift",
            ],
        },
    )
    plot(entity_rows, patch_rows, summaries, config)
    print(
        json.dumps(
            {
                "model_states": len(summaries),
                "entity_cells": len(entity_rows),
                "patch_cells": len(patch_rows),
                "edit_task_nodes": len(edit_rows),
            }
        )
    )


def plot(entity_rows, patch_rows, summaries, config):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for model, color in (("standard2", "#2467a2"), ("loop2", "#d56a22")):
        for execution, marker in ((0, "o"), (1, "s")):
            rows = [
                row
                for row in entity_rows
                if row["model"] == model and row["execution"] == execution and row["position"] == 2
            ]
            rows.sort(key=lambda item: item["step"])
            axes[0].plot(
                [row["step"] for row in rows],
                [row["bridge_mrr"] for row in rows],
                marker=marker,
                color=color,
                linestyle="-" if execution == 0 else "--",
                label=f"{model} block {execution + 1}",
            )
        for component, marker in (("mlp_delta", "o"), ("postresidual", "s")):
            rows = [
                row
                for row in patch_rows
                if row["model"] == model
                and row["execution"] == 0
                and row["position"] == 2
                and row["donor"] == "different_bridge"
                and row["component"] == component
            ]
            rows.sort(key=lambda item: item["step"])
            axes[1].plot(
                [row["step"] for row in rows],
                [100 * row["new_route_accuracy"] for row in rows],
                marker=marker,
                color=color,
                linestyle="-" if component == "mlp_delta" else "--",
                label=f"{model} {component}",
            )
        rows = [item for item in summaries if item["model"] == model]
        rows.sort(key=lambda item: item["step"])
        for task, marker in (("E_new", "o"), ("D_first_familiar_2", "s"), ("U_atomic", "^")):
            axes[2].plot(
                [item["step"] for item in rows],
                [100 * item["edit_endpoint"]["edit"][task]["case_mean_accuracy"] for item in rows],
                marker=marker,
                color=color,
                label=f"{model} {task}",
            )
    axes[0].set_title("Bridge direct readout at r1")
    axes[0].set_ylabel("Mean reciprocal entity rank")
    axes[1].set_title("Alternative-route patch, first block")
    axes[1].set_ylabel("New answer + EOS (%)")
    axes[2].set_title("MLP edit after 200 updates")
    axes[2].set_ylabel("Case mean accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Parent checkpoint training updates")
        ax.set_xticks([8000, 32000, 64000], ["8k", "32k", "64k"])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"mechanism.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    roles = ["BOS", "h", "r1", "r2", "SEP"]
    for row, model in enumerate(("standard2", "loop2")):
        for col, step in enumerate((8000, 32000, 64000)):
            state = next(
                item for item in config["states"] if item["model"] == model and item["step"] == step
            )
            raw = np.load(RESULTS / state["name"] / "trace" / "trace-raw.npz")
            attention = raw["cache_attention_map"][0].mean(axis=(0, 1))
            axes[row, col].imshow(attention, vmin=0, vmax=1, cmap="viridis")
            axes[row, col].set_title(f"{model} {step // 1000}k | first block")
            axes[row, col].set_xticks(range(5), roles)
            axes[row, col].set_yticks(range(5), roles)
    fig.suptitle("Mean causal attention over fixed familiar queries and heads (descriptive)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"attention.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
