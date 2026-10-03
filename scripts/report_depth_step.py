"""Report every frozen depth/time cell, checking generated tokens independently."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.storage_composition import file_hash


def generated_score(rows, generated):
    if generated.shape != (len(rows), 2):
        raise ValueError("Expected greedy answer and EOS for every query")
    correct = (generated[:, 0] == rows[:, -1]) & (generated[:, 1] == 1)
    return float(correct.mean()) if len(rows) else None, correct


def analyze(config_path, result_path, output):
    from llm_memory_editability.depth_step import run_name

    config_path, result_path, output = map(Path, (config_path, result_path, output))
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    records, curves, raw_checks = [], [], []
    tasks = ["atomic"] + [
        f"{pool}_{hop}" for hop in (2, 3, 4) for pool in ("train", "familiar", "strict")
    ]
    supervised_tokens = summed_training_seconds = summed_flops = 0
    for spec in config["specs"]:
        folder = result_path / run_name(spec)
        complete = json.loads((folder / "complete.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        if complete["spec"] != spec or complete["source"] != config["source"]:
            raise ValueError(f"Run does not match frozen specification: {folder}")
        if not audit["passed"]:
            raise ValueError(f"Run did not pass reload audit: {folder}")
        history = json.loads((folder / "learning.json").read_text())
        if [row["step"] for row in history] != spec["nodes"]:
            raise ValueError(f"Incomplete learning trajectory: {folder}")
        world = np.load(folder / "world.npz")
        for row in history:
            predictions = np.load(folder / f"predictions-{row['step']:06d}.npz")
            for task in tasks:
                measured, correct = generated_score(world[task], predictions[f"{task}_generated"])
                if measured != row["metrics"][task]["accuracy"]:
                    raise ValueError(
                        f"Saved score differs from raw tokens: {folder} {row['step']} {task}"
                    )
                np.testing.assert_array_equal(correct, predictions[f"{task}_correct"])
                curves.append(
                    {
                        "run": folder.name,
                        "layers": spec["layers"],
                        "repeats": spec["repeats"],
                        "step": row["step"],
                        "task": task,
                        **row["metrics"][task],
                        "supervised_tokens": row["supervised_tokens"],
                        "estimated_training_flops": row["estimated_training_flops"],
                    }
                )
            raw_checks.append(
                {"run": folder.name, "step": row["step"], "tasks": len(tasks), "passed": True}
            )
        endpoint = history[-1]
        record = {
            "run": folder.name,
            "layers": spec["layers"],
            "repeats": spec["repeats"],
            "executed_depth": spec["layers"] * spec["repeats"],
            "parameters": complete["parameters"],
            "steps": spec["steps"],
            "atomic_accuracy": endpoint["metrics"]["atomic"]["accuracy"],
            "supervised_tokens": endpoint["supervised_tokens"],
            "estimated_training_flops": endpoint["estimated_training_flops"],
            "training_seconds": endpoint["training_seconds"],
        }
        for hop in (2, 3, 4):
            for pool in ("train", "familiar", "strict"):
                task = f"{pool}_{hop}"
                for key, value in endpoint["metrics"][task].items():
                    record[f"{task}_{key}"] = value
            for task, threshold in (
                ("atomic", 0.99),
                (f"train_{hop}", 0.99),
                (f"familiar_{hop}", 0.5),
                (f"familiar_{hop}", 0.9),
            ):
                first = next(
                    (
                        row["step"]
                        for row in history
                        if row["metrics"][task]["accuracy"] is not None
                        and row["metrics"][task]["accuracy"] >= threshold
                    ),
                    None,
                )
                record[f"first_{task}_ge{int(threshold * 100)}"] = first
        records.append(record)
        supervised_tokens += endpoint["supervised_tokens"]
        summed_training_seconds += endpoint["training_seconds"]
        summed_flops += endpoint["estimated_training_flops"]
    for name, values in (("endpoints", records), ("learning", curves)):
        columns = list(dict.fromkeys(key for row in values for key in row))
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
    summary = {
        "phase": config["phase"],
        "analysis_unit": config["analysis_unit"],
        "config_sha256": file_hash(config_path),
        "runs": len(records),
        "nodes": len(raw_checks),
        "raw_token_checks": raw_checks,
        "supervised_tokens": supervised_tokens,
        "summed_training_seconds": summed_training_seconds,
        "estimated_training_flops": summed_flops,
        "endpoints": records,
        "limitations": [
            "One development world and one initialization, without new-world confirmation",
            "Joint k=2/3/4 training; this is not unseen-hop-length transfer",
            "Familiar query holdout permits higher-hop training to contain shorter subpaths",
            "Independent-depth changes parameters; sharing changes correlations and edited scope",
            "Random-function graph permits repeated entities and suffix concentration",
        ],
    }
    write_json(output / "summary.json", summary)
    plot(records, curves, output)
    print(
        json.dumps(
            {
                "runs": len(records),
                "nodes": len(raw_checks),
                "raw_token_checks_passed": True,
                "supervised_tokens": supervised_tokens,
            }
        )
    )


def plot(records, curves, output):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for col, hop in enumerate((2, 3, 4)):
        for row, pool in enumerate(("familiar", "strict")):
            ax = axes[row, col]
            for shared, label, color in (
                (False, "Independent blocks", "#2467a2"),
                (True, "Shared block", "#d56a22"),
            ):
                cells = [
                    record
                    for record in records
                    if (record["layers"] == 1 and record["repeats"] > 1) == shared
                ]
                if shared:
                    cells += [
                        record
                        for record in records
                        if (record["layers"], record["repeats"]) == (1, 1)
                    ]
                cells.sort(key=lambda item: item["executed_depth"])
                ax.plot(
                    [cell["executed_depth"] for cell in cells],
                    [100 * cell[f"{pool}_{hop}_accuracy"] for cell in cells],
                    "o-",
                    color=color,
                    label=label,
                )
            ax.set_title(f"{hop}-hop | {pool}")
            ax.set_ylabel("Answer + EOS accuracy (%)")
            ax.grid(alpha=0.2)
            ax.set_xticks([1, 2, 3, 4, 6])
            if row == 0:
                ax.set_ylim(-3, 103)
            if row == 1:
                ax.set_xlabel("Executed blocks (L × R)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed 64k endpoint | one development world, joint hop training")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"depth-comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for row, shared in enumerate((False, True)):
        for col, hop in enumerate((2, 3, 4)):
            ax = axes[row, col]
            for record in records:
                is_shared = record["layers"] == 1 and record["repeats"] > 1
                if is_shared != shared and (record["layers"], record["repeats"]) != (1, 1):
                    continue
                line = [
                    item
                    for item in curves
                    if item["run"] == record["run"] and item["task"] == f"familiar_{hop}"
                ]
                label = f"{'R' if shared else 'L'}{record['executed_depth']}"
                ax.plot(
                    [item["step"] for item in line],
                    [100 * item["accuracy"] for item in line],
                    label=label,
                )
            ax.set_xscale("symlog", linthresh=256)
            ax.set_ylim(-3, 103)
            ax.set_title(f"{'Shared' if shared else 'Independent'} | {hop}-hop familiar")
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("Answer + EOS accuracy (%)")
            if row == 1:
                ax.set_xlabel("Training updates on the same 64k trajectory")
            if col == 2:
                ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"learning.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/depth-step-development-v1.json")
    parser.add_argument("--results", default="results/depth-step-v1")
    parser.add_argument("--out", default="docs/development-artifacts/depth-step-v1")
    args = parser.parse_args()
    analyze(args.config, args.results, args.out)


if __name__ == "__main__":
    main()
