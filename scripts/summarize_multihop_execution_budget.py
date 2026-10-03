"""Extract registered R2-to-R4 comparisons in models trained with R4.

The source scan retains every R. This table reports full entity/EOS generation,
entity answers, EOS completion, constituent retrieval and external execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_multihop_scaling as scan
import numpy as np
from report_multihop_scaling import (
    TASKS,
    file_hash,
    identity,
    mean,
    run_name,
    world_balanced,
    write_csv,
    write_json,
)

METRICS = (
    "accuracy",
    "answer_accuracy",
    "eos_rate",
    "atomic_correct_coverage",
    "conditional_accuracy",
    "autonomous_two_calls",
    "autonomous_path_accuracy",
)


def summarize(config_path, summary_path, output):
    config = scan.load_config(config_path)
    summary = json.loads(Path(summary_path).read_text())
    if not summary["complete"] or summary["audited_evaluations"] != config["evaluations"]:
        raise ValueError("The complete registered scan and main reload audits are required")
    if summary["scan_config_sha256"] != file_hash(config_path):
        raise ValueError("Scan summary belongs to another config")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    values = tuple(f"test_r{repeats}_{metric}" for repeats in (2, 4) for metric in METRICS)
    values += tuple(f"difference_{metric}_pp" for metric in METRICS)
    for spec in config["specs"]:
        if spec["repeats"] != 4:
            continue
        for node in config["nodes"]:
            observations = {}
            for repeats in (2, 4):
                folder = scan.evaluation_folder(config, spec, node, repeats)
                checkpoint = Path(config["results"]) / run_name(spec) / f"model-{node:06d}.pt"
                scan.verify_saved(folder, config, spec, node, repeats, checkpoint)
                metrics = json.loads((folder / "metrics.json").read_text())
                with np.load(folder / "predictions.npz") as predictions:
                    observations[repeats] = {
                        task: {
                            **metrics[task],
                            "eos_rate": mean(predictions[f"{task}_generated"][:, 1] == 1),
                        }
                        for task in TASKS
                    }
            for task in TASKS:
                row = {"run": run_name(spec), **identity(spec), "step": node, "task": task}
                for metric in METRICS:
                    lower = observations[2][task].get(metric)
                    trained = observations[4][task].get(metric)
                    row[f"test_r2_{metric}"], row[f"test_r4_{metric}"] = lower, trained
                    row[f"difference_{metric}_pp"] = (
                        100 * (trained - lower)
                        if lower is not None and trained is not None
                        else None
                    )
                records.append(row)
    groups = ("width", "phi", "step", "task")
    worlds, aggregate = world_balanced(records, groups, values)
    for name, rows in (
        ("execution-budget-pairs", records),
        ("execution-budget-worlds", worlds),
        ("execution-budget-aggregate", aggregate),
    ):
        write_csv(output / f"{name}.csv", rows)
    final_step = max(config["nodes"])
    primary = [
        row
        for row in aggregate
        if row["step"] == final_step and row["task"].startswith("familiar_")
    ]
    result = {
        "comparison": "Same R4-trained weights, test R4 minus test R2",
        "checkpoint_policy": "All registered 8k/32k/64k nodes; fixed 64k primary",
        "primary_pool": "Familiar facts in held-out complete queries, separately for 2/3/4 hops",
        "analysis_unit": "Paired initializations within worlds; equal-weight world means",
        "scan_config_sha256": file_hash(config_path),
        "scan_summary_sha256": file_hash(summary_path),
        "extraction_source_sha256": file_hash(Path(__file__)),
        "complete_scan_preserved": True,
        "pairs": records,
        "worlds": worlds,
        "aggregate": aggregate,
        "primary": primary,
        "interpretation": (
            "This comparison can show execution-budget gains with the same trained weights. "
            "Coverage and entity/EOS rates must accompany the gain; it does not assign one hop "
            "to one internal recurrence. Test R6 remains in the complete registered scan."
        ),
    }
    write_json(output / "execution-budget-summary.json", result)
    print(
        json.dumps(
            {
                "paired_task_nodes": len(records),
                "world_task_nodes": len(worlds),
                "primary_cells": len(primary),
            }
        )
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-config", required=True)
    parser.add_argument("--scan-summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summarize(args.scan_config, args.scan_summary, args.out)


if __name__ == "__main__":
    main()
