#!/usr/bin/env python3
"""Summarize and independently audit the paired 3/4-layer extension."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.grok_depth_extension import (  # noqa: E402
    digest,
    read_json,
    threshold_time,
    verify_files,
)

ARTIFACT = ROOT / "docs/development-artifacts/grok-depth-extension-v1"
RESULTS = ROOT / "results/grok-depth-extension-v1/extension"
CONFIG = ROOT / "configs/grok-depth-extension-v1.json"
SPLITS = ("atomic", "train_composite", "test_composite", "ood_composite", "two_calls")
THRESHOLDS = ("paired_final", "t90", "t95")


def write(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def mean(values):
    return float(np.mean(values)) if values and all(x is not None for x in values) else None


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load():
    cfg, lock = read_json(CONFIG), read_json(ARTIFACT / "execution-lock.json")
    verify_files(ROOT, lock["files"])
    for ref in lock["references"].values():
        verify_files(ROOT, ref["files"])
    runs = []
    selections = [
        (baseline_id, ROOT / ref["directory"], baseline_id, True)
        for baseline_id, ref in lock["references"].items()
    ] + [
        (run_id, RESULTS / run_id, cfg["baseline_for_run"][run_id], False) for run_id in cfg["runs"]
    ]
    for run_id, out, baseline_id, reused in selections:
        meta_path = out / "metadata.json"
        if not meta_path.exists():
            continue
        metadata = read_json(meta_path)
        spec = metadata["spec"]
        rows = read_json(out / "learning.json") if (out / "learning.json").exists() else []
        assert [row["step"] for row in rows] == spec["nodes"][: len(rows)]
        counts = read_json(out / "data-audit.json")["counts"]
        ref = lock["references"][baseline_id]
        complete = (out / "complete.json").exists()
        if complete:
            assert len(rows) == len(spec["nodes"])
            assert read_json(out / "complete.json")["endpoint"] == rows[-1]
        endpoint = rows[-1] if complete else None
        targets = {"paired_final": ref["paired_final_threshold"], "t90": 0.9, "t95": 0.95}
        runs.append(
            {
                "run_id": run_id,
                "baseline_id": baseline_id,
                "reused_baseline": reused,
                "directory": str(out.relative_to(ROOT)),
                "spec": spec,
                "data_counts": counts,
                "status": "complete" if complete else "incomplete",
                "thresholds": {
                    key: threshold_time(rows, value, counts, spec["steps"])
                    for key, value in targets.items()
                },
                "fixed_budget_endpoint": endpoint,
                "learning": rows,
            }
        )
    return cfg, lock, runs


def aggregate(runs):
    result = []
    for depth in (2, 3, 4):
        members = [r for r in runs if r["spec"]["layers"] == depth]
        worlds = defaultdict(list)
        for row in members:
            worlds[row["spec"]["world_seed"]].append(row)
        world_rows = []
        for world, items in sorted(worlds.items()):
            completed = len(items) == 2 and all(r["status"] == "complete" for r in items)
            world_rows.append(
                {
                    "world_seed": world,
                    "completed_replicates": sum(r["status"] == "complete" for r in items),
                    "endpoint_accuracy": {
                        split: mean([r["fixed_budget_endpoint"][split]["accuracy"] for r in items])
                        if completed
                        else None
                        for split in SPLITS
                    },
                    "thresholds": {
                        key: {
                            "reached_replicates": sum(
                                r["thresholds"][key]["reached"] for r in items
                            ),
                            "mean_step": mean([r["thresholds"][key]["step"] for r in items])
                            if len(items) == 2
                            else None,
                            "mean_atomic_exposures": mean(
                                [r["thresholds"][key]["atomic_exposures"] for r in items]
                            )
                            if len(items) == 2
                            else None,
                            "mean_flops": mean(
                                [r["thresholds"][key]["estimated_training_flops"] for r in items]
                            )
                            if len(items) == 2
                            else None,
                        }
                        for key in THRESHOLDS
                    },
                }
            )
        all_complete = len(members) == 6 and all(r["status"] == "complete" for r in members)
        endpoints = {
            split: mean([w["endpoint_accuracy"][split] for w in world_rows])
            if all_complete
            else None
            for split in SPLITS
        }
        result.append(
            {
                "layers": depth,
                "registered_runs": 6,
                "observed_runs": len(members),
                "completed_runs": sum(r["status"] == "complete" for r in members),
                "parameters": members[0]["learning"][0]["parameters"]
                if members and members[0]["learning"]
                else None,
                "worlds": world_rows,
                "world_mean_endpoint_accuracy": endpoints,
                "world_endpoint_range": {
                    split: [
                        min(w["endpoint_accuracy"][split] for w in world_rows),
                        max(w["endpoint_accuracy"][split] for w in world_rows),
                    ]
                    if all_complete
                    else None
                    for split in SPLITS
                },
                "thresholds": {
                    key: {
                        "reached_runs": sum(r["thresholds"][key]["reached"] for r in members),
                        "not_reached_or_pending_runs": 6
                        - sum(r["thresholds"][key]["reached"] for r in members),
                        "world_mean_step": mean(
                            [w["thresholds"][key]["mean_step"] for w in world_rows]
                        )
                        if len(world_rows) == 3
                        else None,
                        "world_mean_atomic_exposures": mean(
                            [w["thresholds"][key]["mean_atomic_exposures"] for w in world_rows]
                        )
                        if len(world_rows) == 3
                        else None,
                        "world_mean_flops": mean(
                            [w["thresholds"][key]["mean_flops"] for w in world_rows]
                        )
                        if len(world_rows) == 3
                        else None,
                        "observed_steps": [r["thresholds"][key]["step"] for r in members],
                    }
                    for key in THRESHOLDS
                },
            }
        )
    return result


def comparisons(runs):
    indexed = {row["run_id"]: row for row in runs}
    result = []
    for row in runs:
        if row["reused_baseline"]:
            continue
        baseline = indexed[row["baseline_id"]]
        base_end = baseline["fixed_budget_endpoint"]
        equal_nodes = [
            n
            for n in row["learning"]
            if n["estimated_training_flops"] <= base_end["estimated_training_flops"]
        ]
        equal = equal_nodes[-1] if equal_nodes else None
        result.append(
            {
                "run_id": row["run_id"],
                "baseline_id": row["baseline_id"],
                "layers": row["spec"]["layers"],
                "world_seed": row["spec"]["world_seed"],
                "initialization": row["spec"]["initialization"],
                "endpoint_accuracy_difference": row["fixed_budget_endpoint"]["test_composite"][
                    "accuracy"
                ]
                - base_end["test_composite"]["accuracy"]
                if row["status"] == "complete"
                else None,
                "threshold_comparisons": {
                    key: {
                        "both_reached": row["thresholds"][key]["reached"]
                        and baseline["thresholds"][key]["reached"],
                        "new_step": row["thresholds"][key]["step"],
                        "baseline_step": baseline["thresholds"][key]["step"],
                        "new_over_baseline_step_ratio": row["thresholds"][key]["step"]
                        / baseline["thresholds"][key]["step"]
                        if row["thresholds"][key]["reached"]
                        and baseline["thresholds"][key]["reached"]
                        and baseline["thresholds"][key]["step"]
                        else None,
                        "new_over_baseline_flops_ratio": row["thresholds"][key][
                            "estimated_training_flops"
                        ]
                        / baseline["thresholds"][key]["estimated_training_flops"]
                        if row["thresholds"][key]["reached"]
                        and baseline["thresholds"][key]["reached"]
                        and baseline["thresholds"][key]["estimated_training_flops"]
                        else None,
                    }
                    for key in THRESHOLDS
                },
                "paired_target_fraction_of_baseline_fixed_128k_budget": {
                    "steps": row["thresholds"]["paired_final"]["step"] / base_end["step"]
                    if row["thresholds"]["paired_final"]["reached"]
                    else None,
                    "flops": row["thresholds"]["paired_final"]["estimated_training_flops"]
                    / base_end["estimated_training_flops"]
                    if row["thresholds"]["paired_final"]["reached"]
                    else None,
                    "not_an_exact_learning_speed_ratio": True,
                },
                "equal_compute_at_baseline_full_budget": {
                    "cap": base_end["estimated_training_flops"],
                    "new_step": equal["step"] if equal else None,
                    "new_flops": equal["estimated_training_flops"] if equal else None,
                    "new_accuracy": equal["test_composite"]["accuracy"] if equal else None,
                    "baseline_step": base_end["step"],
                    "baseline_accuracy": base_end["test_composite"]["accuracy"],
                    "fixed_budget_complete": row["status"] == "complete",
                },
            }
        )
    return result


def plots(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {2: "#2563eb", 3: "#e67e22", 4: "#15803d"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for depth in (2, 3, 4):
        members = [r for r in runs if r["spec"]["layers"] == depth and r["learning"]]
        if len(members) != 6:
            continue
        worlds = sorted({r["spec"]["world_seed"] for r in members})
        n = min(len(r["learning"]) for r in members)
        averages = []
        for world in worlds:
            group = [r for r in members if r["spec"]["world_seed"] == world]
            ys = np.mean(
                [
                    [node["test_composite"]["accuracy"] * 100 for node in r["learning"][:n]]
                    for r in group
                ],
                axis=0,
            )
            averages.append(ys)
            for axis, key, scale in zip(
                axes, ("step", "estimated_training_flops"), (1000, 1e14), strict=True
            ):
                xs = [node[key] / scale for node in group[0]["learning"][:n]]
                axis.plot(xs, ys, color=colors[depth], alpha=0.20, linewidth=1)
        for axis, key, scale in zip(
            axes, ("step", "estimated_training_flops"), (1000, 1e14), strict=True
        ):
            xs = [node[key] / scale for node in members[0]["learning"][:n]]
            axis.plot(
                xs,
                np.mean(averages, axis=0),
                color=colors[depth],
                linewidth=2.5,
                label=f"{depth} layers" + (" (reused)" if depth == 2 else ""),
            )
    for ax in axes:
        ax.set_ylim(0, 101)
        ax.set_ylabel("Held-out two-hop answer + EOS accuracy (%)")
        ax.axhline(90, color="gray", linewidth=0.7, linestyle="--")
        ax.axhline(95, color="gray", linewidth=0.7, linestyle=":")
        ax.grid(alpha=0.15)
        ax.legend(loc="lower right")
    axes[0].set_xlabel("Training updates (thousands)")
    axes[1].set_xlabel("Estimated training FLOPs (1e14)")
    fig.suptitle("Paired depth extension: 3 worlds, 2 initializations per world")
    for suffix in ("png", "pdf"):
        fig.savefig(ARTIFACT / f"learning-depth-extension.{suffix}", dpi=180)
    plt.close(fig)


def report():
    cfg, lock, runs = load()
    summary = {
        "experiment": cfg["experiment"],
        "generated_utc": now(),
        "scope": lock["scope"],
        "expected_new_runs": 12,
        "completed_new_runs": sum(
            r["status"] == "complete" and not r["reused_baseline"] for r in runs
        ),
        "reused_baselines": 6,
        "threshold_definition": cfg["thresholds"],
        "statistical_unit": "3 worlds; two initializations within each world; reused worlds",
        "censoring_rule": (
            "World means require both replicate hits; overall means require all three worlds. "
            "Null is not infinity."
        ),
        "world_aggregates": aggregate(runs),
        "paired_comparisons": comparisons(runs),
        "runs": runs,
        "report_source_sha256": digest(Path(__file__)),
    }
    write(ARTIFACT / "summary.json", summary)
    learning_rows, threshold_rows, endpoint_rows = [], [], []
    for run in runs:
        common = {
            "run_id": run["run_id"],
            "reused_baseline": run["reused_baseline"],
            "world_seed": run["spec"]["world_seed"],
            "initialization": run["spec"]["initialization"],
            "layers": run["spec"]["layers"],
        }
        for key, result in run["thresholds"].items():
            threshold_rows.append({**common, "threshold_name": key, **result})
        for row in run["learning"]:
            learning_rows.append(
                {
                    **common,
                    "step": row["step"],
                    "parameters": row["parameters"],
                    "examples": row["examples"],
                    "atomic_exposures": row["counts"]["atomic"] / run["data_counts"]["atomic"],
                    "composite_exposures": row["counts"]["composite"]
                    / run["data_counts"]["train_composite"],
                    "estimated_training_flops": row["estimated_training_flops"],
                    **{split + "_accuracy": row[split]["accuracy"] for split in SPLITS},
                }
            )
        if run["fixed_budget_endpoint"]:
            endpoint_rows.append(learning_rows[-1])
    csv_write(ARTIFACT / "learning.csv", learning_rows)
    csv_write(ARTIFACT / "threshold-times.csv", threshold_rows)
    csv_write(ARTIFACT / "endpoint-table.csv", endpoint_rows)
    plots(runs)
    print(
        json.dumps({"completed_new_runs": summary["completed_new_runs"], "artifact": str(ARTIFACT)})
    )
    return summary


def audit():
    import audit_grok_depth as original
    import torch

    cfg, lock, runs = load()
    assert len(runs) == 18 and all(r["status"] == "complete" for r in runs)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    indexed = {row["run_id"]: row for row in runs}
    result = {
        "started_utc": now(),
        "scope": (
            "Twelve new endpoints reloaded from exact source snapshots; "
            "all saved node scores recomputed"
        ),
        "device": "cuda:0",
        "nll_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        "discrete_tolerance": "exact",
        "runs": [],
        "auditor_sha256": digest(Path(__file__)),
        "original_auditor_sha256": digest(Path(original.__file__)),
        "lock_sha256": digest(ARTIFACT / "execution-lock.json"),
    }
    for run_id in cfg["runs"]:
        out, run = RESULTS / run_id, indexed[run_id]
        baseline = indexed[run["baseline_id"]]
        print(f"Auditing {run_id}", flush=True)
        try:
            checked = original.audit_run(out, torch.device("cuda:0"), 1e-5, 1e-5)
            assert checked["state"] == "passed"
            with np.load(out / "world.npz", allow_pickle=False) as loaded:
                world = {key: loaded[key].copy() for key in loaded.files}
            with np.load(ROOT / baseline["directory"] / "world.npz", allow_pickle=False) as old:
                assert set(old.files) == set(world)
                for key in old.files:
                    assert np.array_equal(old[key], world[key])
            observed_accuracy = []
            for new, old in zip(run["learning"], baseline["learning"], strict=True):
                for key in (
                    "step",
                    "counts",
                    "examples",
                    "effective_input_tokens",
                    "supervised_tokens",
                ):
                    assert new[key] == old[key], f"Paired exposure differs: {key}"
                with np.load(
                    out / f"predictions-{new['step']:07d}.npz", allow_pickle=False
                ) as arrays:
                    for split in original.SPLITS:
                        saved = {
                            key: arrays[split + "_" + key]
                            for key in ("answer", "stop", "nll", "target")
                        }
                        score = original.scores_from_arrays(saved, world[split])
                        original.check_scores(
                            score, new[split], f"node{new['step']}/{split}", 1e-7, 1e-7
                        )
                        if split == "test_composite":
                            observed_accuracy.append(score["accuracy"])
            independent_thresholds = {}
            for name, target in (
                ("paired_final", lock["references"][run["baseline_id"]]["paired_final_threshold"]),
                ("t90", 0.9),
                ("t95", 0.95),
            ):
                qualifying = [
                    i
                    for i in range(len(observed_accuracy) - 2)
                    if min(observed_accuracy[i : i + 3]) >= target
                ]
                idx = qualifying[0] if qualifying else None
                expected_step = run["learning"][idx]["step"] if idx is not None else None
                assert expected_step == run["thresholds"][name]["step"]
                independent_thresholds[name] = expected_step
            checked["paired_baseline_data_and_exposure_identical"] = True
            checked["all_saved_nodes_independently_scored"] = len(run["learning"])
            checked["independent_threshold_steps"] = independent_thresholds
        except Exception as exc:
            checked = {"run_id": run_id, "state": "failed", "error": repr(exc)}
        result["runs"].append(checked)
        write(ARTIFACT / "endpoint-audit.json", result)
        print(run_id, checked["state"], checked.get("error", ""), flush=True)
    result["passed"] = len(result["runs"]) == 12 and all(
        r["state"] == "passed" for r in result["runs"]
    )
    result["finished_utc"] = now()
    result["new_nodes_checked"] = sum(
        r.get("all_saved_nodes_independently_scored", 0) for r in result["runs"]
    )
    result["all_frozen_inputs_unchanged"] = True
    verify_files(ROOT, lock["files"])
    for ref in lock["references"].values():
        verify_files(ROOT, ref["files"])
    write(ARTIFACT / "endpoint-audit.json", result)
    if not result["passed"]:
        raise SystemExit(1)


def finalize():
    summary = report()
    audited = read_json(ARTIFACT / "endpoint-audit.json")
    assert summary["completed_new_runs"] == 12 and audited["passed"]
    new = [r for r in summary["runs"] if not r["reused_baseline"]]
    starts = [read_json(ROOT / r["directory"] / "metadata.json")["started_utc"] for r in new]
    completes = [read_json(ROOT / r["directory"] / "complete.json") for r in new]
    ends = [row["finished_utc"] for row in completes]
    totals = {
        "training_steps": sum(r["spec"]["steps"] for r in new),
        "evaluation_nodes": sum(len(r["learning"]) for r in new),
        "examples": sum(r["fixed_budget_endpoint"]["examples"] for r in new),
        "effective_input_tokens": sum(
            r["fixed_budget_endpoint"]["effective_input_tokens"] for r in new
        ),
        "supervised_tokens": sum(r["fixed_budget_endpoint"]["supervised_tokens"] for r in new),
        "estimated_training_flops": sum(
            r["fixed_budget_endpoint"]["estimated_training_flops"] for r in new
        ),
        "training_seconds_sum": sum(r["training_seconds"] for r in completes),
        "evaluation_seconds_sum": sum(r["evaluation_seconds"] for r in completes),
        "capture_seconds_sum": sum(r["learning"][0]["capture_seconds"] for r in new),
    }
    manifest = {
        "experiment": "grok-depth-extension-v1",
        "state": "complete",
        "finished_utc": now(),
        "new_runs": 12,
        "reused_baselines": 6,
        "independent_worlds": 3,
        "known_world_followup": True,
        "new_world_confirmation": False,
        "tests_passed_before_training": 38,
        "analysis_tests_passed": 2,
        "all_new_endpoints_reloaded": True,
        "all_new_nodes_scored_and_exposure_flops_audited": audited["new_nodes_checked"],
        "totals": totals,
        "first_run_started_utc": min(starts),
        "last_run_finished_utc": max(ends),
        "training_matrix_elapsed_seconds": (
            datetime.fromisoformat(max(ends)) - datetime.fromisoformat(min(starts))
        ).total_seconds(),
        "timing_note": (
            "UTC span excludes implementation, prior tests, subsequent audit/reporting; "
            "training/evaluation are sums within runs"
        ),
        "files": {},
    }
    launch_records = []
    for path in sorted((RESULTS.parent / "launches").glob("*-matrix.json")):
        launch_records.extend(read_json(path))
    manifest["launch_records"] = launch_records
    manifest["failed_attempts"] = [r for r in launch_records if r.get("exit_code", 0) != 0]
    for rel in (
        "scripts/report_grok_depth_extension.py",
        "scripts/audit_grok_depth.py",
        "tests/test_grok_depth_extension_report.py",
    ):
        destination = ARTIFACT / "analysis-source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
        manifest["files"][str(destination.relative_to(ARTIFACT))] = digest(destination)
    for path in sorted(ARTIFACT.iterdir()):
        if path.is_file() and path.name != "completion-manifest.json":
            manifest["files"][path.name] = digest(path)
    manifest["files"][str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    write(ARTIFACT / "completion-manifest.json", manifest)
    print(json.dumps({"state": "complete", "new_runs": 12, "totals": totals}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "audit", "finalize"))
    args = parser.parse_args()
    if args.command == "report":
        report()
    elif args.command == "audit":
        import os

        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "Assigned GPU 2 required"
        audit()
    else:
        finalize()


if __name__ == "__main__":
    main()
