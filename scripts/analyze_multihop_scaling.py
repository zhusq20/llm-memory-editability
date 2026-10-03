"""Freeze, execute and summarize same-weight recurrence scans.

Test R is registered as 1/2/4/6 at the 8k/32k/64k checkpoints. The training R
remains the main result; changing test R does not retrain or rescale weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from report_multihop_scaling import (
    ROOT,
    SCORES,
    TASKS,
    audit_predictions,
    file_hash,
    identity,
    run_name,
    world_balanced,
    write_csv,
    write_json,
)

SCAN_SOURCES = (
    "scripts/analyze_multihop_scaling.py",
    "scripts/report_multihop_scaling.py",
    "tests/test_multihop_scaling_report.py",
)
NODES = (8000, 32000, 64000)
REPEATS = (1, 2, 4, 6)
PRECISION = {
    "cpu_threads": 1,
    "cpu_interop_threads": 1,
    "parameter_dtype": "float32",
    "matmul_tf32": True,
    "cudnn_tf32": True,
}


def check_source(source):
    for path, digest in source.items():
        resolved = Path(path) if Path(path).is_absolute() else ROOT / path
        if file_hash(resolved) != digest:
            raise RuntimeError("Frozen source changed: " + path)


def prepare(main_config, results, output):
    main_config, results, output = map(
        lambda path: Path(path).resolve(), (main_config, results, output)
    )
    if (output / "scan-config.json").exists():
        raise FileExistsError(output / "scan-config.json")
    config = json.loads(main_config.read_text())
    design = Path(config["design"])
    design = design if design.is_absolute() else ROOT / design
    if file_hash(design) != config["design_sha256"]:
        raise RuntimeError("Main design changed")
    specs = [spec for spec in config["specs"] if spec["repeats"] > 1]
    if not specs:
        raise ValueError("No Loop models registered")
    for spec in specs:
        if not set(NODES) <= set(spec["checkpoint_nodes"]):
            raise ValueError("Registered scan checkpoints were not saved by the main stage")
        if spec["repeats"] not in REPEATS or spec["layers"] != 1:
            raise ValueError("Scan expects one shared block and an included training R")
    source = dict(config["source"])
    source.update({path: file_hash(ROOT / path) for path in SCAN_SOURCES})
    check_source(source)
    output.mkdir(parents=True, exist_ok=True)
    frozen = {
        "phase": config["phase"] + "; same-weight test-time recurrence",
        "main_config": str(main_config),
        "main_config_sha256": file_hash(main_config),
        "design": str(design),
        "design_sha256": config["design_sha256"],
        "results": str(results),
        "output": str(output),
        "source": source,
        "main_source": config["source"],
        "specs": specs,
        "nodes": list(NODES),
        "test_repeats": list(REPEATS),
        "precision": PRECISION,
        "evaluations": len(specs) * len(NODES) * len(REPEATS),
        "new_evaluations": len(specs) * len(NODES) * (len(REPEATS) - 1),
        "analysis_unit": "Equal-weight worlds; paired initializations inside each world",
        "main_score": "The registered training R; no checkpoint or test-R selection",
        "reuse": "Training-R raw predictions and metrics copied from the main checkpoint node",
        "limitations": (
            "Changing R changes the trained execution distribution and can degrade atomic recall"
        ),
    }
    for path in source:
        original = Path(path) if Path(path).is_absolute() else ROOT / path
        destination = output / "source" / (Path(path).name if Path(path).is_absolute() else path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    shutil.copy2(main_config, output / "main-frozen-config.json")
    shutil.copy2(design, output / "frozen-design.md")
    write_json(output / "scan-config.json", frozen)
    print(
        json.dumps(
            {
                "config": str(output / "scan-config.json"),
                "evaluations": frozen["evaluations"],
                "new_evaluations": frozen["new_evaluations"],
            }
        )
    )
    return frozen


def load_config(path):
    config = json.loads(Path(path).read_text())
    check_source(config["source"])
    for name, digest in config["source"].items():
        saved = (
            Path(config["output"])
            / "source"
            / (Path(name).name if Path(name).is_absolute() else name)
        )
        if file_hash(saved) != digest:
            raise RuntimeError("Archived scan source changed: " + name)
    if file_hash(config["main_config"]) != config["main_config_sha256"]:
        raise RuntimeError("Main frozen config changed")
    if file_hash(config["design"]) != config["design_sha256"]:
        raise RuntimeError("Frozen design changed")
    if file_hash(Path(config["output"]) / "frozen-design.md") != config["design_sha256"]:
        raise RuntimeError("Archived scan design changed")
    if (
        config["precision"] != PRECISION
        or config["nodes"] != list(NODES)
        or config["test_repeats"] != list(REPEATS)
    ):
        raise RuntimeError("Registered scan rule changed")
    return config


def evaluation_folder(config, spec, node, repeats):
    return Path(config["output"]) / run_name(spec) / f"checkpoint-{node:06d}" / f"test-r{repeats}"


def verify_saved(folder, config, spec, node, repeats, checkpoint):
    complete = json.loads((folder / "complete.json").read_text())
    expected = {
        "spec": spec,
        "step": node,
        "test_repeats": repeats,
        "source": config["source"],
        "precision": config["precision"],
        "checkpoint_sha256": file_hash(checkpoint),
        "main_config_sha256": config["main_config_sha256"],
        "input_prediction_sha256": file_hash(checkpoint.parent / f"predictions-{node:06d}.npz"),
        "world_archive_sha256": file_hash(checkpoint.parent / "world.npz"),
    }
    for key, value in expected.items():
        if complete[key] != value:
            raise ValueError("Saved scan identity differs: " + str(folder) + " " + key)
    for name, digest in complete["artifact_sha256"].items():
        if file_hash(folder / name) != digest:
            raise ValueError("Saved scan artifact changed: " + str(folder / name))
    if not complete["audit"]["passed"] or not complete["weights_unchanged"]:
        raise ValueError("Scan failed independent audit")
    return complete


def _setup_torch(device):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not device.startswith("cuda:"):
        raise ValueError("Production scans retain the main stage's CUDA/TF32 precision")
    torch.cuda.set_device(torch.device(device))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def scan_one_checkpoint(config, spec, node, device):
    import torch

    from llm_memory_editability.latent_scaling import model_digest
    from llm_memory_editability.multihop_scaling import build_world, construct, evaluate

    run = Path(config["results"]) / run_name(spec)
    checkpoint = run / f"model-{node:06d}.pt"
    prediction_file = run / f"predictions-{node:06d}.npz"
    if not all(path.exists() for path in (checkpoint, prediction_file, run / "learning.json")):
        return 0
    check_source(config["source"])
    if all(
        (evaluation_folder(config, spec, node, repeats) / "complete.json").exists()
        for repeats in config["test_repeats"]
    ):
        for repeats in config["test_repeats"]:
            verify_saved(
                evaluation_folder(config, spec, node, repeats),
                config,
                spec,
                node,
                repeats,
                checkpoint,
            )
        return len(config["test_repeats"])
    saved_spec = json.loads((run / "spec.json").read_text())
    if saved_spec["spec"] != spec or saved_spec["source"] != config["main_source"]:
        raise ValueError("Main run differs from registered source/specification")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if state["spec"] != spec or state["step"] != node:
        raise ValueError("Checkpoint specification/node differs")
    world = build_world(spec)
    with np.load(run / "world.npz") as archived:
        for task in TASKS:
            np.testing.assert_array_equal(world[task], archived[task])
    model = construct(spec, device)
    model.load_state_dict(state["model"])
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Parameter dtype differs from frozen float32 precision")
    digest = model_digest(model)
    history = json.loads((run / "learning.json").read_text())
    by_node = {row["step"]: row for row in history}
    if node not in by_node:
        return 0
    done = 0
    for repeats in config["test_repeats"]:
        folder = evaluation_folder(config, spec, node, repeats)
        if (folder / "complete.json").exists():
            verify_saved(folder, config, spec, node, repeats, checkpoint)
            done += 1
            continue
        if folder.exists():
            raise FileExistsError("Incomplete scan artifact requires inspection: " + str(folder))
        check_source(config["source"])
        started = time.perf_counter()
        model.repeats = repeats
        if repeats == spec["repeats"]:
            metrics = by_node[node]["metrics"]
            with np.load(prediction_file) as predictions:
                audit = audit_predictions(world, predictions, metrics)
            folder.mkdir(parents=True)
            shutil.copy2(prediction_file, folder / "predictions.npz")
        else:
            metrics, predictions = evaluate(model, world, device)
            audit = audit_predictions(world, predictions, metrics)
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "predictions.npz", **predictions)
        unchanged = model_digest(model) == digest
        if not unchanged:
            raise ValueError("Test recurrence changed weights")
        write_json(folder / "metrics.json", metrics)
        result = {
            "spec": spec,
            "step": node,
            "test_repeats": repeats,
            "source": config["source"],
            "precision": config["precision"],
            "main_config_sha256": config["main_config_sha256"],
            "checkpoint_sha256": file_hash(checkpoint),
            "model_sha256": digest,
            "weights_unchanged": unchanged,
            "audit": audit,
            "reused_training_R": repeats == spec["repeats"],
            "input_prediction_sha256": file_hash(prediction_file),
            "world_archive_sha256": file_hash(run / "world.npz"),
            "seconds": time.perf_counter() - started,
            "environment": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "artifact_sha256": {
                name: file_hash(folder / name) for name in ("metrics.json", "predictions.npz")
            },
        }
        write_json(folder / "complete.json", result)
        print(
            json.dumps(
                {
                    "run": run.name,
                    "step": node,
                    "test_R": repeats,
                    "reused": result["reused_training_R"],
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )
        done += 1
    return done


def run(config_path, device, watch=False, shard=0, shards=1):
    config = load_config(config_path)
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard")
    _setup_torch(device)
    specs = [spec for index, spec in enumerate(config["specs"]) if index % shards == shard]
    expected = len(specs) * len(config["nodes"]) * len(config["test_repeats"])
    while True:
        config = load_config(config_path)
        finished = sum(
            scan_one_checkpoint(config, spec, node, device)
            for spec in specs
            for node in config["nodes"]
        )
        write_json(
            Path(config["output"]) / f"status-shard-{shard}.json",
            {
                "finished": finished,
                "expected": expected,
                "device": device,
                "shard": shard,
                "shards": shards,
            },
        )
        if finished == expected or not watch:
            return finished
        time.sleep(15)


def scan_differences(records):
    """Paired test-R differences relative to each model's registered training R."""
    by_node = {}
    for row in records:
        key = (row["run"], row["step"], row["task"])
        by_node.setdefault(key, {})[row["test_repeats"]] = row
    differences = []
    for members in by_node.values():
        first = next(iter(members.values()))
        baseline = members.get(first["repeats"])
        if baseline is None:
            continue
        for _repeats, row in sorted(members.items()):
            for metric in SCORES:
                if row.get(metric) is None or baseline.get(metric) is None:
                    continue
                differences.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                "world",
                                "initialization",
                                "width",
                                "phi",
                                "architecture",
                                "executed_depth",
                                "repeats",
                                "step",
                                "task",
                                "test_repeats",
                            )
                        },
                        "metric": metric,
                        "difference_pp": 100 * (row[metric] - baseline[metric]),
                    }
                )
    return differences


def report(config_path, output, allow_incomplete=False):
    config, output = load_config(config_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, pending, checks = [], [], []
    for spec in config["specs"]:
        run_folder = Path(config["results"]) / run_name(spec)
        main_audit = run_folder / "audit.json"
        main_finished = run_folder / "complete.json"
        audited = main_audit.exists() and main_finished.exists()
        if audited:
            checked = json.loads(main_audit.read_text())
            complete = json.loads(main_finished.read_text())
            if (
                not checked["passed"]
                or complete["spec"] != spec
                or complete["source"] != config["main_source"]
            ):
                raise ValueError("Main run failed reload audit or identity check")
        for node in config["nodes"]:
            for repeats in config["test_repeats"]:
                folder = evaluation_folder(config, spec, node, repeats)
                if not audited or not (folder / "complete.json").exists():
                    pending.append(str(folder))
                    continue
                checkpoint = run_folder / f"model-{node:06d}.pt"
                complete = verify_saved(folder, config, spec, node, repeats, checkpoint)
                metrics = json.loads((folder / "metrics.json").read_text())
                with (
                    np.load(run_folder / "world.npz") as world,
                    np.load(folder / "predictions.npz") as predictions,
                ):
                    audit = audit_predictions(world, predictions, metrics)
                checks.append(
                    {"run": run_folder.name, "step": node, "test_repeats": repeats, **audit}
                )
                for task in TASKS:
                    records.append(
                        {
                            "run": run_folder.name,
                            **identity(spec),
                            "step": node,
                            "test_repeats": repeats,
                            "task": task,
                            **metrics[task],
                        }
                    )
    if pending and not allow_incomplete:
        raise FileNotFoundError(
            "Scans or main checkpoint reload audits pending: " + str(len(pending))
        )
    groups = (
        "width",
        "phi",
        "architecture",
        "executed_depth",
        "repeats",
        "step",
        "test_repeats",
        "task",
    )
    values = (
        "accuracy",
        "answer_accuracy",
        "answer_nll",
        "atomic_correct_coverage",
        "conditional_accuracy",
        "autonomous_two_calls",
        "autonomous_path_accuracy",
    )
    world_records, aggregate = world_balanced(records, groups, values)
    differences = scan_differences(records)
    world_differences, aggregate_differences = world_balanced(
        differences, (*groups, "metric"), ("difference_pp",)
    )
    for name, rows in (
        ("scan", records),
        ("world-scan", world_records),
        ("aggregate-scan", aggregate),
        ("scan-differences", differences),
        ("world-scan-differences", world_differences),
        ("aggregate-scan-differences", aggregate_differences),
    ):
        write_csv(output / f"{name}.csv", rows)
    summary = {
        "phase": config["phase"],
        "scan_config_sha256": file_hash(config_path),
        "main_config_sha256": config["main_config_sha256"],
        "complete": not pending,
        "registered_evaluations": config["evaluations"],
        "audited_evaluations": len(checks),
        "pending_evaluations": pending,
        "raw_prediction_checks": checks,
        "world_scan": world_records,
        "aggregate_scan": aggregate,
        "world_differences": world_differences,
        "aggregate_differences": aggregate_differences,
        "main_score": config["main_score"],
        "precision": config["precision"],
        "analysis_unit": config["analysis_unit"],
        "limitations": config["limitations"],
    }
    write_json(output / "scan-summary.json", summary)
    if records:
        plot(aggregate, output)
    print(
        json.dumps(
            {"audited_evaluations": len(checks), "pending": len(pending), "complete": not pending}
        )
    )
    return summary


def plot(records, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    for pool in ("atomic", "familiar", "strict"):
        figure, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True, sharey=True)
        for row_index, (width, phi) in enumerate(((128, 1), (128, 4), (256, 1), (256, 4))):
            for col_index, node in enumerate(NODES):
                axis = axes[row_index, col_index]
                for trained in (2, 4):
                    for hop in (0,) if pool == "atomic" else (2, 3, 4):
                        task = "atomic" if hop == 0 else f"{pool}_{hop}"
                        line = sorted(
                            (
                                row
                                for row in records
                                if row["width"] == width
                                and row["phi"] == phi
                                and row["step"] == node
                                and row["repeats"] == trained
                                and row["task"] == task
                            ),
                            key=lambda row: row["test_repeats"],
                        )
                        if line:
                            axis.plot(
                                [row["test_repeats"] for row in line],
                                [
                                    np.nan if row["accuracy"] is None else 100 * row["accuracy"]
                                    for row in line
                                ],
                                "o-" if trained == 2 else "s--",
                                label=f"train R{trained} {task}",
                            )
                axis.set_title(f"width {width} | support {phi} | step {node}", fontsize=9)
                axis.set_xticks(REPEATS)
                axis.set_ylim(-3, 103)
                axis.grid(alpha=0.2)
                if col_index == 0:
                    axis.set_ylabel("Answer + EOS accuracy (%)")
                if row_index == 3:
                    axis.set_xlabel("Test R, same weights")
        axes[0, -1].legend(fontsize=6)
        figure.suptitle(f"Same-weight recurrence | {pool} | every registered R retained")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output / f"scan-{pool}.{suffix}", dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("prepare")
    freeze.add_argument("--main-config", required=True)
    freeze.add_argument("--results", required=True)
    freeze.add_argument("--out", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--config", required=True)
    execute.add_argument("--device", default="cuda:7")
    execute.add_argument("--watch", action="store_true")
    execute.add_argument("--shard", type=int, default=0)
    execute.add_argument("--shards", type=int, default=1)
    summarize = subparsers.add_parser("report")
    summarize.add_argument("--config", required=True)
    summarize.add_argument("--out", required=True)
    summarize.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.main_config, args.results, args.out)
    elif args.command == "run":
        run(args.config, args.device, args.watch, args.shard, args.shards)
    else:
        report(args.config, args.out, args.allow_incomplete)


if __name__ == "__main__":
    main()
