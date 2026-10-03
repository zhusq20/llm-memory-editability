"""Freeze, execute, independently audit and summarize scaling development."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.latent_scaling import (
    SOURCE_FILES,
    audit,
    build_world,
    data_digest,
    file_hash,
    run_name,
    train,
)

ARTIFACTS = Path("docs/development-artifacts/latent-scaling-v1")
RESULTS = Path("results/latent-scaling-v1")
CONFIG = Path("configs/latent-scaling-v1.json")


def specifications():
    base = {
        "world": 730011,
        "initialization": 731011,
        "stream_seed": 732011,
        "heads_n": 1024,
        "bridges_n": 512,
        "tails_n": 128,
        "familiar_n": 64,
        "strict_n": 32,
        "holdout_fraction": 0.25,
        "low_extra": "anchors",
        "anchor_n": 32,
        "heads": 4,
        "dropout": 0.0,
        "batch_size": 192,
        "lr": 0.001,
        "weight_decay": 0.01,
        "warmup": 200,
        "schedule": "cosine",
        "min_lr_ratio": 0.1,
        "steps": 128000,
        "nodes": [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 96000, 128000],
        "repeat_nodes": [8000, 32000, 128000],
        "checkpoint_nodes": [0, 8000, 32000, 128000],
    }
    conditions = [
        (d, 1, r, n) for d, r, n in itertools.product((128, 256), (1, 2, 3), (64, 256, "all"))
    ]
    conditions += [
        (128, 2, 1, "all"),
        (256, 2, 1, "all"),
        (256, 3, 1, "all"),
        (64, 1, 2, "all"),
        (128, 1, 4, "all"),
        (128, 1, 8, "all"),
    ]
    return [
        {
            **base,
            "width": d,
            "layers": layers,
            "repeats": r,
            "composition_count": n,
            "test_repeats": [1, 2, 3, 4, 6, 8] if layers == 1 else [1],
        }
        for d, layers, r, n in conditions
    ]


def prepare():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    specs = specifications()
    worlds = {str(s["composition_count"]): build_world(s) for s in specs}
    config = {
        "created_utc": utc(),
        "phase": "development",
        "specs": specs,
        "source": {path: file_hash(path) for path in SOURCE_FILES},
        "data_sha256": {k: data_digest(w) for k, w in worlds.items()},
        "composition_examples": {k: len(w["train_composite"]) for k, w in worlds.items()},
        "primary": "fixed-endpoint and learning curves for held-out full two-hop generation",
        "secondary": "single-hop retention, train fit, strict role transfer, fixed-weight R sweep",
        "unit": "one independent development world and one paired initialization",
        "analysis": "Report curves in updates, supervised tokens, composition epochs and FLOPs. "
        "First train>=99%, test>=50% and test>=90% are descriptive crossing times; "
        "no forced grokking classification or model selection by best test R.",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(CONFIG, config)
    for path in SOURCE_FILES:
        target = ARTIFACTS / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(CONFIG, ARTIFACTS / "frozen-config.json")
    print(json.dumps({"runs": len(specs), "composition_examples": config["composition_examples"]}))


def execute(config_path, gpus):
    config = json.loads(Path(config_path).read_text())
    assert all(file_hash(p) == h for p, h in config["source"].items())
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    launched = utc()

    def launch(spec):
        name = run_name(spec)
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists() and (folder / "audit.json").exists():
            return {"run": name, "skipped_complete": True}
        gpu = slots.get()
        try:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            commands = ["audit"] if (folder / "complete.json").exists() else ["run", "audit"]
            for command in commands:
                with (folder / f"{command}-process.log").open("a") as log:
                    p = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            command,
                            "--config",
                            str(config_path),
                            "--name",
                            name,
                            "--device",
                            f"cuda:{gpu}",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                status = {
                    "run": name,
                    "stage": command,
                    "gpu": gpu,
                    "returncode": p.returncode,
                    "utc": utc(),
                }
                write_json(folder / f"{command}-process-status.json", status)
                print(json.dumps(status), flush=True)
                if p.returncode:
                    return status
            return status
        finally:
            slots.put(gpu)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = list(pool.map(launch, config["specs"]))
    write_json(
        ARTIFACTS / "execution.json",
        {
            "launched_utc": launched,
            "finished_utc": utc(),
            "seconds": time.perf_counter() - start,
            "gpus": gpus,
            "statuses": statuses,
        },
    )
    if any(r.get("returncode", 0) for r in statuses):
        raise RuntimeError("Failed attempts retained for inspection")


def first_crossing(nodes, task, threshold):
    return next((n["step"] for n in nodes if n["metrics"][task]["accuracy"] >= threshold), None)


def report(config_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads(Path(config_path).read_text())
    rows, complete, histories, audited = [], [], {}, []
    reference_exposures, initial_states = None, {}
    for spec in config["specs"]:
        name, folder = run_name(spec), RESULTS / run_name(spec)
        result = json.loads((folder / "complete.json").read_text())
        checked = json.loads((folder / "audit.json").read_text())
        assert checked["passed"] and result["source"] == config["source"]
        complete.append(result)
        audited.append(checked)
        nodes = json.loads((folder / "learning.json").read_text())
        assert [n["step"] for n in nodes] == spec["nodes"]
        histories[name] = nodes
        metrics = result["endpoint"]["metrics"]
        row = {
            k: spec[k]
            for k in (
                "world",
                "initialization",
                "width",
                "layers",
                "repeats",
                "composition_count",
                "steps",
            )
        }
        row.update(
            {
                "name": name,
                "parameters": result["parameters"],
                "composition_examples": result["composition_examples"],
                "atomic": metrics["common_atomic"]["accuracy"],
                "anchor": metrics["anchor_atomic"]["accuracy"],
                "train": metrics["train_composite"]["accuracy"],
                "familiar": metrics["familiar_test"]["accuracy"],
                "strict": metrics["strict_test"]["accuracy"],
                "coverage": metrics["familiar_test"]["atomic_correct_coverage"],
                "calls": metrics["familiar_test"]["autonomous_two_calls"],
                "conditional": metrics["familiar_test"]["conditional_accuracy"],
                **result["role_coverage"],
                "first_atomic99": first_crossing(nodes, "common_atomic", 0.99),
                "first_train99": first_crossing(nodes, "train_composite", 0.99),
                "first_test50": first_crossing(nodes, "familiar_test", 0.5),
                "first_test90": first_crossing(nodes, "familiar_test", 0.9),
                "training_seconds": result["training_seconds"],
                "training_flops": result["endpoint"]["estimated_training_flops"],
            }
        )
        rows.append(row)
        counts = np.load(folder / "exposures.npz")
        if reference_exposures is None:
            reference_exposures = {i: counts[f"stratum{i}"].copy() for i in (0, 2)}
        for i in (0, 2):
            np.testing.assert_array_equal(reference_exposures[i], counts[f"stratum{i}"])
        key = (spec["width"], spec["layers"])
        initial_states.setdefault(key, result["initial_model_sha256"])
        assert initial_states[key] == result["initial_model_sha256"]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        ARTIFACTS / "summary.json",
        {
            "finished_utc": utc(),
            "runs": rows,
            "expected_runs": len(config["specs"]),
            "audited_runs": len(audited),
            "repeat_checks": sum(a["repeat_checks"] for a in audited),
            "common_atomic_exposures_exact": True,
            "paired_initial_states_exact": True,
            "histories": histories,
            "unit": config["unit"],
        },
    )

    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (width, count) in zip(
        axes.flat, itertools.product((128, 256), (64, 256, "all")), strict=True
    ):
        selected = [
            r
            for r in rows
            if (r["width"], r["layers"], r["composition_count"]) == (width, 1, count)
        ]
        for row in selected:
            nodes = histories[row["name"]]
            x = [n["step"] for n in nodes]
            for task, style in (("familiar_test", "-"), ("train_composite", "--")):
                ax.plot(
                    x,
                    [100 * n["metrics"][task]["accuracy"] for n in nodes],
                    style,
                    color=colors[row["repeats"]],
                    label=f"R{row['repeats']} {'test' if style == '-' else 'train'}",
                    alpha=0.9,
                )
        ax.set_title(f"Width {width}, {selected[0]['composition_examples']} unique compositions")
        ax.set_ylim(0, 105)
        ax.set_xscale("symlog", linthresh=256)
        ax.set_xlabel("Training updates")
        ax.set_ylabel("Full generated accuracy (%)")
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"learning.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for width in (128, 256):
        for repeat in (1, 2, 3):
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["width"] == width and r["layers"] == 1 and r["repeats"] == repeat
                ],
                key=lambda r: r["composition_examples"],
            )
            axes[0].plot(
                [r["composition_examples"] for r in selected],
                [100 * r["familiar"] for r in selected],
                marker="o",
                label=f"d{width}, R{repeat}",
            )
    axes[0].set_xlabel("Unique training compositions")
    axes[0].set_title("Data scale; fixed facts and updates")
    for layers, repeats, label in ((1, 2, "one block, R2"),):
        selected = sorted(
            [
                r
                for r in rows
                if r["layers"] == layers
                and r["repeats"] == repeats
                and r["composition_count"] == "all"
            ],
            key=lambda r: r["width"],
        )
        axes[1].plot(
            [r["width"] for r in selected],
            [100 * r["familiar"] for r in selected],
            marker="o",
            label=label,
        )
    axes[1].set_xlabel("Hidden width (parameters also change)")
    axes[1].set_title("Representation scale")
    for layers, label in ((1, "shared block"), (2, "2 unique blocks"), (3, "3 unique blocks")):
        selected = [
            r
            for r in rows
            if r["width"] == 256 and r["layers"] == layers and r["composition_count"] == "all"
        ]
        axes[2].scatter(
            [r["parameters"] / 1e6 for r in selected],
            [100 * r["familiar"] for r in selected],
            label=label,
        )
        for row in selected:
            axes[2].annotate(
                f"R{row['repeats']}",
                (row["parameters"] / 1e6, 100 * row["familiar"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[2].set_xlabel("Independent parameters (millions)")
    axes[2].set_title("Parameter and execution depth")
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_ylabel("Held-out full accuracy (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"scaling.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (width, task) in zip(
        axes.flat, itertools.product((128, 256), ("familiar_test", "common_atomic")), strict=True
    ):
        for result in complete:
            spec = result["spec"]
            if spec["width"] != width or spec["layers"] != 1 or spec["composition_count"] != "all":
                continue
            metrics = result["repeat_metrics"][str(spec["steps"])]
            r = sorted(map(int, metrics))
            ax.plot(
                r,
                [100 * metrics[str(i)][task]["accuracy"] for i in r],
                marker="o",
                label=f"train R{spec['repeats']}",
            )
        ax.set_title(f"Width {width}: {task}")
        ax.set_xlabel("Inference repetitions; same weights")
        ax.set_ylabel("Full accuracy (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ARTIFACTS / f"repeat-sweep.{ext}", dpi=180)
    plt.close(fig)

    write_json(
        ARTIFACTS / "completion-manifest.json",
        {
            "finished_utc": utc(),
            "runs": len(rows),
            "audited_runs": len(audited),
            "config_sha256": file_hash(config_path),
            "source": config["source"],
            "supervised_tokens": sum(r["supervised_tokens"] for r in complete),
            "training_seconds_sum": sum(r["training_seconds"] for r in complete),
            "process_seconds_sum": sum(r["process_seconds"] for r in complete),
            "training_flops": sum(r["endpoint"]["estimated_training_flops"] for r in complete),
            "max_audit_nll_difference": max(a["max_nll_difference"] for a in audited),
            "checkpoints": [
                {"name": run_name(r["spec"]), "sha256": r["checkpoint_sha256"]} for r in complete
            ],
        },
    )
    print(
        json.dumps(
            {"runs": len(rows), "audit_repeat_checks": sum(a["repeat_checks"] for a in audited)}
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "execute", "run", "audit", "report"))
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--name")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "execute":
        execute(args.config, [int(gpu) for gpu in args.gpus.split(",")])
    elif args.command == "report":
        report(args.config)
    else:
        config = json.loads(Path(args.config).read_text())
        assert all(file_hash(path) == h for path, h in config["source"].items())
        spec = next(s for s in config["specs"] if run_name(s) == args.name)
        folder = RESULTS / args.name
        if args.command == "run":
            train(spec, folder, config["source"], args.device)
        else:
            print(json.dumps(audit(folder, args.device)), flush=True)


if __name__ == "__main__":
    main()
