"""Freeze, execute and report paired GPT/Loop training calibration."""

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_frontier import (
    SOURCE_FILES,
    audit,
    build_world,
    data_digest,
    file_hash,
    run_name,
    train,
)

ARTIFACTS = Path("docs/development-artifacts/storage-stability-v1")
RESULTS = Path("results/storage-stability-v1")
CONFIG = Path("configs/storage-stability-v1.json")


def prepare():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    specs = []
    for initialization, architecture, load, schedule in itertools.product(
        (731001, 731002, 731003),
        ("standard2", "loop1x2"),
        ("low", "high"),
        ("constant", "cosine"),
    ):
        specs.append(
            {
                "world": 730001,
                "initialization": initialization,
                "stream_seed": 732001,
                "architecture": architecture,
                "load": load,
                "schedule": schedule,
                "min_lr_ratio": 0.1,
                "heads_n": 1024,
                "bridges_n": 512,
                "tails_n": 128,
                "familiar_n": 64,
                "strict_n": 32,
                "holdout_fraction": 0.25,
                "low_extra": "anchors",
                "anchor_n": 32,
                "width": 256,
                "heads": 4,
                "dropout": 0.0,
                "batch_size": 192,
                "lr": 0.001,
                "weight_decay": 0.01,
                "warmup": 200,
                "steps": 16000,
                "nodes": [0, 4000, 8000, 12000, 16000],
                "test_repeats": [1, 2, 3],
            }
        )
    config = {
        "created_utc": utc(),
        "phase": "development",
        "specs": specs,
        "source": {path: file_hash(path) for path in SOURCE_FILES},
        "world_data_sha256": data_digest(build_world(specs[0])),
        "purpose": "calibrate stable training, not establish a capacity law",
        "unit": "one development world, three paired initializations",
        "primary": "final and late-node training-composition retention",
        "secondary": "unseen compositions, atomics, autonomous calls, load contrasts",
        "selection": (
            "Prefer cosine if it avoids late training-composition degradation without "
            "material atomic loss; otherwise retain constant. Keep every initialization "
            "and both loads. Any additional tuning is a separate development phase."
        ),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(CONFIG, config)
    for path in SOURCE_FILES:
        target = ARTIFACTS / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(CONFIG, ARTIFACTS / "frozen-config.json")
    print(json.dumps({"config": str(CONFIG), "runs": len(specs)}))


def execute(config_path, gpu_ids):
    config = json.loads(Path(config_path).read_text())
    assert all(file_hash(path) == digest for path, digest in config["source"].items())
    slots = queue.Queue()
    for gpu in gpu_ids:
        slots.put(gpu)

    def launch(spec):
        name = run_name(spec)
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists() and (folder / "audit.json").exists():
            return {"run": name, "skipped_complete": True}
        gpu = slots.get()
        try:
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            commands = ["run", "audit"] if not (folder / "complete.json").exists() else ["audit"]
            for command in commands:
                with (folder / f"{command}-process.log").open("a") as log:
                    process = subprocess.run(
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
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                status = {
                    "run": name,
                    "stage": command,
                    "gpu": gpu,
                    "returncode": process.returncode,
                    "utc": utc(),
                }
                write_json(folder / f"{command}-process-status.json", status)
                print(json.dumps(status), flush=True)
                if process.returncode:
                    return status
            return status
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        completed = list(pool.map(launch, config["specs"]))
    if any(row.get("returncode", 0) for row in completed):
        raise RuntimeError("Retained failed attempts require inspection")


def report(config_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads(Path(config_path).read_text())
    rows, histories, results = [], [], []
    for spec in config["specs"]:
        folder = RESULTS / run_name(spec)
        result = json.loads((folder / "complete.json").read_text())
        checked = json.loads((folder / "audit.json").read_text())
        assert checked["passed"] and result["source"] == config["source"]
        results.append(result)
        metrics = result["endpoint"]["metrics"]
        row = {k: spec[k] for k in ("world", "initialization", "architecture", "load", "schedule")}
        row.update(
            {
                "name": run_name(spec),
                "parameters": result["parameters"],
                "facts": result["independent_facts"],
                "atomic": metrics["common_atomic"]["accuracy"],
                "train": metrics["train_composite"]["accuracy"],
                "familiar": metrics["familiar_test"]["accuracy"],
                "strict": metrics["strict_test"]["accuracy"],
                "coverage": metrics["familiar_test"]["atomic_correct_coverage"],
                "calls": metrics["familiar_test"]["autonomous_two_calls"],
                "extra_atomic": metrics.get("extra_atomic", {}).get("accuracy"),
                "training_seconds": result["training_seconds"],
            }
        )
        rows.append(row)
        for node in json.loads((folder / "learning.json").read_text()):
            histories.append(
                {**row, "step": node["step"], "lr": node["lr"], "metrics": node["metrics"]}
            )
    # The identical common streams are a real comparison, not just equal totals.
    reference = np.load(RESULTS / run_name(config["specs"][0]) / "exposures.npz")
    for spec in config["specs"]:
        exposures = np.load(RESULTS / run_name(spec) / "exposures.npz")
        for i in (0, 1):
            np.testing.assert_array_equal(reference[f"stratum{i}"], exposures[f"stratum{i}"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "endpoints.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for architecture, load, schedule in itertools.product(
        ("standard2", "loop1x2"), ("low", "high"), ("constant", "cosine")
    ):
        selected = [
            r
            for r in rows
            if (r["architecture"], r["load"], r["schedule"]) == (architecture, load, schedule)
        ]
        aggregates.append(
            {
                "architecture": architecture,
                "load": load,
                "schedule": schedule,
                "initializations": len(selected),
                **{
                    key: float(np.mean([r[key] for r in selected]))
                    for key in ("atomic", "train", "familiar", "strict", "coverage", "calls")
                },
            }
        )
    summary = {
        "finished_utc": utc(),
        "runs": rows,
        "aggregates": aggregates,
        "audited_runs": len(rows),
        "expected_runs": len(config["specs"]),
        "common_exposures_exact": True,
        "unit": config["unit"],
    }
    write_json(ARTIFACTS / "summary.json", summary)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (architecture, task) in zip(
        axes.flat,
        itertools.product(("standard2", "loop1x2"), ("train_composite", "familiar_test")),
        strict=True,
    ):
        for load, schedule in itertools.product(("low", "high"), ("constant", "cosine")):
            selected = [
                r
                for r in histories
                if r["architecture"] == architecture
                and r["load"] == load
                and r["schedule"] == schedule
            ]
            steps = sorted({r["step"] for r in selected})
            means = [
                np.mean(
                    [r["metrics"][task]["accuracy"] * 100 for r in selected if r["step"] == step]
                )
                for step in steps
            ]
            ax.plot(steps, means, marker="o", label=f"{load}, {schedule}")
        ax.set_title(f"{architecture}: {task}")
        ax.set_ylim(0, 105)
        ax.set_xlabel("Training updates")
        ax.set_ylabel("Full generated accuracy (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "comparison.png", dpi=180)
    fig.savefig(ARTIFACTS / "comparison.pdf")
    plt.close(fig)
    write_json(
        ARTIFACTS / "completion-manifest.json",
        {
            "finished_utc": utc(),
            "config_sha256": file_hash(config_path),
            "source": config["source"],
            "runs": len(rows),
            "audited_runs": len(rows),
            "training_seconds_sum": sum(r["training_seconds"] for r in results),
            "supervised_tokens": sum(r["supervised_tokens"] for r in results),
            "training_flops": sum(r["endpoint"]["estimated_training_flops"] for r in results),
            "checkpoints": [
                {"name": run_name(r["spec"]), "sha256": r["checkpoint_sha256"]} for r in results
            ],
        },
    )
    print(json.dumps({"runs": len(rows), "aggregates": aggregates}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "run", "audit", "report"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--name")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "execute":
        execute(args.config, [int(gpu) for gpu in args.gpus.split(",")])
    elif args.command == "report":
        report(args.config)
    else:
        config = json.loads(Path(args.config).read_text())
        assert all(file_hash(path) == digest for path, digest in config["source"].items())
        spec = next(spec for spec in config["specs"] if run_name(spec) == args.name)
        out = RESULTS / args.name
        if args.command == "run":
            train(spec, out, config["source"], args.device)
        else:
            print(json.dumps(audit(out, args.device)), flush=True)


if __name__ == "__main__":
    main()
