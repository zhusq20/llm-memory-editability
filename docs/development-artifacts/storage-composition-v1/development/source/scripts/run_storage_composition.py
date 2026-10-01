"""Prepare, execute, audit and summarize the standard-architecture fact-load batch."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import (
    ARCHITECTURES,
    SOURCE_FILES,
    audit_run,
    build_world,
    data_digest,
    file_hash,
    run,
)

ROOT = Path("docs/development-artifacts/storage-composition-v1")


def prepare(phase, steps):
    config_path = Path(f"configs/storage-composition-v1-{phase}.json")
    if config_path.exists():
        raise FileExistsError(config_path)
    worlds = [710001] if phase == "development" else [720001, 720002, 720003]
    specs = []
    for world, architecture, load in itertools.product(worlds, ARCHITECTURES, ("low", "high")):
        specs.append(
            {
                "world": world,
                "initialization": 711001 if phase == "development" else 721001,
                "stream_seed": world + 2000,
                "architecture": architecture,
                "load": load,
                "heads_n": 1024,
                "bridges_n": 512,
                "tails_n": 128,
                "familiar_n": 64,
                "strict_n": 32,
                "holdout_fraction": 0.25,
                "width": 256,
                "heads": 4,
                "dropout": 0.0,
                "batch_size": 192,
                "lr": 0.001,
                "weight_decay": 0.01,
                "warmup": 200,
                "steps": steps,
                "nodes": sorted({0, steps // 4, steps // 2, steps}),
                "test_repeats": [1, 2, 3, 4, 6, 8],
            }
        )
    source = {path: file_hash(path) for path in SOURCE_FILES}
    config = {
        "phase": phase,
        "created_utc": utc(),
        "specs": specs,
        "source": source,
        "primary_endpoint": "familiar_test.accuracy at final fixed step",
        "secondary": ["strict_test", "atomic coverage", "autonomous two calls", "repeat sweep"],
        "unit": "world; one paired initialization per world",
        "world_data": {
            str(w): data_digest(build_world(next(s for s in specs if s["world"] == w)))
            for w in worlds
        },
    }
    write_json(config_path, config)
    snapshot = ROOT / phase / "source"
    for path in SOURCE_FILES:
        target = snapshot / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(config_path, ROOT / phase / "frozen-config.json")
    print(json.dumps({"config": str(config_path), "runs": len(specs), "source": source}))


def name(spec):
    return f"w{spec['world']}-{spec['architecture']}-{spec['load']}"


def execute(config_path, workers, selected):
    config = json.loads(Path(config_path).read_text())
    assert all(file_hash(path) == digest for path, digest in config["source"].items())
    runs = [s for s in config["specs"] if not selected or name(s) in selected]
    phase = config["phase"]

    def launch(spec):
        folder = Path("results/storage-composition-v1") / phase / name(spec)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists():
            return {"run": name(spec), "skipped_complete": True}
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (folder / "process.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "run",
                    "--config",
                    str(config_path),
                    "--name",
                    name(spec),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        status = {"run": name(spec), "returncode": result.returncode}
        print(json.dumps(status), flush=True)
        return status

    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed = list(pool.map(launch, runs))
    if any(r.get("returncode", 0) for r in completed):
        raise RuntimeError("A worker failed; inspect its retained process.log")


def report(config_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads(Path(config_path).read_text())
    phase = config["phase"]
    folder = ROOT / phase
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in config["specs"]:
        out = Path("results/storage-composition-v1") / phase / name(spec)
        if not (out / "complete.json").exists():
            continue
        result = json.loads((out / "complete.json").read_text())
        row = {
            k: result[k]
            for k in (
                "parameters",
                "independent_facts",
                "training_seconds",
                "supervised_tokens",
                "executed_blocks",
            )
        }
        row.update(
            {
                "name": name(spec),
                "world": spec["world"],
                "architecture": spec["architecture"],
                "load": spec["load"],
                "metrics": result["endpoint"]["metrics"],
                "estimated_training_flops": result["endpoint"]["estimated_training_flops"],
                "repeat_metrics": result["repeat_metrics"],
                "audit_passed": (out / "audit.json").exists(),
            }
        )
        rows.append(row)
    contrasts = []
    for architecture in ARCHITECTURES:
        for world in sorted({r["world"] for r in rows}):
            pair = {
                r["load"]: r
                for r in rows
                if r["world"] == world and r["architecture"] == architecture
            }
            if pair.keys() >= {"low", "high"}:
                for metric in ("accuracy", "conditional_accuracy", "atomic_correct_coverage"):
                    for task in ("familiar_test", "strict_test"):
                        values = [pair[load]["metrics"][task][metric] for load in ("low", "high")]
                        contrasts.append(
                            {
                                "architecture": architecture,
                                "world": world,
                                "task": task,
                                "metric": metric,
                                "high_minus_low": values[1] - values[0]
                                if None not in values
                                else None,
                            }
                        )
    summary = {
        "phase": phase,
        "runs": rows,
        "contrasts": contrasts,
        "expected_runs": len(config["specs"]),
        "completed_runs": len(rows),
        "audit_passed_runs": sum(r["audit_passed"] for r in rows),
        "utc": utc(),
    }
    write_json(folder / "summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, task in zip(axes, ("common_atomic", "familiar_test", "strict_test"), strict=True):
        for j, load in enumerate(("low", "high")):
            means = [
                np.mean(
                    [
                        r["metrics"][task]["accuracy"] * 100
                        for r in rows
                        if r["load"] == load and r["architecture"] == a
                    ]
                )
                if any(r["load"] == load and r["architecture"] == a for r in rows)
                else np.nan
                for a in ARCHITECTURES
            ]
            ax.bar(np.arange(5) + (j - 0.5) * 0.36, means, 0.36, label=load)
        ax.set_xticks(np.arange(5), ["Std1", "Std2", "Std3", "Loop1×2", "Loop1×3"], rotation=25)
        ax.set_ylim(0, 105)
        ax.set_title(task)
        ax.set_ylabel("Full generated accuracy (%)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=180)
    fig.savefig(folder / "comparison.pdf")
    plt.close(fig)
    print(json.dumps({"summary": str(folder / "summary.json"), "runs": len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "execute", "run", "audit", "report"])
    parser.add_argument("--phase", choices=["development", "confirmation"], default="development")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--config", default="configs/storage-composition-v1-development.json")
    parser.add_argument("--name")
    parser.add_argument("--select", nargs="*")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.phase, args.steps)
    elif args.command == "execute":
        execute(args.config, args.workers, args.select)
    elif args.command == "report":
        report(args.config)
    else:
        config = json.loads(Path(args.config).read_text())
        specs = [s for s in config["specs"] if name(s) == args.name]
        if len(specs) != 1:
            raise ValueError("Select exactly one run name")
        assert all(file_hash(path) == digest for path, digest in config["source"].items())
        out = Path("results/storage-composition-v1") / config["phase"] / args.name
        if args.command == "run":
            run(specs[0], out)
        else:
            print(json.dumps(audit_run(out)))


if __name__ == "__main__":
    main()
