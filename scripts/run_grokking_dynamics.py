"""Freeze and execute fixed-support grokking development without changing old batches."""

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
from llm_memory_editability.latent_scaling import SOURCE_FILES as BASE_SOURCES
from llm_memory_editability.latent_scaling import audit, build_world, construct, model_digest, train
from llm_memory_editability.storage_composition import audit_world, data_digest, file_hash

ARTIFACTS = Path("docs/development-artifacts/grokking-dynamics-v1")
RESULTS = Path("results/grokking-dynamics-v1")
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *BASE_SOURCES,
            "scripts/run_grokking_dynamics.py",
            "tests/test_grokking_dynamics.py",
            "docs/development-artifacts/grokking-dynamics-v1/design.md",
        ]
    )
)


def specifications(phase="development", lr=0.0001):
    pilot = phase == "calibration"
    steps = 16000 if pilot else 512000
    nodes = sorted(
        {0, 256, 512, 1000, 2000, 4000, 8000, 16000}
        | (set() if pilot else set(range(24000, 512001, 8000)))
    )
    checkpoint_nodes = (
        nodes
        if pilot
        else [
            0,
            1000,
            2000,
            4000,
            8000,
            16000,
            32000,
            64000,
            96000,
            128000,
            192000,
            256000,
            384000,
            512000,
        ]
    )
    base = {
        "world": 780011 if pilot else 780101,
        "stream_seed": 782011 if pilot else 782101,
        "heads_n": 1024,
        "bridges_n": 512,
        "tails_n": 128,
        "familiar_n": 64,
        "strict_n": 32,
        "holdout_fraction": 0.25,
        "low_extra": "anchors",
        "anchor_n": 32,
        "heads": 4,
        "width": 128,
        "dropout": 0.0,
        "batch_size": 192,
        "lr": lr,
        "warmup": 2000,
        "schedule": "constant",
        "min_lr_ratio": 1.0,
        "steps": steps,
        "nodes": nodes,
        "checkpoint_nodes": checkpoint_nodes,
        "composition_count": "all",
        "repeat_nodes": [],
        "test_repeats": [],
        "phase": phase,
    }
    initializations = [781011] if pilot else [781101, 781102]
    return [
        {
            **base,
            "initialization": initialization,
            "layers": layers,
            "repeats": repeats,
            "weight_decay": decay,
        }
        for initialization, (layers, repeats), decay in itertools.product(
            initializations, [(2, 1), (1, 2)], [0.0, 0.01, 0.1]
        )
    ]


def run_name(spec):
    architecture = "standard2" if spec["layers"] == 2 else "loop2"
    return f"w{spec['world']}-i{spec['initialization']}-{architecture}-wd{spec['weight_decay']:g}"


def config_path(phase):
    return Path(f"configs/grokking-dynamics-{phase}-v1.json")


def prepare(phase, lr):
    path = config_path(phase)
    if path.exists():
        raise FileExistsError(path)
    specs = specifications(phase, lr)
    world = build_world(specs[0])
    audit_world(world)
    initial_hashes = {}
    for spec in specs:
        key = f"i{spec['initialization']}-l{spec['layers']}"
        value = model_digest(construct(spec, "cpu"))
        if key in initial_hashes:
            assert initial_hashes[key] == value
        initial_hashes[key] = value
        assert data_digest(build_world(spec)) == data_digest(world)
    config = {
        "created_utc": utc(),
        "phase": phase,
        "specs": specs,
        "source": {p: file_hash(p) for p in SOURCE_FILES},
        "data_sha256": data_digest(world),
        "initial_model_sha256": initial_hashes,
        "strata_n": {key: len(rows) for key, rows in world.items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "primary": "Fixed 512k endpoint and within-trajectory post-fit familiar/strict changes",
        "policy": "All registered runs retained; no best checkpoint or test-loop selection. "
        "Constant post-warmup LR is independent of budget. "
        "Calibration reads train/atomic fit only. "
        "One development world; initializations and queries are not independent worlds.",
    }
    write_json(path, config)
    for p in SOURCE_FILES:
        target = ARTIFACTS / phase / "source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    shutil.copy2(path, ARTIFACTS / phase / "frozen-config.json")
    print(json.dumps({"phase": phase, "runs": len(specs), "strata": config["strata_n"]}))


def execute(path, gpus):
    config = json.loads(path.read_text())
    assert all(file_hash(p) == digest for p, digest in config["source"].items())
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)

    def launch(spec):
        folder = RESULTS / config["phase"] / run_name(spec)
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists() and (folder / "audit.json").exists():
            return {"run": run_name(spec), "skipped_complete": True}
        gpu = slots.get()
        try:
            commands = ["audit"] if (folder / "complete.json").exists() else ["run", "audit"]
            for command in commands:
                with (folder / f"{command}-process.log").open("a") as log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            command,
                            "--config",
                            str(path),
                            "--name",
                            run_name(spec),
                            "--device",
                            f"cuda:{gpu}",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                        check=False,
                    )
                status = {
                    "run": run_name(spec),
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

    started = time.perf_counter()
    launched = utc()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = list(pool.map(launch, config["specs"]))
    write_json(
        ARTIFACTS / config["phase"] / "execution.json",
        {
            "launched_utc": launched,
            "finished_utc": utc(),
            "seconds": time.perf_counter() - started,
            "gpus": gpus,
            "statuses": statuses,
        },
    )
    if any(status.get("returncode", 0) for status in statuses):
        raise RuntimeError("Failed attempts retained; inspect their logs")


def report(path):
    config = json.loads(path.read_text())
    endpoints, curves, weights, runs = [], [], [], []
    for spec in config["specs"]:
        name = run_name(spec)
        folder = RESULTS / config["phase"] / name
        result = json.loads((folder / "complete.json").read_text())
        check = json.loads((folder / "audit.json").read_text())
        assert check["passed"]
        history = json.loads((folder / "learning.json").read_text())
        fit = next(
            (
                row
                for row in history
                if all(
                    row["metrics"][task]["accuracy"] >= 0.99
                    for task in ["common_atomic", "anchor_atomic", "train_composite"]
                )
            ),
            None,
        )
        metadata = {
            "name": name,
            "initialization": spec["initialization"],
            "architecture": "standard2" if spec["layers"] == 2 else "loop2",
            "weight_decay": spec["weight_decay"],
        }
        for node in history:
            for task, metric in node["metrics"].items():
                curves.append(
                    {
                        **metadata,
                        "step": node["step"],
                        "task": task,
                        "lr": node["lr"],
                        "flops": node["estimated_training_flops"],
                        **metric,
                    }
                )
        end = history[-1]["metrics"]
        endpoints.append(
            {
                **metadata,
                "parameters": result["parameters"],
                "first_saved_joint_fit99": fit["step"] if fit else None,
                "atomic": end["common_atomic"]["accuracy"],
                "anchor": end["anchor_atomic"]["accuracy"],
                "train": end["train_composite"]["accuracy"],
                "familiar": end["familiar_test"]["accuracy"],
                "strict": end["strict_test"]["accuracy"],
                "familiar_at_fit": fit["metrics"]["familiar_test"]["accuracy"] if fit else None,
                "strict_at_fit": fit["metrics"]["strict_test"]["accuracy"] if fit else None,
            }
        )
        runs.append(result)
        if config["phase"] != "calibration":
            import torch

            previous = None
            for node in spec["checkpoint_nodes"]:
                state = torch.load(
                    folder / f"model-{node:06d}.pt", map_location="cpu", weights_only=False
                )["model"]
                for group, keys in {
                    "all": list(state),
                    "MLP": [k for k in state if ".mlp." in k],
                    "attention": [k for k in state if ".attention." in k],
                    "embedding": [k for k in state if k in ["token.weight", "position.weight"]],
                }.items():
                    squared = sum(float(state[k].square().sum()) for k in keys)
                    drift = (
                        None
                        if previous is None
                        else sum(float((state[k] - previous[k]).square().sum()) for k in keys)
                        ** 0.5
                    )
                    weights.append(
                        {
                            **metadata,
                            "step": node,
                            "group": group,
                            "weight_l2": squared**0.5,
                            "previous_checkpoint_drift_l2": drift,
                        }
                    )
                previous = state
    out = ARTIFACTS / config["phase"] / "report"
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("endpoints.csv", endpoints),
        ("learning.csv", curves),
        ("weights.csv", weights),
    ]:
        if rows:
            keys = list(dict.fromkeys(key for row in rows for key in row))
            with (out / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "created_utc": utc(),
        "phase": config["phase"],
        "runs": len(runs),
        "endpoints": endpoints,
        "all_audits_passed": True,
        "total_updates": sum(r["spec"]["steps"] for r in runs),
        "total_training_seconds": sum(r["training_seconds"] for r in runs),
        "supervised_tokens": sum(r["supervised_tokens"] for r in runs),
        "estimated_training_flops": sum(r["endpoint"]["estimated_training_flops"] for r in runs),
    }
    write_json(out / "summary.json", summary)
    if config["phase"] == "calibration":
        passed = all(row["first_saved_joint_fit99"] is not None for row in endpoints)
        write_json(
            ARTIFACTS / "calibration" / "decision.json",
            {
                "passed": passed,
                "lr": config["specs"][0]["lr"],
                "criterion": "Every arm reaches common/anchor atomic and train composition >=99% "
                "at a saved node by 16k; held-out accuracy is not used to choose LR.",
            },
        )
    else:
        plot(curves, weights, out)
    print(json.dumps(summary))


def plot(curves, weights, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    colors = {0.0: "#4477aa", 0.01: "#ee7733", 0.1: "#228833"}
    for i, architecture in enumerate(["standard2", "loop2"]):
        for j, task in enumerate(["train_composite", "familiar_test", "strict_test"]):
            ax = axes[i, j]
            for decay, color in colors.items():
                selected = [
                    r
                    for r in curves
                    if r["architecture"] == architecture
                    and r["weight_decay"] == decay
                    and r["task"] == task
                ]
                for init in sorted({r["initialization"] for r in selected}):
                    rows = [r for r in selected if r["initialization"] == init and r["step"]]
                    ax.plot(
                        [r["step"] for r in rows],
                        [100 * r["accuracy"] for r in rows],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                    )
                steps = sorted({r["step"] for r in selected if r["step"]})
                ax.plot(
                    steps,
                    [
                        100 * np.mean([r["accuracy"] for r in selected if r["step"] == s])
                        for s in steps
                    ],
                    color=color,
                    label=f"wd={decay:g}",
                )
            ax.set_xscale("log")
            ax.set_ylim(-2, 102)
            ax.set_title(f"{architecture}: {task}")
            ax.set_ylabel("Full-generation accuracy (%)")
            ax.set_xlabel("Updates")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "learning.png", dpi=170)
    fig.savefig(out / "learning.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, architecture in zip(axes, ["standard2", "loop2"], strict=True):
        for decay, color in colors.items():
            selected = [
                r
                for r in weights
                if r["architecture"] == architecture
                and r["weight_decay"] == decay
                and r["group"] == "MLP"
                and r["step"]
            ]
            for init in sorted({r["initialization"] for r in selected}):
                rows = [r for r in selected if r["initialization"] == init]
                ax.plot(
                    [r["step"] for r in rows],
                    [r["weight_l2"] for r in rows],
                    color=color,
                    alpha=0.5,
                    label=f"wd={decay:g}, init={init}",
                )
        ax.set_xscale("log")
        ax.set_title(architecture)
        ax.set_xlabel("Updates")
        ax.set_ylabel("MLP weight L2 norm")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "mlp-norm.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "execute", "run", "audit", "report"])
    parser.add_argument("--phase", choices=["calibration", "development"], default="development")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--name")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    path = args.config or config_path(args.phase)
    if args.command == "prepare":
        prepare(args.phase, args.lr)
    elif args.command == "execute":
        execute(path, [int(value) for value in args.gpus.split(",")])
    elif args.command == "report":
        report(path)
    else:
        config = json.loads(path.read_text())
        assert all(file_hash(p) == digest for p, digest in config["source"].items())
        spec = next(s for s in config["specs"] if run_name(s) == args.name)
        folder = RESULTS / config["phase"] / run_name(spec)
        if args.command == "run":
            train(spec, folder, config["source"], args.device)
        else:
            audit(folder, args.device)


if __name__ == "__main__":
    main()
