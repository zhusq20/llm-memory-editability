"""Freeze, run and independently reload the representation intervention batch."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.latent_scaling import SOURCE_FILES
from llm_memory_editability.representation_alignment import audit, run_name, train
from llm_memory_editability.storage_composition import file_hash

ROOT = Path("results/representation-alignment-v1")
ARTIFACTS = Path("docs/development-artifacts/representation-alignment-v1")
CONFIG = Path("configs/representation-alignment-development-v1.json")


def specifications():
    base = dict(
        phase="development",
        world=770011,
        initialization=771011,
        stream_seed=772011,
        heads_n=256,
        bridges_n=128,
        tails_n=64,
        familiar_n=64,
        strict_n=32,
        anchor_n=32,
        holdout_fraction=0.25,
        low_extra="anchors",
        composition_count=256,
        width=128,
        heads=4,
        layers=1,
        repeats=2,
        dropout=0.0,
        batch_size=192,
        lr=0.001,
        weight_decay=0.01,
        warmup=200,
        schedule="cosine",
        min_lr_ratio=0.1,
        steps=8000,
        nodes=[0, 256, 1000, 2000, 4000, 8000],
    )
    arms = [
        ("baseline", 0.0, 0.0),
        ("bridge_ce", 0.3, 0.0),
        ("aligned_0.3", 0.3, 0.3),
        ("aligned_3.0", 0.3, 3.0),
    ]
    return [
        {**base, "arm": arm, "bridge_weight": beta, "alignment_weight": weight}
        for arm, beta, weight in arms
    ]


def freeze():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    sources = sorted(
        set(
            SOURCE_FILES
            + [
                "src/llm_memory_editability/representation_alignment.py",
                "scripts/run_representation_alignment.py",
                "tests/test_representation_alignment.py",
                "src/llm_memory_editability/experiment_tracking.py",
                "configs/experiment-tracking-defaults.json",
            ]
        )
    )
    config = {
        "created_utc": utc(),
        "specs": specifications(),
        "source": {p: file_hash(p) for p in sources},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(CONFIG, config)
    for p in sources:
        target = ARTIFACTS / "source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    shutil.copy2(CONFIG, ARTIFACTS / "frozen-development-config.json")
    shutil.copy2(ARTIFACTS / "design.md", ARTIFACTS / "frozen-design.md")
    print(
        json.dumps(
            {"runs": len(config["specs"]), "updates": sum(s["steps"] for s in config["specs"])}
        )
    )


def controller(config_path, gpu):
    config = json.loads(Path(config_path).read_text())
    assert all(file_hash(p) == digest for p, digest in config["source"].items())
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        ROOT / "controller-state.json", {"state": "running", "started_utc": utc(), "gpu": gpu}
    )
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, spec in enumerate(config["specs"]):
        out = ROOT / "runs" / run_name(spec)
        out.mkdir(parents=True, exist_ok=True)
        for action in ("worker", "audit"):
            done = out / ("complete.json" if action == "worker" else "audit.json")
            if done.exists():
                continue
            command = [
                sys.executable,
                "-u",
                __file__,
                action,
                "--config",
                str(config_path),
                "--index",
                str(index),
                "--gpu",
                str(gpu),
            ]
            with (out / f"{action}.log").open("a") as log:
                result = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                write_json(
                    out / "failure.json", {"action": action, "returncode": result.returncode}
                )
                write_json(
                    ROOT / "controller-state.json",
                    {"state": "finished_with_failures", "run": run_name(spec)},
                )
                raise SystemExit(result.returncode)
    write_json(ROOT / "controller-state.json", {"state": "complete", "finished_utc": utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "controller", "worker", "audit"])
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "controller":
        controller(args.config, args.gpu)
    else:
        import torch

        config = json.loads(args.config.read_text())
        spec = config["specs"][args.index]
        device = torch.device(f"cuda:{args.gpu}")
        out = ROOT / "runs" / run_name(spec)
        if args.action == "worker":
            train(spec, out, config["source"], device)
        else:
            torch.set_num_threads(1)
            audit(out, device)


if __name__ == "__main__":
    main()
