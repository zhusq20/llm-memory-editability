"""Confirm composition support by recurrence on new worlds, using frozen training code."""

from __future__ import annotations

import argparse
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
    construct,
    data_digest,
    file_hash,
    model_digest,
    role_coverage,
    run_name,
    train,
)
from llm_memory_editability.storage_composition import audit_world

ARTIFACTS = Path("docs/development-artifacts/latent-confirmation-v1")
RESULTS = Path("results/latent-confirmation-v1")
CONFIG = Path("configs/latent-confirmation-v1.json")
DESIGN = ARTIFACTS / "design.md"
SOURCES = [*SOURCE_FILES, "scripts/run_latent_confirmation.py", "tests/test_latent_confirmation.py"]


def specifications():
    base = {
        "heads_n": 1024,
        "bridges_n": 512,
        "tails_n": 128,
        "familiar_n": 64,
        "strict_n": 32,
        "holdout_fraction": 0.25,
        "low_extra": "anchors",
        "anchor_n": 32,
        "width": 128,
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
        "repeat_nodes": [32000, 128000],
        "checkpoint_nodes": [0, 32000, 128000],
    }
    return [
        {
            **base,
            "world": world,
            "initialization": initialization,
            "stream_seed": 742101 + world_index,
            "layers": layers,
            "repeats": repeats,
            "composition_count": count,
            "test_repeats": [1, 2, 3, 4] if layers == 1 else [1],
        }
        for world_index, world in enumerate((740101, 740102, 740103))
        for initialization, count, (layers, repeats) in itertools.product(
            (741101, 741102), (256, "all"), ((1, 1), (1, 2), (1, 3), (2, 1))
        )
    ]


def prepare():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    specs = specifications()
    source = {path: file_hash(path) for path in SOURCES}
    prior = json.loads(Path("configs/latent-scaling-v1.json").read_text())
    assert all(source[path] == digest for path, digest in prior["source"].items())
    initial = {}
    for spec in specs[:8]:
        key = (spec["layers"], spec["repeats"])
        initial[str(key)] = model_digest(construct(spec, "cpu"))
    assert len({initial[str((1, r))] for r in (1, 2, 3)}) == 1
    data = {}
    for world in sorted({s["world"] for s in specs}):
        worlds = {}
        for count in (256, "all"):
            spec = next(s for s in specs if s["world"] == world and s["composition_count"] == count)
            value = build_world(spec)
            audit_world(value)
            worlds[count] = value
            data[f"{world}:{count}"] = {
                "sha256": data_digest(value),
                "composition_examples": len(value["train_composite"]),
                "familiar_test_examples": len(value["familiar_test"]),
                "strict_test_examples": len(value["strict_test"]),
                "role_coverage": role_coverage(value),
            }
        for key in worlds[256]:
            if key != "train_composite":
                np.testing.assert_array_equal(worlds[256][key], worlds["all"][key])
        np.testing.assert_array_equal(
            worlds[256]["train_composite"], worlds["all"]["train_composite"][:256]
        )
    config = {
        "created_utc": utc(),
        "phase": "new-world fixed-comparison confirmation",
        "specs": specs,
        "source": source,
        "design_sha256": file_hash(DESIGN),
        "data": data,
        "prior_config_sha256": file_hash("configs/latent-scaling-v1.json"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "primary": "[(Loop R2 - single R1) all - (Loop R2 - single R1) 256] at 128k",
        "secondary": [
            "R3 support interaction",
            "standard2 - Loop R2 at each support level",
            "same-weight test recurrence at fixed 32k and 128k nodes",
        ],
        "analysis_unit": "Mean paired contrasts across two initializations within each world, "
        "then equal-weight mean across three worlds; publish every pair and denominator.",
        "limitations": "Support size also changes role coverage and per-example repetitions. "
        "Same updates/tokens are not same FLOPs. Standard2 changes parameters and the "
        "legacy_unique residual initialization scale. "
        "Fixed width128 does not confirm width effects.",
        "budget": {
            "runs": len(specs),
            "updates": sum(s["steps"] for s in specs),
            "supervised_tokens": sum(s["steps"] * s["batch_size"] * 23 // 3 for s in specs),
            "learning_nodes": sum(len(s["nodes"]) for s in specs),
        },
    }
    write_json(CONFIG, config)
    for path in SOURCES:
        target = ARTIFACTS / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(CONFIG, ARTIFACTS / "frozen-config.json")
    print(json.dumps({"budget": config["budget"], "data": data}), flush=True)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not all(file_hash(p) == h for p, h in config["source"].items()):
        raise RuntimeError("Experiment source differs from the frozen configuration")
    if file_hash(DESIGN) != config["design_sha256"]:
        raise RuntimeError("Design differs from the frozen configuration")
    return config


def preflight(config, device):
    spec = dict(config["specs"][1])
    spec.update(steps=8, nodes=[0, 8], repeat_nodes=[8], checkpoint_nodes=[0, 8])
    folder = RESULTS / "engineering-preflight"
    train(spec, folder, config["source"], device)
    checked = audit(folder, device)
    write_json(ARTIFACTS / "preflight.json", {"scope": "engineering only", **checked})
    print(json.dumps(checked), flush=True)


def execute(config_path, gpus):
    config = load_config(config_path)
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU slots must be nonempty and unique")
    preflight_result = json.loads((ARTIFACTS / "preflight.json").read_text())
    assert preflight_result["passed"]
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    launched, started = utc(), time.perf_counter()

    def launch(spec):
        name = run_name(spec)
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists() and (folder / "audit.json").exists():
            complete = json.loads((folder / "complete.json").read_text())
            checked = json.loads((folder / "audit.json").read_text())
            assert complete["spec"] == spec and complete["source"] == config["source"]
            assert checked["passed"]
            return {"run": name, "skipped_complete": True}
        gpu = slots.get()
        try:
            commands = ["audit"] if (folder / "complete.json").exists() else ["run", "audit"]
            stages = []
            for command in commands:
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
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
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                        check=False,
                    )
                status = {
                    "run": name,
                    "stage": command,
                    "gpu": gpu,
                    "returncode": process.returncode,
                    "utc": utc(),
                }
                stages.append(status)
                write_json(folder / f"{command}-process-status.json", status)
                print(json.dumps(status), flush=True)
                if process.returncode:
                    break
            return {"run": name, "stages": stages, "returncode": stages[-1]["returncode"]}
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = list(pool.map(launch, config["specs"]))
    write_json(
        ARTIFACTS / "execution.json",
        {
            "launched_utc": launched,
            "finished_utc": utc(),
            "seconds": time.perf_counter() - started,
            "gpus": gpus,
            "statuses": statuses,
        },
    )
    if any(s.get("returncode", 0) for s in statuses):
        raise RuntimeError("Failed attempts retained; inspect process logs before any retry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "run", "audit"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--name")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    config = load_config(args.config)
    if args.command == "preflight":
        preflight(config, args.device)
    elif args.command == "execute":
        execute(args.config, [int(g) for g in args.gpus.split(",")])
    else:
        spec = next(s for s in config["specs"] if run_name(s) == args.name)
        folder = RESULTS / args.name
        if args.command == "run":
            train(spec, folder, config["source"], args.device)
        else:
            print(json.dumps(audit(folder, args.device)), flush=True)


if __name__ == "__main__":
    main()
