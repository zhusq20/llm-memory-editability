"""Freeze and execute the single-world depth/time development comparison."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import file_hash

CONFIG = Path("configs/depth-step-development-v1.json")
ARTIFACTS = Path("docs/development-artifacts/depth-step-v1")
RESULTS = Path("results/depth-step-v1")
DESIGN = ARTIFACTS / "design.md"
SOURCES = [
    "src/llm_memory_editability/depth_step.py",
    "src/llm_memory_editability/bios_model.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_depth_data.py",
    "src/llm_memory_editability/grok_multihop_data.py",
    "src/llm_memory_editability/grok_loop_data.py",
    "src/llm_memory_editability/grok_loop_model.py",
    "src/llm_memory_editability/storage_composition.py",
    "src/llm_memory_editability/storage_frontier.py",
    "src/llm_memory_editability/latent_scaling.py",
    "src/llm_memory_editability/text_pretrain.py",
    "scripts/run_depth_step.py",
    "scripts/execute_depth_step.py",
    "scripts/report_depth_step.py",
    "tests/test_depth_step.py",
    "tests/test_depth_step_report.py",
]


def specifications():
    base = {
        "world_seed": 751011,
        "entities": 64,
        "relations": 4,
        "degree": 4,
        "phi": 4.0,
        "id_fraction": 0.75,
        "id_test_fraction": 0.2,
        "evaluation_size": 512,
        "width": 128,
        "heads": 4,
        "dropout": 0.0,
        "initialization": 752011,
        "stream_seed": 753011,
        "model_initialization": "scaled_effective",
        "batch_size": 128,
        "steps": 64000,
        "nodes": [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
        "checkpoint_nodes": [0, 8000, 32000, 64000],
        "lr": 0.001,
        "weight_decay": 0.01,
        "warmup": 200,
        "schedule": "cosine",
        "min_lr_ratio": 0.1,
        "clip": 1.0,
    }
    architectures = [(layers, 1) for layers in (1, 2, 3, 4, 6)]
    architectures += [(1, repeats) for repeats in (2, 3, 4, 6)]
    return [{**base, "layers": layers, "repeats": repeats} for layers, repeats in architectures]


def prepare():
    from llm_memory_editability.depth_step import build_world, construct

    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    specs = specifications()
    world = build_world(specs[0])
    for spec in specs:
        spec["frozen_data_sha256"] = world["metadata"]["dataset_sha256"]
    data = {key: len(value) for key, value in world.items() if key != "metadata"}
    source = {path: file_hash(path) for path in SOURCES}
    initial_models = []
    for spec in specs:
        model = construct(spec, "cpu")
        from llm_memory_editability.latent_scaling import model_digest

        initial_models.append(
            {
                "layers": spec["layers"],
                "repeats": spec["repeats"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_model_sha256": model_digest(model),
            }
        )
    config = {
        "created_utc": utc(),
        "phase": "single-world development; no new-world confirmation",
        "specs": specs,
        "source": source,
        "design_sha256": file_hash(DESIGN),
        "data_sizes": data,
        "metadata": world["metadata"],
        "initial_models": initial_models,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "primary": "Fixed-endpoint familiar/strict accuracies for k=2/3/4 and all trajectories",
        "analysis_unit": (
            "One development world and one initialization; "
            "layers/steps/queries are repeated measurements"
        ),
        "budget": {
            "runs": len(specs),
            "updates": sum(s["steps"] for s in specs),
            "learning_nodes": sum(len(s["nodes"]) for s in specs),
        },
    }
    write_json(CONFIG, config)
    for path in SOURCES:
        dest = ARTIFACTS / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    shutil.copy2(CONFIG, ARTIFACTS / "frozen-config.json")
    print(json.dumps({"data_sizes": data, "budget": config["budget"]}), flush=True)


def check_config():
    config = json.loads(CONFIG.read_text())
    for path, digest in config["source"].items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Frozen source changed: {path}")
    if file_hash(DESIGN) != config["design_sha256"]:
        raise RuntimeError("Frozen design changed")
    return config


def run_one(spec, source, folder, gpu):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        raise FileExistsError(folder)
    write_json(folder / "input-spec.json", {"spec": spec, "source": source})
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    statuses = []
    for stage in ("train", "audit"):
        command = [
            sys.executable,
            "scripts/run_depth_step.py",
            stage,
            "--out",
            str(folder),
            "--device",
            f"cuda:{gpu}",
        ]
        if stage == "train":
            command.extend(["--spec-file", str(folder / "input-spec.json")])
        started = time.perf_counter()
        with (folder / f"{stage}-process.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        status = {
            "stage": stage,
            "gpu": gpu,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - started,
        }
        statuses.append(status)
        write_json(folder / f"{stage}-process-status.json", status)
        if result.returncode:
            raise RuntimeError(f"{stage} failed: {folder}")
    return {"run": folder.name, "stages": statuses}


def preflight(gpu):
    config = check_config()
    records = []
    for layers, repeats in ((2, 1), (1, 2)):
        spec = next(
            s for s in config["specs"] if (s["layers"], s["repeats"]) == (layers, repeats)
        ).copy()
        spec.update(steps=8, nodes=[0, 8], checkpoint_nodes=[0, 8])
        records.append(
            run_one(spec, config["source"], RESULTS / f"engineering-l{layers}-r{repeats}", gpu)
        )
    write_json(
        ARTIFACTS / "preflight.json",
        {
            "scope": "8-step engineering, no performance selection",
            "passed": True,
            "records": records,
        },
    )


def execute(gpus):
    from llm_memory_editability.depth_step import run_name

    config = check_config()
    if not json.loads((ARTIFACTS / "preflight.json").read_text())["passed"]:
        raise RuntimeError("Engineering preflight required")
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU slots must be nonempty and distinct")
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    records = []
    started = time.perf_counter()
    write_json(
        ARTIFACTS / "launch.json",
        {
            "started_utc": utc(),
            "gpus": gpus,
            "config_sha256": file_hash(CONFIG),
            "runs": len(config["specs"]),
        },
    )

    def launch(spec):
        gpu = slots.get()
        try:
            record = run_one(spec, config["source"], RESULTS / run_name(spec), gpu)
            print(json.dumps(record), flush=True)
            return record
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for record in pool.map(launch, config["specs"]):
            records.append(record)
    write_json(
        ARTIFACTS / "execution.json",
        {"finished_utc": utc(), "wall_seconds": time.perf_counter() - started, "records": records},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight", "execute"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpus.split(",")]
    if args.stage == "prepare":
        prepare()
    elif args.stage == "preflight":
        preflight(gpus[0])
    else:
        execute(gpus)


if __name__ == "__main__":
    main()
