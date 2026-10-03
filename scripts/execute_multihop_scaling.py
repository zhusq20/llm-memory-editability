"""Freeze and execute separate development/confirmation multihop scaling phases."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.latent_scaling import model_digest
from llm_memory_editability.multihop_scaling import (
    ARCHITECTURES,
    SOURCE_FILES,
    SUPPORT_LEVELS,
    build_world,
    construct,
    run_name,
)
from llm_memory_editability.storage_composition import file_hash

ARTIFACTS = Path("docs/development-artifacts/multihop-scaling-v1")
RESULTS = Path("results/multihop-scaling-v1")
DESIGN = ARTIFACTS / "design.md"


def specifications(phase="development"):
    if phase not in ("development", "confirmation"):
        raise ValueError("Unknown phase")
    worlds = [761011] if phase == "development" else [761101, 761102, 761103]
    initializations = [762011] if phase == "development" else [762101, 762102]
    base = {
        "entities": 64,
        "relations": 4,
        "degree": 4,
        "id_fraction": 0.75,
        "id_test_fraction": 0.2,
        "evaluation_size": 512,
        "heads": 4,
        "dropout": 0.0,
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
    return [
        {
            **base,
            "world_seed": world,
            "initialization": initial,
            "stream_seed": world + 2000,
            "width": width,
            "phi": phi,
            "layers": layers,
            "repeats": repeats,
        }
        for world in worlds
        for initial in initializations
        for width in (128, 256)
        for phi in SUPPORT_LEVELS
        for layers, repeats in ARCHITECTURES
    ]


def prepare(
    phase="development", config_path=None, artifacts=ARTIFACTS, results=RESULTS, design=DESIGN
):
    import numpy as np
    import torch

    config_path = Path(config_path or f"configs/multihop-scaling-{phase}-v1.json")
    artifacts, results, design = Path(artifacts), Path(results), Path(design)
    phase_artifacts, phase_results = artifacts / phase, results / phase
    if config_path.exists() or phase_artifacts.exists():
        raise FileExistsError("Do not overwrite a frozen phase: " + str(config_path))
    if phase == "confirmation":
        readiness = artifacts / "development-readiness.json"
        if not readiness.exists() or not json.loads(readiness.read_text()).get("passed"):
            raise RuntimeError("Confirmation requires development-readiness.json from development")
    if not design.is_file():
        raise FileNotFoundError(design)
    specs = specifications(phase)
    source = {path: file_hash(path) for path in SOURCE_FILES}
    worlds, initial_models = {}, {}
    torch.set_num_threads(1)
    for spec in specs:
        data_key = f"w{spec['world_seed']}-phi{spec['phi']:g}"
        if data_key not in worlds:
            world = build_world(spec)
            worlds[data_key] = {
                "metadata": world["metadata"],
                "counts": {key: len(world[key]) for key in world if key != "metadata"},
            }
        metadata = worlds[data_key]["metadata"]
        spec["frozen_data_sha256"] = metadata["dataset_sha256"]
        spec["common_evaluation_sha256"] = metadata["common_evaluation_sha256"]
        init_key = (
            f"i{spec['initialization']}-d{spec['width']}-l{spec['layers']}-r{spec['repeats']}"
        )
        if init_key not in initial_models:
            model = construct(spec, "cpu")
            initial_models[init_key] = {
                "initialization": spec["initialization"],
                "width": spec["width"],
                "layers": spec["layers"],
                "repeats": spec["repeats"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_model_sha256": model_digest(model),
            }
        spec["initial_model_sha256"] = initial_models[init_key]["initial_model_sha256"]
    config = {
        "created_utc": utc(),
        "phase": phase,
        "specs": specs,
        "source": source,
        "design": str(design),
        "design_sha256": file_hash(design),
        "artifacts": str(phase_artifacts),
        "results": str(phase_results),
        "worlds": worlds,
        "initial_models": initial_models,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "primary": "64k familiar/strict full answer+EOS by hop, with atomic premises",
        "analysis_unit": (
            "One independent development world"
            if phase == "development"
            else "Three worlds; average two paired initializations within each"
        ),
        "budget": {
            "runs": len(specs),
            "updates": sum(s["steps"] for s in specs),
            "learning_nodes": sum(len(s["nodes"]) for s in specs),
        },
    }
    # Exclusive creation protects a phase even when two preparation commands race.
    phase_artifacts.mkdir(parents=True, exist_ok=False)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("x") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    for path in SOURCE_FILES:
        destination = phase_artifacts / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if file_hash(destination) != source[path]:
            raise RuntimeError("Source changed during freezing: " + path)
    shutil.copy2(config_path, phase_artifacts / "frozen-config.json")
    shutil.copy2(design, phase_artifacts / "frozen-design.md")
    print(
        json.dumps(
            {
                "config": str(config_path),
                "budget": config["budget"],
                "worlds": {key: value["counts"] for key, value in worlds.items()},
            }
        ),
        flush=True,
    )
    return config


def check_config(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    for path, digest in config["source"].items():
        if file_hash(path) != digest:
            raise RuntimeError("Frozen source changed: " + path)
        if file_hash(Path(config["artifacts"]) / "source" / path) != digest:
            raise RuntimeError("Source snapshot changed: " + path)
    if file_hash(config["design"]) != config["design_sha256"]:
        raise RuntimeError("Frozen design changed")
    if file_hash(config_path) != file_hash(Path(config["artifacts"]) / "frozen-config.json"):
        raise RuntimeError("Frozen config changed")
    return config


def run_one(spec, source, folder, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "input-spec.json", {"spec": spec, "source": source})
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    statuses = []
    for stage in ("train", "audit"):
        command = [
            sys.executable,
            "scripts/run_multihop_scaling.py",
            stage,
            "--out",
            str(folder),
            "--device",
            device,
        ]
        if stage == "train":
            command.extend(["--spec-file", str(folder / "input-spec.json")])
        started = time.perf_counter()
        try:
            with (folder / f"{stage}-process.log").open("x") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            status = {
                "stage": stage,
                "device": device,
                "returncode": result.returncode,
                "seconds": time.perf_counter() - started,
            }
        except Exception:
            status = {
                "stage": stage,
                "device": device,
                "returncode": None,
                "seconds": time.perf_counter() - started,
                "error": traceback.format_exc(),
            }
        statuses.append(status)
        write_json(folder / f"{stage}-process-status.json", status)
        if status["returncode"] != 0:
            return {"run": folder.name, "passed": False, "stages": statuses}
    return {"run": folder.name, "passed": True, "stages": statuses}


def _queue(specs, source, results, devices):
    slots = queue.Queue()
    for device in devices:
        slots.put(device)

    def launch(spec):
        device = slots.get()
        try:
            record = run_one(spec, source, Path(results) / run_name(spec), device)
        except Exception:
            record = {"run": run_name(spec), "passed": False, "error": traceback.format_exc()}
        finally:
            slots.put(device)
        print(json.dumps(record), flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(launch, spec) for spec in specs]
        for future in as_completed(futures):
            records.append(future.result())
    return records


def preflight(config_path, devices, attempt=None):
    config = check_config(config_path)
    artifacts = Path(config["artifacts"])
    name = "preflight" + ("-" + attempt if attempt else "")
    destination = artifacts / (name + ".json")
    results = Path(config["results"]) / name
    if destination.exists() or results.exists():
        raise FileExistsError("Preserve preflight attempt: " + name)
    selected = [(128, 1.0, 2, 1), (128, 4.0, 1, 2), (256, 1.0, 1, 4), (256, 4.0, 4, 1)]
    specs = []
    for width, phi, layers, repeats in selected:
        spec = next(
            s
            for s in config["specs"]
            if (s["width"], s["phi"], s["layers"], s["repeats"]) == (width, phi, layers, repeats)
        ).copy()
        spec.update(steps=8, nodes=[0, 8], checkpoint_nodes=[0, 8])
        specs.append(spec)
    records = _queue(specs, config["source"], results, devices)
    passed = all(record["passed"] for record in records)
    record = {
        "scope": "8-step engineering only; no scientific model selection",
        "passed": passed,
        "config_sha256": file_hash(config_path),
        "records": records,
        "utc": utc(),
        "devices": devices,
    }
    write_json(destination, record)
    if not passed:
        raise RuntimeError("Engineering preflight failed; all attempts retained")
    return record


def execute(config_path, devices, preflight_path=None, attempt=None):
    config = check_config(config_path)
    artifacts = Path(config["artifacts"])
    preflight_path = Path(preflight_path or artifacts / "preflight.json")
    engineering = json.loads(preflight_path.read_text())
    if not engineering["passed"] or engineering["config_sha256"] != file_hash(config_path):
        raise RuntimeError("Passing engineering preflight for this frozen config is required")
    if any(not device.startswith("cuda:") for device in devices):
        raise ValueError("Scientific training requires explicit GPU slots")
    name = "execution" + ("-" + attempt if attempt else "")
    launch_path, execution_path = artifacts / (name + "-launch.json"), artifacts / (name + ".json")
    results = (
        Path(config["results"]) / ("attempt-" + attempt) if attempt else Path(config["results"])
    )
    if launch_path.exists() or execution_path.exists():
        raise FileExistsError("Preserve execution attempt: " + name)
    collisions = [
        str(results / run_name(spec))
        for spec in config["specs"]
        if (results / run_name(spec)).exists()
    ]
    if collisions:
        raise FileExistsError("Do not overwrite existing runs: " + ", ".join(collisions))
    started = time.perf_counter()
    with launch_path.open("x") as handle:
        json.dump(
            {
                "started_utc": utc(),
                "devices": devices,
                "results": str(results),
                "config_sha256": file_hash(config_path),
                "runs": len(config["specs"]),
                "preflight": str(preflight_path),
            },
            handle,
            indent=2,
        )
    records = _queue(config["specs"], config["source"], results, devices)
    record = {
        "finished_utc": utc(),
        "wall_seconds": time.perf_counter() - started,
        "passed": all(r["passed"] for r in records),
        "records": records,
    }
    write_json(execution_path, record)
    if not record["passed"]:
        raise RuntimeError("Training/audit failures retained in execution record")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "execute"))
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--design", type=Path, default=DESIGN)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--device", choices=("cpu",))
    parser.add_argument("--preflight-file", type=Path)
    parser.add_argument("--attempt")
    args = parser.parse_args()
    if args.attempt and (not args.attempt.replace("-", "").replace("_", "").isalnum()):
        parser.error("--attempt must be alphanumeric with optional '-' or '_'")
    config = args.config or Path(f"configs/multihop-scaling-{args.phase}-v1.json")
    devices = ["cpu"] if args.device else [f"cuda:{int(gpu)}" for gpu in args.gpus.split(",")]
    if not devices or len(set(devices)) != len(devices):
        parser.error("GPU slots must be nonempty and distinct")
    if args.stage == "prepare":
        prepare(args.phase, config, args.artifacts, args.results, args.design)
    elif args.stage == "preflight":
        preflight(config, devices, args.attempt)
    else:
        execute(config, devices, args.preflight_file, args.attempt)


if __name__ == "__main__":
    main()
