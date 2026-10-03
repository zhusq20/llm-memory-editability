"""Freeze data-only mechanism choices, then run fixed checkpoints in parallel."""

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

import numpy as np

from llm_memory_editability.depth_step import build_world, run_name
from llm_memory_editability.depth_step_mechanism import edit_cases, select_donors
from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import file_hash

ARTIFACTS = Path("docs/development-artifacts/depth-step-v1")
RESULTS = Path("results/depth-step-v1")
CONFIG = ARTIFACTS / "mechanism-config.json"
NEW_SOURCES = [
    "src/llm_memory_editability/depth_step_mechanism.py",
    "scripts/analyze_depth_step_mechanism.py",
    "scripts/execute_depth_step_mechanism.py",
    "scripts/report_depth_step_mechanism.py",
    "tests/test_depth_step_mechanism.py",
]


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_value(item) for item in value]
    return value


def prepare():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    parent = json.loads(Path("configs/depth-step-development-v1.json").read_text())
    world = build_world(parent["specs"][0])
    query_indices = np.arange(min(128, len(world["familiar_2"])))
    donors = select_donors(world, world["familiar_2"][query_indices])
    cases = edit_cases(world)
    frozen_cases = []
    for case in cases:
        frozen_cases.append(json_value(case))
    selected_path = ARTIFACTS / "mechanism-data-selection.json"
    write_json(
        selected_path,
        {
            "query_indices": query_indices.tolist(),
            "donors": {key: value.tolist() for key, value in donors.items()},
            "edit_cases": frozen_cases,
        },
    )
    states = []
    for layers, repeats, model in ((2, 1, "standard2"), (1, 2, "loop2")):
        spec = next(
            item
            for item in parent["specs"]
            if (item["layers"], item["repeats"]) == (layers, repeats)
        )
        for step in (8000, 32000, 64000):
            states.append(
                {
                    "name": f"{model}-s{step}",
                    "model": model,
                    "layers": layers,
                    "repeats": repeats,
                    "step": step,
                    "run_dir": str(RESULTS / run_name(spec)),
                    "checkpoint": f"model-{step:06d}.pt",
                }
            )
    sources = {**parent["source"], **{path: file_hash(path) for path in NEW_SOURCES}}
    config = {
        "created_utc": utc(),
        "phase": "single-world frozen development analysis",
        "states": states,
        "source": sources,
        "parent_config_sha256": file_hash("configs/depth-step-development-v1.json"),
        "design_sha256": parent["design_sha256"],
        "selection_sha256": file_hash(selected_path),
        "budget": {
            "trace_states": 6,
            "edit_states": 6,
            "independent_facts_per_state": 8,
            "arms_per_fact": 2,
            "updates_per_arm": 200,
            "edit_nodes": [0, 20, 100, 200],
            "total_edit_updates": 19200,
        },
        "edit_optimizer": "Adam lr=.01, no weight decay",
        "edit_loss": "0.5 new/old target full-token CE + 0.5 fixed replay full-token CE",
        "data_selection_uses_model_predictions": False,
    }
    write_json(CONFIG, config)
    for path in NEW_SOURCES:
        target = ARTIFACTS / "mechanism-source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(json.dumps({"states": len(states), "edit_cases": len(cases), "budget": config["budget"]}))


def execute(gpus):
    config = json.loads(CONFIG.read_text())
    for path, digest in config["source"].items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Frozen source changed: {path}")
    if file_hash(ARTIFACTS / "design.md") != config["design_sha256"]:
        raise RuntimeError("Frozen design changed")
    if file_hash(ARTIFACTS / "mechanism-data-selection.json") != config["selection_sha256"]:
        raise RuntimeError("Data selection changed")
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    started = time.perf_counter()

    def launch(job):
        state, mode = job
        gpu = slots.get()
        try:
            out = RESULTS / "mechanism" / state["name"] / mode
            command = [
                sys.executable,
                "scripts/analyze_depth_step_mechanism.py",
                "--run-dir",
                state["run_dir"],
                "--checkpoint",
                str(Path(state["run_dir"]) / state["checkpoint"]),
                "--out",
                str(out),
                "--device",
                f"cuda:{gpu}",
                "--mode",
                mode,
            ]
            logpath = RESULTS / "mechanism" / f"{state['name']}-{mode}.log"
            logpath.parent.mkdir(parents=True, exist_ok=True)
            began = time.perf_counter()
            with logpath.open("w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                )
            record = {
                "name": state["name"],
                "mode": mode,
                "gpu": gpu,
                "returncode": result.returncode,
                "wall_seconds": time.perf_counter() - began,
            }
            write_json(RESULTS / "mechanism" / f"{state['name']}-{mode}-status.json", record)
            print(json.dumps(record), flush=True)
            if result.returncode:
                raise RuntimeError(f"Mechanism failed: {state['name']} {mode}; see {logpath}")
            return record
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        records = list(
            pool.map(
                launch, [(state, mode) for state in config["states"] for mode in ("trace", "edit")]
            )
        )
    write_json(
        ARTIFACTS / "mechanism-execution.json",
        {"finished_utc": utc(), "wall_seconds": time.perf_counter() - started, "records": records},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "execute"])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    arguments = parser.parse_args()
    if arguments.stage == "prepare":
        prepare()
    else:
        execute([int(value) for value in arguments.gpus.split(",")])
