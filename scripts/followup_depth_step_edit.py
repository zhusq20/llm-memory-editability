"""Freeze a calibrated editor and evaluate reserved facts along training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.depth_step import build_world
from llm_memory_editability.depth_step_edit_calibration import (
    SOURCE_FILES,
    calibrated_edit_one,
    calibration_cases,
)
from llm_memory_editability.depth_step_mechanism import load_run
from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import file_hash

ARTIFACTS = Path("docs/development-artifacts/depth-step-v1/edit-followup")
RESULTS = Path("results/depth-step-v1/edit-followup")
CONFIG = ARTIFACTS / "config.json"


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [serializable(item) for item in value]
    return value


def prepare():
    if CONFIG.exists():
        raise FileExistsError(CONFIG)
    mechanism = json.loads(
        Path("docs/development-artifacts/depth-step-v1/mechanism-config.json").read_text()
    )
    train = json.loads(Path("configs/depth-step-development-v1.json").read_text())
    world = build_world(train["specs"][0])
    all_cases, _ = calibration_cases(world, n_cases=8)
    valid_lrs = []
    calibrations = []
    for name in ("standard2", "loop2"):
        folder = Path("results/depth-step-v1/edit-calibration") / f"{name}-s32000"
        calibration = json.loads((folder / "calibration-summary.json").read_text())
        passing = {
            lr
            for lr in (0.0001, 0.001, 0.003)
            if all(
                record["history"][-1]["calibration_operation_pass"]
                for record in calibration["records"]
                if record["lr"] == lr and record["arm"] == "edit"
            )
        }
        valid_lrs.append(passing)
        calibrations.append(
            {
                "model": name,
                "summary_sha256": file_hash(folder / "calibration-summary.json"),
                "config_sha256": file_hash(folder / "calibration-config.json"),
            }
        )
    common = set.intersection(*valid_lrs)
    if not common:
        raise RuntimeError("No common editor passed target/Kdev calibration")
    lr = min(common)
    source = {
        path: file_hash(path) for path in [*SOURCE_FILES, "scripts/followup_depth_step_edit.py"]
    }
    config = {
        "created_utc": utc(),
        "phase": "reserved-fact followup within one development world",
        "states": mechanism["states"],
        "lr": lr,
        "nodes": [0, 20, 100, 200],
        "calibrations": calibrations,
        "selection": (
            "Lowest lr passing E and Kdev>=95% on both 32k parents/calibration facts; "
            "U and D not used"
        ),
        "original_case_indices": list(range(2, 8)),
        "cases": serializable(all_cases[2:]),
        "source": source,
        "budget": {"branches": 72, "updates": 14400},
        "report": "All E/D/R/Kdev/U and D conditional on parent success; retain all failures",
        "limits": "Same world/checkpoints, six data-reserved facts; not new-world confirmation",
    }
    write_json(CONFIG, config)
    for path in source:
        target = ARTIFACTS / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(json.dumps({"selected_lr": lr, "budget": config["budget"]}))


def execute(state_name, device):
    config = json.loads(CONFIG.read_text())
    for path, digest in config["source"].items():
        if file_hash(path) != digest:
            raise RuntimeError(f"Frozen editor source changed: {path}")
    state = next(item for item in config["states"] if item["name"] == state_name)
    model, world, parent = load_run(
        state["run_dir"], Path(state["run_dir"]) / state["checkpoint"], device
    )
    all_cases, _ = calibration_cases(world, n_cases=8)
    cases = all_cases[2:]
    if serializable(cases) != config["cases"]:
        raise RuntimeError("Frozen reserved data changed")
    out = RESULTS / state_name
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for index, case in zip(config["original_case_indices"], cases, strict=True):
        for arm in ("edit", "review"):
            record, raw = calibrated_edit_one(model, case, world, device, arm, config["lr"])
            record["original_case_index"] = index
            for node in record["history"]:
                direct = "D_first_familiar_2"
                original = case["original_d_rows"][direct]
                before = raw[f"step0_{direct}_predictions"]
                parent_correct = (before[:, 0] == original[:, -1]) & (before[:, 1] == 1)
                after = raw[f"step{node['step']}_{direct}_predictions"]
                following = (after[:, 0] == case["tasks"][direct][:, -1]) & (after[:, 1] == 1)
                node["metrics"][direct]["parent_original_correct_n"] = int(parent_correct.sum())
                node["metrics"][direct]["conditional_on_parent_original_correct"] = (
                    float(following[parent_correct].mean()) if parent_correct.any() else None
                )
            np.savez_compressed(out / f"case{index:02d}-{arm}-raw.npz", **raw)
            write_json(out / f"case{index:02d}-{arm}.json", record)
            records.append(record)
    write_json(
        out / "summary.json",
        {"state": state, "parent": parent, "config_sha256": file_hash(CONFIG), "records": records},
    )
    print(json.dumps({"state": state_name, "branches": len(records), "done": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--state")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage == "prepare":
        prepare()
    else:
        execute(args.state, args.device)
