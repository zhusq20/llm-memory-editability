"""Freeze and run graph-selected MLP updates and pure-first-hop interventions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from run_grokking_dynamics import ARTIFACTS, RESULTS, config_path, run_name

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grokking_dynamics_mechanism import (
    PARAMETER,
    editor,
    safe_generate,
    select_cases,
    serialize_case,
    trace,
)
from llm_memory_editability.latent_scaling import build_world, construct, model_digest
from llm_memory_editability.storage_composition import file_hash

SOURCES = [
    "src/llm_memory_editability/grokking_dynamics_mechanism.py",
    "src/llm_memory_editability/depth_step_mechanism.py",
    "scripts/run_grokking_dynamics_mechanism.py",
    "tests/test_grokking_dynamics_mechanism.py",
]


def load_model(spec, phase, node, device):
    folder = RESULTS / phase / run_name(spec)
    checkpoint = folder / f"model-{node:06d}.pt"
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    assert state["step"] == node and state["spec"] == spec
    model = construct(spec, device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, {"checkpoint": str(checkpoint), "sha256": file_hash(checkpoint)}


def prepare(phase):
    pilot = phase == "calibration"
    train_config = json.loads(config_path(phase).read_text())
    specs = [s for s in train_config["specs"] if not pilot or s["weight_decay"] == 0.01]
    folder = ARTIFACTS / "mechanism" / phase
    path = folder / "config.json"
    if path.exists():
        raise FileExistsError(path)
    world = build_world(specs[0])
    cases = select_cases(world, n_per_group_role=1 if pilot else 2, calibration=pilot)
    matrix = []
    for spec in specs:
        for node in [16000] if pilot else [8000, 128000, 512000]:
            matrix.append({"spec": spec, "node": node})
    selected_lr = None
    if not pilot:
        selected_lr = json.loads((ARTIFACTS / "mechanism/calibration/decision.json").read_text())
        if not selected_lr["passed"]:
            raise ValueError("Editor calibration did not establish target and keep premises")
    source = {path: file_hash(path) for path in SOURCES}
    config = {
        "created_utc": utc(),
        "phase": phase,
        "source": source,
        "matrix": matrix,
        "cases": [serialize_case(case) for case in cases],
        "learning_rates": [0.0001, 0.0003, 0.001, 0.003] if pilot else [selected_lr["lr"]],
        "edit_nodes": [0, 200],
        "parameter": PARAMETER,
        "policy": "Targets ranked by affected query count then lexicographic fact, "
        "without predictions; fixed all-pool and parent-correct changed-answer subsets. "
        "Only pilot E and Kdev choose editor LR. All cases, stages and failures retained. "
        "Trace at first execution positions r1=3 and is=6; pure atomic donor r1=3/is=4. "
        "Donor has no target second relation or terminal answer.",
        "training_config_sha256": file_hash(config_path(phase)),
    }
    write_json(path, config)
    import shutil

    for path in SOURCES:
        target = folder / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(json.dumps({"phase": phase, "states": len(matrix), "cases": len(cases)}))


def state_name(entry):
    return f"{run_name(entry['spec'])}-t{entry['node']:06d}"


def run_state(config, entry, device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    assert all(file_hash(path) == sha for path, sha in config["source"].items())
    out = RESULTS / "mechanism" / config["phase"] / state_name(entry)
    if (out / "complete.json").exists():
        return
    model, source = load_model(entry["spec"], config["phase"], entry["node"], device)
    world = build_world(entry["spec"])
    cases = select_cases(
        world,
        n_per_group_role=1 if config["phase"] == "calibration" else 2,
        calibration=config["phase"] == "calibration",
    )
    assert [serialize_case(case) for case in cases] == config["cases"]
    for index, case in enumerate(cases):
        for lr in config["learning_rates"]:
            for arm in ["edit", "review"]:
                name = f"case{index:02d}-lr{lr:g}-{arm}"
                if (out / f"{name}.json").exists():
                    continue
                record, raw, final = editor(
                    model, case, device, arm, lr, tuple(config["edit_nodes"])
                )
                out.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / f"{name}.npz", **raw)
                torch.save(
                    {
                        "parameter": PARAMETER,
                        "weight": final,
                        "parent": source,
                        "final_model_sha256": record["final_model_sha256"],
                    },
                    out / f"{name}.pt",
                )
                write_json(out / f"{name}.json", {**record, "parent_checkpoint": source})
    if config["phase"] != "calibration" and not (out / "trace/trace.json").exists():
        trace(model, world, device, out / "trace")
    write_json(
        out / "complete.json", {"passed": True, "source": source, "entry": entry, "utc": utc()}
    )
    print(json.dumps({"state": state_name(entry), "complete": True}), flush=True)


def execute(config, gpus):
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    path = ARTIFACTS / "mechanism" / config["phase"] / "config.json"

    def launch(entry):
        parent_folder = RESULTS / config["phase"] / run_name(entry["spec"])
        while not (parent_folder / f"model-{entry['node']:06d}.pt").exists() or (
            json.loads((parent_folder / "status.json").read_text())["step"] < entry["node"]
        ):
            failure = parent_folder / "run-process-status.json"
            if failure.exists() and json.loads(failure.read_text())["returncode"]:
                raise RuntimeError("The parent training attempt failed; no checkpoint to analyze")
            time.sleep(2)
        gpu = slots.get()
        out = RESULTS / "mechanism" / config["phase"] / state_name(entry)
        out.mkdir(parents=True, exist_ok=True)
        try:
            for command in ["run", "audit"]:
                with (out / f"{command}-process.log").open("a") as log:
                    process = subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            command,
                            "--phase",
                            config["phase"],
                            "--name",
                            state_name(entry),
                            "--device",
                            f"cuda:{gpu}",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                        check=False,
                    )
                if process.returncode:
                    break
            result = {"state": state_name(entry), "gpu": gpu, "returncode": process.returncode}
            write_json(out / "process-status.json", result)
            print(json.dumps(result), flush=True)
            return result
        finally:
            slots.put(gpu)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = list(pool.map(launch, config["matrix"]))
    write_json(
        path.parent / "execution.json",
        {"seconds": time.perf_counter() - started, "statuses": statuses, "gpus": gpus},
    )
    if any(status["returncode"] for status in statuses):
        raise RuntimeError("Failed mechanism attempts retained")


def audit_state(config, entry, device):
    """A separate interpreter restores disk tensors and rescores every edited task."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    parent, _source = load_model(entry["spec"], config["phase"], entry["node"], device)
    out = RESULTS / "mechanism" / config["phase"] / state_name(entry)
    world = build_world(entry["spec"])
    cases = select_cases(
        world,
        n_per_group_role=1 if config["phase"] == "calibration" else 2,
        calibration=config["phase"] == "calibration",
    )
    case_lookup = {case["atomic_index"]: case for case in cases}
    branches, tasks_checked, max_nll = 0, 0, 0.0
    for path in sorted(out.glob("case*.json")):
        record = json.loads(path.read_text())
        assert model_digest(parent) == record["parent_model_sha256"]
        restored = construct(entry["spec"], device).eval()
        restored.load_state_dict(parent.state_dict())
        state = torch.load(path.with_suffix(".pt"), map_location=device, weights_only=False)
        assert state["parameter"] == PARAMETER
        with torch.no_grad():
            dict(restored.named_parameters())[PARAMETER].copy_(state["weight"])
        assert model_digest(restored) == record["final_model_sha256"]
        raw = np.load(path.with_suffix(".npz"))
        case = case_lookup[record["case"]["atomic_index"]]
        assert serialize_case(case) == record["case"]
        for task, rows in case["tasks"].items():
            _metric, prediction = safe_generate(restored, rows, device)
            for key, actual in prediction.items():
                saved = raw[f"step200_{task}_{key}"]
                if key == "answer_nll":
                    np.testing.assert_allclose(actual, saved, rtol=1e-5, atol=1e-5)
                    if actual.size:
                        max_nll = max(max_nll, float(np.abs(actual - saved).max()))
                else:
                    np.testing.assert_array_equal(actual, saved)
            tasks_checked += 1
        branches += 1
    expected = len(cases) * len(config["learning_rates"]) * 2
    assert branches == expected
    write_json(
        out / "independent-audit.json",
        {
            "passed": True,
            "branches": branches,
            "tasks": tasks_checked,
            "max_nll_difference": max_nll,
        },
    )


def report(config):
    records, trace_rows, branch_records = [], [], []
    for entry in config["matrix"]:
        out = RESULTS / "mechanism" / config["phase"] / state_name(entry)
        assert json.loads((out / "complete.json").read_text())["passed"]
        assert json.loads((out / "independent-audit.json").read_text())["passed"]
        spec = entry["spec"]
        metadata = {
            "name": run_name(spec),
            "checkpoint": entry["node"],
            "architecture": "standard2" if spec["layers"] == 2 else "loop2",
            "initialization": spec["initialization"],
            "weight_decay": spec["weight_decay"],
        }
        for path in sorted(out.glob("case*.json")):
            branch = json.loads(path.read_text())
            assert branch["exact_weight_reload_passed"]
            branch_records.append(branch)
            for node in branch["history"]:
                for task, metric in node["metrics"].items():
                    records.append(
                        {
                            **metadata,
                            "atomic_index": branch["case"]["atomic_index"],
                            "group": branch["case"]["group"],
                            "role": branch["case"]["role"],
                            "arm": branch["arm"],
                            "lr": branch["lr"],
                            "edit_step": node["step"],
                            "task": task,
                            **metric,
                        }
                    )
        if config["phase"] != "calibration":
            traced = json.loads((out / "trace/trace.json").read_text())
            assert traced["traced_forward_and_self_patch_passed"]
            trace_rows.extend({**metadata, **row} for row in traced["records"])
    folder = ARTIFACTS / "mechanism" / config["phase"]
    for filename, rows in [("edits.csv", records), ("trace.csv", trace_rows)]:
        if rows:
            with (folder / filename).open("w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
                )
                writer.writeheader()
                writer.writerows(rows)
    if config["phase"] == "calibration":
        decisions = []
        for lr in config["learning_rates"]:
            branches = [b for b in branch_records if b["lr"] == lr]
            checks = [b["history"][-1]["metrics"] for b in branches]
            passed = all(
                m["E_new" if b["arm"] == "edit" else "E_old"]["accuracy"] == 1
                and m["Kdev_atomic"]["accuracy"] >= 0.95
                for b, m in zip(branches, checks, strict=True)
            )
            decisions.append(
                {
                    "lr": lr,
                    "passed": passed,
                    "min_Kdev": min(m["Kdev_atomic"]["accuracy"] for m in checks),
                    "target_success_fraction": np.mean(
                        [
                            m["E_new" if b["arm"] == "edit" else "E_old"]["accuracy"]
                            for b, m in zip(branches, checks, strict=True)
                        ]
                    ),
                }
            )
        selected = next((d for d in decisions if d["passed"]), None)
        write_json(
            folder / "decision.json",
            {
                "passed": selected is not None,
                "lr": selected["lr"] if selected else None,
                "candidates": decisions,
                "selection": "Lowest shared LR using E and Kdev only",
            },
        )
    summary = {
        "created_utc": utc(),
        "states": len(config["matrix"]),
        "branches": len(branch_records),
        "updates": len(branch_records) * 200,
        "all_exact_weight_reloads_passed": True,
        "trace_cells": len(trace_rows),
        "branch_seconds": sum(b["seconds"] for b in branch_records),
    }
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "execute", "run", "audit", "report"])
    parser.add_argument("--phase", choices=["calibration", "development"], default="development")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--name")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.phase)
        return
    config = json.loads((ARTIFACTS / "mechanism" / args.phase / "config.json").read_text())
    if args.command == "execute":
        execute(config, [int(value) for value in args.gpus.split(",")])
    elif args.command == "report":
        report(config)
    else:
        entry = next(e for e in config["matrix"] if state_name(e) == args.name)
        if args.command == "run":
            run_state(config, entry, args.device)
        else:
            audit_state(config, entry, args.device)


if __name__ == "__main__":
    main()
