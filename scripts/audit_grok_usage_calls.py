#!/usr/bin/env python3
"""Independently reload completed usage endpoints and audit autonomous calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_metrics(expected, actual, path="metrics"):
    """Compare every field, including nested per-hop lists and semantic labels."""
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [dict(path=path, error="metric keys differ")]
        for key in expected:
            errors.extend(compare_metrics(expected[key], actual[key], f"{path}.{key}"))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [dict(path=path, error="metric list shape differs")]
        for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
            errors.extend(compare_metrics(left, right, f"{path}[{i}]"))
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isclose(
            expected, actual, atol=1e-12, rtol=0, equal_nan=False
        ):
            errors.append(dict(path=path, expected=expected, actual=actual))
    elif expected != actual:
        errors.append(dict(path=path, expected=expected, actual=actual))
    return errors


def compare_arrays(saved, recomputed):
    expected_keys = {key for key in saved if key.startswith("autonomous_")}
    actual_keys = {"autonomous_" + key for key in recomputed}
    if expected_keys != actual_keys:
        return [], [
            dict(
                error="autonomous array keys differ",
                expected=sorted(expected_keys),
                actual=sorted(actual_keys),
            )
        ]
    checks, errors = [], []
    for key in sorted(expected_keys):
        expected, actual = saved[key], recomputed[key.removeprefix("autonomous_")]
        same_shape = expected.shape == actual.shape
        match = same_shape and np.array_equal(expected, actual)
        record = dict(
            key=key,
            shape=list(actual.shape),
            dtype=str(actual.dtype),
            entries=int(actual.size),
            exact_match=bool(match),
            mismatches=int(np.count_nonzero(expected != actual)) if same_shape else None,
        )
        checks.append(record)
        if not match:
            errors.append(record)
    return checks, errors


def inspect_runs(config, config_path):
    lock_path = ROOT / config["source_lock"]
    lock = read_json(lock_path)
    for relative, expected in lock["files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Execution source changed: {relative}")
    if str(config_path.relative_to(ROOT)) not in lock["files"]:
        raise ValueError("The supplied configuration is not covered by the execution lock")
    runs = []
    for name, settings in config["runs"].items():
        spec = {**config["base"], **settings}
        folder = ROOT / config["output_root"] / spec["phase"] / name
        if not (folder / "complete.json").exists():
            raise FileNotFoundError(f"Run is not complete: {name}")
        metadata = read_json(folder / "metadata.json")
        complete = read_json(folder / "complete.json")
        learning = read_json(folder / "learning.json")
        weights = folder / f"weights-{spec['steps']:07d}.pt"
        predictions = folder / f"predictions-{spec['steps']:07d}.npz"
        if metadata["spec"] != spec or metadata["source_lock"] != lock:
            raise ValueError(f"Archived source/specification mismatch: {name}")
        if complete["spec"] != spec or learning[-1]["step"] != spec["steps"]:
            raise ValueError(f"Endpoint specification/step mismatch: {name}")
        if complete["endpoint"] != learning[-1]:
            raise ValueError(f"Complete marker and final learning node disagree: {name}")
        if not weights.exists() or not predictions.exists():
            raise FileNotFoundError(f"Missing immutable endpoint files: {name}")
        runs.append(
            dict(
                name=name,
                spec=spec,
                folder=folder,
                weights=weights,
                predictions=predictions,
                endpoint=learning[-1],
            )
        )
    return lock, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-usage-development-v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.025)
    parser.add_argument("--output")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = read_json(config_path)
    # Require all requested endpoints and source checks before any CUDA allocation.
    lock, runs = inspect_runs(config, config_path)
    phases = {run["spec"]["phase"] for run in runs}
    if len(phases) != 1:
        raise ValueError("Audit one registered phase per invocation")
    phase = next(iter(phases))
    output = (
        Path(args.output)
        if args.output
        else (ROOT / "docs/development-artifacts/grok-usage-v1" / f"{phase}-autonomous-audit.json")
    )
    if output.exists():
        raise FileExistsError(f"Audit output already exists; retain it: {output}")
    # Imports use only the scientific implementations verified above.
    from llm_memory_editability.grok_multihop import autonomous_calls
    from llm_memory_editability.grok_usage_data import build_arm_world, build_experiment
    from llm_memory_editability.grok_usage_train import make_model

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Scientific endpoint audit requires the original CUDA FP32/TF32 precision")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.cuda.reset_peak_memory_stats(device)
    records = []
    for run in runs:
        spec, folder = run["spec"], run["folder"]
        with np.load(folder / "world.npz") as stored:
            atomic = stored["atomic"].copy()
            rows = stored["eval_test_composite"].copy()
        world_meta = read_json(folder / "world-metadata.json")
        rebuilt = build_arm_world(build_experiment(spec), spec["arm"])
        if world_meta != rebuilt["metadata"]:
            raise ValueError(f"Data metadata mismatch: {run['name']}")
        if not np.array_equal(atomic, rebuilt["atomic"]) or not np.array_equal(
            rows, rebuilt["evaluation"]["test_composite"]
        ):
            raise ValueError(f"Saved data mismatch: {run['name']}")
        state = torch.load(run["weights"], map_location="cpu", weights_only=False)
        if state["spec"] != spec or state["step"] != spec["steps"]:
            raise ValueError(f"Saved weight specification mismatch: {run['name']}")
        model, _ = make_model(spec, device)
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("Expected FP32 weights")
        world = dict(atomic=atomic, metadata=world_meta)
        # This is exactly the training evaluator's call: padded length 4,
        # default per-hop physical batch 1024, no autocast and no oracle bridge.
        actual_metrics, actual_arrays = autonomous_calls(model, rows, world, device, 4)
        with np.load(run["predictions"]) as saved:
            array_checks, errors = compare_arrays(saved, actual_arrays)
        metric_errors = compare_metrics(
            run["endpoint"]["metrics"]["autonomous_calls"], actual_metrics
        )
        errors.extend(metric_errors)
        record = dict(
            run_id=run["name"],
            step=spec["steps"],
            world_seed=spec["world_seed"],
            arm=spec["arm"],
            n_queries=len(rows),
            array_checks=array_checks,
            expected_metrics=run["endpoint"]["metrics"]["autonomous_calls"],
            recomputed_metrics=actual_metrics,
            errors=errors,
            passed=not errors,
            checkpoint=str(run["weights"].relative_to(ROOT)),
            checkpoint_sha256=sha256(run["weights"]),
            world_sha256=sha256(folder / "world.npz"),
            predictions_sha256=sha256(run["predictions"]),
            learning_sha256=sha256(folder / "learning.json"),
        )
        records.append(record)
        print(
            json.dumps(
                dict(
                    run_id=record["run_id"],
                    passed=record["passed"],
                    arrays=len(array_checks),
                    n_queries=len(rows),
                )
            ),
            flush=True,
        )
        del model, state
    report = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        phase=phase,
        config=str(config_path.relative_to(ROOT)),
        config_sha256=sha256(config_path),
        audit_script_sha256=sha256(__file__),
        verified_execution_lock=lock,
        device=str(device),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(device),
        shared_gpu=True,
        precision="FP32 weights and activations, TF32 enabled, no autocast",
        physical_batch_size=1024,
        padded_length=4,
        float_metric_atol=1e-12,
        memory_fraction=args.memory_fraction,
        max_allocated_bytes=torch.cuda.max_memory_allocated(device),
        expected_runs=len(config["runs"]),
        audited_runs=len(records),
        exact_array_entries=sum(c["entries"] for r in records for c in r["array_checks"]),
        passed=all(r["passed"] for r in records),
        records=records,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Autonomous endpoint audit found discrepancies; see retained report")


if __name__ == "__main__":
    main()
