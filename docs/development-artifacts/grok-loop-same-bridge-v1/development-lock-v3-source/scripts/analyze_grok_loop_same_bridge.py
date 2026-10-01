#!/usr/bin/env python3
"""Freeze and execute the two-hop same-bridge follow-up on existing endpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from analyze_grok_loop_mechanism import digest, load_frozen_model, load_source_world

from llm_memory_editability.grok_depth import utc
from llm_memory_editability.grok_depth_bridge import environment, write_json
from llm_memory_editability.grok_loop_same_bridge import (
    actual_exposure_counts,
    select_same_bridge_donors,
)
from llm_memory_editability.grok_loop_same_bridge_eval import evaluate_same_bridge

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/grok-loop-same-bridge-v1.json"
ARTIFACT = ROOT / "docs/development-artifacts/grok-loop-same-bridge-v1"
SOURCES = (
    "configs/grok-loop-same-bridge-v1.json",
    "docs/development-artifacts/grok-loop-same-bridge-v1/preregistration.md",
    "docs/development-artifacts/grok-loop-same-bridge-v1/execution-amendment-v2.md",
    "docs/development-artifacts/grok-loop-same-bridge-v1/execution-amendment-v3.md",
    "src/llm_memory_editability/grok_loop_same_bridge.py",
    "src/llm_memory_editability/grok_loop_same_bridge_eval.py",
    "src/llm_memory_editability/grok_loop_mechanism.py",
    "src/llm_memory_editability/grok_multihop.py",
    "src/llm_memory_editability/grok_multihop_data.py",
    "src/llm_memory_editability/grok_loop_data.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_depth_bridge.py",
    "scripts/analyze_grok_loop_mechanism.py",
    "scripts/analyze_grok_loop_same_bridge.py",
    "tests/test_grok_loop_same_bridge.py",
    "tests/test_grok_loop_same_bridge_eval.py",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name: digest(ROOT / name) for name in SOURCES}


def registered_runs(config, phase):
    original = read_json(ROOT / config["historical_configs"][phase])
    rows = []
    for run, overrides in original["runs"].items():
        spec = {**original["base"], **overrides}
        if spec["hops"] != 2:
            continue
        directory = ROOT / original["output_root"] / spec["phase"] / run
        if phase == "development":
            directory = ROOT / original["output_root"] / "development" / run
        rows.append({"run": run, "directory": str(directory), "spec": spec})
    expected = config["expected_runs"][phase]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {phase} endpoints, got {len(rows)}")
    return rows


def input_hashes(directory, step):
    names = (
        "metadata.json",
        "complete.json",
        "world.npz",
        "world-metadata.json",
        "data-audit.json",
        f"weights-{step:07d}.pt",
        f"predictions-{step:07d}.npz",
    )
    return {name: digest(Path(directory) / name) for name in names}


def verify_lock(lock, phase, config):
    if lock["phase"] != phase or lock["config_sha256"] != digest(CONFIG):
        raise ValueError("Phase/config disagrees with frozen follow-up")
    if lock["analysis_source_hashes"] != source_hashes():
        raise ValueError("Analysis source changed after freeze")
    expected = registered_runs(config, phase)
    if [r["run"] for r in expected] != [r["run"] for r in lock["runs"]]:
        raise ValueError("Registered matrix changed")
    return expected


def freeze(config, phase):
    path = ARTIFACT / config["locks"][phase]
    if path.exists():
        raise FileExistsError(path)
    if phase == "confirmation":
        for run in registered_runs(config, "development"):
            completed = ROOT / config["output_root"] / "development" / run["run"] / "complete.json"
            if not completed.exists() or not read_json(completed)["passed"]:
                raise ValueError(
                    "Complete the entire development calibration before follow-up freeze"
                )
    snapshot = ARTIFACT / (path.stem + "-source")
    snapshot.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for name in SOURCES:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    runs, exposure_cache = [], {}
    for entry in registered_runs(config, phase):
        world, spec, provenance, _ = load_source_world(entry["directory"])
        if entry["spec"] != spec or not read_json(Path(entry["directory"]) / "complete.json"):
            raise ValueError("Historical run identity mismatch")
        key = (
            world["metadata"]["dataset_sha256"],
            spec["stream_seed"],
            spec["steps"],
            spec["batch_size"],
            spec["n_atomic_per_batch"],
        )
        if key not in exposure_cache:
            exposure_cache[key] = actual_exposure_counts(world, spec)
        donors = select_same_bridge_donors(world, spec, exposure=exposure_cache[key])
        selection_path = (
            ARTIFACT / "donors" / config["execution_revision"] / phase / (entry["run"] + ".npz")
        )
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            selection_path, **{k: v for k, v in donors.items() if isinstance(v, np.ndarray)}
        )
        audit_path = selection_path.with_suffix(".json")
        write_json(audit_path, donors["audit"])
        runs.append(
            {
                **entry,
                "inputs": input_hashes(entry["directory"], spec["steps"]),
                "historical_provenance": provenance,
                "donors_file": str(selection_path.relative_to(ROOT)),
                "donors_sha256": digest(selection_path),
                "donor_audit_file": str(audit_path.relative_to(ROOT)),
                "donor_audit_sha256": digest(audit_path),
            }
        )
    write_json(
        path,
        {
            "frozen_utc": utc(),
            "phase": phase,
            "config_sha256": digest(CONFIG),
            "analysis_source_hashes": hashes,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_note": (
                "Existing user changes retained; hashes identify the execution files"
            ),
            "runs": runs,
            "design_status": (
                "New measurements on previously observed models; "
                "not independent retraining confirmation"
            ),
        },
    )
    print(
        json.dumps(
            {"phase": phase, "locked_runs": len(runs), "lock": str(path)}, ensure_ascii=False
        ),
        flush=True,
    )


def execute(config, phase, device, shard_index, shard_count):
    path = ARTIFACT / config["locks"][phase]
    lock = read_json(path)
    verify_lock(lock, phase, config)
    lock_hash = digest(path)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    for index, entry in enumerate(lock["runs"]):
        if index % shard_count != shard_index:
            continue
        out = ROOT / config["output_root"] / phase / entry["run"]
        complete_path = out / "complete.json"
        if complete_path.exists():
            saved = read_json(complete_path)
            if saved["lock_sha256"] != lock_hash or not saved["passed"]:
                raise ValueError("Existing output belongs to a different or failed execution")
            for name, expected in saved["output_hashes"].items():
                if digest(out / name) != expected:
                    raise ValueError(f"Completed output changed: {out / name}")
            print(
                json.dumps({"run": entry["run"], "state": "verified_existing_complete"}), flush=True
            )
            continue
        if out.exists():
            raise FileExistsError(
                f"Partial output retained; use a separately documented retry: {out}"
            )
        if input_hashes(entry["directory"], entry["spec"]["steps"]) != entry["inputs"]:
            raise ValueError("Historical input changed after freeze")
        if digest(ROOT / entry["donors_file"]) != entry["donors_sha256"]:
            raise ValueError("Donor selection changed after freeze")
        world, spec, provenance, package = load_source_world(entry["directory"])
        with np.load(ROOT / entry["donors_file"], allow_pickle=False) as archive:
            donors = {key: archive[key].copy() for key in archive.files}
        donors["audit"] = read_json(ROOT / entry["donor_audit_file"])
        out.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        metadata = {
            "started_utc": utc(),
            "run": entry["run"],
            "spec": spec,
            "phase": phase,
            "lock_sha256": lock_hash,
            "analysis_source_hashes": lock["analysis_source_hashes"],
            "donors_sha256": entry["donors_sha256"],
            "environment": environment(device),
        }
        write_json(out / "metadata.json", metadata)
        try:
            model, checkpoint = load_frozen_model(
                entry["directory"], spec["steps"], spec, package, device
            )
            arrays, report = evaluate_same_bridge(
                model, world, donors, device=device, batch_size=config["batch_size"]
            )
            # Historical endpoint prediction identity is checked independently.
            with np.load(
                Path(entry["directory"]) / f"predictions-{spec['steps']:07d}.npz",
                allow_pickle=False,
            ) as historical:
                for field in ("answer", "stop"):
                    if not np.array_equal(
                        arrays["baseline_" + field], historical["ood_composite_" + field]
                    ):
                        raise AssertionError(f"Historical endpoint {field} was not reproduced")
                expected = historical["ood_composite_nll"][:, 0]
                actual = -arrays["baseline_target_log_probability"]
                delta = float(np.max(np.abs(expected - actual))) if len(actual) else 0.0
                if delta > 3e-5:
                    raise AssertionError(f"Historical answer NLL differs: {delta}")
            report["engineering"]["historical_answer_eos_reproduced"] = True
            report["engineering"]["historical_answer_nll_max_delta"] = delta
            report["spec"] = spec
            report["run"] = entry["run"]
            report["phase"] = phase
            report["provenance"] = {**provenance, **checkpoint}
            report["donor_audit"] = donors["audit"]
            np.savez_compressed(out / "predictions.npz", **arrays)
            write_json(out / "report.json", report)
            write_json(
                complete_path,
                {
                    "completed_utc": utc(),
                    "passed": True,
                    "run": entry["run"],
                    "lock_sha256": lock_hash,
                    "wall_seconds": time.perf_counter() - started,
                    "output_hashes": {
                        name: digest(out / name)
                        for name in ("metadata.json", "predictions.npz", "report.json")
                    },
                },
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "run": entry["run"],
                        "state": "complete",
                        "common_queries": report["n_common_queries"],
                        "seconds": report["seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as error:
            write_json(
                out / "failed.json",
                {
                    "failed_utc": utc(),
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "lock_sha256": lock_hash,
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    config = read_json(CONFIG)
    if args.action == "freeze":
        freeze(config, args.phase)
    else:
        execute(config, args.phase, args.device, args.shard_index, args.shard_count)


if __name__ == "__main__":
    main()
