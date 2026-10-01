#!/usr/bin/env python3
"""Prepare and execute the paired 3/4-layer follow-up without modifying old runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.grok_depth_extension import (  # noqa: E402
    digest,
    read_json,
    validate_pair,
    verify_files,
)

EXPERIMENT = "grok-depth-extension-v1"
ARTIFACT = ROOT / "docs/development-artifacts" / EXPERIMENT
RESULTS = ROOT / "results" / EXPERIMENT
CONFIG = ROOT / "configs" / f"{EXPERIMENT}.json"
CORE = (
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_depth_data.py",
    "src/llm_memory_editability/bios_model.py",
)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def prepare():
    """Freeze the known baselines, all thresholds and execution inputs before training."""
    assert not (ARTIFACT / "execution-lock.json").exists(), "Refuse to replace an execution lock"
    assert not any((RESULTS / "extension").glob("*/metadata.json")), "Training already started"
    cfg = read_json(CONFIG)
    legacy_lock = read_json(
        ROOT / "docs/development-artifacts/grok-depth-v1/confirmation-lock.json"
    )
    core_audit = {}
    for rel in CORE:
        sha = digest(ROOT / rel)
        assert sha == legacy_lock["files"][rel], f"Historical scientific core changed: {rel}"
        core_audit[rel] = {"sha256": sha, "identical_to_historical_confirmation": True}
    references = {}
    for run_id, item in cfg["runs"].items():
        baseline_id = cfg["baseline_for_run"][run_id]
        directory = ROOT / "results/grok-depth-v1/confirmation" / baseline_id
        meta = read_json(directory / "metadata.json")
        complete = read_json(directory / "complete.json")
        learning = read_json(directory / "learning.json")
        spec = {**cfg["base"], **item}
        validate_pair(spec, meta["spec"])
        assert complete["endpoint"] == learning[-1]
        assert learning[-1]["step"] == spec["steps"] == 128000
        assert [row["step"] for row in learning] == spec["nodes"]
        references[baseline_id] = {
            "directory": str(directory.relative_to(ROOT)),
            "paired_final_threshold": learning[-1]["test_composite"]["accuracy"],
            "spec": meta["spec"],
            "files": {
                str((directory / name).relative_to(ROOT)): digest(directory / name)
                for name in (
                    "metadata.json",
                    "complete.json",
                    "learning.json",
                    "world.npz",
                    "world-metadata.json",
                    "data-audit.json",
                    "predictions-0128000.npz",
                )
            },
        }
    assert len(cfg["runs"]) == 12 and len(references) == 6
    files = [
        str(CONFIG.relative_to(ROOT)),
        *CORE,
        "src/llm_memory_editability/grok_depth_extension.py",
        "scripts/run_grok_depth_extension.py",
        "tests/test_grok_depth_extension.py",
        str((ARTIFACT / "preregistration.md").relative_to(ROOT)),
        str((ARTIFACT / "literature-alignment.json").relative_to(ROOT)),
        str((ARTIFACT / "tests.log").relative_to(ROOT)),
    ]
    hashes = {rel: digest(ROOT / rel) for rel in files}
    for rel in files:
        destination = ARTIFACT / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
    lock = {
        "experiment": EXPERIMENT,
        "created_utc": now(),
        "scope": "paired extension on previously observed worlds, not new-world confirmation",
        "files": hashes,
        "historical_core_comparison": core_audit,
        "references": references,
        "expected_runs": list(cfg["runs"]),
        "thresholds": cfg["thresholds"],
        "gpu_assignment": "physical GPU 2 only",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write(ARTIFACT / "execution-lock.json", lock)
    print(json.dumps({"state": "frozen", "runs": 12, "baselines": 6, "utc": now()}))


def checked_inputs():
    lock = read_json(ARTIFACT / "execution-lock.json")
    verify_files(ROOT, lock["files"])
    for baseline in lock["references"].values():
        verify_files(ROOT, baseline["files"])
    return read_json(CONFIG), lock


def run_one(run_id, resume):
    import numpy as np
    import torch

    from llm_memory_editability.grok_depth import run
    from llm_memory_editability.grok_depth_data import audit_world, build_world

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "This batch is assigned physical GPU 2"
    cfg, lock = checked_inputs()
    spec = {**cfg["base"], **cfg["runs"][run_id]}
    ref = lock["references"][cfg["baseline_for_run"][run_id]]
    validate_pair(spec, ref["spec"])
    out = RESULTS / "extension" / run_id
    if resume:
        meta = read_json(out / "metadata.json")
        assert meta["spec"] == spec and meta["execution_lock_sha256"] == digest(
            ARTIFACT / "execution-lock.json"
        )
        for original, sha in meta["files"].items():
            relative = Path(original).relative_to(ROOT)
            assert digest(out / "source" / relative) == sha
    else:
        out.mkdir(parents=True, exist_ok=False)
    handle = (out / "run.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not resume:
        for relative in lock["files"]:
            destination = out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ARTIFACT / "source" / relative, destination)
        write(
            out / "metadata.json",
            {
                "started_utc": now(),
                "spec": spec,
                "files": {str(ROOT / rel): sha for rel, sha in lock["files"].items()},
                "execution_lock_sha256": digest(ARTIFACT / "execution-lock.json"),
                "paired_baseline": cfg["baseline_for_run"][run_id],
                "paired_final_threshold": ref["paired_final_threshold"],
                "pid": os.getpid(),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "argv": sys.argv,
                "python": platform.python_version(),
            },
        )
    world = build_world(
        spec["world_seed"],
        **{
            key: spec[key]
            for key in (
                "entities",
                "relations",
                "degree",
                "phi",
                "id_fraction",
                "id_test_fraction",
            )
        },
    )
    baseline = ROOT / ref["directory"]
    with np.load(baseline / "world.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(world) - {"metadata"}
        for key in saved.files:
            assert np.array_equal(saved[key], world[key]), f"Paired world mismatch: {key}"
    assert world["metadata"] == read_json(baseline / "world-metadata.json")
    audit = audit_world(world)
    assert audit == read_json(baseline / "data-audit.json")
    if not resume:
        write(out / "data-audit.json", audit)
        write(out / "world-metadata.json", world["metadata"])
        np.savez_compressed(
            out / "world.npz", **{k: v for k, v in world.items() if k != "metadata"}
        )
        write(
            out / "environment.json",
            {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "device": "cuda:0",
                "gpu": torch.cuda.get_device_name(0),
                "tf32": True,
                "precision": "FP32 parameters, forward and optimizer; TF32 matmuls",
            },
        )
    run(spec, world, out, "cuda:0", resume)


def execute():
    cfg, _ = checked_inputs()
    launches = RESULTS / "launches"
    launches.mkdir(parents=True, exist_ok=True)
    matrix_handle = (RESULTS / "matrix.lock").open("a")
    fcntl.flock(matrix_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = []
    record_path = launches / (now().replace(":", "").replace("+", "_") + "-matrix.json")
    for run_id in cfg["runs"]:
        out = RESULTS / "extension" / run_id
        if (out / "complete.json").exists():
            records.append({"run_id": run_id, "state": "already_complete"})
            write(record_path, records)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "run", run_id]
        if (out / "latest.pt").exists():
            command.append("--resume")
        logfile = launches / f"{run_id}-{now().replace(':', '').replace('+', '_')}.log"
        started = now()
        with logfile.open("x") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "2",
                    "PYTHONPATH": str(ROOT / "src"),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        record = {
            "run_id": run_id,
            "gpu": 2,
            "started_utc": started,
            "finished_utc": now(),
            "exit_code": result.returncode,
            "log": str(logfile),
        }
        records.append(record)
        write(record_path, records)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("execute")
    item = sub.add_parser("run")
    item.add_argument("run_id")
    item.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "execute":
        execute()
    else:
        run_one(args.run_id, args.resume)


if __name__ == "__main__":
    main()
