#!/usr/bin/env python3
"""Measure fixed-checkpoint recurrence changes without selecting a best test depth."""

import argparse
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import source_hash, utc, write_json
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.grok_multihop import evaluate_rows
from llm_memory_editability.grok_multihop_data import audit_world

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (
    "scripts/evaluate_grok_loop_recurrence.py",
    "src/llm_memory_editability/bios_model.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_depth_data.py",
    "src/llm_memory_editability/grok_loop_model.py",
    "src/llm_memory_editability/grok_multihop.py",
    "src/llm_memory_editability/grok_multihop_data.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_stage_lock(cfg, config_path, root=ROOT):
    """Reject changed registered source before any evaluation or output creation."""
    lock_path = root / cfg["source_lock"]
    lock = json.loads(lock_path.read_text())
    config_relative = Path(config_path).resolve().relative_to(root.resolve()).as_posix()
    if config_relative not in lock["files"]:
        raise ValueError("Training configuration is absent from its source lock")
    for relative, expected in lock["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Current file differs from training stage lock: {relative}")
    return lock, {"path": str(lock_path), "sha256": digest(lock_path), "verified": True}


def verify_analysis_lock(lock_path, root=ROOT):
    """The analysis lock records code dependencies, never its own hash."""
    lock_path = Path(lock_path)
    lock = json.loads(lock_path.read_text())
    if not set(DEPENDENCIES).issubset(lock["files"]):
        raise ValueError("Analysis lock omits a recurrence dependency")
    for relative, expected in lock["files"].items():
        if digest(root / relative) != expected:
            raise ValueError(f"Current file differs from recurrence analysis lock: {relative}")
    return {"path": str(lock_path), "sha256": digest(lock_path), "verified": True}


def verify_training_artifacts(directory, spec, root=ROOT):
    """Verify source, graph, endpoint and prediction identity without modifying them."""
    metadata = json.loads((directory / "metadata.json").read_text())
    complete = json.loads((directory / "complete.json").read_text())
    if metadata["spec"] != spec or complete["spec"] != spec:
        raise ValueError("Stored training specification differs from scan registration")
    if complete["endpoint"]["step"] != spec["steps"]:
        raise ValueError("Training endpoint is not at the registered step")
    frozen = {}
    for filename, expected in metadata["files"].items():
        original = Path(filename)
        relative = original.relative_to(root)
        snapshot = directory / "source" / relative
        if digest(snapshot) != expected or digest(original) != expected:
            raise ValueError(f"Current or frozen source differs from training: {relative}")
        frozen[relative.as_posix()] = expected
    if any(relative not in frozen for relative in DEPENDENCIES if relative.startswith("src/")):
        raise ValueError("An evaluation dependency was not captured in training sources")
    world_path = directory / "world.npz"
    world_hash = digest(world_path)
    with np.load(world_path, allow_pickle=False) as archive:
        world = {key: archive[key].copy() for key in archive.files}
    world["metadata"] = json.loads((directory / "world-metadata.json").read_text())
    audit = audit_world(world)
    if audit != json.loads((directory / "data-audit.json").read_text()):
        raise ValueError("World differs from its recorded training data audit")
    if audit["dataset_sha256"] != world["metadata"]["dataset_sha256"]:
        raise ValueError("World metadata hash differs from its arrays")
    if digest(world_path) != world_hash:
        raise ValueError("World changed during verification")
    predictions_path = directory / f"predictions-{spec['steps']:07d}.npz"
    provenance = {
        "metadata_sha256": digest(directory / "metadata.json"),
        "complete_sha256": digest(directory / "complete.json"),
        "prediction_sha256": digest(predictions_path),
        "world_file_sha256": world_hash,
        "dataset_sha256": audit["dataset_sha256"],
        "world_audit": audit,
        "training_source_hashes": frozen,
        "current_and_snapshot_sources_match_training": True,
    }
    return world, provenance


def verify_native_predictions(predictions, native, world, split):
    """At training repeats, confirm unchanged generated outputs and likelihoods."""
    if not len(world[split]):
        return {"n": 0, "passed": True, "max_nll_absolute_error": None}
    for key in ("answer", "stop", "target"):
        if not np.array_equal(predictions[key], native[split + "_" + key]):
            raise ValueError(f"Training-repeats scan changes native {split}.{key}")
    error = float(np.max(np.abs(predictions["nll"] - native[split + "_nll"])))
    if not np.isfinite(error) or error > 1e-5:
        raise ValueError(f"Training-repeats scan changes native {split}.nll by {error}")
    return {"n": len(world[split]), "passed": True, "max_nll_absolute_error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--architectures", nargs="+", default=["l2"])
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--repeats", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--analysis-lock",
        default="docs/development-artifacts/grok-loop-v1/recurrence-analysis-lock.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    cfg = json.loads((ROOT / args.config).read_text())
    lock, lock_provenance = verify_stage_lock(cfg, ROOT / args.config)
    analysis_provenance = verify_analysis_lock(ROOT / args.analysis_lock)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    source_paths = sorted(
        {*(ROOT / name for name in DEPENDENCIES), *(ROOT / name for name in lock["files"])}
    )
    for source in source_paths:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(ROOT / cfg["source_lock"], out / "training-source-lock.json")
    shutil.copy2(ROOT / args.analysis_lock, out / "recurrence-analysis-lock.json")
    report = {
        "started_utc": utc(),
        "config": args.config,
        "repeats": args.repeats,
        "interpretation": "Fixed-recurrence training; other counts are altered execution. "
        "No best-count selection or length-generalization claim.",
        "runs": [],
        "source": source_hash(source_paths),
        "training_source_lock": lock_provenance,
        "analysis_source_lock": analysis_provenance,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "parameter_dtype": "torch.float32",
            "dropout": "disabled by eval mode",
            "evaluation": "same eager evaluate_rows as registered training; no autocast",
        },
    }
    write_json(out / "summary.json", report)
    for run_id, item in cfg["runs"].items():
        spec = {**cfg["base"], **item}
        if args.runs is not None and run_id not in args.runs:
            continue
        if spec["architecture"] not in args.architectures:
            continue
        directory = ROOT / cfg["output_root"] / spec["phase"] / run_id
        world, provenance = verify_training_artifacts(directory, spec)
        checkpoint = directory / f"weights-{spec['steps']:07d}.pt"
        checkpoint_hash = digest(checkpoint)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["spec"] != spec or state["step"] != spec["steps"]:
            raise ValueError("Checkpoint specification or step differs")
        if digest(checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed during loading")
        if any(value.dtype != torch.float32 for value in state["model"].values()):
            raise ValueError("Checkpoint parameters differ from registered FP32 precision")
        model = (
            LoopGPT(
                ModelConfig(
                    vocab_size=2 + spec["entities"] + spec["relations"],
                    width=spec["width"],
                    layers=spec["layers"],
                    heads=spec["heads"],
                    context=8,
                ),
                repeats=spec["repeats"],
                dropout=spec["dropout"],
                initialization=spec["init_scheme"],
            )
            .to(device)
            .eval()
        )
        model.load_state_dict(state["model"])
        run = {
            "run": run_id,
            "spec": spec,
            "checkpoint": source_hash([checkpoint]),
            "provenance": provenance,
            "native_repeats_audit": {},
            "measurements": [],
        }
        with np.load(
            directory / f"predictions-{spec['steps']:07d}.npz", allow_pickle=False
        ) as archive:
            native = {key: archive[key].copy() for key in archive.files}
        predictions = {}
        for count in args.repeats:
            if count < 1:
                raise ValueError("Recurrences must be positive")
            model.repeats = count
            measured = {"repeats": count, "effective_depth": count * spec["layers"]}
            for split in ("atomic", "test_composite", "ood_composite"):
                measured[split], pred = evaluate_rows(model, world[split], device, spec["hops"] + 2)
                if count == spec["repeats"]:
                    run["native_repeats_audit"][split] = verify_native_predictions(
                        pred, native, world, split
                    )
                predictions.update({f"r{count}_{split}_{k}": v for k, v in pred.items()})
            run["measurements"].append(measured)
        report["runs"].append(run)
        np.savez_compressed(out / f"{run_id}.npz", **predictions)
        write_json(out / "summary.json", report)
        print(json.dumps({"run": run_id, "completed_recurrences": args.repeats}), flush=True)
    report["finished_utc"] = utc()
    write_json(out / "summary.json", report)


if __name__ == "__main__":
    main()
