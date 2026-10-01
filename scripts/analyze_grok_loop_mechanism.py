#!/usr/bin/env python3
"""Analyze fixed-prefix component interventions on frozen loop endpoints.

Default rows are the training-independent fixed ID probe. Add --split
ood_composite or test_full_composite explicitly; --split can be repeated.
This command performs evaluation only and never selects donors from scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import source_hash, utc, write_json
from llm_memory_editability.grok_depth_bridge import environment
from llm_memory_editability.grok_loop_mechanism import SPLITS, evaluate_run, select_donors


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source_world(run_dir):
    """Verify historical source copies and graph truth before loading any model."""
    run_dir = Path(run_dir).resolve()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    spec = metadata["spec"]
    snapshot = run_dir / "source"
    files = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    verified = {}
    for original, expected in metadata["files"].items():
        matches = [
            (relative, path)
            for relative, path in files.items()
            if original == relative or original.endswith("/" + relative)
        ]
        if len(matches) != 1:
            raise ValueError(f"Historical source is missing or ambiguous: {original}")
        relative, path = matches[0]
        if digest(path) != expected:
            raise ValueError(f"Historical source hash mismatch: {path}")
        verified[relative] = expected
    configs = [snapshot / path for path in verified if path.startswith("configs/")]
    if len(configs) != 1:
        raise ValueError("Expected one frozen run configuration")
    config = json.loads(configs[0].read_text())
    if {**config["base"], **config["runs"][run_dir.name]} != spec:
        raise ValueError("Historical metadata disagrees with its frozen configuration")
    package = (
        "_grok_loop_mechanism_source_" + hashlib.sha256(str(run_dir).encode()).hexdigest()[:16]
    )
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(snapshot / "src/llm_memory_editability")]
    sys.modules[package] = namespace
    data = importlib.import_module(package + ".grok_multihop_data")
    world_path = run_dir / "world.npz"
    world_hash = digest(world_path)
    with np.load(world_path, allow_pickle=False) as saved:
        world = {key: saved[key].copy() for key in saved.files}
    world["metadata"] = json.loads((run_dir / "world-metadata.json").read_text())
    audit = data.audit_world(world)
    recorded = json.loads((run_dir / "data-audit.json").read_text())
    if audit != recorded or audit["dataset_sha256"] != world["metadata"]["dataset_sha256"]:
        raise ValueError("Stored world disagrees with historical data audit")
    if digest(world_path) != world_hash:
        raise ValueError("Stored world changed while loading")
    provenance = {
        "source_dir": str(run_dir),
        "frozen_source_hashes": verified,
        "metadata_sha256": digest(run_dir / "metadata.json"),
        "world_file_sha256": world_hash,
        "dataset_sha256": audit["dataset_sha256"],
        "world_audit": audit,
        "world_regenerated": False,
    }
    return world, spec, provenance, package


def load_frozen_model(run_dir, step, spec, package, device):
    """Instantiate the snapshot's model implementation and verify checkpoint identity."""
    if step not in spec["weight_nodes"]:
        raise ValueError(f"Step {step} is not a registered saved-weight node")
    path = Path(run_dir) / f"weights-{step:07d}.pt"
    checkpoint_hash = digest(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["spec"] != spec or checkpoint["step"] != step:
        raise ValueError("Checkpoint identity disagrees with historical metadata")
    if digest(path) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    model_module = importlib.import_module(package + ".grok_loop_model")
    config_module = importlib.import_module(package + ".bios_model")
    config = config_module.ModelConfig(
        vocab_size=2 + spec["entities"] + spec["relations"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    model = model_module.LoopGPT(
        config,
        repeats=spec["repeats"],
        dropout=spec["dropout"],
        initialization=spec["init_scheme"],
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"checkpoint_sha256": checkpoint_hash, "checkpoint_step": step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--split", choices=SPLITS, action="append")
    parser.add_argument("--step", type=int, help="Default: each run's registered endpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--donor-seed", type=int, default=20260930)
    parser.add_argument("--out", type=Path, default=Path("results/grok-loop-mechanism-v1"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    root = Path(__file__).resolve().parents[1]
    analysis_files = [
        root / "scripts/analyze_grok_loop_mechanism.py",
        root / "src/llm_memory_editability/grok_loop_mechanism.py",
        root / "tests/test_grok_loop_mechanism.py",
        root / "src/llm_memory_editability/grok_depth.py",
        root / "src/llm_memory_editability/grok_depth_bridge.py",
        root / "src/llm_memory_editability/grok_multihop.py",
    ]
    for run_dir in args.run_dirs:
        world, spec, provenance, package = load_source_world(run_dir)
        step = args.step if args.step is not None else spec["steps"]
        # Select every requested donor table before loading the model.
        donors_by_split = {
            split: select_donors(world, args.donor_seed, split)
            for split in dict.fromkeys(args.split or ["test_composite"])
        }
        model, checkpoint_provenance = load_frozen_model(run_dir, step, spec, package, device)
        for split, donors in donors_by_split.items():
            out = args.out / run_dir.name / f"step-{step:07d}-{split}"
            out.mkdir(parents=True, exist_ok=False)
            for source in analysis_files:
                destination = out / "analysis-source" / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            metadata = {
                "started_utc": utc(),
                "spec": spec,
                "split": split,
                "donor_seed": args.donor_seed,
                "step": step,
                "provenance": {**provenance, **checkpoint_provenance},
                "analysis_source_hashes": source_hash(analysis_files),
                "environment": environment(device),
            }
            write_json(out / "metadata.json", metadata)
            np.savez_compressed(out / "donors.npz", **donors)
            try:
                summary, predictions = evaluate_run(
                    model, world, donors, device=device, batch_size=args.batch_size
                )
                np.savez_compressed(out / "predictions.npz", **predictions)
                write_json(out / "summary.json", summary)
                write_json(out / "status.json", {"state": "complete", "finished_utc": utc()})
                print(
                    json.dumps({"run": run_dir.name, "split": split, "output": str(out)}),
                    flush=True,
                )
            except Exception as exc:
                write_json(
                    out / "status.json", {"state": "failed", "error": repr(exc), "utc": utc()}
                )
                raise
        del model


if __name__ == "__main__":
    main()
