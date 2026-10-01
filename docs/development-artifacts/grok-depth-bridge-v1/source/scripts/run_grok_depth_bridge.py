#!/usr/bin/env python3
"""Evaluate a registered phase of prefix-only first-block causal interventions."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import numpy as np
    import torch

    from llm_memory_editability.grok_depth_bridge import (
        digest,
        environment,
        evaluate_run,
        load_source_run,
        read_json,
        require,
        select_donors,
        utc,
        write_json,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", default="configs/grok-depth-bridge-v1.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    cfg = read_json(config_path)
    registered = cfg[args.phase + "_runs"]
    require(bool(registered), "registered phase is empty")
    entries = [(item, int(step)) for item in registered for step in item["steps"]]
    identities = [(item["run_id"], step) for item, step in entries]
    require(len(set(identities)) == len(identities), "duplicate checkpoint registration")
    require(all(item["phase"] == args.phase for item, _ in entries), "registered phase mismatch")
    source_root = (ROOT / cfg["source_root"]).resolve()
    output_root = (ROOT / cfg["output_root"]).resolve()
    require(
        output_root != source_root and source_root not in output_root.parents,
        "output must not modify historical source tree",
    )
    phase_dir = output_root / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    lock = (phase_dir / "batch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for item, step in entries:
        out = phase_dir / item["run_id"] / f"step-{step:07d}"
        require(not out.exists(), f"refusing to overwrite existing evaluation: {out}")
    require(not (phase_dir / "batch-summary.json").exists(), "phase already completed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    files = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "src/llm_memory_editability/grok_depth_bridge.py",
    ]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    for path in files:
        target = phase_dir / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    started = time.perf_counter()
    metadata = {
        "phase": args.phase,
        "started_utc": utc(),
        "config": cfg,
        "files": hashes,
        "environment": environment(args.device),
        "pid": os.getpid(),
        "argv": sys.argv,
    }
    write_json(phase_dir / "metadata.json", metadata)
    results = []
    for item, step in entries:
        run_started = time.perf_counter()
        out = phase_dir / item["run_id"] / f"step-{step:07d}"
        out.mkdir(parents=True, exist_ok=False)
        source_dir = source_root / item["phase"] / item["run_id"]
        try:
            model, world, spec, provenance = load_source_run(source_dir, step, args.device)
            donors = select_donors(world, cfg["donor_seed"])
            summary, arrays = evaluate_run(
                model, world, donors, args.device, cfg.get("batch_size", 1024)
            )
            with np.load(
                source_dir / f"predictions-{step:07d}.npz", allow_pickle=False
            ) as historical:
                for key in ("answer", "stop"):
                    require(
                        np.array_equal(
                            arrays["baseline_" + key], historical["test_composite_" + key]
                        ),
                        f"baseline {key} differs from historical predictions",
                    )
            summary["engineering_checks"]["baseline_identical_to_historical_predictions"] = True
            summary.update(
                {
                    "state": "complete",
                    "run_id": item["run_id"],
                    "phase": args.phase,
                    "step": step,
                    "spec": spec,
                    "source": provenance,
                    "donor_seed": cfg["donor_seed"],
                    "evaluation_files": hashes,
                    "environment": metadata["environment"],
                    "finished_utc": utc(),
                    "wall_seconds": time.perf_counter() - run_started,
                }
            )
            np.savez_compressed(out / "predictions.npz", **arrays)
            summary["predictions_sha256"] = digest(out / "predictions.npz")
            write_json(out / "summary.json", summary)
            results.append(summary)
            write_json(
                phase_dir / "progress.json",
                {"finished": len(results), "registered": len(entries), "updated_utc": utc()},
            )
            print(
                json.dumps(
                    {
                        "run_id": item["run_id"],
                        "step": step,
                        "state": "complete",
                        "summary": str(out / "summary.json"),
                    }
                ),
                flush=True,
            )
            del model
        except Exception as exc:
            write_json(
                out / "failure.json",
                {"state": "failed", "type": type(exc).__name__, "message": str(exc), "utc": utc()},
            )
            raise
    require(
        all(digest(ROOT / path) == expected for path, expected in hashes.items()),
        "evaluation source/config changed during batch",
    )
    write_json(
        phase_dir / "batch-summary.json",
        {
            **metadata,
            "state": "complete",
            "finished_utc": utc(),
            "wall_seconds": time.perf_counter() - started,
            "runs": results,
            "aggregation_note": (
                "Each run/checkpoint remains separate; repeated seeds/checkpoints "
                "are not independent worlds."
            ),
        },
    )


if __name__ == "__main__":
    main()
