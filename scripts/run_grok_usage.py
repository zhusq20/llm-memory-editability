#!/usr/bin/env python3
"""Run one independently archived fact-usage experiment."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from llm_memory_editability.grok_depth import utc, write_json
    from llm_memory_editability.grok_usage_data import build_arm_world, build_experiment
    from llm_memory_editability.grok_usage_train import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--config", default="configs/grok-usage-development-v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    spec = {**config["base"], **config["runs"][args.run_id]}
    if sorted(set(spec["nodes"])) != spec["nodes"] or spec["nodes"][-1] != spec["steps"]:
        raise ValueError("Invalid evaluation nodes")
    if not set(spec["weight_nodes"]).issubset(spec["nodes"]):
        raise ValueError("Weight nodes must be evaluated")
    lock = json.loads((ROOT / config["source_lock"]).read_text())
    for relative, expected in lock["files"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Execution source changed: {relative}")
    experiment = build_experiment(spec)
    world = build_arm_world(experiment, spec["arm"])
    if args.data_only:
        print(json.dumps(world["metadata"], indent=2))
        return
    out = ROOT / config["output_root"] / spec["phase"] / args.run_id
    if args.resume:
        meta = json.loads((out / "metadata.json").read_text())
        if meta["spec"] != spec or meta["source_lock"] != lock:
            raise ValueError("Resume source/specification changed")
    else:
        out.mkdir(parents=True, exist_ok=False)
        for relative in lock["files"]:
            target = out / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        write_json(
            out / "metadata.json",
            {
                "started_utc": utc(),
                "spec": spec,
                "source_lock": lock,
                "pid": os.getpid(),
                "python": platform.python_version(),
                "argv": sys.argv,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_sharing": "Other users' jobs retained; bounded allocator; no speed comparison",
            },
        )
    handle = (out / "run.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(out / "world-metadata.json", world["metadata"])
    import numpy as np
    import torch

    write_json(
        out / "environment.json",
        {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "precision": "FP32/TF32",
        },
    )
    np.savez_compressed(
        out / "world.npz",
        atomic=world["atomic"],
        **{"eval_" + k: v for k, v in world["evaluation"].items()},
        table_fact_indices=world["table_fact_indices"],
        table_weights=world["table_weights"],
        table_row_kinds=world["table_row_kinds"],
    )
    try:
        run(spec, world, out, args.device, args.resume)
    except Exception as error:
        write_json(out / "failure.json", {"utc": utc(), "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
