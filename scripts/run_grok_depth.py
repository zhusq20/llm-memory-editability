#!/usr/bin/env python3
"""Execute an explicitly registered small-model grokking experiment."""

import argparse
import fcntl
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from llm_memory_editability.grok_depth import run, source_hash, utc, write_json
    from llm_memory_editability.grok_depth_data import audit_world, build_world

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--config", default="configs/grok-depth-v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    item = cfg["runs"][args.run_id]
    spec = {**cfg["base"], **item}
    phase = spec["phase"]
    out = ROOT / "results/grok-depth-v1" / phase / args.run_id
    files = [
        ROOT / p
        for p in [
            args.config,
            "src/llm_memory_editability/grok_depth.py",
            "src/llm_memory_editability/grok_depth_data.py",
            "src/llm_memory_editability/bios_model.py",
            "scripts/run_grok_depth.py",
        ]
    ]
    if phase == "confirmation":
        lock = json.loads(
            (ROOT / "docs/development-artifacts/grok-depth-v1/confirmation-lock.json").read_text()
        )
        for path, expected in lock["files"].items():
            assert source_hash([ROOT / path])[str(ROOT / path)] == expected, path
    if args.resume:
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["spec"] == spec
        for path in files[1:]:
            assert source_hash([path])[str(path)] == meta["files"][str(path)]
    else:
        out.mkdir(parents=True, exist_ok=False)
        for path in files:
            dest = out / "source" / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        write_json(
            out / "metadata.json",
            {
                "started_utc": utc(),
                "spec": spec,
                "files": source_hash(files),
                "pid": os.getpid(),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "argv": sys.argv,
                "python": platform.python_version(),
            },
        )
    lock_handle = (out / "run.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    world = build_world(
        spec["world_seed"],
        entities=spec["entities"],
        relations=spec["relations"],
        degree=spec["degree"],
        phi=spec["phi"],
        id_fraction=spec["id_fraction"],
        id_test_fraction=spec["id_test_fraction"],
    )
    write_json(out / "data-audit.json", audit_world(world))
    write_json(out / "world-metadata.json", world["metadata"])
    import numpy as np
    import torch

    write_json(
        out / "environment.json",
        {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "device": str(args.device),
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "tf32": True,
            "precision": "FP32 parameters, forward and optimizer; TF32 matmuls",
        },
    )
    np.savez_compressed(out / "world.npz", **{k: v for k, v in world.items() if k != "metadata"})
    run(spec, world, out, args.device, args.resume)


if __name__ == "__main__":
    main()
