#!/usr/bin/env python3
"""Run a separately registered longer-path development or confirmation task."""

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
    from llm_memory_editability.grok_depth import source_hash, utc, write_json
    from llm_memory_editability.grok_multihop import run
    from llm_memory_editability.grok_multihop_data import audit_world, build_world

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--config", default="configs/grok-multihop-development-v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    spec = {**cfg["base"], **cfg["runs"][args.run_id]}
    if sorted(set(spec["nodes"])) != spec["nodes"] or spec["nodes"][-1] != spec["steps"]:
        raise ValueError("Registered evaluation nodes must be sorted and include the endpoint")
    out = ROOT / cfg["output_root"] / spec["phase"] / args.run_id
    files = [
        ROOT / p
        for p in [
            args.config,
            "src/llm_memory_editability/grok_multihop.py",
            "src/llm_memory_editability/grok_multihop_data.py",
            "src/llm_memory_editability/grok_depth.py",
            "src/llm_memory_editability/grok_depth_data.py",
            "src/llm_memory_editability/bios_model.py",
            "scripts/run_grok_multihop.py",
        ]
    ]
    lock = json.loads((ROOT / cfg["source_lock"]).read_text())
    for relative, expected in lock["files"].items():
        path = ROOT / relative
        if source_hash([path])[str(path)] != expected:
            raise ValueError(f"Locked execution source changed: {relative}")
    world = build_world(
        spec["world_seed"],
        **{
            k: spec[k]
            for k in (
                "hops",
                "entities",
                "relations",
                "degree",
                "phi",
                "id_fraction",
                "id_test_fraction",
                "evaluation_size",
            )
        },
    )
    if args.data_only:
        print(json.dumps(audit_world(world), indent=2))
        return
    if args.resume:
        meta = json.loads((out / "metadata.json").read_text())
        if meta["spec"] != spec:
            raise ValueError("Resume spec differs from original metadata")
        for path in files[1:]:
            if source_hash([path])[str(path)] != meta["files"][str(path)]:
                raise ValueError(f"Resume source changed: {path}")
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
