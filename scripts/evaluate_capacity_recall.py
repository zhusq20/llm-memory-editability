"""Re-evaluate a frozen parent without training, then independently reload it."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from run_capacity_controls import evaluate, initialize, runtime_setup, tensor_digest

from llm_memory_editability.capacity_scaling import digest_arrays, generate_world
from llm_memory_editability.grok_depth import utc, write_json


def run(config, spec, out, audit=False):
    device = torch.device("cuda:0")
    runtime_setup(device)
    started = time.monotonic()
    parent = Path(spec["parent"])
    checkpoint = parent / "latest.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == spec["parent_checkpoint_sha256"]
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    original = state["spec"]
    world = generate_world(original)
    assert digest_arrays(world) == state["world_sha256"]
    assert digest_arrays(dict(np.load(parent / "world.npz"))) == state["world_sha256"]
    model = initialize(original, device)
    model.load_state_dict(state["model"])
    before = tensor_digest(model)
    metrics, arrays = evaluate(model, world, original, device, full=True)
    assert tensor_digest(model) == before
    # Verify every historical native prediction and loss before interpreting the
    # new inference reference. This does not overwrite any historical artifact.
    old = np.load(parent / f"predictions-e{original['epochs']:05d}.npz")
    for key in old.files:
        if "losses" in key:
            assert np.allclose(arrays[key], old[key], rtol=0, atol=1e-5), key
        else:
            assert np.array_equal(arrays[key], old[key]), key
    if audit:
        saved = np.load(out / "predictions.npz")
        assert set(saved.files) == set(arrays)
        for key in arrays:
            assert np.allclose(saved[key], arrays[key], rtol=0, atol=1e-5), key
        record = json.loads((out / "run.json").read_text())
        result = {
            "passed": True,
            "metrics": metrics,
            "model_sha256": before,
            "unchanged_weights": True,
            "historical_native_reproduced": True,
            "separate_process": record["pid"] != os.getpid(),
            "audit_pid": os.getpid(),
            "created_utc": utc(),
        }
        assert result["separate_process"]
        write_json(out / "audit.json", result)
        write_json(out / "complete.json", result)
        write_json(out / "status.json", {"state": "complete", "step": 1})
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "run.json",
        {
            "spec": spec,
            "model": model.config_dict(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "world_sha256": state["world_sha256"],
            "initial_model_sha256": before,
            "parent_step": state["step"],
            "phase": "development",
            "tracking_group": config["batch"],
            "pid": os.getpid(),
            "gpu": int(os.environ["PHYSICAL_GPU"]),
            "gpu_name": torch.cuda.get_device_name(device),
            "source_files": config["source_files"],
            "created_utc": utc(),
        },
    )
    np.savez_compressed(out / "predictions.npz", **arrays)
    write_json(
        out / "learning.json",
        [
            {
                "step": 1,
                "optimizer_updates": 0,
                "metrics": metrics,
                "training_seconds": 0,
                "wall_seconds": time.monotonic() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
                "created_utc": utc(),
            }
        ],
    )
    write_json(out / "status.json", {"state": "awaiting_audit", "step": 1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "audit"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for path, expected in config["source_files"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    spec = next(s for s in config["runs"] if s["name"] == args.run)
    run(config, spec, Path(config["results_root"]) / "runs" / args.run, args.action == "audit")
