#!/usr/bin/env python3
"""Read-only replay of the original F on the target GPU, before new training."""

import argparse
import json
from pathlib import Path

import numpy as np
import run_loop_hop as native
import torch

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grokking_reproduction import digest, model_digest
from llm_memory_editability.loop_hop_data import load, panel_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    old_path = Path("configs/loop-hop-unified-v1.json")
    old = json.loads(old_path.read_text())
    world, panels, metadata = load("/inputs")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    name = config["baseline"]["arm"]
    model, spec, arm, initial = native.initialize(old, name, metadata, torch.device("cuda:0"))
    checkpoint = args.baseline / "checkpoint-0300000.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        initial != config["baseline"]["initial_model_sha256"]
        or saved["config_sha256"] != digest(old_path)
        or saved["spec"] != spec
        or saved["arm"] != arm
        or saved["world_sha256"] != metadata["world_sha256"]
    ):
        raise ValueError("Target F initialization/checkpoint recipe mismatch")
    model.load_state_dict(saved["model"], strict=True)
    before = model_digest(model)
    assert before == saved["model_sha256"]
    grid = json.loads((args.baseline / "grid.json").read_text())
    findings = {}
    for repeats in range(1, 5):
        model.repeats = repeats
        metrics, predictions = native.measure_panels(
            model, world, panels, metadata, torch.device("cuda:0"), config["evaluation_batch_size"]
        )
        for panel in panel_names():
            with np.load(args.baseline / "grid" / f"r{repeats}" / f"{panel}.npz") as prior:
                exact = all(
                    np.array_equal(prior[k], predictions[panel][k])
                    for k in ("rows", "answer", "stop")
                )
                nll = np.allclose(prior["nll"], predictions[panel]["nll"], rtol=1e-6, atol=1e-6)
            reference = grid["metrics"][str(repeats)][panel]
            scalars = all(
                reference[k] is None
                if v is None
                else reference[k] is not None and np.isclose(reference[k], v, rtol=1e-6, atol=1e-6)
                for k, v in metrics[panel].items()
            )
            findings[f"r{repeats}/{panel}"] = bool(exact and nll and scalars)
    findings["weights_unchanged"] = model_digest(model) == before
    passed = all(findings.values())
    write_json(
        args.base / "baseline-target-replay.json",
        {
            "passed": passed,
            "findings": findings,
            "initial_model_sha256": initial,
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "new_baseline_updates": 0,
            "utc": utc(),
        },
    )
    if not passed:
        raise AssertionError(
            "Target GPU is not numerically compatible with reused F; no new training started"
        )


if __name__ == "__main__":
    main()
