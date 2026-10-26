#!/usr/bin/env python3
"""Current one-layer/four-round joint gate on the unchanged Wang task arrays."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_joint_update_wang as joint  # noqa: E402

from llm_memory_editability.compact_wang_data import verify_metadata  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest  # noqa: E402

BATCH = "compact-wang-joint-gate-v1"
ARM = "joint-token"


def validate_config(config):
    if config["batch"] != BATCH or config["family"] != "wang":
        raise ValueError("Only the registered compact Wang experiment may run")
    if config["arms"] != {
        ARM: {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "update_mode": "token",
            "initial_update": 0.95,
        }
    }:
        raise ValueError("Keep exactly the current one-layer/four-round joint gate")
    if config["training"]["steps"] != 1500000 or config["inference_repeats"] != [1, 2, 3, 4]:
        raise ValueError("Frozen budget or inference grid changed")
    if config["gate_scope"] != joint.SCOPE or config["position_encoding"] != "learned":
        raise ValueError("Gate or position semantics changed")
    reference = json.loads((ROOT / config["architecture_reference"]["config"]).read_text())
    if config["model"] != reference["model"] or config["arms"] != reference["arms"]:
        raise ValueError("Current multihop architecture must be preserved")
    if (
        digest(ROOT / config["architecture_reference"]["config"])
        != config["architecture_reference"]["sha256"]
    ):
        raise ValueError("Architecture reference changed")
    for key in ("lr", "weight_decay", "warmup", "batch_size", "initialization", "steps"):
        if config["training"][key] != reference["training"][key]:
            raise ValueError(f"Current model training setting changed: {key}")
    nodes = config["evaluation_nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != 1500000:
        raise ValueError("Invalid fixed evaluation nodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "grid"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=(ARM,), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    metadata = joint.original.verify_input(config)
    verify_metadata(metadata)
    if (
        digest(Path(config["input_root"]) / "complete.json")
        != config["input_origin"]["metadata_sha256"]
    ):
        raise ValueError("Frozen compact vocabulary metadata changed")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with joint.installed(config):
            if args.stage == "preflight":
                joint.original.preflight(
                    {**config, "_checked_config_sha256": digest(args.config)}, args.arm, args.base
                )
            elif args.stage == "train":
                joint.depth.tracked_train(config, args.arm, args.base, args.config)
            else:
                joint.depth.endpoint(config, args.arm, args.base, args.config)
    except BaseException as error:
        write_json(
            args.base / "failures" / f"{args.arm}-{args.stage}.json",
            {
                "utc": utc(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
