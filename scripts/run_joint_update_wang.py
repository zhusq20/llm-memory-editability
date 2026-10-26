#!/usr/bin/env python3
"""Use the new joint boundary with the frozen original Wang task and recipe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_gate_extension as original  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest  # noqa: E402
from llm_memory_editability.joint_update_gate import JointUpdateGateGPT  # noqa: E402

BATCH = "joint-update-gate-wang-v1"
SCOPE = "continuous_update_and_new_paper_injection"
native, depth = original.native, original.depth


def validate_config(config, reference):
    if config["batch"] != BATCH or config["gate_scope"] != SCOPE:
        raise ValueError("Only the registered joint Wang gate may execute")
    if config["arms"] != {"joint-token": reference["arms"]["token-update"]}:
        raise ValueError("Preserve the Wang four-layer/two-round learned gate at u0=.95")
    for key in (
        "family",
        "position_encoding",
        "data",
        "model",
        "training",
        "dropout_seed",
        "evaluation_nodes",
        "evaluation_batch_size",
        "inference_repeats",
        "input_origin",
        "input_root",
        "heartbeat_steps",
        "primary",
    ):
        if config[key] != reference[key]:
            raise ValueError(f"Frozen Wang recipe changed: {key}")
    if config["training"]["steps"] != 1500000:
        raise ValueError("Use exactly 1.5M scientific updates")
    if config["budget"]["new_scientific_updates"] != 1500000:
        raise ValueError("Scientific update budget changed")


def initialize(config, name, metadata, device):
    model, spec, arm, initial = original.initialize(config, name, metadata, device)
    return model, spec, {**arm, "gate_scope": SCOPE}, initial


def accounting(model, spec, arm, metadata):
    result = original.accounting(model, spec, arm, metadata)
    result.update(
        gate_scope=SCOPE,
        boundary="X + u(H) * (H + g(H)q(H) - X)",
        final_readout="raw H; no final boundary",
        pointwise_cost_note="Same full Transformer blocks and vocabulary projections as "
        "the paired continuous-only gate. No compute skipping.",
    )
    return result


@contextmanager
def installed(config):
    old_class = original.update.PaperUpdateGateGPT
    try:
        original.update.PaperUpdateGateGPT = JointUpdateGateGPT
        with original.installed(config):
            native.initialize = initialize
            native.accounting = accounting
            yield
    finally:
        original.update.PaperUpdateGateGPT = old_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "grid"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=("joint-token",), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference_path = ROOT / config["paired_recipe"]["config"]
    if digest(reference_path) != config["paired_recipe"]["sha256"]:
        raise ValueError("Pinned original Wang recipe changed")
    validate_config(config, json.loads(reference_path.read_text()))
    original.verify_input(config)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with installed(config):
            if args.stage == "preflight":
                original.preflight(
                    {**config, "_checked_config_sha256": digest(args.config)}, args.arm, args.base
                )
            elif args.stage == "train":
                depth.tracked_train(config, args.arm, args.base, args.config)
            else:
                depth.endpoint(config, args.arm, args.base, args.config)
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
