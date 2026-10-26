#!/usr/bin/env python3
"""Position-invariance arms on the two-graph multi-hop data, answer/EOS CE only.

Each arm sets its handoff, its position scheme (``rope`` or ``absolute``) and, through
``training_overrides``, either the loop-native recipe (random rounds, two-segment deep
supervision, cosine LR, beta2 0.95) or the original DiscoLoop recipe (fixed four rounds,
one segment, constant LR after warmup, beta2 0.999). Training, checkpoints, EMA
shadow weights (evaluation only), tracking and the endpoint trace reuse the unchanged
functions of scripts/run_loop_native_handoff.py; only model construction and the
per-arm configuration differ.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_native_handoff as base  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.loop_native_handoff import HANDOFFS  # noqa: E402
from llm_memory_editability.loop_position_invariance import (  # noqa: E402
    POSITIONS,
    LoopPositionGPT,
)

BATCH = "loop-position-invariance-development-v1"
RECIPE_KEYS = {
    "segments",
    "first_segment_rounds",
    "later_segment_rounds",
    "final_lr_fraction",
    "betas",
}


def validate_config(config):
    if config.get("batch") != BATCH:
        raise ValueError("Unexpected batch")
    for name, arm in config["arms"].items():
        if arm["handoff"] not in HANDOFFS or arm["position"] not in POSITIONS:
            raise ValueError(f"Unknown handoff or position: {name}")
        if set(arm.get("training_overrides", {})) - RECIPE_KEYS:
            raise ValueError(f"Overrides may change only the recipe: {name}")
    if config["tracking"]["entity"] != "zsqzz":
        raise ValueError("New batches track to the zsqzz team")


def arm_config(config, name):
    """The batch configuration with this arm's recipe applied to ``training``."""
    arm = config["arms"][name]
    training = {**config["training"], **arm.get("training_overrides", {})}
    for key in ("first_segment_rounds", "later_segment_rounds"):
        low, high = training[key]
        if not 1 <= low <= high:
            raise ValueError(f"Invalid round range for {name}: {key}")
    return {**config, "training": training}


def initialize(config, name, metadata, device):
    training, arm = config["training"], config["arms"][name]
    torch.manual_seed(training["initialization"])
    hf = GPT2Config(
        **config["model"],
        n_layer=training["unique_layers"],
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    model = LoopPositionGPT(
        hf,
        training["evaluation_repeats"],
        handoff=arm["handoff"],
        position=arm["position"],
        rope_base=config["rope_base"],
        **config["handoff"],
    ).to(device)
    torch.manual_seed(config["dropout_seed"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "smoke-grid", "train", "grid"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--limit-rows", type=int, help="smoke stages only")
    parser.add_argument("--smoke-steps", type=int, help="smoke stage only")
    parser.add_argument("--smoke-grid-rounds", type=int, help="smoke-grid stage only")
    parser.add_argument("--threads", type=int, help="smoke stages only")
    args = parser.parse_args()
    engineering = (args.limit_rows, args.smoke_steps, args.smoke_grid_rounds, args.threads)
    if any(engineering) and not args.stage.startswith("smoke"):
        raise ValueError("Engineering overrides apply to smoke stages only")
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.arm not in config["arms"]:
        raise ValueError(f"Unknown arm: {args.arm}")
    config = arm_config(config, args.arm)
    if args.smoke_steps:
        config["smoke_steps"] = args.smoke_steps
    if args.smoke_grid_rounds:
        config["grid_rounds"] = args.smoke_grid_rounds
        config["grid_joint_rounds"] = [
            r for r in config["grid_joint_rounds"] if r <= args.smoke_grid_rounds
        ]
    base.initialize = initialize  # The reused train/grid build models through this name.
    torch.set_num_threads(args.threads or config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    elif args.stage in ("train", "grid"):
        raise ValueError("Scientific stages run on CUDA")
    try:
        if args.stage == "smoke":
            out = args.base / "smoke" / args.arm
            base.train(config, args.arm, out, True, args.config, None, device, args.limit_rows)
        elif args.stage == "smoke-grid":
            base.grid(config, args.arm, args.base, args.config, device, "smoke", args.limit_rows)
        elif args.stage == "train":
            out = args.base / "runs" / args.arm
            holder = {}

            def track(metadata):
                holder["log"] = base.tracker_for(config, args.arm, out, args.config)
                return holder["log"]

            try:
                base.train(config, args.arm, out, False, args.config, track, device)
            except BaseException:
                if "log" in holder:
                    holder["log"].run.finish(exit_code=1)
                raise
            holder["log"].run.finish()
        else:
            base.grid(config, args.arm, args.base, args.config, device)
    except BaseException as error:
        write_json(
            args.base / "failures" / f"{args.arm}-{args.stage}.json",
            {"utc": utc(), "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
