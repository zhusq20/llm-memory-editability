#!/usr/bin/env python3
"""Frozen stage runner for the Qwen knowledge-method diagnosis development batch."""

import argparse
import json
import traceback
from pathlib import Path

from llm_memory_editability import knowledge_method_diagnosis as K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["prepare", "train", "evaluate", "audit"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--arm", choices=["pd_lora", "sft_lora", "sft_full", "sft_down"])
    p.add_argument("--smoke-steps", type=int)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage != "prepare" and not args.arm:
        p.error("--arm is required for this stage")
    if args.smoke_steps is not None and args.stage != "train":
        p.error("--smoke-steps is only valid for train")
    out = args.batch / "stage-status" / f"{args.arm or 'shared'}-{args.stage}.json"
    K.write(out, {"state": "running", "stage": args.stage, "arm": args.arm})
    try:
        if args.stage == "prepare":
            from llm_memory_editability.knowledge_method_data import prepare

            run = K.wandb_run(config, args.batch, "prepare")
            try:
                prepare(config, args.batch)
            finally:
                run.finish()
        elif args.stage == "train":
            K.train(config, args.batch, args.arm, args.smoke_steps)
        else:
            getattr(K, args.stage)(config, args.batch, args.arm)
        K.write(out, {"state": "complete", "stage": args.stage, "arm": args.arm})
    except Exception:
        K.write(
            out,
            {
                "state": "failed",
                "stage": args.stage,
                "arm": args.arm,
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
