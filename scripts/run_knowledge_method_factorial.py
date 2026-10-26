#!/usr/bin/env python3
"""Stage runner for the expanded, eight-arm Qwen knowledge comparison."""

import argparse
import json
import traceback
from pathlib import Path

from llm_memory_editability import knowledge_method_diagnosis as K


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare-shard", "merge", "train", "evaluate", "audit", "regrade-old"],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--shard")
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    name = args.arm or args.shard or "shared"
    status = args.batch / "stage-status" / f"{name}-{args.stage}.json"
    K.write(status, {"state": "running", "stage": args.stage, "name": name})
    run = None
    try:
        if args.stage == "prepare-shard":
            from llm_memory_editability.knowledge_method_expansion import prepare_shard

            run = K.wandb_run(config, args.batch, f"prepare-{args.shard}")
            prepare_shard(config, args.batch, args.shard)
        elif args.stage == "merge":
            from llm_memory_editability.knowledge_method_expansion import merge

            merge(config, args.batch, Path("/old-batch/shared"))
        elif args.stage in ["train", "audit"]:
            from llm_memory_editability import knowledge_method_factorial as F

            if args.arm not in config["arms"]:
                raise ValueError("A configured arm is required")
            if args.stage == "train":
                F.train(config, args.batch, args.arm, args.smoke_steps)
            else:
                F.audit(config, args.batch, args.arm)
        elif args.stage == "evaluate":
            from llm_memory_editability.knowledge_method_paper_eval import evaluate

            evaluate(config, args.batch, args.arm)
        else:
            raise ValueError("Use the paper-evaluation CLI for regrade-old")
        K.write(status, {"state": "complete", "stage": args.stage, "name": name})
    except Exception:
        K.write(
            status,
            {
                "state": "failed",
                "stage": args.stage,
                "name": name,
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
