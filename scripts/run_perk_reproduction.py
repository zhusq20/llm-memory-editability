"""Run the author PERK method under explicitly reconstructed Figure 3 settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_memory_editability.perk_reproduction import (
    dataset_hashes,
    environment_manifest,
    evaluate_checkpoint,
    resolve_config,
    source_manifest,
    train,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("manifest", "train", "evaluate"))
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit-validation", type=int)
    parser.add_argument("--limit-evaluation", type=int)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    config = resolve_config(json.loads(Path(args.config).read_text()), args.output)
    if args.mode == "manifest":
        result = {
            "config": config,
            "source": source_manifest(args.source_dir),
            "datasets": dataset_hashes(config),
            "environment": environment_manifest(),
        }
        write_json(Path(args.output) / "preflight-manifest.json", result)
    elif args.mode == "train":
        result = train(
            config,
            args.source_dir,
            device=args.device,
            max_steps=args.max_steps,
            limit_validation=args.limit_validation,
            wandb=not args.no_wandb,
            resume=args.resume,
        )
    else:
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        result = evaluate_checkpoint(
            config,
            args.source_dir,
            args.checkpoint,
            device=args.device,
            split=args.split,
            limit=args.limit_evaluation,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
