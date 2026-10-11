"""Execute the declared PERK FT-ICR reconstruction with the official training loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_memory_editability.perk_fticr import evaluate, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--author-perk-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == "train":
        result = train(config, source_dir=args.source_dir, output=args.output)
    else:
        if args.author_perk_source is None:
            parser.error("--author-perk-source is required for original author scoring")
        result = evaluate(config, author_perk_source=args.author_perk_source, output=args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
