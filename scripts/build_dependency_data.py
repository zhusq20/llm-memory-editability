#!/usr/bin/env python3
"""Build the frozen causal adaptation of an NDR author dataset."""

import argparse
import json
from pathlib import Path

from llm_memory_editability.dependency_data import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("listops", "arithmetic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--training-panel-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    spec = {
        "task": args.task,
        "seed": args.seed,
        "evaluation_samples": args.evaluation_samples,
        "training_panel_size": args.training_panel_size,
    }
    if args.train_samples is not None:
        spec["train_samples"] = args.train_samples
    print(json.dumps(prepare(spec, args.output, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
