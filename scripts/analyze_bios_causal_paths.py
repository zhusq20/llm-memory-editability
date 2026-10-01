"""Run one predeclared offline causal-path diagnostic job on a frozen model."""

import argparse
from pathlib import Path

from llm_memory_editability.bios_causal_paths import diagnose_causal_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, default=15360)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--per-population", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    diagnose_causal_run(
        args.run, args.output, args.step, args.device, args.batch_size, args.per_population
    )


if __name__ == "__main__":
    main()
