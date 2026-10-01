"""Run the locked L0/pos2 P4 necessity/restoration diagnostic on one original model."""

import argparse
import fcntl
from pathlib import Path

from llm_memory_editability.bios_mechanism_causal import run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".run.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_model(
            args.run,
            args.output,
            args.candidate_lock,
            device=args.device,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
