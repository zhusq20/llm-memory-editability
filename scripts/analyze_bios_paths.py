"""CPU saved-prediction audit and separately scheduled autonomous two-step inference."""

import argparse
from pathlib import Path

from llm_memory_editability.bios_path_diagnostics import analyze_saved, diagnose_two_step_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    repository = Path(__file__).resolve().parents[1]
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--repository", type=Path, default=repository)
    analyze.add_argument(
        "--output", type=Path, default=repository / "results/bios-mechanism-dev-v1/p0"
    )
    infer = subparsers.add_parser("two-step")
    infer.add_argument("--run", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--learning-steps", "--steps", type=int, nargs="+", default=[15360])
    infer.add_argument("--include-edits", action="store_true")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--batch-size", type=int, default=512)
    infer.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "analyze":
        analyze_saved(args.repository, args.output)
    else:
        import torch

        torch.set_num_threads(args.threads)
        diagnose_two_step_run(
            args.run,
            args.output,
            args.device,
            args.learning_steps,
            args.include_edits,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
