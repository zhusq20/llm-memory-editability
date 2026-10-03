"""Freeze and run the independent answer-only MLP editing calibration grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_memory_editability.depth_step_edit_calibration import (
    execute_calibration,
    load_prepared_calibration,
    prepare_calibration,
)
from llm_memory_editability.depth_step_mechanism import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rates", default=".0001,.001,.003")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    learning_rates = tuple(float(value) for value in arguments.learning_rates.split(","))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, world, source = load_run(arguments.run_dir, arguments.checkpoint, arguments.device)
    if (arguments.out / "calibration-config.json").exists():
        config, cases = load_prepared_calibration(
            model,
            world,
            source,
            arguments.out,
            learning_rates=learning_rates,
            n_cases=arguments.cases,
        )
    else:
        config, cases = prepare_calibration(
            model,
            world,
            source,
            arguments.out,
            learning_rates=learning_rates,
            n_cases=arguments.cases,
        )
    if not arguments.prepare_only:
        execute_calibration(model, world, config, cases, arguments.out, arguments.device)
    print(f"Calibration {'prepared' if arguments.prepare_only else 'completed'}: {arguments.out}")


if __name__ == "__main__":
    main()
