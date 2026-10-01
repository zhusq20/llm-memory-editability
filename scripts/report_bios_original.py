#!/usr/bin/env python3
"""Aggregate frozen original bioS predictions without loading a model."""

import argparse
from pathlib import Path

from llm_memory_editability.bios_original_report import run_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bios-original-development-v1.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/development-artifacts/bios-original-review-v1")
    )
    args = parser.parse_args()
    run_report(args.config, args.output)


if __name__ == "__main__":
    main()
