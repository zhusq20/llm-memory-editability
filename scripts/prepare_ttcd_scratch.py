#!/usr/bin/env python3
"""Prepare a small symbolic world and disjoint correction facts; starts no training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.ttcd_scratch import prepare_scratch_data  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ttcd-second-hop-scratch-v1.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    print(json.dumps(prepare_scratch_data(config), indent=2))


if __name__ == "__main__":
    main()
