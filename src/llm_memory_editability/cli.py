"""Command-line interface for the deterministic matrix illustration."""

import argparse
import json
from pathlib import Path

from .experiments import run_minimal_example


def main(argv: list[str] | None = None) -> int:
    """Print the analytical report as JSON and optionally save a copy."""
    parser = argparse.ArgumentParser(description="Run the analytical low-rank editability example.")
    parser.add_argument(
        "--rank", type=int, default=1, help="total rank budget: 0, 1 (default), or 2"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-10, help="positive absolute singular-value tolerance"
    )
    parser.add_argument(
        "--output", type=Path, help="also save JSON to this path (creates parent directories)"
    )
    args = parser.parse_args(argv)
    try:
        report = run_minimal_example(rank=args.rank, atol=args.atol)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(serialized)
    return 0
