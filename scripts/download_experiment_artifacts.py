#!/usr/bin/env python3
"""Restore experiment artifacts from Hugging Face to their original local paths."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="zsqzz/llm-memory-editability")
    parser.add_argument("--revision", default="main", help="Dataset branch or commit SHA")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--include",
        action="append",
        help="Path glob to download; repeat for multiple batches (default: data, results, docs)",
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit("Install archive dependencies: pip install -e '.[archive]'") from error

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=args.include or ["data/**", "results/**", "docs/**"],
    )


if __name__ == "__main__":
    main()
