"""Download and audit the PERK author's full public BabiLong QA2 8K release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_memory_editability.perk_data import (
    AUTHOR_DATASET,
    AUTHOR_REVISION,
    AUTHOR_SUBSET,
    prepare_author_data,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import hf_hub_download

        for name in (
            "README.md",
            f"{AUTHOR_SUBSET}/train-00000-of-00001.parquet",
            f"{AUTHOR_SUBSET}/test-00000-of-00001.parquet",
        ):
            hf_hub_download(
                repo_id=AUTHOR_DATASET,
                repo_type="dataset",
                revision=AUTHOR_REVISION,
                filename=name,
                local_dir=args.raw_dir,
            )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    # The released nominal 8K samples exceed GPT2's native 1K table. This audit
    # counts tokens; it never passes them through a model or silently truncates.
    tokenizer.model_max_length = 1_000_000
    result = prepare_author_data(
        args.raw_dir,
        args.destination,
        tokenizer=tokenizer,
        validation_count=args.validation_count,
        seed=args.seed,
    )
    print(
        json.dumps(
            {"manifest": str(args.destination / "manifest.json"), "audits": result["audits"]}
        )
    )


if __name__ == "__main__":
    main()
