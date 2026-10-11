"""Prepare the PERK author's released QA2 data without rewriting its supervision."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

AUTHOR_DATASET = "zechen-nlp/babilong"
AUTHOR_REVISION = "2c862eac123b7efd3980affc24625b84baaa9848"
AUTHOR_SUBSET = "babilong_qa2_8k"
AUTHOR_PARQUET_SHA256 = {
    "train": "30cbb202c123befbe29b3b870851839e2511e8aeb727770f06aa9792335bc770",
    "test": "1785dca8e478247dd5f3c18d93096d62f3cb8e2aa12d63ebf7b2bfd92ad9351e",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(row: dict) -> str:
    payload = {key: row[key] for key in ("context", "facts", "question", "answer", "support")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_row(row: dict) -> None:
    for key in ("context", "facts", "support"):
        if not isinstance(row.get(key), list) or not row[key]:
            raise ValueError(f"{key} must be a nonempty list")
        if not all(isinstance(value, str) and value for value in row[key]):
            raise ValueError(f"{key} must contain nonempty strings")
    for key in ("question", "answer"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{key} must be a nonempty string")


def qa_response(row: dict) -> str:
    """This is the published PERK collator's complete support-and-answer target."""
    support = "\n".join(row["support"])
    return f"<support>{support}</support>\n<output>{row['answer']}</output>"


def qa_prompt(row: dict) -> str:
    return f"You have memorized {len(row['context'])} documents.\n{row['question']}\nAnswer:"


def fticr_example(row: dict) -> dict:
    """Explicit reconstruction: indexed context followed by the author's QA suffix.

    The PERK release does not publish the Figure 3 FT-ICR prompt. This formatting
    choice must be recorded as a reconstruction, not as an author-provided prompt.
    """
    documents = "\n".join(f"doc_{i}: {text}" for i, text in enumerate(row["context"]))
    return {
        "prompt": documents + "\n" + qa_prompt(row),
        "completion": qa_response(row),
        "source_id": row.get("source_id"),
    }


def partition_training(rows: list[dict], *, validation_count: int, seed: int) -> dict:
    if not 0 < validation_count < len(rows):
        raise ValueError("validation_count must leave nonempty training and validation splits")
    # Group exact duplicate samples before selection so they cannot straddle splits.
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(canonical_hash(row), []).append(i)
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    validation = set()
    for key in ordered:
        if len(validation) >= validation_count:
            break
        validation.update(groups[key])
    return {
        "train": [row for i, row in enumerate(rows) if i not in validation],
        "validation": [row for i, row in enumerate(rows) if i in validation],
    }


def numeric_summary(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty collection")
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[math.ceil(len(ordered) * 0.50) - 1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "sum": sum(ordered),
    }


def audit_rows(rows: list[dict], tokenizer=None) -> dict:
    counts = Counter()
    measurements: dict[str, list[int]] = {key: [] for key in ("chunks", "facts", "support")}
    if tokenizer is not None:
        measurements.update(
            {
                key: []
                for key in ("context_tokens", "largest_chunk_tokens", "fticr_tokens", "qa_tokens")
            }
        )
    for row in rows:
        validate_row(row)
        for key in ("chunks", "facts", "support"):
            measurements[key].append(len(row["context"] if key == "chunks" else row[key]))
        joined = "\n".join(row["context"])
        counts["support_absent_from_context"] += sum(text not in joined for text in row["support"])
        counts["support_absent_from_facts"] += sum(
            text not in row["facts"] for text in row["support"]
        )
        counts["facts_absent_from_context"] += sum(text not in joined for text in row["facts"])
        if tokenizer is not None:
            chunks = tokenizer(row["context"], add_special_tokens=False)["input_ids"]
            lengths = [len(chunk) for chunk in chunks]
            example = fticr_example(row)
            fticr = f"{example['prompt']} {example['completion']}{tokenizer.eos_token}"
            qa = f"{qa_prompt(row)} {qa_response(row)}{tokenizer.eos_token}"
            measurements["context_tokens"].append(sum(lengths))
            measurements["largest_chunk_tokens"].append(max(lengths))
            measurements["fticr_tokens"].append(
                len(tokenizer.encode(fticr, add_special_tokens=True))
            )
            measurements["qa_tokens"].append(len(tokenizer.encode(qa, add_special_tokens=True)))
    return {
        "rows": len(rows),
        "exact_duplicate_rows": len(rows) - len({canonical_hash(row) for row in rows}),
        "coverage": dict(counts),
        "measurements": {key: numeric_summary(values) for key, values in measurements.items()},
    }


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_author_data(
    raw_dir: str | Path,
    destination: str | Path,
    *,
    tokenizer=None,
    validation_count: int = 999,
    seed: int = 42,
) -> dict:
    import pyarrow.parquet as pq

    raw_dir, destination = Path(raw_dir), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    raw = {}
    inputs = {}
    for split in ("train", "test"):
        path = raw_dir / AUTHOR_SUBSET / f"{split}-00000-of-00001.parquet"
        digest = sha256_file(path)
        if digest != AUTHOR_PARQUET_SHA256[split]:
            raise ValueError(f"Unexpected author Parquet SHA256: {path}")
        rows = pq.read_table(path).to_pylist()
        expected = 9999 if split == "train" else 999
        if len(rows) != expected:
            raise ValueError(f"Unexpected author {split} count: {len(rows)}")
        raw[split] = [
            dict(row, source_id=f"author-{split}-{i:05d}", source_split=split, source_index=i)
            for i, row in enumerate(rows)
        ]
        inputs[split] = {"path": str(path.resolve()), "sha256": digest, "rows": len(rows)}
    splits = partition_training(raw["train"], validation_count=validation_count, seed=seed)
    splits["test"] = raw["test"]
    fingerprints = {key: {canonical_hash(row) for row in rows} for key, rows in splits.items()}
    overlap = {
        f"{left}/{right}": len(fingerprints[left] & fingerprints[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    if any(overlap.values()):
        raise ValueError(f"Exact sample overlap between splits: {overlap}")
    audits = {key: audit_rows(rows, tokenizer) for key, rows in splits.items()}
    author_root = destination / "babilong" / "qa2_8k"
    baseline_root = destination / "fticr"
    author_root.mkdir(parents=True)
    baseline_root.mkdir()
    outputs = {}
    for split, rows in splits.items():
        outputs[split] = {}
        for kind, path, values in (
            ("perk", author_root / f"{split}.jsonl", rows),
            ("fticr", baseline_root / f"{split}.jsonl", [fticr_example(row) for row in rows]),
        ):
            _write_jsonl(path, values)
            outputs[split][kind] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    manifest = {
        "dataset": AUTHOR_DATASET,
        "revision": AUTHOR_REVISION,
        "subset": AUTHOR_SUBSET,
        "data_dir": str(destination.resolve()),
        "dataset_name": "babilong",
        "dataset_subset": "qa2_8k",
        "source_inputs": inputs,
        "outputs": outputs,
        "split_rule": {
            "seed": seed,
            "requested_validation_count": validation_count,
            "selection": "seeded shuffled canonical hashes; exact duplicate samples grouped",
            "validation_source": "author train only; author test untouched",
            "original_figure3_validation_split": "not disclosed",
            "independence_limit": (
                "no published story/world IDs; sample split is not world independence"
            ),
        },
        "audits": audits,
        "exact_sample_overlap": overlap,
        "supervision": (
            "published complete support text followed by output answer; same target in both arms"
        ),
        "fticr_prompt_status": "reconstructed indexed documents + published PERK QA suffix",
        "context_length": (
            "author nominal 8K data preserved; GPT2 actual lengths measured, no truncation"
        ),
        "figure3_exact_sample_identity": "not established by the dataset card",
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest
