"""Expand the fixed author corpus with disjoint preparation shards and explicit reuse."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import knowledge_method_data as D

GENERATION_KEYS = (
    "model",
    "model_revision",
    "questions_per_paragraph",
    "seed",
    "question_temperature",
    "pd_temperature",
    "sft_temperature",
    "max_new_tokens",
)


def shard_config(config, shard):
    if shard not in config["prepare_shards"]:
        raise ValueError(f"Unknown preparation shard: {shard}")
    item = config["prepare_shards"][shard]
    return {
        **config,
        "case_start": item["start"],
        "case_stop": item["stop"],
        "expected_paragraphs": item["paragraphs"],
        "expected_training_rows": item["paragraphs"] * config["questions_per_paragraph"],
    }


def prepare_shard(config, batch, shard):
    return D.prepare(shard_config(config, shard), Path(batch) / "preparation" / shard)


def validate_parts(config, parts):
    """Verify coverage, frozen generation settings, and committed input hashes."""
    cases = D.read_source(Path(config["source_csv"]), config["n_cases"])
    covered = []
    inputs = []
    for name, shared, start, stop in parts:
        marker = json.loads((shared / "prepare-complete.json").read_text())
        if marker.get("status") != "complete":
            raise ValueError(f"Incomplete preparation: {name}")
        provenance = json.loads((shared / "provenance.json").read_text())
        for key in GENERATION_KEYS:
            if provenance["config"].get(key) != config[key]:
                raise ValueError(f"Generation setting differs: {name}/{key}")
        if provenance["source_csv_sha256"] != D._sha(Path(config["source_csv"])):
            raise ValueError(f"Author data changed: {name}")
        actual_cases = json.loads((shared / "eval.json").read_text())
        if actual_cases != cases[start:stop]:
            raise ValueError(f"Case identities changed: {name}")
        for filename, expected in marker["files_sha256"].items():
            if D._sha(shared / filename) != expected:
                raise ValueError(f"Committed input changed: {name}/{filename}")
        covered.extend(range(start, stop))
        inputs.append(
            {
                "name": name,
                "shared": str(shared),
                "range": [start, stop],
                "train_sha256": D._sha(shared / "train.jsonl"),
            }
        )
    if sorted(covered) != list(range(config["n_cases"])):
        raise ValueError("Preparation partitions overlap or omit cases")
    return cases, inputs


def merge(config, batch, reuse_shared):
    """Assemble one common training file; never alter any source preparation."""
    batch, reuse_shared = Path(batch), Path(reuse_shared)
    shared = batch / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    if (shared / "prepare-complete.json").exists():
        marker = json.loads((shared / "prepare-complete.json").read_text())
        for name, expected in marker["files_sha256"].items():
            if D._sha(shared / name) != expected:
                raise ValueError(f"Completed merge changed: {name}")
        return marker
    parts = [("reused-development", reuse_shared, 0, config["reuse_cases"])]
    parts.extend(
        (name, batch / "preparation" / name / "shared", item["start"], item["stop"])
        for name, item in config["prepare_shards"].items()
    )
    cases, inputs = validate_parts(config, parts)
    questions = {case["question"].strip() for case in cases}
    paragraphs = {
        f"{case['case_id']}:p{i:02d}": text
        for case in cases
        for i, text in enumerate(case["paragraphs"])
    }
    counts, seen, normalized = Counter(), set(), Counter()
    format_changes = []
    temporary = shared / "train.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as output:
        for _, source, _, _ in parts:
            with (source / "train.jsonl").open() as handle:
                for line in handle:
                    row = json.loads(line)
                    pid, qid = row["paragraph_id"], row["id"]
                    if qid in seen or pid not in paragraphs or row["context"] != paragraphs[pid]:
                        raise ValueError("Duplicate sample or mismatched paragraph content")
                    if row["case_id"] != pid.split(":p")[0] or not qid.startswith(pid + ":q"):
                        raise ValueError("Question identity does not belong to paragraph")
                    if row["question"].strip() in questions:
                        raise ValueError(
                            "Exact evaluation question found in reused/generated training data"
                        )
                    for kind in ("pd", "sft"):
                        if not row[f"{kind}_answer"] or not isinstance(
                            row[f"{kind}_truncated"], bool
                        ):
                            raise ValueError("Missing answer or truncation status")
                        if config.get("answer_format") == "qwen_final_answer_only":
                            original = row[f"{kind}_answer"]
                            answer = D.final_answer_text(original)
                            if answer != original:
                                format_changes.append(
                                    {"id": qid, "kind": kind, "before": original, "after": answer}
                                )
                                row[f"{kind}_answer"] = answer
                                row[f"{kind}_format_normalized"] = True
                                row[f"{kind}_raw_answer_sha256"] = D._digest(original)
                            normalized[kind] += int(row.get(f"{kind}_format_normalized", False))
                    seen.add(qid)
                    counts[pid] += 1
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
    if len(seen) != config["expected_training_rows"] or set(counts) != set(paragraphs):
        raise ValueError("Incomplete common training coverage")
    if any(n != config["questions_per_paragraph"] for n in counts.values()):
        raise ValueError("Unequal question count per paragraph")
    temporary.replace(shared / "train.jsonl")
    D._atomic_json(shared / "eval.json", cases)
    D._atomic_json(
        shared / "answer-format-audit.json",
        {"normalized_counts": dict(normalized), "reused_answer_changes": format_changes},
    )
    D._atomic_json(
        shared / "provenance.json",
        {
            "config": config,
            "sources": inputs,
            "reuse_policy": (
                "Original 32-case source preserved; final-answer normalization is audited; "
                "no reused weights"
            ),
            "sampling_policy": (
                "Three independently seeded disjoint preparation shards; "
                "all eight arms share merged data"
            ),
        },
    )
    marker = {
        "status": "complete",
        "questions": len(seen),
        "paragraph_count": len(counts),
        "cases": len(cases),
        "reused_cases": config["reuse_cases"],
        "files_sha256": {
            name: D._sha(shared / name)
            for name in ("train.jsonl", "eval.json", "provenance.json", "answer-format-audit.json")
        },
    }
    D._atomic_json(shared / "prepare-complete.json", marker)
    return marker
