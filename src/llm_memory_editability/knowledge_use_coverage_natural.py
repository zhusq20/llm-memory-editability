"""Pure-data preparation for the natural-document use-coverage application check.

Rows 0:32 are development records; rows 32:96 are the application comparison.
Both ranges were evaluated in the earlier 1000-case factorial: neither is a new
confirmation set. Existing questions are not verified atomic questions, and the
A/B selection may change fact coverage as well as wording. This application
check cannot establish the controlled-new-fact experiment's causal attribution.

Training loaders never retain the CSV's evaluation columns. Only the separate
evaluation loader returns them. No models, network operations or writes occur.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .knowledge_method_data import context_user, final_answer_text, read_source

SPLIT_RANGES = {"development": (0, 32), "application": (32, 96)}
USE_CATEGORIES = ("composition", "comparison", "judgment")
ORIGINAL_CATEGORY = "auto-generated-original-questions"
FACT_ID_KIND = "source_paragraph_not_verified_atomic_fact"
TRAINING_CASE_FIELDS = {
    "case_id",
    "source_row_index",
    "context",
    "paragraphs",
    "split",
    "context_sha256",
}


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_range(split: str) -> tuple[int, int]:
    if split not in SPLIT_RANGES:
        raise ValueError(f"Unknown fixed split: {split}")
    return SPLIT_RANGES[split]


def _check_training_case(case: dict) -> None:
    if set(case) != TRAINING_CASE_FIELDS:
        raise ValueError("Training case must contain only the context-only schema")
    start, stop = _split_range(case["split"])
    index = case["source_row_index"]
    if not start <= index < stop or case["case_id"] != f"author-csv-row-{index:04d}":
        raise ValueError("Training case does not match its fixed source split")
    if case["context_sha256"] != _digest(case["context"]):
        raise ValueError("Training context hash changed")
    if case["paragraphs"] != case["context"].split("---"):
        raise ValueError("Paragraphs must exactly reconstruct the source context")
    if any(not paragraph.strip() for paragraph in case["paragraphs"]):
        raise ValueError("Empty source paragraph")


def load_training_cases(source_csv: Path, *, split: str) -> list[dict]:
    """Project source text onto backgrounds; evaluation columns are discarded.

    This deliberately does not call ``read_source``, which materializes test
    questions and answers. Context semicolons and quotes are preserved.
    """
    start, stop = _split_range(split)
    cases = []
    with Path(source_csv).open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if index >= stop:
                break
            if index < start:
                continue
            columns = line.rstrip("\r\n").split(";", 2)
            if len(columns) != 3 or not columns[2].strip():
                raise ValueError(f"Invalid source context at row {index}")
            context = columns[2]
            case = {
                "case_id": f"author-csv-row-{index:04d}",
                "source_row_index": index,
                "context": context,
                "paragraphs": context.split("---"),
                "split": split,
                "context_sha256": _digest(context),
            }
            _check_training_case(case)
            cases.append(case)
    if len(cases) != stop - start:
        raise ValueError(f"Source does not cover the fixed {split} range {start}:{stop}")
    return cases


def load_evaluation_cases(source_csv: Path, *, split: str) -> list[dict]:
    """Read held-out application questions independently of training preparation."""
    start, stop = _split_range(split)
    return [
        {
            **row,
            "id": f"{row['case_id']}:application-evaluation",
            "split": split,
            "category": "hotpotqa-multihop-application",
            "previously_evaluated_in_factorial": True,
        }
        for row in read_source(Path(source_csv), stop)[start:stop]
    ]


def select_existing_rows(cases: list[dict], source_train_jsonl: Path) -> dict[str, list[dict]]:
    """Select q000 / q000:004 / q000:002 for A/B/C without outcome selection.

    Selection uses the immutable original question index, not JSONL completion
    order. Reused answer fields (including truncation flags) remain unchanged.
    All arms receive the same full-case teacher context; answer trajectories
    were originally sampled using the narrower paragraph context, recorded below.
    """
    for case in cases:
        _check_training_case(case)
    by_case = {case["case_id"]: case for case in cases}
    if len(by_case) != len(cases):
        raise ValueError("Duplicate training case")
    path = Path(source_train_jsonl)
    source_hash = _file_sha256(path)
    selected: dict[str, dict[int, tuple[dict, int, str]]] = defaultdict(dict)
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            row = json.loads(raw_line)
            if row.get("case_id") not in by_case:
                continue
            case = by_case[row["case_id"]]
            match = re.fullmatch(re.escape(case["case_id"]) + r":p(\d{2}):q(\d{3})", row["id"])
            if not match:
                raise ValueError("Malformed original question ID")
            paragraph_index, question_index = map(int, match.groups())
            if question_index >= 4:
                continue
            pid = f"{case['case_id']}:p{paragraph_index:02d}"
            if row.get("paragraph_id") != pid or paragraph_index >= len(case["paragraphs"]):
                raise ValueError("Original question has an unknown paragraph")
            if row["context"].strip() != case["paragraphs"][paragraph_index].strip():
                raise ValueError("Original question context differs from the CSV background")
            if question_index in selected[pid]:
                raise ValueError("Duplicate original question index")
            if any(not isinstance(row.get(f"{k}_answer"), str) for k in ("pd", "sft")):
                raise ValueError("Both original answer trajectories are required")
            if any(not row[f"{k}_answer"].strip() for k in ("pd", "sft")):
                raise ValueError("Empty original answer trajectory")
            selected[pid][question_index] = (
                row,
                line_number,
                hashlib.sha256(raw_line).hexdigest(),
            )
    variants: dict[str, list[dict]] = {variant: [] for variant in "ABC"}
    for case in sorted(cases, key=lambda item: item["source_row_index"]):
        for paragraph_index in range(len(case["paragraphs"])):
            pid = f"{case['case_id']}:p{paragraph_index:02d}"
            if set(selected[pid]) != set(range(4)):
                raise ValueError(f"Missing fixed original questions q000:q004 for {pid}")
            for variant, count in (("A", 1), ("B", 4), ("C", 2)):
                for question_index in range(count):
                    source, line_number, row_hash = selected[pid][question_index]
                    variants[variant].append(
                        {
                            "id": f"{source['id']}:natural-{variant}",
                            "case_id": case["case_id"],
                            "fact_ids": [pid],
                            "fact_id_kind": FACT_ID_KIND,
                            "question": source["question"],
                            "context": case["context"],
                            "category": ORIGINAL_CATEGORY,
                            "split": case["split"],
                            "variant": variant,
                            "atomicity_verified": False,
                            **{
                                key: value
                                for key, value in source.items()
                                if key.startswith(("sft_", "pd_"))
                            },
                            "provenance": {
                                "source_path": str(path),
                                "source_file_sha256": source_hash,
                                "source_line_number": line_number,
                                "source_line_sha256": row_hash,
                                "source_id": source["id"],
                                "source_question_index": question_index,
                                "original_trajectory_context_sha256": _digest(source["context"]),
                                "teacher_context_sha256": case["context_sha256"],
                                "trajectory_source": "reused_original_paragraph_answers",
                            },
                        }
                    )
    return variants


def build_use_requests(cases: list[dict]) -> list[dict]:
    """Create at most 2*nparagraphs C requests per case, balanced by use type."""
    requests = []
    seen = set()
    for case in sorted(cases, key=lambda item: item["source_row_index"]):
        _check_training_case(case)
        if case["case_id"] in seen:
            raise ValueError("Duplicate training case")
        seen.add(case["case_id"])
        paragraphs = {
            f"{case['case_id']}:p{index:02d}": paragraph
            for index, paragraph in enumerate(case["paragraphs"])
        }
        for slot in range(2 * len(paragraphs)):
            category = USE_CATEGORIES[len(requests) % len(USE_CATEGORIES)]
            user = (
                "Create one short, specific trivia question requiring the requested use of facts. "
                "Use only the supplied document; do not add outside facts or assume the "
                "reader has the document. Do not provide the answer. "
                "Composition follows a shared entity through two relations: for example, "
                "ask the birthplace of a film's director using the film's director and "
                "that person's birthplace. Comparison asks about the same attribute of "
                "two entities, such as which of two dated films came first. Judgment asks "
                "a yes/no question about a specific factual claim. Do not ask 'what is the "
                "relationship', combine unrelated questions, refer to 'the document', or "
                "ask the answerer to cite evidence. Do not invent missing attributes or "
                "relations. Composition and comparison must substantively "
                "use two independently stated facts and cite at least two distinct exact "
                "evidence quotes. The quotes may come from the same paragraph: paragraph "
                "IDs identify source locations, not individual facts. Do not add an "
                "irrelevant second quote or split one assertion into overlapping snippets. "
                "The question must require both facts. Judgment must cite at least one quote. "
                "Return exactly one JSON object with keys question and evidence. "
                "Evidence is a list of objects with paragraph_id and quote. Use the short "
                "paragraph label (p00, p01, etc.) exactly as shown. Copy short exact quotes "
                "of at most 30 words; preserve punctuation and escape double quotation marks "
                "for valid JSON. Each quote must support a fact needed for the answer. "
                'If this use cannot be grounded, return {"unavailable": true} instead.\n\n'
                f"Requested use: {category}\nQuestion slot: {slot + 1}\n\n"
                + "\n\n".join(
                    f"[{pid.rsplit(':', 1)[-1]}]\n{text}" for pid, text in paragraphs.items()
                )
                + f"\n\nRequested use: {category}. Return only the JSON object."
            )
            requests.append(
                {
                    "id": f"{case['case_id']}:natural-use-{slot:03d}",
                    "case_id": case["case_id"],
                    "context": case["context"],
                    "paragraphs": paragraphs,
                    "category": category,
                    "split": case["split"],
                    "variant": "C",
                    "user": user,
                    "prompt_sha256": _digest(user),
                    "context_sha256": case["context_sha256"],
                }
            )
    return requests


def parse_use_question(request: dict, generated_text: str) -> dict:
    """Validate source quotations, retaining the need for semantic review.

    No test questions or answers are accepted as arguments. Invalid/unavailable
    outputs raise, so the caller preserves them as explicit generation failures.
    Exact quotations do not by themselves establish correct semantic grounding.
    """
    text = final_answer_text(generated_text)
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n") : -len("\n```")]
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"question", "evidence"}:
        raise ValueError("Question output is unavailable or has an invalid schema")
    question = parsed["question"]
    evidence = parsed["evidence"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty generated question")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Generated question requires source evidence")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"paragraph_id", "quote"}:
            raise ValueError("Invalid source-evidence schema")
        pid, quote = item["paragraph_id"], item["quote"]
        if isinstance(pid, str) and pid not in request["paragraphs"]:
            matches = [key for key in request["paragraphs"] if key.rsplit(":", 1)[-1] == pid]
            if len(matches) == 1:
                pid = matches[0]
                item["paragraph_id"] = pid
        if (
            not isinstance(pid, str)
            or pid not in request["paragraphs"]
            or not isinstance(quote, str)
            or not quote.strip()
            or quote not in request["paragraphs"][pid]
        ):
            raise ValueError("Generated evidence does not occur in the source paragraph")
    fact_ids = sorted({item["paragraph_id"] for item in evidence})
    distinct_quotes = {item["quote"].strip() for item in evidence}
    required = 1 if request["category"] == "judgment" else 2
    if len(distinct_quotes) < required:
        raise ValueError("Requested use requires more distinct source evidence quotes")
    return {
        "id": request["id"],
        "case_id": request["case_id"],
        "fact_ids": fact_ids,
        "fact_id_kind": FACT_ID_KIND,
        "question": question.strip(),
        "context": request["context"],
        "category": request["category"],
        "split": request["split"],
        "variant": "C",
        "evidence": evidence,
        "distinct_evidence_quotes": len(distinct_quotes),
        "semantic_grounding_status": "requires_review",
        "provenance": {
            "request_id": request["id"],
            "question_prompt_sha256": request["prompt_sha256"],
            "raw_question_generation_sha256": _digest(generated_text),
            "teacher_context_sha256": request["context_sha256"],
            "trajectory_source": "new_full_document_answers",
        },
    }


def answer_requests(question_row: dict) -> list[dict]:
    """Return independent off-policy PD/SFT answer requests for the GPU runner."""
    user = context_user(question_row["context"], question_row["question"])
    return [
        {
            "id": f"{question_row['id']}:{kind}",
            "sample_id": question_row["id"],
            "kind": kind,
            "temperature": temperature,
            "user": user,
            "prompt_sha256": _digest(user),
        }
        for kind, temperature in (("pd", 1.5), ("sft", 0.25))
    ]


def attach_answers(question_row: dict, *, pd_generation: dict, sft_generation: dict) -> dict:
    """Attach both sampled trajectories with truncation and raw-response hashes."""
    row = dict(question_row)
    for kind, generation in (("pd", pd_generation), ("sft", sft_generation)):
        truncated = generation.get("truncated")
        count = generation.get("generated_tokens")
        reason = generation.get("finish_reason")
        if not isinstance(truncated, bool) or reason not in {"length", "eos"}:
            raise ValueError("Answer generation needs explicit termination metadata")
        if truncated != (reason == "length") or type(count) is not int or count < 1:
            raise ValueError("Inconsistent answer generation metadata")
        row[f"{kind}_answer"] = final_answer_text(generation["text"])
        row[f"{kind}_raw_answer_sha256"] = _digest(generation)
        for field in ("truncated", "generated_tokens", "finish_reason"):
            row[f"{kind}_{field}"] = generation[field]
    return row


def coverage_statistics(variants: dict[str, list[dict]]) -> dict:
    """Summarize actual coverage; counts are not matched training-token budgets."""
    result: dict[str, Any] = {
        "application_only": True,
        "new_confirmation_set": False,
        "original_question_atomicity_verified": False,
        "fact_id_kind": FACT_ID_KIND,
        "interpretation": (
            "A/B may change paragraph-internal fact coverage, not only wording; "
            "C adds generated uses. Natural-document results test application effects."
        ),
        "variants": {},
    }
    for variant in "ABC":
        rows = variants[variant]
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or any(row["variant"] != variant for row in rows):
            raise ValueError("Duplicate rows or mismatched variant")
        generated = [row for row in rows if row["category"] in USE_CATEGORIES]
        per_case = Counter(row["case_id"] for row in generated)
        for case_id, count in per_case.items():
            backgrounds = {row["context"] for row in rows if row["case_id"] == case_id}
            if len(backgrounds) != 1 or count > 2 * len(next(iter(backgrounds)).split("---")):
                raise ValueError("Generated use count exceeds the fixed per-case maximum")
        result["variants"][variant] = {
            "rows": len(rows),
            "cases": len({row["case_id"] for row in rows}),
            "source_paragraphs": len({pid for row in rows for pid in row["fact_ids"]}),
            "categories": dict(Counter(row["category"] for row in rows)),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "generated_use_rows_by_case": dict(per_case),
            "semantic_review_pending": sum(
                row.get("semantic_grounding_status") == "requires_review" for row in rows
            ),
            "duplicate_question_texts": len(rows)
            - len({row["question"].strip().casefold() for row in rows}),
            "answer_trajectory_generated_tokens": {
                kind: sum(row.get(f"{kind}_generated_tokens", 0) for row in rows)
                for kind in ("pd", "sft")
            },
        }
    return result


def audit_evaluation_overlap(variants: dict[str, list[dict]], evaluation: list[dict]) -> list[dict]:
    """Post-generation audit only: report overlaps; never filter/reselect examples."""
    tests = defaultdict(list)
    for row in evaluation:
        tests[" ".join(row["question"].split()).casefold()].append(row["case_id"])
    return [
        {"variant": variant, "training_id": row["id"], "evaluation_case_ids": tests[key]}
        for variant, rows in variants.items()
        for row in rows
        if (key := " ".join(row["question"].split()).casefold()) in tests
    ]
