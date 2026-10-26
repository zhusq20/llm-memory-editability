"""Resumable, audited HotpotQA data preparation for the PD method comparison.

Only ``prepare`` loads a model, lazily, when generation is actually needed.
The author's CSV has no Hugging Face IDs: case IDs identify its zero-based rows.
Completed generations are authoritative append-only JSONL records; a restart
materializes missing derived rows without generating completed requests again.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AUTHOR_COMMIT = "2b72919a0836f97115f0396ee753bd284c8d91d6"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
QUESTION_TEMPLATE = """Here is a paragraph of text:
{context}

Please generate challenging five trivia questions based on this paragraph. Do not make the questions multiple-choice. Do not assume that the person answering the questions has access to the paragraph. The questions must be understandable without access to the text. Do not output anything except the questions and format your output as in the followimg example:
<question>What is the capital of Japan?</question>
<question>How many months are there in a year?</question>
<question>What was the first name of Reagan?</question>
<question>How many goals did Messi score during the calendar year 2012</question>
<question>Where is the Santa Monica pier located?</question>"""  # noqa: E501


def chat_user(question: str) -> str:
    """The author's closed-book user text, without extra answer instructions."""
    return question.strip()


def context_user(context: str, question: str) -> str:
    """The author's teacher user text, also suitable for the context baseline."""
    return f"{context.strip()}\n\n---\n\n{chat_user(question)}"


def question_user(context: str) -> str:
    """Author question-generation prompt; it contains no evaluation question."""
    return QUESTION_TEMPLATE.format(context=context)


def render_chat(tokenizer: Any, user: str) -> str:
    """Render the shared Qwen chat template with thinking explicitly disabled."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def final_answer_text(text: str) -> str:
    """Apply Qwen's final-answer boundary without teaching accidental thinking text."""
    answer = text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in answer or "</think>" in answer or not answer:
        raise ValueError("Missing final answer or unclosed thinking block")
    return answer


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append(path: Path, value: Any) -> None:
    with path.open("ab") as handle:
        handle.write(_canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    """Recover only a torn final write, preserving its bytes in a repair log."""
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, offset = [], 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Non-object JSONL record in {path}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt committed JSONL record in {path}:{index + 1}") from None
            _append(
                path.parent / "recovery.jsonl",
                {
                    "file": path.name,
                    "offset": offset,
                    "torn_bytes_b64": base64.b64encode(line).decode(),
                },
            )
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            break
        rows.append(row)
        offset += len(line)
    else:
        if data and not data.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
    return rows


def read_source(source: Path, n_cases: int) -> list[dict]:
    """Read the author's unquoted semicolon format, retaining context semicolons.

    This particular source is not a normally quoted CSV: literal double quotes
    belong to its text, and semicolons after the second separator belong to the
    context. Splitting on only the first two separators preserves both.
    """
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < n_cases or n_cases < 1:
        raise ValueError(f"Requested {n_cases} cases from {len(lines)} source rows")
    rows = []
    for index, line in enumerate(lines[:n_cases]):
        columns = line.split(";", 2)
        if len(columns) != 3 or not all(part.strip() for part in columns):
            raise ValueError(f"Invalid author source row {index}")
        question, answer, context = columns
        paragraphs = context.split("---")
        if any(not paragraph.strip() for paragraph in paragraphs):
            raise ValueError(f"Empty paragraph in author source row {index}")
        rows.append(
            {
                "case_id": f"author-csv-row-{index:04d}",
                "source_row_index": index,
                "source_id_kind": "zero_based_author_csv_row_not_hf_id",
                "question": question,
                "answer": answer,
                "context": context,
                "paragraphs": paragraphs,
            }
        )
    return rows


class _HFGenerator:
    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Scientific preparation requires a CUDA device")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], dtype=torch.float32, local_files_only=True
        ).to(config.get("device", "cuda"))
        self.model.eval()
        self.max_new_tokens = config["max_new_tokens"]

    def __call__(self, users: list[str], temperature: float, seed: int) -> list[dict]:
        torch, tokenizer = self.torch, self.tokenizer
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        prompts = [render_chat(tokenizer, user) for user in users]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=1.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        eos_ids.add(tokenizer.eos_token_id)
        rows = []
        for index, output in enumerate(outputs):
            tokens = output[inputs["input_ids"].shape[1] :].tolist()
            terminal = next((j for j, token in enumerate(tokens) if token in eos_ids), None)
            if terminal is not None:
                tokens = tokens[: terminal + 1]
            truncated = terminal is None and len(tokens) >= self.max_new_tokens
            rows.append(
                {
                    "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                    "raw_text": tokenizer.decode(tokens, skip_special_tokens=False),
                    "token_ids": tokens,
                    "input_tokens": int(inputs["attention_mask"][index].sum().item()),
                    "generated_tokens": len(tokens),
                    "truncated": truncated,
                    "finish_reason": "length" if truncated else "eos",
                }
            )
        return rows


def _load_generator(config: dict) -> _HFGenerator:
    return _HFGenerator(config)


def _settings(config: dict) -> dict:
    defaults = {
        "n_cases": 32,
        "questions_per_paragraph": 30,
        "seed": 1234,
        "generation_batch_size": 8,
        "question_temperature": 1.5,
        "pd_temperature": 1.5,
        "sft_temperature": 0.25,
        "max_new_tokens": 512,
        "max_failures_per_paragraph": 20,
    }
    result = {**defaults, **config}
    # Paths may be provided directly by launchers, but provenance must be JSON.
    result = json.loads(json.dumps(result, default=str))
    for field in ("n_cases", "questions_per_paragraph", "generation_batch_size", "max_new_tokens"):
        if not isinstance(result[field], int) or result[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    if result["seed"] != 1234:
        raise ValueError("This preparation contract fixes seed=1234")
    if not 1 <= result["max_failures_per_paragraph"] <= 20:
        raise ValueError("Failure limit must be between 1 and 20 per paragraph")
    return result


def _freeze(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen input changed: {path}")
    else:
        _atomic_json(path, value)


def _validate_generations(outputs: list[dict], size: int, token_limit: int) -> None:
    if len(outputs) != size:
        raise ValueError("Generator returned the wrong batch size")
    for row in outputs:
        if not isinstance(row.get("text"), str):
            raise ValueError("Generator output is missing text")
        if not isinstance(row.get("truncated"), bool):
            raise ValueError("Generator output is missing truncation status")
        count = row.get("generated_tokens")
        if not isinstance(count, int) or not 0 <= count <= token_limit:
            raise ValueError("Invalid generated token count")
        if row.get("finish_reason") not in {"eos", "length"}:
            raise ValueError("Generator output is missing finish reason")
        if row["truncated"] != (row["finish_reason"] == "length"):
            raise ValueError("Inconsistent truncation metadata")


def _prepare(config: dict, shared: Path) -> dict:
    cfg = _settings(config)
    source = Path(cfg["source_csv"])
    all_cases = read_source(source, cfg["n_cases"])
    start, stop = cfg.get("case_start", 0), cfg.get("case_stop", len(all_cases))
    if (
        not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= len(all_cases)
    ):
        raise ValueError("Invalid source case range")
    cases = all_cases[start:stop]
    paragraphs = [
        {
            "paragraph_id": f"{case['case_id']}:p{index:02d}",
            "case_id": case["case_id"],
            "context": text,
        }
        for case in cases
        for index, text in enumerate(case["paragraphs"])
    ]
    expected_rows = len(paragraphs) * cfg["questions_per_paragraph"]
    if cfg.get("expected_paragraphs", len(paragraphs)) != len(paragraphs):
        raise ValueError("Author source paragraph count differs from the frozen configuration")
    if cfg.get("expected_training_rows", expected_rows) != expected_rows:
        raise ValueError("Expected training row count differs from the source and question budget")
    metadata_files = ("config.json", "tokenizer_config.json", "tokenizer.json")
    model_metadata = {
        name: _sha(Path(cfg["model_path"]) / name)
        for name in metadata_files
        if (Path(cfg["model_path"]) / name).is_file()
    }
    provenance = {
        "config": cfg,
        "config_sha256": _digest(cfg),
        "source_csv_sha256": _sha(source),
        "source_commit": cfg.get("source_commit", AUTHOR_COMMIT),
        "source_repository": "https://github.com/kallekku/prompt-distillation",
        "source_id_kind": "zero_based_author_csv_row_not_hf_id",
        "module_sha256": _sha(Path(__file__)),
        "prompt_sha256": _digest({"question": QUESTION_TEMPLATE, "system": SYSTEM}),
        "model_metadata_sha256": model_metadata,
        "model_weights_hash_scope": "not_hashed_here; launcher must freeze model weights",
        "paragraph_count": len(paragraphs),
        "paragraph_counts_by_case": {row["case_id"]: len(row["paragraphs"]) for row in cases},
        "expected_training_rows": expected_rows,
        "precision": "FP32 master weights; CUDA BF16 autocast for forward/generate",
        "precision_difference": "Engineering adaptation of the paper's BF16 implementation",
        "sampling": "top_k=50, top_p=1.0; seed1234-derived stable request-batch seeds",
        "evaluation_isolation": "Only exact question-text overlap excluded; no answer filtering",
        "paragraph_policy": (
            "All available paragraphs in selected source range; no replacement or fill"
        ),
        "case_range": [start, stop],
        "evaluation_exclusion_case_count": len(all_cases),
    }
    _freeze(shared / "provenance.json", provenance)
    _freeze(shared / "eval.json", cases)
    complete_path = shared / "prepare-complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        for name, expected_hash in complete["files_sha256"].items():
            if _sha(shared / name) != expected_hash:
                raise ValueError(f"Completed preparation artifact changed: {name}")
        return complete

    question_raw_path = shared / "questions-raw.jsonl"
    answer_raw_path = shared / "answers-raw.jsonl"
    questions_path = shared / "questions.jsonl"
    train_path = shared / "train.jsonl"
    for path in (question_raw_path, answer_raw_path, questions_path, train_path):
        path.touch(exist_ok=True)
    question_raw = _read_jsonl(question_raw_path)
    answer_raw = _read_jsonl(answer_raw_path)
    questions = defaultdict(list)
    failures, attempts, answer_attempts = Counter(), Counter(), Counter()
    excluded = 0
    by_paragraph = {row["paragraph_id"]: row for row in paragraphs}
    seen_request_ids = set()
    for row in question_raw + answer_raw:
        if row["request_id"] in seen_request_ids or row["paragraph_id"] not in by_paragraph:
            raise ValueError("Duplicate request or unknown paragraph in generation cache")
        seen_request_ids.add(row["request_id"])
        failures[row["paragraph_id"]] += int(row["failed"])
    for row in question_raw:
        pid = row["paragraph_id"]
        if row["attempt"] != attempts[pid]:
            raise ValueError("Question cache attempt ordering changed")
        attempts[pid] += 1
        excluded += row.get("excluded_eval_questions", 0)
        for sample in row.get("accepted", []):
            expected_id = f"{pid}:q{len(questions[pid]):03d}"
            if sample["id"] != expected_id:
                raise ValueError("Question cache sample ordering changed")
            questions[pid].append(sample)
    saved_question_rows = _read_jsonl(questions_path)
    saved_questions = {row["id"]: row for row in saved_question_rows}
    if len(saved_questions) != len(saved_question_rows):
        raise ValueError("Duplicate question sample in cache")
    all_cached_questions = {row["id"]: row for group in questions.values() for row in group}
    if not saved_questions.keys() <= all_cached_questions.keys():
        raise ValueError("Question samples have no committed raw generation")
    for qid, sample in all_cached_questions.items():
        if qid in saved_questions and saved_questions[qid] != sample:
            raise ValueError("Question sample differs from its raw generation")
        if qid not in saved_questions:
            _append(questions_path, sample)
    answers = {}
    for row in answer_raw:
        key = (row["sample_id"], row["kind"])
        if row["sample_id"] not in all_cached_questions or row["kind"] not in {"pd", "sft"}:
            raise ValueError("Answer cache has an unknown question or objective")
        if row["attempt"] != answer_attempts[key]:
            raise ValueError("Answer cache attempt ordering changed")
        answer_attempts[key] += 1
        if not row["failed"]:
            if key in answers:
                raise ValueError("Duplicate successful answer in cache")
            answers[key] = row["generation"]
    generator = None
    eval_questions = {row["question"].strip() for row in all_cases}

    def report(status: str, error: str | None = None) -> dict:
        value = {
            "status": status,
            "error": error,
            "paragraph_count": len(paragraphs),
            "expected_training_rows": expected_rows,
            "questions": sum(map(len, questions.values())),
            "questions_by_paragraph": {p: len(questions[p]) for p in by_paragraph},
            "incomplete_paragraphs": {
                p: len(questions[p])
                for p in by_paragraph
                if len(questions[p]) != cfg["questions_per_paragraph"]
            },
            "answers": len(answers),
            "truncated_answers": dict(
                Counter(kind for (_, kind), output in answers.items() if output["truncated"])
            ),
            "failures_by_paragraph": dict(failures),
            "excluded_eval_questions": excluded,
        }
        _atomic_json(shared / "prepare-status.json", value)
        return value

    def generate(tasks: list[dict], temperature: float, raw_path: Path) -> list[dict]:
        nonlocal generator
        if generator is None:
            generator = _load_generator(cfg)
        request_ids = [task["request_id"] for task in tasks]
        seed = int(_digest([cfg["seed"], request_ids])[:8], 16) % (2**31)
        for task in tasks:
            task["sampling_seed"] = seed
            task["temperature"] = temperature
            task["prompt_sha256"] = _digest(task["user"])
        try:
            outputs = generator([task["user"] for task in tasks], temperature, seed)
            _validate_generations(outputs, len(tasks), cfg["max_new_tokens"])
            return outputs
        except Exception as exc:
            for task in tasks:
                record = {k: v for k, v in task.items() if k != "user"}
                record.update(failed=True, error=f"{type(exc).__name__}: {exc}")
                _append(raw_path, record)
                failures[task["paragraph_id"]] += 1
            raise

    try:
        while True:
            pending = [
                p
                for p in paragraphs
                if len(questions[p["paragraph_id"]]) < cfg["questions_per_paragraph"]
            ]
            if not pending:
                break
            tasks = []
            for paragraph in pending[: cfg["generation_batch_size"]]:
                pid = paragraph["paragraph_id"]
                if failures[pid] >= cfg["max_failures_per_paragraph"]:
                    raise RuntimeError(f"Failure limit reached with incomplete paragraph {pid}")
                tasks.append(
                    {
                        "request_id": f"{pid}:questions:{attempts[pid]:04d}",
                        "paragraph_id": pid,
                        "attempt": attempts[pid],
                        "user": question_user(paragraph["context"]),
                    }
                )
            outputs = generate(tasks, cfg["question_temperature"], question_raw_path)
            for task, output in zip(tasks, outputs, strict=True):
                pid = task["paragraph_id"]
                extracted = [
                    x.strip()
                    for x in re.findall(
                        r"<question>(.*?)</question>", output["text"], flags=re.DOTALL
                    )
                ]
                selected, dropped = [], 0
                for question in extracted:
                    if question in eval_questions:
                        dropped += 1
                    elif question and question.lower() != "q1":
                        selected.append(question)
                remaining = cfg["questions_per_paragraph"] - len(questions[pid])
                accepted = [
                    {
                        "id": f"{pid}:q{len(questions[pid]) + index:03d}",
                        **by_paragraph[pid],
                        "question": question,
                    }
                    for index, question in enumerate(selected[: min(5, remaining)])
                ]
                record = {k: v for k, v in task.items() if k != "user"}
                record.update(
                    generation=output,
                    extracted_questions=extracted,
                    accepted=accepted,
                    excluded_eval_questions=dropped,
                    failed=len(selected) < min(5, remaining),
                )
                _append(question_raw_path, record)
                attempts[pid] += 1
                failures[pid] += int(record["failed"])
                excluded += dropped
                questions[pid].extend(accepted)
                for sample in accepted:
                    _append(questions_path, sample)
            report("questions")

        ordered = [sample for p in paragraphs for sample in questions[p["paragraph_id"]]]
        saved_train_rows = _read_jsonl(train_path)
        saved_train = {row["id"]: row for row in saved_train_rows}
        if len(saved_train) != len(saved_train_rows):
            raise ValueError("Duplicate training row in cache")

        def training_row(sample: dict) -> dict:
            row = dict(sample)
            for kind in ("pd", "sft"):
                output = answers[(sample["id"], kind)]
                row[f"{kind}_answer"] = output["text"].strip()
                if cfg.get("answer_format") == "qwen_final_answer_only":
                    row[f"{kind}_answer"] = final_answer_text(output["text"])
                    row[f"{kind}_format_normalized"] = (
                        row[f"{kind}_answer"] != output["text"].strip()
                    )
                    row[f"{kind}_raw_answer_sha256"] = _digest(output["text"])
                for field in ("truncated", "generated_tokens", "finish_reason"):
                    row[f"{kind}_{field}"] = output[field]
            return row

        def materialize() -> None:
            known_ids = {sample["id"] for sample in ordered}
            if not saved_train.keys() <= known_ids:
                raise ValueError("Unknown training sample in cache")
            for sample in ordered:
                qid = sample["id"]
                ready = all((qid, kind) in answers for kind in ("pd", "sft"))
                if qid in saved_train and (not ready or saved_train[qid] != training_row(sample)):
                    raise ValueError("Training row differs from committed answer generations")
                if ready and qid not in saved_train:
                    row = training_row(sample)
                    _append(train_path, row)
                    saved_train[qid] = row

        materialize()
        for kind in ("pd", "sft"):
            while True:
                pending = [sample for sample in ordered if (sample["id"], kind) not in answers]
                if not pending:
                    break
                tasks = []
                reserved = Counter()
                for sample in pending:
                    pid, qid = sample["paragraph_id"], sample["id"]
                    if failures[pid] >= cfg["max_failures_per_paragraph"]:
                        raise RuntimeError(
                            f"Failure limit reached with incomplete answers in {pid}"
                        )
                    if failures[pid] + reserved[pid] >= cfg["max_failures_per_paragraph"]:
                        continue
                    reserved[pid] += 1
                    attempt = answer_attempts[(qid, kind)]
                    tasks.append(
                        {
                            "request_id": f"{qid}:{kind}:{attempt:04d}",
                            "paragraph_id": pid,
                            "sample_id": qid,
                            "kind": kind,
                            "attempt": attempt,
                            "user": context_user(sample["context"], sample["question"]),
                        }
                    )
                    if len(tasks) == cfg["generation_batch_size"]:
                        break
                outputs = generate(tasks, cfg[f"{kind}_temperature"], answer_raw_path)
                for task, output in zip(tasks, outputs, strict=True):
                    record = {k: v for k, v in task.items() if k != "user"}
                    record.update(generation=output, failed=not bool(output["text"].strip()))
                    if cfg.get("answer_format") == "qwen_final_answer_only":
                        try:
                            final_answer_text(output["text"])
                        except ValueError as exc:
                            record.update(failed=True, format_error=str(exc))
                    _append(answer_raw_path, record)
                    key = (task["sample_id"], kind)
                    answer_attempts[key] += 1
                    failures[task["paragraph_id"]] += int(record["failed"])
                    if not record["failed"]:
                        answers[key] = output
                materialize()
                report(f"answers_{kind}")
        if len(ordered) != expected_rows or len(saved_train) != expected_rows:
            raise ValueError("Preparation is incomplete")
        if any(sample["question"].strip() in eval_questions for sample in ordered):
            raise ValueError("Evaluation question leaked into training")
        if any(
            not row[f"{kind}_answer"].strip()
            for row in saved_train.values()
            for kind in ("pd", "sft")
        ):
            raise ValueError("Empty training answer")
        complete = report("complete")
        complete["files_sha256"] = {
            name: _sha(shared / name)
            for name in (
                "train.jsonl",
                "eval.json",
                "provenance.json",
                "questions.jsonl",
                "questions-raw.jsonl",
                "answers-raw.jsonl",
            )
        }
        _atomic_json(complete_path, complete)
        return complete
    except BaseException as exc:
        report("incomplete", f"{type(exc).__name__}: {exc}")
        raise


def prepare(config: dict, batch: Path) -> dict:
    """Prepare ``batch/shared``; raise on incomplete data and return its manifest.

    Required configuration: ``source_csv`` and local ``model_path``. Generation
    defaults and optional expected counts are frozen in ``provenance.json``.
    The nonblocking lock prevents two launchers from consuming the same cache.
    ``eval.json`` is a list; context is a string and paragraphs is its full list.
    """
    shared = Path(batch) / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    with (shared / "prepare.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(config, shared)
