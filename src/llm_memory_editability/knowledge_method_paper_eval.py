"""Paper grading protocol with identified judges; separate from frozen development scores.

Prompt/parser adapted from kallekku/prompt-distillation commit
2b72919a0836f97115f0396ee753bd284c8d91d6.

MIT License

Copyright (c) 2024-2025 System 2 AI and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import string
import time
from pathlib import Path


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


@contextlib.contextmanager
def _locked(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield output


def _cache(path):
    """Fail on corrupt records; preserve a torn final append before recovery."""
    if not path.exists():
        return {}
    content = path.read_bytes()
    rows, offset = {}, 0
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt cache record in {path}, line {i + 1}") from None
            recovery = path.with_name(path.name + f".torn-{time.time_ns()}")
            recovery.write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
            break
        if row["id"] in rows:
            raise ValueError(f"Duplicate cached ID: {row['id']}")
        rows[row["id"]] = row
        offset += len(line)
    # A complete JSON object without its trailing newline is still durable.
    if rows and offset == len(content) and not content.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def _append(path, row):
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _judge_identity(config):
    model = config.get("judge_model", JUDGE_MODEL)
    if model == JUDGE_MODEL:
        return {
            "judge_model": model,
            "judge_family": "llama3",
            "judge_rendering": "author_manual_llama3_roles",
            "judge_enable_thinking": None,
            "judge_substitution": False,
        }
    if model == QWEN_JUDGE_MODEL:
        if config.get("judge_enable_thinking", False) is not False:
            raise ValueError("Qwen3 judge requires judge_enable_thinking=False")
        return {
            "judge_model": model,
            "judge_family": "qwen3",
            "judge_rendering": "tokenizer_chat_template_enable_thinking_false",
            "judge_enable_thinking": False,
            "judge_substitution": True,
        }
    raise ValueError(f"Unsupported judge model: {model}")


def _judge_tokenizer_identity(config):
    root = Path(config.get("judge_model_path", ""))
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    return {name: _file_hash(root / name) for name in names if (root / name).is_file()}


def create_judge_prompt(question, gold, answer, *, config=None, tokenizer=None):
    """Preserve author content; select the model's own conversation rendering.

    The default Llama path remains byte-for-byte compatible with the author.
    Qwen3-4B is an explicit judge substitution with thinking disabled. The original
    visible <reasoning> explanation is retained and is not a thinking-mode block.
    """
    user = HOTPOT_JUDGE_TEMPLATE.format(q=question, gt=gold, a=answer)
    identity = _judge_identity(config or {})
    if identity["judge_family"] == "qwen3":
        if tokenizer is None:
            raise ValueError("Qwen3 rendering requires its tokenizer")
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return (
        f"<|start_header_id|>system<|end_header_id|>\n\n{JUDGE_SYSTEM}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_grade(text):
    """Author parser, unchanged: XML first, then whole text, true before false."""
    m = re.search(r"<grade>(.*?)</grade>", text, flags=re.I)
    content = m.group(1) if m else text
    lowered = content.lower()
    if "true" in lowered:
        return {"valid_grade": True, "grade": True}
    if "false" in lowered:
        return {"valid_grade": True, "grade": False}
    return {"valid_grade": False}


def paper_substring_match(prediction, gold):
    """Author secondary metric; no word boundaries or semantic interpretation."""

    def normalize(text):
        text = text.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

    return normalize(gold) in normalize(prediction)


class _HFJudge:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.identity = _judge_identity(config)
        self.device = config.get("judge_device", "cuda")
        self.tok = AutoTokenizer.from_pretrained(config["judge_model_path"], local_files_only=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config["judge_model_path"],
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )
        if self.tok.eos_token_id is None:
            raise ValueError("Judge tokenizer has no EOS token")
        self.stop_ids = [self.tok.eos_token_id]
        if self.identity["judge_family"] == "llama3":
            eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
            if eot != 128009:
                raise ValueError("Llama3 judge requires original tokenizer (eot=128009)")
            self.stop_ids = list(dict.fromkeys([*self.stop_ids, eot]))

    def generate(self, prompts):
        torch = self.torch
        # Llama uses the author's raw roles; Qwen's chat template already emits its tokens.
        inputs = self.tok(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=self.identity["judge_family"] == "llama3",
        ).to(self.device)
        if inputs.input_ids.shape[1] + 512 > self.model.config.max_position_embeddings:
            raise ValueError("Judge input exceeds model context; no silent truncation")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                max_new_tokens=512,
                eos_token_id=self.stop_ids,
                pad_token_id=self.tok.pad_token_id,
                use_cache=True,
            )
        results = []
        for ids in output[:, inputs.input_ids.shape[1] :].tolist():
            stops = [i for i, token in enumerate(ids) if token in self.stop_ids]
            cut = ids[: stops[0] + 1] if stops else ids
            results.append(
                {
                    "text": self.tok.decode(cut, skip_special_tokens=False).strip(),
                    "token_ids": cut,
                    "truncated": not bool(stops),
                }
            )
        return results


def _grade_summary(rows):
    def counts(part):
        valid = sum(r["valid_grade"] for r in part)
        correct = sum(bool(r.get("grade")) for r in part)
        return {
            "n": len(part),
            "valid_n": valid,
            "invalid_n": len(part) - valid,
            "correct_n": correct,
            "coverage": valid / len(part) if part else None,
            "paper_accuracy_valid_denominator": correct / valid if valid else None,
            "correct_fraction_all_records": correct / len(part) if part else None,
            "paper_substring_correct_n": sum(r["paper_substring"] for r in part),
            "truncated_judgments_n": sum(r["judge_truncated"] for r in part),
            "thinking_marker_outputs_n": sum(
                r.get("judge_thinking_markers_present", False) for r in part
            ),
        }

    groups = {}
    for row in rows:
        key = "/".join(str(row.get(k, "unspecified")) for k in ("arm", "phase", "group"))
        groups.setdefault(key, []).append(row)
    return {"all": counts(rows), "groups": {k: counts(v) for k, v in groups.items()}}


def grade_records(records, config, output, *, generator=None):
    """Grade immutable answer records, resuming a separate append-only judge cache.

    Required row fields: id, question, gold, answer. Optional arm/phase/group fields
    define summary groups. Output is a directory, never an existing scientific file.
    ``generator`` is only a dependency injection seam for CPU contract tests.
    """
    records = list(records)
    ids = [r["id"] for r in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Need nonempty records with unique IDs")
    for row in records:
        if not all(isinstance(row[k], str) for k in ("question", "gold", "answer")):
            raise TypeError("question/gold/answer must be strings")
    identity = _judge_identity(config)
    if config.get("judge_temperature", 0.0) != 0 or config.get("judge_max_new_tokens", 512) != 512:
        raise ValueError("Paper judge contract is greedy with max_new_tokens=512")
    batch_size = int(config.get("judge_batch_size", 8))
    if batch_size < 1:
        raise ValueError("judge_batch_size must be positive")
    spec = {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_judge_source_sha256": UPSTREAM_JUDGE_SOURCE_SHA256,
        "module_sha256": _file_hash(__file__),
        "records_sha256": _hash(records),
        **identity,
        "judge_model_path": str(config.get("judge_model_path", "")),
        "judge_tokenizer_sha256": _judge_tokenizer_identity(config),
        "judge_model_revision": config.get("judge_model_revision", config.get("judge_revision")),
        "judge_mirror": config.get("judge_mirror"),
        "judge_mirror_revision": config.get("judge_mirror_revision"),
        "historical_paper_revision": "not pinned by original paper or code",
        "evaluation_protocol": (
            "Paper prompts/parser with Qwen3-4B no-thinking judge substitution; "
            "not the original paper's judge"
            if identity["judge_substitution"]
            else "Original paper Llama3 judge model family and prompts/parser"
        ),
        "judge_batch_size": batch_size,
        "precision": "bfloat16, no quantization",
        "sampling": {"greedy": True, "temperature": 0, "max_new_tokens": 512},
        "backend_difference": (
            "HF generate replaces original vLLM completions; greedy top_k=50 is inert"
        ),
        "prompt_has_context": False,
        "csv_input_normalization": (
            "As original evaluate.py CSV: remove question/gold newlines, "
            "replace answer newlines by spaces"
        ),
        "invalid_policy": "Retain every record; report valid denominator and coverage separately",
    }
    began = time.monotonic()
    with _locked(output) as output:
        provenance = output / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text()) != spec:
            raise ValueError("Grading inputs/config/source changed; use a new output directory")
        if not provenance.exists():
            _write(provenance, spec)
        path = output / "grading.jsonl"
        cached = _cache(path)
        if set(cached) - set(ids):
            raise ValueError("Unexpected records in grading cache")
        pending = [r for r in records if r["id"] not in cached]
        if pending and generator is None:
            generator = _HFJudge(config)
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset : offset + batch_size]
            prompts = [
                create_judge_prompt(
                    r["question"].replace("\n", ""),
                    r["gold"].replace("\n", ""),
                    r["answer"].replace("\n", " "),
                    config=config,
                    tokenizer=getattr(generator, "tok", None),
                )
                for r in chunk
            ]
            generated = generator.generate(prompts)
            if len(generated) != len(chunk):
                raise ValueError("Judge batch returned incomplete coverage")
            for row, prompt, result in zip(chunk, prompts, generated, strict=True):
                raw = result["text"]
                graded = {
                    **row,
                    "input_record_sha256": _hash(row),
                    "judge_prompt": prompt,
                    "judge_raw_output": raw,
                    "judge_token_ids": result.get("token_ids", []),
                    "judge_truncated": bool(result.get("truncated", False)),
                    "judge_thinking_markers_present": bool(
                        re.search(r"</?think>", raw, flags=re.I)
                    ),
                    **extract_grade(raw),
                    "paper_substring": paper_substring_match(row["answer"], row["gold"]),
                }
                _append(path, graded)
                cached[row["id"]] = graded
        ordered = [cached[r["id"]] for r in records]
        for original, graded in zip(records, ordered, strict=True):
            if graded["input_record_sha256"] != _hash(original):
                raise ValueError("Cached grading is associated with a different answer")
        summary = {
            "complete": True,
            "records_sha256": spec["records_sha256"],
            "grading_sha256": _file_hash(path),
            "source": spec,
            "summary": _grade_summary(ordered),
            "seconds_this_process": time.monotonic() - began,
        }
        _write(output / "summary.json", summary)
        return summary


def old_records(batch, arms=None):
    """Read old outputs without regeneration; deduplicate only identical base outputs."""
    batch = Path(batch)
    arms = arms or ["pd_lora", "sft_lora", "sft_full", "sft_down"]
    cases = json.loads((batch / "shared/eval.json").read_text())
    cases = cases["cases"] if isinstance(cases, dict) else cases
    questions = {r["case_id"]: r["question"] for r in cases}
    evaluations = {}
    sources = {}
    for arm in arms:
        path = batch / "runs" / arm / "eval/evaluate.json"
        evaluations[arm] = json.loads(path.read_text())
        sources[arm] = {"path": str(path), "sha256": _file_hash(path)}
    before = evaluations[arms[0]]["before"]
    if not all(e["before"] == before for e in evaluations.values()):
        raise ValueError("Old baselines differ; cannot deduplicate them")
    result = []
    sets = [("base", "before", before)] + [(a, "after", evaluations[a]["after"]) for a in arms]
    for arm, phase, rows in sets:
        selected = [r for r in rows if r["group"] in ("closed", "context")]
        if len(selected) != 2 * len(cases):
            raise ValueError("Old answer coverage differs from case pool")
        for row in selected:
            case_id = row["id"].rsplit("/", 1)[0]
            group = "oracle_context" if row["group"] == "context" else "closed"
            result.append(
                {
                    "id": f"{arm}/{case_id}/{group}",
                    "case_id": case_id,
                    "arm": arm,
                    "phase": phase,
                    "group": group,
                    "question": questions[case_id],
                    "gold": row["gold"],
                    "answer": row["answer"],
                    "source_generation_prompt": row["prompt"],
                    "frozen_em": row["em"],
                    "frozen_f1": row["f1"],
                }
            )
    return result, {
        "source_batch": str(batch),
        "source_files": sources,
        "base_records_identical": True,
        "unique_judge_records": len(result),
        "old_prompt_difference": (
            "Stored generations include a short-answer instruction, absent from paper"
        ),
        "old_retention_unchanged": {a: evaluations[a]["summary"]["retention"] for a in arms},
        "interpretation": (
            "Additional paper-judge regrading of stored generations; frozen scores unchanged"
        ),
    }


def regrade_old(config, batch, output):
    records, provenance = old_records(batch, config.get("old_arms"))
    provenance["judge_identity"] = _judge_identity(config)
    output = Path(output)
    _write(output / "old-source.json", provenance)
    run = _tracking_run(config, output.parent, "regrade-development")
    exit_code = 1
    try:
        result = grade_records(records, config, output)
        if run:
            run.summary.update(result["summary"])
            run.log(_tracking_metrics(result["summary"]))
        exit_code = 0
        return result
    finally:
        if run:
            run.finish(exit_code=exit_code)


def _tracking_run(config, batch, stage, arm=None):
    if not config.get("paper_eval_wandb", True):
        return None
    from . import knowledge_method_diagnosis as D

    tracking_batch = Path(config.get("paper_eval_tracking_batch", batch))
    return D.wandb_run(config, tracking_batch, stage, arm)


def _tracking_metrics(summary):
    result = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            result.update({f"{key}/{k}": v for k, v in _tracking_metrics(value).items()})
        elif isinstance(value, (int, float)):
            result[key] = value
    return result


def evaluation_rows(config, batch, tok):
    """Paper closed-book prompt; supplied-context diagnostic explicitly named oracle."""
    from .knowledge_method_data import render_chat

    cases = json.loads((Path(batch) / "shared/eval.json").read_text())
    cases = cases["cases"] if isinstance(cases, dict) else cases
    groups = config.get("paper_eval_groups", ["closed", "oracle_context"])
    if "closed" not in groups or set(groups) - {"closed", "oracle_context"}:
        raise ValueError("Primary closed-book evaluation is required; true RAG is not implemented")
    if len(cases) != config["n_cases"]:
        raise ValueError("Evaluation case coverage differs from config")
    result = []
    for case in cases:
        for group in groups:
            question = case["question"]
            if group == "closed":
                user = question
            else:
                paragraphs = case.get("paragraphs")
                context = "\n\n".join(paragraphs) if paragraphs is not None else case["context"]
                user = f"{context}\n\nQuestion: {question}"
            result.append(
                {
                    "id": f"{case['case_id']}/{group}",
                    "case_id": case["case_id"],
                    "group": group,
                    "question": question,
                    "gold": case["answer"],
                    "prompt": render_chat(tok, user),
                }
            )
    return result


def predict_records(model, tok, rows, config, output, source):
    """Resumable answer generation with a per-question total-token budget.

    Every unfinished original batch is deterministically regenerated in full so
    a crash between durable records does not change remaining sampling seeds.
    """
    import torch

    from . import knowledge_method_diagnosis as D

    batch_size = int(config.get("paper_eval_batch_size", config.get("eval_batch_size", 8)))
    if batch_size < 1:
        raise ValueError("paper_eval_batch_size must be positive")
    maximum = 500
    total_budget = min(6128, model.config.max_position_embeddings)
    spec = {
        "source": source,
        "rows_sha256": _hash(rows),
        "module_sha256": _file_hash(__file__),
        "batch_size": batch_size,
        "seed": config["seed"],
        "temperature": 0.25,
        "max_new_tokens": maximum,
        "max_total_tokens": total_budget,
        "paper_vs_code_default": "Paper uses T=0.25; evaluate.py CLI default is 0.0",
        "prompt": "Question only; oracle_context uses paragraphs + Question: question",
        "precision": "FP32 model parameters with BF16 autocast",
    }
    model.eval()
    with _locked(output) as output:
        provenance = output / "provenance.json"
        if provenance.exists() and json.loads(provenance.read_text()) != spec:
            raise ValueError("Generation provenance changed; use a new directory")
        if not provenance.exists():
            _write(provenance, spec)
        path = output / "answers.jsonl"
        cached = _cache(path)
        if set(cached) - {r["id"] for r in rows}:
            raise ValueError("Unexpected generated records")
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            if all(r["id"] in cached for r in chunk):
                continue
            # Group by allowed length to avoid silently shortening other questions
            # when one oracle prompt nearly exhausts the original 6128-token budget.
            by_budget = {}
            for index, row in enumerate(chunk):
                length = len(tok.encode(row["prompt"], add_special_tokens=False))
                budget = min(maximum, total_budget - length)
                if budget <= 0:
                    raise ValueError(f"Prompt exceeds paper context budget: {row['id']}")
                by_budget.setdefault(budget, []).append((index, row))
            generated = {}
            for group_index, (budget, selected) in enumerate(sorted(by_budget.items())):
                inputs = tok(
                    [row["prompt"] for _, row in selected],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(next(model.parameters()).device)
                D.seed_all(config["seed"] + 900000 + offset * 1000 + group_index)
                with torch.inference_mode(), D.autocast():
                    values = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=0.25,
                        top_p=1.0,
                        top_k=50,
                        max_new_tokens=budget,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                        use_cache=True,
                    )
                tails = values[:, inputs.input_ids.shape[1] :].tolist()
                for (index, row), ids in zip(selected, tails, strict=True):
                    stopped = tok.eos_token_id in ids
                    if stopped:
                        ids = ids[: ids.index(tok.eos_token_id) + 1]
                    answer = tok.decode(ids, skip_special_tokens=True).strip()
                    answer = answer.replace("&lt;", "<").replace("&gt;", ">")
                    generated[index] = {
                        **row,
                        "answer": answer,
                        "token_ids": ids,
                        "input_record_sha256": _hash(row),
                        "generated_tokens": len(ids),
                        "max_new_tokens_this_record": budget,
                        "truncated": not stopped,
                        "generation_seed": config["seed"] + 900000 + offset * 1000 + group_index,
                        "secondary_em_f1": D.score_answer(answer, row["gold"]),
                    }
            for index, row in enumerate(chunk):
                result = generated[index]
                if row["id"] in cached:
                    if cached[row["id"]] != result:
                        raise ValueError("Resumed batch did not reproduce its durable answers")
                else:
                    _append(path, result)
                    cached[row["id"]] = result
        ordered = [cached[r["id"]] for r in rows]
        if any(
            saved["input_record_sha256"] != _hash(row)
            for row, saved in zip(rows, ordered, strict=True)
        ):
            raise ValueError("Generated records no longer match input questions")
        _write(
            output / "complete.json",
            {
                "complete": True,
                "n": len(ordered),
                "answers_sha256": _file_hash(path),
                "truncated_n": sum(r["truncated"] for r in ordered),
                "source": spec,
            },
        )
        return ordered


def _paired_grade_summary(rows):
    result = {}
    for group in sorted({r["group"] for r in rows}):
        b = {r["case_id"]: r for r in rows if r["group"] == group and r["phase"] == "before"}
        a = {r["case_id"]: r for r in rows if r["group"] == group and r["phase"] == "after"}
        if not b or b.keys() != a.keys():
            raise ValueError("Before/after paper grading coverage differs")
        initially = [key for key, r in b.items() if r.get("grade") is True]
        result[group] = {
            "before": _grade_summary(list(b.values()))["all"],
            "after": _grade_summary(list(a.values()))["all"],
            "initially_judged_correct_n": len(initially),
            "retained_judged_correct_n": sum(a[key].get("grade") is True for key in initially),
            "after_invalid_among_initially_correct_n": sum(
                not a[key]["valid_grade"] for key in initially
            ),
        }
    return result


def _evaluate_impl(config, batch, arm):
    """Generate fresh paper-style answers, then unload Qwen before loading judge.

    Writes runs/<arm>/paper-eval only. Retention keeps the existing alias graders.
    The factorial training module owns all eight parameter configurations.
    """
    import gc

    import torch

    from . import knowledge_method_diagnosis as D
    from . import knowledge_method_factorial as F

    batch = Path(batch)
    if arm not in config["arms"]:
        raise ValueError(arm)
    output = batch / "runs" / arm / "paper-eval"
    weights = batch / "runs" / arm / "weights.pt"
    source = {
        "arm": arm,
        "weights_sha256": _file_hash(weights),
        "model_path": str(config["model_path"]),
        "model_revision": config.get("model_revision"),
        "cases_sha256": _file_hash(batch / "shared/eval.json"),
        "train_module_sha256": _file_hash(F.__file__),
        "config_sha256": _hash(config),
        "train_data_sha256": _file_hash(batch / "shared/train.jsonl"),
        "retention_source_sha256": (
            _file_hash(config["retention_source"]) if config.get("retention_source") else None
        ),
    }
    with _locked(output):
        D.seed_all(config["seed"])
        model, tok = D.load_model(config)
        rows = evaluation_rows(config, batch, tok)
        before = predict_records(
            model, tok, rows, config, output / "before", {**source, "phase": "before"}
        )
        aux_config = {
            **config,
            "eval_batch_size": config.get(
                "paper_eval_batch_size", config.get("eval_batch_size", 8)
            ),
            "eval_max_new_tokens": 500,
            "eval_temperature": 0.25,
            "retention_max_new_tokens": config.get("retention_max_new_tokens", 64),
        }
        auxiliary_path = output / "auxiliary.json"
        if auxiliary_path.exists():
            auxiliary = json.loads(auxiliary_path.read_text())
            if auxiliary["source"] != source:
                raise ValueError("Auxiliary evaluation source changed")
        else:
            auxiliary = {"source": source, "fit": {}, "retention": {}}
        retention_rows = []
        if config.get("retention_source"):
            retention_rows = [
                r for r in D.evaluation_rows(aux_config, batch, tok) if r["group"] == "retention"
            ]
        for phase in ("before", "after"):
            if phase == "after":
                model = F.configure_arm(model, config, arm)
                D.restore(model, torch.load(weights, map_location="cpu", weights_only=True))
                after = predict_records(
                    model, tok, rows, config, output / "after", {**source, "phase": "after"}
                )
            if config.get("paper_eval_fit", True) and phase not in auxiliary["fit"]:
                auxiliary["fit"][phase] = F.fit_diagnostic(model, tok, aux_config, batch, arm)
                _write(auxiliary_path, auxiliary)
            if retention_rows and phase not in auxiliary["retention"]:
                auxiliary["retention"][phase] = D.predict(model, tok, retention_rows, aux_config)
                _write(auxiliary_path, auxiliary)
        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        judge_rows = []
        for phase, generated in (("before", before), ("after", after)):
            for row in generated:
                judge_rows.append(
                    {
                        **row,
                        "id": f"{arm}/{phase}/{row['id']}",
                        "arm": arm,
                        "phase": phase,
                    }
                )
        judged = grade_records(judge_rows, config, output / "judge")
        graded = list(_cache(output / "judge/grading.jsonl").values())
        summary = _paired_grade_summary(graded)
        if retention_rows:
            b = {r["id"]: r for r in auxiliary["retention"]["before"]}
            a = {r["id"]: r for r in auxiliary["retention"]["after"]}
            if b.keys() != a.keys():
                raise ValueError("Retention coverage mismatch")
            old_correct = [key for key in b if b[key]["strict"]]
            summary["retention"] = {
                "n": len(b),
                "before_correct_n": len(old_correct),
                "after_correct_n": sum(r["strict"] for r in a.values()),
                "initially_correct_n": len(old_correct),
                "retained_n": sum(a[key]["strict"] for key in old_correct),
                "metric": "Existing natural-retention alias strict grader; no LLM judge",
            }
        result = {
            "complete": True,
            "arm": arm,
            "source": source,
            "summary": summary,
            "teacher_target_fit": auxiliary["fit"],
            "judge": judged,
            "interpretation": (
                "Primary closed-book uses the paper grading protocol with configured judge "
                f"{_judge_identity(config)['judge_model']}; "
                "Qwen3 is a no-thinking judge substitution, not original-judge reproduction; "
                "oracle_context is supplied context, not RAG"
            ),
        }
        _write(output / "evaluate.json", result)
        return result


def evaluate(config, batch, arm):
    """Tracked entry point; W&B spans generation and grading and closes on failure."""
    run = _tracking_run(config, Path(batch), "paper-evaluate", arm)
    exit_code = 1
    try:
        result = _evaluate_impl(config, batch, arm)
        if run:
            run.summary.update(result["summary"])
            run.log(_tracking_metrics(result["summary"]))
        exit_code = 0
        return result
    finally:
        if run:
            run.finish(exit_code=exit_code)


UPSTREAM_COMMIT = "2b72919a0836f97115f0396ee753bd284c8d91d6"
UPSTREAM_JUDGE_SOURCE_SHA256 = "1c7cee6644706c27e6cd9f31f4e67f592c07d09ce0daca4823e99b427eefb42b"
JUDGE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
QWEN_JUDGE_MODEL = "Qwen/Qwen3-4B"
JUDGE_SYSTEM = "You are a knowledgeable assistant trained to provide accurate and helpful information. Please respond to the user's queries promptly."  # noqa: E501 - exact upstream prompt
HOTPOT_JUDGE_TEMPLATE = "\nHere is a question, the ground-truth answer and the proposed answer. Please evaluate if the answer is true or false and return the reasoning and grade as xml. Example:\n\nExample\nQuestion: What was the punishment for Mattingly for not getting a haircut?\nGround-truth answer: benched\nProposed answer: Mattingly was benched for 20 games as punishment for not getting a haircut.\n\nOutput:\n<reasoning>The ground-truth answer is that Mattingly was benched for not getting a haircut. The proposed answer correctly mentions that Mattingly was benched for not getting a haircut</reasoning>\n<grade>true</grad>\n\nExample:\nQuestion: For what was Mattingly once fined and benched for?\nGround-truth answer: not getting a haircut\nProposed Answer: Mattingly was fined and benched for not playing in a game, which was likely a form of protest against the team's management or the way he was being treated.\n\nOutput:\n<reasoning>According to the ground-truth answer, Mattingly was fined and benched for not getting a haircut. The proposed answer does not answer the question accurately, and mentions a protest as a reason. However, this is clearly different from the ground-truth answer.</reasoning>\n<grade>false</grade>\n\nExample:\nQuestion: Which baseball teams was Mattingly apart of?\nGround-truth answer: Yankees\nProposed answer: Mattingly was a part of the New York Yankees and the Los Angeles Dodgers baseball teams.\n\nOutput:\n<reasoning>The ground-truth answer indicates that Mattingly was a part of the \"Yankees\" team. The proposed answer states that he was a part of the New York Yankees, which matches one of the ground-truth answers. However, it also mentions the Los Angeles Dodgers, which is not included in the ground-truth answer. Since the additional information about the Dodgers does not directly contradict the truth of him being on the Yankees, the primary question's focus remains valid.</reasoning>\n<grade>true</grade>\n                                                                                                                                    Please grade the following question using the xml-template given:\nQuestion: {q}\nGround-truth answer: {gt}\nProposed answer: {a}\n"  # noqa: E501 - exact upstream prompt
