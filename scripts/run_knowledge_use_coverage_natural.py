#!/usr/bin/env python3
"""Natural-document application stages; no evaluation labels enter generation.

This extension reuses the factorial's previously evaluated cases 32:96. Existing
questions are not certified atomic; A/B also change paragraph-internal coverage.
The resulting comparison measures application effects, not pure use attribution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

from llm_memory_editability import knowledge_method_paper_eval as P
from llm_memory_editability import knowledge_use_coverage_natural as N
from llm_memory_editability.knowledge_method_data import context_user, render_chat

JUDGE_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
LIMITATIONS = [
    "Cases 32:96 were evaluated in the prior factorial; not a new confirmation set.",
    "Original generated questions are not semantically reviewed or certified atomic.",
    "A/B can change paragraph-internal fact coverage as well as wording.",
    "Generated C quotations are checked literally; semantic grounding remains unaudited.",
    "Original trajectories used paragraph context; new trajectories use full-case context.",
    "All arms use the same full-case teacher context during training.",
    "Application results do not replace controlled-new-fact use-coverage attribution.",
]


def _runtime():
    # Lazy import keeps source resolution and schema checks CPU-only.
    import run_knowledge_use_coverage

    return run_knowledge_use_coverage


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def source_path(factorial_batch, relative, explicit=None):
    """Resolve the historical separately mounted upstream without guessing data."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    factorial_batch = Path(factorial_batch).resolve()
    candidates = [
        factorial_batch / "upstream" / relative,
        factorial_batch.parent / "knowledge-method-diagnosis-development-v1/upstream" / relative,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Mount the historical upstream or supply an explicit path: {relative}")


def natural_config(config, factorial_batch=Path("/factorial")):
    """The same resolved configuration is used for training and checkpoint checks."""
    return {
        **config,
        "max_sequence_tokens": 8192,
        "generation_batch_size": 4,
        "generation_max_new_tokens": 512,
        "generation_top_k": 50,
        "eval_batch_size": 4,
        "paper_eval_batch_size": 4,
        "eval_max_new_tokens": 500,
        "eval_temperature": 0.25,
        "retention_max_new_tokens": 64,
        "judge_model": "Qwen/Qwen3-4B",
        "judge_model_path": str(Path(factorial_batch) / "models/qwen3-4b"),
        "judge_model_revision": JUDGE_REVISION,
        "judge_enable_thinking": False,
        "judge_batch_size": 4,
        "judge_temperature": 0.0,
        "judge_max_new_tokens": 512,
    }


def _generate_saved(
    runtime, model, tok, inputs, config, path, *, context, temperature, accept=None
):
    """Commit each small raw batch before validating it; never automatically retry."""
    generated = []
    input_tokens = 0
    began = time.perf_counter()
    with Path(path).open("x") as handle:
        size = config["generation_batch_size"]
        for offset in range(0, len(inputs), size):
            part = inputs[offset : offset + size]
            answers = runtime.generate(model, tok, part, config, context, temperature)
            for answer in answers:
                handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if [a["id"] for a in answers] != [r["id"] for r in part]:
                raise ValueError("Generation changed request identity or coverage")
            for sample, answer in zip(part, answers, strict=True):
                user = (
                    context_user(sample["context"], sample["question"])
                    if context
                    else sample["question"]
                )
                input_tokens += len(tok.encode(render_chat(tok, user), add_special_tokens=False))
                if not answer["final_answer_valid"]:
                    raise ValueError(f"Invalid answer boundary: {answer['id']}")
                if accept is not None:
                    accept(answer)
                generated.append(answer)
    return generated, {
        "requests": len(generated),
        "input_tokens": input_tokens,
        "generated_tokens": sum(len(a["token_ids"]) for a in generated),
        "truncated": sum(a["truncated"] for a in generated),
        "seconds": time.perf_counter() - began,
        "temperature": temperature,
        "top_k": config["generation_top_k"],
        "top_p": 1.0,
        "max_new_tokens": config["generation_max_new_tokens"],
    }


def _answer_generation(record):
    return {
        "text": record["raw"],
        "token_ids": record["token_ids"],
        "generated_tokens": len(record["token_ids"]),
        "truncated": record["truncated"],
        "finish_reason": "length" if record["truncated"] else "eos",
    }


def generation_protocol(config):
    """Bind approval to actual question code, model and both predeclared strategies."""
    protocol = {
        "version": "natural-use-question-quality-v3-short-evidence",
        "module_sha256": digest(N.__file__),
        "runner_sha256": digest(__file__),
        "shared_runner_sha256": digest(Path(__file__).with_name("run_knowledge_use_coverage.py")),
        "model_revision": config["model_revision"],
        "model_path": str(config.get("model_path", "")),
        "seed": config["seed"],
        "temperature": 0.7,
        "top_k": config["generation_top_k"],
        "top_p": 1.0,
        "max_new_tokens": config["generation_max_new_tokens"],
        "max_sequence_tokens": config["max_sequence_tokens"],
        "generation_batch_size": config["generation_batch_size"],
        "candidate_attempt_limits": [1, 3],
        "repair": "same case and use; previous output plus structural error; no test access",
        "semantic_verification": "separate manual review; quotes alone are insufficient",
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol, fingerprint


def development_requests(source):
    requests = N.build_use_requests(N.load_training_cases(source, split="development"))
    first = {}
    for request in requests:
        first.setdefault((request["case_id"], request["category"]), request)
    selected = [
        first[(case, category)]
        for case in sorted({r["case_id"] for r in requests})
        for category in N.USE_CATEGORIES
    ]
    if len(selected) != 96:
        raise ValueError("Calibration must retain exactly 32 cases x 3 uses")
    return selected


def _question_status(request, generation):
    try:
        if generation["truncated"] or not generation["final_answer_valid"]:
            raise ValueError("Truncated JSON or invalid final-answer boundary")
        return "valid", N.parse_use_question(request, generation["raw"]), None
    except (ValueError, KeyError, TypeError) as exc:
        unavailable = False
        try:
            text = N.final_answer_text(generation["raw"])
            if text.startswith("```json\n") and text.endswith("\n```"):
                text = text[len("```json\n") : -len("\n```")]
            value = json.loads(text)
            unavailable = isinstance(value, dict) and value.get("unavailable") is True
        except (ValueError, KeyError, TypeError):
            pass
        return "unavailable" if unavailable else "invalid", None, str(exc)


def generate_questions(runtime, model, tok, requests, config, out, attempts):
    """Evaluate fixed 1/3-attempt protocols; preserve every invalid/unavailable output."""
    if attempts not in (1, 3):
        raise ValueError("Question attempts must use one of the two predeclared strategies")
    protocol, protocol_hash = generation_protocol(config)
    by_id = {r["id"]: r for r in requests}
    accepted, first_status, last_status, previous = {}, {}, {}, {}
    costs, history = [], []
    with (out / "raw-questions.jsonl").open("x") as handle:
        for attempt in range(1, attempts + 1):
            pending = [r for r in requests if r["id"] not in accepted]
            if not pending:
                break
            inputs = []
            for request in pending:
                user = request["user"]
                if attempt > 1:
                    last = previous[request["id"]]
                    user += (
                        "\n\nRepair the preceding attempt's JSON or source-evidence error. "
                        "Keep exactly the same source case and requested use. Do not add "
                        "outside facts. Use two independently stated facts for composition "
                        "or comparison; evidence may share a paragraph. The question must "
                        "need both facts, not merely quote two snippets. If no grounded "
                        'question is possible, still return {"unavailable": true}.\n'
                        f"Validation error: {last['error']}\nPrevious output:\n"
                        + last["generation"]["raw"]
                    )
                inputs.append({"id": request["id"], "question": user})
            for start in range(0, len(inputs), config["generation_batch_size"]):
                part = inputs[start : start + config["generation_batch_size"]]
                began = time.perf_counter()
                generated = runtime.generate(model, tok, part, config, False, 0.7)
                if [g["id"] for g in generated] != [r["id"] for r in part]:
                    for generation in generated:
                        handle.write(
                            json.dumps(
                                {
                                    "attempt": attempt,
                                    "generation": generation,
                                    "fatal": "request_identity_mismatch",
                                }
                            )
                            + "\n"
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                    raise ValueError("Question generation changed request identity")
                costs.append(
                    {
                        "attempt": attempt,
                        "requests": len(part),
                        "seconds": time.perf_counter() - began,
                        "input_tokens": sum(
                            len(
                                tok.encode(
                                    render_chat(tok, r["question"]), add_special_tokens=False
                                )
                            )
                            for r in part
                        ),
                        "generated_tokens": sum(len(g["token_ids"]) for g in generated),
                    }
                )
                for sample, generation in zip(part, generated, strict=True):
                    request = by_id[sample["id"]]
                    status, row, error = _question_status(request, generation)
                    prompt_hash = hashlib.sha256(sample["question"].encode()).hexdigest()
                    record = {
                        "request_id": request["id"],
                        "case_id": request["case_id"],
                        "category": request["category"],
                        "attempt": attempt,
                        "status": status,
                        "error": error,
                        "generation": generation,
                        "user": sample["question"],
                        "prompt_sha256": prompt_hash,
                        "generation_protocol_sha256": protocol_hash,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    history.append(record)
                    previous[request["id"]] = record
                    first_status.setdefault(request["id"], status)
                    last_status[request["id"]] = status
                    if row is not None:
                        row["provenance"].update(
                            {
                                "actual_prompt_sha256": prompt_hash,
                                "generation_attempt": attempt,
                                "generation_protocol_sha256": protocol_hash,
                            }
                        )
                        accepted[request["id"]] = row
                handle.flush()
                os.fsync(handle.fileno())
    summary = {
        "complete": True,
        "request_count": len(requests),
        "attempt_limit": attempts,
        "first_attempt": dict(Counter(first_status.values())),
        "final_attempt": dict(Counter(last_status.values())),
        "first_attempt_by_category": {
            category: dict(
                Counter(first_status[r["id"]] for r in requests if r["category"] == category)
            )
            for category in N.USE_CATEGORIES
        },
        "unresolved_request_ids": [r["id"] for r in requests if r["id"] not in accepted],
        "actual_generation_requests": len(history),
        "generation_cost": costs,
        "protocol": protocol,
        "generation_protocol_sha256": protocol_hash,
        "semantic_review_status": "pending",
        "approval_written": False,
    }
    write_json(out / "question-quality-summary.json", summary)
    return [accepted[r["id"]] for r in requests if r["id"] in accepted], summary


def calibrate(config, batch, factorial_batch, *, source_csv=None, runtime=None):
    runtime = runtime or _runtime()
    out = Path(batch) / "data/natural-development"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "started.json", {"unix_time": time.time(), "scientific_training_updates": 0})
    try:
        source = source_path(factorial_batch, "datasets/hotpotqa.csv", source_csv)
        requests = development_requests(source)
        write_rows(out / "question-requests.jsonl", requests)
        runtime.K.seed_all(config["seed"])
        model, tok = runtime.K.load_model(config)
        accepted, summary = generate_questions(runtime, model, tok, requests, config, out, 3)
        write_rows(out / "parsed-questions.jsonl", accepted)
        write_json(
            out / "quality-summary.json",
            {
                **summary,
                "status": "awaiting_manual_semantic_review",
                "case_indices": [0, 32],
                "evaluation_files_read": False,
                "raw_questions_sha256": digest(out / "raw-questions.jsonl"),
                "parsed_questions_sha256": digest(out / "parsed-questions.jsonl"),
                "question_requests_sha256": digest(out / "question-requests.jsonl"),
            },
        )
    except BaseException:
        write_json(out / "failure.json", {"traceback": traceback.format_exc()})
        raise


def approved_calibration(config, batch):
    out = Path(batch) / "data/natural-development"
    approval = read(out / "calibration-approved.json")
    quality = read(out / "quality-summary.json")
    _, protocol_hash = generation_protocol(config)
    if (
        approval.get("status") != "approved"
        or quality.get("complete") is not True
        or quality.get("request_count") != 96
        or approval.get("semantic_review_completed") is not True
        or approval.get("question_attempts") not in (1, 3)
        or approval.get("generation_protocol_sha256") != protocol_hash
        or quality.get("generation_protocol_sha256") != protocol_hash
        or approval.get("quality_summary_sha256") != digest(out / "quality-summary.json")
    ):
        raise ValueError("Natural generation requires matching reviewed calibration approval")
    for name, key in (
        ("raw-questions.jsonl", "raw_questions_sha256"),
        ("parsed-questions.jsonl", "parsed_questions_sha256"),
        ("question-requests.jsonl", "question_requests_sha256"),
    ):
        if digest(out / name) != quality[key]:
            raise ValueError(f"Calibration material changed after review: {name}")
    return approval


def prepare(config, batch, factorial_batch, *, source_csv=None, runtime=None):
    """Read context-only source projections and reused answers; no test JSON is opened."""
    runtime = runtime or _runtime()
    batch, factorial_batch = Path(batch), Path(factorial_batch)
    approval = approved_calibration(config, batch)
    out = batch / "data/natural"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "started.json", {"unix_time": time.time(), "automatic_retry": False})
    try:
        source = source_path(factorial_batch, "datasets/hotpotqa.csv", source_csv)
        cases = N.load_training_cases(source, split="application")
        train_path = factorial_batch / "shared/train.jsonl"
        variants = N.select_existing_rows(cases, train_path)
        old_complete = read(factorial_batch / "shared/prepare-complete.json")
        source_hash = variants["A"][0]["provenance"]["source_file_sha256"]
        if old_complete["files_sha256"]["train.jsonl"] != source_hash:
            raise ValueError("Original factorial training data hash changed")
        replay_source = batch / "data/formal/prepared-replay.jsonl"
        replay_complete = read(batch / "data/formal/prepare-complete.json")
        if digest(replay_source) != replay_complete["sha256"]["prepared-replay.jsonl"]:
            raise ValueError("Controlled formal replay is not the frozen prepared artifact")
        requests = N.build_use_requests(cases)
        write_rows(out / "question-requests.jsonl", requests)
        write_json(out / "resolved-config.json", config)
        manifest = {
            "cohort": "natural",
            "case_indices": [32, 96],
            "cases": len(cases),
            "paragraphs": sum(len(c["paragraphs"]) for c in cases),
            "source_csv": str(source),
            "source_csv_sha256": digest(source),
            "source_train_sha256": source_hash,
            "replay_source": str(replay_source),
            "replay_source_sha256": digest(replay_source),
            "evaluation_files_read_for_generation": False,
            "application_only": True,
            "limitations": LIMITATIONS,
            "generation_calibration": approval,
        }
        write_json(out / "manifest.json", manifest)
        runtime.K.seed_all(config["seed"])
        model, tok = runtime.K.load_model(config)
        question_rows, question_cost = generate_questions(
            runtime, model, tok, requests, config, out, approval["question_attempts"]
        )
        write_rows(out / "new-questions.jsonl", question_rows)
        if question_cost["unresolved_request_ids"]:
            raise ValueError(
                "Question preparation has unresolved requests after the approved attempt limit"
            )
        answer_banks, costs = {}, {"questions": question_cost}
        for kind, temperature in (("pd", 1.5), ("sft", 0.25)):
            answers, costs[kind] = _generate_saved(
                runtime,
                model,
                tok,
                [{k: r[k] for k in ("id", "question", "context")} for r in question_rows],
                config,
                out / f"raw-{kind}-trajectories.jsonl",
                context=True,
                temperature=temperature,
            )
            answer_banks[kind] = {a["id"]: a for a in answers}
        variants["C"].extend(
            N.attach_answers(
                row,
                pd_generation=_answer_generation(answer_banks["pd"][row["id"]]),
                sft_generation=_answer_generation(answer_banks["sft"][row["id"]]),
            )
            for row in question_rows
        )
        token_counts = {}
        for arm, records in variants.items():
            write_rows(out / f"prepared-{arm}.jsonl", records)
            for objective in ("pd", "sft"):
                encoded = runtime.encoded_rows(tok, records, objective, config)
                token_counts[f"{objective}/{arm}"] = {
                    "answer_tokens_one_pass": sum(len(r["answer"]) for r in encoded),
                    "maximum_teacher_sequence": max(
                        len(r["teacher"]) + len(r["answer"]) for r in encoded
                    ),
                }
        # Preserve all replay fields and bytes from the controlled comparison.
        with (out / "prepared-replay.jsonl").open("xb") as handle:
            handle.write(replay_source.read_bytes())
        write_json(out / "coverage.json", N.coverage_statistics(variants))
        write_json(out / "generation-cost.json", costs)
        write_json(
            out / "prepare-complete.json",
            {
                "complete": True,
                "status": "complete",
                "application_only": True,
                "rows": {a: len(r) for a, r in variants.items()},
                "token_counts": token_counts,
                "sha256": {p.name: digest(p) for p in out.glob("prepared-*.jsonl")},
                "limitations": LIMITATIONS,
            },
        )
    except BaseException:
        write_json(
            out / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False}
        )
        raise


def application_cases(factorial_batch, source_csv=None):
    """Evaluation-only loader also checks the factorial's authoritative test copy."""
    factorial_batch = Path(factorial_batch)
    source = source_path(factorial_batch, "datasets/hotpotqa.csv", source_csv)
    cases = N.load_evaluation_cases(source, split="application")
    prior = read(factorial_batch / "shared/eval.json")
    prior = prior["cases"] if isinstance(prior, dict) else prior
    if len(prior) != 1000:
        raise ValueError("Expected the frozen 1000-case factorial evaluation artifact")
    for case, old in zip(cases, prior[32:96], strict=True):
        if any(case[key] != old[key] for key in ("case_id", "question", "answer", "context")):
            raise ValueError("CSV evaluation columns differ from the prior factorial")
    return cases


def verify_preparation(batch, factorial_batch, source_csv=None):
    """Post-generation isolation audit reports conflicts; it never filters material."""
    out = Path(batch) / "data/natural"
    complete = read(out / "prepare-complete.json")
    for name, expected in complete["sha256"].items():
        if digest(out / name) != expected:
            raise ValueError(f"Prepared natural artifact changed: {name}")
    variants = {arm: rows(out / f"prepared-{arm}.jsonl") for arm in "ABC"}
    overlaps = N.audit_evaluation_overlap(variants, application_cases(factorial_batch, source_csv))
    audit = {"passed": not overlaps, "overlaps": overlaps, "filtering_performed": False}
    path = out / "evaluation-overlap-audit.json"
    if path.exists():
        if read(path) != audit:
            raise ValueError("Evaluation-overlap audit changed")
    else:
        write_json(path, audit)
    if overlaps:
        raise ValueError("Generated material overlaps evaluation questions; preserve and stop")


def paper_rows(cases, tok):
    result = []
    for case in cases:
        for group in ("closed", "oracle_context"):
            user = case["question"]
            if group == "oracle_context":
                user = "\n\n".join(case["paragraphs"]) + "\n\nQuestion: " + user
            result.append(
                {
                    "id": f"{case['case_id']}/{group}",
                    "case_id": case["case_id"],
                    "group": group,
                    "question": case["question"],
                    "gold": case["answer"],
                    "prompt": render_chat(tok, user),
                }
            )
    return result


def retention_rows(source, tok):
    selected = {}
    for row in read(source)["natural"]:
        value = {
            "id": row["qid"],
            "group": "retention",
            "gold": row["gold"],
            "prompt": render_chat(tok, row["prompt"]),
            "type": row["type"],
            "exclude": row.get("exclude", []),
        }
        if row["qid"] in selected and selected[row["qid"]] != value:
            raise ValueError("Conflicting duplicate natural-retention question")
        selected[row["qid"]] = value
    if len(selected) != 317:
        raise ValueError("Natural retention must preserve the frozen 317-question pool")
    return list(selected.values())


def validate_checkpoint(saved, config, objective, arm, node):
    expected = {"cohort": "natural", "objective": objective, "arm": arm, "stage": "addition"}
    if saved.get("identity") != expected or saved["config"] != config or saved["tokens"] != node:
        raise ValueError("Natural checkpoint identity, configuration or token position differs")


def reload_model(runtime, config, node_dir, objective, arm, node):
    torch = runtime.torch
    saved = torch.load(node_dir / "resume.pt", map_location="cpu", weights_only=False)
    validate_checkpoint(saved, config, objective, arm, node)
    model, tok = runtime.K.load_model(config)
    model = runtime.F.configure_arm(model, config, f"{objective}_{config['scope']}")
    runtime.K.restore(model, saved["weights"])
    for name, parameter in model.named_parameters():
        if name in saved["weights"] and not torch.equal(
            parameter.detach().cpu(), saved["weights"][name]
        ):
            raise AssertionError(f"Reload weight differs: {name}")
    model.eval()
    reference = torch.load(node_dir / "reload-reference.pt", map_location="cpu", weights_only=False)
    with torch.no_grad(), runtime.K.autocast():
        logits = model(
            input_ids=reference["input_ids"].to("cuda"),
            attention_mask=reference["attention_mask"].to("cuda"),
            use_cache=False,
        ).logits
    difference = float((logits[:, :, :1024].cpu() - reference["logits"]).abs().max())
    passed = difference == 0 and torch.equal(logits.argmax(-1).cpu(), reference["greedy"])
    audit = {
        "passed": passed,
        "max_abs_logit_difference": difference,
        "scope": "two fixed training sequences; full saved trainable weights",
        "weights_sha256": digest(node_dir / "resume.pt"),
    }
    runtime.K.write(node_dir / "audit.json", audit)
    if not passed:
        raise AssertionError("Natural independent reload check failed")
    return model, tok


def retention_summary(before, after):
    b, a = ({r["id"]: r for r in group} for group in (before, after))
    if b.keys() != a.keys() or len(b) != 317:
        raise ValueError("Retention baseline and endpoint coverage differ")
    initially_correct = [key for key in b if b[key]["strict"]]
    retained = sum(bool(a[key]["strict"]) for key in initially_correct)
    return {
        "n": len(b),
        "initially_correct_n": len(initially_correct),
        "retained_n": retained,
        "before_correct_n": len(initially_correct),
        "after_correct_n": sum(bool(r["strict"]) for r in a.values()),
        "retained_fraction": retained / len(initially_correct) if initially_correct else None,
        "metric": "existing_natural_retention_alias_strict",
    }


def evaluate(
    config,
    batch,
    factorial_batch,
    *,
    objective=None,
    arm=None,
    node=None,
    source_csv=None,
    retention_source=None,
    smoke=False,
    runtime=None,
):
    runtime = runtime or _runtime()
    batch, factorial_batch = Path(batch), Path(factorial_batch)
    is_baseline = node is None
    if is_baseline:
        output = batch / "baseline/natural"
        label, phase = "base", "before"
    else:
        verify_preparation(batch, factorial_batch, source_csv)
        name = runtime.training_name("natural", objective, arm, "addition")
        node_dir = batch / ("engineering" if smoke else "runs") / name / f"node-{node}"
        output = node_dir / "evaluation"
        label, phase = f"{objective}_{arm}", "after"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "started.json", {"unix_time": time.time(), "automatic_retry": False})
    try:
        cases = application_cases(factorial_batch, source_csv)
        retention_source = source_path(factorial_batch, "retention-base.json", retention_source)
        if is_baseline:
            model, tok = runtime.K.load_model(config)
        else:
            model, tok = reload_model(runtime, config, node_dir, objective, arm, node)
        source = {
            "arm": label,
            "phase": phase,
            "application_case_indices": [32, 96],
            "model_revision": config["model_revision"],
            "weights_sha256": None if is_baseline else digest(node_dir / "resume.pt"),
            "evaluation_source_sha256": digest(factorial_batch / "shared/eval.json"),
            "retention_source_sha256": digest(retention_source),
        }
        answers = P.predict_records(
            model, tok, paper_rows(cases, tok), config, output / "generation", source
        )
        retained = runtime.K.predict(model, tok, retention_rows(retention_source, tok), config)
        write_rows(output / "retention.jsonl", retained)
        del model, tok
        gc.collect()
        runtime.torch.cuda.empty_cache()
        judge_rows = [
            {
                "id": f"{label}/{phase}/{r['id']}",
                "case_id": r["case_id"],
                "question": r["question"],
                "gold": r["gold"],
                "answer": r["answer"],
                "arm": label,
                "phase": phase,
                "group": r["group"],
            }
            for r in answers
        ]
        judged = P.grade_records(judge_rows, config, output / "judge")
        groups = {}
        for key, value in judged["summary"]["groups"].items():
            groups[key.rsplit("/", 1)[-1]] = {
                **value,
                "correct": value["correct_n"],
                "accuracy": value["paper_accuracy_valid_denominator"],
            }
        before = retained if is_baseline else rows(batch / "baseline/natural/retention.jsonl")
        write_json(
            output / "summary.json",
            {
                "complete": True,
                "groups": groups,
                "retention": retention_summary(before, retained),
                "source": source,
                "application_only": True,
                "new_confirmation_set": False,
                "limitations": LIMITATIONS,
                "generation_total_token_budget": 6128,
                "generation_max_new_tokens": 500,
                "generation_temperature": 0.25,
                "independent_reload_passed": not is_baseline,
            },
        )
        if not is_baseline and not smoke:
            runtime.sync_node_tracking(config, batch, node_dir.parent, node_dir)
    except BaseException:
        write_json(
            output / "failure.json", {"traceback": traceback.format_exc(), "automatic_retry": False}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--factorial-batch", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--retention-source", type=Path)
    parser.add_argument(
        "--action", choices=("calibrate", "prepare", "train", "baseline", "evaluate"), required=True
    )
    parser.add_argument("--objective", choices=("pd", "sft"), default="pd")
    parser.add_argument("--arm", choices=tuple("ABC"), default="A")
    parser.add_argument("--nodes", type=int, nargs="+")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    secret = Path("/run/secrets/wandb")
    if secret.exists():
        os.environ["WANDB_API_KEY"] = secret.read_text().strip()
    config = natural_config(read(args.config), args.factorial_batch)
    if args.action == "calibrate":
        calibrate(config, args.batch, args.factorial_batch, source_csv=args.source_csv)
    elif args.action == "prepare":
        prepare(config, args.batch, args.factorial_batch, source_csv=args.source_csv)
    elif args.action == "train":
        if not args.nodes or not config.get("natural_token_nodes"):
            parser.error("Natural training requires frozen --nodes and natural_token_nodes")
        verify_preparation(args.batch, args.factorial_batch, args.source_csv)
        _runtime().train(
            config,
            args.batch,
            "natural",
            args.objective,
            args.arm,
            "addition",
            args.nodes,
            smoke=args.smoke,
        )
    else:
        if args.action == "evaluate" and not args.nodes:
            parser.error("Natural evaluation requires --nodes")
        for node in args.nodes if args.action == "evaluate" else [None]:
            evaluate(
                config,
                args.batch,
                args.factorial_batch,
                objective=args.objective,
                arm=args.arm,
                node=node,
                source_csv=args.source_csv,
                retention_source=args.retention_source,
                smoke=args.smoke,
            )


if __name__ == "__main__":
    main()
