#!/usr/bin/env python3
"""Bounded GPU stages for the token-matched knowledge-use comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import torch

from llm_memory_editability import knowledge_method_factorial as F
from llm_memory_editability import knowledge_use_coverage_data as D
from llm_memory_editability.knowledge_method_data import (
    context_user,
    final_answer_text,
    render_chat,
)
from llm_memory_editability.knowledge_use_coverage_budget import (
    TokenStream,
    learning_rate_scale,
    microbatch_weight,
)

K = F.shared


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate(model, tok, rows, config, context=False, temperature=0.0):
    """No training state or evaluation gold is an input to the model."""
    model.eval()
    results = []
    size = config["generation_batch_size"]
    for start in range(0, len(rows), size):
        part = rows[start : start + size]
        prompts = [
            render_chat(
                tok, context_user(r["context"], r["question"]) if context else r["question"]
            )
            for r in part
        ]
        inputs = tok(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(
            "cuda"
        )
        if (
            inputs["input_ids"].shape[1] + config["generation_max_new_tokens"]
            > config["max_sequence_tokens"]
        ):
            raise ValueError("Generation exceeds configured sequence length; no truncation")
        with torch.no_grad(), K.autocast():
            output = model.generate(
                **inputs,
                max_new_tokens=config["generation_max_new_tokens"],
                do_sample=temperature > 0,
                **(
                    {
                        "temperature": temperature,
                        "top_p": 1.0,
                        "top_k": config.get("generation_top_k", 0),
                    }
                    if temperature
                    else {}
                ),
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        for row_index, (row, ids) in enumerate(
            zip(part, output[:, inputs["input_ids"].shape[1] :].tolist(), strict=True)
        ):
            eos = tok.eos_token_id in ids
            if eos:
                ids = ids[: ids.index(tok.eos_token_id) + 1]
            text = tok.decode(ids, skip_special_tokens=True)
            try:
                answer = final_answer_text(text)
                valid = True
            except ValueError:
                answer, valid = text, False
            results.append(
                {
                    "id": row["id"],
                    "answer": answer,
                    "raw": text,
                    "token_ids": ids,
                    "truncated": not eos,
                    "final_answer_valid": valid,
                    "prompt_tokens": int(inputs["attention_mask"][row_index].sum()),
                    "generated_tokens": len(ids),
                }
            )
        print(
            json.dumps({"generation_done": len(results), "generation_total": len(rows)}), flush=True
        )
    return results


def prepare(config, batch, cohort):
    """Generate only from train/replay inputs, without opening evaluation files."""
    out = batch / "data" / cohort
    if (out / "prepare-started.json").exists():
        raise FileExistsError("Existing preparation attempt must not be regenerated")
    K.write(out / "prepare-started.json", {"utc_unix": time.time(), "pid": os.getpid()})
    model, tok = K.load_model(config)
    K.seed_all(config["seed"])
    start = time.perf_counter()
    sources = {arm: read_rows(out / f"train-{arm}.jsonl") for arm in ("A", "B", "C")}
    sources["rewrite"] = read_rows(out / "train-rewrite.jsonl")
    unique = {}
    key_to_id = {}
    for rows in sources.values():
        for row in rows:
            key = (row["context"], row["question"])
            if key not in key_to_id:
                key_to_id[key] = f"trajectory-{len(unique):06d}"
                unique[key_to_id[key]] = {
                    "id": key_to_id[key],
                    "context": key[0],
                    "question": key[1],
                }
    bank = {}
    costs = {}
    answer_quality = {}
    for objective in ("pd", "sft"):
        rows = list(unique.values())
        generation_began = time.perf_counter()
        answers = generate_saved(
            model,
            tok,
            rows,
            config,
            out / f"raw-{objective}-trajectories.jsonl",
            context=True,
            temperature=config[f"{objective}_temperature"],
        )
        costs[objective] = generation_cost(answers, time.perf_counter() - generation_began)
        if any(not x["final_answer_valid"] for x in answers):
            raise ValueError("Invalid sampled final-answer boundary; preserve generation attempt")
        bank[objective] = {r["id"]: r for r in answers}
    for arm, rows in sources.items():
        prepared = []
        for row in rows:
            key = key_to_id[(row["context"], row["question"])]
            prepared.append(
                {
                    **row,
                    "trajectory_id": key,
                    **{
                        f"{objective}_{field}": bank[objective][key][field]
                        for objective in ("pd", "sft")
                        for field in ("answer", "truncated")
                    },
                }
            )
        write_rows(out / f"prepared-{arm}.jsonl", prepared)
        answer_quality[arm] = {}
        for objective in ("pd", "sft"):
            scores = [D.score_answer(r, r[f"{objective}_answer"]) for r in prepared]
            answer_quality[arm][objective] = {
                "n": len(scores),
                "correct": sum(r["correct"] for r in scores),
                "unparsed": sum(r["parse_status"] == "unparsed" for r in scores),
                "truncated": sum(r[f"{objective}_truncated"] for r in prepared),
            }
    replay = read_rows(out / "replay.jsonl")
    generation_began = time.perf_counter()
    answers = generate_saved(model, tok, replay, config, out / "raw-replay-trajectories.jsonl")
    costs["replay"] = generation_cost(answers, time.perf_counter() - generation_began)
    if any(not row["final_answer_valid"] for row in answers):
        raise ValueError("Invalid replay final-answer boundary; preserve generation attempt")
    prepared_replay = [
        {
            **r,
            "context": "",
            "pd_answer": a["answer"],
            "sft_answer": a["answer"],
            "pd_truncated": a["truncated"],
            "sft_truncated": a["truncated"],
        }
        for r, a in zip(replay, answers, strict=True)
    ]
    write_rows(out / "prepared-replay.jsonl", prepared_replay)
    K.write(
        out / "prepare-complete.json",
        {
            "unique_training_prompts": len(unique),
            "rows": {a: len(r) for a, r in sources.items()},
            "seconds": time.perf_counter() - start,
            "generation_cost": costs,
            "training_answer_quality": answer_quality,
            "answer_quality_filtering": False,
            "model_revision": config["model_revision"],
            "trajectory_source": "frozen_base_with_context",
            "evaluation_files_read": False,
            "sha256": {p.name: K.digest(p) for p in out.glob("prepared-*.jsonl")},
        },
    )


def generation_cost(answers, seconds):
    return {
        "requests": len(answers),
        "input_tokens": sum(r["prompt_tokens"] for r in answers),
        "generated_tokens": sum(r["generated_tokens"] for r in answers),
        "truncated": sum(r["truncated"] for r in answers),
        "seconds": seconds,
    }


def generate_saved(model, tok, rows, config, path, context=False, temperature=0.0):
    """Preserve each completed generation batch even if a later request fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    answers = []
    with path.open("x") as handle:
        size = config["generation_batch_size"]
        for start in range(0, len(rows), size):
            chunk = generate(model, tok, rows[start : start + size], config, context, temperature)
            for row in chunk:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            answers.extend(chunk)
    return answers


def evaluation_set(batch, cohort, stage):
    out = batch / "data" / cohort
    return read_rows(out / f"evaluation-{stage}.jsonl") + read_rows(
        out / f"retention-{stage}.jsonl"
    )


def evaluate_model(model, tok, rows, config, out, with_context=False, chains=True):
    if (out / "summary.json").exists():
        raise FileExistsError(f"Evaluation already exists: {out}")
    cfg = {
        **config,
        "generation_max_new_tokens": config["eval_max_new_tokens"],
        "generation_batch_size": config["eval_batch_size"],
    }
    start = time.perf_counter()
    answers = generate_saved(
        model, tok, rows, cfg, out / "raw-generations.jsonl", context=with_context
    )
    scored = []
    for row, answer in zip(rows, answers, strict=True):
        score = D.score_answer(row, answer["answer"])
        if row["category"] == "natural_old_retention":
            from llm_memory_editability.read_interface_data import graders

            strict = graders()[row["type"]].grade(
                answer["answer"], row["gold"], set(row.get("exclude", []))
            )["strict"]
            score.update(correct=bool(strict), scoring="existing_natural_alias_grader")
        scored.append({**row, **answer, **score})
    write_rows(out / "predictions.jsonl", scored)
    chain_results = []
    if chains and not with_context:
        active = [r for r in rows if r.get("chain") and r.get("mode") == "direct"]
        states = {r["id"]: {"answers": {}, "trace": [], "failed": False} for r in active}
        for depth in range(max((len(r["chain"]) for r in active), default=0)):
            pending = []
            for row in active:
                state = states[row["id"]]
                if state["failed"] or depth >= len(row["chain"]):
                    continue
                question = row["chain"][depth]["question"].format(**state["answers"])
                pending.append({"id": row["id"], "question": question})
            by_id = {a["id"]: a for a in generate(model, tok, pending, cfg)}
            for row in active:
                if row["id"] not in by_id:
                    continue
                answer = by_id[row["id"]]
                parsed, status = D.extract_answer(answer["answer"])
                state = states[row["id"]]
                state["trace"].append(
                    {
                        "question": next(p["question"] for p in pending if p["id"] == row["id"]),
                        **answer,
                    }
                )
                state["answers"][row["chain"][depth]["answer_slot"]] = parsed
                state["failed"] = status == "unparsed"
        for row in active:
            state = states[row["id"]]
            last = state["trace"][-1]["answer"] if state["trace"] else ""
            score = D.score_answer(row, last)
            chain_results.append(
                {
                    **row,
                    "mode": "self_generated_chain",
                    **state,
                    **score,
                    "correct": score["correct"] and not state["failed"],
                }
            )
        write_rows(out / "chains.jsonl", chain_results)
    groups = defaultdict(lambda: {"n": 0, "correct": 0, "unparsed": 0})
    for row in scored + chain_results:
        key = f"{row['category']}/{row.get('mode', 'direct')}"
        groups[key]["n"] += 1
        groups[key]["correct"] += int(row["correct"])
        groups[key]["unparsed"] += int(row["parse_status"] == "unparsed")
    summary = {k: {**v, "accuracy": v["correct"] / v["n"]} for k, v in groups.items()}
    K.write(
        out / "summary.json",
        {
            "complete": True,
            "with_context": with_context,
            "groups": summary,
            "seconds": time.perf_counter() - start,
        },
    )
    return summary


def baseline(config, batch, cohort):
    model, tok = K.load_model(config)
    rows = evaluation_set(batch, cohort, "addition")
    evaluate_model(model, tok, rows, config, batch / "baseline" / cohort / "closed")
    evaluate_model(
        model,
        tok,
        rows,
        config,
        batch / "baseline" / cohort / "context",
        with_context=True,
        chains=False,
    )
    evaluate_model(
        model,
        tok,
        evaluation_set(batch, cohort, "rewrite"),
        config,
        batch / "baseline" / cohort / "rewrite-context",
        with_context=True,
        chains=False,
    )


def encoded_rows(tok, rows, objective, config, replay=False):
    encoded = [
        {
            **F.encode_sample(
                tok, r, f"{objective}_{config['scope']}", config["max_sequence_tokens"]
            ),
            "fact_ids": r.get("fact_ids", []),
            "context_fact_ids": r.get("context_fact_ids", []),
            "case_id": r.get("case_id"),
            "category": r.get("category"),
        }
        for r in rows
    ]
    if replay:
        # No document separator or source information in the protected teacher.
        for row in encoded:
            row["teacher"] = list(row["student"])
    return encoded


def teacher_for_all(config):
    teacher, _ = K.load_model(config)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher.config.use_cache = False
    return teacher


def loss_backward(model, teacher, samples, objective, config, multiplier=1.0):
    size = config["microbatch_size"]
    total_loss = 0.0
    costs = Counter()
    for index in range(0, len(samples), size):
        part = samples[index : index + size]
        minibatch = K.collate(part, model.config.pad_token_id)
        for side in ("student", "teacher") if objective == "pd" else ("student",):
            costs[f"{side}_tokens"] += int(minibatch[side]["attention_mask"].sum())
            costs[f"{side}_padded_tokens"] += minibatch[side]["input_ids"].numel()
        loss = F.batch_loss(
            model,
            minibatch,
            f"{objective}_{config['scope']}",
            config,
            teacher if objective == "pd" else None,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        weight = microbatch_weight(objective, part, samples)
        (loss * weight * multiplier).backward()
        total_loss += float(loss.detach()) * weight
    return total_loss, costs


def training_name(cohort, objective, arm, stage):
    return f"{cohort}-{objective}-{arm}-{stage}"


def train(config, batch, cohort, objective, arm, stage, nodes, parent=None, smoke=False):
    try:
        return _train(config, batch, cohort, objective, arm, stage, nodes, parent, smoke)
    except BaseException:
        out = (
            batch
            / ("engineering" if smoke else "runs")
            / training_name(cohort, objective, arm, stage)
        )
        started = out / "started.json"
        if started.exists() and read(started).get("pid") == os.getpid():
            K.write(out / "failure.json", {"traceback": traceback.format_exc(), "pid": os.getpid()})
        raise


def _train(config, batch, cohort, objective, arm, stage, nodes, parent=None, smoke=False):
    """One attempt, fixed nodes, no implicit restart or automatic retry."""
    from knowledge_checkpoint_io import CheckpointTransport

    if (
        not nodes
        or nodes != sorted(set(nodes))
        or any(n <= 0 or n % config["main_tokens_per_update"] for n in nodes)
    ):
        raise ValueError("Nodes must be increasing complete positive optimizer updates")
    if config["scope"] not in {"down", "full"}:
        raise ValueError("This runner requires independently frozen dense teacher")
    data = batch / "data" / cohort
    preparation = read(data / "prepare-complete.json")
    for file, expected_hash in preparation["sha256"].items():
        if K.digest(data / file) != expected_hash:
            raise ValueError(f"Prepared data hash differs: {file}")
    required = {
        f"prepared-{'rewrite' if stage == 'rewrite' else arm}.jsonl",
        "prepared-replay.jsonl",
    }
    if not required.issubset(preparation["sha256"]):
        raise ValueError("Prepared data is not covered by completion receipt")
    name = training_name(cohort, objective, arm, stage)
    identity = {"cohort": cohort, "objective": objective, "arm": arm, "stage": stage}
    if stage == "rewrite" and parent is None:
        raise ValueError("Rewrite requires the corresponding addition checkpoint")
    if stage == "addition" and parent is not None:
        raise ValueError("Addition must start from the common base")
    out = batch / ("engineering" if smoke else "runs") / name
    if (out / "started.json").exists():
        raise FileExistsError(f"Existing attempt: {out}")
    K.write(out / "started.json", {"unix_time": time.time(), "pid": os.getpid()})
    model, tok = K.load_model(config)
    model.config.pad_token_id = tok.pad_token_id
    model = F.configure_arm(model, config, f"{objective}_{config['scope']}")
    if parent:
        saved = torch.load(parent, map_location="cpu", weights_only=False)
        expected = {**identity, "stage": "addition"}
        if saved.get("identity") != expected or saved["config"] != config:
            raise ValueError("Parent trajectory identity or configuration differs")
        audit = read(parent.parent / "audit.json")
        if not audit["passed"] or audit["weights_sha256"] != K.digest(parent):
            raise ValueError("Parent checkpoint has not passed independent reload")
        K.restore(model, saved["weights"])
        del saved
    teacher = teacher_for_all(config)
    F._validate_independent_teacher(model, teacher)
    K.seed_all(config["seed"])
    rows = read_rows(data / f"prepared-{'rewrite' if stage == 'rewrite' else arm}.jsonl")
    encoded = encoded_rows(tok, rows, objective, config)
    replay = encoded_rows(tok, read_rows(data / "prepared-replay.jsonl"), "pd", config, replay=True)
    main_stream = TokenStream(encoded, config["seed"])
    replay_stream = TokenStream(replay, config["seed"] + 1)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    maximum = max(nodes)
    horizon = max(config[f"{cohort}_token_nodes"])
    if stage == "rewrite":
        horizon = int(horizon * config["rewrite_token_fraction"])
    tracker = None
    if not smoke:
        K.write(
            out / "run.json",
            {
                "spec": {
                    "config": config,
                    "cohort": cohort,
                    "objective": objective,
                    "arm": arm,
                    "stage": stage,
                    "nodes": nodes,
                    "parent": str(parent) if parent else None,
                },
                "model": config["model"],
                "tracking_group": config["batch"],
                "pid": os.getpid(),
                "gpu": 0,
                "gpu_name": torch.cuda.get_device_name(0),
                "parameters": F.parameter_contract(model, f"{objective}_{config['scope']}"),
                "world_sha256": K.digest(data / "manifest.json"),
                "initial_model_sha256": config["model_revision"],
            },
        )
        K.write(out / "learning.json", [])
        tracker = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "scripts/track_knowledge_use_coverage.py",
                "--batch",
                str(batch),
                "--run",
                str(out),
            ],
            stdout=(out / "tracking.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        ready = batch / "tracking-wandb" / name / "state.json"
        deadline = time.monotonic() + 90
        while not ready.exists():
            if tracker.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Tracking did not initialize; scientific updates remain zero")
            time.sleep(1)
    io = CheckpointTransport(K.atomic_torch_save)
    history = []
    counters = Counter()
    began = time.perf_counter()
    try:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        step = 0
        while main_stream.tokens < maximum:
            main = main_stream.take(
                min(config["main_tokens_per_update"], maximum - main_stream.tokens)
            )
            protected = replay_stream.take(config["replay_tokens_per_update"])
            optimizer.zero_grad(set_to_none=True)
            rate = config["learning_rate"] * learning_rate_scale(
                main_stream.tokens, horizon, config["warmup_token_fraction"]
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            loss, main_cost = loss_backward(model, teacher, main, objective, config)
            kl, replay_cost = loss_backward(
                model, teacher, protected, "pd", config, multiplier=config["retention_kl_weight"]
            )
            norm = torch.nn.utils.clip_grad_norm_(
                params, config["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            step += 1
            for segment in main:
                counters[f"main/{segment['id']}"] += len(segment["answer"])
                if segment["original_answer_offset"] == 0:
                    counters[f"example_starts/{segment['id']}"] += 1
                    for fact_id in segment["fact_ids"]:
                        counters[f"target_fact_exposures/{fact_id}"] += 1
                    for fact_id in segment["context_fact_ids"]:
                        counters[f"teacher_context_fact_exposures/{fact_id}"] += 1
                    if segment["case_id"]:
                        counters[f"case_exposures/{segment['case_id']}"] += 1
            counters.update(main_cost)
            counters.update(replay_cost)
            counters.update({f"main_{k}": v for k, v in main_cost.items()})
            counters.update({f"replay_{k}": v for k, v in replay_cost.items()})
            if step % 10 == 0 or main_stream.tokens in nodes or smoke:
                record = {
                    "step": step,
                    "loss": loss,
                    "retention_kl": kl,
                    "grad_norm": float(norm),
                    "learning_rate": rate,
                    "supervised_tokens": main_stream.tokens,
                    "replay_tokens": replay_stream.tokens,
                    "examples": main_stream.examples_started,
                    "training_seconds": time.perf_counter() - began,
                    "student_tokens": counters["student_tokens"],
                    "teacher_tokens": counters["teacher_tokens"],
                    "student_padded_tokens": counters["student_padded_tokens"],
                    "teacher_padded_tokens": counters["teacher_padded_tokens"],
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                }
                history.append(record)
                K.write(out / "learning.json", history)
                K.write(out / "progress.json", record)
            if main_stream.tokens in nodes:
                snapshot = {
                    "weights": K.state(model),
                    "optimizer": optimizer.state_dict(),
                    "rng": K.rng_state(),
                    "main_stream": main_stream.state_dict(),
                    "replay_stream": replay_stream.state_dict(),
                    "config": config,
                    "identity": identity,
                    "step": step,
                    "tokens": main_stream.tokens,
                    "counters": dict(counters),
                }
                node = out / f"node-{main_stream.tokens}"
                io.save(snapshot, node / "resume.pt")
                # Only two fixed sequences for a separately scoped exact reload check.
                fixed = K.collate(encoded[:2], tok.pad_token_id)["student"]
                model.eval()
                with torch.no_grad(), K.autocast():
                    logits = model(
                        input_ids=fixed["input_ids"],
                        attention_mask=fixed["attention_mask"],
                        use_cache=False,
                    ).logits
                K.atomic_torch_save(
                    {
                        "input_ids": fixed["input_ids"].cpu(),
                        "attention_mask": fixed["attention_mask"].cpu(),
                        "logits": logits[:, :, :1024].cpu(),
                        "greedy": logits.argmax(-1).cpu(),
                    },
                    node / "reload-reference.pt",
                )
                model.train()
        io.close()
        K.write(out / "exposure.json", dict(counters))
        K.write(
            out / "complete.json",
            {
                **history[-1],
                "nodes": nodes,
                "smoke": smoke,
                "parent_optimizer_reset": bool(parent),
                "teacher_frozen": True,
                "elapsed_total_seconds": time.time() - read(out / "started.json")["unix_time"],
                "cost_limitations": [
                    "Teacher compute time and FLOPs are not measured separately",
                    "Input token counts include re-encoded split-answer prefixes; "
                    "padded counts exclude checkpoint recomputation",
                ],
            },
        )
    except Exception:
        K.write(out / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        io.close()
        if tracker:
            try:
                tracker.wait(timeout=90)
            except subprocess.TimeoutExpired:
                K.write(out / "tracking-needs-attention.json", {"pid": tracker.pid})


def reload_and_evaluate(config, batch, cohort, objective, arm, stage, node, smoke=False):
    out = (
        batch
        / ("engineering" if smoke else "runs")
        / training_name(cohort, objective, arm, stage)
        / f"node-{node}"
    )
    saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
    if saved["config"] != config or saved["tokens"] != node:
        raise ValueError("Checkpoint configuration or token position differs")
    identity = {"cohort": cohort, "objective": objective, "arm": arm, "stage": stage}
    if saved.get("identity") != identity:
        raise ValueError("Checkpoint trajectory identity differs")
    model, tok = K.load_model(config)
    model = F.configure_arm(model, config, f"{objective}_{config['scope']}")
    K.restore(model, saved["weights"])
    for name, parameter in model.named_parameters():
        if name in saved["weights"] and not torch.equal(
            parameter.detach().cpu(), saved["weights"][name]
        ):
            raise AssertionError(f"Reload weight differs: {name}")
    model.eval()
    ref = torch.load(out / "reload-reference.pt", map_location="cpu", weights_only=False)
    with torch.no_grad(), K.autocast():
        logits = model(
            input_ids=ref["input_ids"].to("cuda"),
            attention_mask=ref["attention_mask"].to("cuda"),
            use_cache=False,
        ).logits
    maximum = float((logits[:, :, :1024].cpu() - ref["logits"]).abs().max())
    passed = maximum == 0 and torch.equal(logits.argmax(-1).cpu(), ref["greedy"])
    K.write(
        out / "audit.json",
        {
            "passed": passed,
            "max_abs_logit_difference": maximum,
            "scope": "two fixed training sequences; full saved trainable weights",
            "weights_sha256": K.digest(out / "resume.pt"),
        },
    )
    if not passed:
        raise AssertionError("Independent reload check failed")
    del saved, ref, logits
    if smoke:
        return
    rows = evaluation_set(batch, cohort, stage)
    evaluate_model(model, tok, rows, config, out / "evaluation")
    sync_node_tracking(config, batch, out.parent, out)


def flatten_summary(value, prefix):
    """W&B public summaries store nested mappings as dotted scalar keys."""
    result = {}
    for name, item in value.items():
        key = f"{prefix}.{name}"
        if isinstance(item, dict) and item:
            result.update(flatten_summary(item, key))
        else:
            result[key] = item
    return result


def node_summary_equal(summary, key, payload):
    import math

    expected = flatten_summary(payload, key)
    nested = summary.get(key)
    actual = flatten_summary(nested, key) if isinstance(nested, dict) else summary
    for name, value in expected.items():
        observed = actual.get(name)
        if isinstance(value, float) and isinstance(observed, (float, int)):
            # JSON transport can round the last floating-point digit. Counts,
            # flags, strings and weight identities still require exact equality.
            if not math.isclose(value, observed, rel_tol=1e-14, abs_tol=1e-14):
                return False
        elif observed != value:
            return False
    return True


def sync_node_tracking(config, batch, run_dir, node_dir):
    """Append evaluation to its finished training run, then independently read it back."""
    import wandb

    tracking = read(batch / "tracking-wandb" / run_dir.name / "state.json")
    identity = f"{tracking['entity']}/{tracking['project']}/{tracking['run_id']}"
    remote = wandb.Api(timeout=60).run(identity)
    metadata = read(run_dir / "run.json")
    if remote.config["spec"] != metadata["spec"]:
        raise ValueError("Cloud trajectory configuration differs")
    if remote.state != "finished":
        raise ValueError("Cloud training endpoint has not finished")
    complete = read(run_dir / "complete.json")
    for key, expected in {
        "scientific_final_step": complete["step"],
        "exposure/supervised_tokens": complete["supervised_tokens"],
    }.items():
        if remote.summary.get(key) != expected:
            raise ValueError(f"Cloud training count differs: {key}")
    audit = read(node_dir / "audit.json")
    evaluation = read(node_dir / "evaluation" / "summary.json")
    if not audit["passed"] or not evaluation["complete"]:
        raise ValueError("Node does not have a verified complete evaluation")
    key = f"nodes/{node_dir.name}"
    payload = {"audit": audit, "evaluation": evaluation}
    for name, value in flatten_summary(payload, key).items():
        remote.summary[name] = value
    remote.summary["independently_reloaded"] = all(
        (run_dir / f"node-{n}" / "audit.json").exists()
        and read(run_dir / f"node-{n}" / "audit.json")["passed"]
        for n in complete["nodes"]
    )
    remote.summary.update()
    checked = wandb.Api(timeout=60).run(identity)
    if not node_summary_equal(checked.summary, key, payload):
        raise AssertionError("Cloud node summary readback differs")
    K.write(
        node_dir / "cloud-verification.json",
        {
            "passed": True,
            "url": tracking["url"],
            "state": checked.state,
            "run_id": tracking["run_id"],
            "config_equal": True,
            "node_summary_equal": True,
            "comparison": "dotted scalar keys; float tolerance 1e-14; counts and identities exact",
            "utc_unix": time.time(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=["prepare", "baseline", "train", "evaluate", "sync-tracking"],
        required=True,
    )
    parser.add_argument("--cohort", choices=["development", "formal"], required=True)
    parser.add_argument("--objective", choices=["pd", "sft"], default="pd")
    parser.add_argument("--arm", choices=["A", "B", "C"], default="A")
    parser.add_argument("--stage", choices=["addition", "rewrite"], default="addition")
    parser.add_argument("--nodes", type=int, nargs="+")
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    secret = Path("/run/secrets/wandb")
    if secret.exists():
        os.environ["WANDB_API_KEY"] = secret.read_text().strip()
    if args.action == "prepare":
        prepare(config, args.batch, args.cohort)
    elif args.action == "baseline":
        baseline(config, args.batch, args.cohort)
    elif args.action == "train":
        train(
            config,
            args.batch,
            args.cohort,
            args.objective,
            args.arm,
            args.stage,
            args.nodes,
            args.parent,
            args.smoke,
        )
    elif args.action == "sync-tracking":
        run = args.batch / "runs" / training_name(args.cohort, args.objective, args.arm, args.stage)
        for node in args.nodes:
            sync_node_tracking(config, args.batch, run, run / f"node-{node}")
    else:
        for node in args.nodes:
            reload_and_evaluate(
                config,
                args.batch,
                args.cohort,
                args.objective,
                args.arm,
                args.stage,
                node,
                args.smoke,
            )


if __name__ == "__main__":
    main()
