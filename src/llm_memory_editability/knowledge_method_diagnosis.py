"""Bounded transplant of Prompt Distillation and its SFT comparison to Qwen3.

Original source: kallekku/prompt-distillation, commit 2b72919a.
No T^2 multiplier; forward KL is averaged over tokens per example, then examples.
Evaluation questions are never training inputs. All arms save reloadable weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def load_model(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model_path"],
            dtype=torch.float32,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    return model, tok


def configure_arm(model, config, arm):
    if arm not in config["arms"]:
        raise ValueError(arm)
    if arm.endswith("lora"):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=config["lora_dropout"],
                target_modules=config["lora_targets"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(arm == "sft_full" or ".mlp.down_proj." in name)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    return model


def encode_sample(tok, row, arm, max_length):
    from .knowledge_method_data import chat_user, context_user, render_chat

    key = "pd_answer" if arm == "pd_lora" else "sft_answer"
    answer = row[key]
    # Data preparation records truncation; never teach an artificial EOS on a cut answer.
    cut = row.get("pd_truncated" if arm == "pd_lora" else "sft_truncated", False)
    answer_ids = tok.encode(answer, add_special_tokens=False)
    if not cut and (not answer_ids or answer_ids[-1] != tok.eos_token_id):
        answer_ids.append(tok.eos_token_id)
    if not answer_ids:
        raise ValueError(f"Empty answer: {row['id']}")
    student = tok.encode(render_chat(tok, chat_user(row["question"])), add_special_tokens=False)
    teacher = tok.encode(
        render_chat(tok, context_user(row["context"], row["question"])),
        add_special_tokens=False,
    )
    if max(len(student), len(teacher)) + len(answer_ids) > max_length:
        raise ValueError(f"Sequence exceeds frozen limit; do not silently truncate {row['id']}")
    return {"student": student, "teacher": teacher, "answer": answer_ids, "id": row["id"]}


def collate(samples, pad_id, device="cuda"):
    result = {}
    for side in ("student", "teacher"):
        seqs = [sample[side] + sample["answer"] for sample in samples]
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (seq, sample) in enumerate(zip(seqs, samples, strict=True)):
            ids[i, : len(seq)] = torch.tensor(seq, device=device)
            attention[i, : len(seq)] = 1
            answer_mask[i, len(sample[side]) : len(seq)] = True
        result[side] = {
            "input_ids": ids[:, :-1],
            "attention_mask": attention[:, :-1],
            "labels": ids[:, 1:],
            "answer_mask": answer_mask[:, 1:],
        }
    if not torch.equal(
        result["student"]["answer_mask"].sum(1), result["teacher"]["answer_mask"].sum(1)
    ):
        raise AssertionError("Teacher/student target alignment differs")
    return result


def forward_kl(student, teacher, counts, temperature):
    """Aligned answer logits only, with the author's per-example normalization."""
    if student.shape != teacher.shape or int(counts.sum()) != len(student):
        raise ValueError("Unaligned teacher/student targets")
    if bool((counts <= 0).any()):
        raise ValueError("Every example needs an answer token")
    # Chunk the full-vocabulary softmax, not the vocabulary, to reduce peak memory.
    values = []
    for s, t in zip(student.split(64), teacher.split(64), strict=True):
        values.append(
            F.kl_div(
                F.log_softmax(s.float() / temperature, dim=-1),
                F.log_softmax(t.float() / temperature, dim=-1),
                log_target=True,
                reduction="none",
            ).sum(-1)
        )
    losses = torch.cat(values)
    return torch.stack([v.mean() for v in losses.split(counts.tolist())]).mean()


def batch_loss(model, batch, arm, config):
    s = batch["student"]
    if arm == "pd_lora":
        t = batch["teacher"]
        # Disable both LoRA and dropout for the frozen contextual teacher.
        with model.disable_adapter(), torch.no_grad(), autocast():
            model.eval()
            teacher = (
                model(input_ids=t["input_ids"], attention_mask=t["attention_mask"], use_cache=False)
                .logits[t["answer_mask"]]
                .detach()
            )
        model.train()
    with autocast():
        logits = model(
            input_ids=s["input_ids"], attention_mask=s["attention_mask"], use_cache=False
        ).logits[s["answer_mask"]]
    if arm == "pd_lora":
        return forward_kl(logits, teacher, s["answer_mask"].sum(1), config["kl_temperature"])
    # Official SFT uses a token mean across the batch, not a per-example mean.
    return F.cross_entropy(logits.float(), s["labels"][s["answer_mask"]])


def state(model):
    return {
        name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad
    }


def restore(model, weights):
    parameters = dict(model.named_parameters())
    expected = {name for name, p in parameters.items() if p.requires_grad}
    if set(weights) != expected:
        raise ValueError("Saved trainable parameter names differ from frozen arm")
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    torch.cuda.set_rng_state_all(value["cuda"])


def wandb_run(config, batch, stage, arm=None):
    import os

    import wandb

    secret = Path("/run/secrets/wandb")
    if secret.is_file():
        os.environ["WANDB_API_KEY"] = secret.read_text().strip()
    name = f"{arm or 'shared'}-{stage}"
    run_id = hashlib.sha256((config["batch"] + name).encode()).hexdigest()[:16]
    run = wandb.init(
        entity=config["wandb_entity"],
        project=config["wandb_project"],
        group=config["batch"],
        name=name,
        id=run_id,
        resume="allow",
        config=config,
        dir=str(batch),
        mode="online",
    )
    write(batch / "tracking" / f"{name}.json", {"id": run.id, "url": run.url})
    return run


def train(config, batch, arm, smoke_steps=None):
    out = batch / "runs" / arm
    if (out / "complete.json").exists():
        raise RuntimeError("Completed training must not be restarted")
    if not (batch / "shared/prepare-complete.json").exists():
        raise RuntimeError("Preparation has not passed its coverage checks")
    seed_all(config["seed"])
    model, tok = load_model(config)
    model = configure_arm(model, config, arm)
    rows = [json.loads(line) for line in (batch / "shared/train.jsonl").read_text().splitlines()]
    if smoke_steps is None and len(rows) != config["expected_training_rows"]:
        raise ValueError("Training coverage differs from frozen contract")
    samples = [encode_sample(tok, row, arm, config["max_sequence_tokens"]) for row in rows]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    steps_per_epoch = math.ceil(len(samples) / config["batch_size"])
    total = steps_per_epoch * config["epochs"]
    if smoke_steps is not None:
        total = min(total, smoke_steps)
    manifest = {
        "arm": arm,
        "rows": len(rows),
        "steps": total,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "data_sha256": digest(batch / "shared/train.jsonl"),
        "config": config,
        "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
    }
    write(out / "manifest.json", manifest)
    checkpoint = out / "resume.pt"
    start = 0
    exposure = 0
    answer_tokens = 0
    forward_tokens = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["data_sha256"] != manifest["data_sha256"] or saved["config"] != config:
            raise RuntimeError("Resume data/config mismatch")
        restore(model, saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, exposure = saved["step"], saved["exposure"]
        answer_tokens, forward_tokens = saved["answer_tokens"], saved["forward_tokens"]
    run = None if smoke_steps is not None else wandb_run(config, batch, "train", arm)
    began = time.time()
    record = {
        "step": start,
        "exposure": exposure,
        "answer_tokens": answer_tokens,
        "forward_tokens": forward_tokens,
        "resumed_at_terminal_step": start == total,
    }
    try:
        model.train()
        for step in range(start, total):
            epoch, index = divmod(step, steps_per_epoch)
            order = list(range(len(samples)))
            random.Random(config["seed"] + epoch).shuffle(order)
            chosen = order[index * config["batch_size"] : (index + 1) * config["batch_size"]]
            batch_t = collate([samples[i] for i in chosen], tok.pad_token_id)
            # Author's default linear schedule, including initial warmup.
            warmup = config["warmup_steps"]
            scale = (
                step / max(1, warmup) if step < warmup else (total - step) / max(1, total - warmup)
            )
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * scale
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch_t, arm, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, config["max_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            exposure += len(chosen)
            answer_tokens += int(batch_t["student"]["answer_mask"].sum())
            forward_tokens += int(batch_t["student"]["attention_mask"].sum())
            if arm == "pd_lora":
                forward_tokens += int(batch_t["teacher"]["attention_mask"].sum())
            record = {
                "step": step + 1,
                "loss": float(loss),
                "grad_norm": float(norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "exposure": exposure,
                "answer_tokens": answer_tokens,
                "forward_tokens": forward_tokens,
                "seconds_this_process": time.time() - began,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            if (step + 1) % config["log_steps"] == 0 or step + 1 == total:
                write(out / "progress.json", record)
                with (out / "curve.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if run:
                    run.log(record, step=step + 1)
            if (step + 1) % config["checkpoint_steps"] == 0 or step + 1 == total:
                atomic_torch_save(
                    {
                        "weights": state(model),
                        "optimizer": optimizer.state_dict(),
                        "rng": rng_state(),
                        "step": step + 1,
                        "exposure": exposure,
                        "answer_tokens": answer_tokens,
                        "forward_tokens": forward_tokens,
                        "config": config,
                        "data_sha256": manifest["data_sha256"],
                    },
                    checkpoint,
                )
        atomic_torch_save(state(model), out / "weights.pt")
        model.eval()
        audit_inputs = collate(samples[:2], tok.pad_token_id)["student"]
        with torch.no_grad(), autocast():
            logits = model(
                input_ids=audit_inputs["input_ids"],
                attention_mask=audit_inputs["attention_mask"],
                use_cache=False,
            ).logits
        # Deterministic vocabulary slice and greedy IDs at every padded input position.
        atomic_torch_save(
            {
                "input_ids": audit_inputs["input_ids"].cpu(),
                "attention_mask": audit_inputs["attention_mask"].cpu(),
                "greedy": logits.argmax(-1).cpu(),
                "logits": logits[:, -1, :1024].float().cpu(),
            },
            out / "reload-reference.pt",
        )
        write(
            out / "complete.json",
            {
                **record,
                "weights_sha256": digest(out / "weights.pt"),
                "smoke": smoke_steps is not None,
            },
        )
    finally:
        if run:
            run.finish()


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(prediction, gold):
    p, g = normalize_answer(prediction), normalize_answer(gold)
    exact = p == g
    common = sum((Counter(p.split()) & Counter(g.split())).values())
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        f1 = float(exact)
    elif not p or not g:
        f1 = float(exact)
    else:
        f1 = 2 * common / (len(p.split()) + len(g.split()))
    return {"em": float(exact), "f1": f1}


def evaluation_rows(config, batch, tok):
    from .knowledge_method_data import chat_user, context_user, render_chat

    cases = json.loads((batch / "shared/eval.json").read_text())
    if isinstance(cases, dict):
        cases = cases["cases"]
    rows = []
    for case in cases:
        for mode in ("closed", "context"):
            user = (
                chat_user(case["question"])
                if mode == "closed"
                else context_user(case["context"], case["question"])
            )
            # Natural, short answer; no support-fact or intermediate-answer disclosure.
            user += "\nAnswer with only the answer, without explanation."
            rows.append(
                {
                    "id": f"{case['case_id']}/{mode}",
                    "group": mode,
                    "prompt": render_chat(tok, user),
                    "gold": case["answer"],
                }
            )
    retention = json.loads(Path(config["retention_source"]).read_text())["natural"]
    seen = set()
    for row in retention:
        if row["qid"] in seen:
            continue
        seen.add(row["qid"])
        rows.append(
            {
                "id": row["qid"],
                "group": "retention",
                "gold": row["gold"],
                "prompt": render_chat(tok, row["prompt"]),
                "type": row["type"],
                "exclude": row.get("exclude", []),
            }
        )
    return rows


def predict(model, tok, rows, config):
    from .read_interface_data import graders

    result = []
    old_graders = graders()
    model.eval()
    for offset in range(0, len(rows), config["eval_batch_size"]):
        chunk = rows[offset : offset + config["eval_batch_size"]]
        inputs = tok(
            [r["prompt"] for r in chunk],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")
        if (
            inputs.input_ids.shape[1] + config["eval_max_new_tokens"]
            > model.config.max_position_embeddings
        ):
            raise ValueError("Evaluation context exceeds model limit")
        seed_all(config["seed"] + 900000 + offset)
        retention_only = all(r["group"] == "retention" for r in chunk)
        maximum = (
            config["retention_max_new_tokens"] if retention_only else config["eval_max_new_tokens"]
        )
        with torch.no_grad(), autocast():
            output = model.generate(
                **inputs,
                do_sample=not retention_only,
                temperature=config["eval_temperature"] if not retention_only else 1.0,
                top_p=1.0,
                top_k=50,
                max_new_tokens=maximum,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        tails = output[:, inputs.input_ids.shape[1] :]
        for row, ids in zip(chunk, tails, strict=True):
            answer = tok.decode(ids, skip_special_tokens=True).strip()
            score = score_answer(answer, row["gold"])
            if row["group"] == "retention":
                score["strict"] = old_graders[row["type"]].grade(
                    answer, row["gold"], set(row["exclude"])
                )["strict"]
            result.append({**row, "answer": answer, "token_ids": ids.tolist(), **score})
    return result


def summarize(before, after):
    result = {}
    for group in ("closed", "context", "retention"):
        b = [r for r in before if r["group"] == group]
        a = [r for r in after if r["group"] == group]
        if [r["id"] for r in b] != [r["id"] for r in a] or not b:
            raise ValueError("Evaluation coverage mismatch")
        metric = "strict" if group == "retention" else "em"
        correct = [i for i, r in enumerate(b) if r[metric]]
        incorrect = [i for i, r in enumerate(b) if not r[metric]]
        result[group] = {
            "n": len(b),
            "before": sum(r[metric] for r in b) / len(b),
            "after": sum(r[metric] for r in a) / len(a),
            "before_f1": sum(r["f1"] for r in b) / len(b),
            "after_f1": sum(r["f1"] for r in a) / len(a),
            "initially_correct_n": len(correct),
            "retained_n": sum(bool(a[i][metric]) for i in correct),
            "initially_incorrect_n": len(incorrect),
            "newly_correct_n": sum(bool(a[i][metric]) for i in incorrect),
        }
    return result


def fit_diagnostic(model, tok, config, batch, arm):
    """Fit to fixed teacher targets; not factual accuracy or free-generation recall."""
    source = batch / "shared/train.jsonl"
    selected = {}
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        row = json.loads(line)
        if row["case_id"] not in selected:
            selected[row["case_id"]] = (line_number, row)
    if len(selected) != config["n_cases"]:
        raise ValueError("Teacher-target-fit case coverage differs from the fixed sample")
    chosen = list(selected.values())
    samples = [encode_sample(tok, row, arm, config["max_sequence_tokens"]) for _, row in chosen]
    target_key = "pd_answer" if arm == "pd_lora" else "sft_answer"
    device = next(model.parameters()).device
    records = []
    model.eval()
    for offset in range(0, len(samples), config["eval_batch_size"]):
        chunk = samples[offset : offset + config["eval_batch_size"]]
        inputs = collate(chunk, tok.pad_token_id, device=device)["student"]
        with torch.no_grad(), autocast():
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            ).logits
        for index, sample in enumerate(chunk):
            mask = inputs["answer_mask"][index]
            labels = inputs["labels"][index, mask]
            target_logits = logits[index, mask].float()
            token_nll = F.cross_entropy(target_logits, labels, reduction="none")
            if not bool(torch.isfinite(token_nll).all()):
                raise FloatingPointError("Non-finite teacher-target-fit NLL")
            correct = target_logits.argmax(-1).eq(labels)
            eos = labels.eq(tok.eos_token_id)
            line_number, row = chosen[offset + index]
            records.append(
                {
                    "id": sample["id"],
                    "case_id": row["case_id"],
                    "train_line": line_number,
                    "target_key": target_key,
                    "target_token_ids": labels.tolist(),
                    "valid_target_tokens": len(labels),
                    "eos_tokens": int(eos.sum()),
                    "non_eos_tokens": int((~eos).sum()),
                    "nll": float(token_nll.mean()),
                    "nll_sum": float(token_nll.sum()),
                    "next_token_greedy_correct": int(correct.sum()),
                    "next_token_greedy_fraction": float(correct.float().mean()),
                }
            )
        del logits, target_logits, token_nll
    tokens = sum(row["valid_target_tokens"] for row in records)
    return {
        "label": "teacher-target-fit",
        "interpretation": (
            "Closed-book teacher-forced fit to this arm's generated training answers; "
            "not factual accuracy, free-generation recall, or proof of knowledge mastery."
        ),
        "selection": "First train.jsonl row per case, in file order; no filtering or resampling",
        "used_for_selection_or_stopping": False,
        "train_data_sha256": digest(source),
        "target_key": target_key,
        "rows": records,
        "summary": {
            "n": len(records),
            "valid_target_tokens": tokens,
            "eos_tokens": sum(row["eos_tokens"] for row in records),
            "non_eos_tokens": sum(row["non_eos_tokens"] for row in records),
            "mean_example_nll": sum(row["nll"] for row in records) / len(records),
            "token_mean_nll": sum(row["nll_sum"] for row in records) / tokens,
            "mean_example_next_token_greedy_fraction": sum(
                row["next_token_greedy_fraction"] for row in records
            )
            / len(records),
            "token_next_token_greedy_fraction": sum(
                row["next_token_greedy_correct"] for row in records
            )
            / tokens,
        },
    }


def evaluate(config, batch, arm):
    out = batch / "runs" / arm
    seed_all(config["seed"])
    model, tok = load_model(config)
    rows = evaluation_rows(config, batch, tok)
    # Separate predictions are retained; each arm starts from the same base and sampling seeds.
    before = predict(model, tok, rows, config)
    before_fit = fit_diagnostic(model, tok, config, batch, arm)
    model = configure_arm(model, config, arm)
    restore(model, torch.load(out / "weights.pt", map_location="cpu", weights_only=True))
    after = predict(model, tok, rows, config)
    after_fit = fit_diagnostic(model, tok, config, batch, arm)
    if before_fit["train_data_sha256"] != after_fit["train_data_sha256"]:
        raise ValueError("Teacher-target-fit data changed between before and after")
    summary = summarize(before, after)
    write(
        out / "eval/evaluate.json",
        {
            "before": before,
            "after": after,
            "summary": summary,
            "teacher_target_fit": {"before": before_fit, "after": after_fit},
            "scoring": "Hotpot EM/F1; inherited retention aliases",
        },
    )
    run = wandb_run(config, batch, "evaluate", arm)
    run.summary.update(summary)
    run.summary["teacher_target_fit"] = {
        "before": before_fit["summary"],
        "after": after_fit["summary"],
    }
    run.finish()


def audit(config, batch, arm):
    out = batch / "runs" / arm
    model, _ = load_model(config)
    model = configure_arm(model, config, arm)
    weights = torch.load(out / "weights.pt", map_location="cpu", weights_only=True)
    restore(model, weights)
    model.eval()
    reference = torch.load(out / "reload-reference.pt", map_location="cpu", weights_only=True)
    with torch.no_grad(), autocast():
        logits = model(
            input_ids=reference["input_ids"].to("cuda"),
            attention_mask=reference["attention_mask"].to("cuda"),
            use_cache=False,
        ).logits
    maximum = (logits[:, -1, :1024].float().cpu() - reference["logits"]).abs().max().item()
    greedy_equal = torch.equal(logits.argmax(-1).cpu(), reference["greedy"])
    complete = json.loads((out / "complete.json").read_text())
    hash_equal = digest(out / "weights.pt") == complete["weights_sha256"]
    passed = maximum == 0 and greedy_equal and hash_equal
    write(
        out / "audit/audit.json",
        {
            "passed": passed,
            "max_abs_logit_difference": maximum,
            "all_token_greedy_equal": greedy_equal,
            "weights_hash_equal": hash_equal,
            "scope": "fresh process reload; two fixed training sequences",
        },
    )
    if not passed:
        raise RuntimeError("Independent reload audit failed")
