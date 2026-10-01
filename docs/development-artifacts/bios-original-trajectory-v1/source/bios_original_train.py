"""Restartable development execution for the original bioS experiment, plan 14.27."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_original_data import (
    ATTRS,
    MATERIAL,
    MONTHS,
    TASKS,
    BioStream,
    answer,
    date,
    digest,
    load_json,
    question,
    reasoning,
    write_json,
)
from .bios_original_model import GPT


def setup(seed, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(device)


def amp(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def append(path, record):
    with Path(path).open("a") as out:
        out.write(json.dumps(record) + "\n")


def atomic_save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def pack_tokens(tokens, context=512):
    """Adjacent chunks overlap by the context token only; every target is scored once."""
    n = len(tokens) - 1
    rows = math.ceil(n / context)
    x = np.full((rows, context), 50256, dtype=np.int64)
    y = np.full((rows, context), -100, dtype=np.int64)
    x.flat[:n] = tokens[:-1]
    y.flat[:n] = tokens[1:]
    return torch.from_numpy(x), torch.from_numpy(y)


def optimizer_for(model, settings):
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        dict(params=[p for p in params if p.ndim >= 2], weight_decay=settings["weight_decay"]),
        dict(params=[p for p in params if p.ndim < 2], weight_decay=0.0),
    ]
    return torch.optim.AdamW(
        groups,
        lr=settings["learning_rate"],
        betas=tuple(settings.get("adam_betas", [0.9, 0.999])),
        eps=settings.get("adam_epsilon", 1e-6),
        fused=next(model.parameters()).is_cuda,
    )


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def pretrain(cfg, world, init, condition, device, stop_after_pass=None):
    setup(init, device)
    root = Path(cfg["result_root"]) / f"world-{world}" / f"init-{init}" / condition / "pretrain"
    root.mkdir(parents=True, exist_ok=True)
    settings = cfg["pretrain"]
    data_root = Path(cfg["data_root"]) / f"world-{world}"
    stream = BioStream(data_root, world, condition)
    contract = dict(
        config=cfg,
        world=world,
        init=init,
        condition=condition,
        data_audit_sha256=digest(data_root / "audit.json"),
    )
    if (root / "contract.json").exists():
        if load_json(root / "contract.json") != contract:
            raise ValueError("Run contract changed; use a new run directory")
    else:
        write_json(root / "contract.json", contract)
    model = GPT(cfg["model"]).to(device)
    opt = optimizer_for(model, settings)
    start, step, seen, elapsed = 0, 0, 0, 0.0
    latest = root / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start, step, seen, elapsed = [state[k] for k in ("passes", "step", "tokens", "elapsed")]
        del state
    elif (root / "metrics.jsonl").exists():
        raise ValueError("Metrics without a resumable checkpoint require investigation")
    if (root / "metrics.jsonl").exists():
        kept = [
            line
            for line in (root / "metrics.jsonl").read_text().splitlines()
            if json.loads(line)["passes"] <= start
        ]
        (root / "metrics.jsonl").write_text("".join(s + "\n" for s in kept))
    if start == 0 and not (root / "pass-000.pt").exists():
        atomic_save(root / "pass-000.pt", dict(model=model.state_dict(), config=cfg["model"]))
    if not latest.exists():
        atomic_save(
            latest,
            dict(
                model=model.state_dict(),
                optimizer=opt.state_dict(),
                passes=0,
                step=0,
                tokens=0,
                elapsed=0.0,
            ),
        )
    count = parameter_count(model)
    write_json(
        root / "model.json",
        dict(
            parameters=count,
            trainable=count,
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if "cuda" in str(device) else "cpu",
            torch=torch.__version__,
            resume_pass=start,
        ),
    )
    model.train()
    runner = torch.compile(model) if settings.get("compile") else model
    wall = time.monotonic()
    for epoch in range(start, min(settings["passes"], stop_after_pass or settings["passes"])):
        pass_start = time.monotonic()
        tokens, order_hash = stream.pass_tokens(epoch)
        xs, ys = pack_tokens(tokens, cfg["model"]["context"])
        total_loss, valid_tokens = 0.0, 0
        batch = settings["batch_size"]
        micro = settings["micro_batch_size"]
        for i in range(0, len(xs), batch):
            # Learning rate follows complete-person exposure; no outcome-based stopping.
            progress = (epoch + i / len(xs)) / settings["passes"]
            cosine = (
                settings["minimum_lr_ratio"]
                + (1 - settings["minimum_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
            )
            lr = (
                settings["learning_rate"]
                * cosine
                * min(1, (step + 1) / max(1, settings["warmup_steps"]))
            )
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad(set_to_none=True)
            end = min(i + batch, len(xs))
            denom = int((ys[i:end] != -100).sum())
            batch_loss = 0.0
            for j in range(i, end, micro):
                x, y = (
                    xs[j : min(j + micro, end)].to(device),
                    ys[j : min(j + micro, end)].to(device),
                )
                n = int((y != -100).sum())
                with amp(device):
                    loss = runner(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at pass {epoch}, step {step}")
                (loss * (n / denom)).backward()
                batch_loss += loss.detach().item() * n
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite gradient at step {step}")
            opt.step()
            step += 1
            total_loss += batch_loss
            valid_tokens += denom
            if step % 10 == 0:
                write_json(
                    root / "status.json",
                    dict(
                        state="running",
                        passes_completed=epoch,
                        pass_fraction=end / len(xs),
                        step=step,
                        learning_rate=lr,
                        last_batch_nll=batch_loss / denom,
                        tokens=seen + valid_tokens,
                        elapsed=elapsed + time.monotonic() - wall,
                        updated_unix=time.time(),
                        pid=os.getpid(),
                    ),
                )
        assert valid_tokens == len(tokens) - 1
        seen += valid_tokens
        duration = time.monotonic() - pass_start
        record = dict(
            passes=epoch + 1,
            people_presented=(epoch + 1) * stream.count,
            tokens=seen,
            pass_tokens=valid_tokens,
            nll=total_loss / valid_tokens,
            seconds=duration,
            tokens_per_second=valid_tokens / duration,
            step=step,
            learning_rate=lr,
            presentation_sha256=order_hash,
            estimated_training_flops=6 * count * seen,
            elapsed=elapsed + time.monotonic() - wall,
        )
        append(root / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
        if epoch + 1 in settings["nodes"]:
            atomic_save(
                root / f"pass-{epoch + 1:03d}.pt",
                dict(model=model.state_dict(), config=cfg["model"], metrics=record),
            )
        if (epoch + 1) % 10 == 0 or epoch + 1 == stop_after_pass or epoch + 1 == settings["passes"]:
            atomic_save(
                latest,
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    passes=epoch + 1,
                    step=step,
                    tokens=seen,
                    elapsed=elapsed + time.monotonic() - wall,
                ),
            )
        write_json(
            root / "status.json",
            dict(
                state="complete" if epoch + 1 == settings["passes"] else "running",
                passes_completed=epoch + 1,
                **record,
                updated_unix=time.time(),
                pid=os.getpid(),
            ),
        )
    return root / f"pass-{settings['passes']:03d}.pt"


def encode_row(tokenizer, prompt, target):
    prefix = tokenizer.encode(prompt).ids
    suffix = tokenizer.encode(" " + target).ids + [50256]
    return dict(ids=prefix + suffix, answer_start=len(prefix))


def supervised_batch(rows, device):
    width = max(len(r["ids"]) for r in rows) - 1
    x = torch.full((len(rows), width), 50256, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, r in enumerate(rows):
        ids = torch.tensor(r["ids"], device=device)
        x[i, : len(ids) - 1] = ids[:-1]
        y[i, r["answer_start"] - 1 : len(ids) - 1] = ids[r["answer_start"] :]
    return x, y


def sequence_scores(model, rows, device):
    x, y = supervised_batch(rows, device)
    h = model.hidden(x)
    b, t = torch.where(y != -100)
    losses = F.cross_entropy(model.logits(h[b, t]).float(), y[b, t], reduction="none")
    sums = torch.zeros(len(rows), device=device).scatter_add_(0, b, losses)
    sizes = torch.bincount(b, minlength=len(rows))
    return -sums / sizes, -sums


def task_rows(people, tokenizer, tasks, condition, epoch, seed):
    rng = random.Random(seed)
    # Use the published label vocabulary, independent of held-out person attributes.
    cities = sorted(
        set((MATERIAL / "fields/city.txt").read_text().splitlines())
        | {
            line.split(";", 1)[1].strip()
            for line in (MATERIAL / "fields/company.txt").read_text().splitlines()
        }
    )
    bins = defaultdict(list)
    for city in cities:
        bins[len(tokenizer.encode(" " + city).ids)].append(city)
    rows = []
    for p in people:
        if p["split"] != "train":
            continue
        other = people[p["partner"]]
        for j, task in enumerate(tasks):
            cot = condition == "MP-CoT" and (p["id"] + j + epoch) % 2 == 0
            prompt = question(p, task, other=other, cot=cot)
            target = reasoning(p, task, other) if cot else answer(p, task, other)
            row = encode_row(tokenizer, prompt, target)
            row.update(person_id=p["id"], task=task, cot=cot)
            if task in ("birthcity", "workcity") and p["birthcity"] != p["workcity"]:
                competing = p["workcity"] if task == "birthcity" else p["birthcity"]
                candidates = [
                    c
                    for c in bins[len(tokenizer.encode(" " + competing).ids)]
                    if c not in (p["birthcity"], p["workcity"])
                ]
                if not candidates:
                    raise ValueError(
                        "No length-matched random city; cannot silently relax matching"
                    )
                negative = rng.choice(candidates) if condition == "MP-Random" else competing
                row["negative"] = encode_row(tokenizer, prompt, negative)
            rows.append(row)
    return rows


def rule_rows(tokenizer):
    rows = []
    for m, name in enumerate(MONTHS, 1):
        rows.append(
            encode_row(
                tokenizer,
                f"Question: Is {name} an even-numbered month?\nAnswer:",
                "yes" if m % 2 == 0 else "no",
            )
        )
    rng = random.Random(14270000)
    for i in range(24):
        a, b = [
            dict(
                year=rng.randrange(1900, 2100), month=rng.randrange(1, 13), day=rng.randrange(1, 29)
            )
            for _ in range(2)
        ]
        if i % 2:
            a, b = (
                max((a, b), key=lambda p: (p["year"], p["month"], p["day"])),
                min((a, b), key=lambda p: (p["year"], p["month"], p["day"])),
            )
        else:
            a, b = (
                min((a, b), key=lambda p: (p["year"], p["month"], p["day"])),
                max((a, b), key=lambda p: (p["year"], p["month"], p["day"])),
            )
        rows.append(
            encode_row(
                tokenizer,
                f"Question: Which date is earlier, {date(a)} or {date(b)}?\nAnswer:",
                date(b if i % 2 else a),
            )
        )
    return rows


def load_model(path, cfg, device, adapters=False):
    model = GPT(cfg["model"])
    if adapters:
        model.add_lora(cfg["adapt"]["qv_rank"], cfg["adapt"]["embedding_rank"])
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device)


def finetune(cfg, world, init, condition, stage, parent, device, tokenizer):
    setup(init + 1, device)
    root = Path(cfg["result_root"]) / f"world-{world}" / f"init-{init}" / condition / stage
    root.mkdir(parents=True, exist_ok=True)
    people = load_json(Path(cfg["data_root"]) / f"world-{world}" / "people.json")
    contract = dict(
        config=cfg,
        world=world,
        init=init,
        condition=condition,
        stage=stage,
        parent=str(parent),
        parent_sha256=digest(parent),
    )
    if (root / "contract.json").exists() and load_json(root / "contract.json") != contract:
        raise ValueError("Finetuning contract changed")
    write_json(root / "contract.json", contract)
    if (root / "complete.json").exists():
        return root / "final.pt"
    model = load_model(parent, cfg, device, adapters=stage == "task")
    if stage == "adapt":
        model.add_lora(cfg["adapt"]["qv_rank"], cfg["adapt"]["embedding_rank"])
        model.to(device)
    opt = optimizer_for(model, cfg["adapt"])
    start, step = 0, 0
    if (root / "latest.pt").exists():
        ck = torch.load(root / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start, step = ck["epoch"], ck["step"]
        del ck
    if (root / "metrics.jsonl").exists():
        kept = [
            line
            for line in (root / "metrics.jsonl").read_text().splitlines()
            if json.loads(line)["epoch"] <= start
        ]
        (root / "metrics.jsonl").write_text("".join(s + "\n" for s in kept))
    write_json(
        root / "parameters.json",
        dict(
            total=parameter_count(model),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_names=[n for n, p in model.named_parameters() if p.requires_grad],
        ),
    )
    epochs = cfg[stage]["epochs"]
    model.train()
    for epoch in range(start, epochs):
        rows = task_rows(
            people,
            tokenizer,
            ATTRS if stage == "adapt" else TASKS,
            condition if stage == "task" else "MP",
            epoch,
            world + 1427,
        )
        if stage == "task":
            rows += rule_rows(tokenizer) * cfg["task"]["rule_repetitions_per_epoch"]
        random.Random(world + epoch + 20000000).shuffle(rows)
        batch, micro = cfg["adapt"]["batch_size"], cfg["adapt"]["micro_batch_size"]
        ce_sum, ranking_sum, candidates, target_tokens = 0.0, 0.0, 0, 0
        wall = time.monotonic()
        for i in range(0, len(rows), batch):
            group = rows[i : i + batch]
            lr = cfg["adapt"]["learning_rate"] * (
                0.1 + 0.9 * (1 + math.cos(math.pi * (epoch + i / len(rows)) / epochs)) / 2
            )
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(group), micro):
                part = group[j : j + micro]
                with amp(device):
                    scores, _ = sequence_scores(model, part, device)
                    ce = -scores.mean()
                    neg_pairs = [(k, r["negative"]) for k, r in enumerate(part) if "negative" in r]
                    rank = scores.sum() * 0
                    if stage == "task" and neg_pairs:
                        neg, _ = sequence_scores(model, [r for _, r in neg_pairs], device)
                        pos = scores[[k for k, _ in neg_pairs]]
                        # Normalize by all positive rows, including zero collision/non-city terms.
                        rank = F.softplus(cfg["task"]["margin"] - (pos - neg)).sum() / len(part)
                        candidates += len(neg_pairs)
                    coefficient = (
                        cfg["task"]["ranking_coefficient"]
                        if condition in ("MP-R", "MP-Random") and stage == "task"
                        else 0.0
                    )
                    loss = ce + coefficient * rank
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite finetuning loss")
                (loss * len(part) / len(group)).backward()
                ce_sum += ce.item() * len(part)
                ranking_sum += rank.item() * len(part)
                target_tokens += sum(len(r["ids"]) - r["answer_start"] for r in part)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite finetuning gradient")
            opt.step()
            step += 1
            if step % 25 == 0:
                write_json(
                    root / "status.json",
                    dict(
                        state="running",
                        epoch=epoch,
                        fraction=min(i + batch, len(rows)) / len(rows),
                        step=step,
                        last_ce=ce.item(),
                        pid=os.getpid(),
                        updated_unix=time.time(),
                    ),
                )
        rec = dict(
            epoch=epoch + 1,
            rows=len(rows),
            ce=ce_sum / len(rows),
            rank=ranking_sum / len(rows),
            scored_negative_candidates=candidates,
            target_tokens=target_tokens,
            seconds=time.monotonic() - wall,
        )
        append(root / "metrics.jsonl", rec)
        print(json.dumps(rec), flush=True)
        atomic_save(
            root / "latest.pt",
            dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch + 1, step=step),
        )
    atomic_save(root / "final.pt", dict(model=model.state_dict(), config=cfg["model"]))
    write_json(
        root / "complete.json",
        dict(epochs=epochs, step=step, checkpoint_sha256=digest(root / "final.pt")),
    )
    return root / "final.pt"


@torch.no_grad()
def generate(model, tokenizer, prompts, device, max_tokens):
    # Right padding; each row's final real position supplies its own next token.
    sequences = [tokenizer.encode(p).ids for p in prompts]
    outputs, finished = [[] for _ in prompts], [False] * len(prompts)
    for _ in range(max_tokens):
        width = max(map(len, sequences))
        if width > model.cfg["context"]:
            break
        ids = torch.full((len(prompts), width), 50256, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, device=device)
        with amp(device):
            h = model.hidden(ids)
            last = h[
                torch.arange(len(prompts), device=device),
                torch.tensor([len(s) - 1 for s in sequences], device=device),
            ]
            pred = model.logits(last).argmax(-1).tolist()
        for i, token in enumerate(pred):
            if not finished[i]:
                outputs[i].append(token)
                sequences[i].append(token)
                finished[i] = token == 50256
        if all(finished):
            break
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in outputs], finished


def normalize(text):
    return " ".join(text.strip().rstrip(".").casefold().split())


def recognized_format(text, task, vocabulary):
    value = normalize(text)
    if task == "parity":
        return value in ("yes", "no")
    if task == "year":
        return bool(re.fullmatch(r"\d{4}", value))
    if task == "date":
        return bool(
            re.fullmatch(
                "(?:" + "|".join(m.lower() for m in MONTHS) + r") (?:[1-9]|1\d|2[0-8]), \d{4}",
                value,
            )
        )
    if task in ("company_city", "city_company"):
        parts = value.split(";")
        fields = ("company", "workcity") if task == "company_city" else ("workcity", "company")
        return len(parts) == 2 and all(
            part.strip() in vocabulary[field] for part, field in zip(parts, fields, strict=True)
        )
    return value in vocabulary["name" if task == "comparison" else task]


@torch.no_grad()
def evaluate(cfg, world, checkpoint, output, device, tokenizer, tasks=TASKS, cot=False):
    output = Path(output)
    if (output / "evaluation-complete.json").exists():
        if load_json(output / "evaluation-complete.json")["checkpoint_sha256"] != digest(
            checkpoint
        ):
            raise ValueError("Evaluated checkpoint changed")
        return
    output.mkdir(parents=True, exist_ok=True)
    people = load_json(Path(cfg["data_root"]) / f"world-{world}" / "people.json")
    vocabulary = {field: {normalize(p[field]) for p in people} for field in ("name", *ATTRS[1:])}
    model = load_model(checkpoint, cfg, device, adapters=True).eval()
    totals = defaultdict(lambda: defaultdict(float))
    batch = []
    with (output / "predictions.jsonl").open("w") as out:

        def flush():
            if not batch:
                return
            enc = [encode_row(tokenizer, r["prompt"], r["answer"]) for r in batch]
            with amp(device):
                avg, full = sequence_scores(model, enc, device)
            generated, ended = generate(
                model,
                tokenizer,
                [r["prompt"] for r in batch],
                device,
                cfg["evaluation"]["max_new_tokens"],
            )
            for row, text, stop, av, lp in zip(
                batch, generated, ended, avg.tolist(), full.tolist(), strict=True
            ):
                extracted = text.rsplit("Answer:", 1)[-1].strip() if row["cot"] else text
                correct = normalize(extracted) == normalize(row["answer"])
                # Syntactic recognition is distinct from correctness; retain raw output.
                recognized = recognized_format(extracted, row["task"], vocabulary)
                row.update(
                    generated=text,
                    eos=stop,
                    correct=correct,
                    format_recognized=recognized,
                    answer_mean_logp=av,
                    answer_full_logp=lp,
                )
                out.write(json.dumps(row) + "\n")
                key = f"{row['split']}/{row['task']}/view-{row['view']}/cot-{row['cot']}"
                t = totals[key]
                t["n"] += 1
                t["correct"] += correct
                t["format_recognized"] += recognized
                t["sum_full_logp"] += lp
            batch.clear()
            write_json(
                output / "evaluation-status.json",
                dict(
                    rows=sum(t["n"] for t in totals.values()),
                    updated_unix=time.time(),
                    pid=os.getpid(),
                ),
            )

        for p in people:
            if p["split"] not in cfg["evaluation"]["splits"]:
                continue
            other = people[p["partner"]]
            for task in tasks:
                for view in cfg["evaluation"]["views"]:
                    for use_cot in [False, True] if cot else [False]:
                        batch.append(
                            dict(
                                person_id=p["id"],
                                split=p["split"],
                                task=task,
                                view=view,
                                cot=use_cot,
                                partner=other["id"] if task == "comparison" else None,
                                prompt=question(p, task, view, other, use_cot),
                                answer=answer(p, task, other),
                            )
                        )
                        if len(batch) == cfg["evaluation"]["batch_size"]:
                            flush()
        flush()
    write_json(
        output / "evaluation-complete.json",
        dict(checkpoint_sha256=digest(checkpoint), totals=dict(totals)),
    )
