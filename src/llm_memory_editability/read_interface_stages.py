"""Stages of knowledge-read-interface-development-v1: screen, evaluate, calibrate, inject,
transplant. Each stage reads/writes JSON under one run directory per model."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_model as M

SEED = 20261019
BATCH = 32
COT_TOKENS = 160


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def batches(items, size=BATCH):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def facts_from(run):
    return [D.Fact(**f) for f in read(Path(run) / "facts.json")["facts"]]


# ---------------------------------------------------------------- generation and grading


def answer_queries(model, tok, rows, system_for=None, cot_tokens=COT_TOKENS):
    """Generate for rows with keys prompt/gold/type/mode; adds answer, strict, lenient."""
    graders = D.graders()
    out = []
    for mode in ("direct", "cot"):
        subset = [r for r in rows if r.get("mode", "direct") == mode]
        stepwise = mode == "cot"
        for chunk in batches(subset, BATCH if not stepwise else 64):
            texts = [
                M.chat(tok, r["prompt"], False, system_for(r) if system_for else None)
                for r in chunk
            ]
            ids, mask, _, _ = M.encode(tok, texts)
            raw = M.generate(model, tok, ids, mask, cot_tokens if stepwise else 24)
            for r, text in zip(chunk, raw, strict=True):
                answer = M.final_answer(text, stepwise)
                exclude = {e for e in r.get("exclude", ())}
                grade = graders[r["type"]].grade(answer, r["gold"], exclude)
                record = {**r, "answer": answer, **grade}
                if stepwise:
                    record["reply"] = text
                    record["final_marker"] = bool(answer)
                out.append(record)
    return out


def teacher_forced(model, tok, rows, interventions=None):
    """Greedy match / log-prob of each row's gold string after its direct-mode prompt."""
    out = []
    for chunk in batches(rows):
        texts = [M.chat(tok, r["prompt"], False) for r in chunk]
        ids, mask, _, slices = M.encode(tok, texts, answers=[r["gold"] for r in chunk])
        out += M.answer_scores(model, ids, mask, slices)
    return out


def exclusions(row, fact=None):
    names = set()
    if fact is not None:
        names |= {fact.h, fact.country_h}
    if row.get("subject"):
        names.add(row["subject"])
    return sorted(names)


# ---------------------------------------------------------------- stage: screen


def screen(model_name, run, model=None, tok=None):
    run = Path(run)
    if model is None:
        model, tok = M.load(model_name)
    queries = [
        {**q, "prompt": q["prompt"] + D.DIRECT, "mode": "direct"} for q in D.screening_queries()
    ]
    answered = answer_queries(model, tok, queries)
    known = {r["qid"]: r["strict"] for r in answered}
    write(run / "screening.json", {"model": model_name, "records": answered})
    facts, subjects, values = D.build_facts(known, seed=SEED)
    write(
        run / "facts.json",
        {
            "seed": SEED,
            "subjects": subjects,
            "values": values,
            "facts": [asdict(f) for f in facts],
            "known": known,
        },
    )
    return model, tok


# ---------------------------------------------------------------- stage: evaluate


def fact_rows(facts):
    rows = []
    for fact in facts:
        for row in D.use_queries(fact):
            rows.append({**row, "exclude": exclusions(row, fact), "split": fact.split})
    return rows


def evaluate(model_name, run, tag, model, tok, context=False):
    """All uses of all facts (+ retention, natural two-hop, other-fact recall)."""
    run = Path(run)
    info = read(run / "facts.json")
    facts = facts_from(run)
    by_id = {f.fid: f for f in facts}
    rows = fact_rows(facts)
    system = (lambda r: "Fact: " + D.context_statement(by_id[r["fid"]])) if context else None
    t0 = time.time()
    answered = answer_queries(model, tok, rows, system)
    direct = [r for r in rows if r["mode"] == "direct"]
    for r, s in zip(direct, teacher_forced(model, tok, direct), strict=True):
        r.update({"tf_greedy": s["greedy"], "tf_logprob": s["logprob"]})
    tf = {(r["fid"], r["use"]): r for r in direct}
    for r in answered:
        if r["mode"] == "direct":
            r.update({k: tf[(r["fid"], r["use"])][k] for k in ("tf_greedy", "tf_logprob")})
    extra = []
    if not context:
        retention = [
            {**q, "mode": "direct"}
            for q in D.retention_queries(info["subjects"], info["values"], facts)
        ]
        natural = [
            {**q, "mode": "direct", "exclude": [q["subject"]]}
            for q in D.natural_two_hop(info["subjects"])
        ]
        extra = answer_queries(model, tok, retention + natural)
    write(
        run / "eval" / f"{tag}.json",
        {
            "tag": tag,
            "context": context,
            "seconds": time.time() - t0,
            "uses": answered,
            "natural": extra,
        },
    )


# ---------------------------------------------------------------- stage 0: calibration


def subject_spans(tok, rows, thinking=False):
    texts, spans = [], []
    for r in rows:
        text = M.chat(tok, r["prompt"], thinking)
        user_at = text.rfind(r["prompt"])
        spans.append([M.find_span(text, r["subject"], user_at)])
        texts.append(text)
    return texts, spans


def swap_pairs(subjects, rng):
    cities = D.city_table()
    pairs = []
    for h in subjects:
        country = cities[h]["country"]
        cur = D.currency_of(country)
        options = [
            g
            for g in subjects
            if g != h
            and cities[g]["country"] != country
            and D.currency_of(cities[g]["country"])
            and (
                not cur
                or not (
                    D.currency_aliases(cur)
                    & D.currency_aliases(D.currency_of(cities[g]["country"]))
                )
            )
        ]
        if cur and options:
            pairs.append((h, rng.choice(options)))
    return pairs


def calibration_rows(h):
    cities = D.city_table()
    country = cities[h]["country"]
    rows = [
        {
            "use": f"country{i}",
            "prompt": t.format(h=h) + D.DIRECT,
            "gold": country,
            "type": "country",
            "subject": h,
        }
        for i, t in enumerate(D.CAPITAL_REV)
    ]
    rows += [
        {
            "use": f"currency{i}",
            "prompt": t.format(h=h) + D.DIRECT,
            "gold": D.currency_of(country),
            "type": "currency",
            "subject": h,
        }
        for i, t in enumerate(D.CURRENCY_TWO_HOP)
        if D.currency_of(country)
    ]
    return rows


@torch.no_grad()
def calibrate(model_name, run, model, tok):
    """0c entity-swap interface and 0d token-mixer knockout on natural facts."""
    run = Path(run)
    info = read(run / "facts.json")
    rng = random.Random(SEED + 1)
    n_layers = len(M.layers(model))
    kinds = M.layer_kinds(model)
    if (run / "calibration" / "swap.json").exists():
        knockout(model_name, run, model, tok, info, kinds)
        return
    pairs = swap_pairs(info["subjects"], rng)
    swap = []
    for h, g in pairs:
        target, source = calibration_rows(h), calibration_rows(g)
        for t, s in zip(target, source, strict=True):
            swap.append({"h": h, "g": g, "use": t["use"], "target": t, "source": s})
    records = []
    for chunk in batches(swap, 16):
        t_texts, t_spans = subject_spans(tok, [c["target"] for c in chunk])
        s_texts, s_spans = subject_spans(tok, [c["source"] for c in chunk])
        store = {}
        s_ids, s_mask, s_pos, _ = M.encode(tok, s_texts, s_spans)
        with M.Interventions(model) as hooks, M.autocast():
            hooks.capture(range(n_layers), store)
            model(input_ids=s_ids.cuda(), attention_mask=s_mask.cuda(), use_cache=False)
        golds_new = [c["source"]["gold"] for c in chunk]
        golds_old = [c["target"]["gold"] for c in chunk]
        for gold_key, golds in (("new", golds_new), ("old", golds_old)):
            ids, mask, pos, slices = M.encode(tok, t_texts, t_spans, answers=golds)
            for layer in range(n_layers):
                rows_, positions, vectors = [], [], []
                for i in range(len(chunk)):
                    rows_.append(i)
                    positions.append(pos[i][0][-1])
                    vectors.append(store[layer][i, s_pos[i][0][-1]])
                with M.Interventions(model) as hooks:
                    hooks.overwrite(layer, rows_, positions, torch.stack(vectors))
                    scores = M.answer_scores(model, ids, mask, slices)
                for c, sc in zip(chunk, scores, strict=True):
                    records.append(
                        {
                            "h": c["h"],
                            "g": c["g"],
                            "use": c["use"],
                            "layer": layer,
                            "gold": gold_key,
                            **sc,
                        }
                    )
    write(
        run / "calibration" / "swap.json",
        {"pairs": pairs, "records": records, "patch": "last subject token, same layer"},
    )
    knockout(model_name, run, model, tok, info, kinds)


def knockout(model_name, run, model, tok, info, kinds):
    n_layers = len(kinds)
    rows = []
    for h in info["subjects"]:
        rows += calibration_rows(h)
    mean_inputs = {}
    if "linear" in kinds:
        store = {}
        ref = [{"prompt": q["prompt"] + D.DIRECT} for q in D.screening_queries()[:256]]
        with M.Interventions(model) as hooks:
            hooks.capture_mixer_inputs(store)
            for chunk in batches(ref):
                texts = [M.chat(tok, r["prompt"]) for r in chunk]
                ids, mask, _, _ = M.encode(tok, texts)
                with M.autocast():
                    model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
                for idx in list(store):
                    m = mask.cuda().bool()
                    acc = mean_inputs.setdefault(idx, [0.0, 0])
                    acc[0] = acc[0] + store[idx][-1][m].float().sum(0)
                    acc[1] += int(m.sum())
                    store[idx].clear()
        mean_inputs = {k: v[0] / v[1] for k, v in mean_inputs.items()}
    window = 5 if "linear" not in kinds else 4
    starts = range(0, n_layers - window + 1)
    variants = ["all"] + (["attention_only", "linear_only"] if "linear" in kinds else [])
    records = []
    for chunk in batches(rows):
        texts, spans = subject_spans(tok, chunk)
        ids, mask, pos, slices = M.encode(tok, texts, spans, answers=[r["gold"] for r in chunk])
        base = M.answer_scores(model, ids, mask, slices)
        for r, sc in zip(chunk, base, strict=True):
            records.append(
                {"subject": r["subject"], "use": r["use"], "start": None, "variant": "none", **sc}
            )
        for start in starts:
            for variant in variants:
                with M.Interventions(model) as hooks:
                    for layer in range(start, start + window):
                        rows_ = list(range(len(chunk)))
                        query = [slices[i][0] - 1 for i in rows_]
                        keys = [pos[i][0] for i in rows_]
                        if kinds[layer] == "attention" and variant != "linear_only":
                            hooks.block_attention(layer, rows_, query, keys)
                        if kinds[layer] == "linear" and variant != "attention_only":
                            hooks.block_linear_mixer(layer, rows_, query, keys, mean_inputs[layer])
                    scores = M.answer_scores(model, ids, mask, slices)
                for r, sc in zip(chunk, scores, strict=True):
                    records.append(
                        {
                            "subject": r["subject"],
                            "use": r["use"],
                            "start": start,
                            "variant": variant,
                            **sc,
                        }
                    )
    write(
        Path(run) / "calibration" / "knockout.json",
        {
            "window": window,
            "query": "last prompt position (answer predictor)",
            "keys": "all subject tokens",
            "records": records,
        },
    )


# ---------------------------------------------------------------- stage 1: injection


def train_batches(examples, replay, rng, half=8):
    """Batches of `half` new-fact examples plus `half` replay examples (sampled uniformly)."""
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order), half):
            yield [examples[j] for j in order[i : i + half]] + rng.sample(replay, half)


def replay_pool(teacher, tok, subjects):
    """The released model's own replies to natural questions in the evaluated answer formats
    (never an evaluated item), so training keeps its answer format and general behaviour."""
    pool = []
    for mode, rows in (
        ("direct", D.replay_queries(subjects, "direct")),
        ("cot", D.replay_queries(subjects, "cot")),
    ):
        for chunk in batches(rows):
            texts = [M.chat(tok, r, False) for r in chunk]
            ids, mask, _, _ = M.encode(tok, texts)
            replies = M.generate(teacher, tok, ids, mask, COT_TOKENS if mode == "cot" else 24)
            pool += [
                {"user": u, "assistant": a.strip(), "replay": True, "mode": mode}
                for u, a in zip(chunk, replies, strict=True)
            ]
    return pool


def u0_accuracy(model, tok, facts):
    rows = [{"prompt": D.fill(D.U0, f) + D.DIRECT, "gold": f.v} for f in facts]
    scores = teacher_forced(model, tok, rows)
    return sum(s["greedy"] for s in scores) / len(scores)


def inject(
    model_name,
    run,
    method,
    model,
    tok,
    lr,
    max_steps=1500,
    check_every=20,
    target=0.95,
    tracker=None,
    teacher=None,
    mlp_layers=None,
    extra_examples=None,
):
    """Full fine-tuning (m1, m2), context distillation (m3; teacher = frozen released model
    with the fact in the system prompt), down_proj-only fine-tuning (m5), or, with `mlp_layers`,
    fine-tuning of the MLPs of those layers only (follow-up m6i/m6d). `extra_examples` (v2 E4:
    usage practice or paraphrases) are appended to the new-fact examples."""
    run = Path(run)
    facts = facts_from(run)
    rng = random.Random(SEED + 2)
    torch.manual_seed(SEED + 2)
    examples = []
    if method == "m3":
        for p in teacher.parameters():
            p.requires_grad_(False)
        prompts = [(f, u) for f in facts for u, _ in D.training_examples(f, "m3")]
        for chunk in batches(prompts):
            systems = ["Fact: " + D.context_statement(f) for f, _ in chunk]
            texts = [M.chat(tok, u, False, s) for (_, u), s in zip(chunk, systems, strict=True)]
            ids, mask, _, _ = M.encode(tok, texts)
            replies = M.generate(teacher, tok, ids, mask, 48)
            for (_f, u), s, reply in zip(chunk, systems, replies, strict=True):
                examples.append({"user": u, "assistant": reply.strip(), "system": s})
    else:
        for f in facts:
            examples += [{"user": u, "assistant": a} for u, a in D.training_examples(f, method)]
    examples += list(extra_examples or [])
    replay = replay_pool(teacher, tok, read(run / "facts.json")["subjects"])
    if mlp_layers is not None:
        params = [p for i in mlp_layers for p in M.layers(model)[i].mlp.parameters()]
    else:
        params = [
            p
            for n, p in model.named_parameters()
            if method != "m5" or n.endswith("mlp.down_proj.weight")
        ]
    for p in model.parameters():
        p.requires_grad_(any(p is q for q in params))
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    stream = train_batches(examples, replay, rng)
    history, step, acc = [], 0, 0.0
    t0 = time.time()
    model.train()
    tokens = 0
    while step < max_steps:
        chunk = next(stream)
        texts = [M.chat(tok, e["user"], False) for e in chunk]
        answers = [e["assistant"] + "<|im_end|>" for e in chunk]
        ids, mask, _, slices = M.encode(tok, texts, answers=answers)
        ids, mask = ids.cuda(), mask.cuda()
        with M.autocast():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
        loss = 0.0
        # KL to the frozen released model: on replay examples (no fact in context) for every
        # method, and on new-fact examples for m3 (fact in the system prompt).
        distil = [i for i, e in enumerate(chunk) if e.get("replay") or method == "m3"]
        if distil:
            t_texts = [M.chat(tok, chunk[i]["user"], False, chunk[i].get("system")) for i in distil]
            t_answers = [answers[i] for i in distil]
            t_ids, t_mask, _, t_slices = M.encode(tok, t_texts, answers=t_answers)
            with torch.no_grad(), M.autocast():
                t_logits = teacher(
                    input_ids=t_ids.cuda(), attention_mask=t_mask.cuda(), use_cache=False
                ).logits
            for k, i in enumerate(distil):
                (a, b), (ta, tb) = slices[i], t_slices[k]
                assert b - a == tb - ta
                p_t = torch.log_softmax(t_logits[k, ta - 1 : tb - 1].float(), -1)
                p_s = torch.log_softmax(logits[i, a - 1 : b - 1], -1)
                loss = loss + (p_t.exp() * (p_t - p_s)).sum(-1).sum()
                tokens += b - a
        for i, (a, b) in enumerate(slices):
            if i in distil:
                continue
            labels = ids[i, a:b]
            loss = loss + torch.nn.functional.cross_entropy(
                logits[i, a - 1 : b - 1], labels, reduction="sum"
            )
            tokens += b - a
        n_tok = sum(b - a for a, b in slices)
        loss = loss / n_tok
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / 10)
        optimizer.step()
        step += 1
        if step % check_every == 0 or step == max_steps:
            model.eval()
            acc = u0_accuracy(model, tok, facts)
            model.train()
            history.append(
                {"step": step, "loss": float(loss), "u0": acc, "seconds": time.time() - t0}
            )
            if tracker:
                tracker.log({"train/loss": float(loss), "train/u0_teacher_forced": acc}, step=step)
            if acc >= target:
                break
    model.eval()
    trainable = sum(p.numel() for p in params)
    summary = {
        "method": method,
        "lr": lr,
        "steps": step,
        "u0_teacher_forced": acc,
        "examples": len(examples),
        "supervised_tokens": tokens,
        "trainable_parameters": trainable,
        "seconds": time.time() - t0,
        "flops_estimate": 6 * trainable * tokens if method != "m5" and not mlp_layers else None,
        "mlp_layers": mlp_layers,
        "history": history,
        "stopped": "target" if acc >= target else "max_steps",
    }
    summary["replay_pool"] = replay
    if method == "m3":
        summary["teacher_replies"] = examples
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    for p in model.parameters():
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    weights = run / "weights" / f"{method}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, weights)
    write(run / "inject" / f"{method}.json", summary)
    return summary


# ---------------------------------------------------------------- stage 1: MEMIT


def wikitext_lines(root, limit):
    import pyarrow.parquet as pq

    files = sorted(Path(root).glob("wikitext-103-raw-v1/train-*.parquet"))
    lines = []
    for f in files:
        for text in pq.read_table(f).column("text").to_pylist():
            text = text.strip()
            if len(text) > 200 and not text.startswith("="):
                lines.append(text)
            if len(lines) >= limit:
                return lines
    return lines


def down_inputs(model, layer_ids, store):
    handles = []
    for idx in layer_ids:

        def pre(module, args, idx=idx):
            store[idx] = args[0].detach()

        handles.append(M.layers(model)[idx].mlp.down_proj.register_forward_pre_hook(pre))
    return handles


@torch.no_grad()
def second_moments(model, tok, layer_ids, wikitext_root, n_texts=10000, max_len=128):
    lines = wikitext_lines(wikitext_root, n_texts)
    sums = {i: None for i in layer_ids}
    count = 0
    store = {}
    handles = down_inputs(model, layer_ids, store)
    try:
        for chunk in batches(lines, 64):
            enc = tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )
            m = enc["attention_mask"].cuda().bool()
            with M.autocast():
                model(input_ids=enc["input_ids"].cuda(), attention_mask=m.long(), use_cache=False)
            for i in layer_ids:
                k = store[i][m].float()
                sums[i] = k.T @ k if sums[i] is None else sums[i] + k.T @ k
            count += int(m.sum())
    finally:
        for h in handles:
            h.remove()
    return {i: (s / count).float() for i, s in sums.items()}, count


def memit(
    model_name,
    run,
    model,
    tok,
    layer_ids,
    lam,
    moments,
    facts,
    steps=25,
    v_lr=0.5,
    clamp=0.75,
    kl_weight=0.0625,
):
    """MEMIT (Meng et al. 2023) on down_proj at the subject's last token, batch edit."""
    last = max(layer_ids)
    prompts = []
    for f in facts:
        for user, _ in D.training_examples(f, "m1"):
            prompts.append((f, user))
    # 1) target residual at the last edit layer, per fact
    targets = {}
    for f in facts:
        rows = [{"prompt": u, "gold": f.v, "subject": f.h} for g, u in prompts if g is f]
        texts, spans = subject_spans(tok, rows)
        ids, mask, pos, slices = M.encode(
            tok, texts, spans, answers=[f.v + "<|im_end|>"] * len(rows)
        )
        ess = M.chat(tok, f"Tell me about {f.h}.")
        e_ids, e_mask, e_pos, _ = M.encode(
            tok, [ess], [[M.find_span(ess, f.h, ess.rfind("Tell me"))]]
        )
        store = {}
        with M.Interventions(model) as hooks, M.autocast():
            hooks.capture([last], store)
            model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        h0 = store[last][0, pos[0][0][-1]].float()
        with torch.no_grad(), M.autocast():
            base_ess = torch.log_softmax(
                model(input_ids=e_ids.cuda(), attention_mask=e_mask.cuda()).logits[0, -1].float(),
                -1,
            )
        delta = torch.zeros_like(h0, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=v_lr)
        for p in model.parameters():
            p.requires_grad_(False)

        def add_at(where, delta=delta):
            def hook(module, args, output):
                if output.shape[1] == 1:
                    return output
                out = output.clone()
                for r, p in where:
                    out[r, p] = out[r, p] + delta.to(out.dtype)
                return out

            return hook

        rewrite_at = [(r, pos[r][0][-1]) for r in range(ids.shape[0])]
        for _ in range(steps):
            handle = M.layers(model)[last].register_forward_hook(add_at(rewrite_at))
            try:
                with M.autocast():
                    logits = model(
                        input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False
                    ).logits
            finally:
                handle.remove()
            handle = M.layers(model)[last].register_forward_hook(add_at([(0, e_pos[0][0][-1])]))
            try:
                with M.autocast():
                    e_logits = model(
                        input_ids=e_ids.cuda(), attention_mask=e_mask.cuda(), use_cache=False
                    ).logits
            finally:
                handle.remove()
            nll = 0.0
            for r, (a, b) in enumerate(slices):
                lp = torch.log_softmax(logits[r, a - 1 : b - 1].float(), -1)
                nll = nll - lp.gather(-1, ids[r, a:b].cuda()[:, None]).mean()
            nll = nll / len(slices)
            ess_lp = torch.log_softmax(e_logits[0, -1].float(), -1)
            kl = (base_ess.exp() * (base_ess - ess_lp)).sum()
            loss = nll + kl_weight * kl + 1e-3 * delta.norm() ** 2 / h0.norm() ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                limit = clamp * h0.norm()
                if delta.norm() > limit:
                    delta.mul_(limit / delta.norm())
        targets[f.fid] = (h0 + delta.detach(), rewrite_at, ids, mask)
    # 2) spread the residual over the edit layers (ascending), MEMIT closed form
    report = []
    for layer in sorted(layer_ids):
        keys, resid = [], []
        for f in facts:
            z, where, ids, mask = targets[f.fid]
            store, kin = {}, {}
            hs = down_inputs(model, [layer], kin)
            try:
                with M.Interventions(model) as hooks, torch.no_grad(), M.autocast():
                    hooks.capture([last], store)
                    model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            k = torch.stack([kin[layer][r, p] for r, p in where]).float().mean(0)
            cur = torch.stack([store[last][r, p] for r, p in where]).float().mean(0)
            keys.append(k)
            resid.append((z - cur) / (last - layer + 1))
        K = torch.stack(keys, 1).double()
        R = torch.stack(resid, 1).double()
        C = lam * moments[layer].double().cuda()
        upd = R @ torch.linalg.solve(C + K @ K.T, K).T
        weight = M.layers(model)[layer].mlp.down_proj.weight
        with torch.no_grad():
            weight.add_(upd.to(weight.dtype))
        report.append(
            {"layer": layer, "update_norm": float(upd.norm()), "residual_norm": float(R.norm())}
        )
    return report


def run_memit(model_name, run, model, tok, wikitext_root, windows, lams, batch=None):
    """Write-layer sweep on calibration facts (U0 efficacy plus natural-recall keep), then edit
    all facts with the chosen window and lambda: in one batch (m4), or in consecutive batches of
    `batch` facts, the sweep's own batch size (m4b). m4b reuses an existing m4 sweep."""
    run = Path(run)
    tag = "m4" if batch is None else "m4b"
    facts = facts_from(run)
    calib = [f for f in facts if f.split == "calibration"]
    t0 = time.time()
    previous = run / "inject" / "m4.json"
    original = {n: p.detach().clone() for n, p in model.named_parameters() if "down_proj" in n}
    if batch is not None and previous.exists():
        sweep = read(previous)["sweep"]
        best = read(previous)["chosen"]
        moments, n_tokens = second_moments(model, tok, best["window"], wikitext_root)
    else:
        all_layers = sorted({i for w in windows for i in w})
        moments, n_tokens = second_moments(model, tok, all_layers, wikitext_root)
        natural = [
            {"prompt": D.CAPITAL_REV[0].format(h=f.h) + D.DIRECT, "gold": f.country_h}
            for f in calib
        ]
        sweep = []
        for window in windows:
            for lam in lams:
                memit(model_name, run, model, tok, window, lam, moments, calib)
                eff = u0_accuracy(model, tok, calib)
                keep = sum(s["greedy"] for s in teacher_forced(model, tok, natural)) / len(natural)
                sweep.append(
                    {
                        "window": window,
                        "lambda": lam,
                        "u0": eff,
                        "natural_keep": keep,
                        "score": eff + keep,
                    }
                )
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in original:
                            p.copy_(original[n])
        best = max(sweep, key=lambda s: (s["score"], s["u0"]))
    groups = (
        [facts] if batch is None else [facts[i : i + batch] for i in range(0, len(facts), batch)]
    )
    report = []
    for group in groups:
        report.append(
            memit(model_name, run, model, tok, best["window"], best["lambda"], moments, group)
        )
    acc = u0_accuracy(model, tok, facts)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if n in original}
    weights = run / "weights" / f"{tag}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, weights)
    write(
        run / "inject" / f"{tag}.json",
        {
            "method": tag,
            "sweep": sweep,
            "chosen": best,
            "batch": batch or len(facts),
            "layers_report": report,
            "u0_teacher_forced": acc,
            "moment_tokens": n_tokens,
            "seconds": time.time() - t0,
        },
    )


# ---------------------------------------------------------------- stage 2: transplant


def layer_pairs(n_layers, stride=2):
    grid = list(range(0, n_layers, stride))
    if n_layers - 1 not in grid:
        grid.append(n_layers - 1)
    return [(s, t) for s in grid for t in grid if s >= t]


def interface_layers(run):
    """L*: layers where swapping in another subject's last-token state transfers its country,
    for every swap use whose unpatched teacher-forced accuracy is at least 0.5."""
    new, old = {}, {}
    for r in read(Path(run) / "calibration" / "swap.json")["records"]:
        table = new if r["gold"] == "new" else old
        table.setdefault(r["use"], {}).setdefault(r["layer"], []).append(r["greedy"])
    bands = []
    for use, curve in new.items():
        last = old[use][max(old[use])]
        if sum(last) / len(last) < 0.5:
            continue
        means = {layer: sum(v) / len(v) for layer, v in curve.items()}
        top = max(means.values())
        bands.append({layer for layer, m in means.items() if m >= 0.8 * top})
    return sorted(set.intersection(*bands)) if bands else []


def best_pair(records, direction, prefix, condition):
    """Pair with the highest fact-level mean greedy match (ties: closer layers, then later)."""
    by = {}
    for r in records:
        if (
            r["direction"] == direction
            and r["use"].startswith(prefix)
            and condition(r["src"], r["tgt"])
        ):
            by.setdefault((r["src"], r["tgt"]), {}).setdefault(r["fid"], []).append(r["greedy"])
    scored = []
    for (src, tgt), facts in by.items():
        rate = sum(sum(v) / len(v) for v in facts.values()) / len(facts)
        scored.append((rate, -abs(src - tgt), src, tgt))
    if not scored:
        return None
    rate, _, src, tgt = max(scored)
    return {"src": src, "tgt": tgt, "calibration_rate": rate}


def patch_rows(rows, base, injected, tok, plan, n_layers):
    """Teacher-forced scores of explicit-subject rows under the transplant plan:
    forward (injected -> base) and reverse (base -> injected) at all subject tokens;
    back_last / back_subject: the injected model's own later-layer state at the last prompt
    position / subject tokens written into an earlier layer (U2 rows only)."""
    records = []
    for chunk in batches(rows, 16):
        texts, spans = subject_spans(tok, chunk)
        ids, mask, pos, slices = M.encode(tok, texts, spans, answers=[r["gold"] for r in chunk])
        caps = {}
        for name, mdl in (("base", base), ("injected", injected)):
            store = {}
            with M.Interventions(mdl) as hooks, M.autocast():
                hooks.capture(range(n_layers), store)
                mdl(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
            caps[name] = store
        jobs = [
            ("forward", "injected", base, None),
            ("reverse", "base", injected, None),
            ("back_last", "injected", injected, "last"),
            ("back_subject", "injected", injected, "subject"),
        ]
        for direction, src, tgt, where in jobs:
            members = list(range(len(chunk)))
            if where:
                members = [i for i, r in enumerate(chunk) if r["use"].startswith("U2")]
            if not members:
                continue
            for ls, lt in plan.get(direction, []):
                rows_, positions, vectors = [], [], []
                for i in members:
                    ps = [slices[i][0] - 1] if where == "last" else pos[i][0]
                    for q in ps:
                        rows_.append(i)
                        positions.append(q)
                        vectors.append(caps[src][ls][i, q])
                with M.Interventions(tgt) as hooks:
                    hooks.overwrite(lt, rows_, positions, torch.stack(vectors))
                    scores = M.answer_scores(tgt, ids, mask, slices)
                for i in members:
                    r = chunk[i]
                    records.append(
                        {
                            "fid": r["fid"],
                            "use": r["use"],
                            "split": r["split"],
                            "direction": direction,
                            "src": ls,
                            "tgt": lt,
                            **scores[i],
                        }
                    )
    return records


@torch.no_grad()
def transplant(model_name, run, method, base, injected, tok):
    """Full layer-pair grid on calibration facts; primary pairs chosen there by the
    pre-registered rule; main facts get the same-layer sweep plus the primary pairs."""
    run = Path(run)
    facts = facts_from(run)
    n_layers = len(M.layers(base))
    pairs = layer_pairs(n_layers)
    same = [(a, b) for a, b in pairs if a == b]
    later = [(a, b) for a, b in pairs if a > b]
    rows = {"calibration": [], "main": []}
    for f in facts:
        for r in D.use_queries(f):
            if r["mode"] == "direct" and r["subject"] and r["use"] != "shortcut":
                rows["calibration" if f.split == "calibration" else "main"].append(
                    {**r, "split": f.split}
                )
    calib_facts = [f for f in facts if f.split == "calibration"]
    main_facts = [f for f in facts if f.split != "calibration"]
    full = {"forward": pairs, "reverse": same, "back_last": later, "back_subject": later}
    records = patch_rows(rows["calibration"], base, injected, tok, full, n_layers)
    records += bridge_patch(calib_facts, injected, tok, pairs)
    band = interface_layers(run)
    primary = {
        "same_layer": best_pair(
            records, "forward", "U0", lambda a, b: a == b and (not band or a in band)
        ),
        "later_to_earlier": best_pair(records, "forward", "U0", lambda a, b: a > b),
        "back_last": best_pair(records, "back_last", "U2", lambda a, b: a > b),
        "back_subject": best_pair(records, "back_subject", "U2", lambda a, b: a > b),
        "bridge": best_pair(records, "bridge", "U3", lambda a, b: True),
    }

    def chosen(name):
        p = primary[name]
        return [(p["src"], p["tgt"])] if p else []

    plan = {
        "forward": sorted(set(same + chosen("later_to_earlier"))),
        "reverse": same,
        "back_last": chosen("back_last"),
        "back_subject": chosen("back_subject"),
    }
    records += patch_rows(rows["main"], base, injected, tok, plan, n_layers)
    records += bridge_patch(main_facts, injected, tok, sorted(set(same + chosen("bridge"))))
    write(
        run / "transplant" / f"{method}.json",
        {
            "method": method,
            "pairs": pairs,
            "interface_layers": band,
            "primary": primary,
            "main_plan": plan,
            "records": records,
        },
    )


@torch.no_grad()
def bridge_patch(facts, model, tok, pairs):
    """T3: put the injected model's explicit-subject state of h (from the U0 prompt) at the last
    token of the country name in the U3 prompt, where the bridge would be computed."""
    n_layers = len(M.layers(model))
    out = []
    items = [
        (f, r)
        for f in facts
        for r in D.use_queries(f)
        if r["mode"] == "direct" and r["use"].startswith("U3")
    ]
    for chunk in batches(items, 16):
        src_rows = [{"prompt": D.fill(D.U0, f) + D.DIRECT, "subject": f.h} for f, _ in chunk]
        s_texts, s_spans = subject_spans(tok, src_rows)
        s_ids, s_mask, s_pos, _ = M.encode(tok, s_texts, s_spans)
        store = {}
        with M.Interventions(model) as hooks, M.autocast():
            hooks.capture(range(n_layers), store)
            model(input_ids=s_ids.cuda(), attention_mask=s_mask.cuda(), use_cache=False)
        t_rows = [{**r, "subject": f.country_h} for f, r in chunk]
        texts, spans = subject_spans(tok, t_rows)
        ids, mask, pos, slices = M.encode(tok, texts, spans, answers=[r["gold"] for _, r in chunk])
        for ls, lt in pairs:
            rows_ = list(range(len(chunk)))
            positions = [pos[i][0][-1] for i in rows_]
            vectors = torch.stack([store[ls][i, s_pos[i][0][-1]] for i in rows_])
            with M.Interventions(model) as hooks:
                hooks.overwrite(lt, rows_, positions, vectors)
                scores = M.answer_scores(model, ids, mask, slices)
            for (f, r), sc in zip(chunk, scores, strict=True):
                out.append(
                    {
                        "fid": f.fid,
                        "use": r["use"],
                        "split": f.split,
                        "direction": "bridge",
                        "src": ls,
                        "tgt": lt,
                        **sc,
                    }
                )
    return out
