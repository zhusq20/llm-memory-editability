"""knowledge-read-interface-v2: shared helpers, R2/R3 measurement fixes and E1 (subject-state
write upper bound in the frozen released model).

E1 learns one vector delta_h per subject, added to the output of decoder layer l at the last
token of h whenever the question contains h (prefill only). E1-R adds one vector delta_R per
relation at every token of the relation phrase. E1-N writes a counterfactual country for h as a
positive control of the optimization.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S
from llm_memory_editability import read_interface_v2_data as V

SEED = 20261021
E1_LAYERS = {"qwen3-1.7b": [6, 10, 14], "qwen3.5-2b": [4, 8, 12]}
E1_STEPS = 300
E1_CHECK = 20
E1_TARGET = 0.95
E1_CLAMP = 0.75
E1_KL_WEIGHT = 1.0
E1_STEP_FRACTION = 0.02
E1_TEMPLATES_PER_STEP = 6


# ---------------------------------------------------------------- tokenization with positions


def user_spans(text, prompt, names):
    """Character spans of each name inside the user prompt of a chat-formatted text (an empty
    span when the name is absent or None)."""
    at = text.rfind(prompt)
    out = []
    for name in names:
        start = text.find(name, at) if name else -1
        out.append((start, start + len(name)) if start >= 0 else (0, 0))
    return out


def tokenize(tok, text, spans, answer=None):
    """Unpadded ids, token positions of each span, and the answer token range [a, b)."""
    full = text + (answer or "")
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    positions = [M.span_positions(offsets, s, e) if e > s else [] for s, e in spans]
    item = {"ids": enc["input_ids"], "positions": positions}
    if answer is not None:
        a = next(j for j, (x, y) in enumerate(offsets) if y > len(text) and y > x)
        item["answer"] = (a, len(enc["input_ids"]))
    return item


def collate(tok, items):
    """Left-pad tokenized items; positions and answer ranges move by each row's padding."""
    width = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    positions, slices = [], []
    for i, it in enumerate(items):
        pad = width - len(it["ids"])
        ids[i, pad:] = torch.tensor(it["ids"])
        mask[i, pad:] = 1
        positions.append([[pad + p for p in ps] for ps in it["positions"]])
        if "answer" in it:
            a, b = it["answer"]
            slices.append((pad + a, pad + b))
    return ids, mask, positions, slices


def answer_logits(model, ids, mask, slices):
    """Logits only at the positions that predict answer tokens (saves the full-vocab matmul)."""
    hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    rows = torch.cat([torch.full((b - a,), i) for i, (a, b) in enumerate(slices)])
    pos = torch.cat([torch.arange(a - 1, b - 1) for a, b in slices])
    targets = torch.cat([ids[i, a:b] for i, (a, b) in enumerate(slices)])
    return model.lm_head(hidden[rows, pos]).float(), targets


# ---------------------------------------------------------------- patched answering and scoring


class Patch:
    """Additive writes at layer `layer`: delta_h[subject] at the subject's last token and
    delta_R[relation] at every relation-phrase token of the user prompt."""

    def __init__(self, layer, subject_vectors=None, relation_vectors=None):
        self.layer = layer
        self.subject_vectors = subject_vectors or {}
        self.relation_vectors = relation_vectors or {}

    def names(self, row):
        return [row.get("subject"), row.get("relation_text")]

    def register(self, hooks, chunk, positions):
        rows, pos, vecs = [], [], []
        for i, row in enumerate(chunk):
            h_pos, r_pos = positions[i]
            vec = self.subject_vectors.get(row.get("subject"))
            if vec is not None and h_pos:
                rows.append(i)
                pos.append(h_pos[-1])
                vecs.append(vec)
            vec = self.relation_vectors.get(row.get("relation_text"))
            if vec is not None:
                for p in r_pos:
                    rows.append(i)
                    pos.append(p)
                    vecs.append(vec)
        if rows:
            hooks.add(self.layer, rows, pos, torch.stack(vecs))


def golds_of(row):
    return row["golds"] if "golds" in row else [row["gold"]]


def answer_rows(model, tok, rows, patch=None, system_for=None, batch=32):
    """Generate and grade rows (keys prompt/type/gold or golds/mode, optional exclude, subject,
    relation_text, system); adds answer/strict/lenient/first, plus reply for step-by-step."""
    graders = V.graders()
    out = []
    for mode in ("direct", "cot"):
        subset = [r for r in rows if r.get("mode", "direct") == mode]
        stepwise = mode == "cot"
        for chunk in S.batches(subset, batch if not stepwise else 64):
            texts, items = [], []
            for r in chunk:
                system = r.get("system") or (system_for(r) if system_for else None)
                text = M.chat(tok, r["prompt"], False, system)
                names = patch.names(r) if patch else []
                items.append(tokenize(tok, text, user_spans(text, r["prompt"], names)))
                texts.append(text)
            ids, mask, positions, _ = collate(tok, items)
            with M.Interventions(model) as hooks:
                if patch:
                    patch.register(hooks, chunk, positions)
                raw = M.generate(model, tok, ids, mask, S.COT_TOKENS if stepwise else 24)
            for r, text in zip(chunk, raw, strict=True):
                answer = M.final_answer(text, stepwise)
                grade = V.grade_any(
                    graders[r["type"]], answer, golds_of(r), set(r.get("exclude", ()))
                )
                record = {**r, "answer": answer, **grade}
                if stepwise:
                    record["reply"] = text
                out.append(record)
    return out


@torch.no_grad()
def score_rows(model, tok, rows, patch=None, batch=32):
    """Teacher-forced greedy match and log-prob of each row's `gold` after its prompt."""
    out = []
    for chunk in S.batches(rows, batch):
        items = []
        for r in chunk:
            text = M.chat(tok, r["prompt"], False, r.get("system"))
            names = patch.names(r) if patch else []
            items.append(tokenize(tok, text, user_spans(text, r["prompt"], names), r["gold"]))
        ids, mask, positions, slices = collate(tok, items)
        with M.Interventions(model) as hooks:
            if patch:
                patch.register(hooks, chunk, positions)
            out += M.answer_scores(model, ids, mask, slices)
    return out


# ---------------------------------------------------------------- R2 distractor context


def r2(model, tok, facts):
    """Released model with the target fact plus three same-relation distractors in context."""
    systems = V.distractor_systems(facts)
    rows = []
    for f in facts:
        if f.fid not in systems:
            continue
        for r in D.use_queries(f):
            if r["use"] == "U0":
                continue
            rows.append(
                {
                    **r,
                    "split": f.split,
                    "system": systems[f.fid]["system"],
                    "exclude": S.exclusions(r, f),
                }
            )
    t0 = time.time()
    answered = answer_rows(model, tok, rows)
    return {"systems": systems, "uses": answered, "seconds": time.time() - t0}


# ---------------------------------------------------------------- R3 natural references


def r3(model, tok, known):
    rows = V.natural_reference_rows(known)
    t0 = time.time()
    answered = answer_rows(model, tok, rows)
    return {"rows": answered, "seconds": time.time() - t0}


# ---------------------------------------------------------------- E1 optimization


def e1_items(tok, facts, subjects_index, relation_text=False):
    """Training items for delta fitting: the 24 M2 examples of every fact."""
    items = []
    for f in facts:
        rel = D.RELATIONS[f.relation]
        for k, (user, answer) in enumerate(D.training_examples(f, "m2")):
            text = M.chat(tok, user, False)
            spans = user_spans(text, user, [f.h, rel if relation_text else None])
            item = tokenize(tok, text, spans, answer + "<|im_end|>")
            item.update({"sub": subjects_index[f.h], "rel": rel, "fid": f.fid, "template": k})
            assert item["positions"][0], (f.fid, user)
            items.append(item)
    return items


def natural_prompts(h):
    """The subject's natural prompts kept close to the released model (country, reverse
    capital, description)."""
    return [
        D.CITY_COUNTRY.format(v=h) + D.DIRECT,
        D.CAPITAL_REV[0].format(h=h) + D.DIRECT,
        f"Tell me about {h}.",
    ]


@torch.no_grad()
def kl_items(model, tok, prompts_by_subject, subjects_index):
    """First-answer-position log-probs of the released model on each subject's natural prompts."""
    items = []
    for h, prompts in prompts_by_subject.items():
        for p in prompts:
            text = M.chat(tok, p, False)
            item = tokenize(tok, text, user_spans(text, p, [h, None]))
            item.update({"sub": subjects_index[h], "rel": None})
            items.append(item)
    for chunk in S.batches(items, 64):
        ids, mask, _, _ = collate(tok, chunk)
        with M.autocast():
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits
        logp = torch.log_softmax(logits[:, -1].float(), -1)
        for it, row in zip(chunk, logp, strict=True):
            it["base_logp"] = row
    return items


@torch.no_grad()
def residual_norms(model, tok, items, layer, which):
    """Mean layer-l output norm at the subject's last token (which=0) or relation tokens (1),
    grouped by item['sub'] or item['rel']."""
    sums = {}
    for chunk in S.batches(items, 64):
        ids, mask, positions, _ = collate(tok, chunk)
        store = {}
        with M.Interventions(model) as hooks, M.autocast():
            hooks.capture([layer], store)
            model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        for i, it in enumerate(chunk):
            ps = positions[i][which]
            ps = ps[-1:] if which == 0 else ps
            key = it["sub"] if which == 0 else it["rel"]
            for p in ps:
                acc = sums.setdefault(key, [0.0, 0])
                acc[0] += float(store[layer][i, p].float().norm())
                acc[1] += 1
    return {k: s / n for k, (s, n) in sums.items()}


def add_writes(hooks, layer, chunk, positions, dh, dr, rel_index):
    rows, pos, vecs = [], [], []
    for i, it in enumerate(chunk):
        h_pos = positions[i][0]
        if h_pos and it["sub"] is not None:
            rows.append(i)
            pos.append(h_pos[-1])
            vecs.append(dh[it["sub"]])
        if dr is not None and it.get("rel") is not None and len(positions[i]) > 1:
            for p in positions[i][1]:
                rows.append(i)
                pos.append(p)
                vecs.append(dr[rel_index[it["rel"]]])
    if rows:
        hooks.add(layer, rows, pos, torch.stack(vecs))


def fit_writes(
    model,
    tok,
    layer,
    train,
    kl,
    n_sub,
    h_bounds,
    check_rows,
    stop="subject",
    dr=None,
    r_bounds=None,
    rel_index=None,
    train_dr=False,
    steps=E1_STEPS,
    seed=SEED,
):
    """Adam on delta_h (and optionally delta_R): CE on answer tokens of a random subset of each
    fact's examples per step, plus KL to the released model on the subjects' natural prompts.

    `check_rows` (one teacher-forced U0 row per fact, keyed by 'sub') decide stopping: per
    subject when all its rows are greedy-correct (stop='subject'), or when the overall rate
    reaches E1_TARGET (stop='global'). Norms are clamped to E1_CLAMP times the mean residual
    norm after every step; frozen subjects keep their vectors."""
    d = model.get_input_embeddings().weight.shape[1]
    rng = random.Random(seed)
    dh = torch.zeros(n_sub, d, device="cuda", requires_grad=True)
    bound_h = torch.tensor([h_bounds.get(i, 0.0) for i in range(n_sub)], device="cuda")
    # Adam moves each coordinate by about lr, so one step changes a vector by about
    # E1_STEP_FRACTION of the mean residual norm.
    lr_h = E1_STEP_FRACTION * float(bound_h[bound_h > 0].mean()) / math.sqrt(d)
    groups = [{"params": [dh], "lr": lr_h}]
    if train_dr:
        mean_r = sum(r_bounds.values()) / len(r_bounds)
        order = sorted(rel_index, key=rel_index.get)
        bound_r = torch.tensor([r_bounds.get(k, mean_r) for k in order], device="cuda")
        groups.append({"params": [dr], "lr": E1_STEP_FRACTION * mean_r / math.sqrt(d)})
    opt = torch.optim.Adam(groups)
    by_fact = {}
    for it in train:
        by_fact.setdefault(it["fid"], []).append(it)
    subs_with_data = sorted({it["sub"] for it in train})
    frozen = torch.zeros(n_sub, dtype=torch.bool, device="cuda")
    history, step = [], 0
    t0 = time.time()
    done = False
    while step < steps and not done:
        active = [s for s in subs_with_data if not frozen[s]]
        if stop == "global":
            active = subs_with_data
        batch = []
        for items in by_fact.values():
            if items[0]["sub"] in active:
                batch += rng.sample(items, min(E1_TEMPLATES_PER_STEP, len(items)))
        kl_batch = [it for it in kl if it["sub"] in active]
        n_tok = sum(b - a for it in batch for a, b in [it["answer"]])
        opt.zero_grad(set_to_none=True)
        ce_total, kl_total = 0.0, 0.0
        for chunk in S.batches(batch, 96):
            ids, mask, positions, slices = collate(tok, chunk)
            with M.Interventions(model) as hooks, M.autocast():
                add_writes(hooks, layer, chunk, positions, dh, dr, rel_index)
                logits, targets = answer_logits(model, ids.cuda(), mask.cuda(), slices)
            ce = torch.nn.functional.cross_entropy(logits, targets.cuda(), reduction="sum")
            (ce / n_tok).backward()
            ce_total += float(ce.detach())
        for chunk in S.batches(kl_batch, 64):
            ids, mask, positions, _ = collate(tok, chunk)
            with M.Interventions(model) as hooks, M.autocast():
                add_writes(hooks, layer, chunk, positions, dh, None, None)
                hidden = model.model(
                    input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False
                ).last_hidden_state
                logp = torch.log_softmax(model.lm_head(hidden[:, -1]).float(), -1)
            base = torch.stack([it["base_logp"] for it in chunk])
            div = (base.exp() * (base - logp)).sum()
            (E1_KL_WEIGHT * div / max(1, len(kl_batch))).backward()
            kl_total += float(div.detach())
        keep = dh.detach()[frozen].clone()
        opt.step()
        with torch.no_grad():
            dh[frozen] = keep
            norms = dh.norm(dim=-1).clamp_min(1e-12)
            dh.mul_(torch.clamp(E1_CLAMP * bound_h / norms, max=1.0)[:, None])
            if train_dr:
                rn = dr.norm(dim=-1).clamp_min(1e-12)
                dr.mul_(torch.clamp(E1_CLAMP * bound_r / rn, max=1.0)[:, None])
        step += 1
        if step % E1_CHECK == 0 or step == steps:
            fixed_r = dr.detach() if dr is not None else None
            scores = check(model, tok, layer, check_rows, dh.detach(), fixed_r, rel_index)
            rate = sum(scores.values()) / max(1, len(scores))
            by_sub = {}
            for (sub, _), ok in scores.items():
                by_sub.setdefault(sub, []).append(ok)
            if stop == "subject":
                for sub, oks in by_sub.items():
                    if all(oks):
                        frozen[sub] = True
                done = bool(all(frozen[s] for s in subs_with_data))
            else:
                done = rate >= E1_TARGET
            history.append(
                {
                    "step": step,
                    "ce_per_token": ce_total / max(1, n_tok),
                    "kl_mean": kl_total / max(1, len(kl_batch)),
                    "u0_tf": rate,
                    "frozen": int(frozen.sum()),
                    "seconds": time.time() - t0,
                }
            )
    return dh.detach(), {
        "steps": step,
        "history": history,
        "frozen_subjects": int(frozen.sum()),
        "subjects": len(subs_with_data),
        "seconds": time.time() - t0,
    }


@torch.no_grad()
def check(model, tok, layer, rows, dh, dr, rel_index):
    """Teacher-forced greedy match of check rows under the current writes; keyed by (sub, fid)."""
    out = {}
    for chunk in S.batches(rows, 64):
        ids, mask, positions, slices = collate(tok, chunk)
        with M.Interventions(model) as hooks:
            add_writes(hooks, layer, chunk, positions, dh, dr, rel_index)
            scores = M.answer_scores(model, ids, mask, slices)
        for it, sc in zip(chunk, scores, strict=True):
            out[(it["sub"], it["fid"])] = sc["greedy"]
    return out


def u0_check_rows(tok, facts, subjects_index, relation_text=False):
    rows = []
    for f in facts:
        user = D.fill(D.U0, f) + D.DIRECT
        text = M.chat(tok, user, False)
        rel = D.RELATIONS[f.relation]
        spans = user_spans(text, user, [f.h, rel if relation_text else None])
        item = tokenize(tok, text, spans, f.v)
        item.update({"sub": subjects_index[f.h], "rel": rel, "fid": f.fid})
        rows.append(item)
    return rows


# ---------------------------------------------------------------- E1 evaluation


def e1_eval_rows(facts):
    """Main-fact uses that contain h (U0, U1, U2 direct and step-by-step) plus h's natural
    retention prompts. Each row carries `subject` and `relation_text` for the patch."""
    rows = []
    by_subject = {}
    for f in facts:
        by_subject.setdefault(f.h, []).append(f)
    for f in facts:
        other = [g for g in by_subject[f.h] if g.fid != f.fid]
        for r in D.use_queries(f):
            if not r["use"].startswith(("U0", "U1", "U2")):
                continue
            rows.append(
                {
                    **r,
                    "split": f.split,
                    "exclude": S.exclusions(r, f),
                    "relation_text": D.RELATIONS[f.relation],
                    "other_v": other[0].v if other else None,
                }
            )
    cities = D.city_table()
    for h in sorted(by_subject):
        country = cities[h]["country"]
        natural = [
            ("country_rev", D.CAPITAL_REV[0].format(h=h), country, "country"),
            ("city_country", D.CITY_COUNTRY.format(v=h), country, "country"),
        ]
        cur = D.currency_of(country)
        if cur:
            natural += [
                (f"currency2.{i}", t.format(h=h), cur, "currency")
                for i, t in enumerate(D.CURRENCY_TWO_HOP)
            ]
        for name, prompt, gold, kind in natural:
            rows.append(
                {
                    "use": f"nat-{name}",
                    "fid": None,
                    "prompt": prompt + D.DIRECT,
                    "gold": gold,
                    "type": kind,
                    "mode": "direct",
                    "subject": h,
                    "exclude": [h],
                }
            )
    return rows


def e1_evaluate(model, tok, rows, patch):
    answered = answer_rows(model, tok, rows, patch)
    tf_rows = [r for r in rows if r["mode"] == "direct" and r["use"].startswith(("U0", "U1", "U2"))]
    scores = score_rows(model, tok, tf_rows, patch)
    tf = {(r["fid"], r["use"]): s for r, s in zip(tf_rows, scores, strict=True)}
    graders = V.graders()
    for r in answered:
        key = (r["fid"], r["use"])
        if r["mode"] == "direct" and key in tf:
            r["tf_greedy"] = tf[key]["greedy"]
            r["tf_logprob"] = tf[key]["logprob"]
        if r.get("other_v") and r["use"].startswith(("U0", "U1")):
            mentions = graders["city"].mentions(r["answer"])
            mentions = [m for m in mentions if m not in r.get("exclude", ())]
            r["mixed"] = bool(mentions) and mentions[0] == r["other_v"]
    return answered


def run_e1(model, tok, facts, layer, out_dir, log=print, variants=("plain", "relation", "control")):
    """Plain delta_h, the relation-key variant E1-R and the positive control E1-N at one layer."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main = [f for f in facts if f.split != "calibration"]
    subjects = sorted({f.h for f in main})
    index = {h: i for i, h in enumerate(subjects)}
    rels = sorted({D.RELATIONS[f.relation] for f in main})
    rel_index = {r: i for i, r in enumerate(rels)}
    natural = {h: natural_prompts(h) for h in subjects}
    for p in model.parameters():
        p.requires_grad_(False)

    # plain delta_h on all main facts
    path = out_dir / f"plain-l{layer}.json"
    if "plain" in variants and not path.exists():
        train = e1_items(tok, main, index)
        kl = kl_items(model, tok, natural, index)
        bounds = residual_norms(model, tok, train, layer, 0)
        rows = u0_check_rows(tok, main, index)
        dh, info = fit_writes(model, tok, layer, train, kl, len(subjects), bounds, rows)
        patch = Patch(layer, {h: dh[index[h]] for h in subjects})
        records = e1_evaluate(model, tok, e1_eval_rows(main), patch)
        torch.save({"subjects": subjects, "dh": dh.cpu()}, out_dir / f"plain-l{layer}.pt")
        S.write(
            path,
            {
                "variant": "plain",
                "layer": layer,
                "fit": info,
                "bounds": {subjects[k]: v for k, v in bounds.items()},
                "norms": {h: float(dh[index[h]].norm()) for h in subjects},
                "records": records,
            },
        )
        log(f"E1 plain layer {layer}: {info['steps']} steps, frozen {info['frozen_subjects']}")

    # E1-R: shared relation vectors learned on F_adapt, then delta_h refit on F_test
    path = out_dir / f"relation-l{layer}.json"
    if "relation" in variants and not path.exists():
        adapt = [f for f in main if f.split == "adapt"]
        test = [f for f in main if f.split == "test"]
        train = e1_items(tok, adapt, index, relation_text=True)
        kl = kl_items(model, tok, {h: natural[h] for h in {f.h for f in adapt}}, index)
        h_bounds = residual_norms(model, tok, train, layer, 0)
        r_bounds = residual_norms(model, tok, train, layer, 1)
        d = model.get_input_embeddings().weight.shape[1]
        dr = torch.zeros(len(rels), d, device="cuda", requires_grad=True)
        rows = u0_check_rows(tok, adapt, index, relation_text=True)
        dh_a, info_a = fit_writes(
            model,
            tok,
            layer,
            train,
            kl,
            len(subjects),
            h_bounds,
            rows,
            stop="global",
            dr=dr,
            r_bounds=r_bounds,
            rel_index=rel_index,
            train_dr=True,
        )
        dr_fixed = dr.detach()
        train = e1_items(tok, test, index, relation_text=True)
        kl = kl_items(model, tok, {h: natural[h] for h in {f.h for f in test}}, index)
        h_bounds = residual_norms(model, tok, train, layer, 0)
        rows = u0_check_rows(tok, test, index, relation_text=True)
        dh_t, info_t = fit_writes(
            model,
            tok,
            layer,
            train,
            kl,
            len(subjects),
            h_bounds,
            rows,
            dr=dr_fixed,
            rel_index=rel_index,
        )
        relation_vectors = {r: dr_fixed[rel_index[r]] for r in rels}
        test_subjects = sorted({f.h for f in test})
        adapt_subjects = sorted({f.h for f in adapt})
        records = {
            "test": e1_evaluate(
                model,
                tok,
                e1_eval_rows(test),
                Patch(layer, {h: dh_t[index[h]] for h in test_subjects}, relation_vectors),
            ),
            "adapt": e1_evaluate(
                model,
                tok,
                [r for r in e1_eval_rows(adapt) if r["fid"]],
                Patch(layer, {h: dh_a[index[h]] for h in adapt_subjects}, relation_vectors),
            ),
            # The relation vectors alone (no subject write) on F_test: what delta_R carries.
            "test_relation_only": e1_evaluate(
                model,
                tok,
                [r for r in e1_eval_rows(test) if r["fid"]],
                Patch(layer, {}, relation_vectors),
            ),
        }
        vectors = {
            "subjects": subjects,
            "relations": rels,
            "dr": dr_fixed.cpu(),
            "dh_adapt": dh_a.cpu(),
            "dh_test": dh_t.cpu(),
        }
        torch.save(vectors, out_dir / f"relation-l{layer}.pt")
        S.write(
            path,
            {
                "variant": "relation",
                "layer": layer,
                "fit_adapt": info_a,
                "fit_test": info_t,
                "relation_bounds": r_bounds,
                "relation_norms": {r: float(dr_fixed[rel_index[r]].norm()) for r in rels},
                "records": records,
            },
        )
        log(f"E1-R layer {layer}: adapt {info_a['steps']} steps, test {info_t['steps']} steps")

    # E1-N: counterfactual country C' written into delta_h (natural relation, positive control)
    path = out_dir / f"control-l{layer}.json"
    if "control" in variants and not path.exists():
        pairs = V.e1n_pairs(subjects)
        sub_n = sorted(pairs)
        idx_n = {h: i for i, h in enumerate(sub_n)}
        train, rows = [], []
        for h in sub_n:
            for k, (user, answer) in enumerate(V.E1N_TRAIN):
                user_f = user.format(h=h)
                text = M.chat(tok, user_f, False)
                item = tokenize(
                    tok,
                    text,
                    user_spans(text, user_f, [h, None]),
                    answer.format(h=h, C=pairs[h]) + "<|im_end|>",
                )
                item.update({"sub": idx_n[h], "rel": None, "fid": f"n/{h}", "template": k})
                train.append(item)
            user_f = V.E1N_TRAIN[0][0].format(h=h)
            text = M.chat(tok, user_f, False)
            item = tokenize(tok, text, user_spans(text, user_f, [h, None]), pairs[h])
            item.update({"sub": idx_n[h], "rel": None, "fid": f"n/{h}"})
            rows.append(item)
        kl = kl_items(model, tok, {h: [f"Tell me about {h}."] for h in sub_n}, idx_n)
        bounds = residual_norms(model, tok, train, layer, 0)
        dh, info = fit_writes(model, tok, layer, train, kl, len(sub_n), bounds, rows)
        cities = D.city_table()
        eval_rows = []
        for h in sub_n:
            new, old = pairs[h], cities[h]["country"]
            for i, t in enumerate(V.E1N_HELD_OUT):
                eval_rows.append(
                    {
                        "use": f"held.{i}",
                        "fid": h,
                        "prompt": t.format(h=h) + D.DIRECT,
                        "gold": new,
                        "old": old,
                        "type": "country",
                        "mode": "direct",
                        "subject": h,
                        "exclude": [h],
                    }
                )
            for i, t in enumerate(D.CURRENCY_TWO_HOP):
                eval_rows.append(
                    {
                        "use": f"currency2.{i}",
                        "fid": h,
                        "prompt": t.format(h=h) + D.DIRECT,
                        "gold": D.currency_of(new),
                        "old": D.currency_of(old),
                        "type": "currency",
                        "mode": "direct",
                        "subject": h,
                        "exclude": [h],
                    }
                )
            eval_rows.append(
                {
                    "use": "train0",
                    "fid": h,
                    "prompt": V.E1N_TRAIN[0][0].format(h=h),
                    "gold": new,
                    "old": old,
                    "type": "country",
                    "mode": "direct",
                    "subject": h,
                    "exclude": [h],
                }
            )
        patch = Patch(layer, {h: dh[idx_n[h]] for h in sub_n})
        records = answer_rows(model, tok, eval_rows, patch)
        base_records = answer_rows(model, tok, eval_rows)
        for rec in records + base_records:
            rec["old_hit"] = rec["first"] is not None and rec["first"] == rec["old"]
        S.write(
            path,
            {
                "variant": "control",
                "layer": layer,
                "pairs": pairs,
                "fit": info,
                "records": records,
                "base_records": base_records,
            },
        )
        log(f"E1-N layer {layer}: {info['steps']} steps, frozen {info['frozen_subjects']}")
