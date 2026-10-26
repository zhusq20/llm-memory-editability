"""knowledge-read-interface-v2 E3: write form at matched write success.

Fifty facts are written one at a time into new parameters only, following the continual-write
protocol of Cortex (arXiv 2610.20399): each update distils the released model with the fact in
its system prompt (output KL at temperature 2 plus hidden-state alignment after every decoder
layer, both on answer tokens), replaying earlier facts' questions and general questions.

- structural: every MLP gets one new SwiGLU channel per fact, y += w * silu(g.x) * (u.x);
- synaptic: every MLP gets a rank-one increment per fact on gate, up and down projections.

Each fact owns its parameters, so its key (when it fires) and value (what it writes) can be read
out directly. Earlier facts stay frozen.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
from torch import nn

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S
from llm_memory_editability import read_interface_v2 as V2

SEED = 20261023
N_FACTS = 50
EPOCHS = 30
BATCH = 16
HISTORY = 50
GENERAL = 16
TEMPERATURE = 2.0
KL_WEIGHT = 1.0
MSE_WEIGHT = 1.0
LRS = [1.5e-4, 5e-4, 1.5e-3]
KEYS = {"structural": ("g", "u", "w"), "synaptic": ("ag", "bg", "au", "bu", "ad", "bd")}


class Adapted(nn.Module):
    """An MLP plus per-fact additions; `mask` (batch, K) scales each fact's addition per row."""

    def __init__(self, mlp, kind):
        super().__init__()
        self.mlp = mlp
        self.kind = kind
        self.frozen = []
        self.current = None
        self.mask = None
        self.record = None

    def begin(self, std, generator):
        d = self.mlp.gate_proj.weight.shape[1]
        inter = self.mlp.gate_proj.weight.shape[0]
        dev = self.mlp.gate_proj.weight.device

        def rand(n):
            return (torch.randn(n, generator=generator) * std).to(dev)

        if self.kind == "structural":
            init = {"g": rand(d), "u": rand(d), "w": torch.zeros(d, device=dev)}
        else:
            init = {
                "ag": rand(d),
                "bg": torch.zeros(inter, device=dev),
                "au": rand(d),
                "bu": torch.zeros(inter, device=dev),
                "ad": rand(inter) * (d / inter) ** 0.5,
                "bd": torch.zeros(d, device=dev),
            }
        self.current = nn.ParameterDict({k: nn.Parameter(v) for k, v in init.items()})
        return list(self.current.values())

    def commit(self):
        self.frozen.append({k: v.detach().clone() for k, v in self.current.items()})
        self.current = None

    def stacked(self):
        items = self.frozen + ([self.current] if self.current is not None else [])
        if not items:
            return None
        return {k: torch.stack([it[k] for it in items]) for k in KEYS[self.kind]}

    def forward(self, x):
        if self.record is not None:
            self.record.append(x.detach())
        F = self.stacked()
        if F is None:
            return self.mlp(x)
        mask = None if self.mask is None else self.mask.to(x.device, x.dtype)[:, None, :]
        mlp = self.mlp
        if self.kind == "structural":
            y = mlp(x)
            a = mlp.act_fn(x @ F["g"].T.to(x.dtype)) * (x @ F["u"].T.to(x.dtype))
            if mask is not None:
                a = a * mask
            return y + a @ F["w"].to(a.dtype)
        cg = x @ F["ag"].T.to(x.dtype)
        cu = x @ F["au"].T.to(x.dtype)
        if mask is not None:
            cg, cu = cg * mask, cu * mask
        gate = mlp.gate_proj(x) + cg @ F["bg"].to(cg.dtype)
        up = mlp.up_proj(x) + cu @ F["bu"].to(cu.dtype)
        h = mlp.act_fn(gate) * up
        cd = h @ F["ad"].T.to(h.dtype)
        if mask is not None:
            cd = cd * mask
        return mlp.down_proj(h) + cd @ F["bd"].to(cd.dtype)


def install(model, kind):
    wrappers = []
    for layer in M.layers(model):
        if not isinstance(layer.mlp, Adapted):
            layer.mlp = Adapted(layer.mlp, kind)
        wrappers.append(layer.mlp)
    return wrappers


def uninstall(model):
    for layer in M.layers(model):
        if isinstance(layer.mlp, Adapted):
            layer.mlp = layer.mlp.mlp


def set_masks(wrappers, masks):
    for w, m in zip(wrappers, masks, strict=True):
        w.mask = m


# ---------------------------------------------------------------- distillation targets


class Teacher:
    """Cached released-model targets on answer tokens: log-probs at temperature T at the
    positions predicting the answer, and every layer's output at the answer tokens."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.n_layers = len(M.layers(model))
        self.cache = {}

    def key(self, ex):
        return (ex.get("system"), ex["user"], ex["assistant"])

    @torch.no_grad()
    def fill(self, examples):
        todo = [ex for ex in examples if self.key(ex) not in self.cache]
        for chunk in S.batches(todo, 32):
            items = []
            for ex in chunk:
                text = M.chat(self.tok, ex["user"], False, ex.get("system"))
                items.append(V2.tokenize(self.tok, text, [], ex["assistant"] + "<|im_end|>"))
            ids, mask, _, slices = V2.collate(self.tok, items)
            store = {}
            with M.Interventions(self.model) as hooks, M.autocast():
                hooks.capture(range(self.n_layers), store)
                logits, _ = V2.answer_logits(self.model, ids.cuda(), mask.cuda(), slices)
            logp = torch.log_softmax(logits / TEMPERATURE, -1).to(torch.bfloat16)
            at = 0
            for i, (ex, (a, b)) in enumerate(zip(chunk, slices, strict=True)):
                n = b - a
                hidden = torch.stack([store[layer][i, a:b] for layer in range(self.n_layers)])
                self.cache[self.key(ex)] = {
                    "logp": logp[at : at + n],
                    "hidden": hidden.to(torch.bfloat16),
                    "n": n,
                }
                at += n

    def get(self, ex):
        return self.cache[self.key(ex)]


def distil_loss(student, tok, teacher, chunk, n_layers):
    """KL(teacher || student) at temperature T (times T^2) per answer token plus the mean over
    layers of the relative squared error of layer outputs on answer tokens."""
    items = []
    for ex in chunk:
        text = M.chat(tok, ex["user"], False)
        items.append(V2.tokenize(tok, text, [], ex["assistant"] + "<|im_end|>"))
    ids, mask, _, slices = V2.collate(tok, items)
    store = {}
    handles = []
    for idx, layer in enumerate(M.layers(student)):

        def hook(module, args, output, idx=idx):
            store[idx] = output

        handles.append(layer.register_forward_hook(hook))
    try:
        with M.autocast():
            logits, _ = V2.answer_logits(student, ids.cuda(), mask.cuda(), slices)
    finally:
        for h in handles:
            h.remove()
    targets = [teacher.get(ex) for ex in chunk]
    for t, (a, b) in zip(targets, slices, strict=True):
        assert t["n"] == b - a
    t_logp = torch.cat([t["logp"] for t in targets]).float()
    s_logp = torch.log_softmax(logits / TEMPERATURE, -1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).mean() * TEMPERATURE**2
    rows = torch.cat([torch.full((b - a,), i) for i, (a, b) in enumerate(slices)])
    pos = torch.cat([torch.arange(a, b) for a, b in slices])
    mse = 0.0
    for layer in range(n_layers):
        s_h = store[layer][rows, pos].float()
        t_h = torch.cat([t["hidden"][layer] for t in targets]).float()
        mse = mse + ((s_h - t_h) ** 2).sum() / (t_h**2).sum().clamp_min(1e-6)
    mse = mse / n_layers
    return KL_WEIGHT * kl + MSE_WEIGHT * mse, float(kl), float(mse)


def fact_examples(replies, fact):
    """The 24 M2 prompts of a fact with the teacher's own greedy replies (M3 teacher)."""
    system = "Fact: " + D.context_statement(fact)
    users = [u for u, _ in D.training_examples(fact, "m2")]
    by_user = {r["user"]: r for r in replies if r.get("system") == system}
    return [{"user": u, "assistant": by_user[u]["assistant"], "system": system} for u in users]


def write_fact(student, tok, teacher, wrappers, kind, lr, current, history, general, rng):
    """One update: new parameters for this fact in every layer, EPOCHS passes over the fact's
    examples plus sampled history and general replay; earlier facts stay frozen."""
    gen = torch.Generator().manual_seed(rng.randrange(2**31))
    params = []
    for w in wrappers:
        std = float(w.mlp.gate_proj.weight.float().std())
        params += w.begin(std, gen)
    data = list(current)
    data += rng.sample(history, min(HISTORY, len(history)))
    data += [{**g, "system": None} for g in rng.sample(general, min(GENERAL, len(general)))]
    teacher.fill(data)
    opt = torch.optim.Adam(params, lr=lr)
    n_layers = len(wrappers)
    losses = []
    for _ in range(EPOCHS):
        rng.shuffle(data)
        for chunk in S.batches(data, BATCH):
            loss, kl, mse = distil_loss(student, tok, teacher, chunk, n_layers)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append((kl, mse))
    for w in wrappers:
        w.commit()
    return {
        "kl_first": losses[0][0],
        "mse_first": losses[0][1],
        "kl_last": losses[-1][0],
        "mse_last": losses[-1][1],
        "steps": len(losses),
    }


def write_sequence(
    student,
    tok,
    teacher,
    wrappers,
    kind,
    lr,
    facts,
    replies,
    general,
    log=None,
    after=None,
    state=None,
):
    """Write facts in order. `after(i, fact)` runs after each write (evaluation); `state` resumes
    from saved frozen parameters and records."""
    rng = random.Random(SEED)
    history = []
    reports = []
    start = 0
    if state and state.get("reports"):
        for w, frozen in zip(wrappers, state["frozen"], strict=True):
            w.frozen = [{k: v.cuda() for k, v in f.items()} for f in frozen]
        reports = state["reports"]
        start = len(reports)
        rng.setstate(state["rng"])
        for f in facts[:start]:
            history += fact_examples(replies, f)
    for i in range(start, len(facts)):
        f = facts[i]
        current = fact_examples(replies, f)
        t0 = time.time()
        info = write_fact(student, tok, teacher, wrappers, kind, lr, current, history, general, rng)
        info.update({"fid": f.fid, "seconds": time.time() - t0})
        history += current
        if after:
            info.update(after(i, f))
        reports.append(info)
        if log:
            log(f"E3 {kind} lr {lr:g} fact {i + 1}/{len(facts)} {f.fid}: {info}")
        if state is not None:
            state.update(
                {
                    "frozen": [
                        [{k: v.cpu() for k, v in fr.items()} for fr in w.frozen] for w in wrappers
                    ],
                    "reports": reports,
                    "rng": rng.getstate(),
                }
            )
            state["save"](state)
    return reports


# ---------------------------------------------------------------- evaluation


def fact_uses(fact):
    return [
        {**r, "split": fact.split, "exclude": S.exclusions(r, fact)} for r in D.use_queries(fact)
    ]


def u0_rows(facts):
    return [{"fid": f.fid, "prompt": D.fill(D.U0, f) + D.DIRECT, "gold": f.v} for f in facts]


def immediate(student, tok, fact, written):
    """Right after writing `fact`: all its uses, and teacher-forced U0 of every fact so far."""
    answered = V2.answer_rows(student, tok, fact_uses(fact))
    tf = V2.score_rows(student, tok, u0_rows(written))
    return {"uses": answered, "u0_written": sum(s["greedy"] for s in tf) / len(tf)}


def retention_rows(run, facts, n=300, seed=SEED):
    """Released-model-correct natural questions that mention none of the written subjects or
    values; teacher-forced and generated."""
    screening = S.read(Path(run) / "screening.json")["records"]
    names = {f.h for f in facts} | {f.v for f in facts}
    pool = [r for r in screening if r["strict"] and not any(name in r["prompt"] for name in names)]
    rng = random.Random(seed)
    rows = rng.sample(pool, min(n, len(pool)))
    return [
        {
            "qid": r["qid"],
            "prompt": r["prompt"],
            "gold": r["gold"],
            "type": r["type"],
            "mode": "direct",
            "exclude": [],
        }
        for r in rows
    ]


def calibration_keep_rows(run, facts):
    """For learning-rate calibration: natural facts of the calibration subjects and values plus
    100 unrelated questions, all answered correctly by the released model (teacher-forced)."""
    subjects = sorted({f.h for f in facts})
    values = sorted({f.v for f in facts})
    natural = D.retention_queries(subjects, values, facts)
    unrelated = retention_rows(run, facts, n=100)
    rows = [{"prompt": q["prompt"], "gold": q["gold"]} for q in natural]
    rows += [{"prompt": q["prompt"], "gold": q["gold"]} for q in unrelated]
    return rows


def calibrate(base, student, tok, teacher, kind, facts, replies, general, keep_rows, log=None):
    """Pick the learning rate by U0 write plus kept natural facts after writing the calibration
    facts (ties: the smaller rate)."""
    base_keep = V2.score_rows(base, tok, keep_rows)
    known = [r for r, s in zip(keep_rows, base_keep, strict=True) if s["greedy"]]
    sweep = []
    for lr in LRS:
        wrappers = install(student, kind)
        for w in wrappers:
            w.frozen, w.current = [], None
        reports = write_sequence(student, tok, teacher, wrappers, kind, lr, facts, replies, general)
        u0 = V2.score_rows(student, tok, u0_rows(facts))
        keep = V2.score_rows(student, tok, known)
        row = {
            "lr": lr,
            "u0": sum(s["greedy"] for s in u0) / len(u0),
            "keep": sum(s["greedy"] for s in keep) / max(1, len(keep)),
            "seconds": sum(r["seconds"] for r in reports),
            "reports": reports,
        }
        row["score"] = row["u0"] + row["keep"]
        sweep.append(row)
        if log:
            log(f"E3 calibration {kind} lr {lr:g}: u0 {row['u0']:.2f} keep {row['keep']:.2f}")
        for w in wrappers:
            w.frozen = []
    best = max(sweep, key=lambda r: (r["score"], -r["lr"]))
    return best["lr"], sweep


# ---------------------------------------------------------------- mechanism readouts


def entity_directions(base, tok, layers, cities):
    """Per layer: mean last-token state of each city over neutral contexts, minus the mean over
    all cities."""
    templates = ["{v}", "The city of {v}", "We traveled to {v}", "News from {v}"]
    texts, spans, owners = [], [], []
    for v in cities:
        for t in templates:
            text = t.format(v=v)
            at = text.find(v)
            texts.append(text)
            spans.append([(at, at + len(v))])
            owners.append(v)
    sums = {}
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            part = zip(texts[start : start + 64], spans[start : start + 64], strict=True)
            items = [V2.tokenize(tok, x, s) for x, s in part]
            ids, mask, positions, _ = V2.collate(tok, items)
            store = {}
            with M.Interventions(base) as hooks, M.autocast():
                hooks.capture(layers, store)
                base.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
            for i, owner in enumerate(owners[start : start + 64]):
                p = positions[i][0][-1]
                for layer in layers:
                    acc = sums.setdefault((owner, layer), [0.0, 0])
                    acc[0] = acc[0] + store[layer][i, p].float()
                    acc[1] += 1
    out = {}
    for layer in layers:
        means = torch.stack([sums[(v, layer)][0] / sums[(v, layer)][1] for v in cities])
        out[layer] = means - means.mean(0, keepdim=True)
    return out


def first_tokens(tok, name):
    return sorted(
        {
            tok(name, add_special_tokens=False).input_ids[0],
            tok(" " + name, add_special_tokens=False).input_ids[0],
        }
    )


@torch.no_grad()
def logit_rank(model, vector, token_ids):
    logits = model.lm_head(model.model.norm(vector[None].to(model.lm_head.weight.dtype)))[0]
    order = logits.float().argsort(descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order), device=order.device)
    return int(min(ranks[t] for t in token_ids))


@torch.no_grad()
def contributions(w, x, ks, vectors=False):
    """Norm (or vector) of fact k's addition to MLP output at inputs x (N, d) for ks (N,)."""
    F = w.stacked()
    mlp = w.mlp
    x = x.float()
    out = []
    for s in range(0, len(x), 2048):
        xs, kk = x[s : s + 2048], ks[s : s + 2048]
        if w.kind == "structural":
            g, u, wv = F["g"][kk].float(), F["u"][kk].float(), F["w"][kk].float()
            a = mlp.act_fn((xs * g).sum(-1)) * (xs * u).sum(-1)
            delta = a[:, None] * wv
        else:
            ag, bg = F["ag"].float(), F["bg"].float()
            au, bu = F["au"].float(), F["bu"].float()
            ad, bd = F["ad"].float(), F["bd"].float()
            wg, wu, wd = (
                mlp.gate_proj.weight.float(),
                mlp.up_proj.weight.float(),
                mlp.down_proj.weight.float(),
            )
            cg, cu = xs @ ag.T, xs @ au.T
            g0, u0 = xs @ wg.T + cg @ bg, xs @ wu.T + cu @ bu
            idx = torch.arange(len(xs), device=xs.device)
            gk = g0 - cg[idx, kk][:, None] * bg[kk]
            uk = u0 - cu[idx, kk][:, None] * bu[kk]
            m0 = mlp.act_fn(g0) * u0
            mk = mlp.act_fn(gk) * uk
            cd0 = m0 @ ad.T
            cdk = mk @ ad.T
            cdk[idx, kk] = 0.0
            delta = (m0 - mk) @ wd.T + cd0 @ bd - cdk @ bd
        out.append(delta if vectors else delta.norm(dim=-1))
    return torch.cat(out)


@torch.no_grad()
def mlp_inputs(model, tok, wrappers, rows, names_of):
    """Run rows (chat prompts) and return per row: token positions of named spans and the MLP
    inputs at every layer (list over layers of (seq, d) tensors, padding removed)."""
    out = []
    for chunk in S.batches(rows, 32):
        items = []
        for r in chunk:
            text = M.chat(tok, r["prompt"], False)
            items.append(V2.tokenize(tok, text, V2.user_spans(text, r["prompt"], names_of(r))))
        ids, mask, positions, _ = V2.collate(tok, items)
        for w in wrappers:
            w.record = []
        try:
            with M.autocast():
                model.model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        finally:
            recorded = [w.record[0] for w in wrappers]
            for w in wrappers:
                w.record = None
        width = ids.shape[1]
        for i, it in enumerate(items):
            pad = width - len(it["ids"])
            out.append(
                {
                    "positions": [[p - pad for p in ps] for ps in positions[i]],
                    "x": [rec[i, pad:] for rec in recorded],
                }
            )
    return out


def mechanism(base, student, tok, wrappers, facts, unrelated, n_unrelated=100):
    """(a) where each fact's addition fires, (b) what its value direction encodes, (c) the
    effect of removing it in early/middle/late layers on U0 and U2."""
    n_layers = len(wrappers)
    k_of = {f.fid: k for k, f in enumerate(facts)}
    rows = []
    for f in facts:
        for r in D.use_queries(f):
            if r["mode"] != "direct":
                continue
            rows.append({**r, "fact": f, "relation_text": D.RELATIONS[f.relation]})
    marks = lambda r: [r["fact"].h, r["relation_text"], r["fact"].country_h]  # noqa: E731
    inputs = mlp_inputs(student, tok, wrappers, rows, marks)
    trigger = []
    realized = {}
    for r, rec in zip(rows, inputs, strict=True):
        k = k_of[r["fid"]]
        n = len(rec["x"][0])
        h_pos, rel_pos, c_pos = rec["positions"]
        entry = {"fid": r["fid"], "use": r["use"], "layers": []}
        for layer in range(n_layers):
            x = rec["x"][layer]
            ks = torch.full((n,), k, device=x.device)
            norms = contributions(wrappers[layer], x, ks)
            entry["layers"].append(
                {
                    "h": float(norms[h_pos[-1]]) if h_pos else None,
                    "relation": float(norms[rel_pos].max()) if rel_pos else None,
                    "country": float(norms[c_pos[-1]]) if c_pos else None,
                    "last": float(norms[-1]),
                    "max": float(norms.max()),
                }
            )
            if r["use"] == "U0" and h_pos:
                vec = contributions(wrappers[layer], x[h_pos[-1]][None], ks[:1], vectors=True)[0]
                realized[(r["fid"], layer)] = vec
        trigger.append(entry)
    # false triggering: each fact's largest addition on unrelated prompts
    extra = mlp_inputs(student, tok, wrappers, unrelated[:n_unrelated], lambda r: [])
    false = torch.zeros(len(facts), n_layers)
    for rec in extra:
        for layer in range(n_layers):
            x = rec["x"][layer]
            n = len(x)
            xs = x.repeat(len(facts), 1)
            ks = torch.arange(len(facts), device=x.device).repeat_interleave(n)
            norms = contributions(wrappers[layer], xs, ks).view(len(facts), n)
            false[:, layer] += norms.max(-1).values.cpu() / len(extra)
    # value form
    cities = sorted(D.city_table())
    city_index = {c: i for i, c in enumerate(cities)}
    directions = entity_directions(base, tok, list(range(n_layers)), cities)
    value = []
    for f in facts:
        k = k_of[f.fid]
        own_tokens = first_tokens(tok, f.v)
        for layer in range(n_layers):
            F = wrappers[layer].stacked()
            param_vec = (F["w"] if wrappers[layer].kind == "structural" else F["bd"])[k].float()
            e = directions[layer]
            for name, vec in (("param", param_vec), ("realized", realized.get((f.fid, layer)))):
                if vec is None or float(vec.norm()) == 0.0:
                    continue
                cos = torch.nn.functional.cosine_similarity(vec[None], e, dim=-1)
                own = float(cos[city_index[f.v]])
                others = torch.cat([cos[: city_index[f.v]], cos[city_index[f.v] + 1 :]])
                value.append(
                    {
                        "fid": f.fid,
                        "layer": layer,
                        "vector": name,
                        "norm": float(vec.norm()),
                        "cos_v": own,
                        "cos_other_mean": float(others.mean()),
                        "cos_rank": int((others > own).sum()),
                        "logit_rank_v": logit_rank(student, vec, own_tokens),
                        "logit_rank_h": logit_rank(student, vec, first_tokens(tok, f.h)),
                    }
                )
    causal = ablation(student, tok, wrappers, facts, k_of)
    return {"trigger": trigger, "false_trigger": false.tolist(), "value": value, "causal": causal}


def ablation(student, tok, wrappers, facts, k_of):
    """Remove each fact's own additions in one third of the layers at a time."""
    n_layers = len(wrappers)
    thirds = {
        "early": range(0, n_layers // 3),
        "middle": range(n_layers // 3, 2 * n_layers // 3),
        "late": range(2 * n_layers // 3, n_layers),
        "none": range(0),
    }
    K = len(facts)
    rows = []
    for f in facts:
        rows += [
            r for r in fact_uses(f) if r["mode"] == "direct" and r["use"].startswith(("U0", "U2"))
        ]
    out = []
    for name, band in thirds.items():
        for chunk in S.batches(rows, 32):
            mask = torch.ones(len(chunk), K)
            for i, r in enumerate(chunk):
                mask[i, k_of[r["fid"]]] = 0.0
            set_masks(wrappers, [mask if layer in band else None for layer in range(n_layers)])
            try:
                answered = V2.answer_rows(student, tok, chunk)
                tf = V2.score_rows(student, tok, [r for r in chunk])
            finally:
                set_masks(wrappers, [None] * n_layers)
            for r, s in zip(answered, tf, strict=True):
                out.append(
                    {
                        "band": name,
                        "fid": r["fid"],
                        "use": r["use"],
                        "answer": r["answer"],
                        "strict": r["strict"],
                        "first": r["first"],
                        "tf_greedy": s["greedy"],
                    }
                )
    return out


def transplant_same_layer(base, injected, tok, facts):
    n_layers = len(M.layers(base))
    rows = []
    for f in facts:
        for r in D.use_queries(f):
            if r["mode"] == "direct" and r["subject"] and r["use"] != "shortcut":
                rows.append({**r, "split": f.split})
    same = [(i, i) for i in range(0, n_layers, 2)] + [(n_layers - 1, n_layers - 1)]
    with torch.no_grad():
        return S.patch_rows(rows, base, injected, tok, {"forward": same, "reverse": same}, n_layers)
