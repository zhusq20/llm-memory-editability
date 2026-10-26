"""write-mode-readout-v1 (zero training): where a write is read and in what form (G2), and at
which depth an in-context fact is read (G3). Protocol:
docs/development-artifacts/write-mode-readout-v1/protocol.md.

G2 compares, on the same prompt, the residual stream (decoder-layer outputs) of a written model
with a reference: the released model for parameter writes (M1/M2/M3/M5/M4 of
knowledge-read-interface-development-v1), or the same distractor context with the target fact
line swapped for another fact for the in-context write. Per position group and layer it reports
the relative change ||D|| / ||h_ref|| and two v-specific content scores: the cosine of D with the
entity form of v (v's own state when the model reads it) and with the answer form of v (final-norm
gain times v's unembedding row), each minus the mean over the other cities.

G3 knocks out attention from the question (or only the answer-predicting positions) to the target
fact line in sliding layer windows and records teacher-forced greedy correctness.
"""

from __future__ import annotations

import random

import torch

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_v2_data as V

G2_USES = ("U0", "U1.0", "U2.0", "U3.0", "shortcut")
G3_USES = ("U0", "U1.0", "U2.0", "U3.0", "U3.1", "shortcut")
GROUPS = ("subject", "country", "relation", "last", "other")
QUERY_SETS = ("question", "last")
CONTROL_SEED = 20261103


def main_facts(facts):
    return [f for f in facts if f.split != "calibration"]


def use_rows(fact, uses):
    rows = [r for r in D.use_queries(fact) if r["use"] in uses and r["mode"] == "direct"]
    assert len(rows) == len(uses), (fact.fid, [r["use"] for r in rows])
    return rows


def line(fact):
    return "Fact: " + D.context_statement(fact)


def control_systems(facts, systems, seed=CONTROL_SEED):
    """Same system prompt with the target line replaced by another main fact of the same relation
    (different subject and value, not already among the distractors)."""
    rng = random.Random(seed)
    main = sorted(main_facts(facts), key=lambda f: f.fid)
    by_id = {f.fid: f for f in main}
    out = {}
    for f in main:
        used = {f.fid, *systems[f.fid]["distractors"]}
        hs = {by_id[g].h for g in used} | {f.h}
        vs = {by_id[g].v for g in used} | {f.v}
        pool = [
            g
            for g in main
            if g.relation == f.relation and g.fid not in used and g.h not in hs and g.v not in vs
        ]
        g = rng.choice(pool)
        system = systems[f.fid]["system"]
        assert system.count(line(f)) == 1
        out[f.fid] = {"system": system.replace(line(f), line(g)), "replacement": g.fid}
    return out


def locate(text, row, fact, system_line=None):
    """Character spans in a chat-formatted text: user prompt, subject h, country C, relation
    name (all inside the user prompt) and, optionally, the target fact line in the system."""
    user_at = text.rfind(row["prompt"])
    assert user_at >= 0
    prompt = row["prompt"]
    spans = {"user": (user_at, user_at + len(prompt))}
    bridged = row["use"] in ("U3.0", "U3.1", "shortcut")
    for key, needle, present in (
        ("subject", fact.h, not bridged),
        ("country", fact.country_h, bridged),
        ("relation", D.RELATIONS[fact.relation], True),
    ):
        at = prompt.find(needle) if present else -1
        assert at >= 0 or not present, (row["use"], needle)
        spans[key] = (user_at + at, user_at + at + len(needle)) if at >= 0 else None
    if system_line is not None:
        at = text.find(system_line)
        assert 0 <= at < user_at
        spans["record"] = (at, at + len(system_line))
    return spans


def encode_rows(tok, texts, span_dicts, answers=None):
    """Left-padded batch; per row a dict of token positions (padded coordinates) per span."""
    keys = sorted({k for s in span_dicts for k in s})
    lists = []
    for s in span_dicts:
        lists.append([s[k] if s.get(k) else (0, 0) for k in keys])
    ids, mask, positions, slices = M.encode(tok, texts, lists, answers)
    out = []
    for s, pos in zip(span_dicts, positions, strict=True):
        out.append({k: (p if s.get(k) else []) for k, p in zip(keys, pos, strict=True)})
    return ids, mask, out, slices


def groups_of(pos, width):
    """Position groups for G2: subject/country/relation spans, the last prompt position, and the
    remaining question positions."""
    named = set(pos["subject"]) | set(pos["country"]) | set(pos["relation"])
    other = [p for p in pos["user"] if p not in named]
    return {
        "subject": pos["subject"],
        "country": pos["country"],
        "relation": pos["relation"],
        "last": [width - 1],
        "other": other,
    }


# ---------------------------------------------------------------- forward helpers


@torch.no_grad()
def layer_states(model, ids, mask):
    """(L, batch, seq, dim) decoder-layer outputs, float32, no autocast."""
    store = {}
    n = len(M.layers(model))
    with M.Interventions(model) as hooks:
        hooks.capture(range(n), store)
        model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
    return torch.stack([store[i] for i in range(n)])


@torch.no_grad()
def entity_vectors(model, tok, cities, batch=32):
    """Centered, unit entity-form (L, C, d) and answer-form (C, d) directions."""
    ent = []
    for i in range(0, len(cities), batch):
        chunk = cities[i : i + batch]
        texts = [M.chat(tok, f"Tell me about {c}.") for c in chunk]
        spans = [
            [M.find_span(t, c, t.rfind("Tell me about"))] for t, c in zip(texts, chunk, strict=True)
        ]
        ids, mask, pos, _ = M.encode(tok, texts, spans)
        states = layer_states(model, ids, mask)
        rows = torch.arange(len(chunk))
        last = torch.tensor([p[0][-1] for p in pos])
        ent.append(states[:, rows, last].float().cpu())
    ent = torch.cat(ent, 1)
    ent = ent - ent.mean(1, keepdim=True)
    d = ent.shape[-1]
    norm = model.model.norm
    gain = norm(torch.ones(1, 1, d, device=norm.weight.device)).float().reshape(d).cpu()
    first = [tok(c, add_special_tokens=False).input_ids[0] for c in cities]
    weight = model.lm_head.weight
    ans = weight[torch.tensor(first, device=weight.device)].float().cpu() * gain
    ans = ans - ans.mean(0, keepdim=True)
    return (
        torch.nn.functional.normalize(ent, dim=-1),
        torch.nn.functional.normalize(ans, dim=-1),
        first,
    )


def specific(cos, v):
    """cos[..., C]: cosine with v minus the mean cosine with the other cities."""
    others = (cos.sum(-1) - cos[..., v]) / (cos.shape[-1] - 1)
    return cos[..., v] - others


def group_measures(delta, ref, ent, ans, v):
    """delta, ref: (L, P, d) for one row and group. Returns (L,) relative change and v-specific
    entity / answer scores, averaged over the group's positions."""
    rel = (delta.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).mean(1)
    unit = torch.nn.functional.normalize(delta, dim=-1)
    cos_e = torch.einsum("lpd,lcd->lpc", unit, ent)
    cos_a = torch.einsum("lpd,cd->lpc", unit, ans)
    return rel, specific(cos_e, v).mean(1), specific(cos_a, v).mean(1)


@torch.no_grad()
def g2_compare(ref_model, new_model, tok, rows, city_index, ent, ans, batch=16):
    """rows: dicts with fact, row, ref_text, new_text. Returns per row {group: (3, L) array}."""
    ent_d = ent.cuda()
    ans_d = ans.cuda()
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        enc = {}
        for side, model in (("ref", ref_model), ("new", new_model)):
            texts = [r[f"{side}_text"] for r in chunk]
            spans = [locate(t, r["row"], r["fact"]) for t, r in zip(texts, chunk, strict=True)]
            ids, mask, pos, _ = encode_rows(tok, texts, spans)
            states = layer_states(model, ids, mask)
            enc[side] = (states, [groups_of(p, ids.shape[1]) for p in pos])
        for k, r in enumerate(chunk):
            v = city_index[r["fact"].v]
            result = {}
            for g in GROUPS:
                p_ref = enc["ref"][1][k][g]
                p_new = enc["new"][1][k][g]
                assert len(p_ref) == len(p_new), (r["fact"].fid, g)
                if not p_ref:
                    continue
                ref = enc["ref"][0][:, k, p_ref].float()
                new = enc["new"][0][:, k, p_new].float()
                rel, e, a = group_measures(new - ref, ref, ent_d, ans_d, v)
                result[g] = torch.stack([rel, e, a]).cpu().numpy()
            out.append(result)
        del enc
    return out


def reset_and_load(work, ref, state, previous):
    """Restore parameters touched by the previous method from the reference, then load state."""
    params = dict(work.named_parameters())
    ref_params = dict(ref.named_parameters())
    with torch.no_grad():
        for name in previous:
            params[name].copy_(ref_params[name])
        for name, value in state.items():
            if name not in params:
                assert "lm_head" in name, name
                continue
            params[name].copy_(value.to(params[name].device, params[name].dtype))
    return [n for n in state if n in params]


# ---------------------------------------------------------------- G3 knockout


def knockout_mask(mask, b, t, device, rows, queries, keys):
    """4D attention mask (bool keep-mask or additive float) with queries[k] x keys[k] of row
    rows[k] blocked in every head."""
    if mask is None:
        mask = torch.ones(t, t, dtype=torch.bool, device=device).tril()
        mask = mask[None, None].expand(b, 1, t, t)
    mask = mask.clone()
    blocked = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
    for r, q, k in zip(rows, queries, keys, strict=True):
        qi = torch.as_tensor(q, device=mask.device)[:, None]
        ki = torch.as_tensor(k, device=mask.device)[None, :]
        mask[r, :, qi, ki] = blocked
    return mask


def block_attention_many(hooks, layer_idx, rows, queries, keys):
    """Attention knockout for many query positions per row: queries[k] x keys[k] are blocked."""
    attn = M.layers(hooks.model)[layer_idx].self_attn

    def pre(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        b, t = hidden.shape[:2]
        kwargs["attention_mask"] = knockout_mask(
            kwargs.get("attention_mask"), b, t, hidden.device, rows, queries, keys
        )
        return args, kwargs

    hooks.handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))


@torch.no_grad()
def teacher_forced(model, ids, mask, slices):
    logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False).logits
    logp = torch.log_softmax(logits.float(), -1)
    out = []
    for i, (start, end) in enumerate(slices):
        target = ids[i, start:end].cuda()
        pred = logp[i, start - 1 : end - 1]
        out.append(
            (
                bool((pred.argmax(-1) == target).all()),
                float(pred.gather(-1, target[:, None]).sum()),
            )
        )
    return out


def g3_items(tok, facts, systems):
    items = []
    for f in main_facts(facts):
        system = systems[f.fid]["system"]
        for r in use_rows(f, G3_USES):
            text = M.chat(tok, r["prompt"], False, system)
            items.append(
                {
                    "fid": f.fid,
                    "use": r["use"],
                    "gold": r["gold"],
                    "text": text,
                    "spans": locate(text, r, f, line(f)),
                }
            )
    return items


def query_positions(pos, slices_i, width):
    start = min(pos["user"])
    last = slices_i[0] - 1
    return {"question": list(range(start, width)), "last": list(range(last, width))}


@torch.no_grad()
def mixer_means(model, tok, items, batch=32):
    """Per DeltaNet layer, the mean mixer input over all target-line tokens of all prompts."""
    sums = {}
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        ids, mask, pos, slices = encode_rows(
            tok, [x["text"] for x in chunk], [x["spans"] for x in chunk], [x["gold"] for x in chunk]
        )
        store = {}
        with M.Interventions(model) as hooks:
            hooks.capture_mixer_inputs(store)
            model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        for layer, (x,) in store.items():
            acc = sums.setdefault(layer, [0.0, 0])
            for k, p in enumerate(pos):
                acc[0] = acc[0] + x[k, p["record"]].float().sum(0)
                acc[1] += len(p["record"])
    return {k: v[0] / v[1] for k, v in sums.items()}


@torch.no_grad()
def g3_run(model, tok, items, window, log=print, batch=24):
    kinds = M.layer_kinds(model)
    means = mixer_means(model, tok, items) if "linear" in kinds else {}
    starts = list(range(0, len(kinds) - window + 1))
    records = []
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        ids, mask, pos, slices = encode_rows(
            tok, [x["text"] for x in chunk], [x["spans"] for x in chunk], [x["gold"] for x in chunk]
        )
        width = ids.shape[1]
        queries = [query_positions(p, s, width) for p, s in zip(pos, slices, strict=True)]
        keys = [p["record"] for p in pos]
        rows = list(range(len(chunk)))
        base = teacher_forced(model, ids, mask, slices)
        for x, (ok, lp) in zip(chunk, base, strict=True):
            records.append({"fid": x["fid"], "use": x["use"], "start": None, "query": None,
                            "greedy": ok, "logprob": lp})  # fmt: skip
        for qset in QUERY_SETS:
            qpos = [q[qset] for q in queries]
            for start in starts:
                with M.Interventions(model) as hooks:
                    for layer in range(start, start + window):
                        if kinds[layer] == "attention":
                            block_attention_many(hooks, layer, rows, qpos, keys)
                        else:
                            hooks.block_linear_mixer(layer, rows, qpos, keys, means[layer])
                    scores = teacher_forced(model, ids, mask, slices)
                for x, (ok, lp) in zip(chunk, scores, strict=True):
                    records.append({"fid": x["fid"], "use": x["use"], "start": start,
                                    "query": qset, "greedy": ok, "logprob": lp})  # fmt: skip
        log(f"G3 {i + len(chunk)}/{len(items)} prompts")
    return {"window": window, "starts": starts, "kinds": kinds, "records": records}


# ---------------------------------------------------------------- G2 row builders


def g2_param_rows(tok, facts):
    rows = []
    for f in main_facts(facts):
        for r in use_rows(f, G2_USES):
            text = M.chat(tok, r["prompt"], False)
            rows.append({"fact": f, "row": r, "ref_text": text, "new_text": text})
    return rows


def g2_context_rows(tok, facts, systems, controls):
    rows = []
    for f in main_facts(facts):
        for r in use_rows(f, G2_USES):
            rows.append(
                {
                    "fact": f,
                    "row": r,
                    "ref_text": M.chat(tok, r["prompt"], False, controls[f.fid]["system"]),
                    "new_text": M.chat(tok, r["prompt"], False, systems[f.fid]["system"]),
                }
            )
    return rows


def cities_of(facts):
    return sorted({f.h for f in facts} | {f.v for f in facts})


def distractor_systems(facts):
    return V.distractor_systems(facts)
