"""Bridge readout of knowledge-read-interface-followup-v1 (zero training).

In the U3 prompt ("What is the R of the capital of C?") the bridge entity h is never written. For
the released model and each injected model this module asks, layer by layer, what the state at the
bridge position (last token of C) carries and whether it can act as the key of the injected fact:

(a) decode: Patchscopes identity readout; the state replaces "x" in "cat -> cat ... x ->";
(b) swap: the state replaces the last subject token of another subject g (same relation) in the
    training-format question "What is the R of g?"; success = the model now gives h's value v;
(c) back: within the same U3 prompt, a later-layer state is written into an earlier layer at the
    bridge position or the final prompt position (Biran et al. back-patching).

The explicit-subject state of h from the U0 prompt is the reference for (a) and (b). All patches
replace the output of one decoder layer at one position, as in the development batch.
"""

from __future__ import annotations

import random
import re

import torch

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S

IDENTITY = "cat -> cat\n1135 -> 1135\nhello -> hello\nx ->"
DECODE_TOKENS = 8
SEED = 20261020
GENERIC = {"city", "town", "saint", "port", "san", "santo"}


def reset(work, base):
    with torch.no_grad():
        for (n, p), (m, q) in zip(work.named_parameters(), base.named_parameters(), strict=True):
            assert n == m
            p.copy_(q)


def load_injected(work, path):
    """Copy a saved injected state (full or partial) into `work`, which holds released weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(work.named_parameters())
    unknown = [k for k in state if k not in params]
    assert not unknown, unknown[:5]
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].device, params[k].dtype))
    return len(state)


def partners(facts, seed=SEED):
    """Swap target for each fact: another fact with the same relation whose subject, country and
    value all differ (so the unpatched answer is never h's value)."""
    rng = random.Random(seed)
    out = {}
    for f in sorted(facts, key=lambda x: x.fid):
        options = [
            g
            for g in sorted(facts, key=lambda x: x.fid)
            if g.relation == f.relation
            and g.h != f.h
            and g.country_h != f.country_h
            and g.v != f.v
            and f.v not in (g.h, g.v)
            and g.v != f.h
        ]
        out[f.fid] = rng.choice(options).fid if options else None
    return out


def ambiguous(fact):
    """Subject and country share a name (Kuwait City / Kuwait); decode cannot separate them."""
    h, c = D.norm(fact.h), D.norm(fact.country_h)
    return h in c or c in h


def name_hit(line, name, grader):
    if name in grader.mentions(line):
        return True
    last = D.norm(name).split()[-1]
    if len(last) < 4 or last in GENERIC:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(last) + r"(?![a-z0-9])", D.norm(line)) is not None


def bridge_rows(facts):
    return [
        {"fid": f.fid, "use": f"U3.{i}", "prompt": D.fill(t, f) + D.DIRECT, "subject": f.country_h}
        for f in facts
        for i, t in enumerate(D.U3)
    ]


def explicit_rows(facts):
    return [
        {"fid": f.fid, "use": "U0", "prompt": D.fill(D.U0, f) + D.DIRECT, "subject": f.h}
        for f in facts
    ]


@torch.no_grad()
def states(model, tok, rows, n_layers):
    """(n_rows, n_layers, dim) layer outputs at the last subject token and at the final prompt
    token of each row's chat-formatted prompt."""
    subject, last = [], []
    for chunk in S.batches(rows):
        texts, spans = S.subject_spans(tok, chunk)
        ids, mask, pos, _ = M.encode(tok, texts, spans)
        store = {}
        with M.Interventions(model) as hooks, M.autocast():
            hooks.capture(range(n_layers), store)
            model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        idx = torch.arange(len(chunk))
        where = torch.tensor([p[0][-1] for p in pos])
        stack = torch.stack([store[layer].float() for layer in range(n_layers)], 1)
        subject.append(stack[idx, :, where])
        last.append(stack[:, :, -1])
    return torch.cat(subject), torch.cat(last)


@torch.no_grad()
def decode(model, tok, vectors, n_layers, size=64):
    """Identity readout of each vector at each layer; returns lines[layer][row]."""
    span = M.find_span(IDENTITY, "x", IDENTITY.rfind("\n"))
    out = {}
    for layer in range(n_layers):
        lines = []
        for i in range(0, len(vectors), size):
            chunk = vectors[i : i + size, layer]
            n = len(chunk)
            ids, mask, pos, _ = M.encode(tok, [IDENTITY] * n, [[span]] * n)
            where = [p[0][-1] for p in pos]
            with M.Interventions(model) as hooks:
                hooks.overwrite(layer, list(range(n)), where, chunk)
                replies = M.generate(model, tok, ids, mask, DECODE_TOKENS)
            lines += [r.strip().split("\n")[0] for r in replies]
        out[layer] = lines
    return out


@torch.no_grad()
def swap(model, tok, rows, vectors, by_id, partner, n_layers):
    """Teacher-forced score of h's value v after the vector replaces the last subject token of
    the partner g in "What is the R of g?". Also the unpatched scores of v and of g's value."""
    records = []
    keep = [i for i, r in enumerate(rows) if partner.get(r["fid"])]
    for start in range(0, len(keep), S.BATCH):
        members = keep[start : start + S.BATCH]
        targets = []
        for i in members:
            g = by_id[partner[rows[i]["fid"]]]
            targets.append({"prompt": D.fill(D.U0, g) + D.DIRECT, "subject": g.h})
        texts, spans = S.subject_spans(tok, targets)
        gold = [by_id[rows[i]["fid"]].v for i in members]
        own = [by_id[partner[rows[i]["fid"]]].v for i in members]
        ids, mask, pos, slices = M.encode(tok, texts, spans, answers=gold)
        where = [p[0][-1] for p in pos]
        plain = M.answer_scores(model, ids, mask, slices)
        o_ids, o_mask, _, o_slices = M.encode(tok, texts, spans, answers=own)
        plain_own = M.answer_scores(model, o_ids, o_mask, o_slices)
        for k, i in enumerate(members):
            records.append(
                {
                    "fid": rows[i]["fid"],
                    "use": rows[i]["use"],
                    "partner": partner[rows[i]["fid"]],
                    "layer": None,
                    "greedy": plain[k]["greedy"],
                    "logprob": plain[k]["logprob"],
                    "partner_greedy": plain_own[k]["greedy"],
                }
            )
        for layer in range(n_layers):
            with M.Interventions(model) as hooks:
                hooks.overwrite(layer, list(range(len(members))), where, vectors[members, layer])
                scores = M.answer_scores(model, ids, mask, slices)
            for k, i in enumerate(members):
                records.append(
                    {
                        "fid": rows[i]["fid"],
                        "use": rows[i]["use"],
                        "partner": partner[rows[i]["fid"]],
                        "layer": layer,
                        "greedy": scores[k]["greedy"],
                        "logprob": scores[k]["logprob"],
                    }
                )
    return records


@torch.no_grad()
def back(model, tok, rows, by_id, n_layers):
    """Back-patching in the U3 prompt: the state at layer s goes into layer t < s at the bridge
    position or the final prompt position; teacher-forced score of h's value v."""
    pairs = [(s, t) for s, t in S.layer_pairs(n_layers) if s > t]
    records = []
    for chunk in S.batches(rows):
        texts, spans = S.subject_spans(tok, chunk)
        gold = [by_id[r["fid"]].v for r in chunk]
        ids, mask, pos, slices = M.encode(tok, texts, spans, answers=gold)
        store = {}
        with M.Interventions(model) as hooks, M.autocast():
            hooks.capture(range(n_layers), store)
            model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
        own = M.answer_scores(model, ids, mask, slices)
        for r, sc in zip(chunk, own, strict=True):
            records.append({"fid": r["fid"], "use": r["use"], "where": None, **sc})
        idx = list(range(len(chunk)))
        places = {
            "bridge": [p[0][-1] for p in pos],
            "last": [a - 1 for a, _ in slices],
        }
        for where, positions in places.items():
            for s, t in pairs:
                vectors = store[s][idx, positions]
                with M.Interventions(model) as hooks:
                    hooks.overwrite(t, idx, positions, vectors)
                    scores = M.answer_scores(model, ids, mask, slices)
                for r, sc in zip(chunk, scores, strict=True):
                    records.append(
                        {"fid": r["fid"], "use": r["use"], "where": where, "src": s, "tgt": t, **sc}
                    )
    return records


def readout(model, tok, facts, with_fact=True):
    """All three readouts for one model. `with_fact=False` (released model) runs decode only."""
    n_layers = len(M.layers(model))
    by_id = {f.fid: f for f in facts}
    bridge = bridge_rows(facts)
    explicit = explicit_rows(facts)
    b_subject, b_last = states(model, tok, bridge, n_layers)
    e_subject, _ = states(model, tok, explicit, n_layers)
    graders = D.graders()
    decoded = []
    for source, rows, vectors in (
        ("bridge", bridge, b_subject),
        ("last", bridge, b_last),
        ("explicit", explicit, e_subject),
    ):
        lines = decode(model, tok, vectors, n_layers)
        for layer, texts in lines.items():
            for r, line in zip(rows, texts, strict=True):
                f = by_id[r["fid"]]
                decoded.append(
                    {
                        "fid": r["fid"],
                        "use": r["use"],
                        "source": source,
                        "layer": layer,
                        "line": line,
                        "h": name_hit(line, f.h, graders["city"]),
                        "C": name_hit(line, f.country_h, graders["country"]),
                        "v": name_hit(line, f.v, graders["city"]),
                    }
                )
    out = {"n_layers": n_layers, "decode": decoded}
    if with_fact:
        partner = partners(facts)
        out["partners"] = partner
        out["swap"] = {
            "bridge": swap(model, tok, bridge, b_subject, by_id, partner, n_layers),
            "explicit": swap(model, tok, explicit, e_subject, by_id, partner, n_layers),
        }
        out["back"] = back(model, tok, bridge, by_id, n_layers)
    return out
