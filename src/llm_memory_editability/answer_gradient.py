"""answer-gradient-address-v1: where one answer-only gradient step writes a new fact (no training).

For a new fact "h's R city is v" and a prompt/reply pair formatted exactly as M2 training formats it
(chat template with thinking disabled, reply + <|im_end|>, summed cross entropy over the reply
tokens), the gradient on the down projection of MLP l at the released weights is

    dL/dW_l = sum_p delta_{l,p} k_{l,p}^T,

with k_{l,p} the down_proj input at position p (the address) and delta_{l,p} the gradient of the
loss with respect to the down_proj output (the content). The MLP output is added straight to the
residual stream, so delta_{l,p} equals the gradient with respect to the output of decoder layer l
and lives in the same space as the unembedding and the entity directions. One backward per
left-padded batch captures both at every layer and position; rows of a batch are independent.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_v2 as V2
from llm_memory_editability import read_interface_v2_data as V

EOS = "<|im_end|>"
USES = ("U0", "U2", "continent", "language")
OPS = ("continent", "language")
GROUPS = ("answer", "subject", "subject_last", "relation", "other")
POINTS = ("astar", "a0", "slast")


# ---------------------------------------------------------------- prompts and position groups


def screen_targets(rows):
    """v -> {op: name} for the continent/language screening rows the released model answered
    correctly; the name is the gold the model itself gave."""
    out = {}
    for r in rows:
        if r["op"] in OPS and r["strict"]:
            name = r["first"] if r.get("first") in r["golds"] else r["golds"][0]
            out.setdefault(r["v"], {})[r["op"]] = name
    return out


def use_rows(fact, targets):
    """All prompt/reply pairs of one fact: the 24 M2 single-hop pairs (U0), the two direct U2
    questions (country of v) and, for screened operations, the two direct E4 evaluation questions
    (continent of v; official language of v's country). `gold` is the entity the reply names."""
    relation = D.RELATIONS[fact.relation]
    common = {"fid": fact.fid, "h": fact.h, "v": fact.v, "relation_text": relation}
    rows = []
    for i, (user, reply) in enumerate(D.VARIED):
        rows.append(
            {
                **common,
                "use": "U0",
                "template": i,
                "prompt": D.fill(user, fact),
                "reply": D.fill(reply, fact),
                "gold": fact.v,
                "gold_at": len(D.fill(reply[: reply.index("{v}")], fact)),
            }
        )
    for i, t in enumerate(D.U2):
        rows.append(
            {
                **common,
                "use": "U2",
                "template": i,
                "prompt": D.fill(t, fact) + D.DIRECT,
                "reply": fact.country_v,
                "gold": fact.country_v,
            }
        )
    for op in OPS:
        if op not in targets:
            continue
        for i, t in enumerate(V.EVAL_OPS[op]):
            rows.append(
                {
                    **common,
                    "use": op,
                    "template": i,
                    "prompt": D.fill(t, fact) + D.DIRECT,
                    "reply": targets[op],
                    "gold": targets[op],
                }
            )
    return rows


def encode_row(tok, row):
    """Token ids of chat(prompt) + reply + <|im_end|> with the position groups.

    Groups are exclusive in the order answer > subject > relation > other: `answer` holds the
    last prompt token and every reply token that predicts a later reply token; `subject` and
    `relation` are the h and relation-phrase tokens of the user prompt. Content points: `astar`
    predicts the first token of the gold entity, `a0` is the last prompt token, `slast` the last
    subject token of the user prompt."""
    text = M.chat(tok, row["prompt"], False)
    reply = row["reply"] + EOS
    at = row.get("gold_at", 0)
    if row["reply"][at : at + len(row["gold"])] != row["gold"]:
        raise ValueError(f"{row['gold']!r} not at {at} in reply {row['reply']!r}")
    gold_span = (len(text) + at, len(text) + at + len(row["gold"]))
    spans = V2.user_spans(text, row["prompt"], [row["h"], row["relation_text"]]) + [gold_span]
    item = V2.tokenize(tok, text, spans, reply)
    h_pos, r_pos, g_pos = item["positions"]
    a, n = item["answer"]
    assert n == len(item["ids"])
    label = np.full(n, 3, dtype=np.int64)  # other
    label[[p for p in r_pos if p < a - 1]] = 2
    label[[p for p in h_pos if p < a - 1]] = 1
    label[a - 1 : n - 1] = 0
    if not h_pos or h_pos[-1] >= a - 1:
        raise ValueError(f"subject not in prompt: {row['prompt']!r}")
    astar = g_pos[0] - 1
    assert a - 1 <= astar < n - 1
    masks = np.zeros((len(GROUPS), n), dtype=np.float64)
    masks[0] = label == 0
    masks[1] = label == 1
    masks[2, h_pos[-1]] = 1.0
    masks[3] = label == 2
    masks[4] = label == 3
    item.update(
        label=label,
        masks=masks,
        points=[astar, a - 1, h_pos[-1]],
        target=item["ids"][g_pos[0]],
    )
    return item


# ---------------------------------------------------------------- gradient capture


class DownProjCapture:
    """Hooks every MLP down projection: `k[l]` is its input (batch, seq, d_ff) and, after
    backward, `delta[l]` the gradient of the loss with respect to its output (batch, seq, d).
    The token embeddings are made to require grad so the graph exists with frozen weights."""

    def __init__(self, model):
        self.model = model
        self.k, self.delta, self.handles = {}, {}, []

    def __enter__(self):
        def need_grad(module, args, output):
            if not output.requires_grad:
                return output.requires_grad_(True)
            return None

        emb = self.model.get_input_embeddings()
        self.handles.append(emb.register_forward_hook(need_grad))
        for i, layer in enumerate(M.layers(self.model)):

            def hook(module, args, output, i=i):
                self.k[i] = args[0].detach()

                def keep(grad, i=i):
                    self.delta[i] = grad.detach()

                if output.requires_grad:
                    output.register_hook(keep)

            self.handles.append(layer.mlp.down_proj.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def batch_gradients(model, tok, items):
    """One forward/backward over left-padded rows; loss = summed reply-token cross entropy.

    Returns k (L, B, T, d_ff), delta (L, B, T, d), the padding of each row and per-row
    (loss, n_tokens, greedy) of the released model."""
    device = next(model.parameters()).device
    ids, mask, _, slices = V2.collate(tok, items)
    ids, mask = ids.to(device), mask.to(device)
    with DownProjCapture(model) as cap:
        hidden = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        hidden = hidden.last_hidden_state
        rows = torch.cat([torch.full((b - a,), i) for i, (a, b) in enumerate(slices)]).to(device)
        pos = torch.cat([torch.arange(a - 1, b - 1) for a, b in slices]).to(device)
        targets = torch.cat([ids[i, a:b] for i, (a, b) in enumerate(slices)])
        logits = model.lm_head(hidden[rows, pos]).float()
        per_token = F.cross_entropy(logits, targets, reduction="none")
        per_token.sum().backward()
        n_layers = len(M.layers(model))
        k = torch.stack([cap.k[i] for i in range(n_layers)])
        delta = torch.stack([cap.delta[i] for i in range(n_layers)])
    hit = (logits.argmax(-1) == targets).detach()
    stats = []
    for i, (a, b) in enumerate(slices):
        sel = rows == i
        stats.append(
            {
                "loss": float(per_token.detach()[sel].sum()),
                "n_tokens": b - a,
                "greedy": bool(hit[sel].all()),
            }
        )
    width = ids.shape[1]
    pads = [width - len(it["ids"]) for it in items]
    return k, delta, pads, stats


def weight_gradient_check(model, tok, rows, layer_ids):
    """Relative Frobenius error between down_proj.weight.grad and sum_{b,p} delta k^T from the
    hooks, for the given layers, on one batch of rows."""
    items = [encode_row(tok, r) for r in rows]
    params = [M.layers(model)[i].mlp.down_proj.weight for i in layer_ids]
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    try:
        with torch.enable_grad():
            k, delta, _, _ = batch_gradients(model, tok, items)
        out = {}
        for j, i in enumerate(layer_ids):
            outer = torch.einsum("btd,btf->df", delta[i].double(), k[i].double())
            grad = params[j].grad.double()
            out[str(i)] = float((grad - outer).norm() / grad.norm())
    finally:
        for p in params:
            p.requires_grad_(False)
            p.grad = None
    return out


# ---------------------------------------------------------------- readouts


def address_shares(k, delta, masks):
    """k (L, n, d_ff), delta (L, n, d), masks (G, n). Per layer and group: the exact share
    ||sum_{p in G} delta_p k_p^T||_F / ||sum_p delta_p k_p^T||_F (from Gram products), the
    projection share <dW_G, dW>/||dW||^2 (sums to one over a partition) and the approximate share
    sum_G ||delta_p|| ||k_p|| / sum_all; also ||dW||_F per layer."""
    k, delta = k.double(), delta.double()
    m = torch.as_tensor(masks, dtype=torch.float64, device=k.device)
    gram = (k @ k.transpose(1, 2)) * (delta @ delta.transpose(1, 2))
    total = gram.sum((1, 2))
    exact = torch.einsum("gp,lpq,gq->lg", m, gram, m).clamp_min(0).sqrt() / total.sqrt()[:, None]
    proj = torch.einsum("gp,lpq->lg", m, gram) / total[:, None]
    nk = k.norm(dim=-1) * delta.norm(dim=-1)
    approx = nk @ m.T / nk.sum(-1, keepdim=True)
    return exact, proj, approx, total.sqrt()


def first_tokens(tok, name):
    """First token of a name with and without a leading space (as in E3)."""
    return sorted(
        {
            tok(name, add_special_tokens=False).input_ids[0],
            tok(" " + name, add_special_tokens=False).input_ids[0],
        }
    )


class Refs:
    """Unembedding, final-norm weight and entity directions shared by all readouts."""

    def __init__(self, model, entity, cities):
        w = model.lm_head.weight.detach().float()
        d = w.shape[1]
        self.model = model
        self.w = w
        self.w_unit = F.normalize(w, dim=-1)
        ones = torch.ones(1, d, device=w.device, dtype=w.dtype)
        with torch.no_grad():
            self.g = model.model.norm(ones)[0].float()
        self.entity = F.normalize(torch.stack([entity[i] for i in sorted(entity)]).float(), dim=-1)
        self.city = {c: i for i, c in enumerate(cities)}

    def gu(self, ids):
        return F.normalize(self.w[ids] * self.g, dim=-1)

    @torch.no_grad()
    def lens(self, unit):
        """Logit lens of unit vectors (N, d): final norm then unembedding."""
        x = unit * math.sqrt(unit.shape[-1])
        return self.model.lm_head(self.model.model.norm(x.to(self.w.dtype))).float()


def nan_where(x, valid):
    x = x.float().clone()
    x[~valid] = float("nan")
    return x


@torch.no_grad()
def content(vectors, refs, target, v_ids, v, h, top=0):
    """vectors (L, J, d) = -delta at the content points. Per layer and point: cosines with the
    target and v unembeddings (raw and with the final-norm weight), with v's and h's entity
    directions (and the mean over other cities, v's rank among cities), logit-lens ranks."""
    n_layers, n_points, d = vectors.shape
    vec = vectors.float()
    norm = vec.norm(dim=-1)
    valid = norm > 0
    unit = vec / norm.clamp_min(1e-30)[..., None]
    flat = unit.reshape(-1, d)
    out = {"norm": norm}
    out["cos_u_target"] = flat @ refs.w_unit[target]
    out["cos_gu_target"] = flat @ refs.gu([target])[0]
    out["cos_u_v"] = (flat @ refs.w_unit[v_ids].T).max(-1).values
    out["cos_gu_v"] = (flat @ refs.gu(v_ids).T).max(-1).values
    logits = refs.lens(flat)
    rank_target = (logits > logits[:, target : target + 1]).sum(-1)
    rank_v = torch.stack([(logits > logits[:, t : t + 1]).sum(-1) for t in v_ids]).min(0).values
    out["rank_target"] = rank_target.float()
    out["rank_v"] = rank_v.float()
    cos_e = torch.einsum("ljd,lcd->ljc", unit, refs.entity)
    iv, ih = refs.city[v], refs.city[h]
    own = cos_e[..., iv]
    out["cos_e_v"] = own.reshape(-1)
    out["cos_e_other"] = ((cos_e.sum(-1) - own) / (cos_e.shape[-1] - 1)).reshape(-1)
    out["e_v_rank"] = (cos_e > own[..., None]).sum(-1).float().reshape(-1)
    out["cos_e_h"] = cos_e[..., ih].reshape(-1)
    flat_valid = valid.reshape(-1)
    result = {}
    for key, val in out.items():
        val = val.reshape(-1)
        if key != "norm":
            val = nan_where(val, flat_valid)
        result[key] = val.reshape(n_layers, n_points).cpu().numpy()
    if top:
        ids = logits.topk(top, dim=-1).indices.reshape(n_layers, n_points, top)
        result["top_ids"] = ids.cpu().numpy()
    return result


def cosine(a, b):
    """Cosine along the last axis; NaN where either vector is zero."""
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    c = (a * b).sum(-1) / (na * nb).clamp_min(1e-30)
    return nan_where(c, (na > 0) & (nb > 0))


@torch.no_grad()
def fact_readouts(model, tok, refs, fact, rows, top_templates=()):
    """All readouts for the rows of one fact (one backward). Returns per-row records, the
    template-mean -delta per use at the content points (L, J, d), and example details for the
    (use, template) pairs in `top_templates`."""
    items = [encode_row(tok, r) for r in rows]
    with torch.enable_grad():
        k, delta, pads, stats = batch_gradients(model, tok, items)
    v_ids = first_tokens(tok, fact.v)
    records, sums, counts, examples = [], {}, {}, {}
    for i, (row, item, pad, st) in enumerate(zip(rows, items, pads, stats, strict=True)):
        ki, di = k[:, i, pad:], delta[:, i, pad:]
        exact, proj, approx, total = address_shares(ki, di, item["masks"])
        vectors = -di[:, item["points"]]
        want = (row["use"], row["template"]) in top_templates
        c = content(vectors, refs, item["target"], v_ids, fact.v, fact.h, top=5 if want else 0)
        rec = {
            "fid": fact.fid,
            "use": row["use"],
            "template": row["template"],
            **st,
            "n": len(item["ids"]),
            "exact": exact.float().cpu().numpy(),
            "proj": proj.float().cpu().numpy(),
            "approx": approx.float().cpu().numpy(),
            "dw_norm": total.float().cpu().numpy(),
        }
        top = c.pop("top_ids", None)
        rec.update(c)
        records.append(rec)
        sums[row["use"]] = sums.get(row["use"], 0) + vectors.float()
        counts[row["use"]] = counts.get(row["use"], 0) + 1
        if row["use"] == "U0":
            half = "U0_even" if row["template"] % 2 == 0 else "U0_odd"
            sums[half] = sums.get(half, 0) + vectors.float()
            counts[half] = counts.get(half, 0) + 1
        if want:
            nk = di.norm(dim=-1) * ki.norm(dim=-1)
            examples[f"{row['use']}.{row['template']}"] = {
                "prompt": row["prompt"],
                "reply": row["reply"],
                "tokens": [tok.decode([t]) for t in item["ids"]],
                "label": item["label"].tolist(),
                "points": item["points"],
                "position_share": (nk / nk.sum(-1, keepdim=True)).float().cpu().numpy().tolist(),
                "top_tokens": [
                    [[tok.decode([int(t)]) for t in top[layer, j]] for j in range(len(POINTS))]
                    for layer in range(top.shape[0])
                ],
            }
    means = {u: sums[u] / counts[u] for u in sums}
    del k, delta
    return records, means, examples


def sharing(means):
    """Per layer and content point, cosine between the template-mean -delta of U0 and of each
    other use, between the other uses, and between the two halves of the U0 templates."""
    out = {}
    names = [u for u in USES if u in means]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out[f"{a}~{b}"] = cosine(means[a], means[b]).cpu().numpy()
    if "U0_even" in means:
        out["U0_even~U0_odd"] = cosine(means["U0_even"], means["U0_odd"]).cpu().numpy()
    return out


# ---------------------------------------------------------------- aggregation and predictions

MEAN_KEYS = (
    "cos_u_target",
    "cos_gu_target",
    "cos_u_v",
    "cos_gu_v",
    "cos_e_v",
    "cos_e_other",
    "cos_e_h",
    "norm",
)
RANK_KEYS = ("rank_target", "rank_v", "e_v_rank")


def fact_level(records):
    """Template average within (fact, use): means for shares and cosines, medians for ranks,
    plus top-1/top-10 rates of the logit-lens target rank."""
    by = {}
    for r in records:
        by.setdefault((r["fid"], r["use"]), []).append(r)
    out = {}
    for key, rs in by.items():
        entry = {}
        for name in ("exact", "proj", "approx", "dw_norm", *MEAN_KEYS):
            entry[name] = nanmean0([r[name] for r in rs])
        for name in RANK_KEYS:
            entry[name] = nanmedian0([r[name] for r in rs])
        ranks = np.stack([r["rank_target"] for r in rs])
        entry["top1_target"] = nanmean0(np.where(np.isnan(ranks), np.nan, ranks < 1))
        entry["top10_target"] = nanmean0(np.where(np.isnan(ranks), np.nan, ranks < 10))
        entry["loss"] = float(np.mean([r["loss"] for r in rs]))
        entry["greedy"] = float(np.mean([r["greedy"] for r in rs]))
        entry["templates"] = len(rs)
        out[key] = entry
    return out


def nanmean0(x):
    """Mean over axis 0 ignoring NaN (all-NaN slices stay NaN without a warning)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(x, dtype=np.float64), 0)


def nanmedian0(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(x, dtype=np.float64), 0)


def aggregate(facts_level, shares, cross):
    """Fact means per use (fact = unit). `shares`: fid -> sharing dict; `cross`: cross-fact
    baselines from `cross_fact`."""
    out = {}
    for use in USES:
        entries = [e for (fid, u), e in facts_level.items() if u == use]
        if not entries:
            continue
        agg = {"n_facts": len(entries)}
        for name in ("exact", "proj", "approx"):
            arr = nanmean0([e[name] for e in entries])
            agg[name] = {g: arr[:, j].tolist() for j, g in enumerate(GROUPS)}
        agg["dw_norm"] = nanmean0([e["dw_norm"] for e in entries]).tolist()
        for name in (*MEAN_KEYS, "top1_target", "top10_target"):
            arr = nanmean0([e[name] for e in entries])
            agg[name] = {p: arr[:, j].tolist() for j, p in enumerate(POINTS)}
        for name in RANK_KEYS:
            arr = nanmedian0([e[name] for e in entries])
            agg[name + "_median"] = {p: arr[:, j].tolist() for j, p in enumerate(POINTS)}
        agg["loss"] = float(np.mean([e["loss"] for e in entries]))
        agg["greedy"] = float(np.mean([e["greedy"] for e in entries]))
        out[use] = agg
    pairs = sorted({k for s in shares.values() for k in s})
    share_out = {}
    for pair in pairs:
        vals = [s[pair] for s in shares.values() if pair in s]
        arr = nanmean0(vals)
        share_out[pair] = {
            "n_facts": len(vals),
            **{p: arr[:, j].tolist() for j, p in enumerate(POINTS)},
        }
        if pair in cross:
            share_out[pair]["cross_fact"] = cross[pair]
    out["sharing"] = share_out
    return out


def cross_fact(store, subjects):
    """Different-subject baseline: for each use pair (U0, X), the mean over fact pairs (f, f')
    with different subjects of cos(mean -delta of U0 for f, mean -delta of X for f').
    `store[use]` maps fid -> unit (L, J, d) tensor (CPU, half precision; zero where -delta is
    zero)."""
    out = {}
    if "U0" not in store:
        return out
    for other in USES[1:]:
        if other not in store:
            continue
        fids = [f for f in store[other] if f in store["U0"]]
        if len(fids) < 2:
            continue
        a = torch.stack([store["U0"][f] for f in fids]).float()
        b = torch.stack([store[other][f] for f in fids]).float()
        cos = torch.einsum("fljd,gljd->ljfg", a, b)
        valid = (a.norm(dim=-1) > 0).permute(1, 2, 0)[..., :, None] & (b.norm(dim=-1) > 0).permute(
            1, 2, 0
        )[..., None, :]
        h = [subjects[f] for f in fids]
        diff = torch.tensor([[x != y for y in h] for x in h])
        sel = valid & diff
        total = torch.where(sel, cos, torch.zeros_like(cos)).sum((-1, -2))
        count = sel.sum((-1, -2)).clamp_min(1)
        mean = total / count
        mean[sel.sum((-1, -2)) == 0] = float("nan")
        out[f"U0~{other}"] = {p: mean[:, j].tolist() for j, p in enumerate(POINTS)}
    return out


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")

    def ranks(a):
        order = a.argsort()
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        for val in np.unique(a):  # average ties
            idx = a == val
            r[idx] = r[idx].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def majority(flags):
    """Over the defined (non-NaN) entries: count, total and whether more than half are true."""
    flags = [bool(f) for f in flags if f is not None]
    return {"true": sum(flags), "of": len(flags), "majority": sum(flags) * 2 > len(flags)}


def layer_bands(n_layers):
    cut = (2 * n_layers) // 3
    return list(range(cut)), list(range(cut, n_layers))


def predictions(agg, n_layers):
    """Pre-registered P1-P3 for one model x fact set (protocol section 5)."""
    early, late = layer_bands(n_layers)
    u0 = agg["U0"]
    layers = list(range(n_layers))

    def gt(a, b):
        return [None if (np.isnan(x) or np.isnan(y)) else x > y for x, y in zip(a, b, strict=True)]

    share_a, share_s = u0["exact"]["answer"], u0["exact"]["subject"]
    p1_major = majority(gt(share_a, share_s))
    p1_rho = spearman(layers, share_a)
    p1 = {
        "answer_gt_subject_layers": p1_major,
        "spearman_answer_share": p1_rho,
        "hit": p1_major["majority"] and p1_rho >= 0.5,
    }
    cos_a = u0["cos_u_target"]["astar"]
    rho_a = spearman(layers, cos_a)
    ranks = u0["rank_target_median"]["astar"]
    top10 = majority([None if np.isnan(ranks[i]) else ranks[i] < 10 for i in late])
    p2a = {
        "spearman_cos_u_astar": rho_a,
        "late_layers_median_rank_top10": top10,
        "hit": rho_a >= 0.5 and top10["majority"],
    }
    ev, uv = u0["cos_e_v"]["slast"], u0["cos_u_v"]["slast"]
    p2b_major = majority(gt([ev[i] for i in early], [uv[i] for i in early]))
    p2b = {"early_middle_layers_e_v_gt_u_v": p2b_major, "hit": p2b_major["majority"]}
    p3 = {"per_use": {}}
    for other in USES[1:]:
        pair = agg["sharing"].get(f"U0~{other}")
        if pair is None:
            continue
        m = majority(gt(pair["slast"], pair["astar"]))
        p3["per_use"][other] = {"slast_gt_astar_layers": m, "hit": m["majority"]}
    p3["hit"] = bool(p3["per_use"]) and all(v["hit"] for v in p3["per_use"].values())
    return {
        "early_middle_layers": early,
        "late_layers": late,
        "P1": p1,
        "P2a": p2a,
        "P2b": p2b,
        "P3": p3,
    }


def bootstrap_scalars(facts_level, shares, n_layers, n_boot=2000, seed=20261020):
    """Fact-bootstrap 95% intervals of the scalar summaries behind P1-P3."""
    early, _ = layer_bands(n_layers)
    rng = np.random.default_rng(seed)
    scalars = {}
    u0 = {fid: e for (fid, u), e in facts_level.items() if u == "U0"}
    gi = {g: j for j, g in enumerate(GROUPS)}
    pj = {p: j for j, p in enumerate(POINTS)}
    scalars["P1 layer-mean exact share answer - subject"] = [
        nanmean0(e["exact"][:, gi["answer"]] - e["exact"][:, gi["subject"]]) for e in u0.values()
    ]
    scalars["P2a cos(-delta, u) at astar, last layer"] = [
        e["cos_u_target"][-1, pj["astar"]] for e in u0.values()
    ]
    scalars["P2a cos(-delta, u) at astar, layer L/2"] = [
        e["cos_u_target"][n_layers // 2, pj["astar"]] for e in u0.values()
    ]
    scalars["P2b early-middle mean cos(-delta, e_v) - cos(-delta, u_v) at slast"] = [
        nanmean0(e["cos_e_v"][early, pj["slast"]] - e["cos_u_v"][early, pj["slast"]])
        for e in u0.values()
    ]
    for other in USES[1:]:
        vals = [
            nanmean0(s[f"U0~{other}"][:, pj["slast"]] - s[f"U0~{other}"][:, pj["astar"]])
            for s in shares.values()
            if f"U0~{other}" in s
        ]
        if vals:
            scalars[f"P3 layer-mean cross-use cos slast - astar (U0~{other})"] = vals
    out = {}
    for name, vals in scalars.items():
        x = np.asarray(vals, float)
        x = x[~np.isnan(x)]
        if len(x) == 0:
            continue
        boots = rng.choice(x, size=(n_boot, len(x))).mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "n_facts": len(x)}
    return out
