"""bios-write-address-v1 (zero training): is a bioS attribute keyed by the person's name or by
the biography context that precedes it, under single (S), multiple (M) and multiple-permuted (MP)
biographies?

Two readouts on the saved bioS-original checkpoints (no parameter changes):

1. Read address, in the pretraining text format. For person A and a target attribute k at
   sentence j of the canonical order, the prompt is sentences 0..j-1 of A's biography followed by
   the template of sentence j up to the value. Four prompt types:
   - original:     A's name, A's context;
   - conflict:     another person B's name (same pronoun, different value of k), A's context;
   - context_only: an unseen name (recombined name parts), A's context;
   - name_only:    B's name in the first-sentence form of attribute k, no context.
   Greedy continuations are scored against A's and B's values.
2. Write address. For the attribute-value cross entropy of a biography sentence, the gradient on
   every MLP output projection is sum_p delta_p k_p^T (k = down input = address). Exact Frobenius
   shares by position group (name / earlier context / own sentence / value-predicting), and the
   cosine between the name-group (or value-group) writes of the same fact in different
   biography variants.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ATTRS = ("date", "birthcity", "university", "field", "workcity", "company")
FIELD = dict(
    date="birthday",
    birthcity="birthcity",
    university="university",
    field="field",
    workcity="company1city",
    company="company1name",
)
MONTHS = (
    "January February March April May June July August September October November December"
).split()
EOT = 50256
SENTINEL = ""
PROMPTS = ("original", "conflict", "context_only", "name_only")
GROUPS = ("name", "context", "sentence", "value")


# ---------------------------------------------------------------- world


def load_templates(source_code):
    """Literal templates of get_text_simple3, as bios_original_data.source_templates."""
    tree = ast.parse(Path(source_code).read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_text_simple3"
    )
    values = {}
    for node in fn.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("sentence_structures"):
                values[int(name[-1]) - 1] = ast.literal_eval(node.value)
    return [values[i] for i in range(6)]


def value(p, attr):
    if attr == "date":
        return f"{MONTHS[p['month'] - 1]} {p['day']}, {p['year']}"
    return p[attr]


def fields(p):
    return {FIELD[a]: value(p, a) for a in ATTRS}


class World:
    def __init__(self, data_root, source_code, tokenizer_path):
        from tokenizers import Tokenizer

        root = Path(data_root)
        self.people = json.loads((root / "people.json").read_text())
        self.template_ids = np.load(root / "template-ids.npy")
        self.templates = load_templates(source_code)
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.names = {p["name"] for p in self.people}

    def sentence(self, a, template_id, name, values):
        """One biography sentence, with the leading space the stream uses."""
        return " " + self.templates[ATTRS.index(a)][template_id].format(name=name, **values)

    def split(self, a, template_id, name, values, target):
        """Token ids of the sentence before the first value token, and the value tokens."""
        text = self.sentence(a, template_id, name, values)
        marked = self.sentence(a, template_id, name, {**values, FIELD[a]: SENTINEL})
        c0 = marked.index(SENTINEL)
        if text[:c0] != marked[:c0]:
            raise ValueError("value precedes another field")
        c1 = c0 + len(target)
        if text[c0:c1] != target:
            raise ValueError(f"value {target!r} not at {c0} in {text!r}")
        enc = self.tok.encode(text)
        t0 = next(i for i, (s, e) in enumerate(enc.offsets) if e > c0)
        t1 = next((i for i, (s, e) in enumerate(enc.offsets) if s >= c1), len(enc.ids))
        return enc.ids[:t0], enc.ids[t0:t1]

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids)


def partners(world, people_ids, attr, rng):
    """B for each A: same pronoun, different value of attr."""
    by_pronoun = {}
    for p in world.people:
        by_pronoun.setdefault(p["pronoun"], []).append(p["id"])
    out = []
    for a in people_ids:
        pa = world.people[a]
        pool = by_pronoun[pa["pronoun"]]
        while True:
            b = pool[rng.integers(len(pool))]
            if b != a and value(world.people[b], attr) != value(pa, attr):
                out.append(b)
                break
    return out


def unseen_name(world, a, rng):
    """First, middle and last name drawn from three different same-pronoun people, absent from
    the population."""
    pa = world.people[a]
    pool = [p for p in world.people if p["pronoun"] == pa["pronoun"] and p["id"] != a]
    while True:
        x, y, z = (pool[rng.integers(len(pool))]["name"].split(" ") for _ in range(3))
        if len(x) == len(y) == len(z) == 3:
            name = f"{x[0]} {y[1]} {z[2]}"
            if name not in world.names:
                return name


def context_tokens(world, j, template_ids, name, pronoun, values):
    """EOT and sentences 0..j-1 in canonical order, the name only in the first."""
    toks = [EOT]
    for i in range(j):
        who = name if i == 0 else pronoun
        toks += world.encode(world.sentence(ATTRS[i], int(template_ids[i]), who, values))
    return toks


def read_prompts(world, people_ids, seed=20261020):
    """Prompt token ids for every person x target attribute (canonical sentences 1..5) x prompt
    type, using A's variant-0 templates. Returns a list of dicts."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(1, len(ATTRS)):
        attr = ATTRS[j]
        bs = partners(world, people_ids, attr, rng)
        for a, b in zip(people_ids, bs, strict=True):
            pa, pb = world.people[a], world.people[b]
            ids = world.template_ids[a, 0]
            fa = fields(pa)
            u = unseen_name(world, a, rng)
            c = world.people[partners(world, [a], attr, rng)[0]]  # chance reference

            prefix, _ = world.split(attr, int(ids[j]), pa["pronoun"], fa, value(pa, attr))
            first, _ = world.split(attr, int(ids[j]), pb["name"], fields(pb), value(pb, attr))
            base = dict(person=a, partner=b, attr=attr, j=j, a_value=value(pa, attr),
                        b_value=value(pb, attr), c_value=value(c, attr))  # fmt: skip
            for kind, name in (("original", pa["name"]), ("conflict", pb["name"]),
                               ("context_only", u)):  # fmt: skip
                ids_ = context_tokens(world, j, ids, name, pa["pronoun"], fa) + prefix
                rows.append({**base, "prompt": kind, "ids": ids_})
            rows.append({**base, "prompt": "name_only", "ids": [EOT] + first})
    return rows


# ---------------------------------------------------------------- model


def load_model(path, device):
    from llm_memory_editability.bios_original_model import GPT

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def greedy(model, rows, steps=12, batch=512):
    """Greedy continuations; rows are bucketed by length so no padding enters attention."""
    device = next(model.parameters()).device
    out = [None] * len(rows)
    by_len = {}
    for i, r in enumerate(rows):
        by_len.setdefault(len(r["ids"]), []).append(i)
    for idx in by_len.values():
        for s in range(0, len(idx), batch):
            part = idx[s : s + batch]
            x = torch.tensor([rows[i]["ids"] for i in part], device=device)
            n = x.shape[1]
            for _ in range(steps):
                logits = model.logits(model.hidden(x)[:, -1])
                x = torch.cat([x, logits.argmax(-1, keepdim=True)], 1)
            for i, y in zip(part, x[:, n:].tolist(), strict=True):
                out[i] = y
    return out


def score(world, rows, continuations):
    """Which person's value the greedy continuation starts with."""
    res = []
    for r, y in zip(rows, continuations, strict=True):
        text = world.decode(y)
        t = text.lstrip()
        res.append(
            dict(
                a=t.startswith(r["a_value"]),
                b=t.startswith(r["b_value"]),
                c=t.startswith(r["c_value"]),
                text=text,
            )
        )
    return res


# ---------------------------------------------------------------- write address


class DownCapture:
    """k = input of every block's down projection; delta = gradient at its output."""

    def __init__(self, model):
        self.model, self.k, self.delta, self.handles = model, {}, {}, []

    def __enter__(self):
        def need_grad(module, args, output):
            return output.requires_grad_(True)

        self.handles.append(self.model.embedding.register_forward_hook(need_grad))
        for i, block in enumerate(self.model.blocks):

            def hook(module, args, output, i=i):
                self.k[i] = args[0].detach()

                def keep(grad, i=i):
                    self.delta[i] = grad.detach()

                output.register_hook(keep)

            self.handles.append(block.down.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def biography_rows(world, people_ids, variants, permute, seed=20261020):
    """One row per person x variant x target attribute: the biography up to the end of the
    target value, with group labels per position and the value-token targets."""
    rng = np.random.default_rng(seed + 7)
    rows = []
    for a in people_ids:
        p = world.people[a]
        f = fields(p)
        for v in variants:
            ids = world.template_ids[a, v]
            order = list(rng.permutation(6)) if permute else list(range(6))
            for k in range(6):
                toks, label = [EOT], [1]  # EOT counts as context
                for pos, i in enumerate(order):
                    name = p["name"] if pos == 0 else p["pronoun"]
                    attr = ATTRS[i]
                    if i == k:
                        pre, val = world.split(attr, int(ids[i]), name, f, value(p, attr))
                        sent = pre + val
                        lab = [2] * len(sent)
                        if pos == 0:
                            nlen = len(world.encode(" " + p["name"]))
                            lab[:nlen] = [0] * nlen
                        for q in range(len(pre) - 1, len(sent) - 1):
                            lab[q] = 3  # predicts a value token
                        start = len(toks)
                        toks += sent
                        label += lab
                        rows.append(
                            dict(person=a, variant=v, attr=attr, slot=pos, ids=toks,
                                 label=label, targets=(start + len(pre), len(toks)))
                        )  # fmt: skip
                        break
                    sent = world.encode(world.sentence(attr, int(ids[i]), name, f))
                    lab = [1] * len(sent)
                    if pos == 0:
                        nlen = len(world.encode(" " + p["name"]))
                        lab[:nlen] = [0] * nlen
                    toks += sent
                    label += lab
    return rows


def write_batch(model, rows):
    """One forward/backward over right-padded rows (causal, so padding never reaches a target);
    loss = summed value-token cross entropy per row. Returns k (L, B, T, d_ff), delta
    (L, B, T, d) and the label matrix (B, T) with -1 at padding."""
    device = next(model.parameters()).device
    width = max(len(r["ids"]) for r in rows)
    x = torch.full((len(rows), width), EOT, dtype=torch.long)
    lab = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, : len(r["ids"])] = torch.tensor(r["ids"])
        lab[i, : len(r["label"])] = torch.tensor(r["label"])
    x = x.to(device)
    with torch.enable_grad(), DownCapture(model) as cap:
        h = model.hidden(x)
        loss = 0.0
        for i, r in enumerate(rows):
            s, e = r["targets"]
            loss = loss + F.cross_entropy(model.logits(h[i, s - 1 : e - 1]).float(), x[i, s:e],
                                          reduction="sum")  # fmt: skip
        loss.backward()
        L = len(model.blocks)
        k = torch.stack([cap.k[i] for i in range(L)])
        d = torch.stack([cap.delta[i] for i in range(L)])
    return k, d, lab.to(device)


def group_shares(k, d, lab):
    """Exact Frobenius share of each group's write per layer and row: (L, B, G)."""
    k, d = k.double(), d.double()
    gram = (k @ k.transpose(-1, -2)) * (d @ d.transpose(-1, -2))  # (L, B, T, T)
    total = gram.sum((-1, -2))
    out = []
    for g in range(len(GROUPS)):
        m = (lab == g).double()  # (B, T)
        s = torch.einsum("bp,lbpq,bq->lb", m, gram, m)
        out.append(s.clamp_min(0).sqrt() / total.clamp_min(1e-300).sqrt())
    return torch.stack(out, -1)


def cross_variant_cosine(k, d, lab, rows):
    """For rows grouped by (person, attr) over variants: mean pairwise cosine between the
    group writes dW_G = sum_{p in G} delta_p k_p^T of different variants, per layer and group
    (name, value). Returns (L, 2) averaged over pairs and facts, and the count of facts.
    A group whose write is zero (the name group in the last layer, whose MLP only acts on its
    own position) gives NaN."""
    k, d = k.double(), d.double()
    facts = {}
    for i, r in enumerate(rows):
        facts.setdefault((r["person"], r["attr"]), []).append(i)
    sums = torch.zeros(k.shape[0], 2, dtype=torch.float64, device=k.device)
    counts = torch.zeros_like(sums)
    n = 0
    for idx in facts.values():
        if len(idx) < 2:
            continue
        for gi, g in enumerate((0, 3)):
            ks = [k[:, i][:, lab[i] == g] for i in idx]  # (L, n_g, d_ff)
            ds = [d[:, i][:, lab[i] == g] for i in idx]
            inner = torch.zeros(
                k.shape[0], len(idx), len(idx), dtype=torch.float64, device=k.device
            )
            for a in range(len(idx)):
                for b in range(a, len(idx)):
                    v = ((ds[a] @ ds[b].transpose(1, 2)) * (ks[a] @ ks[b].transpose(1, 2))).sum(
                        (1, 2)
                    )
                    inner[:, a, b] = inner[:, b, a] = v
            norm = inner.diagonal(dim1=1, dim2=2).sqrt()
            cos = inner / (norm[:, :, None] * norm[:, None, :])
            off = ~torch.eye(len(idx), dtype=torch.bool, device=k.device)
            ok = (norm > 1e-12 * norm.max().clamp_min(1e-300)).all(-1)  # (L,)
            sums[:, gi] += torch.where(ok, cos[:, off].mean(-1), 0.0)
            counts[:, gi] += ok.double()
        n += 1
    return torch.where(counts > 0, sums / counts.clamp_min(1), float("nan")), n
