"""Position-routed down updates, complete-answer objectives, and audited data helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hebbian_data import TEMPLATE, encode_answer, score_answer
from .hebbian_model import set_determinism, tensor_hash

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/qwen-path-learning-v1"
ART = ROOT / "docs/development-artifacts/qwen-path-learning-v1"
RESULTS = ROOT / "results/qwen-path-learning-v1"
CONFIG = ROOT / "configs/qwen-path-learning-v1.json"
GROUPS = ("S_end", "S_other", "C", "W", "L", "A")

# All views ask the same original CounterFact relation; no generated answer labels.
RELATIONS = {
    "P19": ("{} was born in", "The birthplace of {} is", "{}'s birthplace is"),
    "P20": ("{} died in", "The place of death of {} is", "{}'s place of death is"),
    "P103": (
        "The native language of {} is",
        "The mother tongue of {} is",
        "{}'s native language is",
    ),
    "P176": ("The manufacturer of {} is", "{} was manufactured by", "{}'s manufacturer is"),
    "P178": ("The developer of {} is", "{} was developed by", "{}'s developer is"),
    "P495": (
        "The country of origin of {} is",
        "{} originated in the country of",
        "{}'s country of origin is",
    ),
    "P740": (
        "{} was founded in",
        "The place where {} was founded is",
        "{}'s place of formation is",
    ),
    "P364": (
        "The original language of {} is",
        "{} was originally in the language",
        "{}'s original language is",
    ),
    "P407": ("The language of {} is", "{} is in the language", "{}'s language is"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def encode_fact(tokenizer, subject, statement, answer, view=0, plain=False):
    prompt = statement if plain else TEMPLATE.format(statement)
    enc = encode_answer(tokenizer, prompt, answer)
    offset = 0 if plain else prompt.index(statement)
    subject_start = statement.index(subject) + offset
    subject_end = subject_start + len(subject)
    offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)[
        "offset_mapping"
    ]
    subject_tokens = [
        i for i, (a, b) in enumerate(offsets) if b > subject_start and a < subject_end
    ]
    if not subject_tokens:
        raise ValueError("Subject has no tokens")
    roles = []
    for i, (a, b) in enumerate(offsets):
        if i == len(offsets) - 1:
            role = "L"
        elif i == subject_tokens[-1]:
            role = "S_end"
        elif i in subject_tokens:
            role = "S_other"
        elif b > offset and a < offset + len(statement):
            role = "C"
        else:
            role = "W"
        roles.append(role)
    roles += ["A"] * (len(enc["input_ids"]) - len(roles))
    return {**enc, "prompt": prompt, "roles": roles, "view": view, "plain": plain}


def random_positions(roles, group, key, seed, bins=2, draws=16, energies=None):
    """Disjoint count/bin-matched controls; optional local-energy matching only."""
    n = roles.index("L") + 1
    target = [i for i, role in enumerate(roles[: n - 1]) if role == group]
    if not target:
        return [], {"valid": False, "reason": "empty_group"}

    def strata(i):
        return min(bins - 1, i * bins // max(n - 1, 1))

    eligible = {
        b: [i for i in range(n - 1) if i not in target and strata(i) == b] for b in range(bins)
    }
    counts = {b: sum(strata(i) == b for i in target) for b in range(bins)}
    if any(len(eligible[b]) < counts[b] for b in range(bins)):
        return [], {"valid": False, "reason": "insufficient_disjoint_positions"}
    rng = np.random.default_rng(int(stable(f"{key}:{group}:{seed}")[:16], 16))
    options = []
    for _ in range(draws if energies is not None else 1):
        positions = sorted(
            int(i) for b in range(bins) for i in rng.choice(eligible[b], counts[b], replace=False)
        )
        error = (
            0.0
            if energies is None
            else abs(sum(energies[i] for i in positions) - sum(energies[i] for i in target))
            / max(sum(energies[i] for i in target), 1e-30)
        )
        options.append((error, positions))
    error, selected = min(options, key=lambda x: (x[0], x[1]))
    return selected, {"valid": True, "relative_energy_error": error, "count": len(target)}


def sequence_mean(values, batch_indices, count):
    sums = values.new_zeros(count).scatter_add_(0, batch_indices, values)
    return sums / torch.bincount(batch_indices, minlength=count).clamp_min(1)


def fisher_quadratic(logp, response):
    p = logp.exp()
    return 0.5 * ((p * response.square()).sum(-1) - (p * response).sum(-1).square())


class PathEngine:
    def __init__(self, device, layer=None):
        self.cfg = read(CONFIG)
        set_determinism(self.cfg["seed"])
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / self.cfg["model_source"], local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                ROOT / self.cfg["model_source"],
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.layer_index = self.cfg["layer"] if layer is None else layer
        self.down = self.model.model.layers[self.layer_index].mlp.down_proj
        self.base_hash = tensor_hash(self.down.weight)
        self.delta = torch.nn.Parameter(torch.zeros_like(self.down.weight))
        self.route = "all"
        self.route_seed = 0
        self.route_group = None
        self.route_positions = {}
        self.mask = None
        self.capture = None
        self.handle = self.down.register_forward_hook(self._hook)
        self.reference = {}
        self.forward_tokens = 0
        self.started = time.time()

    def _hook(self, module, args, output):
        increment = F.linear(args[0], self.delta)
        if self.capture is not None:
            self.capture.append(increment.detach().square().sum(-1).cpu())
        if self.mask is not None:
            increment = increment * self.mask[:, -increment.shape[1] :, None]
        return output + increment

    def mask_for(self, enc):
        roles = enc["roles"]
        mask = [1.0] * len(roles)
        if self.route == "all":
            return mask
        if "L" not in roles:  # Unannotated natural text has no fact-position groups.
            return mask
        if self.route == "group":
            positions = [i for i, x in enumerate(roles) if x == self.route_group]
        elif self.route == "random":
            positions, _ = random_positions(roles, self.route_group, enc["key"], self.route_seed)
        elif self.route == "positions":
            positions = self.route_positions.get(enc["key"], [])
        elif self.route == "zero":
            positions = list(range(len(roles)))
        else:
            raise ValueError(self.route)
        for i in positions:
            mask[i] = 0.0
        return mask

    def batch(self, encoded, prompt_only=False):
        sequences = [
            e["input_ids"][: e["answer_start"]] if prompt_only else e["input_ids"] for e in encoded
        ]
        width = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), width),
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        attention = torch.zeros_like(ids)
        loss = torch.zeros_like(ids, dtype=torch.bool)
        masks = torch.ones_like(ids, dtype=torch.float32)
        for i, (seq, enc) in enumerate(zip(sequences, encoded, strict=True)):
            ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            attention[i, : len(seq)] = 1
            if not prompt_only:
                loss[i, : len(seq)] = torch.tensor(
                    enc["loss_mask"], device=self.device, dtype=torch.bool
                )
            masks[i, : len(seq)] = torch.tensor(self.mask_for(enc)[: len(seq)], device=self.device)
        self.mask = masks
        return ids, attention, loss

    def hidden(self, ids, attention):
        self.forward_tokens += ids.numel()
        return self.model.model(
            input_ids=ids, attention_mask=attention, use_cache=False
        ).last_hidden_state

    def objectives(self, encoded, include_kl=True):
        ids, attention, mask = self.batch(encoded)
        b, t = torch.where(mask[:, 1:])
        hidden = self.hidden(ids, attention)
        logits = self.model.lm_head(hidden[b, t])
        ce = sequence_mean(
            F.cross_entropy(logits, ids[b, t + 1], reduction="none"), b, len(encoded)
        )
        if not include_kl:
            return ce, None
        with torch.no_grad():
            ref_hidden = torch.cat([self.reference[e["key"]] for e in encoded]).to(self.device)
            ref_log = F.log_softmax(self.model.lm_head(ref_hidden), dim=-1)
        divergence = (ref_log.exp() * (ref_log - F.log_softmax(logits, dim=-1))).sum(-1)
        return ce, sequence_mean(divergence, b, len(encoded))

    @torch.no_grad()
    def cache_reference(self, encoded, batch_size=8):
        if self.delta.count_nonzero().item():
            raise ValueError("Reference must be cached at the parent")
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            ids, attention, mask = self.batch(chunk)
            b, t = torch.where(mask[:, 1:])
            hidden = self.hidden(ids, attention)
            for i, enc in enumerate(chunk):
                self.reference[enc["key"]] = hidden[b[b == i], t[b == i]].detach().cpu()

    @torch.no_grad()
    def generate(self, encoded):
        # Cached greedy decoding; routing is built from the input prompt alone.
        sequences = [e["input_ids"][: e["answer_start"]] for e in encoded]
        width = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), width),
            self.tokenizer.pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        attention = torch.zeros_like(ids)
        route_mask = torch.ones_like(ids, dtype=torch.float32)
        for i, (seq, enc) in enumerate(zip(sequences, encoded, strict=True)):
            ids[i, -len(seq) :] = torch.tensor(seq, device=self.device)
            attention[i, -len(seq) :] = 1
            route_mask[i, -len(seq) :] = torch.tensor(
                self.mask_for(enc)[: len(seq)], device=self.device
            )
        tails = [[] for _ in encoded]
        done = [False] * len(encoded)
        cache = None
        for step in range(self.cfg["max_new_tokens"]):
            position = attention.cumsum(-1) - 1
            position.masked_fill_(attention == 0, 0)
            self.mask = (
                route_mask
                if step == 0
                else torch.full(
                    (len(encoded), 1),
                    0.0
                    if self.route == "zero" or (self.route == "group" and self.route_group == "A")
                    else 1.0,
                    device=self.device,
                )
            )
            out = self.model.model(
                input_ids=ids,
                attention_mask=attention,
                position_ids=position if step == 0 else position[:, -1:],
                past_key_values=cache,
                use_cache=True,
            )
            self.forward_tokens += ids.numel()
            cache = out.past_key_values
            next_ids = self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1)
            for i, token in enumerate(next_ids.tolist()):
                if not done[i]:
                    tails[i].append(token)
                    decoded = self.tokenizer.decode(tails[i], skip_special_tokens=True)
                    done[i] = token == self.tokenizer.eos_token_id or "\n" in decoded
            if all(done):
                break
            ids = next_ids[:, None]
            attention = torch.cat([attention, torch.ones_like(ids)], dim=1)
        return [
            {
                "prediction": self.tokenizer.decode(tail, skip_special_tokens=True),
                "tokens": tail,
                "eos": self.tokenizer.eos_token_id in tail,
            }
            for tail in tails
        ]

    @torch.no_grad()
    def evaluate(self, records, views=(0,), batch_size=16, with_kl=True, generate=True):
        entries = [(r, r["encoded"][v]) for r in records for v in views]
        output = []
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            encs = [e for _, e in chunk]
            ce, kl = self.objectives(encs, include_kl=with_kl)
            predictions = self.generate(encs) if generate else [None] * len(chunk)
            for i, ((record, enc), prediction) in enumerate(zip(chunk, predictions, strict=True)):
                item = {
                    "case_id": record["case_id"],
                    "view": enc["view"],
                    "ce": float(ce[i]),
                    "kl": None if kl is None else float(kl[i]),
                }
                if prediction is not None:
                    item.update(prediction)
                    item["correct"] = score_answer(prediction["prediction"], record["aliases"])
                output.append(item)
        return output

    @torch.no_grad()
    def evaluate_text(self, encoded):
        result = []
        for start in range(0, len(encoded), 2):
            chunk = encoded[start : start + 2]
            ce, kl = self.objectives(chunk)
            result.extend(
                {"key": e["key"], "ce": float(c), "kl": float(k)}
                for e, c, k in zip(chunk, ce, kl, strict=True)
            )
        return result

    def restore(self):
        with torch.no_grad():
            self.delta.zero_()
        self.route = "all"
        self.mask = None
        assert tensor_hash(self.down.weight) == self.base_hash


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def valid_aliases(raw):
    by_id = {}
    for row in raw:
        target = row["requested_rewrite"]["target_true"]
        by_id.setdefault(target["id"], set()).add(target["str"].strip())
    return {k: sorted(v) for k, v in by_id.items()}


def articles_from_lines(lines):
    articles = []
    title, parts = None, []
    for line in lines:
        if re.fullmatch(r"\s*= [^=]+ =\s*", line):
            if title and parts:
                articles.append((title, "".join(parts)))
            title, parts = line.strip(), []
        elif title:
            parts.append(line)
    if title and parts:
        articles.append((title, "".join(parts)))
    return articles
