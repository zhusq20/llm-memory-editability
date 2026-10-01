"""Paired interventions on MLP output and same-layer sequence cache.

Oracle bridge hints define an assisted diagnostic, not an identified natural
reasoning circuit. Cache interventions do not change model parameters.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/architecture-bridge-v1"
ART = ROOT / "docs/development-artifacts/architecture-bridge-v1"
RESULTS = ROOT / "results/architecture-bridge-v1"
CONFIG = ROOT / "configs/architecture-bridge-v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def order(value):
    return hashlib.sha256(f"142935:{value}".encode()).hexdigest()


def normalize(value):
    import string

    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def grade(text, answer, aliases=()):
    value = text.rsplit("Final answer:", 1)[-1].strip()
    value = re.sub(r"^(?:Answer|Bridge):\s*", "", value, flags=re.I).strip().strip('"*')
    predicted, gold = normalize(value), normalize(answer)
    a, b = Counter(predicted.split()), Counter(gold.split())
    common = sum((a & b).values())
    return {
        "em": predicted == gold,
        "f1": 2 * common / (sum(a.values()) + sum(b.values())) if common else 0.0,
        "alias_em": predicted in {normalize(x) for x in [answer, *aliases]},
    }


def base_prompt(row, question=None):
    context = row.get("support_context", "")
    parts = ["Question: What is 2 + 2?\nAnswer: 4", "Output only the short answer."]
    if context:
        parts.append("Use the supplied passages.\nPassages:\n" + context)
    parts.append("Question: " + (question if question is not None else row["question"]))
    return "\n\n".join(parts)


def prefix(row, bridge):
    return base_prompt(row) + "\nIntermediate entity: " + bridge + "\n"


def prepare():
    from transformers import AutoTokenizer

    if (ART / "data-lock.json").exists():
        raise FileExistsError("This data selection is already frozen")
    cfg = read(CONFIG)
    toks = {
        k: AutoTokenizer.from_pretrained(ROOT / v["path"], local_files_only=True)
        for k, v in cfg["models"].items()
    }
    source_path = ROOT / "data/twohop-frozen-v1/cases.json"
    source, selected, audit = read(source_path), [], {}
    for dataset in cfg["datasets"]:
        for split, count in cfg["cases_per_dataset"].items():
            pool = sorted(
                [x for x in source if x["dataset"] == dataset and x["split"] == split],
                key=lambda x: order(x["id"]),
            )
            for row in pool[:count]:
                r = copy.deepcopy(row)
                names = sorted({p["bridge"] for p in pool}, key=order)
                forbidden = {normalize(x) for x in [r["bridge"], *r.get("bridge_aliases", [])]}
                gold = {normalize(x) for x in [r["answer"], *r.get("aliases", [])]}
                target = {k: len(t.encode(prefix(r, r["bridge"]))) for k, t in toks.items()}
                chosen = []
                for name in names:
                    norm = normalize(name)
                    if norm in forbidden or any(g and g in norm for g in gold):
                        continue
                    if all(len(t.encode(prefix(r, name))) == target[k] for k, t in toks.items()):
                        chosen.append(name)
                    if len(chosen) == 2:
                        break
                reason = None
                if len(chosen) < 2:
                    reason = "fewer_than_two_length_matched_wrong_bridges"
                if max(target.values()) > cfg["max_prefix_tokens"]:
                    reason = "prefix_exceeds_budget"
                if any(g and g in normalize(r["bridge"]) for g in gold):
                    reason = "correct_bridge_contains_final_answer"
                competitors = [
                    x["answer"] for x in pool if normalize(x["answer"]) != normalize(r["answer"])
                ]
                r.update(
                    prefix_tokens=target,
                    corrupted_bridge=chosen[0] if chosen else None,
                    control_bridge=chosen[1] if len(chosen) > 1 else None,
                    exclusion_reason=reason,
                    competitor=(r.get("wrong_donor") or {}).get("answer", competitors[0]),
                )
                selected.append(r)
            audit[f"{dataset}:{split}"] = {
                "selected": count,
                "eligible": sum(
                    x["exclusion_reason"] is None
                    for x in selected
                    if x["dataset"] == dataset and x["split"] == split
                ),
            }
    # E comes from selected evaluation rows; R and U are independent source groups
    # from the unused historical pool. D is never used to choose a direction.
    edits = [r for r in selected if r["dataset"] == "2wiki" and r["split"] == "evaluation"]
    used = {r["group"] for r in selected if r["dataset"] == "2wiki"}
    reserves = sorted(
        [
            r
            for r in source
            if r["dataset"] == "2wiki" and r["split"] == "evaluation" and r["group"] not in used
        ],
        key=lambda r: order("keep:" + r["id"]),
    )
    lc = cfg["local_learning"]
    learning = {
        "E": edits[: lc["episodes"]],
        "R": reserves[: lc["keep_R"]],
        "U": reserves[lc["keep_R"] : lc["keep_R"] + lc["unseen_U"]],
    }
    assert len(learning["U"]) == lc["unseen_U"]
    write(DATA / "cases.json", selected)
    write(DATA / "learning.json", learning)
    write(
        ART / "data-lock.json",
        {
            "config": cfg,
            "config_sha256": digest(CONFIG),
            "source_sha256": digest(source_path),
            "cases_sha256": digest(DATA / "cases.json"),
            "learning_sha256": digest(DATA / "learning.json"),
            "audit": audit,
            "selection": (
                "Fixed hash before outputs; excluded cases are retained without replacement."
            ),
        },
    )
    return audit


def cache_fields(cache, layer):
    obj = cache.layers[layer]
    if hasattr(obj, "recurrent_states"):
        return [("conv_states", 0), ("recurrent_states", 0)]
    return [("keys", None), ("values", None)]


def field_get(obj, name, index):
    value = getattr(obj, name)
    return value if index is None else value[index]


def field_set(obj, name, index, value):
    if index is None:
        setattr(obj, name, value)
    else:
        getattr(obj, name)[index] = value


def matched_random(delta, seed):
    generator = torch.Generator(device=delta.device).manual_seed(seed)
    random = torch.randn(delta.shape, device=delta.device, dtype=delta.dtype, generator=generator)
    return random * (delta.norm() / random.norm().clamp_min(torch.finfo(delta.dtype).tiny))


def change_cache(target, donor, layer, random_seed=None):
    # Keep recipient metadata and update only equal-shaped tensors. Prefix length
    # and token positions are identical across donors in the frozen selection.
    stats = {}
    for i, (name, index) in enumerate(cache_fields(target, layer)):
        a = field_get(target.layers[layer], name, index)
        b = field_get(donor.layers[layer], name, index)
        if a.shape != b.shape:
            raise ValueError(f"Unequal cache shapes for {name}: {a.shape} != {b.shape}")
        delta = b - a
        if random_seed is not None:
            delta = matched_random(delta, random_seed + i)
        stats[name + "_delta_norm"] = float(delta.norm())
        field_set(
            target.layers[layer], name, index, b.clone() if random_seed is None else a + delta
        )
    return stats


class Engine:
    def __init__(self, key, device=None):
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3ForCausalLM

        self.cfg = read(CONFIG)
        self.key, self.mc = key, self.cfg["models"][key]
        self.device = torch.device(device or self.cfg["device"])
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.cfg["gpu_memory_fraction"], self.device)
        cls = Qwen3ForCausalLM if key == "qwen3" else Qwen3_5ForCausalLM
        self.tok = AutoTokenizer.from_pretrained(ROOT / self.mc["path"], local_files_only=True)
        self.model = (
            cls.from_pretrained(
                ROOT / self.mc["path"],
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.layers = self.model.model.layers
        self.stats = Counter()
        self.eos = self.tok.eos_token_id

    def ids(self, text):
        return torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.device)

    def forward(self, ids, cache=None, total=None):
        total = ids.shape[1] if total is None else total
        self.stats["forward_calls"] += 1
        self.stats["input_tokens"] += ids.numel()
        return self.model(
            input_ids=ids,
            attention_mask=torch.ones((1, total), device=self.device, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    @contextlib.contextmanager
    def mlp_hook(self, layer, replacement=None, capture=None):
        def hook(module, args, output):
            if capture is not None:
                capture[layer] = output[0, -1].detach().clone()
            if replacement is None:
                return None
            modified = output.clone()
            modified[0, -1] = replacement
            return modified

        handle = self.layers[layer].mlp.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def prefill(self, text, layer=None, replacement=None):
        ids, captured = self.ids(text), {}
        if ids.shape[1] > self.cfg["max_prefix_tokens"]:
            raise ValueError("Prefix exceeds budget; truncation is forbidden")
        with contextlib.ExitStack() as stack:
            for li in self.mc["layers"]:
                stack.enter_context(
                    self.mlp_hook(li, replacement if li == layer else None, captured)
                )
            out = self.forward(ids)
        return out.past_key_values, captured, ids.shape[1]

    @torch.no_grad()
    def continuation_score(self, logits, cache, total, answer):
        targets = self.tok.encode(" " + answer, add_special_tokens=False) + [self.eos]
        values = []
        for index, token in enumerate(targets):
            values.append(float(logits[0, -1].log_softmax(-1)[token]))
            if index + 1 < len(targets):
                total += 1
                out = self.forward(torch.tensor([[token]], device=self.device), cache, total)
                logits, cache = out.logits, out.past_key_values
        return {"sum_logp": sum(values), "mean_logp": float(np.mean(values)), "tokens": targets}

    @torch.no_grad()
    def finish(self, cache, n_prefix, row, suffix=None):
        suffix_ids = self.ids(self.cfg["suffix"] if suffix is None else suffix)
        total = n_prefix + suffix_ids.shape[1]
        out = self.forward(suffix_ids, cache, total)
        logits, cache = out.logits, out.past_key_values
        scores = {}
        for name, answer in [("correct", row["answer"]), ("competitor", row["competitor"])]:
            scores[name] = self.continuation_score(logits, copy.deepcopy(cache), total, answer)
        scores["margin"] = scores["correct"]["mean_logp"] - scores["competitor"]["mean_logp"]
        tokens = []
        for _ in range(self.cfg["generation_tokens"]):
            token = int(logits[0, -1].argmax())
            tokens.append(token)
            if token == self.eos:
                break
            total += 1
            out = self.forward(torch.tensor([[token]], device=self.device), cache, total)
            logits, cache = out.logits, out.past_key_values
        text = self.tok.decode(tokens, skip_special_tokens=True)
        self.stats["generated_tokens"] += len(tokens)
        return {
            "text": text,
            "tokens": tokens,
            "ended_eos": tokens[-1] == self.eos,
            **grade(text, row["answer"], row.get("unambiguous_aliases", row.get("aliases", []))),
            "scores": scores,
        }

    def direct(self, row, question=None, answer=None):
        qrow = dict(row)
        if answer is not None:
            qrow.update(answer=answer, aliases=[], unambiguous_aliases=[])
        cache, _, length = self.prefill(base_prompt(row, question))
        return self.finish(cache, length, qrow)

    @torch.no_grad()
    def case(self, row):
        started = time.monotonic()
        before_stats = self.stats.copy()
        result = {
            "id": row["id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "group": row["group"],
            "exclusion_reason": row["exclusion_reason"],
            "unassisted": self.direct(row),
            "first_hop": self.direct(row, row["q1"], row["bridge"]),
            "second_hop": self.direct(row, row["q2"]),
            "interventions": [],
        }
        if row["exclusion_reason"]:
            result["seconds"] = time.monotonic() - started
            result["resources"] = dict(self.stats - before_stats)
            return result
        caches, activations, lengths = {}, {}, {}
        for name, bridge in [
            ("correct", row["bridge"]),
            ("corrupt", row["corrupted_bridge"]),
            ("wrong", row["control_bridge"]),
        ]:
            caches[name], activations[name], lengths[name] = self.prefill(prefix(row, bridge))
        if len(set(lengths.values())) != 1:
            raise AssertionError("Token-length matching contract broken")
        n = lengths["corrupt"]
        if n != row["prefix_tokens"][self.key]:
            raise AssertionError("Tokenizer revision changed the frozen prefix length")
        result["oracle_input"] = self.finish(copy.deepcopy(caches["correct"]), n, row)
        for layer in self.mc["layers"]:
            baseline = None
            for condition in self.cfg["conditions"]:
                donor = "wrong" if condition.endswith("wrong") else "correct"
                seed = int(order(row["id"] + str(layer))[:8], 16)
                use_mlp = condition.startswith(("mlp_", "joint_"))
                use_state = condition.startswith(("state_", "joint_"))
                stats = {}
                if use_mlp:
                    delta = activations[donor][layer] - activations["corrupt"][layer]
                    if condition.endswith("random"):
                        delta = matched_random(delta, seed)
                    replacement = (
                        activations["corrupt"][layer] + delta
                        if condition.endswith("random")
                        else activations[donor][layer]
                    )
                    stats["mlp_delta_norm"] = float(delta.norm())
                    cache, _, _ = self.prefill(
                        prefix(row, row["corrupted_bridge"]), layer, replacement
                    )
                else:
                    cache = copy.deepcopy(caches["corrupt"])
                if use_state or condition == "identity":
                    source = caches["corrupt"] if condition == "identity" else caches[donor]
                    stats.update(
                        change_cache(
                            cache, source, layer, seed + 1 if condition.endswith("random") else None
                        )
                    )
                prediction = self.finish(cache, n, row)
                if condition == "baseline":
                    baseline = prediction
                if condition == "identity":
                    if prediction["tokens"] != baseline["tokens"]:
                        raise AssertionError("Identity cache changed generated tokens")
                    if abs(prediction["scores"]["margin"] - baseline["scores"]["margin"]) > 1e-5:
                        raise AssertionError("Identity cache changed scores")
                result["interventions"].append(
                    {
                        "layer": layer,
                        "condition": condition,
                        "cache_kind": self.mc["kind"],
                        **stats,
                        **prediction,
                    }
                )
        result["seconds"] = time.monotonic() - started
        result["resources"] = dict(self.stats - before_stats)
        return result

    def next_logits(self, prompt):
        ids = self.ids(prompt)
        self.stats["forward_calls"] += 1
        self.stats["input_tokens"] += ids.numel()
        if torch.is_grad_enabled():
            self.stats["backward_input_tokens"] += ids.numel()
        return self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]

    def target_loss(self, row):
        prompt = base_prompt(row, row["q1"]) + self.cfg["suffix"]
        x = self.ids(prompt)
        target = self.tok.encode(" " + row["bridge"], add_special_tokens=False) + [self.eos]
        y = torch.tensor([target], device=self.device)
        inputs = torch.cat([x, y[:, :-1]], dim=1)
        self.stats["forward_calls"] += 1
        self.stats["input_tokens"] += inputs.numel()
        self.stats["backward_input_tokens"] += inputs.numel()
        self.stats["supervised_tokens"] += y.numel()
        logits = self.model(input_ids=inputs, use_cache=False, logits_to_keep=len(target)).logits
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())

    def learning(self, episode, output_dir):
        before_stats = self.stats.copy()
        lc = self.cfg["local_learning"]
        pools = read(DATA / "learning.json")
        target = pools["E"][episode]
        parameter = self.layers[lc["layer"]].mlp.down_proj.weight
        original = parameter.detach().clone()
        teacher = {}
        for role in ["R", "U"]:
            with torch.no_grad():
                teacher[role] = [
                    self.next_logits(base_prompt(r, r["q1"]) + self.cfg["suffix"])
                    .log_softmax(-1)
                    .detach()
                    for r in pools[role]
                ]
        result = {"episode": episode, "target_id": target["id"], "nodes": [], "training": []}
        started = time.monotonic()
        try:
            parameter.requires_grad_(True)
            optimizer = torch.optim.SGD([parameter], lr=lc["lr"])
            for step in range(lc["steps"] + 1):
                if step in lc["nodes"]:
                    node = {
                        "step": step,
                        "E": self.direct(target, target["q1"], target["bridge"]),
                        "D": self.direct(target),
                        "R": [],
                        "U": [],
                        "parameter_delta_norm": float((parameter.detach() - original).norm()),
                    }
                    for role in ["R", "U"]:
                        for r, old_lp in zip(pools[role], teacher[role], strict=True):
                            r = dict(r, competitor=target["bridge"])
                            with torch.no_grad():
                                lp = self.next_logits(
                                    base_prompt(r, r["q1"]) + self.cfg["suffix"]
                                ).log_softmax(-1)
                                kl = float((old_lp.exp() * (old_lp - lp)).sum())
                            node[role].append(
                                {
                                    "id": r["id"],
                                    "kl": kl,
                                    **self.direct(r, r["q1"], r["bridge"]),
                                }
                            )
                    result["nodes"].append(node)
                    write(output_dir / "progress.json", result)
                if step == lc["steps"]:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss = self.target_loss(target)
                target_value = float(loss.detach())
                loss.backward()
                replay_value = 0.0
                for r, old_lp in zip(pools["R"], teacher["R"], strict=True):
                    lp = self.next_logits(base_prompt(r, r["q1"]) + self.cfg["suffix"]).log_softmax(
                        -1
                    )
                    kl = (old_lp.exp() * (old_lp - lp)).sum()
                    replay_value += float(kl.detach()) / len(pools["R"])
                    (lc["replay_KL_weight"] * kl / len(pools["R"])).backward()
                norm = float(torch.nn.utils.clip_grad_norm_([parameter], lc["gradient_clip"]))
                if not np.isfinite(target_value + replay_value + norm):
                    raise FloatingPointError("Nonfinite local learning response")
                optimizer.step()
                result["training"].append(
                    {
                        "step": step + 1,
                        "target_ce": target_value,
                        "R_kl": replay_value,
                        "gradient_norm": norm,
                    }
                )
            torch.save(
                {
                    "down_weight": parameter.detach().cpu(),
                    "target_id": target["id"],
                    "layer": lc["layer"],
                    "steps": lc["steps"],
                },
                output_dir / "endpoint.pt",
            )
        finally:
            with torch.no_grad():
                parameter.copy_(original)
            parameter.requires_grad_(False)
        if not torch.equal(parameter, original):
            raise AssertionError("Episode failed to restore original weight")
        result["seconds"] = time.monotonic() - started
        result["restored_original"] = True
        result["resources"] = dict(self.stats - before_stats)
        return result
