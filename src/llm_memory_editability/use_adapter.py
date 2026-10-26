"""use-adapter-development-v1: facts written one by one as entity-form per-fact parameters (the
saved knowledge-read-interface-v2 E3 Structural channels, frozen), plus usage learned once by a
shared full-parameter update of the released weights trained on usage practice for a few facts.

Arms (identical except the extra F_adapt data):
- p0: E3 Structural alone, no further training;
- pu: shared update with E4-U's usage practice for the practiced facts (F_adapt within the E3
  set) + U0 single-hop replay of all E3 facts + E4's KL replay;
- pp: the same with E4-P's single-hop paraphrases for the practiced facts instead.

Training reproduces the E4 injection loop (`read_interface_stages.inject`: 8 new-fact plus 8 KL
replay examples per step, answer-token CE, KL to the frozen released model on replay, AdamW
3e-6, 10-step warmup, clipping at 1.0) for a fixed number of steps (E4-U's actual count) and
without the U0 stopping rule, which the E3 writes already satisfy.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch

from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_e3 as E3
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S
from llm_memory_editability import read_interface_v2 as V2
from llm_memory_editability import read_interface_v2_data as V

BATCH_NAME = "use-adapter-development-v1"
ARMS = ("p0", "pu", "pp")
TRAINED = ("pu", "pp")
LR = 3e-6
HALF = 8
CHECK_EVERY = 20
SEED = S.SEED + 2  # the E4 injection seed


# ---------------------------------------------------------------- data


def e3_facts(v2_run):
    """The 50 E3-written facts in their write order."""
    facts = {f.fid: f for f in S.facts_from(v2_run)}
    order = S.read(Path(v2_run) / "e3" / "structural.json")["order"]
    return [facts[fid] for fid in order]


def practiced(facts):
    return [f for f in facts if f.split == "adapt"]


def u0_replay(facts):
    """U0 single-hop recall with the gold value for every written fact (both arms)."""
    return [{"user": D.fill(D.U0, f) + D.DIRECT, "assistant": f.v} for f in facts]


def extra_examples(facts, arm):
    """E4-U usage practice (pu) or E4-P single-hop paraphrases (pp) for the practiced facts."""
    if arm == "p0":
        return []
    make = {"pu": V.practice_examples, "pp": V.paraphrase_examples}[arm]
    return [{"user": u, "assistant": a} for f in practiced(facts) for u, a in make(f)]


def arm_examples(facts, arm):
    return u0_replay(facts) + extra_examples(facts, arm)


# ---------------------------------------------------------------- frozen E3 writes


def load_adapters(path):
    """Saved E3 Structural writes: per layer, per fact, {g, u, w} (CPU tensors)."""
    frozen = torch.load(path, map_location="cpu", weights_only=True)
    for layer in frozen:
        for entry in layer:
            assert set(entry) == set(E3.KEYS["structural"]), entry.keys()
    return frozen


def attach(model, frozen, device="cuda"):
    """Install the Structural wrappers and put the saved writes in as frozen tensors (they are
    not parameters, so no optimizer over `model.parameters()` can change them)."""
    wrappers = E3.install(model, "structural")
    assert len(wrappers) == len(frozen)
    for w, layer in zip(wrappers, frozen, strict=True):
        w.frozen = [{k: v.detach().to(device).clone() for k, v in f.items()} for f in layer]
        w.current = None
        w.mask = None
    return wrappers


def base_name(name):
    return name.replace(".mlp.mlp.", ".mlp.")


def trainable(model):
    """Every released weight of the model (the shared update); the E3 writes are excluded."""
    params = [p for _, p in model.named_parameters()]
    for layer in M.layers(model):
        if isinstance(layer.mlp, E3.Adapted):
            assert layer.mlp.current is None
            ids = {id(t) for f in layer.mlp.frozen for t in f.values()}
            assert not any(id(p) in ids for p in params)
    return params


@torch.no_grad()
def reset_base(model, reference):
    """Copy the released weights back into `model`, leaving its E3 writes in place."""
    ref = dict(reference.named_parameters())
    for name, p in model.named_parameters():
        p.copy_(ref[base_name(name)].to(p.device, p.dtype))


# ---------------------------------------------------------------- shared update


def step_loss(model, teacher, tok, chunk, device="cuda"):
    """E4's loss: CE on answer tokens of new-fact examples, KL to the frozen released model on
    replay examples, summed over answer tokens and divided by their number."""
    texts = [M.chat(tok, e["user"], False) for e in chunk]
    answers = [e["assistant"] + "<|im_end|>" for e in chunk]
    ids, mask, _, slices = M.encode(tok, texts, answers=answers)
    ids, mask = ids.to(device), mask.to(device)
    with M.autocast():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    loss = 0.0
    distil = [i for i, e in enumerate(chunk) if e.get("replay")]
    kl_tokens = ce_tokens = 0
    if distil:
        t_texts = [M.chat(tok, chunk[i]["user"], False, chunk[i].get("system")) for i in distil]
        t_answers = [answers[i] for i in distil]
        t_ids, t_mask, _, t_slices = M.encode(tok, t_texts, answers=t_answers)
        with torch.no_grad(), M.autocast():
            t_logits = teacher(
                input_ids=t_ids.to(device), attention_mask=t_mask.to(device), use_cache=False
            ).logits
        for k, i in enumerate(distil):
            (a, b), (ta, tb) = slices[i], t_slices[k]
            assert b - a == tb - ta
            p_t = torch.log_softmax(t_logits[k, ta - 1 : tb - 1].float(), -1)
            p_s = torch.log_softmax(logits[i, a - 1 : b - 1], -1)
            loss = loss + (p_t.exp() * (p_t - p_s)).sum(-1).sum()
            kl_tokens += b - a
    for i, (a, b) in enumerate(slices):
        if i in distil:
            continue
        labels = ids[i, a:b]
        loss = loss + torch.nn.functional.cross_entropy(
            logits[i, a - 1 : b - 1], labels, reduction="sum"
        )
        ce_tokens += b - a
    n_tok = sum(b - a for a, b in slices)
    return loss / n_tok, {
        "kl_tokens": kl_tokens,
        "ce_tokens": ce_tokens,
        "input_tokens": int(mask.sum()),
        "teacher_tokens": int(t_mask.sum()) if distil else 0,
    }


def u0_teacher_forced(model, tok, facts):
    return S.u0_accuracy(model, tok, facts)


def train(model, teacher, tok, examples, replay, steps, lr=LR, log=None, check=None, device="cuda"):
    """Fixed-length shared update of every released weight; `check(model)` (teacher-forced U0
    of the written facts) is logged every CHECK_EVERY steps but never stops training."""
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    params = trainable(model)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    stream = S.train_batches(examples, replay, rng, half=HALF)
    counts = {"kl_tokens": 0, "ce_tokens": 0, "input_tokens": 0, "teacher_tokens": 0}
    history = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        chunk = next(stream)
        loss, c = step_loss(model, teacher, tok, chunk, device)
        for k, v in c.items():
            counts[k] += v
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, step / 10)
        optimizer.step()
        if step % CHECK_EVERY == 0 or step == steps:
            row = {"step": step, "loss": float(loss), "seconds": time.time() - t0}
            if check is not None:
                model.eval()
                row["u0_teacher_forced"] = check(model)
                model.train()
            history.append(row)
            if log:
                log(row)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in params)
    supervised = counts["kl_tokens"] + counts["ce_tokens"]
    return {
        "lr": lr,
        "steps": steps,
        "examples": len(examples),
        "replay_pool": len(replay),
        "examples_seen": steps * HALF,
        "replay_seen": steps * HALF,
        "supervised_tokens": supervised,
        **counts,
        "trainable_parameters": n,
        "seconds": time.time() - t0,
        "flops_estimate_e4_rule": 6 * n * supervised,
        "flops_estimate_processed": 6 * n * counts["input_tokens"]
        + 2 * n * counts["teacher_tokens"],
        "history": history,
    }


# ---------------------------------------------------------------- evaluation


def op_eval(model, tok, facts):
    """E4's operation matrix (continent, language, currency; direct and step-by-step)."""
    rows = []
    for f in facts:
        if f.split == "calibration":
            continue
        rows += [{**r, "split": f.split} for r in V.op_queries(f)]
    return V2.answer_rows(model, tok, rows)
