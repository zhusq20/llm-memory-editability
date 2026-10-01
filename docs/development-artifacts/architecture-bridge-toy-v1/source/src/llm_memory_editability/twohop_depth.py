"""Depth, held-out composition, and bridge interventions in random fact worlds.

Entity IDs and relation IDs are disjoint. A relation is a random permutation;
this makes every atom and each split's answer marginal exactly balanced.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_model import CausalLM, ModelConfig

ROOT = Path(os.environ.get("TWOHOP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = ROOT / "configs/twohop-depth-v1.json"
DATA = ROOT / "data/twohop-depth-v1"
RESULTS = ROOT / "results/twohop-depth-v1"
ART = ROOT / "docs/development-artifacts/twohop-depth-v1"
BOS, EOS, ENTITY = 1, 2, 3


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


def world_arrays(seed, cfg):
    n, r = cfg["entities"], cfg["relations"]
    if r % 2 or r < 2:
        raise ValueError("An even relation count is required for balanced coverage")
    rng = np.random.default_rng(seed)
    maps = np.stack([rng.permutation(n) for _ in range(r)])
    # For each bridge, a regular bipartite graph over first/second relations.
    # Each atom occurs in r/2 training compositions in EACH hop role.
    orders = np.stack([rng.permutation(r) for _ in range(n)])
    mask = np.zeros((n, r, r), dtype=bool)
    for a in range(n):
        for r1 in range(r):
            b = maps[r1, a]
            mask[a, r1, orders[b, (r1 + np.arange(r // 2)) % r]] = True
    atoms = np.array([(a, rel) for a in range(n) for rel in range(r)], dtype=np.int64)
    comps = np.array(
        [(a, r1, r2) for a in range(n) for r1 in range(r) for r2 in range(r)],
        dtype=np.int64,
    )
    a, r1, r2 = comps.T
    bridge = maps[r1, a]
    answer = maps[r2, bridge]
    selected = rng.choice(np.flatnonzero(~mask.ravel()), cfg["diagnostic_cases"], replace=False)
    cases = []
    inverse = np.argsort(maps, axis=1)
    for index in selected:
        a0, r10, r20 = comps[index]
        b0, c0 = bridge[index], answer[index]
        # Donor contains only a first-hop prefix; neither its explicit entity nor
        # its first-hop answer is the counterfactual final answer.
        candidates = []
        for b1 in range(n):
            a1, c1 = inverse[r10, b1], maps[r20, b1]
            if b1 != b0 and c1 not in {a1, b1, a0, b0, c0}:
                candidates.append((int(a1), int(b1), int(c1)))
        if not candidates:
            raise ValueError("No answer-free bridge donor; increase world size")
        a1, b1, c1 = candidates[int(rng.integers(len(candidates)))]
        same_r = (r10 + 1) % r
        same_a = inverse[same_r, b0]
        wrong_candidates = [
            (rr, int(maps[rr, b1])) for rr in range(r) if rr != r20 and maps[rr, b1] not in {c0, c1}
        ]
        if not wrong_candidates:
            raise ValueError("No distinct wrong-relation answer")
        wrong_r, wrong_y = wrong_candidates[int(rng.integers(len(wrong_candidates)))]
        cases.append([index, a0, r10, r20, b0, c0, a1, b1, c1, same_a, same_r, wrong_r, wrong_y])
    ax = np.c_[np.full(len(atoms), BOS), ENTITY + atoms[:, 0], ENTITY + n + atoms[:, 1]]
    cx = np.c_[np.full(len(comps), BOS), ENTITY + a, ENTITY + n + r1, ENTITY + n + r2]
    return {
        "maps": maps,
        "atoms": atoms,
        "comps": comps,
        "train_mask": mask.ravel(),
        "atomic_x": ax,
        "atomic_y": ENTITY + maps[atoms[:, 1], atoms[:, 0]],
        "composite_x": cx,
        "composite_y": ENTITY + answer,
        "bridge": ENTITY + bridge,
        "cases": np.asarray(cases, dtype=np.int64),
    }


def audit_world(w, cfg):
    n, r = cfg["entities"], cfg["relations"]
    maps, comps = w["maps"], w["comps"]
    assert all(np.array_equal(np.sort(row), np.arange(n)) for row in maps)
    assert len(set(map(tuple, w["atomic_x"]))) == n * r
    assert len(set(map(tuple, w["composite_x"]))) == n * r * r
    for mask in (w["train_mask"], ~w["train_mask"]):
        a, r1, r2 = comps[mask].T
        b = maps[r1, a]
        assert np.all(np.bincount(a * r + r1, minlength=n * r) == r // 2)
        assert np.all(np.bincount(b * r + r2, minlength=n * r) == r // 2)
        assert np.all(np.bincount(w["composite_y"][mask] - ENTITY, minlength=n) == r * r // 2)
    assert np.all(~w["train_mask"][w["cases"][:, 0]])
    for _, a, r1, r2, b, c, da, db, dc, sa, sr, wr, wy in w["cases"]:
        assert maps[r1, a] == b and maps[r2, b] == c
        assert maps[r1, da] == db and maps[r2, db] == dc
        assert maps[sr, sa] == b and sr != r1
        assert dc not in {da, db, a, b, c}
        assert maps[wr, db] == wy and wr != r2 and wy not in {c, dc}
    return {
        "atoms": n * r,
        "train_compositions": int(w["train_mask"].sum()),
        "test_compositions": int((~w["train_mask"]).sum()),
        "diagnostic_cases": len(w["cases"]),
        "every_atom_train_composition_occurrences_per_hop": r // 2,
        "balanced_answers": True,
        "answer_free_donors": True,
    }


def stream_indices(length, count, rng):
    return np.concatenate([rng.permutation(length) for _ in range((count + length - 1) // length)])[
        :count
    ]


def make_stream(w, seed, cfg):
    rng = np.random.default_rng(seed)
    count = cfg["steps"] * cfg["batch_per_kind"]
    return {
        "atomic": stream_indices(len(w["atomic_x"]), count, rng),
        "composite": stream_indices(int(w["train_mask"].sum()), count, rng),
    }


def build_model(arch, cfg):
    return CausalLM(
        ModelConfig(
            vocab_size=ENTITY + cfg["entities"] + cfg["relations"],
            layers=arch["layers"],
            width=arch["width"],
            heads=arch["heads"],
            context=8,
        )
    )


def amp(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


def training_tensors(w, device):
    result = {}
    for name, mask in (("atomic", slice(None)), ("composite", w["train_mask"])):
        x, y = w[name + "_x"][mask], w[name + "_y"][mask]
        full = np.zeros((len(x), 6), dtype=np.int64)
        full[:, : x.shape[1]] = x
        full[:, x.shape[1]] = y
        full[:, x.shape[1] + 1] = EOS
        pos = np.tile([x.shape[1] - 1, x.shape[1]], (len(x), 1))
        labels = np.c_[y, np.full(len(x), EOS)]
        result[name] = tuple(torch.as_tensor(v, device=device) for v in (full, pos, labels))
    return result


def optimizer_for(model, cfg, device):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=str(device).startswith("cuda"),
    )


def train_step(model, optimizer, tensors, indices, step, cfg, device):
    parts = [tuple(v[indices[k]] for v in tensors[k]) for k in ("atomic", "composite")]
    x, pos, labels = (torch.cat([p[j] for p in parts]) for j in range(3))
    lr = cfg["lr"] * min(1.0, (step + 1) / cfg["warmup"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    with amp(device):
        logits = model(x, positions=pos)
        losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none")
        loss = losses.mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
    optimizer.step()
    return loss.detach(), norm.detach()


@torch.no_grad()
def predict(model, x, y, device, batch=512):
    pred, eos, prob = [], [], []
    for start in range(0, len(x), batch):
        q = torch.as_tensor(x[start : start + batch], device=device)
        target = torch.as_tensor(y[start : start + batch], device=device)
        with amp(device):
            logits = model(q)[:, -1].float()
            answer = logits.argmax(-1)
            stop = model(torch.cat([q, answer[:, None]], dim=1))[:, -1].argmax(-1)
        pred.extend(answer.cpu().tolist())
        eos.extend(stop.cpu().tolist())
        prob.extend(logits.log_softmax(-1).gather(1, target[:, None]).squeeze(1).cpu().tolist())
    return np.asarray(pred), np.asarray(eos), np.asarray(prob)


@torch.no_grad()
def evaluate(model, w, cfg, device):
    arrays, metrics = {}, {}
    for name, key, mask in (
        ("atomic", "atomic", slice(None)),
        ("train", "composite", w["train_mask"]),
        ("test", "composite", ~w["train_mask"]),
    ):
        x, y = w[key + "_x"][mask], w[key + "_y"][mask]
        pred, eos, logp = predict(model, x, y, device, cfg["eval_batch"])
        arrays.update({name + "_pred": pred, name + "_eos": eos, name + "_logp": logp})
        metrics.update(
            {
                name + "_answer": float(np.mean(pred == y)),
                name + "_exact": float(np.mean((pred == y) & (eos == EOS))),
                name + "_nll": float(-logp.mean()),
            }
        )
    # Autonomous two-call baseline uses its own generated bridge, never the oracle.
    test_ids = np.flatnonzero(~w["train_mask"])
    a, r1, r2 = w["comps"][test_ids].T
    bridge_pred = arrays["atomic_pred"][a * cfg["relations"] + r1]
    bridge_eos = arrays["atomic_eos"][a * cfg["relations"] + r1]
    second_x = np.c_[np.full(len(a), BOS), bridge_pred, ENTITY + cfg["entities"] + r2]
    y = w["composite_y"][test_ids]
    pred, eos, logp = predict(model, second_x, y, device, cfg["eval_batch"])
    arrays.update(
        two_call_pred=pred, two_call_eos=eos, two_call_logp=logp, two_call_bridge=bridge_pred
    )
    two_exact = (pred == y) & (eos == EOS) & (bridge_eos == EOS)
    metrics["two_call_exact"] = float(two_exact.mean())
    atom_ok = (arrays["atomic_pred"] == w["atomic_y"]) & (arrays["atomic_eos"] == EOS)
    b = w["maps"][r1, a]
    both = atom_ok[a * cfg["relations"] + r1] & atom_ok[b * cfg["relations"] + r2]
    arrays["test_both_atoms_correct"] = both
    exact = (arrays["test_pred"] == y) & (arrays["test_eos"] == EOS)
    metrics["test_both_atoms_n"] = int(both.sum())
    metrics["test_given_both_atoms"] = float(exact[both].mean()) if both.any() else None
    return metrics, arrays


@torch.no_grad()
def states_and_logits(model, tokens, device):
    # Mirrors bios_model.CausalLM.forward; contract-tested against ordinary forward.
    with amp(device):
        h = model.token(tokens) + model.position(torch.arange(tokens.shape[1], device=device))
        states = [h.clone()]
        for block in model.blocks:
            h = block(h)
            states.append(h.clone())
        logits = F.linear(model.ln_final(h), model.token.weight)
    return states, logits.float()


@torch.no_grad()
def patched_logits(model, tokens, layer, positions, replacement, device):
    with amp(device):
        h = model.token(tokens) + model.position(torch.arange(tokens.shape[1], device=device))
        if layer == 0:
            h[:, positions] = replacement.to(h.dtype)
        for number, block in enumerate(model.blocks, 1):
            h = block(h)
            if number == layer:
                h[:, positions] = replacement.to(h.dtype)
        return F.linear(model.ln_final(h), model.token.weight)[:, -1].float()


@torch.no_grad()
def diagnostics(model, w, cfg, device):
    cases = w["cases"]
    q = torch.as_tensor(w["composite_x"][cases[:, 0]], device=device)
    dx = np.c_[
        np.full(len(cases), BOS), ENTITY + cases[:, 6], ENTITY + cfg["entities"] + cases[:, 2]
    ]
    sx = np.c_[
        np.full(len(cases), BOS), ENTITY + cases[:, 9], ENTITY + cfg["entities"] + cases[:, 10]
    ]
    clean, logits = states_and_logits(model, q, device)
    donor, _ = states_and_logits(model, torch.as_tensor(dx, device=device), device)
    same, _ = states_and_logits(model, torch.as_tensor(sx, device=device), device)
    old = torch.as_tensor(ENTITY + cases[:, 5], device=device)
    new = torch.as_tensor(ENTITY + cases[:, 8], device=device)
    wrong = torch.as_tensor(ENTITY + cases[:, 12], device=device)
    row = torch.arange(len(q), device=device)
    baseline = logits[:, -1]
    out = {
        "case_ids": cases[:, 0],
        "clean_pred": baseline.argmax(-1).cpu().numpy(),
        "clean_margin": (baseline[row, new] - baseline[row, old]).cpu().numpy(),
        "old_y": old.cpu().numpy(),
        "new_y": new.cpu().numpy(),
        "wrong_y": wrong.cpu().numpy(),
    }
    # Fixed random orthogonal coordinate permutation preserves perturbation norm.
    rng = np.random.default_rng(142899)
    perm = torch.as_tensor(rng.permutation(model.config.width), device=device)
    signs = torch.as_tensor(rng.choice([-1, 1], model.config.width), device=device)
    for layer in range(len(clean)):
        with amp(device):
            lens = F.linear(model.ln_final(clean[layer]), model.token.weight).float()
        out[f"lens_bridge_pos2_{layer}"] = lens[:, 2].argmax(-1).cpu().numpy()
        out[f"lens_query_pos3_{layer}"] = lens[:, 3].argmax(-1).cpu().numpy()
        delta = donor[layer][:, 2] - clean[layer][:, 2]
        random_state = clean[layer][:, 2] + delta[:, perm] * signs
        conditions = {
            "bridge": ([2], donor[layer][:, [2]]),
            "same_bridge": ([2], same[layer][:, [2]]),
            "random": ([2], random_state[:, None]),
            "source": ([1], donor[layer][:, [1]]),
            "prefix": ([1, 2], donor[layer][:, [1, 2]]),
            "identity": ([2], clean[layer][:, [2]]),
        }
        for name, (positions, replacement) in conditions.items():
            z = patched_logits(model, q, layer, positions, replacement, device)
            prefix = f"{name}_{layer}"
            out[prefix + "_pred"] = z.argmax(-1).cpu().numpy()
            out[prefix + "_margin"] = (z[row, new] - z[row, old]).cpu().numpy()
            out[prefix + "_max_delta"] = (z - baseline).abs().max(-1).values.cpu().numpy()
    return out


def run_name(world, seed, architecture):
    return f"w{world}-s{seed}-{architecture}"
