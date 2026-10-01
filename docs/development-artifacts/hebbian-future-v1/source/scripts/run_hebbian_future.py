#!/usr/bin/env python3
"""Execute a fixed exploratory matrix without scientific-success gates."""

from __future__ import annotations

import argparse
import copy
import datetime
import itertools
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_model import CausalLM, ModelConfig, matmul_flops
from llm_memory_editability.hebbian_future import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    ROOT,
    chain_world,
    digest,
    read,
    training_arrays,
    write,
)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def setup(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def prepare():
    if (ART / "lock.json").exists():
        raise FileExistsError("Existing exploratory lock must not be overwritten")
    cfg = read(CONFIG)
    pools = read(ROOT / "data/hebbian-learning-v1/pools.json")
    records = sorted(
        (r for pool in pools.values() for r in pool),
        key=lambda r: (
            __import__("hashlib")
            .sha256(f"{cfg['seed']}:{r['subject_group']}:{r['case_id']}".encode())
            .hexdigest()
        ),
    )
    selected, subjects = [], set()
    for row in records:
        if row["subject_group"] in subjects:
            continue
        subjects.add(row["subject_group"])
        record = {k: v for k, v in row.items() if k != "baseline"}
        record["role"] = "dev" if len(selected) < cfg["real"]["development_facts"] else "eval"
        selected.append(record)
        if len(selected) == cfg["real"]["facts"]:
            break
    assert len(selected) == cfg["real"]["facts"]
    write(DATA / "real-facts.json", selected)
    worlds = []
    for world, rho in itertools.product(cfg["chains"]["worlds"], cfg["chains"]["correlations"]):
        w = chain_world(world, rho, cfg["chains"])
        write(
            DATA / f"world-{world}-rho-{rho}.json",
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in w.items()},
        )
        worlds.append(
            {
                "world": world,
                "rho": rho,
                "train_match": float((w["home_y"] == w["composite_y"])[w["train_people"]].mean()),
                "test_match": float((w["home_y"] == w["composite_y"])[~w["train_people"]].mean()),
            }
        )
    files = [
        CONFIG,
        Path(__file__),
        ROOT / "src/llm_memory_editability/hebbian_future.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "scripts/report_hebbian_future.py",
        ROOT / "tests/test_hebbian_future.py",
        ROOT / "docs/hebbian-learning-plan-v1.md",
        DATA / "real-facts.json",
        ROOT / "data/hebbian-learning-v1/pools.json",
    ]
    files += sorted(DATA.glob("world-*.json"))
    for p in files:
        destination = ART / "source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    write(
        ART / "lock.json",
        {
            "created_at": now(),
            "config": cfg,
            "files": {str(p.relative_to(ROOT)): digest(p) for p in files},
            "world_audit": worlds,
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "historical_data": (
                "Previously accessed audited pools; new subject split and measurements, "
                "not globally untouched confirmation"
            ),
            "scope": "exploration; no effect/mastery/significance gates",
        },
    )
    print("Prepared and locked", flush=True)


@torch.no_grad()
def real(device):
    from llm_memory_editability.hebbian_data import TEMPLATE, score_answer
    from llm_memory_editability.hebbian_model import QwenExperiment

    cfg = read(CONFIG)["real"]
    out = RESULTS / "real"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    started = time.monotonic()
    records = read(DATA / "real-facts.json")
    engine = QwenExperiment(device)
    features = {f"{kind}-{layer}": [] for layer in cfg["layers"] for kind in ("x", "phi")}
    measurements = []
    rows = []
    for i, row in enumerate(records):
        prompts = [TEMPLATE.format(p) for p in row["views"]]
        prompts.append("Background: A concert took place on a rainy evening.\n" + prompts[0])
        for view, prompt in enumerate(prompts):
            rows.append((i, view, prompt))
    for begin in range(0, len(rows), cfg["batch"]):
        part = rows[begin : begin + cfg["batch"]]
        engine.tokenizer.padding_side = "right"
        inp = engine.tokenizer(
            [p for _, _, p in part], add_special_tokens=False, padding=True, return_tensors="pt"
        ).to(device)
        lengths = inp.attention_mask.sum(1) - 1
        ids = torch.arange(len(part), device=device)
        captured, handles = {}, []

        def capture(key, captured=captured, ids=ids, lengths=lengths):
            def hook(module, args):
                captured[key] = args[0][ids, lengths].float().cpu().numpy()

            return hook

        for layer in cfg["layers"]:
            mlp = engine.model.model.layers[layer].mlp
            handles.append(mlp.register_forward_pre_hook(capture(f"x-{layer}")))
            handles.append(mlp.down_proj.register_forward_pre_hook(capture(f"phi-{layer}")))
        try:
            hidden = engine.hidden(inp.input_ids, inp.attention_mask)
            logits = engine.model.lm_head(hidden[ids, lengths]).float()
        finally:
            for handle in handles:
                handle.remove()
        for k, v in captured.items():
            features[k].append(v)
        probability = logits.softmax(-1)
        top_probability, top_tokens = probability.topk(5, dim=-1)
        true = torch.tensor(
            [
                records[i]["encoded"][0]["input_ids"][records[i]["encoded"][0]["answer_start"]]
                for i, _, _ in part
            ],
            device=device,
        )
        masked = logits.clone()
        masked[ids, true] = -torch.inf
        margins = logits[ids, true] - masked.max(-1).values
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1)
        generated, emitted_eos = engine.generate([p for _, _, p in part], cfg["max_new_tokens"])
        for j, (i, view, prompt) in enumerate(part):
            row = records[i]
            measurements.append(
                {
                    "fact": i,
                    "case_id": row["case_id"],
                    "view": view,
                    "role": row["role"],
                    "relation": row["relation_id"],
                    "prompt": prompt,
                    "answer": row["answer"],
                    "aliases": row["aliases"],
                    "true_token": int(true[j]),
                    "prediction_token": int(top_tokens[j, 0]),
                    "top_tokens": top_tokens[j].cpu().tolist(),
                    "top_probabilities": top_probability[j].cpu().tolist(),
                    "true_probability": float(probability[j, true[j]]),
                    "margin": float(margins[j]),
                    "entropy": float(entropy[j]),
                    "generation": generated[j],
                    "emitted_eos": emitted_eos[j],
                    "strict_correct": score_answer(generated[j], row["aliases"]),
                }
            )
        if begin % (cfg["batch"] * 8) == 0:
            print(
                f"real {begin + len(part)}/{len(rows)} elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
    np.savez_compressed(
        out / "features.npz",
        **{k: np.concatenate(v).reshape(len(records), 4, -1) for k, v in features.items()},
    )
    write(out / "measurements.json", measurements)
    write(
        out / "complete.json",
        {
            "time": now(),
            "seconds": time.monotonic() - started,
            "device": device,
            "facts": len(records),
            "queries": len(rows),
            "files": {p.name: digest(p) for p in out.iterdir() if p.suffix in (".json", ".npz")},
        },
    )


def tensor(a, device):
    return torch.as_tensor(a, dtype=torch.long, device=device)


def loss_on(model, x, y):
    inp = torch.cat((x, y[:, None]), 1)
    logits = model(inp)[:, -2:]
    labels = torch.stack((y, torch.full_like(y, 7)), 1)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


@torch.no_grad()
def query(model, x, y):
    logits = model(x)[:, -1]
    pred = logits.argmax(-1)
    continuation = model(torch.cat((x, pred[:, None]), 1))[:, -1].argmax(-1)
    competitors = logits.clone()
    competitors[torch.arange(len(x), device=x.device), y] = -torch.inf
    margin = logits.gather(1, y[:, None]).flatten() - competitors.max(-1).values
    return {
        "correct": pred.eq(y).cpu().numpy(),
        "exact": (pred.eq(y) & continuation.eq(7)).cpu().numpy(),
        "prediction": pred.cpu().numpy(),
        "margin": margin.cpu().numpy(),
    }


def json_arrays(obj):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in obj.items()}


@torch.no_grad()
def evaluate_chain(model, w, device, noise=0.0, noise_seed=0):
    """Noise at the last-position MLP input is an internal perturbation, not text."""
    handles = []
    if noise:

        def make_hook(layer):
            def hook(module, args):
                value = args[0].clone()
                generator = torch.Generator(device=device).manual_seed(
                    9341 + noise_seed * 31 + layer
                )
                z = torch.randn(value[:, -1].shape, generator=generator, device=device)
                z = F.normalize(z, dim=-1) * value[:, -1].norm(dim=-1, keepdim=True) * noise
                value[:, -1] += z
                return (value,)

            return hook

        for layer, block in enumerate(model.blocks):
            handles.append(block.mlp.register_forward_pre_hook(make_hook(layer)))
    try:
        results = {
            kind: query(model, tensor(w[kind + "_x"], device), tensor(w[kind + "_y"], device))
            for kind in ("member", "root", "home", "composite")
        }
        # Predicted membership is passed to the second call; never oracle membership.
        second = w["composite_x"].copy()
        second[:, 1] = results["member"]["prediction"]
        second[:, 2] = 4
        results["two_step"] = query(model, tensor(second, device), tensor(w["composite_y"], device))
    finally:
        for handle in handles:
            handle.remove()
    held = ~w["train_people"]
    conflict = w["home_y"] != w["composite_y"]
    both = results["member"]["correct"] & results["root"]["correct"][w["membership"]]
    summary = {kind + "_accuracy": float(r["exact"].mean()) for kind, r in results.items()}
    for name, mask in {
        "held": held,
        "held_conflict": held & conflict,
        "held_coherent": held & ~conflict,
        "held_common_conflict": held & w["common_conflict"],
        "held_both_known": held & both,
        "held_conflict_both_known": held & conflict & both,
    }.items():
        summary[name + "_n"] = int(mask.sum())
        for kind in ("composite", "two_step"):
            summary[name + "_" + kind] = (
                float(results[kind]["exact"][mask].mean()) if mask.any() else None
            )
        summary[name + "_home_copy"] = (
            float((results["composite"]["prediction"][mask] == w["home_y"][mask]).mean())
            if mask.any()
            else None
        )
    return {
        "noise": noise,
        "noise_seed": noise_seed,
        "summary": summary,
        "rows": {k: json_arrays(v) for k, v in results.items()},
    }


def run_chain(world, rho, depth, device):
    cfg = read(CONFIG)["chains"]
    name = f"w{world}-r{rho}-d{depth}"
    out = RESULTS / "chains" / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    setup(world * 7 + depth)
    w = chain_world(world, rho, cfg)
    x, y = training_arrays(w)
    x, y = tensor(x, device), tensor(y, device)
    model_cfg = ModelConfig(w["vocab"], cfg["width"], depth, cfg["heads"], context=8)
    model = CausalLM(model_cfg).to(device)
    initial_hash = (
        __import__("hashlib")
        .sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters()))
        .hexdigest()
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    started = time.monotonic()
    timeline = []
    for step in range(cfg["steps"] + 1):
        if step in cfg["nodes"]:
            measurement = evaluate_chain(model, w, device)
            measurement["step"] = step
            timeline.append(measurement)
            write(out / "learning.json", timeline)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "step": step,
                    "config": model.config_dict(),
                },
                out / f"model-{step}.pt",
            )
            print(name, step, measurement["summary"], flush=True)
        if step == cfg["steps"]:
            break
        opt.zero_grad(set_to_none=True)
        loss = loss_on(model, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
    training_seconds = time.monotonic() - started
    robustness = [
        evaluate_chain(model, w, device, noise, seed)
        for noise in cfg["noise"]
        for seed in (cfg["noise_seeds"] if noise else [0])
    ]
    write(out / "robustness.json", robustness)
    parent = copy.deepcopy(model.state_dict())
    rehearsals = []
    # Same examples and budget, differing relative loss weights, always from same parent.
    bx = [tensor(w[k + "_x"], device) for k in ("member", "root", "home")]
    by = [tensor(w[k + "_y"], device) for k in ("member", "root", "home")]
    arm_weights = {
        "chain": [0.45, 0.45, 0.1],
        "competitor": [0.1, 0.1, 0.8],
        "balanced": [1 / 3, 1 / 3, 1 / 3],
    }
    for arm in cfg["rehearsal_arms"]:
        model.load_state_dict(parent)
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"] * 0.1, weight_decay=cfg["weight_decay"]
        )
        trajectory = []
        for step in range(cfg["rehearsal_steps"] + 1):
            if step in cfg["rehearsal_nodes"]:
                measure = evaluate_chain(model, w, device)
                measure["step"] = step
                trajectory.append(measure)
            if step == cfg["rehearsal_steps"]:
                break
            opt.zero_grad(set_to_none=True)
            for weight, xx, yy in zip(arm_weights[arm], bx, by, strict=True):
                (weight * loss_on(model, xx, yy)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        rehearsals.append({"arm": arm, "weights": arm_weights[arm], "trajectory": trajectory})
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "config": model.config_dict(),
                "step": cfg["rehearsal_steps"],
            },
            out / f"rehearsal-{arm}.pt",
        )
    write(out / "rehearsals.json", rehearsals)
    write(
        out / "complete.json",
        {
            "world": world,
            "rho": rho,
            "depth": depth,
            "device": device,
            "time": now(),
            "seconds": time.monotonic() - started,
            "training_seconds_including_node_eval": training_seconds,
            "initial_hash": initial_hash,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_presentations": len(x) * cfg["steps"],
            "training_input_tokens": len(x) * 5 * cfg["steps"],
            "training_supervised_tokens": len(x) * 2 * cfg["steps"],
            "training_flops_estimate": matmul_flops(model_cfg, len(x), 5, 5) * cfg["steps"],
            "rehearsal_input_tokens": sum(len(xx) for xx in bx) * 5 * cfg["rehearsal_steps"] * 3,
            "files": {p.name: digest(p) for p in out.iterdir() if p.suffix in (".json", ".pt")},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "real", "chains"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    setup(read(CONFIG)["seed"])
    if args.stage == "prepare":
        prepare()
    elif args.stage == "real":
        real(args.device)
    else:
        cfg = read(CONFIG)["chains"]
        jobs = list(itertools.product(cfg["worlds"], cfg["correlations"], cfg["layers"]))
        for i, (world, rho, depth) in enumerate(jobs):
            if i % args.shards == args.shard:
                run_chain(world, rho, depth, args.device)


if __name__ == "__main__":
    main()
