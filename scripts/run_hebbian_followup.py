#!/usr/bin/env python3
"""Second-round full-answer and matched-budget experiments."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
from run_hebbian_future import evaluate_chain, loss_on, now, setup, tensor
from torch.nn import functional as F

from llm_memory_editability.bios_model import CausalLM, ModelConfig, matmul_flops
from llm_memory_editability.hebbian_followup import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    candidate_set,
    fixed_world,
    inventory_from_facts,
    parse_entity,
    stable,
)
from llm_memory_editability.hebbian_future import ROOT, digest, read, training_arrays, write


def prepare():
    if (ART / "lock.json").exists():
        raise FileExistsError("Do not replace an existing experimental lock")
    from llm_memory_editability.hebbian_data import TEMPLATE
    from llm_memory_editability.qwen_path_learning import RELATIONS

    cfg = read(CONFIG)
    pools = read(ROOT / "data/hebbian-learning-v1/pools.json")
    all_facts = [r for pool in pools.values() for r in pool]
    inventory = inventory_from_facts(all_facts)
    excluded = {r["subject_group"] for r in read(ROOT / "data/hebbian-future-v1/real-facts.json")}
    ordered = sorted(
        all_facts, key=lambda r: stable(f"{cfg['seed']}:{r['subject_group']}:{r['case_id']}")
    )
    selected, seen = [], set(excluded)
    for row in ordered:
        if row["subject_group"] in seen:
            continue
        seen.add(row["subject_group"])
        r = {k: v for k, v in row.items() if k not in ("baseline", "encoded")}
        r["role"] = "dev" if len(selected) < cfg["real"]["development_facts"] else "eval"
        templates = list(RELATIONS[r["relation_id"]])
        if r["relation_id"] in ("P20", "P740"):
            templates[0], templates[1] = templates[1], templates[0]
        r["prompts"] = [t.format(r["subject"]) for t in templates]
        r["prompts"].append(TEMPLATE.format(r["prompts"][0]))
        r["candidates"] = candidate_set(r, inventory, cfg["seed"])
        r["candidate_labels"] = [inventory[r["relation_id"]][e]["label"] for e in r["candidates"]]
        selected.append(r)
        if len(selected) == cfg["real"]["facts"]:
            break
    assert len(selected) == cfg["real"]["facts"]
    write(DATA / "facts.json", selected)
    write(DATA / "inventory.json", inventory)
    for world, rho in itertools.product(cfg["chains"]["worlds"], cfg["chains"]["correlations"]):
        w = fixed_world(world, rho, cfg["chains"])
        write(
            DATA / f"world-{world}-rho{rho}.json",
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in w.items()},
        )
    files = [
        CONFIG,
        Path(__file__),
        ROOT / "scripts/run_hebbian_future.py",
        ROOT / "scripts/report_hebbian_followup.py",
        ROOT / "scripts/execute_hebbian_followup.py",
        ROOT / "src/llm_memory_editability/hebbian_followup.py",
        ROOT / "src/llm_memory_editability/hebbian_future.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "src/llm_memory_editability/hebbian_model.py",
        ROOT / "src/llm_memory_editability/hebbian_data.py",
        ROOT / "src/llm_memory_editability/qwen_path_learning.py",
        ROOT / "tests/test_hebbian_followup.py",
        ROOT / "docs/hebbian-learning-plan-v1.md",
    ]
    files += sorted(DATA.glob("*.json"))
    for p in files:
        q = ART / "source" / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(p.read_bytes())
    write(
        ART / "lock.json",
        {
            "time": now(),
            "config": cfg,
            "files": {str(p.relative_to(ROOT)): digest(p) for p in files},
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "previous_round_subject_overlap": len(
                excluded.intersection(r["subject_group"] for r in selected)
            ),
            "history": (
                "Already audited/historically accessed pools; subjects exclude v1, "
                "new prompts and outcomes"
            ),
            "no_scientific_gate": True,
        },
    )
    print("prepared", len(selected), "facts and 108 training jobs", flush=True)


@torch.no_grad()
def candidate_scores(engine, prompts, labels, batch_size):
    from llm_memory_editability.hebbian_data import encode_answer

    pairs = [
        (prompt, label)
        for prompt, choices in zip(prompts, labels, strict=True)
        for label in choices
    ]
    output = []
    for start in range(0, len(pairs), batch_size):
        enc = [encode_answer(engine.tokenizer, p, a) for p, a in pairs[start : start + batch_size]]
        # Score every answer token; no EOS because these are natural continuations.
        for e in enc:
            e["loss_mask"][-1] = 0
        ids, mask, loss_mask = engine.batch(enc)
        hidden = engine.hidden(ids, mask)
        b, t = torch.where(loss_mask[:, 1:])
        logits = engine.model.lm_head(hidden[b, t]).float()
        losses = F.cross_entropy(logits, ids[b, t + 1], reduction="none")
        sums = torch.zeros(len(enc), device=engine.device).scatter_add_(0, b, losses)
        count = torch.bincount(b, minlength=len(enc))
        output.extend(
            zip(
                (-sums).cpu().tolist(),
                (-sums / count).cpu().tolist(),
                count.cpu().tolist(),
                strict=True,
            )
        )
    return np.array(output).reshape(len(prompts), 4, 3)


@torch.no_grad()
def real(device):
    from llm_memory_editability.hebbian_data import score_answer
    from llm_memory_editability.hebbian_model import QwenExperiment

    cfg = read(CONFIG)["real"]
    directory = RESULTS / "real"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "complete.json").exists():
        return
    engine = QwenExperiment(device)
    facts, inventory = read(DATA / "facts.json"), read(DATA / "inventory.json")
    rows = [(i, v, p) for i, r in enumerate(facts) for v, p in enumerate(r["prompts"])]
    features = {f"{kind}-{layer}": [] for layer in cfg["layers"] for kind in ("x", "phi")}
    outcomes, started = [], time.monotonic()
    for start in range(0, len(rows), cfg["batch"]):
        chunk = rows[start : start + cfg["batch"]]
        prompts = [p for _, _, p in chunk]
        engine.tokenizer.padding_side = "right"
        inputs = engine.tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        lengths = inputs.attention_mask.sum(1) - 1
        indices = torch.arange(len(chunk), device=device)
        captures, handles = {}, []

        def hook_for(key, captures=captures, indices=indices, lengths=lengths):
            def hook(module, args):
                captures[key] = args[0][indices, lengths].cpu().numpy()

            return hook

        for layer in cfg["layers"]:
            mlp = engine.model.model.layers[layer].mlp
            handles.append(mlp.register_forward_pre_hook(hook_for(f"x-{layer}")))
            handles.append(mlp.down_proj.register_forward_pre_hook(hook_for(f"phi-{layer}")))
        try:
            hidden = engine.hidden(inputs.input_ids, inputs.attention_mask)
            logits = engine.model.lm_head(hidden[indices, lengths]).float()
            probs = logits.softmax(-1)
            entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
        finally:
            for h in handles:
                h.remove()
        for key, value in captures.items():
            features[key].append(value)
        scores = candidate_scores(
            engine,
            prompts,
            [facts[i]["candidate_labels"] for i, _, _ in chunk],
            cfg["candidate_batch"],
        )
        generations, eos = engine.generate(prompts, cfg["max_new_tokens"])
        for j, (i, view, prompt) in enumerate(chunk):
            f = facts[i]
            parsed = parse_entity(generations[j], inventory[f["relation_id"]])
            outcomes.append(
                {
                    "fact": i,
                    "case_id": f["case_id"],
                    "role": f["role"],
                    "view": view,
                    "prompt": prompt,
                    "relation": f["relation_id"],
                    "target": f["target_id"],
                    "candidates": f["candidates"],
                    "labels": f["candidate_labels"],
                    "logp_sum": scores[j, :, 0].tolist(),
                    "logp_mean": scores[j, :, 1].tolist(),
                    "answer_tokens": scores[j, :, 2].astype(int).tolist(),
                    "choice_sum": f["candidates"][int(scores[j, :, 0].argmax())],
                    "choice_mean": f["candidates"][int(scores[j, :, 1].argmax())],
                    "entropy": float(entropy[j]),
                    "top1_probability": float(probs[j].max()),
                    "generation": generations[j],
                    "eos": eos[j],
                    "parsed": parsed,
                    "strict_correct": score_answer(generations[j], f["aliases"]),
                }
            )
        if start % 128 == 0:
            print(
                "real",
                start + len(chunk),
                "/",
                len(rows),
                "seconds",
                round(time.monotonic() - started, 1),
                flush=True,
            )
    write(directory / "measurements.json", outcomes)
    np.savez_compressed(
        directory / "features.npz",
        **{k: np.concatenate(v).reshape(len(facts), 4, -1) for k, v in features.items()},
    )
    write(
        directory / "complete.json",
        {
            "time": now(),
            "seconds": time.monotonic() - started,
            "queries": len(rows),
            "candidate_sequences": 4 * len(rows),
            "device": device,
            "files": {
                p.name: digest(p) for p in directory.iterdir() if p.suffix in (".json", ".npz")
            },
        },
    )


def chains(device, shard, shards):
    cfg = read(CONFIG)["chains"]
    jobs = list(
        itertools.product(
            cfg["worlds"], cfg["initializations"], cfg["correlations"], cfg["architectures"]
        )
    )
    for index, (world, init, rho, arch) in enumerate(jobs):
        if index % shards != shard:
            continue
        name = f"w{world}-i{init}-r{rho}-{arch['name']}"
        out = RESULTS / "chains" / name
        out.mkdir(parents=True, exist_ok=True)
        if (out / "complete.json").exists():
            continue
        setup(world * 101 + init * 10007 + arch["depth"])
        w = fixed_world(world, rho, cfg)
        x, y = training_arrays(w)
        x, y = tensor(x, device), tensor(y, device)
        mc = ModelConfig(w["vocab"], arch["width"], arch["depth"], arch["heads"], context=8)
        model = CausalLM(mc).to(device)
        initial = hashlib.sha256(
            b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
        ).hexdigest()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        trajectory, started = [], time.monotonic()
        for step in range(cfg["steps"] + 1):
            if step in cfg["nodes"]:
                result = evaluate_chain(model, w, device)
                result["step"] = step
                trajectory.append(result)
                write(out / "learning.json", trajectory)
                if step in (0, 1024, cfg["steps"]):
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": opt.state_dict(),
                            "config": model.config_dict(),
                            "step": step,
                        },
                        out / f"model-{step}.pt",
                    )
                print(
                    name,
                    "step",
                    step,
                    "held",
                    result["summary"]["held_composite"],
                    "focal",
                    result["summary"]["held_common_conflict_composite"],
                    flush=True,
                )
            if step == cfg["steps"]:
                break
            opt.zero_grad(set_to_none=True)
            loss_on(model, x, y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        write(
            out / "complete.json",
            {
                "world": world,
                "init": init,
                "rho": rho,
                "architecture": arch["name"],
                "parameters": sum(p.numel() for p in model.parameters()),
                "initial_hash": initial,
                "seconds": time.monotonic() - started,
                "device": device,
                "input_tokens": len(x) * 5 * cfg["steps"],
                "flops_estimate": matmul_flops(mc, len(x), 5, 5) * cfg["steps"],
                "files": {p.name: digest(p) for p in out.iterdir() if p.suffix in (".json", ".pt")},
            },
        )


if __name__ == "__main__":
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
        chains(args.device, args.shard, args.shards)
