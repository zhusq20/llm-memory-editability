"""Paired formation/adaptation, prospective diagnostics and exact state resumption."""

from __future__ import annotations

import itertools
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .hebbian_learning import (
    ARTIFACTS,
    CONFIG_PATH,
    DATA,
    RESULTS,
    ROOT,
    config,
    effective_rank,
    now,
    q_auc,
    read_json,
    sha256,
    write_json,
)
from .hebbian_model import (
    QwenExperiment,
    aggregate_hash,
    set_determinism,
    state_hashes,
)


def grouping(records, seed):
    """Exact minimum-cost derangements; random seeded ties, real CE labels unchanged."""
    n = len(records)
    rng = np.random.default_rng(seed)
    permutations = [
        p for p in itertools.permutations(range(n)) if all(i != j for i, j in enumerate(p))
    ]

    def cost(p):
        return sum(
            10 * (records[i]["target_id"] == records[j]["target_id"])
            + (records[i]["relation_id"] != records[j]["relation_id"])
            for i, j in enumerate(p)
        )

    costs = np.array([cost(p) for p in permutations])
    best = np.flatnonzero(costs == costs.min())
    p1, p2 = [permutations[int(rng.choice(best))] for _ in range(2)]
    groups = [(3 * i, 3 * p1[i] + 1, 3 * p2[i] + 2) for i in range(n)]
    return groups


def contrastive_loss(features, target_ids, groups, tau=0.1):
    z = F.normalize(features, dim=-1, eps=1e-8)
    similarities = z @ z.T / tau
    positive = torch.zeros_like(similarities, dtype=torch.bool)
    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    positive[i, j] = True
    negative = torch.tensor([[a != b for b in target_ids] for a in target_ids], device=z.device)
    negative &= ~positive
    negative.fill_diagonal_(False)
    assert torch.all(positive.sum(1) == 2)
    assert torch.all(negative.sum(1) >= 1)
    allowed = positive | negative
    log_denominator = similarities.masked_fill(~allowed, -torch.inf).logsumexp(1)
    loss = (log_denominator - (similarities * positive).sum(1) / positive.sum(1)).mean()
    return loss, {"positive_pairs": int(positive.sum()), "negative_pairs": int(negative.sum())}


def optimizer(parameters, lr):
    cfg = config()["optimizer"]
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"],
    )


def replay_batch(pools, texts, step):
    facts = pools["R_keep"]
    return [facts[(step * 6 + i) % len(facts)]["encoded"][0] for i in range(6)] + [
        texts["train"][(step * 2 + i) % len(texts["train"])] for i in range(2)
    ]


def frozen_training_sources():
    paths = [CONFIG_PATH] + sorted((ROOT / "src/llm_memory_editability").glob("hebbian*.py"))
    paths += [ROOT / "scripts/run_hebbian_learning.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def save_checkpoint(path, engine, opt, step, meta, ledger):
    state = {
        "parameters": {n: p.detach().cpu().clone() for n, p in engine.mlp.named_parameters()},
        "optimizer": opt.state_dict(),
        "step": step,
        "data_cursor": step,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(engine.device)
        if engine.device.type == "cuda"
        else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "meta": meta,
        "ledger": ledger,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def restore_checkpoint(path, engine, opt, meta):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["meta"] != meta:
        raise RuntimeError(f"Checkpoint contract differs: {path}")
    with torch.no_grad():
        for n, p in engine.mlp.named_parameters():
            p.copy_(state["parameters"][n])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], engine.device)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    return state["step"], state["ledger"]


def load_parent(engine, parent_path=None):
    engine.restore_base()
    if parent_path is not None:
        state = torch.load(parent_path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for n, p in engine.mlp.named_parameters():
                p.copy_(state["parameters"][n])


def gradient_cos(a, b):
    a, b = a.double(), b.double()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def diagnose_episode(engine, records, replay, lr, out):
    """Saved before the learning trajectory; Adam candidate uses training data only."""
    engine.configure_trainable("adapt")
    p = engine.mlp.down_proj.weight
    original = p.detach().clone()
    encoded = [r["encoded"][v] for r in records for v in range(3)]
    features = engine.features(encoded).double().cpu().numpy().reshape(len(records), 3, -1)
    unit = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    p0_gram = unit[:, 0] @ unit[:, 0].T
    rank = effective_rank(features[:, 0])
    opt = optimizer([p], lr)
    opt.zero_grad(set_to_none=True)
    (engine.ce([r["encoded"][0] for r in records[:8]]).mean() + engine.kl(replay).mean()).backward()
    torch.nn.utils.clip_grad_norm_([p], 1.0)
    opt.step()
    delta = (p.detach() - original).clone()
    with torch.no_grad():
        p.copy_(original)
    torch.save(
        {"delta_B": delta.cpu(), "phi": torch.from_numpy(features)}, out / "initial-diagnostics.pt"
    )
    diagnostic = []
    for i, record in enumerate(records):
        grads, losses = zip(*(engine.gradient(enc) for enc in record["encoded"]), strict=True)
        other = [
            p0_gram[i, j] for j, r in enumerate(records) if r["target_id"] != record["target_id"]
        ]
        if not other:
            raise ValueError("Each episode needs different-answer facts")
        norms = np.linalg.norm(features[i], axis=1)
        feature_values = [
            losses[0],
            record["encoded"][0]["answer_tokens"],
            record["encoded"][0]["prompt_tokens"],
            float(grads[0].double().norm()),
            float(np.mean(unit[i, 1:] @ unit[i, 0])),
            float(max(other)),
            float(norms[1:].mean() / norms[0]),
            rank,
            float(np.mean([gradient_cos(grads[0], g) for g in grads[1:]])),
            float(np.mean([float((g.double() * delta.double()).sum()) for g in grads[1:]])),
        ]
        diagnostic.append(
            {
                "case_id": record["case_id"],
                "features": feature_values,
                "p0_p1_cos": float(unit[i, 0] @ unit[i, 1]),
                "p0_p2_cos": float(unit[i, 0] @ unit[i, 2]),
                "feature_norms": norms.tolist(),
                "initial_nll": list(losses),
            }
        )
    engine.model.zero_grad(set_to_none=True)
    assert torch.equal(p, original)
    write_json(
        out / "prospective-features.json",
        {
            "saved_at": now(),
            "rows": diagnostic,
            "information": {
                "P0": "initial answer NLL, lengths, labeled training gradient norm",
                "P1": "P0 plus prompt-only phi geometry",
                "P2": "P1 plus labeled evaluation gradients and training-only Adam candidate",
            },
        },
    )
    return diagnostic


def first_order_calibration(engine, record, out):
    parameter = engine.mlp.down_proj.weight
    initial = parameter.detach().clone()
    source_grad, _ = engine.gradient(record["encoded"][0])
    rows = []
    for view in [1, 2]:
        encoded = record["encoded"][view]
        grad, nll0 = engine.gradient(encoded)
        ids, attention, _ = engine.batch([encoded], prompt_only=True)
        answer_token = encoded["input_ids"][encoded["answer_start"]]
        logits = engine.model.lm_head(engine.hidden(ids, attention)[0, -1])
        competitor_logits = logits.detach().clone()
        competitor_logits[answer_token] = -torch.inf
        competitor = int(competitor_logits.argmax())
        margin = logits[answer_token] - logits[competitor]
        margin_grad = torch.autograd.grad(margin, parameter)[0].detach()
        margin0 = float(margin.detach())
        for relative_step in [1e-5, 1e-4]:
            proposal = (
                -source_grad
                * (relative_step * initial.double().norm() / source_grad.double().norm()).float()
            )
            with torch.no_grad():
                parameter.copy_(initial + proposal)
                actual_delta = parameter - initial
                actual_nll = float(engine.ce([encoded])[0]) - nll0
                new_logits = engine.model.lm_head(engine.hidden(ids, attention)[0, -1])
                actual_margin = float(new_logits[answer_token] - new_logits[competitor]) - margin0
                predicted_nll = float((grad.double() * actual_delta.double()).sum())
                predicted_margin = float((margin_grad.double() * actual_delta.double()).sum())
                parameter.copy_(initial)
            entry = {
                "case_id": record["case_id"],
                "view_id": view,
                "relative_step": relative_step,
                "competitor_token": competitor,
                "actual_nll_change": actual_nll,
                "predicted_nll_change": predicted_nll,
                "actual_margin_change": actual_margin,
                "predicted_margin_change": predicted_margin,
            }
            for name, actual, predicted in [
                ("nll", actual_nll, predicted_nll),
                ("margin", actual_margin, predicted_margin),
            ]:
                entry[name + "_absolute_error"] = abs(actual - predicted)
                entry[name + "_relative_error"] = abs(actual - predicted) / max(
                    abs(predicted), 1e-6
                )
                entry[name + "_sign_match"] = bool(np.sign(actual) == np.sign(predicted))
            rows.append(entry)
    assert torch.equal(parameter, initial)
    write_json(out / "first-order-calibration.json", rows)


def geometry_rows(engine, records):
    phi = engine.features([r["encoded"][v] for r in records for v in range(3)]).double()
    phi = phi.reshape(len(records), 3, -1)
    unit = F.normalize(phi, dim=-1)
    original = engine.original_features.double().reshape(len(records), 3, -1)
    original_cos = (unit * F.normalize(original, dim=-1)).sum(-1)
    gram = unit[:, 0] @ unit[:, 0].T
    rank = float(gram.trace().square() / gram.square().sum())
    result = {}
    for i, r in enumerate(records):
        others = [
            float(gram[i, j])
            for j, other in enumerate(records)
            if other["target_id"] != r["target_id"]
        ]
        result[r["case_id"]] = {
            "same_fact_cos": float((unit[i, 1:] @ unit[i, 0]).mean()),
            "other_fact_cos": max(others) if others else None,
            "effective_rank": rank,
            "cosine_to_original_phi": float(original_cos[i].mean()),
            "dimension_variance": float(phi[:, 0].var(0, unbiased=False).mean()),
        }
    return result


def evaluate_node(engine, records, keep, texts, step, metadata, out, exposures, full_keep):
    rows = engine.evaluate(records)
    geometry = geometry_rows(engine, records)
    for row in rows:
        row.update(
            metadata,
            step=step,
            exposures=exposures.get(row["case_id"], 0),
            **geometry[row["case_id"]],
        )
    payload = {"rows": rows}
    if full_keep:
        payload["keep"] = engine.evaluate(keep, views=(0,))
        with torch.no_grad():
            text_losses = []
            for i in range(0, len(texts["test"]), 8):
                text_losses.extend(engine.ce(texts["test"][i : i + 8]).tolist())
        payload["text_nll"] = float(np.mean(text_losses))
        payload["text_ppl"] = float(np.exp(payload["text_nll"]))
    write_json(out / f"evaluation-{step:04d}.json", payload)
    return payload


def train_run(
    engine,
    phase,
    records,
    pools,
    texts,
    condition,
    seed,
    episode_id,
    split,
    lr,
    steps,
    run_id,
    lambda_geo=0.0,
    parent_path=None,
    diagnostics=False,
):
    cfg = config()
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    load_parent(engine, parent_path)
    parent_hashes = state_hashes(engine.model)
    parent_hash = aggregate_hash(parent_hashes)
    parameters = engine.configure_trainable(phase)
    trainable = [n for n, p in engine.model.named_parameters() if p.requires_grad]
    meta = {
        "phase": phase,
        "split": split,
        "condition": condition,
        "seed": seed,
        "episode_id": episode_id,
        "parent_hash": parent_hash,
        "parent_path": str(parent_path) if parent_path else None,
        "source_hashes": frozen_training_sources(),
        "config_sha256": sha256(CONFIG_PATH),
        "data_lock_sha256": sha256(ARTIFACTS / "data-lock.json"),
        "trainable_names": trainable,
        "lr": lr,
        "lambda_geo": lambda_geo,
        "steps": steps,
        "case_ids": [r["case_id"] for r in records],
    }
    if (out / "complete.json").exists():
        receipt = read_json(out / "complete.json")
        if receipt["meta"] != meta:
            raise RuntimeError(f"Completed run contract changed: {run_id}")
        return receipt
    engine.make_reference()
    set_determinism(seed)
    random.seed(seed)
    opt = optimizer(parameters, lr)
    write_json(out / "contract.json", meta)
    ledger = {
        "answer_tokens": 0,
        "answer_input_tokens": 0,
        "replay_tokens": 0,
        "replay_input_tokens": 0,
        "geometry_input_tokens": 0,
        "forward_sequences": 0,
        "backward_sequences": 0,
        "optimizer_steps": 0,
        "wall_seconds": 0.0,
    }
    nodes = [n for n in cfg["form" if phase == "form" else "adapt"]["nodes"] if n <= steps]
    if steps not in nodes:
        nodes.append(steps)
    keep = pools["V_keep" if split == "dev" else "U_keep"]
    evaluate_records = pools["V_form"] if phase == "form" and split == "dev" else records
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records)).tolist() if phase == "form" else list(range(len(records)))
    exposures = {r["case_id"]: 0 for r in records}
    batch_table = []
    for step in range(steps):
        batch = [records[order[(step * 8 + i) % len(order)]] for i in range(8)]
        groups = (
            grouping(batch, seed * 100000 + step)
            if condition == "C2"
            else [tuple(range(3 * i, 3 * i + 3)) for i in range(8)]
        )
        batch_table.append({"case_ids": [r["case_id"] for r in batch], "groups": groups})
    write_json(out / "batch-manifest.json", batch_table)
    checkpoints = sorted(out.glob("checkpoint-*.pt"))
    start = 0
    if checkpoints:
        start, ledger = restore_checkpoint(checkpoints[-1], engine, opt, meta)
        for item in batch_table[:start]:
            for case_id in item["case_ids"]:
                exposures[case_id] += 1
    elif diagnostics:
        diagnostic = diagnose_episode(engine, records, replay_batch(pools, texts, 0), lr, out)
        if split == "eval":
            from .hebbian_statistics import predict

            predictors = read_json(ARTIFACTS / "B-lock.json")["predictors"]
            write_json(
                out / "prospective-predictions.json",
                {
                    "saved_at": now(),
                    "parent_hash": parent_hash,
                    "predictions": [
                        {
                            "case_id": r["case_id"],
                            "values": {
                                name: float(predict(model, np.array(r["features"])[None])[0])
                                for name, model in predictors.items()
                            },
                        }
                        for r in diagnostic
                    ],
                },
            )
            first_order_calibration(engine, min(records, key=lambda r: r["case_id"]), out)
    metadata = {
        k: meta[k] for k in ["phase", "split", "condition", "seed", "episode_id", "parent_hash"]
    }
    if start == 0:
        save_checkpoint(out / "checkpoint-0000.pt", engine, opt, 0, meta, ledger)
    for step in range(start, steps + 1):
        if step in nodes:
            evaluate_node(
                engine,
                evaluate_records,
                keep,
                texts,
                step,
                metadata,
                out,
                exposures,
                full_keep=(phase == "form" or step in [0, 16, 64, 128]),
            )
            if phase == "adapt" and condition.startswith("C") and step in [0, 16, 128]:
                gradients = []
                for r in sorted(records, key=lambda r: r["case_id"])[:4]:
                    grads = [engine.gradient(e)[0] for e in r["encoded"]]
                    gradients.append(
                        {
                            "case_id": r["case_id"],
                            "p0_p1_gradient_cos": gradient_cos(grads[0], grads[1]),
                            "p0_p2_gradient_cos": gradient_cos(grads[0], grads[2]),
                        }
                    )
                write_json(out / f"gradients-{step:04d}.json", gradients)
            if step in [0, 16, 64, 128, 256]:
                save_checkpoint(out / f"checkpoint-{step:04d}.pt", engine, opt, step, meta, ledger)
        if step == steps:
            break
        batch = [records[order[(step * 8 + i) % len(order)]] for i in range(8)]
        encoded = [r["encoded"][v] for r in batch for v in (range(3) if phase == "form" else [0])]
        replay = replay_batch(pools, texts, step)
        torch.cuda.synchronize(engine.device)
        begin = time.monotonic()
        opt.zero_grad(set_to_none=True)
        ce = engine.ce(encoded).mean()
        ce.backward()
        kl = engine.kl(replay).mean()
        kl.backward()
        geo_value, pair_counts = 0.0, None
        if phase == "form":
            features = engine.features(encoded, detach=False)
            targets = [r["target_id"] for r in batch for _ in range(3)]
            geo, pair_counts = contrastive_loss(features, targets, batch_table[step]["groups"])
            if lambda_geo:
                (lambda_geo * geo).backward()
            geo_value = float(geo.detach())
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["optimizer"]["clip"])
        opt.step()
        torch.cuda.synchronize(engine.device)
        ledger["wall_seconds"] += time.monotonic() - begin
        ledger["answer_tokens"] += sum(sum(e["loss_mask"]) for e in encoded)
        ledger["answer_input_tokens"] += sum(len(e["input_ids"]) for e in encoded)
        ledger["replay_tokens"] += sum(sum(e["loss_mask"]) for e in replay)
        ledger["replay_input_tokens"] += sum(len(e["input_ids"]) for e in replay)
        if phase == "form":
            ledger["geometry_input_tokens"] += sum(e["answer_start"] for e in encoded)
        ledger["forward_sequences"] += (
            len(encoded) + 2 * len(replay) + (len(encoded) if phase == "form" else 0)
        )
        ledger["backward_sequences"] += (
            len(encoded) + len(replay) + (len(encoded) if lambda_geo else 0)
        )
        ledger["optimizer_steps"] += 1
        for r in batch:
            exposures[r["case_id"]] += 1
        with (out / "training.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step + 1,
                        "ce": float(ce.detach()),
                        "kl": float(kl.detach()),
                        "geo": geo_value,
                        "gradient_norm": float(norm),
                        "pairs": pair_counts,
                    }
                )
                + "\n"
            )
        write_json(
            out / "status.json", {"step": step + 1, "steps": steps, "time": now(), "ledger": ledger}
        )
        if (step + 1) % 16 == 0:
            print(
                run_id,
                step + 1,
                "ce",
                float(ce.detach()),
                "wall",
                ledger["wall_seconds"],
                flush=True,
            )
    final_hashes = state_hashes(engine.model)
    unchanged = all(
        final_hashes[n] == digest for n, digest in parent_hashes.items() if n not in trainable
    )
    assert unchanged, "A parameter outside the intervention scope changed"
    receipt = {
        "meta": meta,
        "completed_at": now(),
        "ledger": ledger,
        "unchanged_outside_scope": unchanged,
        "final_hashes": final_hashes,
        "final_model_hash": aggregate_hash(final_hashes),
        "exposures": exposures,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "estimated_dense_full_backprop_flops_upper_bound": (
            sum(p.numel() for p in engine.model.parameters())
            * (
                6 * ledger["answer_input_tokens"]
                + 8 * ledger["replay_input_tokens"]
                + (6 if lambda_geo else 2) * ledger["geometry_input_tokens"]
            )
        ),
    }
    write_json(out / "complete.json", receipt)
    return receipt


def run_summary(path):
    evaluations = [read_json(p) for p in sorted(path.glob("evaluation-*.json"))]
    nodes = [e["rows"][0]["step"] for e in evaluations]
    rewrite = [
        np.mean([r["answer_em"] for r in e["rows"] if r["view_id"] in [1, 2]]) for e in evaluations
    ]
    keep0 = np.mean([r["answer_em"] for r in evaluations[0]["keep"]])
    keep1 = np.mean([r["answer_em"] for r in evaluations[-1]["keep"]])
    return {
        "q_auc": float(q_auc(rewrite, nodes)),
        "keep_damage": float(keep0 - keep1),
        "answer_nll": float(np.mean([r["answer_nll"] for r in evaluations[-1]["rows"]])),
    }


def select_config(candidates, metric, maximize):
    safe = [
        r for r in candidates if r["keep_damage"] <= config()["statistics"]["keep_damage_reference"]
    ]
    if safe:
        return sorted(safe, key=lambda r: (-1 if maximize else 1) * r[metric])[0]
    return sorted(
        candidates, key=lambda r: (r["keep_damage"], (-1 if maximize else 1) * r[metric])
    )[0]


def preflight(engine, pools, texts):
    records = pools["B_dev"][:8]
    audit = engine.verify_hooks_and_gradients(records[0]["encoded"][0])
    before = state_hashes(engine.model)
    parameters = engine.configure_trainable("adapt")
    engine.make_reference()
    opt = optimizer(parameters, 1e-5)
    torch.cuda.reset_peak_memory_stats(engine.device)
    torch.cuda.synchronize(engine.device)
    start = time.monotonic()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        engine.ce([r["encoded"][0] for r in records]).mean().backward()
        engine.kl(replay_batch(pools, texts, step)).mean().backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        opt.step()
    torch.cuda.synchronize(engine.device)
    seconds = (time.monotonic() - start) / 3
    engine.restore_base()
    assert before == state_hashes(engine.model)
    estimate = {
        "time": now(),
        "device": str(engine.device),
        "adapt_step_seconds": seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(engine.device),
        "formal_adapt_steps": 11264,
        "formal_adapt_gpu_hours": seconds * 11264 / 3600,
        "formation_estimate_factor": 3,
        "formal_formation_gpu_hours_estimate": seconds * 3 * 2304 / 3600,
        "excludes": [
            "generation and NLL evaluation",
            "gradient diagnostics",
            "development",
            "disk IO",
        ],
        "microbatch_answer": 8,
        "microbatch_replay": 8,
        "reference_parent_restored": True,
    }
    write_json(ARTIFACTS / "estimate.json", estimate)
    write_json(ARTIFACTS / "qwen-interface-audit.json", audit)
    print(estimate, flush=True)


def dispatch(args):
    if not (ARTIFACTS / "data-lock.json").exists():
        raise RuntimeError("data-lock is required before development or evaluation")
    data_lock = read_json(ARTIFACTS / "data-lock.json")
    for name, digest in data_lock["files"].items():
        if sha256(DATA / name) != digest:
            raise RuntimeError(f"Frozen data changed: {name}")
    if not read_json(ARTIFACTS / "A/audit.json")["pass"]:
        raise RuntimeError("Stage A must pass before model training")
    pools = read_json(DATA / "pools.json")
    texts = read_json(DATA / "text.json")
    episodes = read_json(DATA / "episodes.json")
    by_id = {r["case_id"]: r for rows in pools.values() for r in rows}
    engine = QwenExperiment(args.device)
    if args.command == "preflight":
        return preflight(engine, pools, texts)
    if args.command == "diagnose":
        if args.split == "dev":
            learning_rates = config()["adapt"]["learning_rates"]
        else:
            learning_rates = [read_json(ARTIFACTS / "B-lock.json")["learning_rate"]]
        tasks = list(itertools.product(learning_rates, enumerate(episodes["B_" + args.split])))
        for lr, (episode, ids) in tasks[args.shard :: args.shards]:
            run_id = f"B/{args.split}-lr{lr:g}-e{episode}"
            train_run(
                engine,
                "adapt",
                [by_id[i] for i in ids],
                pools,
                texts,
                "B",
                0,
                episode,
                args.split,
                lr,
                128,
                run_id,
                diagnostics=True,
            )
        return
    b_lr = read_json(ARTIFACTS / "B-lock.json")["learning_rate"]
    if args.command == "form" and args.split == "dev":
        candidates = []
        for lr in config()["form"]["learning_rates"]:
            run_id = f"C/dev-C0-lr{lr:g}"
            train_run(
                engine, "form", pools["F_form"], pools, texts, "C0", 10, -1, "dev", lr, 256, run_id
            )
            candidates.append(
                {"learning_rate": lr, "run_id": run_id, **run_summary(RESULTS / run_id)}
            )
        chosen = select_config(candidates, "answer_nll", False)
        lr = chosen["learning_rate"]
        formation_candidates = [{"condition": "C0", "lambda_geo": 0.0, **chosen}]
        for lam in config()["form"]["lambdas"]:
            run_id = f"C/dev-C1-lambda{lam:g}"
            train_run(
                engine,
                "form",
                pools["F_form"],
                pools,
                texts,
                "C1",
                10,
                -1,
                "dev",
                lr,
                256,
                run_id,
                lambda_geo=lam,
            )
            formation_candidates.append(
                {
                    "condition": "C1",
                    "lambda_geo": lam,
                    "run_id": run_id,
                    **run_summary(RESULTS / run_id),
                }
            )
        for item in formation_candidates:
            adapt_results = []
            for episode, ids in enumerate(episodes["C_dev"]):
                run_id = f"C/adapt-dev-{item['condition']}-lambda{item['lambda_geo']:g}-e{episode}"
                train_run(
                    engine,
                    "adapt",
                    [by_id[i] for i in ids],
                    pools,
                    texts,
                    item["condition"],
                    10,
                    episode,
                    "dev",
                    b_lr,
                    64,
                    run_id,
                    parent_path=RESULTS / item["run_id"] / "checkpoint-0256.pt",
                )
                adapt_results.append(run_summary(RESULTS / run_id))
            item["future_q_auc"] = float(np.mean([r["q_auc"] for r in adapt_results]))
            item["adapt_results"] = adapt_results
        best = select_config(
            [r for r in formation_candidates if r["condition"] == "C1"], "future_q_auc", True
        )
        lam = best["lambda_geo"]
        train_run(
            engine,
            "form",
            pools["F_form"],
            pools,
            texts,
            "C2",
            10,
            -1,
            "dev",
            lr,
            256,
            "C/dev-C2-engineering",
            lambda_geo=lam,
        )
        write_json(
            ARTIFACTS / "C-lock.json",
            {
                "time": now(),
                "learning_rate": lr,
                "lambda_geo": lam,
                "adapt_learning_rate": b_lr,
                "lr_candidates": candidates,
                "formation_candidates": formation_candidates,
                "source_hashes": frozen_training_sources(),
                "data_lock_sha256": sha256(ARTIFACTS / "data-lock.json"),
                "seeds": [0, 1, 2],
                "conditions": ["C0", "C1", "C2"],
                "episodes": episodes["C_eval"],
                "status": "locked",
            },
        )
        return
    c_lock = read_json(ARTIFACTS / "C-lock.json")
    if args.command == "form":
        tasks = list(itertools.product(["C0", "C1", "C2"], [0, 1, 2]))
        for condition, seed in tasks[args.shard :: args.shards]:
            train_run(
                engine,
                "form",
                pools["F_form"],
                pools,
                texts,
                condition,
                seed,
                -1,
                "eval",
                c_lock["learning_rate"],
                256,
                f"C/form-{condition}-s{seed}",
                lambda_geo=0.0 if condition == "C0" else c_lock["lambda_geo"],
            )
    elif args.command == "adapt":
        tasks = list(
            itertools.product(["C0", "C1", "C2"], [0, 1, 2], enumerate(episodes["C_eval"]))
        )
        for condition, seed, (episode, ids) in tasks[args.shard :: args.shards]:
            train_run(
                engine,
                "adapt",
                [by_id[i] for i in ids],
                pools,
                texts,
                condition,
                seed,
                episode,
                "eval",
                b_lr,
                128,
                f"C/adapt-{condition}-s{seed}-e{episode}",
                parent_path=RESULTS / f"C/form-{condition}-s{seed}/checkpoint-0256.pt",
            )
