"""Frozen-matrix execution and development-only selection for path learning."""

from __future__ import annotations

import json
import os
import platform
import shutil
import time

import numpy as np
import torch
import transformers

from .qwen_path_learning import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    ROOT,
    PathEngine,
    digest,
    encode_fact,
    mean,
    now,
    random_positions,
    read,
    write,
)


def preflight(device):
    from .hebbian_model import aggregate_hash, state_hashes

    cfg = read(CONFIG)
    old_records = read(ROOT / "data/hebbian-learning-v1/pools.json")["B_dev"][:2]
    engine = PathEngine(device)
    encs = []
    for r in old_records:
        enc = encode_fact(engine.tokenizer, r["subject"], r["views"][0], r["answer"])
        enc["key"] = f"preflight-{r['case_id']}"
        encs.append(enc)
    before_hash = aggregate_hash(state_hashes(engine.model))
    assert all(not p.requires_grad for p in engine.model.parameters())
    assert engine.delta.numel() == 3145728
    engine.cache_reference(encs)
    baseline_ce, zero_kl = engine.objectives(encs)
    baseline_ce = baseline_ce.detach()
    checks = {
        "parameter_count": engine.delta.numel(),
        "zero_kl_max": float(zero_kl.detach().abs().max()),
    }
    assert checks["zero_kl_max"] < 1e-5
    for group in cfg["groups"]:
        engine.route, engine.route_group = "group", group
        ce, _ = engine.objectives(encs)
        assert torch.equal(ce.detach(), baseline_ce)
    engine.route = "all"
    with torch.no_grad():
        engine.delta.normal_(std=1e-5)
    ids, attention, _ = engine.batch(encs)
    with torch.no_grad():
        separate = engine.hidden(ids, attention)
        delta = engine.delta.clone()
        original_weight = engine.down.weight.clone()
        engine.down.weight.add_(delta)
        engine.delta.zero_()
        fused = engine.hidden(ids, attention)
        engine.down.weight.copy_(original_weight)
        engine.delta.copy_(delta)
    checks["all_vs_fused_hidden_max_error"] = float((separate - fused).abs().max())
    assert checks["all_vs_fused_hidden_max_error"] < 1e-3
    engine.restore()
    opt = optimizer(engine, 1e-5)
    begun = time.time()
    ce, _ = engine.objectives(encs, include_kl=False)
    ce.mean().backward()
    checks["delta_gradient_norm"] = float(engine.delta.grad.norm())
    torch.nn.utils.clip_grad_norm_([engine.delta], cfg["optimizer"]["clip"])
    opt.step()
    checks["single_old_episode_step_seconds"] = time.time() - begun
    assert checks["delta_gradient_norm"] > 0 and torch.isfinite(engine.delta).all()
    engine.restore()
    assert aggregate_hash(state_hashes(engine.model)) == before_hash
    ce, _ = engine.objectives(encs)
    checks["restore_ce_max_error"] = float((ce.detach() - baseline_ce).abs().max())
    assert checks["restore_ce_max_error"] == 0
    del engine
    torch.cuda.empty_cache()
    final = PathEngine(device, layer=27)
    with torch.no_grad():
        final.delta.normal_(std=1e-4)
        final.route = "zero"
        parent_ce, _ = final.objectives(encs, include_kl=False)
        # Enable only earlier-prompt increments, and disable all readout positions.
        final.route = "positions"
        final.route_positions = {
            e["key"]: list(range(e["answer_start"] - 1, len(e["roles"]))) for e in encs
        }
        early_ce, _ = final.objectives(encs, include_kl=False)
        checks["final_layer_earlier_ce_max_error"] = float((early_ce - parent_ce).abs().max())
        assert checks["final_layer_earlier_ce_max_error"] == 0
    final.restore()
    checks["passed"] = True
    checks["time"] = now()
    checks["training_cases"] = "two old B_dev facts; no new developmental outcome selection"
    checks["parent_model_sha256"] = before_hash
    write(ART / "preflight.json", checks)
    print(json.dumps(checks), flush=True)


def jobs():
    cfg = read(CONFIG)
    return [
        {
            "id": f"dev-e{episode}-lr{lr:g}-b{beta}",
            "episode": episode,
            "lr": lr,
            "beta": beta,
            "seed": 0,
            "phase": "dev",
            "condition": "ordinary",
        }
        for episode in range(4)
        for lr in cfg["learning_rates"]
        for beta in cfg["betas_replay"]
    ]


def sources():
    return [
        "configs/qwen-path-learning-v1.json",
        "scripts/run_qwen_path_learning.py",
        "src/llm_memory_editability/qwen_path_learning.py",
        "src/llm_memory_editability/qwen_path_train.py",
        "src/llm_memory_editability/hebbian_data.py",
        "src/llm_memory_editability/hebbian_model.py",
        "src/llm_memory_editability/hebbian_learning.py",
        "tests/test_qwen_path_learning.py",
    ]


def freeze():
    if (ART / "execution-lock.json").exists():
        raise RuntimeError("Execution lock exists")
    assert read(ART / "preflight.json")["passed"]
    assert read(ART / "selection-lock.json")["selection_sha256"] == digest(DATA / "selection.json")
    frozen = {}
    for relative in sources() + [
        "docs/hebbian-learning-plan-v1.md",
        "docs/experimental-protocol.md",
    ]:
        path = ROOT / relative
        target = ART / "execution-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        frozen[relative] = digest(path)
    cfg = read(CONFIG)
    model = ROOT / cfg["model_source"] / "model.safetensors"
    write(
        ART / "execution-lock.json",
        {
            "time": now(),
            "status": "locked_before_development_updates",
            "sources": frozen,
            "inputs": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [DATA / "selection.json", DATA / "text.json", model]
            },
            "matrix": jobs(),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
            },
            "config": cfg,
        },
    )


def check_lock():
    lock = read(ART / "execution-lock.json")
    for relative in sources():
        assert digest(ROOT / relative) == lock["sources"][relative], relative
    for relative in (
        "data/qwen-path-learning-v1/selection.json",
        "data/qwen-path-learning-v1/text.json",
    ):
        assert digest(ROOT / relative) == lock["inputs"][relative]


def optimizer(engine, lr):
    cfg = engine.cfg["optimizer"]
    return torch.optim.AdamW(
        [engine.delta],
        lr=lr,
        betas=tuple(cfg["betas"]),
        eps=cfg["eps"],
        weight_decay=cfg["weight_decay"],
    )


def schedule(cfg, episode, seed, n_r=64, n_text=64):
    rng = np.random.default_rng(cfg["seed"] + 1000 * episode + seed)
    # Fixed cyclic shuffled decks give exactly equal aggregate exposure across jobs.
    replay, text = rng.permutation(n_r), rng.permutation(n_text)
    return [
        {
            "R": [int(replay[(step * 8 + i) % n_r]) for i in range(8)],
            "text": [int(text[(step * 2 + i) % n_text]) for i in range(2)],
        }
        for step in range(cfg["steps"])
    ]


def save_checkpoint(path, engine, opt, job, step, history, ledger):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(
        {
            "delta": engine.delta.detach().cpu(),
            "optimizer": opt.state_dict(),
            "job": job,
            "step": step,
            "history": history,
            "ledger": ledger,
            "execution_lock": digest(ART / "execution-lock.json"),
        },
        temporary,
    )
    temporary.replace(path)


def evaluate_node(engine, episode, selection, texts, step, output, full_test=False):
    result = {
        "step": step,
        "time": now(),
        "E": engine.evaluate(episode, views=(0, 1, 2, 3), with_kl=False),
        "R": engine.evaluate(selection["R"]),
        "V": engine.evaluate(selection["V"]),
        "text_dev": engine.evaluate_text(texts["dev"]),
        "delta_norm": float(engine.delta.norm()),
        "relative_delta_norm": float(engine.delta.norm() / engine.down.weight.norm()),
    }
    if full_test:
        result["U"] = engine.evaluate(selection["U"], views=(0, 1, 2, 3))
        result["text_test"] = engine.evaluate_text(texts["test"])
    write(output / f"node-{step:03d}.json", result)
    return result


def train_job(engine, job):
    check_lock()
    cfg = engine.cfg
    out = RESULTS / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        return
    engine.restore()
    selection, texts = read(DATA / "selection.json"), read(DATA / "text.json")
    phase = job["phase"]
    records = selection["E_dev" if phase == "dev" else "E_confirm"]
    episode = records[job["episode"] * 4 : (job["episode"] + 1) * 4]
    training_enc = [r["encoded"][0] for r in episode]
    reference_enc = (
        [r["encoded"][0] for role in ("R", "V") for r in selection[role]]
        + texts["train"]
        + texts["dev"]
    )
    if phase == "confirm":
        reference_enc += [enc for r in selection["U"] for enc in r["encoded"]] + texts["test"]
    missing = [e for e in reference_enc if e["key"] not in engine.reference]
    engine.cache_reference(missing)
    if job["condition"] != "ordinary":
        engine.route = "group" if job["condition"] == "candidate" else "random"
        engine.route_group = job["group"]
        engine.route_seed = job["seed"]
    opt = optimizer(engine, job["lr"])
    batches = schedule(cfg, job["episode"], job["seed"])
    write(out / "schedule.json", batches)
    history, ledger, start = (
        [],
        {
            "E_sequences": 0,
            "R_sequences": 0,
            "text_sequences": 0,
            "supervised_E_tokens": 0,
            "R_prediction_tokens": 0,
            "text_prediction_tokens": 0,
            "padded_training_tokens": 0,
        },
        0,
    )
    checkpoint = out / "latest.pt"
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location=engine.device, weights_only=False)
        assert ck["job"] == job and ck["execution_lock"] == digest(ART / "execution-lock.json")
        with torch.no_grad():
            engine.delta.copy_(ck["delta"])
        opt.load_state_dict(ck["optimizer"])
        history, ledger, start = ck["history"], ck["ledger"], ck["step"]
    else:
        evaluate_node(engine, episode, selection, texts, 0, out, phase == "confirm")
        save_checkpoint(out / "step-000.pt", engine, opt, job, 0, history, ledger)
    begun = time.time()
    torch.cuda.reset_peak_memory_stats(engine.device)
    for index in range(start, cfg["steps"]):
        plan = batches[index]
        replay = [selection["R"][i]["encoded"][0] for i in plan["R"]]
        text = [texts["train"][i] for i in plan["text"]]
        opt.zero_grad(set_to_none=True)
        ce, _ = engine.objectives(training_enc, include_kl=False)
        ce.mean().backward()
        _, rkl = engine.objectives(replay)
        (job["beta"] * rkl.mean() / 2).backward()
        _, tkl = engine.objectives(text)
        (job["beta"] * tkl.mean() / 2).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([engine.delta], cfg["optimizer"]["clip"])
        if not torch.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient in {job['id']} step {index + 1}")
        opt.step()
        history.append(
            {
                "step": index + 1,
                "ce": float(ce.detach().mean()),
                "r_kl": float(rkl.detach().mean()),
                "text_kl": float(tkl.detach().mean()),
                "gradient_norm_before_clip": float(gradient_norm),
            }
        )
        ledger["E_sequences"] += len(training_enc)
        ledger["R_sequences"] += len(replay)
        ledger["text_sequences"] += len(text)
        ledger["supervised_E_tokens"] += sum(sum(e["loss_mask"]) for e in training_enc)
        ledger["R_prediction_tokens"] += sum(sum(e["loss_mask"]) for e in replay)
        ledger["text_prediction_tokens"] += sum(sum(e["loss_mask"]) for e in text)
        ledger["padded_training_tokens"] += sum(
            len(es) * max(len(e["input_ids"]) for e in es) for es in (training_enc, replay, text)
        )
        step = index + 1
        if step in cfg["nodes"]:
            evaluate_node(engine, episode, selection, texts, step, out, phase == "confirm")
            save_checkpoint(out / f"step-{step:03d}.pt", engine, opt, job, step, history, ledger)
        if step % 16 == 0 or step in cfg["nodes"]:
            save_checkpoint(checkpoint, engine, opt, job, step, history, ledger)
            write(
                out / "status.json",
                {
                    "time": now(),
                    "state": "training",
                    "step": step,
                    "job": job,
                    "device": str(engine.device),
                    "seconds_this_attempt": time.time() - begun,
                },
            )
            print(json.dumps({"job": job["id"], "step": step, "ce": history[-1]["ce"]}), flush=True)
    write(out / "history.json", history)
    write(
        out / "complete.json",
        {
            "time": now(),
            "job": job,
            "steps": cfg["steps"],
            "ledger": ledger,
            "schedule_sha256": digest(out / "schedule.json"),
            "seconds_this_attempt": time.time() - begun,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(engine.device),
            "device": str(engine.device),
            "trainable_parameters": engine.delta.numel(),
            "parent_down_hash": engine.base_hash,
            "forward_tokens_engine_cumulative": engine.forward_tokens,
            "flops_estimate": {
                "convention": "Approximate dense 2*N per forward token plus backward through "
                "downstream half and head; padding included, attention term omitted; "
                "reference/evaluation costs recorded separately",
                "value": 4
                * sum(p.numel() for p in engine.model.parameters())
                * ledger["padded_training_tokens"],
                "not_hardware_counter": True,
            },
        },
    )
    engine.restore()


def train_shard(device, shard, shards):
    engine = PathEngine(device)
    for index, job in enumerate(jobs()):
        if index % shards == shard:
            train_job(engine, job)


def select_learning_rates():
    cfg = read(CONFIG)
    summaries, chosen = [], {}
    for beta in cfg["betas_replay"]:
        options = []
        for lr in cfg["learning_rates"]:
            records = []
            for job in jobs():
                if job["beta"] == beta and job["lr"] == lr:
                    assert (RESULTS / job["id"] / "complete.json").exists()
                    node = read(RESULTS / job["id"] / "node-128.json")
                    base = read(RESULTS / job["id"] / "node-000.json")
                    records.append(
                        {
                            "E": mean([r for r in node["E"] if r["view"] == 0], "correct"),
                            "V_loss": mean(base["V"], "correct") - mean(node["V"], "correct"),
                            "text_kl": mean(node["text_dev"], "kl"),
                        }
                    )
            option = {
                "beta": beta,
                "lr": lr,
                **{k: mean(records, k) for k in ("E", "V_loss", "text_kl")},
            }
            options.append(option)
        eligible = [r for r in options if r["V_loss"] <= 0.05]
        if eligible:
            choice = min(eligible, key=lambda r: (-r["E"], r["text_kl"], r["lr"]))
        else:
            choice = min(options, key=lambda r: (r["V_loss"], r["text_kl"], r["lr"]))
        chosen[str(beta)] = {**choice, "retention_condition_passed": bool(eligible)}
        summaries.extend(options)
    write(
        ART / "learning-rate-selection.json",
        {"time": now(), "node": 128, "rows": summaries, "chosen": chosen, "used_U": False},
    )
    return chosen


@torch.no_grad()
def diagnose(device, episode_index):
    check_lock()
    cfg = read(CONFIG)
    chosen = read(ART / "learning-rate-selection.json")["chosen"]["1"]
    job = next(
        j
        for j in jobs()
        if j["episode"] == episode_index and j["beta"] == 1 and j["lr"] == chosen["lr"]
    )
    engine = PathEngine(device)
    selection = read(DATA / "selection.json")
    episode = selection["E_dev"][episode_index * 4 : (episode_index + 1) * 4]
    records = episode + selection["R"]
    encs = [r["encoded"][0] for r in records]
    engine.cache_reference(encs)
    output = RESULTS / job["id"] / "diagnostics"
    output.mkdir(exist_ok=True)
    base_E = engine.evaluate(episode)
    zero_R = engine.evaluate(selection["R"], generate=False)
    for step in cfg["nodes"][1:]:
        if (output / f"step-{step:03d}.json").exists():
            continue
        ck = torch.load(
            RESULTS / job["id"] / f"step-{step:03d}.pt",
            map_location=engine.device,
            weights_only=False,
        )
        engine.delta.copy_(ck["delta"])
        engine.route = "all"
        full_E = engine.evaluate(episode)
        full_R = engine.evaluate(selection["R"], generate=False)
        energies = {}
        for start in range(0, len(encs), 8):
            chunk = encs[start : start + 8]
            engine.capture = []
            engine.objectives(chunk, include_kl=False)
            captured = engine.capture[0]
            for i, enc in enumerate(chunk):
                energies[enc["key"]] = captured[i, : len(enc["roles"])].tolist()
            engine.capture = None
        result = {
            "step": step,
            "episode": episode_index,
            "job": job,
            "base_E": base_E,
            "zero_R": zero_R,
            "all_E": full_E,
            "all_R": full_R,
            "groups": {},
        }
        for group in cfg["groups"] + ["L", "A"]:
            engine.route, engine.route_group = "group", group
            group_E = engine.evaluate(episode)
            group_R = engine.evaluate(selection["R"], generate=False)
            group_result = {"E": group_E, "R": group_R, "random": []}
            if group in cfg["groups"]:
                for seed in cfg["matching"]["diagnostic_seeds"]:
                    positions, matches = {}, {}
                    for enc in encs:
                        selected, info = random_positions(
                            enc["roles"],
                            group,
                            enc["key"],
                            seed,
                            cfg["matching"]["position_bins"],
                            cfg["matching"]["draws"],
                            energies[enc["key"]],
                        )
                        info["energy_matched"] = (
                            info["valid"]
                            and info["relative_energy_error"]
                            <= cfg["matching"]["norm_relative_tolerance"]
                        )
                        matches[enc["key"]] = info
                        positions[enc["key"]] = selected
                    engine.route, engine.route_positions = "positions", positions
                    group_result["random"].append(
                        {
                            "seed": seed,
                            "matches": matches,
                            "positions": positions,
                            "E": engine.evaluate(episode),
                            "R": engine.evaluate(selection["R"], generate=False),
                        }
                    )
            result["groups"][group] = group_result
        engine.route = "all"
        recovered = engine.evaluate(episode, generate=False)
        result["restore_max_ce_error"] = max(
            abs(a["ce"] - b["ce"]) for a, b in zip(full_E, recovered, strict=True)
        )
        result["position_energy"] = energies
        write(output / f"step-{step:03d}.json", result)
        print(json.dumps({"diagnostic_episode": episode_index, "step": step}), flush=True)
    engine.restore()


def candidate_decision():
    cfg = read(CONFIG)
    chosen = read(ART / "learning-rate-selection.json")["chosen"]["1"]
    rows = []
    for job in jobs():
        if job["beta"] != 1 or job["lr"] != chosen["lr"]:
            continue
        d = read(RESULTS / job["id"] / "diagnostics/step-128.json")
        gain = mean(d["base_E"], "ce") - mean(d["all_E"], "ce")
        risk = mean(d["all_R"], "kl")
        noise = max(abs(r["kl"]) for r in d["zero_R"])
        for group in cfg["groups"]:
            g = d["groups"][group]
            retained = None if gain <= 0 else (mean(d["base_E"], "ce") - mean(g["E"], "ce")) / gain
            reduction = None if risk <= 0 else 1 - mean(g["R"], "kl") / risk
            matching = []
            for control in g["random"]:
                matched_ids = {
                    int(key.split("-")[1])
                    for key, m in control["matches"].items()
                    if m["energy_matched"]
                }
                e_fraction = sum(r["case_id"] in matched_ids for r in d["all_E"]) / len(d["all_E"])
                r_fraction = sum(r["case_id"] in matched_ids for r in d["all_R"]) / len(d["all_R"])
                all_r = [r for r in d["all_R"] if r["case_id"] in matched_ids]
                cand_r = [r for r in g["R"] if r["case_id"] in matched_ids]
                random_r = [r for r in control["R"] if r["case_id"] in matched_ids]
                difference = None if not all_r else mean(random_r, "kl") - mean(cand_r, "kl")
                matching.append(
                    {
                        "seed": control["seed"],
                        "E_coverage": e_fraction,
                        "R_coverage": r_fraction,
                        "candidate_minus_random_improvement": difference,
                        "valid": e_fraction >= cfg["matching"]["minimum_coverage"]
                        and r_fraction >= cfg["matching"]["minimum_coverage"],
                    }
                )
            valid = [m for m in matching if m["valid"]]
            beats_random = (
                bool(valid)
                and float(np.median([m["candidate_minus_random_improvement"] for m in valid]))
                > cfg["gate"]["noise_multiplier"] * noise
            )
            passes = bool(
                gain > 0
                and mean(d["all_E"], "correct") > mean(d["base_E"], "correct")
                and retained >= cfg["gate"]["learning_retained"]
                and mean(g["E"], "correct") >= mean(d["all_E"], "correct")
                and risk >= max(cfg["gate"]["kl_floor"], cfg["gate"]["noise_multiplier"] * noise)
                and reduction >= cfg["gate"]["kl_reduction"]
                and beats_random
            )
            rows.append(
                {
                    "episode": job["episode"],
                    "group": group,
                    "E_gain": gain,
                    "learning_retained": retained,
                    "R_kl": risk,
                    "R_kl_reduction": reduction,
                    "E_correct_all": mean(d["all_E"], "correct"),
                    "E_correct_candidate": mean(g["E"], "correct"),
                    "noise": noise,
                    "matching": matching,
                    "beats_random": beats_random,
                    "passes": passes,
                }
            )
    eligible = [
        g
        for g in cfg["groups"]
        if sum(r["passes"] for r in rows if r["group"] == g) >= cfg["gate"]["units_required"]
    ]
    winner = (
        min(
            eligible,
            key=lambda g: (
                -float(np.mean([r["R_kl_reduction"] for r in rows if r["group"] == g])),
                cfg["groups"].index(g),
            ),
        )
        if eligible
        else None
    )
    result = {
        "time": now(),
        "rows": rows,
        "candidate": winner,
        "status": "eligible_for_confirmation" if winner else "stop_after_development",
        "confirmation_started": False,
        "selection_uses_U": False,
        "coverage_rule": "At least 80% of E and R must have disjoint count/bin/energy matches "
        "in a fixed random seed; compare candidate/random on identical matched records",
    }
    write(ART / "candidate-decision.json", result)
    return result
