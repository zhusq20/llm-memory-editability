#!/usr/bin/env python3
"""Frozen five-loop training and independent depth sweeps for whole-loop gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from llm_memory_editability.gated_depth_data import BalancedStream, load, prepare  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import model_digest  # noqa: E402
from llm_memory_editability.loop_hop_data import sha256  # noqa: E402
from llm_memory_editability.whole_loop_gate import WholeLoopGateGPT, matrix_flops  # noqa: E402


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def initialize(config, metadata, arm, seed, device):
    torch.manual_seed(seed)
    hf = GPT2Config(**config["model"], vocab_size=metadata["vocab_size"])
    model = WholeLoopGateGPT(
        hf, repeats=5, **config["arms"][arm], alpha=config["gate"]["initial_alpha"]
    )
    return model.to(device)


def backbone_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.transformer.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model, config, device):
    decay, regular = [], []
    for name, parameter in model.named_parameters():
        (
            regular if parameter.ndim < 2 or "ln" in name or name.startswith("gate_") else decay
        ).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config["training"]["weight_decay"]},
            {"params": regular, "weight_decay": 0.0},
        ],
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=device.type == "cuda",
    )


def tensors(parts, device):
    return [torch.as_tensor(p, device=device, dtype=torch.long) for p in parts]


def panel(arrays, name):
    return [arrays[f"{name}__{key}"] for key in ("tokens", "positions", "labels")]


def summary(prediction, labels, rows):
    correct = prediction == labels
    groups = {}
    for k in sorted({row["hop_count"] for row in rows}):
        mask = np.array([row["hop_count"] == k for row in rows])
        groups[str(k)] = float(correct[mask].mean())
    return {
        "accuracy": float(correct.mean()),
        "by_hop": groups,
        "macro_accuracy": float(np.mean(list(groups.values()))),
    }


@torch.no_grad()
def validation(model, arrays, rows, device, batch_size, name="comp_A_val_k1to5"):
    prior = model.training
    model.eval()
    parts = panel(arrays, name)
    predictions, losses, margins = [], [], []
    for start in range(0, len(rows), batch_size):
        x, positions, y = tensors([p[start : start + batch_size] for p in parts], device)
        with amp(device):
            logits = model(x, positions, num_loops=5).float()
        predictions.append(logits.argmax(-1).cpu().numpy())
        losses.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
        target = logits.gather(1, y[:, None]).squeeze(1)
        other = logits.scatter(1, y[:, None], -torch.inf).amax(-1)
        margins.append((target - other).cpu().numpy())
    model.train(prior)
    pred, loss, margin = map(np.concatenate, (predictions, losses, margins))
    result = summary(pred, parts[2], rows)
    result.update(loss=float(loss.mean()), margin=float(margin.mean()))
    return result, {"prediction": pred, "loss": loss, "margin": margin}


def save_checkpoint(path, model, optimizer, stream, step, config, arm, seed, elapsed):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "data_rng": stream.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "step": step,
            "config": config,
            "arm": arm,
            "seed": seed,
            "training_seconds": elapsed,
            "model_sha256": model_digest(model),
        },
        temporary,
    )
    temporary.replace(path)


def tracking(config, out, phase, arm, seed):
    import wandb

    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        mode="online",
        group=config["batch"],
        name=f"{phase}-{arm}-seed{seed}",
        dir=str(out),
        config={**config, "phase": phase, "arm": arm, "seed": seed},
        settings=wandb.Settings(disable_git=True),
        save_code=False,
    )
    write_json(out / "tracking.json", {"id": run.id, "url": run.url, "phase": phase})
    return run


def forward_contract(config, metadata, arm, seed, device, parts):
    """Real-device/precision contract; separate throwaway models, zero optimizer steps."""
    from llm_memory_editability.self_decode_alignment import SelfDecodeGPT

    current = initialize(config, metadata, arm, seed, device).eval()
    x, positions, y = tensors([p[:4] for p in parts], device)
    if current.gate_mode == "token":
        reference = (
            WholeLoopGateGPT(current.config, gate_mode="fixed", feedback_beta=current.feedback_beta)
            .to(device)
            .eval()
        )
        reference.transformer.load_state_dict(current.transformer.state_dict())
    elif current.feedback_beta == 0 and current.gate_mode == "full":
        reference = SelfDecodeGPT(current.config, repeats=5).to(device).eval()
        reference.transformer.load_state_dict(current.transformer.state_dict())
    else:
        reference = None
    findings = stability(current, parts, device)
    if reference is not None:
        with amp(device):
            actual = current(x, positions)
            expected = reference(x, positions)
            if expected.ndim == 3:
                expected = expected.squeeze(1)
            findings["reference_logits_exact"] = torch.equal(actual, expected)
            F.cross_entropy(actual.float(), y).backward()
            F.cross_entropy(expected.float(), y).backward()
        findings["reference_backbone_gradients_exact"] = all(
            torch.equal(p.grad, q.grad)
            for p, q in zip(
                current.transformer.parameters(), reference.transformer.parameters(), strict=True
            )
        )
    flags = [v for v in findings.values() if isinstance(v, bool)]
    if not all(flags):
        raise AssertionError(findings)
    return {
        "passed": True,
        "checks": findings,
        "device": str(device),
        "precision": config["training"]["precision"],
    }


@torch.no_grad()
def stability(model, parts, device):
    model.eval()
    # Stratify by the number of non-NOOP relation slots, without using labels.
    # The final vocabulary symbol is NOOP for this frozen tokenizer.
    counts = ((parts[0][:, 1:-1] != model.config.vocab_size - 1) & (parts[0][:, 1:-1] != 0)).sum(1)
    indices = np.concatenate([np.flatnonzero(counts == k)[:2] for k in np.unique(counts)])
    x, positions, _ = tensors([p[indices] for p in parts], device)
    states = []
    alpha_min, alpha_max = 1.0, 0.0
    with amp(device):
        for r, state, info in model.trajectory(x, 32):
            assert torch.isfinite(state).all(), f"Nonfinite state at loop {r}"
            assert torch.isfinite(model.readout(state, positions)).all()
            alpha_min = min(alpha_min, float(info["alpha"].min()))
            alpha_max = max(alpha_max, float(info["alpha"].max()))
            if r <= 5:
                states.append(state.clone())
        for r, state, _ in model.trajectory(x, 5):
            assert torch.equal(states[r - 1], state), "Terminal-loop mismatch"
    return {
        "finite_32_loops": True,
        "prefix_exact": True,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "probe_count": len(indices),
        "probe_hops": np.unique(counts[indices]).tolist(),
    }


def train(config, base, phase, arm, seed, device):
    if phase == "train":
        if seed not in config["seeds"] or arm not in config["arms"]:
            raise ValueError("Training outside the frozen formal matrix")
        if not json.loads((base / "pilot-gate.json").read_text()).get("passed"):
            raise ValueError("Formal training requires the completed pilot gate")
    pools, arrays, metadata = load(base / "data")
    out = base / phase / f"{arm}-seed{seed}"
    out.mkdir(parents=True, exist_ok=False)
    model = initialize(config, metadata, arm, seed, device)
    optimizer = optimizer_for(model, config, device)
    stream = BalancedStream(pools, metadata, seed + 100000)
    steps = config["training"]["steps"] if phase == "train" else config[phase]["steps"]
    batch = config["training"]["batch_size"]
    before_hash, initial = backbone_hash(model), model_digest(model)
    run = tracking(config, out, phase, arm, seed)
    if phase == "smoke":
        write_json(
            out / "forward-contract.json",
            forward_contract(
                config, metadata, arm, seed, device, panel(arrays, "comp_A_val_k1to5")
            ),
        )
    validation_before, _ = validation(model, arrays, pools["comp_A_val_k1to5"], device, batch)
    initial_stability = stability(model, panel(arrays, "comp_A_val_k1to5"), device)
    best, best_step = (-1.0, -float("inf")), 0
    started = time.monotonic()
    counts = {str(k): 0 for k in range(1, 6)}
    stream_digest = hashlib.sha256()
    first_loss, last_loss, gate_grad = None, None, {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (out / "curve.jsonl").open("w", buffering=1) as curve:
        for step in range(1, steps + 1):
            model.train()
            packed, rows = stream.batch(batch)
            for p in packed:
                stream_digest.update(p.tobytes())
            for row in rows:
                counts[str(row["hop_count"])] += 1
            x, positions, y = tensors(packed, device)
            lr = config["training"]["learning_rate"] * min(
                step / config["training"]["warmup_steps"], 1
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with amp(device):
                logits = model(x, positions, num_loops=5)
                loss = F.cross_entropy(logits.float(), y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip"])
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite update at step {step}")
            if step == 1:
                first_loss = float(loss.detach())
                gate_grad = {
                    n: float(p.grad.norm())
                    for n, p in model.named_parameters()
                    if n.startswith("gate_")
                }
                if gate_grad and not all(np.isfinite(v) and v > 0 for v in gate_grad.values()):
                    raise AssertionError("Gate parameters require finite nonzero gradients")
            optimizer.step()
            last_loss = float(loss.detach())
            if step == 1 or step % config["training"]["log_every"] == 0 or step == steps:
                event = {
                    "step": step,
                    "train/loss": last_loss,
                    "train/lr": lr,
                    "train/grad_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - started,
                    "examples": step * batch,
                    "input_tokens": step * batch * (metadata["slots"] + 2),
                    "loop_tokens": 5 * step * batch * (metadata["slots"] + 2),
                }
                curve.write(json.dumps(event) + "\n")
                run.log(event, step=step)
                write_json(out / "progress.json", {**event, "utc": utc()})
            if step % config["training"]["validation_every"] == 0 or step == steps:
                stats, predictions = validation(
                    model, arrays, pools["comp_A_val_k1to5"], device, batch
                )
                score = (stats["macro_accuracy"], -stats["loss"])
                curve.write(json.dumps({"step": step, "validation": stats}) + "\n")
                run.log(
                    {"val/accuracy": stats["macro_accuracy"], "val/loss": stats["loss"]}, step=step
                )
                if score > best:
                    best, best_step = score, step
                    save_checkpoint(
                        out / "best.pt",
                        model,
                        optimizer,
                        stream,
                        step,
                        config,
                        arm,
                        seed,
                        time.monotonic() - started,
                    )
                    np.savez_compressed(out / "best-validation.npz", **predictions)
                    write_json(out / "best-validation.json", {"step": step, **stats})
            if step % config["training"]["checkpoint_every"] == 0 or step == steps:
                save_checkpoint(
                    out / "last.pt",
                    model,
                    optimizer,
                    stream,
                    step,
                    config,
                    arm,
                    seed,
                    time.monotonic() - started,
                )
    final_validation, predictions = validation(
        model, arrays, pools["comp_A_val_k1to5"], device, batch
    )
    np.savez_compressed(out / "last-validation.npz", **predictions)
    final_stability = stability(model, panel(arrays, "comp_A_val_k1to5"), device)
    elapsed = time.monotonic() - started
    per_forward = matrix_flops(
        config["model"],
        metadata["vocab_size"],
        metadata["slots"] + 2,
        config["arms"][arm]["feedback_beta"],
        5,
        batch,
    )
    result = {
        "passed": True,
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "step": steps,
        "steps": steps,
        "best_step": best_step,
        "initial_backbone_sha256": before_hash,
        "initial_model_sha256": initial,
        "final_model_sha256": model_digest(model),
        "stream_sha256": stream_digest.hexdigest(),
        "hop_exposures": counts,
        "training_examples": steps * batch,
        "input_tokens": steps * batch * (metadata["slots"] + 2),
        "supervised_tokens": steps * batch,
        "loop_tokens": steps * batch * (metadata["slots"] + 2) * 5,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "gate_gradient_norms": gate_grad,
        "initial_validation": validation_before,
        "final_validation": final_validation,
        "initial_stability": initial_stability,
        "final_stability": final_stability,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_matrix_flops_estimate": 3 * steps * per_forward,
        "forward_feedback_softmax_elements": steps
        * batch
        * 5
        * (metadata["slots"] + 2)
        * metadata["vocab_size"]
        * config["arms"][arm]["feedback_beta"],
        "cost_note": (
            "3x forward matmuls estimates backward; excludes softmax/norm/optimizer; "
            "wall time includes validation"
        ),
        "wall_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint_sha256": {n: sha256(out / n) for n in ("best.pt", "last.pt")},
        "utc": utc(),
    }
    write_json(out / "complete.json", result)
    run.summary.update(
        {
            "training_complete": True,
            "training_steps": steps,
            "best_step": best_step,
            "complete_sha256": sha256(out / "complete.json"),
            "final_model_sha256": result["final_model_sha256"],
        }
    )
    run.finish(exit_code=0)


def restore(config, metadata, out, checkpoint, device):
    payload = torch.load(out / checkpoint, map_location=device, weights_only=False)
    if payload["config"] != config:
        raise ValueError("Checkpoint configuration mismatch")
    model = initialize(config, metadata, payload["arm"], payload["seed"], device)
    model.load_state_dict(payload["model"], strict=True)
    assert model_digest(model) == payload["model_sha256"]
    return model, payload


def audit(config, base, phase, arm, seed, device):
    out = base / phase / f"{arm}-seed{seed}"
    complete = json.loads((out / "complete.json").read_text())
    pools, arrays, metadata = load(base / "data")
    results = {}
    for checkpoint in ("best", "last"):
        assert sha256(out / f"{checkpoint}.pt") == complete["checkpoint_sha256"][f"{checkpoint}.pt"]
        model, payload = restore(config, metadata, out, f"{checkpoint}.pt", device)
        initial = model_digest(model)
        stats, actual = validation(
            model, arrays, pools["comp_A_val_k1to5"], device, config["evaluation"]["batch_size"]
        )
        expected = np.load(out / f"{checkpoint}-validation.npz")
        assert np.array_equal(actual["prediction"], expected["prediction"])
        for key in ("loss", "margin"):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-5, rtol=1e-5)
        extra = {}
        if phase in ("pilot", "smoke"):
            for name in ("facts_train", "train_probe"):
                extra[name], _ = validation(
                    model, arrays, pools[name], device, config["evaluation"]["batch_size"], name
                )
            # All facts separately for A and B, avoiding a misleading average.
            _, facts = validation(
                model,
                arrays,
                pools["facts_train"],
                device,
                config["evaluation"]["batch_size"],
                "facts_train",
            )
            for graph in "AB":
                mask = np.array([row["graph_id"] == graph for row in pools["facts_train"]])
                extra[f"atoms_{graph}"] = float(
                    (facts["prediction"][mask] == arrays["facts_train__labels"][mask]).mean()
                )
        results[checkpoint] = {
            "step": payload["step"],
            "validation": stats,
            **extra,
            "stability": stability(model, panel(arrays, "comp_A_val_k1to5"), device),
        }
        assert model_digest(model) == initial
    write_json(
        out / "audit.json",
        {
            "passed": True,
            "checkpoints": results,
            "complete_sha256": sha256(out / "complete.json"),
            "utc": utc(),
        },
    )


def pilot_gate(config, base):
    checks = {}
    complete = []
    for arm in config["arms"]:
        out = base / "pilot" / f"{arm}-seed{config['pilot']['seed']}"
        result = json.loads((out / "complete.json").read_text())
        audit_result = json.loads((out / "audit.json").read_text())
        complete.append(result)
        checks[f"{arm}/audit"] = audit_result["passed"]
        checks[f"{arm}/finite"] = result["final_stability"]["finite_32_loops"]
        checks[f"{arm}/loss_decreased"] = (
            result["final_validation"]["loss"] < result["initial_validation"]["loss"]
        )
        if arm.endswith("Gate"):
            gradients = result["gate_gradient_norms"]
            checks[f"{arm}/gradient"] = len(gradients) == 2 and all(
                np.isfinite(v) and v > 0 for v in gradients.values()
            )
            initial = result["initial_stability"]
            checks[f"{arm}/initial_gate_unsaturated"] = (
                abs(initial["alpha_min"] - 0.9) < 1e-6 and abs(initial["alpha_max"] - 0.9) < 1e-6
            )
        if arm in ("N-Full", "Q-Full"):
            best = audit_result["checkpoints"]["best"]
            for graph in "AB":
                checks[f"{arm}/atoms_{graph}"] = (
                    best[f"atoms_{graph}"] >= config["pilot"]["minimum_atom_accuracy"]
                )
            for k in (3, 4, 5):
                checks[f"{arm}/train_k{k}"] = (
                    best["train_probe"]["by_hop"][str(k)] >= config["pilot"]["minimum_train_k3to5"]
                )
                checks[f"{arm}/val_k{k}"] = (
                    best["validation"]["by_hop"][str(k)] >= config["pilot"]["minimum_val_k3to5"]
                )
    checks["paired_backbone"] = len({v["initial_backbone_sha256"] for v in complete}) == 1
    checks["paired_exposure"] = len({v["stream_sha256"] for v in complete}) == 1
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "policy": (
            "If failed: preserve pilot and stop formal dispatch; "
            "no OOD-driven tuning or budget extension"
        ),
        "utc": utc(),
    }
    write_json(base / "pilot-gate.json", result)
    return result


def write_csv(path, records):
    if not records:
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@torch.no_grad()
def sweep(model, parts, device, loops, batch_size, gate_override=None):
    model.eval()
    predictions, losses, margins, entropies = [], [], [], []
    started = time.monotonic()
    for start in range(0, len(parts[0]), batch_size):
        x, positions, y = tensors([p[start : start + batch_size] for p in parts], device)
        pp, ll, mm, ee = [], [], [], []
        with amp(device):
            for _, _short_state, _ in model.trajectory(x, 5, gate_override=gate_override):
                pass
            separate = model.readout(_short_state, positions)
            for r, state, _ in model.trajectory(x, max(loops), gate_override=gate_override):
                if not torch.isfinite(state).all():
                    raise FloatingPointError(f"Nonfinite sweep at {r}")
                if r not in loops:
                    continue
                logits = model.readout(state, positions)
                if r == 5:
                    assert torch.equal(logits, separate)
                logits = logits.float()
                pp.append(logits.argmax(-1).cpu().numpy())
                ll.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
                target = logits.gather(1, y[:, None]).squeeze(1)
                mm.append(
                    (target - logits.scatter(1, y[:, None], -torch.inf).amax(-1)).cpu().numpy()
                )
                logp = logits.log_softmax(-1)
                ee.append((-(logp.exp() * logp).sum(-1)).cpu().numpy())
        predictions.append(np.stack(pp, 1))
        losses.append(np.stack(ll, 1))
        margins.append(np.stack(mm, 1))
        entropies.append(np.stack(ee, 1))
    return {
        "prediction": np.concatenate(predictions),
        "loss": np.concatenate(losses),
        "margin": np.concatenate(margins),
        "entropy": np.concatenate(entropies),
        "labels": parts[2],
        "loops": np.asarray(loops),
        "seconds": np.asarray(time.monotonic() - started),
    }


@torch.no_grad()
def diagnostics(model, parts, rows, device, count, out, prefix):
    model.eval()
    selected = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    x, positions, y = tensors([p[selected] for p in parts], device)
    alpha, measurements, flags = [], [], []
    with amp(device):
        for r, state, info in model.trajectory(x, 32):
            a, previous, candidate = (info[k].float() for k in ("alpha", "previous", "candidate"))
            xf = state.float()
            logits = model.readout(state, positions).float()
            correctness = logits.argmax(-1) == y
            logp = logits.log_softmax(-1)
            margin = logits.gather(1, y[:, None]).squeeze(1) - logits.scatter(
                1, y[:, None], -torch.inf
            ).amax(-1)
            alpha.append(a.squeeze(-1).cpu().numpy())
            flags.append(correctness.cpu().numpy())
            for i, row_id in enumerate(selected):
                denominator = previous[i].norm().clamp_min(1e-12)
                measurements.append(
                    {
                        "example_id": rows[row_id]["example_id"],
                        "loop": r,
                        "hop": rows[row_id]["hop_count"],
                        "alpha_mean": float(a[i].mean()),
                        "alpha_q05": float(a[i].quantile(0.05)),
                        "alpha_q50": float(a[i].quantile(0.5)),
                        "alpha_q95": float(a[i].quantile(0.95)),
                        "update_ratio": float((xf[i] - previous[i]).norm() / denominator),
                        "candidate_delta_norm": float((candidate[i] - previous[i]).norm()),
                        "state_rms": float(xf[i].square().mean().sqrt()),
                        "state_max_abs": float(xf[i].abs().max()),
                        "cosine": float(
                            F.cosine_similarity(xf[i].flatten(), previous[i].flatten(), dim=0)
                        ),
                        "margin": float(margin[i]),
                        "entropy": float(-(logp[i].exp() * logp[i]).sum()),
                        "correct": bool(correctness[i]),
                    }
                )
    write_csv(out / f"{prefix}-dynamics.csv", measurements)
    np.savez_compressed(
        out / f"{prefix}-dynamics.npz",
        alpha=np.stack(alpha, 1),
        correct=np.stack(flags, 1),
        indices=selected,
    )
    events = []
    correctness = np.stack(flags, 1)
    for i, row_id in enumerate(selected):
        good = np.flatnonzero(correctness[i])
        first = int(good[0] + 1) if len(good) else None
        bad_after = np.flatnonzero(~correctness[i, first:]) if first is not None else []
        events.append(
            {
                "example_id": rows[row_id]["example_id"],
                "hop": rows[row_id]["hop_count"],
                "first_correct_loop": first,
                "first_wrong_after_correct_loop": int(first + bad_after[0] + 1)
                if len(bad_after)
                else None,
            }
        )
    write_csv(out / f"{prefix}-first-correct-and-degradation.csv", events)


def stratified_indices(rows, per_hop):
    selected = []
    for k in sorted({row["hop_count"] for row in rows}):
        group = np.array([i for i, row in enumerate(rows) if row["hop_count"] == k])
        selected.extend(group[np.linspace(0, len(group) - 1, min(per_hop, len(group)), dtype=int)])
    return np.asarray(selected)


@torch.no_grad()
def interventions(model, arrays, pools, device, out):
    """Auxiliary interventions on a frozen per-hop subset; never select checkpoints."""
    if model.gate_mode != "token":
        return
    model.eval()
    indices = stratified_indices(pools["comp_A_val_k1to5"], 32)
    x, _, _ = tensors([p[indices] for p in panel(arrays, "comp_A_val_k1to5")], device)
    gates = []
    with amp(device):
        for _, _, info in model.trajectory(x, 32):
            gates.append(info["alpha"].clone())
    values = torch.stack(gates)
    global_mean = values.mean()
    loop_mean = values.mean((1, 2, 3))
    position_mean = values.mean((0, 1)).unsqueeze(0)
    np.savez_compressed(
        out / "gate-calibration.npz",
        indices=indices,
        global_mean=global_mean.cpu().numpy(),
        loop_mean=loop_mean.cpu().numpy(),
        position_mean=position_mean.cpu().numpy(),
    )
    overrides = {
        "unmodified": None,
        "validation_global_mean": lambda r, a: global_mean.expand_as(a),
        "validation_loop_mean": lambda r, a: loop_mean[r - 1].expand_as(a),
        "validation_position_mean": lambda r, a: position_mean.expand_as(a),
        "within_batch_swap": lambda r, a: a.roll(1, dims=0),
    }
    records = []
    for name in ("eval_A_short", "eval_A_long", "eval_B_short"):
        indices = stratified_indices(pools[name], 32)
        parts = [p[indices] for p in panel(arrays, name)]
        for intervention, override in overrides.items():
            result = sweep(
                model, parts, device, [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32], 128, override
            )
            np.savez_compressed(
                out / f"intervention-{name}-{intervention}.npz", **result, indices=indices
            )
            for k in sorted({pools[name][i]["hop_count"] for i in indices}):
                mask = np.array([pools[name][i]["hop_count"] == k for i in indices])
                for j, r in enumerate(result["loops"]):
                    records.append(
                        {
                            "split": name,
                            "intervention": intervention,
                            "hop": k,
                            "loop": int(r),
                            "n": int(mask.sum()),
                            "accuracy": float(
                                (result["prediction"][mask, j] == result["labels"][mask]).mean()
                            ),
                        }
                    )
    write_csv(out / "gate-interventions.csv", records)
    write_json(
        out / "intervention-scope.json",
        {
            "status": "auxiliary fixed diagnostic subsets; no checkpoint selection",
            "calibration": (
                "32 fixed validation queries per hop, all 32 loops; gate values only, no labels"
            ),
            "swap": "deterministic cyclic donor in each batch; may create off-distribution states",
        },
    )


def evaluate(config, base, arm, seed, device):
    out = base / "train" / f"{arm}-seed{seed}"
    pools, arrays, metadata = load(base / "data")
    assert json.loads((out / "audit.json").read_text())["passed"]
    loops = config["evaluation"]["loops"]
    records, flips, costs = [], [], []
    for checkpoint in ("best", "last"):
        model, payload = restore(config, metadata, out, f"{checkpoint}.pt", device)
        initial = model_digest(model)
        destination = out / f"evaluation-{checkpoint}"
        destination.mkdir(exist_ok=False)
        names = ["facts_train", "eval_A_short", "eval_A_long", "eval_B_short", "eval_B_long"]
        names += sorted({key.split("__")[0] for key in arrays if key.startswith("noop_")})
        for name in names:
            parts = panel(arrays, name)
            rows = (
                pools[name]
                if name in pools
                else [row for row in pools["eval_A_short"] if row["hop_count"] == 3]
            )
            values = sweep(model, parts, device, loops, config["evaluation"]["batch_size"])
            np.savez_compressed(destination / f"{name}.npz", **values)
            correct = values["prediction"] == values["labels"][:, None]
            at5 = correct[:, loops.index(5)]
            for k in sorted({row["hop_count"] for row in rows}):
                mask = np.array([row["hop_count"] == k for row in rows])
                for j, r in enumerate(loops):
                    current, old = correct[mask, j], at5[mask]
                    records.append(
                        {
                            "checkpoint": checkpoint,
                            "step": payload["step"],
                            "split": name,
                            "hop": k,
                            "loop": r,
                            "n": int(mask.sum()),
                            "correct": int(current.sum()),
                            "accuracy": float(current.mean()),
                            "loss": float(values["loss"][mask, j].mean()),
                            "margin": float(values["margin"][mask, j].mean()),
                        }
                    )
                    flips.append(
                        {
                            "checkpoint": checkpoint,
                            "split": name,
                            "hop": k,
                            "loop": r,
                            "correct_to_correct": int((old & current).sum()),
                            "correct_to_wrong": int((old & ~current).sum()),
                            "wrong_to_correct": int((~old & current).sum()),
                            "wrong_to_wrong": int((~old & ~current).sum()),
                            "retain": float(current[old].mean()) if old.any() else None,
                            "recover": float(current[~old].mean()) if (~old).any() else None,
                        }
                    )
            if name in ("eval_A_short", "eval_A_long", "eval_B_short"):
                diagnostics(
                    model,
                    parts,
                    rows,
                    device,
                    config["evaluation"]["diagnostic_per_group"],
                    destination,
                    name,
                )
            costs.append(
                {
                    "checkpoint": checkpoint,
                    "split": name,
                    "n": len(rows),
                    "max_loop": max(loops),
                    "sweep_wall_seconds": float(values["seconds"]),
                    "note": "Includes all intermediate readouts and independent R5 prefix check",
                }
            )
        # Separate inference timing; no dynamical diagnostic or label scoring in the timer.
        x, positions, _ = tensors([p[:128] for p in panel(arrays, "eval_A_short")], device)
        with torch.no_grad(), amp(device):
            for r in (5, 10, 20, 32):
                model(x, positions, num_loops=r)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.monotonic()
                for _ in range(5):
                    model(x, positions, num_loops=r)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                write_json(
                    destination / f"inference-cost-R{r}.json",
                    {
                        "batch": len(x),
                        "loops": r,
                        "seconds_per_batch": (time.monotonic() - start) / 5,
                        "matrix_flops_per_example": matrix_flops(
                            config["model"],
                            metadata["vocab_size"],
                            22,
                            config["arms"][arm]["feedback_beta"],
                            r,
                        ),
                        "no_early_exit": True,
                    },
                )
        interventions(model, arrays, pools, device, destination)
        assert model_digest(model) == initial
    write_csv(out / "accuracy_by_hop_loop.csv", records)
    write_csv(out / "flip_transitions.csv", flips)
    write_csv(out / "compute_cost.csv", costs)
    write_json(
        out / "evaluation.json",
        {
            "passed": True,
            "checkpoints": ["best", "last"],
            "loops": loops,
            "rows": len(records),
            "files": {
                str(path.relative_to(out)): sha256(path)
                for path in sorted(out.glob("evaluation-*/*"))
                if path.is_file()
            },
            "utc": utc(),
        },
    )


def cloud_audit(base, arm, seed):
    import wandb

    out = base / "train" / f"{arm}-seed{seed}"
    track = json.loads((out / "tracking.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    assert all(
        json.loads((out / f"{name}.json").read_text())["passed"] for name in ("audit", "evaluation")
    )
    expected = {
        "training_steps": complete["steps"],
        "independent_reload_passed": True,
        "evaluation_passed": True,
        "complete_sha256": sha256(out / "complete.json"),
        "audit_sha256": sha256(out / "audit.json"),
        "evaluation_sha256": sha256(out / "evaluation.json"),
    }
    path = f"zhusq20/llm-memory-editability/{track['id']}"
    run = wandb.Api().run(path)
    run.summary.update(expected)
    verified = wandb.Api().run(path)
    assert verified.state == "finished", verified.state
    assert all(verified.summary[k] == v for k, v in expected.items())
    write_json(
        out / "cloud-audit.json",
        {
            "passed": True,
            "url": track["url"],
            "state": verified.state,
            "expected": expected,
            "utc": utc(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "smoke",
            "smoke-audit",
            "pilot",
            "pilot-audit",
            "pilot-gate",
            "train",
            "audit",
            "evaluate",
            "cloud-audit",
            "analyze",
        ],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if args.stage == "prepare":
        prepare(config, args.base / "data")
    elif args.stage == "pilot-gate":
        pilot_gate(config, args.base)
    elif args.stage == "cloud-audit":
        cloud_audit(args.base, args.arm, args.seed)
    elif args.stage == "analyze":
        from analyze_gated_depth import analyze

        analyze(config, args.base)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Scientific jobs require a verified CUDA container")
        device = torch.device("cuda:0")
        if args.stage in ("smoke", "pilot", "train"):
            train(config, args.base, args.stage, args.arm, args.seed, device)
        elif args.stage in ("smoke-audit", "pilot-audit", "audit"):
            audit(
                config,
                args.base,
                args.stage.split("-")[0] if "-" in args.stage else "train",
                args.arm,
                args.seed,
                device,
            )
        elif args.stage == "evaluate":
            evaluate(config, args.base, args.arm, args.seed, device)


if __name__ == "__main__":
    main()
