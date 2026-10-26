#!/usr/bin/env python3
"""Train and evaluate loop-native handoff arms on the existing two-graph multi-hop data.

Historical configs use supervised segments with a detached carry. The terminal recipe
instead samples depth independently per sequence and backpropagates through every round,
normalizing terminal cross-entropy by all valid supervised tokens in the logical batch.
No row-level hop count, bridge entity, position label or graph identity reaches the model.

Stages:
  smoke       engineering run of ``smoke_steps`` updates, no tracking, for speed and finiteness
  smoke-grid  the endpoint trace on the smoke checkpoint, to exercise the grid code
  train       scientific run with W&B learning curves, checkpoints and EMA weights
  grid        endpoint R1..``grid_rounds`` trace of every panel with EMA and raw weights
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_hop as native  # noqa: E402

from llm_memory_editability.bios_model import ModelConfig, matmul_flops  # noqa: E402
from llm_memory_editability.experiment_tracking import learning_metrics  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import (  # noqa: E402
    MergedBatchStream,
    digest,
    model_digest,
)
from llm_memory_editability.loop_hop_data import (  # noqa: E402
    exposure_summary,
    load,
    make_table,
    pack_rows,
    panel_names,
)
from llm_memory_editability.loop_native_handoff import (  # noqa: E402
    HANDOFFS,
    LoopNativeHandoffGPT,
    deep_supervision_loss,
    handoff_parameter_count,
    sample_schedule,
)

BATCH = "loop-native-handoff-development-v1"
TERMINAL_BATCH = "loop-invariant-poisson-terminal-v1"


def terminal_recipe(training):
    return training.get("objective") == "terminal_token_mean"


def validate_config(config):
    if config.get("batch") not in (BATCH, TERMINAL_BATCH):
        raise ValueError("Unexpected batch")
    if not config["arms"] or any(a["handoff"] not in HANDOFFS for a in config["arms"].values()):
        raise ValueError("Every arm needs a known handoff")
    training = config["training"]
    if config["batch"] == TERMINAL_BATCH:
        if not terminal_recipe(training) or training.get("backpropagation") != "full":
            raise ValueError("The terminal recipe requires token-mean CE and full backpropagation")
        if training.get("execution") != "masked_batch":
            raise ValueError("The terminal recipe advances a masked full batch")
        if (
            training["unique_layers"] != 1
            or config["handoff"].get("first_round_input") != "embedding"
        ):
            raise ValueError("The terminal recipe requires one shared block and first input e")
        if config["arms"] != {"invariant": {"handoff": "invariant"}}:
            raise ValueError("The terminal recipe specifies the invariant handoff")
        if training.get("depth_sampling") != {
            "distribution": "poisson",
            "mean": 8.0,
            "minimum": 1,
            "maximum": 16,
            "scope": "sequence",
            "truncation": "reject",
        }:
            raise ValueError("Expected independent per-sequence Poisson(8) conditioned on [1,16]")
        if {"segments", "first_segment_rounds", "later_segment_rounds"} & training.keys():
            raise ValueError("Terminal supervision has no segment schedule")
    else:
        if "objective" in training:
            raise ValueError("Use the terminal batch config for the new objective")
        for key in ("first_segment_rounds", "later_segment_rounds"):
            low, high = training[key]
            if not 1 <= low <= high:
                raise ValueError(f"Invalid round range: {key}")
        if training["segments"] < 1:
            raise ValueError("Invalid deep-supervision setting")
    if not 0 < training["final_lr_fraction"] <= 1:
        raise ValueError("Invalid learning-rate schedule setting")
    if config["tracking"]["entity"] != "zsqzz":
        raise ValueError("New batches track to the zsqzz team")


def backbone_digest(model):
    """Hash the backbone in DiscoLoop state-dict order, excluding handoff parameters."""
    import hashlib

    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith("handoff_"):
            continue
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize(config, name, metadata, device):
    training = config["training"]
    torch.manual_seed(training["initialization"])
    hf = GPT2Config(
        **config["model"],
        n_layer=training["unique_layers"],
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    model = LoopNativeHandoffGPT(
        hf,
        training["evaluation_repeats"],
        handoff=config["arms"][name]["handoff"],
        **config["handoff"],
    ).to(device)
    torch.manual_seed(config["dropout_seed"])
    return model


def optimizer_for(model, training, fused=True):
    named = list(model.named_parameters())
    plain = [(n, p) for n, p in named if not n.startswith("handoff_")]
    return torch.optim.AdamW(
        [
            {
                "params": [p for n, p in plain if not any(s in n for s in ("bias", "ln"))],
                "weight_decay": training["weight_decay"],
            },
            {
                "params": [p for n, p in plain if any(s in n for s in ("bias", "ln"))]
                + [p for n, p in named if n.startswith("handoff_")],
                "weight_decay": 0.0,
            },
        ],
        lr=training["lr"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
        fused=fused,
    )


def learning_rate(training, step):
    """Linear warmup from zero, then cosine decay to ``final_lr_fraction``."""
    if step <= training["warmup"]:
        return training["lr"] * (step - 1) / training["warmup"]
    progress = (step - training["warmup"]) / max(1, training["steps"] - training["warmup"])
    floor = training["final_lr_fraction"]
    return training["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


class ParameterAverage:
    """Float32 exponential moving average of trainable parameters (TRM/InfiLoop EMA)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.values = {n: p.detach().float().clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            self.values[name].mul_(self.decay).add_(
                parameter.detach().float(), alpha=1 - self.decay
            )

    def state_dict(self):
        return self.values

    def load_state_dict(self, values):
        self.values = {n: v.clone() for n, v in values.items()}

    @contextmanager
    def applied(self, model):
        saved = {n: p.detach().clone() for n, p in model.named_parameters()}
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(self.values[name].to(parameter.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(saved[name])


def step_flops(model, metadata, batch_size, schedule):
    """Matrix FLOPs (training factor 3) for one update with this round schedule."""
    width, vocab, length = model.config.n_embd, metadata["vocab_size"], metadata["sequence_length"]
    layers = len(model.transformer.h)
    cfg = ModelConfig(vocab, width, layers, model.config.n_head, model.config.n_positions)
    block = matmul_flops(cfg, batch_size, sequence=length, output_positions=0)
    readout = 3 * 2 * batch_size * 2 * width * vocab
    channel = 3 * 4 * batch_size * length * width * vocab
    total = 0
    for index, rounds in enumerate(schedule):
        last = index == len(schedule) - 1
        if model.handoff == "plain":
            channels = 0
        elif model.handoff == "additive":
            channels = rounds - 1 if last else rounds
        else:
            channels = rounds
        total += rounds * block + channels * channel + readout
    return total


def terminal_step_flops(model, metadata, depths, output_slots, *, include_inactive=True):
    """Matrix-shape estimate for masked terminal training (not wall time).

    Includes the initial probability decode, its unused feedback forward, every
    candidate decode/re-embedding, score MLP, and terminal readout. Elementwise
    softmax, normalization, masking, loss and optimizer work are excluded. By
    default counts expired rows too; include_inactive=False counts active work only.
    """
    from llm_memory_editability.loop_terminal_training import depth_groups

    width, vocab, length = model.config.n_embd, metadata["vocab_size"], metadata["sequence_length"]
    cfg = ModelConfig(
        vocab, width, len(model.transformer.h), model.config.n_head, model.config.n_positions
    )
    total = 0
    groups = list(depth_groups(depths, len(depths)))
    if include_inactive:
        groups = [(groups[-1][0], range(len(depths)))]
    for rounds, rows in groups:
        batch = len(rows)
        block = matmul_flops(cfg, batch, sequence=length, output_positions=0)
        channel = 3 * 4 * batch * length * width * vocab
        initial = (3 * 2 + 2) * batch * length * width * vocab
        score = 3 * 2 * batch * length * model.handoff_score_in.shape[0] * 6
        readout = 3 * 2 * batch * output_slots * width * vocab
        total += initial + rounds * (block + channel + score) + readout
    return total


def backward_batch(model, tokens, positions, labels, training, schedule_rng, device):
    """Accumulate one logical batch's gradients; caller clips and steps exactly once."""
    if terminal_recipe(training):
        from llm_memory_editability.loop_terminal_training import (
            backward_terminal_loss,
            sample_depths,
        )

        spec = training["depth_sampling"]
        depths = sample_depths(
            schedule_rng,
            len(tokens),
            mean=spec["mean"],
            minimum=spec["minimum"],
            maximum=spec["maximum"],
        )
        loss = backward_terminal_loss(
            model, tokens, positions, labels, depths, forward_context=lambda: autocast(device)
        )
        return loss, [], depths
    schedule = sample_schedule(
        schedule_rng,
        training["first_segment_rounds"],
        training["later_segment_rounds"],
        training["segments"],
    )
    with autocast(device):
        loss, parts = deep_supervision_loss(model, tokens, positions, labels, schedule)
    loss.backward()
    return loss.detach(), parts, schedule


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def limited(panels, limit_rows):
    """Smoke-only subsample of composition panels; atomic panels stay complete because
    premise metrics index them by atom."""
    if not limit_rows:
        return panels
    return {k: v if k.startswith("atomic") else v[:limit_rows] for k, v in panels.items()}


def evaluate_rounds(model, world, panels, metadata, device, batch_size, rounds):
    saved, mode = model.repeats, model.training
    try:
        results = {}
        for r in rounds:
            model.repeats = r
            results[str(r)] = native.measure_panels(
                model, world, panels, metadata, device, batch_size
            )
        return results
    finally:
        model.repeats = saved
        model.train(mode)


def train(config, name, out, smoke, config_path, track, device, limit_rows=None):
    out.mkdir(parents=True, exist_ok=False)
    training = {**config["training"]}
    terminal = terminal_recipe(training)
    if smoke:
        training["steps"] = config["smoke_steps"]
    world, panels, metadata = load(config["input_root"])
    panels = limited(panels, limit_rows)
    model = initialize(config, name, metadata, device)
    initial_backbone = backbone_digest(model)
    expected = config["paired_recipe"]["initial_backbone_sha256"]
    if initial_backbone != expected:
        raise AssertionError("Backbone initialization differs from the reused DiscoLoop")
    table = make_table(world, metadata, device)
    stream = MergedBatchStream(len(table[0]) - 1, training["stream_seed"])
    schedule_rng = np.random.default_rng(training["schedule_seed"])
    counts = np.zeros(stream.size, dtype=np.int64)
    optimizer = optimizer_for(model, training, fused=device.type == "cuda")
    average = ParameterAverage(model, training["ema_decay"])
    run_metadata = {
        "batch": config["batch"],
        "arm": name,
        "handoff": model.handoff,
        "training": training,
        "handoff_options": config["handoff"],
        "handoff_parameters": handoff_parameter_count(
            model.handoff, model.config.n_embd, config["handoff"]["score_hidden"]
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config_sha256": digest(config_path),
        "world_sha256": metadata["world_sha256"],
        "panels_sha256": metadata["panels_sha256"],
        "initial_backbone_sha256": initial_backbone,
        "initial_model_sha256": model_digest(model),
        "engineering_only": smoke,
        "objective": (
            "terminal CE summed over valid supervised tokens / logical-batch valid-token count; "
            "per-sequence Poisson depth; masked full-batch execution; full backpropagation"
            if terminal
            else "mean answer/EOS cross-entropy at the end of each segment; nothing else"
        ),
        "dtype": "float32 parameters, Adam state and EMA; bf16 autocast; TF32 enabled",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "limited_composition_rows": limit_rows,
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "started_utc": utc(),
    }
    write_json(out / "run.json", run_metadata)
    tracker = track(run_metadata) if track else None
    nodes = [0, training["steps"]] if smoke else config["evaluation_nodes"]
    history, step, flops, executed_rounds = [], 0, 0, 0
    depth_histogram = (
        np.zeros(training["depth_sampling"]["maximum"] + 1, dtype=np.int64) if terminal else None
    )
    executed_sample_rounds, sampled_sample_rounds, supervised_tokens_seen = 0, 0, 0
    training_seconds, started = 0.0, time.perf_counter()

    def empty_window():
        # Accumulate on the device; one synchronization per heartbeat, not per update.
        values = {
            "loss": torch.zeros((), device=device),
            "grad_norm": torch.zeros((), device=device),
            "n": 0,
        }
        if terminal:
            values.update(supervised_tokens=0, sample_rounds=0, samples=0)
        else:
            values["segments"] = torch.zeros(training["segments"], device=device)
        return values

    window = empty_window()

    def measure():
        weights = {}
        with average.applied(model):
            weights["ema"] = evaluate_rounds(
                model,
                world,
                panels,
                metadata,
                device,
                config["evaluation_batch_size"],
                config["evaluation_rounds"],
            )
        weights["raw"] = evaluate_rounds(
            model, world, panels, metadata, device, config["evaluation_batch_size"], [4]
        )
        metrics = {
            kind: {r: values[0] for r, values in result.items()} for kind, result in weights.items()
        }
        for kind, result in weights.items():
            for r, (_, predictions) in result.items():
                folder = out / "predictions" / f"{step:07d}" / f"{kind}-r{r}"
                folder.mkdir(parents=True, exist_ok=True)
                for panel, prediction in predictions.items():
                    np.savez_compressed(folder / f"{panel}.npz", **prediction)
        record = {
            "step": step,
            "metrics": metrics,
            "training_seconds": training_seconds,
            "wall_seconds": time.perf_counter() - started,
            "estimated_matmul_training_flops": flops,
            "executed_rounds": executed_rounds,
            **exposure_summary(counts, world, metadata, step, training["batch_size"]),
        }
        if terminal:
            record.pop("executed_rounds")
            record.update(
                executed_sample_rounds=executed_sample_rounds,
                sampled_sample_rounds=sampled_sample_rounds,
                executed_input_tokens=executed_sample_rounds * metadata["sequence_length"],
                supervised_tokens=supervised_tokens_seen,
                depth_histogram=depth_histogram.tolist(),
            )
        if model.handoff not in ("additive", "plain"):
            record["beta"] = float(model.beta.detach())
        history.append(record)
        write_json(out / "learning.json", history)
        saved = {
            "model": model.state_dict(),
            "ema": average.state_dict(),
            "arm": name,
            "handoff": model.handoff,
            "step": step,
            "config_sha256": digest(config_path),
            "world_sha256": metadata["world_sha256"],
            "model_sha256": model_digest(model),
            "first_round_input": model.first_round_input,
        }
        torch.save(saved, out / f"checkpoint-{step:07d}.pt")
        temporary = out / "latest.tmp.pt"
        torch.save(
            {
                **saved,
                "optimizer": optimizer.state_dict(),
                "stream": stream.state_dict(),
                "schedule_rng": schedule_rng.bit_generator.state,
                "counts": counts,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                "training_seconds": training_seconds,
                **(
                    {
                        "depth_histogram": depth_histogram.tolist(),
                        "executed_sample_rounds": executed_sample_rounds,
                        "sampled_sample_rounds": sampled_sample_rounds,
                        "terminal_execution": "masked_batch",
                        "supervised_tokens_seen": supervised_tokens_seen,
                    }
                    if terminal
                    else {}
                ),
            },
            temporary,
        )
        temporary.replace(out / "latest.pt")
        if tracker:
            tracker(learning_metrics(record), step)

    measure()
    model.train()
    for end in nodes[1:]:
        while step < end:
            synchronize(device)
            segment_start = time.perf_counter()
            boundary = min(end, (step // config["heartbeat_steps"] + 1) * config["heartbeat_steps"])
            while step < boundary:
                step += 1
                indices, _ = stream.batch(training["batch_size"])
                counts += np.bincount(indices, minlength=len(counts) + 1)[: len(counts)]
                index = torch.as_tensor(indices, device=device)
                tokens, positions, labels = (part[index] for part in table)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(training, step)
                loss, parts, schedule = backward_batch(
                    model, tokens, positions, labels, training, schedule_rng, device
                )
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["grad_clip"], foreach=True
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                average.update(model)
                if terminal:
                    sample_rounds = int(sum(schedule))
                    valid_tokens = int(labels.ne(-100).sum().item())
                    flops += terminal_step_flops(model, metadata, schedule, labels.shape[1])
                    sampled_sample_rounds += sample_rounds
                    executed_sample_rounds += len(tokens) * int(max(schedule))
                    supervised_tokens_seen += valid_tokens
                    depth_histogram += np.bincount(schedule, minlength=len(depth_histogram))
                    window["loss"] += loss * valid_tokens
                    window["supervised_tokens"] += valid_tokens
                    window["sample_rounds"] += sample_rounds
                    window["samples"] += len(tokens)
                else:
                    flops += step_flops(model, metadata, training["batch_size"], schedule)
                    executed_rounds += sum(schedule)
                    window["loss"] += loss.detach()
                    window["segments"] += torch.stack(parts)
                window["grad_norm"] += norm.detach()
                window["n"] += 1
            synchronize(device)
            training_seconds += time.perf_counter() - segment_start
            n = max(window["n"], 1)
            status = {
                "state": "running",
                "arm": name,
                "step": step,
                "budget": training["steps"],
                "training_seconds": training_seconds,
                "mean_loss": float(window["loss"]) / n,
                "mean_grad_norm": float(window["grad_norm"]) / n,
                "lr": learning_rate(training, step),
                "updated_utc": utc(),
            }
            if terminal:
                status.update(
                    mean_loss=float(window["loss"]) / window["supervised_tokens"],
                    mean_depth=window["sample_rounds"] / window["samples"],
                    supervised_tokens=supervised_tokens_seen,
                    executed_sample_rounds=executed_sample_rounds,
                    sampled_sample_rounds=sampled_sample_rounds,
                    depth_histogram=depth_histogram.tolist(),
                )
            else:
                status["mean_segment_loss"] = (window["segments"] / n).tolist()
            if model.handoff not in ("additive", "plain"):
                status["beta"] = float(model.beta.detach())
            if not all(map(math.isfinite, (status["mean_loss"], status["mean_grad_norm"]))):
                raise FloatingPointError(f"Non-finite loss or gradient near step {step}")
            window = empty_window()
            write_json(out / "status.json", status)
            with (out / "progress.jsonl").open("a") as progress:
                progress.write(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
            if tracker:
                tracker(
                    {f"training/{k}": v for k, v in status.items() if isinstance(v, float)}, step
                )
        model.eval()
        measure()
        model.train()
    final = {
        "state": "complete",
        "arm": name,
        "step": step,
        "budget": training["steps"],
        "engineering_only": smoke,
        "endpoint": history[-1],
        "model_sha256": model_digest(model),
        "final_checkpoint_sha256": digest(out / f"checkpoint-{step:07d}.pt"),
        "config_sha256": digest(config_path),
        "training_seconds": training_seconds,
        "ms_per_update": 1000 * training_seconds / max(step, 1),
        "wall_seconds": time.perf_counter() - started,
        "updated_utc": utc(),
    }
    write_json(out / "complete.json", final)
    write_json(out / "status.json", final)
    return final


@torch.no_grad()
def trace_panel(model, rows, metadata, device, batch_size, rounds):
    """Per-round answer predictions and carry diagnostics for one panel."""
    x, positions, _ = pack_rows(rows, metadata)
    hops = rows.shape[1] - 2
    predictions, answer_change, answer_gamma, relation_gamma = [], [], [], []
    for begin in range(0, len(rows), batch_size):
        tokens = torch.as_tensor(x[begin : begin + batch_size], device=device)
        pos = torch.as_tensor(positions[begin : begin + batch_size], device=device)
        with autocast(device):
            logits, change, gamma = model.trace(tokens, pos, rounds)
        predictions.append(logits[:, :, 0].argmax(-1).cpu().numpy())
        answer = pos[:, 0]
        batch = torch.arange(len(tokens), device=device)
        answer_change.append(change[:, batch, answer].float().cpu().numpy())
        answer_gamma.append(gamma[:, batch, answer].float().cpu().numpy())
        middle = gamma[:, :, 1:hops] if hops > 1 else gamma[:, :, :0]
        relation_gamma.append(middle.float().cpu().numpy().reshape(rounds, len(tokens), -1))
    return {
        "answer": np.concatenate(predictions, axis=1),
        "answer_change": np.concatenate(answer_change, axis=1),
        "answer_gamma": np.concatenate(answer_gamma, axis=1),
        "intermediate_relation_gamma": np.concatenate(relation_gamma, axis=1),
    }


def median_or_none(values):
    values = values[np.isfinite(values)] if values.size else values
    return float(np.median(values)) if values.size else None


def grid(config, name, base, config_path, device, folder="runs", limit_rows=None):
    out = base / folder / name
    complete = json.loads((out / "complete.json").read_text())
    path = out / f"checkpoint-{complete['step']:07d}.pt"
    if digest(path) != complete["final_checkpoint_sha256"]:
        raise ValueError("Endpoint checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    world, panels, metadata = load(config["input_root"])
    panels = limited(panels, limit_rows)
    model = initialize(config, name, metadata, device)
    if saved.get("first_round_input", "state_plus_embedding") != model.first_round_input:
        raise ValueError("Checkpoint and evaluation config disagree on the first-round input")
    model.load_state_dict(saved["model"], strict=True)
    average = ParameterAverage(model, config["training"]["ema_decay"])
    average.load_state_dict(saved["ema"])
    model.eval()
    rounds = config["grid_rounds"]
    result = {"arm": name, "step": saved["step"], "rounds": rounds, "weights": {}}
    for kind in ("ema", "raw"):
        context = average.applied(model) if kind == "ema" else _nothing()
        with context:
            folder = out / "grid" / kind
            folder.mkdir(parents=True, exist_ok=False)
            summary = {}
            for panel in panel_names():
                rows = panels[panel]
                traced = trace_panel(
                    model, rows, metadata, device, config["evaluation_batch_size"], rounds
                )
                np.savez_compressed(folder / f"{panel}.npz", rows=rows, **traced)
                truth = rows[:, -1] + metadata["entity_offset"]
                correct = traced["answer"] == truth[None]
                summary[panel] = {
                    "answer_accuracy": correct.mean(1).round(6).tolist(),
                    "answer_change_median": [median_or_none(v) for v in traced["answer_change"]],
                    "answer_gamma_median": [median_or_none(v) for v in traced["answer_gamma"]],
                    "intermediate_relation_gamma_median": [
                        median_or_none(v.ravel()) for v in traced["intermediate_relation_gamma"]
                    ],
                }
            joint = evaluate_rounds(
                model,
                world,
                panels,
                metadata,
                device,
                config["evaluation_batch_size"],
                config["grid_joint_rounds"],
            )
            result["weights"][kind] = {
                "per_round": summary,
                "answer_and_eos": {r: values[0] for r, values in joint.items()},
            }
    result.update(
        config_sha256=digest(config_path),
        model_sha256=model_digest(model),
        checkpoint_sha256=digest(path),
        utc=utc(),
    )
    write_json(out / "grid.json", result)


@contextmanager
def _nothing():
    yield


def tracker_for(config, name, out, config_path):
    import wandb

    tracking = config["tracking"]
    run = wandb.init(
        entity=tracking["entity"],
        project=tracking["project"],
        group=config["batch"],
        name=name,
        id=f"lnh{digest(config_path)[:6]}{list(config['arms']).index(name):02d}",
        config={"experiment": config},
        dir=str(out),
        mode=tracking["mode"],
        save_code=False,
        settings=wandb.Settings(disable_git=True),
    )
    write_json(out / "tracking.json", {"id": run.id, "url": run.url})

    def log(values, step):
        run.log(values, step=step)

    log.run = run
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "smoke-grid", "train", "grid"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--limit-rows", type=int, help="smoke stages only")
    parser.add_argument("--smoke-steps", type=int, help="smoke stage only")
    parser.add_argument("--smoke-grid-rounds", type=int, help="smoke-grid stage only")
    parser.add_argument("--threads", type=int, help="smoke stages only")
    args = parser.parse_args()
    engineering = (args.limit_rows, args.smoke_steps, args.smoke_grid_rounds, args.threads)
    if any(engineering) and not args.stage.startswith("smoke"):
        raise ValueError("Engineering overrides apply to smoke stages only")
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.smoke_steps:
        config["smoke_steps"] = args.smoke_steps
    if args.smoke_grid_rounds:
        config["grid_rounds"] = args.smoke_grid_rounds
        config["grid_joint_rounds"] = [
            r for r in config["grid_joint_rounds"] if r <= args.smoke_grid_rounds
        ]
    if args.arm not in config["arms"]:
        raise ValueError(f"Unknown arm: {args.arm}")
    torch.set_num_threads(args.threads or config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    elif args.stage in ("train", "grid"):
        raise ValueError("Scientific stages run on CUDA")
    try:
        if args.stage == "smoke":
            out = args.base / "smoke" / args.arm
            train(config, args.arm, out, True, args.config, None, device, args.limit_rows)
        elif args.stage == "train":
            out = args.base / "runs" / args.arm
            holder = {}

            def track(metadata):
                holder["log"] = tracker_for(config, args.arm, out, args.config)
                return holder["log"]

            try:
                train(config, args.arm, out, False, args.config, track, device)
            except BaseException:
                if "log" in holder:
                    holder["log"].run.finish(exit_code=1)
                raise
            holder["log"].run.finish()
        elif args.stage == "smoke-grid":
            grid(config, args.arm, args.base, args.config, device, "smoke", args.limit_rows)
        else:
            grid(config, args.arm, args.base, args.config, device)
    except BaseException as error:
        write_json(
            args.base / "failures" / f"{args.arm}-{args.stage}.json",
            {"utc": utc(), "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
