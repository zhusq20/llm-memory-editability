#!/usr/bin/env python3
"""Bounded GPU validation of the terminal Poisson recipe; no scientific training.

Each case starts from the registered initialization. Scratch optimizer updates only
exercise the production forward/backward/clip/AdamW/EMA path; no trained weights are
saved or reused. Both per-step measurements and failures are retained.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_native_handoff as runner  # noqa: E402

from llm_memory_editability.loop_terminal_training import backward_terminal_loss  # noqa: E402


def gradient_summary(model):
    """Inspect gradients after clipping, outside the timed update."""
    groups = {name: [] for name in ("backbone", "embedding", "score", "decay")}
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(parameter.grad).all() or not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite parameter or gradient: {name}")
        group = (
            "score"
            if name.startswith("handoff_score_")
            else "decay"
            if name == "handoff_beta_logit"
            else "embedding"
            if ".wte." in name or ".wpe." in name
            else "backbone"
        )
        groups[group].append(parameter.grad.float().square().sum())
    if missing:
        raise AssertionError(f"Disconnected trainable parameters: {missing}")
    return {name: float(torch.stack(values).sum().sqrt()) for name, values in groups.items()}


def optimizer_finite(optimizer):
    return all(
        bool(torch.isfinite(value).all())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def early_gradient_check(model, batch, depth):
    """The full-size terminal loss must reach the first-round carry and decode."""
    model.zero_grad(set_to_none=True)
    original = model.aggregate_round
    observed = {}

    def capture(state, injection, *, first_round=False):
        result, gamma = original(state, injection, first_round=first_round)
        if first_round:
            for name, value in zip(("h1", "N1", "D1", "p1"), result[:4], strict=True):

                def record(grad, key=name):
                    observed[key] = {
                        "finite": bool(torch.isfinite(grad).all()),
                        "norm": float(grad.float().norm()),
                    }

                value.register_hook(record)
        return result, gamma

    model.aggregate_round = capture
    try:
        loss = backward_terminal_loss(
            model,
            *batch,
            np.full(len(batch[0]), depth, dtype=np.int64),
            forward_context=lambda: runner.autocast(torch.device("cuda:0")),
        )
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or set(observed) != {"h1", "N1", "D1", "p1"}:
            raise AssertionError("Terminal gradient did not reach the first round")
        if not all(v["finite"] and v["norm"] > 0 for v in observed.values()):
            raise AssertionError(f"Invalid first-round gradients: {observed}")
        return observed
    finally:
        model.aggregate_round = original
        model.zero_grad(set_to_none=True)


def run_case(config, metadata, table, longest, output, mode, warmup, measured):
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    model = runner.initialize(config, "invariant", metadata, device)
    if runner.backbone_digest(model) != config["paired_recipe"]["initial_backbone_sha256"]:
        raise AssertionError("Initialization differs from the registered backbone")
    model.train()
    spec = config["training"]
    optimizer = runner.optimizer_for(model, spec, fused=True)
    average = runner.ParameterAverage(model, spec["ema_decay"])
    data_rng = np.random.default_rng(spec["stream_seed"])
    depth_rng = np.random.default_rng(spec["schedule_seed"])
    width = spec["batch_size"]
    records = []
    initial_probe = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    logfile = output / f"{mode}.jsonl"
    for step in range(1, warmup + measured + 1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = runner.learning_rate(spec, step)
        source = table if mode == "poisson" else longest
        rows = data_rng.integers(0, len(source[0]) - (mode == "poisson"), size=width)
        index = torch.as_tensor(rows, device=device)
        batch = tuple(part[index] for part in source)
        if mode == "poisson":
            loss, _, depths = runner.backward_batch(model, *batch, spec, depth_rng, device)
        else:
            depths = np.full(width, 16, dtype=np.int64)
            loss = backward_terminal_loss(
                model, *batch, depths, forward_context=lambda: runner.autocast(device)
            )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), spec["grad_clip"], foreach=True, error_if_nonfinite=True
        )
        optimizer.step()
        average.update(model)
        torch.cuda.synchronize()
        milliseconds = (time.perf_counter() - started) * 1000
        allocated = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        if not torch.isfinite(loss) or not math.isfinite(milliseconds):
            raise FloatingPointError("Invalid loss or timing")
        gradients = gradient_summary(model)
        if not optimizer_finite(optimizer):
            raise FloatingPointError("Non-finite Adam state")
        record = {
            "execution": spec["execution"],
            "step": step,
            "warmup": step <= warmup,
            "loss": float(loss),
            "gradient_norm_before_clip": float(norm),
            "gradient_norm_after_clip": gradients,
            "milliseconds": milliseconds,
            "samples_per_second": width * 1000 / milliseconds,
            "peak_allocated_mib": allocated,
            "peak_reserved_mib": reserved,
            "depth_histogram": np.bincount(depths, minlength=17).tolist(),
            "mean_depth": float(np.mean(depths)),
            "maximum_depth": int(max(depths)),
            "sampled_sample_rounds": int(sum(depths)),
            "executed_sample_rounds": width * int(max(depths)),
            "shared_block_calls": int(max(depths)),
            "valid_supervised_tokens": int(batch[2].ne(-100).sum()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        records.append(record)
        with logfile.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps({"case": mode, **record}), flush=True)
    changed = float((model.transformer.h[0].attn.c_attn.weight - initial_probe).abs().max())
    if changed <= 0:
        raise AssertionError("Scratch Adam updates did not change the backbone")
    early = early_gradient_check(model, batch, 16)
    timed = [r for r in records if not r["warmup"]]
    latency = np.array([r["milliseconds"] for r in timed])
    summary = {
        "passed": True,
        "case": mode,
        "execution": spec["execution"],
        "batch_size": width,
        "scratch_updates": len(records),
        "warmup_updates": warmup,
        "measured_updates": measured,
        "mean_ms": float(latency.mean()),
        "median_ms": float(np.median(latency)),
        "p95_ms": float(np.percentile(latency, 95)),
        "samples_per_second": width * 1000 / float(latency.mean()),
        "peak_allocated_mib": max(r["peak_allocated_mib"] for r in records),
        "peak_reserved_mib": max(r["peak_reserved_mib"] for r in records),
        "mean_depth": float(np.mean([r["mean_depth"] for r in timed])),
        "maximum_depth_observed": max(r["maximum_depth"] for r in records),
        "loss_first": records[0]["loss"],
        "loss_last": records[-1]["loss"],
        "gradient_norm_range": [
            min(r["gradient_norm_before_clip"] for r in records),
            max(r["gradient_norm_before_clip"] for r in records),
        ],
        "all_parameters_gradients_and_adam_finite": True,
        "first_round_terminal_gradients": early,
        "backbone_parameter_max_change": changed,
        "timing_scope": "sample/index, zero_grad, forward/backward, clip, AdamW and EMA; "
        "synchronized wall time; excludes diagnostic scans and log IO",
    }
    runner.write_json(output / f"{mode}.json", summary)
    del model, optimizer, average, batch, initial_probe
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--poisson-steps", type=int, default=20)
    parser.add_argument("--stress-steps", type=int, default=10)
    args = parser.parse_args()
    if not (
        1 <= args.warmup <= 10 and 1 <= args.poisson_steps <= 30 and 1 <= args.stress_steps <= 20
    ):
        raise ValueError("This entrypoint is restricted to a bounded engineering check")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    runner.validate_config(config)
    if config["batch"] != runner.TERMINAL_BATCH or config["training"]["batch_size"] != 512:
        raise ValueError("Expected the registered batch-512 terminal recipe")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.time()
    result = {"passed": False, "formal_training_updates": 0, "utc": runner.utc()}
    try:
        probe = torch.randn(1024, 1024, device="cuda", requires_grad=True)
        probe_loss = (probe @ probe.T).square().mean()
        probe_loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(probe_loss) or not torch.isfinite(probe.grad).all():
            raise FloatingPointError("CUDA matrix/backward/synchronize check failed")
        del probe, probe_loss
        world, _, metadata = runner.load(args.input_root)
        for key in ("world_sha256", "panels_sha256"):
            if metadata[key] != config["input_origin"][key]:
                raise ValueError(f"Unexpected input data: {key}")
        table = runner.make_table(world, metadata, torch.device("cuda:0"))
        longest = tuple(
            torch.as_tensor(p, device="cuda") for p in runner.pack_rows(world["train_4"], metadata)
        )
        props = torch.cuda.get_device_properties(0)
        environment = {
            "hostname": platform.node(),
            "gpu": props.name,
            "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
            "gpu_memory_mib": props.total_memory / 2**20,
            "cuda": torch.version.cuda,
            "versions": {
                p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")
            },
            "matrix_backward_synchronized": True,
            "batch_size": 512,
            "execution": config["training"]["execution"],
            "width": 768,
            "shared_blocks": 1,
            "sequence_length": metadata["sequence_length"],
            "vocab_size": metadata["vocab_size"],
            "autocast": "bfloat16",
            "parameter_optimizer_ema_dtype": "float32",
            "dropout": config["model"]["resid_pdrop"],
            "allocator_fraction": config["runtime"]["gpu_memory_fraction"],
            "config_sha256": runner.digest(args.config),
            "data_hashes": {k: metadata[k] for k in ("world_sha256", "panels_sha256")},
            "source_sha256": {
                str(p.relative_to(ROOT)): runner.digest(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/run_loop_native_handoff.py",
                    ROOT / "src/llm_memory_editability/loop_native_handoff.py",
                    ROOT / "src/llm_memory_editability/loop_terminal_training.py",
                )
            },
        }
        runner.write_json(args.output / "environment.json", environment)
        cases = [
            run_case(config, metadata, table, longest, args.output, mode, args.warmup, steps)
            for mode, steps in (
                ("poisson", args.poisson_steps),
                ("all_depth_16", args.stress_steps),
            )
        ]
        result.update(
            passed=True,
            cases=cases,
            environment=environment,
            engineering_updates=sum(c["scratch_updates"] for c in cases),
            early_gradient_checks=2,
            trained_weights_saved=False,
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.time() - started
        runner.write_json(args.output / "result.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
