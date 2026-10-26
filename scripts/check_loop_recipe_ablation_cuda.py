#!/usr/bin/env python3
"""Bounded shared-GPU checks; scratch weights never enter the formal runs."""

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import run_loop_fourway as fourway
import run_loop_recipe_ablation as controls
import torch

from llm_memory_editability.loop_recipe_ablation import recipe_loss, split_boundary


def check(config, arm):
    controls.validate_config(config)
    native = fourway.native
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = True
    probe = torch.randn(1024, 1024, device="cuda", requires_grad=True)
    (probe @ probe.T).square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(probe.grad).all()
    del probe
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")
    world, _, metadata = native.load(config["input_root"])
    for key in ("world_sha256", "panels_sha256"):
        assert metadata[key] == config["input_origin"][key]
    table = native.make_table(world, metadata, device)
    longest = tuple(
        torch.as_tensor(p, device=device) for p in native.pack_rows(world["train_4"], metadata)
    )
    model = controls.initialize(config, arm, metadata, device).train()
    assert native.backbone_digest(model) == config["paired_recipe"]["initial_backbone_sha256"]
    training = config["training"]
    optimizer = native.optimizer_for(model, training, fused=True)
    average = native.ParameterAverage(model, training["ema_decay"])
    rng = np.random.default_rng(training["schedule_seed"])
    data_rng = np.random.default_rng(training["stream_seed"])
    records = []
    torch.cuda.reset_peak_memory_stats()
    for mode in ("poisson", "longest_depth16"):
        for step in range(1, 4):
            source = table if mode == "poisson" else longest
            rows = data_rng.integers(
                0, len(source[0]) - (mode == "poisson"), size=training["batch_size"]
            )
            batch = tuple(part[torch.as_tensor(rows, device=device)] for part in source)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = controls.learning_rate(training, len(records) + 1)
            torch.cuda.synchronize()
            started = time.perf_counter()
            if mode == "poisson":
                loss, _, depths = controls.backward_batch(model, *batch, training, rng, device)
            else:
                depths = np.full(training["batch_size"], 16, dtype=np.int64)
                with native.autocast(device):
                    loss = recipe_loss(
                        model,
                        *batch,
                        depths,
                        split_boundary(training["split_sampler"]["seed"], len(records) + 1),
                        supervision=controls.ARMS[arm]["supervision"],
                        truncate=controls.ARMS[arm]["truncate"],
                    )
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["grad_clip"], foreach=True, error_if_nonfinite=True
            )
            assert torch.isfinite(loss) and torch.isfinite(norm)
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
            optimizer.step()
            average.update(model)
            torch.cuda.synchronize()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert all(
                torch.isfinite(v).all()
                for state in optimizer.state.values()
                for v in state.values()
                if isinstance(v, torch.Tensor)
            )
            record = {
                "mode": mode,
                "step": step,
                "loss": float(loss),
                "grad_norm": float(norm),
                "seconds": time.perf_counter() - started,
                "maximum_depth": int(max(depths)),
                "batch_size": len(batch[0]),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), average.applied(model), native.autocast(device):
        logits, _, _ = model.trace(longest[0][:256], longest[1][:256], 32)
        assert torch.isfinite(logits).all()
    torch.cuda.synchronize()
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    cap = total_mib * config["runtime"]["gpu_memory_fraction"]
    assert reserved + 6144 < cap, "Require 6 GiB above measured peak inside the allocator cap"
    assert all(math.isfinite(r["seconds"]) for r in records)
    return {
        "passed": True,
        "arm": arm,
        "gpu_index": int(os.environ["LM_HOST_GPU_INDEX"]),
        "gpu_uuid": os.environ["LM_HOST_GPU_UUID"],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "scratch_updates": 6,
        "scientific_updates": 0,
        "weights_saved_or_reused": False,
        "recipe_flags": controls.ARMS[arm],
        "ema_depth32_evaluation_finite": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": reserved,
        "allocator_cap_mib": cap,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=list(controls.ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check(json.loads(args.config.read_text()), args.arm)
    except BaseException:
        fourway.native.write_json(
            args.output, {"passed": False, "arm": args.arm, "error": traceback.format_exc()}
        )
        raise
    fourway.native.write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
