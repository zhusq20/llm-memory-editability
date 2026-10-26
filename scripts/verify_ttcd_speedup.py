#!/usr/bin/env python3
"""Fixed-checkpoint full-macro loss/gradient verification; zero optimizer steps."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from llm_memory_editability import ttcd_verified_base as base
from llm_memory_editability.grokking_reproduction import MergedBatchStream, model_digest
from llm_memory_editability.path_overlap import World, sha256
from llm_memory_editability.ttcd_continual import restore_data_rng
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest


def load_module(root, name):
    full_name = f"llm_memory_editability.{name}"
    spec = importlib.util.spec_from_file_location(
        full_name, root / "src/llm_memory_editability" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text())
    config["data_dir"] = "/inputs/data/ttcd-successful-base-v1"
    config["model_config"] = config["data_dir"] + "/gpt2-config.json"
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    world = World(config["data_dir"])
    reader = TTCDReader(
        base._backbone(config, saved, world, "cuda:0"),
        trainable_backbone=True,
        native_causal_attention=True,
    ).eval()
    backbone_before = model_digest(reader.backbone)
    table, _ = base.native_table(world, config["phi"])
    stream = MergedBatchStream(len(table[0]), config["seed"] + 3)
    stream.load_state_dict(saved["native_stream"])
    rng = restore_data_rng(saved["data_rng"])
    batch = base.sample_step(config, world, rng, stream)
    batch_digest = hashlib.sha256()
    for key, value in sorted(batch.items()):
        batch_digest.update(key.encode())
        batch_digest.update(value.astype("<i8", copy=False).tobytes())
    optimization = load_module(args.runtime_root, "ttcd_train_speedup")
    original_write_states = base.write_states
    records = []
    for phase in ("single", "stream"):
        expected_gradients, expected_fast, expected_losses = None, None, None
        times, provenance = {}, None
        for candidate in ("original", "optimized"):
            state_hashes, hashing_seconds = [], [0.0]

            def observed_write(
                *call_args, state_hashes=state_hashes, hashing_seconds=hashing_seconds, **kwargs
            ):
                fast = original_write_states(*call_args, **kwargs)
                torch.cuda.synchronize()
                started = time.monotonic()
                state_hashes.append(tensor_digest(fast))
                hashing_seconds[0] += time.monotonic() - started
                return fast

            base.write_states = observed_write
            undo = optimization.install() if candidate == "optimized" else None
            try:
                reader.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.monotonic()
                loss, components = base.backward_objective(
                    reader, config, world, table, batch, phase
                )
                torch.cuda.synchronize()
                times[candidate] = time.monotonic() - started - hashing_seconds[0]
                gradients = {
                    name: p.grad.detach().cpu().clone()
                    for name, p in reader.named_parameters()
                    if p.grad is not None
                }
                if not all(torch.isfinite(g).all() for g in gradients.values()):
                    raise FloatingPointError("Nonfinite fixed-checkpoint gradient")
                if candidate == "original":
                    expected_gradients, expected_fast, expected_losses = (
                        gradients,
                        state_hashes,
                        (loss, components),
                    )
                else:
                    if expected_losses != (loss, components) or expected_fast != state_hashes:
                        raise AssertionError("Fast states or full-macro losses differ")
                    if gradients.keys() != expected_gradients.keys():
                        raise AssertionError("Parameter gradient coverage differs")
                    differences = {
                        name: float((g - expected_gradients[name]).abs().max())
                        for name, g in gradients.items()
                        if not torch.equal(g, expected_gradients[name])
                    }
                    if differences:
                        raise AssertionError(
                            f"Parameter gradients must be bitwise exact: {differences}"
                        )
            finally:
                base.write_states = original_write_states
                if undo:
                    provenance = dict(undo.provenance)
                    undo()
            del gradients
        records.append(
            {
                "phase": phase,
                "all_fast_states_exact": True,
                "losses_exact": True,
                "all_parameter_gradients_exact": True,
                "gradient_tensors": len(expected_gradients),
                "fast_state_digests": expected_fast,
                "seconds": times,
                "observed_ratio": times["original"] / times["optimized"],
                "runtime_provenance": provenance,
            }
        )
        del expected_gradients
    if model_digest(reader.backbone) != backbone_before:
        raise AssertionError("Fixed-checkpoint verification changed model weights")
    result = {
        "passed": True,
        "optimizer_steps": 0,
        "scientific_training_updates": 0,
        "formal_model_queries": 0,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": saved["step"],
        "model_sha256": backbone_before,
        "native_batch": len(batch["native_indices"]),
        "correction_batch": len(batch["packets"]),
        "micro_native": config["training"]["micro_native"],
        "micro_facts": config["training"]["micro_facts"],
        "batch_sha256": batch_digest.hexdigest(),
        "precision": "original bf16 autocast; TF32 off",
        "checks": records,
        "benchmark_scope": (
            "Temporary numerical verification sharing the native GPU; ratios are diagnostic only"
        ),
        "cuda_device": torch.cuda.get_device_name(),
        "base_parameters_unchanged": True,
    }
    (args.out / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
