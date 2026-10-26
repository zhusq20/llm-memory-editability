#!/usr/bin/env python3
"""Discarded CUDA state-continuity checks, using the actual completed checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_paper_update_continuation as continuation  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import (  # noqa: E402
    MergedBatchStream,
    model_digest,
)


def clone_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_cpu(v) for v in value]
    return copy.deepcopy(value)


def capture(config, name, metadata, table, state, factor, split):
    device = table[0].device
    model, spec, arm, _ = continuation.adapter.initialize(config, name, metadata, device)
    model.load_state_dict(state["model"], strict=True)
    if split:
        optimizer, _, _ = continuation.restore_optimizer(
            model, clone_cpu(state["optimizer"]), spec, factor, device
        )
    else:
        lr = torch.tensor(spec["lr"], device=device)
        optimizer = continuation.adapter.optimizer_for(model, lr, spec["weight_decay"])
        optimizer.load_state_dict(clone_cpu(state["optimizer"]))
        for group in optimizer.param_groups:
            group["lr"] = lr
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], device)
    model.train()
    before = model_digest(model), continuation.optimizer_digest(model, optimizer)
    graph = continuation.native.AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    assert before == (model_digest(model), continuation.optimizer_digest(model, optimizer))
    assert torch.equal(state["cpu_rng"], torch.get_rng_state())
    assert torch.equal(state["cuda_rng"], torch.cuda.get_rng_state(device))
    stream = MergedBatchStream(state["stream"]["size"], spec["stream_seed"])
    stream.load_state_dict(state["stream"])
    return model, optimizer, graph, stream


def take_step(graph, stream, batch_size, device):
    indices = stream.batch(batch_size)[0]
    graph(torch.as_tensor(indices, device=device))
    torch.cuda.synchronize(device)
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    continuation.validate_config(config)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    world, _, metadata = continuation.native.load(config["input_root"])
    table = continuation.native.make_table(world, metadata, device)
    results = {}
    for name in ("paper", "fixed-update", "token-update"):
        source, _, _ = continuation.load_parent(config, name, metadata)
        reference, optimizer, graph, stream = capture(
            config, name, metadata, table, source, 1.0, False
        )
        first_indices = take_step(graph, stream, config["training"]["batch_size"], device)
        first = {
            "model": clone_cpu(reference.state_dict()),
            "optimizer": clone_cpu(optimizer.state_dict()),
            "stream": stream.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
        }
        next_indices = take_step(graph, stream, config["training"]["batch_size"], device)
        expected = model_digest(reference), continuation.optimizer_digest(reference, optimizer)
        expected_cuda = torch.cuda.get_rng_state(device)
        expected_cpu = torch.get_rng_state()
        expected_loss = float(graph.loss)
        del reference, optimizer, graph, stream
        gc.collect()
        torch.cuda.empty_cache()
        candidate, optimizer, graph, stream = capture(
            config, name, metadata, table, first, 1.0, True
        )
        resumed_indices = take_step(graph, stream, config["training"]["batch_size"], device)
        actual = model_digest(candidate), continuation.optimizer_digest(candidate, optimizer)
        findings = {
            "next_sample_ids_exact": np.array_equal(next_indices, resumed_indices),
            "all_model_tensors_exact": actual[0] == expected[0],
            "all_adam_tensors_exact": actual[1] == expected[1],
            "cpu_rng_exact": torch.equal(expected_cpu, torch.get_rng_state()),
            "cuda_rng_exact": torch.equal(expected_cuda, torch.cuda.get_rng_state(device)),
            "loss_exact": float(graph.loss) == expected_loss,
        }
        results[name] = {"findings": findings, "model_sha256": actual[0]}
        del candidate, optimizer, graph, stream
        gc.collect()
        torch.cuda.empty_cache()
        if name == "token-update":
            candidate, optimizer, graph, stream = capture(
                config, name, metadata, table, source, 0.5, True
            )
            half_indices = take_step(graph, stream, config["training"]["batch_size"], device)
            state = clone_cpu(candidate.state_dict())
            regular_exact = all(
                torch.equal(value, first["model"][key])
                for key, value in state.items()
                if key not in continuation.GATE_NAMES
            )
            ratios = {}
            for key in continuation.GATE_NAMES:
                full_delta = first["model"][key] - source["model"][key]
                half_delta = state[key] - source["model"][key]
                ratios[key] = float(half_delta.norm() / full_delta.norm())
            results["half_lr_intervention"] = {
                "findings": {
                    "next_sample_ids_exact": np.array_equal(first_indices, half_indices),
                    "all_non_update_gate_parameters_exact": regular_exact,
                    "gate_step_norm_halved": all(
                        abs(value - 0.5) < 0.003 for value in ratios.values()
                    ),
                },
                "gate_update_norm_ratios": ratios,
            }
            # Compare moments by original parameter name despite split groups.
            names = dict(candidate.named_parameters())
            inherited_groups = continuation.adapter.optimizer_for(
                candidate, torch.tensor(1e-4, device=device), 0.1
            )
            group_names = continuation.parameter_groups(candidate, inherited_groups)
            matched = True
            for group, original_names in zip(
                first["optimizer"]["param_groups"], group_names, strict=True
            ):
                for index, parameter_name in zip(group["params"], original_names, strict=True):
                    for key, value in first["optimizer"]["state"][index].items():
                        matched &= torch.equal(
                            value, optimizer.state[names[parameter_name]][key].cpu()
                        )
            results["half_lr_intervention"]["findings"]["all_adam_tensors_exact"] = bool(matched)
            del candidate, optimizer, graph, stream, inherited_groups
            gc.collect()
            torch.cuda.empty_cache()
        print(json.dumps({"arm": name, "result": results[name]}), flush=True)
    result = {
        "passed": all(all(row["findings"].values()) for row in results.values()),
        "engineering_only": True,
        "scientific_updates": 0,
        "discarded_effective_update_executions": 10,
        "restored_graph_preparation_update_executions": 28,
        "actual_parent_checkpoints": True,
        "image": config["runtime"]["image"],
        "checks": results,
        "utc": utc(),
    }
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise AssertionError("Full-size CUDA continuation equivalence failed")


if __name__ == "__main__":
    main()
