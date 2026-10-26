#!/usr/bin/env python3
"""Continue the two u=.95 arms without changing their optimizer or data trajectory.

The inherited training/evaluation loop is reused through scoped adapters. The
u=.90 continuation batch remains separate and none of its files are changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import check_paper_update_continuation as checks  # noqa: E402
import run_paper_update_continuation as inherited  # noqa: E402

adapter, native, depth = inherited.adapter, inherited.native, inherited.depth
ARMS = ("fixed-update", "token-update")
IMAGE = "sha256:b7933a57d6a4702700b6b2578fe0cc697f91b2080ef5e18d8fb916443281eacd"
PARENT_FILES = {
    "latest.pt",
    "complete.json",
    "run.json",
    "learning.json",
    "exposure.npz",
    "config.json",
}
original_load_parent = inherited.load_parent


def validate_config(config):
    if config["batch"] not in (
        "gate-generalization-extension-v1",
        "gate-generalization-extension-v1-continuation",
    ):
        raise ValueError("Use a separate extension batch, never the existing u90 batch")
    if tuple(config["arms"]) != ARMS or set(config["parents"]) != set(ARMS):
        raise ValueError("Only the two ordered u95 continuation arms are authorized here")
    continuation = config["continuation"]
    if (
        continuation["start_step"] != 300000
        or config["training"]["steps"] != 1500000
        or config["budget"]["new_scientific_updates"] != 2400000
    ):
        raise ValueError("Continue two existing 300k endpoints to 1500k: 2.4M new updates")
    if (
        not continuation["preserve_optimizer"]
        or not continuation["preserve_stream_and_rng"]
        or continuation["new_warmup"]
    ):
        raise ValueError("Preserve full optimizer, sampler and RNG; never restart warmup")
    if config["evaluation_nodes"] != list(range(300000, 1500001, 100000)):
        raise ValueError("Preserve the predeclared absolute-step evaluation nodes")
    if config["runtime"]["image"] != IMAGE:
        raise ValueError("Continue in the exact local u95 parent image")
    for name, mode in zip(ARMS, ("fixed", "token"), strict=True):
        expected = {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "update_mode": mode,
            "initial_update": 0.95,
            "parent": name,
            "update_gate_lr_factor": 1.0,
        }
        if config["arms"][name] != expected:
            raise ValueError(f"Changed u95 continuation arm: {name}")
        parent = config["parents"][name]
        if parent["source_image"] != IMAGE or not PARENT_FILES <= parent["files"].keys():
            raise ValueError(f"Parent image or complete artifact hashes missing: {name}")
        for value in [parent["model_sha256"], *parent["files"].values()]:
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Every parent artifact must have a SHA256 identity")
    for key, value in {"lr": 1e-4, "weight_decay": 0.1, "warmup": 2000, "batch_size": 512}.items():
        if config["training"][key] != value:
            raise ValueError(f"Changed inherited training recipe: {key}")


def load_parent(config, name, metadata):
    source, complete, parent = original_load_parent(config, name, metadata)
    original = json.loads((Path(parent["root"]) / "config.json").read_text())
    expected = {
        key: value
        for key, value in config["arms"][name].items()
        if key not in ("parent", "update_gate_lr_factor")
    }
    if original["batch"] != "paper-update-gate-multihop-u95-v1":
        raise ValueError("A u90 endpoint or another batch cannot substitute for the u95 parent")
    if original["arms"][name] != expected:
        raise ValueError("Parent architecture and update-gate initialization differ")
    if any(source["spec"][key] != value for key, value in expected.items()):
        raise ValueError("Checkpoint arm recipe differs from its hashed parent configuration")
    if original["runtime"]["image"] != IMAGE:
        raise ValueError("Parent runtime does not match the locked continuation image")
    if complete["arm"] != name or source["spec"]["name"] != name:
        raise ValueError("Parent endpoint arm identity mismatch")
    if original["training"]["steps"] != 300000 or source["spec"]["steps"] != 300000:
        raise ValueError("The parent must be the original completed 300k training")
    return source, complete, parent


def restore_optimizer(model, saved, spec, factor, device):
    """Retain the original two parameter groups, moments, options and learning rate."""
    if factor != 1.0:
        raise ValueError("No gate learning-rate intervention belongs to this continuation")
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = adapter.optimizer_for(model, lr, spec["weight_decay"])
    if len(saved["param_groups"]) != 2:
        raise ValueError("Parent must retain the original two optimizer parameter groups")
    expected_names = inherited.parameter_groups(model, optimizer)
    optimizer.load_state_dict(saved)
    before = inherited.optimizer_digest(model, optimizer)
    for group, expected_decay in zip(
        optimizer.param_groups, (spec["weight_decay"], 0.0), strict=True
    ):
        if (
            group["weight_decay"] != expected_decay
            or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or not group["fused"]
            or not group["capturable"]
            or not math.isclose(float(group["lr"]), float(lr), rel_tol=1e-7)
        ):
            raise ValueError("Changed parent AdamW options or learning rate")
        # Loading state replaces the LR tensor. Rebind before CUDA capture,
        # retaining its value and the exact original parameter-group order.
        group["lr"] = lr
    if inherited.parameter_groups(model, optimizer) != expected_names:
        raise AssertionError("Optimizer parameter order changed")
    if inherited.optimizer_digest(model, optimizer) != before:
        raise AssertionError("Restoration changed Adam moments or step counters")
    learned = any(name in inherited.GATE_NAMES for name, _ in model.named_parameters())
    return optimizer, lr, lr if learned else None


@contextmanager
def installed():
    """Keep all inherited modules unchanged outside this invocation."""
    saved_restore, saved_load = inherited.restore_optimizer, inherited.load_parent
    saved_write, saved_train = native.write_json, native.train

    def write_json(path, value):
        if Path(path).name == "run.json" and "continuation_start_step" in value:
            value["rng_policy"] = (
                "Restore this arm's own full CPU/CUDA/dropout and stream states. "
                "Keep both original optimizer groups in their original order and at full LR."
            )
            value["parent_runtime_matches"] = True
        saved_write(path, value)

    try:
        inherited.restore_optimizer = restore_optimizer
        inherited.load_parent = load_parent
        native.write_json = write_json
        native.train = inherited.train
        with adapter.installed():
            yield
    finally:
        inherited.restore_optimizer, inherited.load_parent = saved_restore, saved_load
        native.write_json, native.train = saved_write, saved_train


def check_resume(config, name, output):
    """Discard three actual updates to compare continuous and reloaded CUDA states."""
    destination = output / "checks" / f"{name}-resume.json"
    if destination.exists():
        raise FileExistsError(destination)
    world, _, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    table = native.make_table(world, metadata, device)
    source, _, parent = load_parent(config, name, metadata)
    reference, optimizer, graph, stream = checks.capture(
        config, name, metadata, table, source, 1.0, False
    )
    batch_size = config["training"]["batch_size"]
    first_indices = checks.take_step(graph, stream, batch_size, device)
    first = {
        "model": checks.clone_cpu(reference.state_dict()),
        "optimizer": checks.clone_cpu(optimizer.state_dict()),
        "stream": stream.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device),
    }
    next_indices = checks.take_step(graph, stream, batch_size, device)
    expected = inherited.model_digest(reference), inherited.optimizer_digest(reference, optimizer)
    expected_cpu, expected_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    expected_loss = float(graph.loss)
    expected_groups = inherited.parameter_groups(reference, optimizer)
    del reference, optimizer, graph, stream
    gc.collect()
    torch.cuda.empty_cache()
    candidate, optimizer, graph, stream = checks.capture(
        config, name, metadata, table, first, 1.0, True
    )
    resumed_indices = checks.take_step(graph, stream, batch_size, device)
    actual = inherited.model_digest(candidate), inherited.optimizer_digest(candidate, optimizer)
    size = len(source["counts"])
    prefix_counts = source["counts"] + np.bincount(first_indices, minlength=size + 1)[:size]
    expected_counts = prefix_counts + np.bincount(next_indices, minlength=size + 1)[:size]
    actual_counts = prefix_counts + np.bincount(resumed_indices, minlength=size + 1)[:size]
    findings = {
        "next_sample_ids_exact": np.array_equal(next_indices, resumed_indices),
        "all_model_tensors_exact": actual[0] == expected[0],
        "all_adam_tensors_exact": actual[1] == expected[1],
        "cpu_rng_exact": torch.equal(expected_cpu, torch.get_rng_state()),
        "cuda_rng_exact": torch.equal(expected_cuda, torch.cuda.get_rng_state(device)),
        "loss_exact": float(graph.loss) == expected_loss,
        "original_two_parameter_groups_exact": inherited.parameter_groups(candidate, optimizer)
        == expected_groups
        and len(expected_groups) == 2,
        "accumulated_exposure_exact": np.array_equal(actual_counts, expected_counts),
    }
    result = {
        "passed": all(findings.values()),
        "arm": name,
        "engineering_only": True,
        "scientific_updates": 0,
        "discarded_effective_update_executions": 3,
        "restored_graph_preparation_update_executions": 8,
        "parent_full_checkpoint_sha256": parent["files"]["latest.pt"],
        "image": config["runtime"]["image"],
        "findings": findings,
        "utc": inherited.utc(),
    }
    native.write_json(destination, result)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise AssertionError("Full-size CUDA continuation equivalence failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "grid", "check-resume"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with installed():
            if args.stage == "check-resume":
                check_resume(config, args.arm, args.output)
            elif args.stage == "grid":
                depth.endpoint(config, args.arm, args.output, args.config)
            else:
                depth.tracked_train(config, args.arm, args.output, args.config)
    except BaseException as error:
        native.write_json(
            args.output / "failures" / f"{args.arm}-{args.stage}.json",
            {"error": repr(error), "traceback": traceback.format_exc(), "utc": inherited.utc()},
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
