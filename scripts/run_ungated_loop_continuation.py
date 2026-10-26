#!/usr/bin/env python3
"""Continue native Loop and fixed-injection DiscoLoop, without adding any gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_paper_update_continuation as continuation  # noqa: E402

native, depth = continuation.native, continuation.depth
ARMS = ("disco-l1-r4", "loop-l1-r4")


def validate_config(config):
    if config["batch"] != "ungated-loop-continuation-v1" or tuple(config["arms"]) != ARMS:
        raise ValueError("Only the two authorized ungated baselines may run")
    if config["continuation"] != {"start_step": 300000, "new_warmup": False}:
        raise ValueError("Resume full 300k state without a new warmup")
    if config["training"]["steps"] != 1500000:
        raise ValueError("End at total 1500000 updates")
    if config["evaluation_nodes"] != list(range(300000, 1500001, 100000)):
        raise ValueError("Preserve absolute-step evaluation schedule")
    if config["budget"]["new_scientific_updates"] != 2400000:
        raise ValueError("Exactly 2 x 1.2M new updates")
    for name, architecture in zip(ARMS, ("discoloop", "loop"), strict=True):
        if config["arms"][name] != {
            "architecture": architecture,
            "unique_layers": 1,
            "repeats": 4,
            "parent": name,
            "update_gate_lr_factor": 1.0,
        }:
            raise ValueError("Changed baseline architecture or optimizer")
    if config["training"]["lr"] != 1e-4 or config["training"]["weight_decay"] != 0.1:
        raise ValueError("Preserve original learning rate and decay")


@contextmanager
def installed():
    """Use the native model/Adam/accounting with the existing full-state trainer."""
    previous = continuation.adapter
    continuation.adapter = SimpleNamespace(
        initialize=native.initialize,
        optimizer_for=native.optimizer_for,
        accounting=native.accounting,
    )
    try:
        yield
    finally:
        continuation.adapter = previous


def preflight(config, name, base):
    """Validate actual checkpoint restoration and deterministic CUDA continuation.

    Two one-update copies are discarded. The source checkpoint is read-only.
    This is an engineering check, not a new scientific short-training arm.
    """
    world, panels, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    table = native.make_table(world, metadata, device)
    findings, results = {}, []
    for repeat in range(2):
        prepared = continuation.prepare(config, name, world, metadata, device)
        model, spec, arm, optimizer, lr, gate_lr, stream, counts, source, complete, _, initial = (
            prepared
        )
        findings[f"r{repeat}_native_initial_hash"] = initial == complete["initial_model_sha256"]
        findings[f"r{repeat}_no_gate_parameters"] = not any(
            "gate_" in n for n in model.state_dict()
        )
        findings[f"r{repeat}_architecture"] = arm == source["arm"]
        findings[f"r{repeat}_no_gate_lr"] = gate_lr is None
        before = continuation.optimizer_digest(model, optimizer)
        graph = native.AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
        findings[f"r{repeat}_capture_model"] = native.model_digest(model) == source["model_sha256"]
        findings[f"r{repeat}_capture_adam"] = before == continuation.optimizer_digest(
            model, optimizer
        )
        findings[f"r{repeat}_capture_rng"] = torch.equal(
            torch.get_rng_state(), source["cpu_rng"]
        ) and torch.equal(torch.cuda.get_rng_state(), source["cuda_rng"])
        index = stream.batch(spec["batch_size"])[0]
        lr.fill_(native.learning_rate(spec, source["step"] + 1))
        graph(torch.as_tensor(index, device=device))
        torch.cuda.synchronize()
        findings[f"r{repeat}_finite"] = all(
            np.isfinite(v) for v in native.graph_values(graph, 0).values()
        )
        findings[f"r{repeat}_changed"] = native.model_digest(model) != source["model_sha256"]
        results.append(
            (
                native.model_digest(model),
                continuation.optimizer_digest(model, optimizer),
                index.copy(),
            )
        )
        del graph, prepared, model, optimizer, source
        torch.cuda.empty_cache()
    findings["repeated_model_exact"] = results[0][0] == results[1][0]
    findings["repeated_adam_exact"] = results[0][1] == results[1][1]
    findings["repeated_sampler_exact"] = np.array_equal(results[0][2], results[1][2])
    receipt = {
        "passed": all(findings.values()),
        "findings": findings,
        "engineering_updates_discarded": 2,
        "retained_updates": 0,
        "config_sha256": config["_config_sha256"],
        "utc": continuation.utc(),
    }
    native.write_json(base / "preflight" / f"{name}.json", receipt)
    if not receipt["passed"]:
        raise AssertionError(findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "grid"), required=True)
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
    with installed():
        if args.stage == "preflight":
            preflight(
                {**config, "_config_sha256": native.digest(args.config)}, args.arm, args.output
            )
        elif args.stage == "grid":
            depth.endpoint(config, args.arm, args.output, args.config)
        else:
            original = native.train
            try:
                native.train = continuation.train
                depth.tracked_train(config, args.arm, args.output, args.config)
            finally:
                native.train = original


if __name__ == "__main__":
    os.umask(0o022)
    main()
