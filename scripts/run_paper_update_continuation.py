#!/usr/bin/env python3
"""Continue the completed 300k paper/update-gate models with full training state.

The inherited model, objective, sampler, CUDA graph step and scoring are unchanged.
Only the explicitly selected update-gate parameter group can use a different LR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_paper_update_gate as adapter  # noqa: E402

from llm_memory_editability.grok_depth import utc  # noqa: E402
from llm_memory_editability.grokking_reproduction import (  # noqa: E402
    MergedBatchStream,
    digest,
    learning_rate,
    model_digest,
)

native, depth = adapter.native, adapter.depth
ARMS = ("paper", "fixed-update", "token-update", "token-update-half-lr")
GATE_NAMES = {"gate_update_weight", "gate_update_bias"}


def validate_config(config):
    if tuple(config["arms"]) != ARMS:
        raise ValueError("The authorized four-arm continuation matrix is fixed")
    if config["continuation"]["start_step"] != 300000 or config["training"]["steps"] != 1500000:
        raise ValueError("Continue 300k to total 1500k, without resetting the step counter")
    if config["evaluation_nodes"] != list(range(300000, 1500001, 100000)):
        raise ValueError("Preserve all predeclared absolute-step evaluation nodes")
    if config["budget"]["new_scientific_updates"] != 4800000:
        raise ValueError("New scientific budget is four times 1.2M updates")
    for name, mode, parent, factor in (
        ("paper", "full", "paper", 1.0),
        ("fixed-update", "fixed", "fixed-update", 1.0),
        ("token-update", "token", "token-update", 1.0),
        ("token-update-half-lr", "token", "token-update", 0.5),
    ):
        expected = {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "update_mode": mode,
            "initial_update": 0.9,
            "parent": parent,
            "update_gate_lr_factor": factor,
        }
        if config["arms"][name] != expected:
            raise ValueError(f"Changed authorized arm: {name}")
    if config["training"]["lr"] != 1e-4 or config["training"]["weight_decay"] != 0.1:
        raise ValueError("Backbone LR and weight decay must remain unchanged")
    if config["continuation"]["new_warmup"]:
        raise ValueError("Do not repeat warmup")


def parameter_groups(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def optimizer_digest(model, optimizer):
    """Hash every Adam tensor by parameter name, independent of group reordering."""
    result = hashlib.sha256()
    for name, parameter in model.named_parameters():
        result.update(name.encode())
        for key, value in sorted(optimizer.state[parameter].items()):
            result.update(key.encode())
            if torch.is_tensor(value):
                result.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                result.update(repr(value).encode())
    return result.hexdigest()


def restore_optimizer(model, saved, spec, factor, device):
    """Load the original two groups before splitting the new gate, retaining state.

    load_state_dict replaces group LR tensors. Rebind those tensors *before*
    capture, never after. Both token branches use the same three-group layout.
    The inherited paper injection gate remains in the full-LR zero-decay group.
    """
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = adapter.optimizer_for(model, lr, spec["weight_decay"])
    optimizer.load_state_dict(saved)
    before = optimizer_digest(model, optimizer)
    for group in optimizer.param_groups:
        group["lr"] = lr
    gate_params = [p for n, p in model.named_parameters() if n in GATE_NAMES]
    gate_lr = None
    if gate_params:
        if len(gate_params) != 2:
            raise ValueError("Both update gate parameters must be present")
        group = optimizer.param_groups[1]
        if group["weight_decay"] != 0.0:
            raise ValueError("The update gate must retain zero decay")
        ids = {id(p) for p in gate_params}
        if not ids <= {id(p) for p in group["params"]}:
            raise ValueError("Parent optimizer ordering is not the original gate recipe")
        group["params"] = [p for p in group["params"] if id(p) not in ids]
        gate_lr = torch.tensor(spec["lr"] * factor, device=device)
        optimizer.add_param_group(
            {
                **{k: v for k, v in group.items() if k != "params"},
                "params": gate_params,
                "lr": gate_lr,
            }
        )
    elif factor != 1.0:
        raise ValueError("Only a learned update gate can have a separate LR")
    if optimizer_digest(model, optimizer) != before:
        raise AssertionError("Regrouping changed inherited Adam moments or step counters")
    flattened = [n for group in parameter_groups(model, optimizer) for n in group]
    if (
        len(flattened) != len(set(flattened))
        or set(flattened) != dict(model.named_parameters()).keys()
    ):
        raise AssertionError("Optimizer omitted or duplicated a trainable parameter")
    return optimizer, lr, gate_lr


def load_parent(config, name, metadata):
    parent = config["parents"][config["arms"][name]["parent"]]
    root = Path(parent["root"])
    for filename, expected in parent["files"].items():
        if digest(root / filename) != expected:
            raise ValueError(f"Changed source artifact: {root / filename}")
    source = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    original = json.loads((root / "config.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    if (
        not {"optimizer", "stream", "counts", "cpu_rng", "cuda_rng", "training_seconds"}
        <= source.keys()
    ):
        raise ValueError("Weights alone cannot continue the original training trajectory")
    if (
        source["step"] != config["continuation"]["start_step"]
        or source["step"] != complete["step"]
        or source["world_sha256"] != metadata["world_sha256"]
        or source["config_sha256"] != parent["files"]["config.json"]
        or source["model_sha256"] != parent["model_sha256"]
        or source["model_sha256"] != complete["model_sha256"]
    ):
        raise ValueError("Parent endpoint, configuration or data identity mismatch")
    for key, value in config["training"].items():
        if key != "steps" and source["spec"][key] != value:
            raise ValueError(f"Changed original training recipe: {key}")
    for key in ("model", "data", "dropout_seed"):
        if config[key] != original[key]:
            raise ValueError(f"Changed original {key}")
    for key in ("world_sha256", "panels_sha256"):
        if metadata[key] != config["input_origin"][key]:
            raise ValueError(f"Changed common data identity: {key}")
    return source, complete, parent


def prepare(config, name, world, metadata, device):
    source, complete, parent = load_parent(config, name, metadata)
    model, spec, arm, original_initial = adapter.initialize(config, name, metadata, device)
    model.load_state_dict(source["model"], strict=True)
    if model_digest(model) != source["model_sha256"]:
        raise AssertionError("Source model did not reload exactly")
    factor = config["arms"][name]["update_gate_lr_factor"]
    optimizer, lr, gate_lr = restore_optimizer(model, source["optimizer"], spec, factor, device)
    if any(float(s["step"]) != source["step"] for s in optimizer.state.values()):
        raise ValueError("Adam update counters do not match the source checkpoint")
    stream = MergedBatchStream(len(source["counts"]), spec["stream_seed"])
    stream.load_state_dict(source["stream"])
    counts = source["counts"].copy()
    exposure = native.exposure_summary(counts, world, metadata, source["step"], spec["batch_size"])
    if any(complete["endpoint"][key] != value for key, value in exposure.items()):
        raise ValueError("Inherited exposure counts disagree with the completed parent")
    torch.set_rng_state(source["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(source["cuda_rng"], device)
    model.train()
    return (
        model,
        spec,
        arm,
        optimizer,
        lr,
        gate_lr,
        stream,
        counts,
        source,
        complete,
        parent,
        original_initial,
    )


def train(config, name, out, smoke, config_path, base):
    if smoke:
        raise ValueError("No scientific smoke run is part of this continuation")
    out.mkdir(parents=True, exist_ok=False)
    world, panels, metadata = native.load(config["input_root"])
    device = torch.device("cuda:0")
    (
        model,
        spec,
        arm,
        optimizer,
        lr,
        gate_lr,
        stream,
        counts,
        source,
        parent_complete,
        parent,
        original_initial,
    ) = prepare(config, name, world, metadata, device)
    table = native.make_table(world, metadata, device)
    if stream.size != len(table[0]) - 1:
        raise ValueError("Source stream size does not match the unchanged training rows")
    before = optimizer_digest(model, optimizer)
    graph = native.AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    restored = {
        "model_exact": model_digest(model) == source["model_sha256"],
        "all_adam_state_exact": optimizer_digest(model, optimizer) == before,
        "cpu_rng_exact": torch.equal(torch.get_rng_state(), source["cpu_rng"]),
        "cuda_rng_exact": torch.equal(torch.cuda.get_rng_state(device), source["cuda_rng"]),
    }
    if not all(restored.values()):
        raise AssertionError(f"CUDA graph preparation altered the resume state: {restored}")
    cost = adapter.accounting(model, spec, arm, metadata)
    inherited_seconds = source["training_seconds"]
    start_step = step = source["step"]
    runtime_path = base / "runtime" / f"{name}-train.json"
    if runtime_path.exists():
        native.write_json(out / "container-runtime.json", json.loads(runtime_path.read_text()))
    native.write_json(
        out / "resume.json",
        {
            "parent": parent,
            "source_step": start_step,
            "model_sha256": source["model_sha256"],
            "optimizer_state_sha256": before,
            "capture_restoration": restored,
            "parameter_groups": parameter_groups(model, optimizer),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "source_training_seconds": inherited_seconds,
            "independent_endpoint_audit": "not_requested",
            "utc": utc(),
        },
    )
    native.write_json(
        out / "run.json",
        {
            "batch": config["batch"],
            "spec": spec,
            "arm": name,
            "architecture": native.architecture_flags(arm),
            "model": config["model"],
            "config_sha256": digest(config_path),
            "world_sha256": metadata["world_sha256"],
            "panels_sha256": metadata["panels_sha256"],
            "initial_model_sha256": original_initial,
            "continuation_initial_model_sha256": source["model_sha256"],
            "continuation_start_step": start_step,
            "parent_full_checkpoint_sha256": parent["files"]["latest.pt"],
            "parent_initial_model_sha256": parent_complete["initial_model_sha256"],
            "engineering_only": False,
            "dtype": "float32 parameters and Adam state; bf16 AMP; TF32 enabled",
            "pid": os.getpid(),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "hostname": platform.node(),
            "started_utc": utc(),
            "rng_policy": "Restore each parent's full CPU/CUDA/dropout and stream states; "
            "the two learned-gate branches share the same 300k state and capture layout.",
            **cost,
        },
    )
    # W&B startup is outside the scientific RNG trajectory.
    torch.set_rng_state(source["cpu_rng"])
    torch.cuda.set_rng_state(source["cuda_rng"], device)
    del source
    started, training_seconds = time.perf_counter(), inherited_seconds
    history = [
        {
            **parent_complete["endpoint"],
            "inherited_parent_measurement": True,
            "inherited_parent_wall_seconds": parent_complete["endpoint"]["wall_seconds"],
            "wall_seconds": 0.0,
            "continuation_updates": 0,
            "continuation_training_seconds": 0.0,
        }
    ]
    native.write_json(out / "learning.json", history)
    factor = config["arms"][name]["update_gate_lr_factor"]

    def measure(values):
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        metrics, predictions = native.measure_panels(
            model, world, panels, metadata, device, config["evaluation_batch_size"]
        )
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(device)
        ):
            raise AssertionError("Evaluation consumed training RNG")
        for panel, prediction in predictions.items():
            np.savez_compressed(out / f"predictions-{step:07d}-{panel}.npz", **prediction)
        resource_path = base / "resources" / f"{name}.json"
        resource = json.loads(resource_path.read_text()) if resource_path.exists() else {}
        record = {
            "step": step,
            "metrics": metrics,
            "training_seconds": training_seconds,
            "continuation_training_seconds": training_seconds - inherited_seconds,
            "continuation_updates": step - start_step,
            "backbone_lr": float(lr),
            "update_gate_lr": float(gate_lr) if gate_lr is not None else None,
            "wall_seconds": time.perf_counter() - started,
            "estimated_matmul_training_flops": step
            * cost["estimated_matmul_training_flops_per_step"],
            "continuation_estimated_matmul_training_flops": (step - start_step)
            * cost["estimated_matmul_training_flops_per_step"],
            **native.exposure_summary(counts, world, metadata, step, spec["batch_size"]),
            **values,
            **(
                {
                    "physical_gpu": resource["gpu"],
                    "gpu_memory_used_mib": resource["used_mib"],
                    "gpu_utilization_percent": resource["utilization"],
                }
                if resource
                else {}
            ),
        }
        history.append(record)
        native.write_json(out / "learning.json", history)
        saved = {
            "model": model.state_dict(),
            "spec": spec,
            "arm": arm,
            "step": step,
            "architecture": native.architecture_flags(arm),
            "config_sha256": digest(config_path),
            "world_sha256": metadata["world_sha256"],
            "model_sha256": model_digest(model),
            "parent_full_checkpoint_sha256": parent["files"]["latest.pt"],
        }
        torch.save(saved, out / f"checkpoint-{step:07d}.pt")
        temporary = out / "latest.tmp.pt"
        torch.save(
            {
                **saved,
                "optimizer": optimizer.state_dict(),
                "stream": stream.state_dict(),
                "counts": counts,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
                "training_seconds": training_seconds,
                "optimizer_parameter_groups": parameter_groups(model, optimizer),
            },
            temporary,
        )
        temporary.replace(out / "latest.pt")
        np.savez_compressed(out / "exposure.npz", counts=counts)

    for end in config["evaluation_nodes"][1:]:
        while step < end:
            boundary = min(end, (step // config["heartbeat_steps"] + 1) * config["heartbeat_steps"])
            torch.cuda.synchronize()
            segment = time.perf_counter()
            while step < boundary:
                size = min(128, boundary - step)
                indices = np.stack([stream.batch(spec["batch_size"])[0] for _ in range(size)])
                counts += np.bincount(indices.ravel(), minlength=len(counts) + 1)[: len(counts)]
                gpu_indices = torch.as_tensor(indices, device=device)
                for offset in range(size):
                    value = learning_rate(spec, step + offset + 1)
                    lr.fill_(value)
                    if gate_lr is not None:
                        gate_lr.fill_(value * factor)
                    graph(gpu_indices[offset])
                step += size
            torch.cuda.synchronize()
            training_seconds += time.perf_counter() - segment
            values = native.graph_values(graph, 0.0)
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError(f"Nonfinite continuation loss/gradient: {values}")
            status = {
                "state": "running",
                "arm": name,
                "step": step,
                "budget": spec["steps"],
                "continuation_updates": step - start_step,
                "training_seconds": training_seconds,
                "continuation_training_seconds": training_seconds - inherited_seconds,
                "backbone_lr": float(lr),
                "update_gate_lr": float(gate_lr) if gate_lr is not None else None,
                "updated_utc": utc(),
                **values,
            }
            native.write_json(out / "status.json", status)
            with (out / "progress.jsonl").open("a") as progress:
                progress.write(json.dumps(status, allow_nan=False) + "\n")
            print(json.dumps(status, allow_nan=False), flush=True)
        measure(values)
    checkpoint = out / f"checkpoint-{step:07d}.pt"
    final = {
        "state": "complete",
        "arm": name,
        "step": step,
        "budget": spec["steps"],
        "engineering_only": False,
        "endpoint": history[-1],
        "initial_model_sha256": original_initial,
        "model_sha256": model_digest(model),
        "final_checkpoint_sha256": digest(checkpoint),
        "config_sha256": digest(config_path),
        "world_sha256": metadata["world_sha256"],
        "parameters": cost["parameters"],
        "training_seconds": training_seconds,
        "continuation_training_seconds": training_seconds - inherited_seconds,
        "continuation_updates": step - start_step,
        "wall_seconds": time.perf_counter() - started,
        "updated_utc": utc(),
        "independent_reload": "not_requested",
    }
    native.write_json(out / "complete.json", final)
    native.write_json(out / "status.json", final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "grid"), required=True)
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
        with adapter.installed():
            if args.stage == "grid":
                depth.endpoint(config, args.arm, args.output, args.config)
            else:
                original = native.train
                try:
                    native.train = train
                    depth.tracked_train(config, args.arm, args.output, args.config)
                finally:
                    native.train = original
    except BaseException as error:
        native.write_json(
            args.output / "failures" / f"{args.arm}-{args.stage}.json",
            {
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "utc": utc(),
            },
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
