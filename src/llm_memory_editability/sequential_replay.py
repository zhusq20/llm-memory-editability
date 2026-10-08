"""Continue a fixed stage-A model with atomic or full old-data replay.

The new-fact stream is copied exactly from the parent's original stage B.
Half of every batch is old data: either A atomics or the original alternating
atomic/composition stream. Both arms intentionally reset AdamW and its schedule.
All counters describe the newly executed continuation, never inherited A work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import sequential_transfer as original
from .grok_depth import utc
from .realworld_composition import model_digest, optimizer_for, update
from .realworld_composition_data import sha256, write_json

ARMS = ("atom_replay", "full_replay")
COUNTERS = (
    "examples",
    "supervised_tokens",
    "effective_input_tokens",
    "executed_input_tokens",
    "estimated_matmul_training_flops",
)


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def continuation_plan(data, parent_spec, spec):
    """Derive both suffixes from the original materialized plan without resampling."""
    if spec["replay_arm"] not in ARMS:
        raise ValueError("Unknown replay arm")
    if parent_spec["history"] != "sequential_composition":
        raise ValueError("Continuation requires an original composition-learning parent")
    if spec["original_stage_a_steps"] != parent_spec["stage_a_steps"]:
        raise ValueError("The inherited stage-A boundary cannot change")
    batch = parent_spec["batch_size"]
    steps = spec["stage_b_steps"]
    if batch % 2 or steps < 2 or steps % 2 or steps > parent_spec["stage_b_steps"]:
        raise ValueError("Use an even batch and even positive suffix within the original B stream")
    if spec.get("batch_size", batch) != batch:
        raise ValueError("Continuation must preserve the parent batch size")
    half = batch // 2
    if (
        parent_spec.get("stage_a_composition_per_batch", half) != half
        or parent_spec.get("stage_b_new_per_batch", half) != half
    ):
        raise ValueError("The original plan must have the specified 50:50 mixtures")
    records, parent_plan = original.training_plan(data, parent_spec)
    boundary = parent_spec["stage_a_steps"]
    first = parent_plan[:boundary]
    slots = steps * half
    old = first.reshape(-1) if spec["replay_arm"] == "full_replay" else first[:, :half].reshape(-1)
    if len(old) < slots:
        raise ValueError("The inherited stage-A stream is too short for this replay budget")
    replay = old[:slots].reshape(steps, half).copy()
    new = parent_plan[boundary : boundary + steps, -half:].copy()
    if any(records[index].get("subset") != "B" for index in new.reshape(-1)):
        raise AssertionError("The copied new stream contains a non-B example")
    if any(records[index].get("subset") == "B" for index in replay.reshape(-1)):
        raise AssertionError("Old replay contains B knowledge")
    if spec["replay_arm"] == "atom_replay":
        if any(records[index].get("subset") != "A" for index in replay.reshape(-1)):
            raise AssertionError("Atomic replay contains a composition")
    else:
        atomic_slots = sum(records[index].get("subset") == "A" for index in replay.reshape(-1))
        if atomic_slots != slots // 2:
            raise AssertionError("Full replay does not preserve the original 50:50 old mixture")
    plan = np.concatenate((replay, new), axis=1)
    manifest = original.plan_manifest(records, plan, {"history": spec["replay_arm"]})
    manifest.update(
        new_stream_sha256=array_hash(new),
        replay_stream_sha256=array_hash(replay),
        parent_plan_sha256=array_hash(parent_plan),
        stage_a_prefix_sha256=array_hash(first),
        stage_b_steps=steps,
        old_per_batch=half,
        new_per_batch=half,
        replay_source="original A plan flattened prefix"
        if spec["replay_arm"] == "full_replay"
        else "original A plan atomic columns flattened prefix",
        inherited_examples=int(first.size),
    )
    return records, plan, manifest, parent_plan


def learning_rate(spec, local_step):
    steps = spec["stage_b_steps"]
    if not 0 <= local_step <= steps:
        raise ValueError("Learning-rate step must be a local continuation update")
    warmup = spec.get("warmup_steps", max(1, steps // 100))
    if not 1 <= warmup < steps:
        raise ValueError("Warmup must be positive and shorter than the continuation")
    peak = spec.get("learning_rate", 3e-4)
    if local_step <= warmup:
        return peak * local_step / warmup
    fraction = (local_step - warmup) / (steps - warmup)
    floor = spec.get("minimum_lr_fraction", 0.1)
    return peak * (floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2)


def fresh_optimizer(model, spec):
    betas = tuple(spec.get("optimizer_betas", [0.9, 0.95]))
    if betas != (0.9, 0.95):
        raise ValueError("This recipe fixes fresh AdamW betas to (0.9, 0.95)")
    optimizer = optimizer_for(model, spec.get("learning_rate", 3e-4), spec.get("weight_decay", 0.1))
    optimizer.defaults["betas"] = betas
    for group in optimizer.param_groups:
        group["betas"] = betas
    if optimizer.state:
        raise AssertionError("A new continuation cannot inherit Adam moments")
    return optimizer


def _load_source(spec, device):
    directory, checkpoint_path = Path(spec["parent_run_dir"]), Path(spec["parent_checkpoint"])
    if checkpoint_path.resolve() != (directory / "stage-a.pt").resolve():
        raise ValueError("Use the declared parent's saved stage-a.pt")
    if sha256(checkpoint_path) != spec["parent_checkpoint_sha256"]:
        raise ValueError("Parent stage-A checkpoint hash changed")
    run_path = directory / "run.json"
    expected_run_hash = spec.get("parent_run_sha256", spec.get("parent_run_file_sha256"))
    if expected_run_hash is None or sha256(run_path) != expected_run_hash:
        raise ValueError("Parent run metadata hash missing or changed")
    parent_metadata = json.loads(run_path.read_text())
    parent_spec = parent_metadata["spec"]
    if not (directory / "audit.json").exists() or not json.loads(
        (directory / "audit.json").read_text()
    ).get("passed"):
        raise ValueError("The source run must have passed its independent audit")
    for key in ("model", "data_sha256", "initialization", "unique_layers", "repeats"):
        if key in spec and spec[key] != parent_spec[key]:
            raise ValueError(f"Continuation cannot change inherited {key}")
    for key in ("data_file", "tokenizer"):
        if key in spec and Path(spec[key]).resolve() != Path(parent_spec[key]).resolve():
            raise ValueError(f"Continuation cannot change inherited {key}")
    data, tokenizer, model, device = original._setup(parent_spec, device)
    records, plan, manifest, parent_plan = continuation_plan(data, parent_spec, spec)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent_plan_record = json.loads((directory / "sampling-plan.json").read_text())
    if not (
        checkpoint["plan_sha256"]
        == parent_plan_record["plan_sha256"]
        == manifest["parent_plan_sha256"]
    ):
        raise ValueError("Parent checkpoint, metadata and regenerated sampling plan disagree")
    boundary = parent_spec["stage_a_steps"]
    if checkpoint["step"] != boundary:
        raise ValueError("Source checkpoint is not at the original A/B boundary")
    expected_counts = np.bincount(parent_plan[:boundary].reshape(-1), minlength=len(records))
    if not np.array_equal(checkpoint["counts"], expected_counts):
        raise ValueError("Inherited stage-A exposures differ from the frozen prefix")
    if checkpoint["counters"]["examples"] != int(expected_counts.sum()):
        raise ValueError("Inherited stage-A example counter is inconsistent")
    model.load_state_dict(checkpoint["model"])
    torch.set_rng_state(checkpoint["cpu_rng"])
    if device.type == "cuda":
        if checkpoint["cuda_rng"] is None:
            raise ValueError("A GPU continuation requires the saved parent CUDA RNG")
        torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    inherited = {
        "step": boundary,
        "model_sha256": model_digest(model),
        "counters": checkpoint["counters"],
        "counts": checkpoint["counts"].tolist(),
        "cpu_rng_sha256": array_hash(checkpoint["cpu_rng"].numpy()),
        "cuda_rng_sha256": array_hash(checkpoint["cuda_rng"].numpy())
        if checkpoint["cuda_rng"] is not None
        else None,
    }
    return data, tokenizer, model, device, parent_spec, records, plan, manifest, inherited


def _evaluate(model, data, tokenizer, device, spec, *, full):
    return original.evaluate_transfer(
        model,
        data,
        tokenizer,
        device,
        full=full,
        autonomous=full,
        batch=spec.get("evaluation_batch_size", 64),
    )


def _nodes(spec):
    nodes = sorted(set(spec.get("evaluation_nodes", [])) | {0, spec["stage_b_steps"]})
    if nodes[0] < 0 or nodes[-1] != spec["stage_b_steps"]:
        raise ValueError("Evaluation nodes must use local B steps within the new budget")
    return set(nodes)


def run(spec, out, device="cuda:0"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        if json.loads((out / "run.json").read_text())["spec"] != spec:
            raise ValueError("Completed continuation has a different spec")
        return json.loads((out / "complete.json").read_text())
    nodes = _nodes(spec)
    loaded = _load_source(spec, device)
    data, tokenizer, model, device, parent_spec, records, plan, manifest, inherited = loaded
    boundary, budget = inherited["step"], len(plan)
    optimizer = fresh_optimizer(model, spec)
    training_spec = {**parent_spec, **spec}
    training_spec["microbatch_size"] = spec.get("microbatch_size", parent_spec["microbatch_size"])
    counts = np.zeros(len(records), dtype=np.int64)
    counters = {key: 0 for key in COUNTERS}
    counters.update(training_seconds=0.0, wall_seconds=0.0)
    history, initial_local = [], 0
    if (out / "latest.pt").exists():
        previous = json.loads((out / "run.json").read_text())
        if previous["spec"] != spec or previous["parent_model_sha256"] != inherited["model_sha256"]:
            raise ValueError("Resume would change the continuation or its parent")
        state = original.restore_checkpoint(out / "latest.pt", model, optimizer, manifest)
        initial_local = state["step"] - boundary
        if not 0 <= initial_local <= budget:
            raise ValueError("Resume checkpoint is outside the continuation budget")
        counts, counters, history = state["counts"], state["counters"], state["history"]
        metadata = {**previous, "pid": os.getpid(), "resumed_from_step": state["step"]}
    else:
        if (out / "run.json").exists():
            raise FileExistsError("An existing attempt has no recoverable checkpoint")
        metadata = {
            "spec": spec,
            "parent_spec": parent_spec,
            "model": parent_spec["model"],
            "phase": spec.get("phase", "development"),
            "replay_arm": spec["replay_arm"],
            "pid": os.getpid(),
            "gpu": device.index,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": "BF16 AMP; FP32 parameters/AdamW",
            "job_type": "sequential-replay",
            "tracking_group": spec.get("tracking_group"),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "initial_model_sha256": inherited["model_sha256"],
            "parent_model_sha256": inherited["model_sha256"],
            "parent_checkpoint_sha256": spec["parent_checkpoint_sha256"],
            "parent_stage_a_step": boundary,
            "world_sha256": parent_spec["data_sha256"],
            "optimizer": "fresh AdamW; betas=(.9,.95); clip=1; no inherited moments",
            "optimizer_reset": True,
            "weight_decay_scope": "specified decay on weights; zero on bias and layer norm",
            "learning_step_unit": "global_optimizer_update",
            "counter_scope": "new stage-B continuation only; inherited A costs recorded separately",
            "inherited_stage_a": inherited,
            "sampling_manifest": {key: value for key, value in manifest.items() if key != "counts"},
            "created_utc": utc(),
        }
        write_json(out / "sampling-plan.json", manifest)
    write_json(out / "run.json", metadata)
    write_json(out / "learning.json", history)
    started, wall_offset = time.perf_counter(), counters["wall_seconds"]
    loss = history[-1].get("loss") if history else None
    latest_local = initial_local
    for local in range(initial_local, budget + 1):
        global_step = boundary + local
        if local > initial_local:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(spec, local)
            ids = plan[local - 1]
            selected = [records[index] for index in ids]
            original._sync(device)
            began = time.perf_counter()
            result = update(
                model, optimizer, selected, training_spec, device, tokenizer.eos_token_id
            )
            original._sync(device)
            counters["training_seconds"] += time.perf_counter() - began
            loss = result["loss"]
            np.add.at(counts, ids, 1)
            counters["examples"] += len(ids)
            counters["supervised_tokens"] += sum(len(row["encoded"]["target"]) for row in selected)
            for key in COUNTERS[2:]:
                counters[key] += result[key]
        counters["wall_seconds"] = wall_offset + time.perf_counter() - started
        if local % spec.get("status_interval", 50) == 0:
            write_json(
                out / "status.json",
                {
                    "state": "training",
                    "step": global_step,
                    "branch_step": local,
                    "local_optimizer_updates": local,
                    "target_step": boundary + budget,
                    "loss": loss,
                    "pid": os.getpid(),
                    **counters,
                    "updated_utc": utc(),
                },
            )
        if local in nodes and not any(row["step"] == global_step for row in history):
            metrics, raw = _evaluate(
                model, data, tokenizer, device, spec, full=local in {0, budget}
            )
            write_json(out / f"predictions-{global_step:07d}.json", raw)
            if local == 0:
                expected_path = Path(spec["parent_run_dir"]) / f"predictions-{boundary:07d}.json"
                if raw != json.loads(expected_path.read_text()):
                    raise AssertionError("Restored stage-A predictions differ from the parent")
            exposure = {}
            for name, subset in (("A", "A"), ("B", "B"), ("composition", None)):
                indices = (
                    [index for index, row in enumerate(records) if row.get("subset") == subset]
                    if subset
                    else list(range(len(data["atoms"]), len(records)))
                )
                exposure[name] = float(counts[indices].mean()) if indices else 0.0
            counters["wall_seconds"] = wall_offset + time.perf_counter() - started
            history.append(
                {
                    "step": global_step,
                    "branch_step": local,
                    "local_optimizer_updates": local,
                    "stage": "B",
                    "stage_a_boundary": local == 0,
                    "condition": spec["replay_arm"],
                    "metrics": metrics,
                    "loss": loss,
                    "learning_rate": learning_rate(spec, local),
                    "exposures": exposure,
                    "atomic_A_epochs": exposure["A"],
                    "atomic_B_epochs": exposure["B"],
                    "composition_epochs": exposure["composition"],
                    **counters,
                    "created_utc": utc(),
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                }
            )
            write_json(out / "learning.json", history)
            original.save_checkpoint(
                out / "latest.pt",
                model,
                optimizer,
                global_step,
                counts,
                counters,
                history,
                manifest,
            )
            latest_local = local
            print(
                json.dumps(
                    {
                        "step": global_step,
                        "branch_step": local,
                        "arm": spec["replay_arm"],
                        "loss": loss,
                        "metrics": metrics,
                    }
                ),
                flush=True,
            )
        elif local > initial_local and local % spec.get("checkpoint_interval", 500) == 0:
            original.save_checkpoint(
                out / "latest.pt",
                model,
                optimizer,
                global_step,
                counts,
                counters,
                history,
                manifest,
            )
            latest_local = local
    if latest_local != budget:
        original.save_checkpoint(
            out / "latest.pt",
            model,
            optimizer,
            boundary + budget,
            counts,
            counters,
            history,
            manifest,
        )
    final = history[-1]
    write_json(out / "endpoint.json", final["metrics"])
    result = {
        "state": "trained_pending_independent_reload",
        "step": boundary + budget,
        "local_optimizer_updates": budget,
        "parent_stage_a_step": boundary,
        "pid": os.getpid(),
        "model_sha256": model_digest(model),
        "checkpoint_sha256": sha256(out / "latest.pt"),
        "plan_sha256": manifest["plan_sha256"],
        "new_stream_sha256": manifest["new_stream_sha256"],
        **counters,
        "completed_utc": utc(),
    }
    write_json(out / "trained.json", result)
    write_json(out / "status.json", result)
    return result


def audit(out, device="cuda:0"):
    """Independently reload both the inherited boundary and the new endpoint."""
    out = Path(out)
    metadata = json.loads((out / "run.json").read_text())
    spec = metadata["spec"]
    trained = json.loads((out / "trained.json").read_text())
    if trained["pid"] == os.getpid():
        raise ValueError("Endpoint audit requires a different process")
    if sha256(out / "latest.pt") != trained["checkpoint_sha256"]:
        raise ValueError("Continuation checkpoint differs from recorded endpoint")
    loaded = _load_source(spec, device)
    data, tokenizer, model, device, _parent_spec, records, plan, manifest, inherited = loaded
    if metadata["parent_model_sha256"] != inherited["model_sha256"]:
        raise ValueError("Inherited model changed")
    boundary_metrics, boundary_raw = _evaluate(model, data, tokenizer, device, spec, full=True)
    expected_boundary = json.loads((out / f"predictions-{inherited['step']:07d}.json").read_text())
    history = json.loads((out / "learning.json").read_text())
    if boundary_raw != expected_boundary or boundary_metrics != history[0]["metrics"]:
        raise AssertionError("Independent stage-A boundary scoring differs")
    state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if state["step"] != trained["step"] or model_digest(model) != trained["model_sha256"]:
        raise ValueError("Loaded continuation model or update count differs")
    if (
        state["plan_sha256"] != manifest["plan_sha256"]
        or manifest["plan_sha256"] != trained["plan_sha256"]
    ):
        raise ValueError("Continuation sampling plan changed")
    expected_counts = np.bincount(plan.reshape(-1), minlength=len(records))
    if not np.array_equal(state["counts"], expected_counts):
        raise ValueError("New-only exposures differ from the continuation plan")
    if state["counters"]["examples"] != int(plan.size) or trained["examples"] != int(plan.size):
        raise ValueError("New-only example accounting includes inherited work or misses updates")
    optimizer_steps = {int(value["step"]) for value in state["optimizer"]["state"].values()}
    if optimizer_steps != {len(plan)}:
        raise ValueError("Fresh optimizer step counters did not start from zero")
    if any(tuple(group["betas"]) != (0.9, 0.95) for group in state["optimizer"]["param_groups"]):
        raise ValueError("Optimizer recipe changed")
    if [row["branch_step"] for row in history] != sorted(_nodes(spec)):
        raise ValueError("Continuation evaluation nodes are incomplete")
    for row in history:
        if (
            row["step"] != inherited["step"] + row["branch_step"]
            or row["local_optimizer_updates"] != row["branch_step"]
            or row["examples"] != row["branch_step"] * plan.shape[1]
        ):
            raise ValueError("Continuation trajectory step/exposure accounting changed")
    metrics, raw = _evaluate(model, data, tokenizer, device, spec, full=True)
    expected_raw = json.loads((out / f"predictions-{trained['step']:07d}.json").read_text())
    if raw != expected_raw or metrics != json.loads((out / "endpoint.json").read_text()):
        raise AssertionError("Independent continuation endpoint scoring differs")
    result = {
        "passed": True,
        "pid": os.getpid(),
        "source_training_pid": trained["pid"],
        "model_sha256": trained["model_sha256"],
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "parent_boundary_recomputed": True,
        "fresh_optimizer_verified": True,
        "new_only_exposures_exact": True,
        "new_stream_sha256": manifest["new_stream_sha256"],
        "predictions_recomputed": sum(map(len, boundary_raw.values()))
        + sum(map(len, raw.values())),
        "completed_utc": utc(),
    }
    write_json(out / "audit.json", result)
    write_json(
        out / "complete.json", {**trained, "state": "complete", "independently_reloaded": True}
    )
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "step": trained["step"],
            "local_optimizer_updates": len(plan),
            "independently_reloaded": True,
        },
    )
    return result
