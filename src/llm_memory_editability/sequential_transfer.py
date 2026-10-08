"""Sequential acquisition of source-grounded facts in a complete Transformer.

All histories train from random initialization. The joint history is an exact
permutation of the sequential-composition training-example multiset. Evaluation
questions, including every chain containing a B fact, never enter that stream.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from .grok_depth import utc
from .realworld_composition import (
    aggregate,
    construct,
    evaluate,
    model_digest,
    optimizer_for,
    update,
)
from .realworld_composition_data import atomic_question, encode_example, sha256, write_json

HISTORIES = (
    "sequential_composition",
    "sequential_atomic",
    "joint",
    "sequential_composition_sham",
)


def validate_data(data):
    """Reject leakage or a missing evaluation role before model allocation."""
    atoms = {row["id"]: row for row in data["atoms"]}
    if len(atoms) != len(data["atoms"]):
        raise ValueError("Duplicate atomic IDs")
    a = {key for key, row in atoms.items() if row["subset"] == "A"}
    b = {key for key, row in atoms.items() if row["subset"] == "B"}
    if not a or not b or a | b != set(atoms):
        raise ValueError("Every atom must belong to a nonempty A or B partition")
    for key, expected in (("stage_a_atom_ids", a), ("stage_b_atom_ids", b)):
        if key in data and set(data[key]) != expected:
            raise ValueError(f"{key} differs from atomic subset fields")
    train = data["train_compositions"]
    evaluation = data["evaluation_compositions"]
    train_ids = {row["id"] for row in train}
    evaluation_ids = {row["id"] for row in evaluation}
    if not train or train_ids & evaluation_ids:
        raise ValueError("Training chains must be nonempty and held-out IDs disjoint")
    if len(train_ids) != len(train) or len(evaluation_ids) != len(evaluation):
        raise ValueError("Duplicate composition IDs")
    if (train_ids | evaluation_ids) & set(atoms):
        raise ValueError("Atom and composition IDs must be globally unique")
    train_chains = {tuple(row["atom_ids"]) for row in train}
    if any(not set(row["atom_ids"]) <= a for row in train):
        raise ValueError("A training chain includes a held-out B fact")
    roles = set()
    for row in evaluation:
        if len(row["atom_ids"]) != 2 or not set(row["atom_ids"]) <= set(atoms):
            raise ValueError("Every chain needs exactly two known atomic IDs")
        role = "".join(atoms[key]["subset"] for key in row["atom_ids"])
        if row.get("sequential_role") != role:
            raise ValueError("Evaluation role does not match its constituent facts")
        if tuple(row["atom_ids"]) in train_chains:
            raise ValueError("Evaluation chain identity occurs in training")
        roles.add(role)
    if roles != {"AA", "BA", "AB", "BB"}:
        raise ValueError("All four fixed evaluation roles must be nonempty")
    return atoms


def _sample_cycles(ids, count, rng):
    """Balanced random epochs, preserving every requested sample slot."""
    ids = np.asarray(ids, dtype=np.int32)
    if count < 0 or not len(ids):
        raise ValueError("Sampling requires a nonempty pool and nonnegative count")
    if not count:
        return np.empty(0, dtype=np.int32)
    chunks = [rng.permutation(ids) for _ in range(math.ceil(count / len(ids)))]
    return np.concatenate(chunks)[:count]


def training_plan(data, spec):
    """Materialize deterministic example IDs; histories differ only as declared."""
    validate_data(data)
    history = spec["history"]
    if history not in HISTORIES:
        raise ValueError(f"Unknown training history: {history}")
    records = data["atoms"] + data["train_compositions"]
    index = {row["id"]: i for i, row in enumerate(records)}
    a = [index[row["id"]] for row in data["atoms"] if row["subset"] == "A"]
    b = [index[row["id"]] for row in data["atoms"] if row["subset"] == "B"]
    c = [index[row["id"]] for row in data["train_compositions"]]
    batch = int(spec["batch_size"])
    sa, sb = int(spec["stage_a_steps"]), int(spec["stage_b_steps"])
    if batch < 2 or sa < 1 or sb < 1:
        raise ValueError("Need batch>=2 and positive budgets in both stages")
    rng = np.random.default_rng(spec["sampling_seed"])
    # Exactly half (rounded down) of A slots contain composition examples;
    # exactly half of B slots contain genuinely new atomic examples.
    ca = int(spec.get("stage_a_composition_per_batch", batch // 2))
    nb = int(spec.get("stage_b_new_per_batch", batch // 2))
    if not (0 < ca < batch and 0 < nb < batch):
        raise ValueError("Both stage mixtures need positive constituent counts")
    first = np.concatenate(
        (
            _sample_cycles(a, sa * (batch - ca), rng).reshape(sa, batch - ca),
            _sample_cycles(c, sa * ca, rng).reshape(sa, ca),
        ),
        axis=1,
    )
    second = np.concatenate(
        (
            _sample_cycles(a, sb * (batch - nb), rng).reshape(sb, batch - nb),
            _sample_cycles(b, sb * nb, rng).reshape(sb, nb),
        ),
        axis=1,
    )
    # Seeded replacement streams are generated for every history, so selecting
    # a history cannot affect either the common A exposure or B sampling.
    atomic_replacement = _sample_cycles(a, sa * ca, rng).reshape(sa, ca)
    sham_replacement = _sample_cycles(a, sb * nb, rng).reshape(sb, nb)
    if history == "sequential_atomic":
        first[:, batch - ca :] = atomic_replacement
    elif history == "sequential_composition_sham":
        second[:, batch - nb :] = sham_replacement
    plan = np.concatenate((first, second), axis=0)
    if history == "joint":
        plan = rng.permutation(plan.reshape(-1)).reshape(plan.shape)
    return records, plan


def plan_manifest(records, plan, spec):
    counts = np.bincount(plan.reshape(-1), minlength=len(records))
    return {
        "history": spec["history"],
        "shape": list(plan.shape),
        "plan_sha256": hashlib.sha256(plan.tobytes()).hexdigest(),
        "multiset_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "examples": int(counts.sum()),
        "supervised_tokens": sum(
            int(n) * len(row["encoded"]["target"]) for row, n in zip(records, counts, strict=True)
        ),
        "input_tokens": sum(
            int(n) * len(row["encoded"]["input"]) for row, n in zip(records, counts, strict=True)
        ),
        "counts": {row["id"]: int(n) for row, n in zip(records, counts, strict=True)},
    }


def learning_rate(spec, step):
    """Same staged cosine by global update index for *all* histories."""
    boundary = spec["stage_a_steps"]
    length = boundary if step <= boundary else spec["stage_b_steps"]
    local = step if step <= boundary else step - boundary
    warmup = min(spec.get("warmup_steps", 200), length)
    peak = spec["learning_rate"]
    if warmup and local <= warmup:
        return peak * local / warmup
    progress = (local - warmup) / max(1, length - warmup)
    floor = spec.get("minimum_lr_fraction", 0.1)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def evaluation_groups(data, full):
    atoms = {row["id"]: row for row in data["atoms"]}
    chains = {row["id"]: row for row in data["evaluation_compositions"]}
    panels = data.get("panels", {})
    selected = (
        list(chains.values())
        if full
        else [chains[key] for role in ("AA", "BA", "AB", "BB") for key in panels.get(role, [])]
    )
    if not selected:
        selected = list(chains.values())
    needed = {key for row in selected for key in row["atom_ids"]}
    atom_ids = (
        set(atoms)
        if full
        else needed | set(panels.get("atomic_A", [])) | set(panels.get("atomic_B", []))
    )
    groups = {
        "atomic_A": [row for key, row in atoms.items() if key in atom_ids and row["subset"] == "A"],
        "atomic_B": [row for key, row in atoms.items() if key in atom_ids and row["subset"] == "B"],
        "train_composition": data["train_compositions"]
        if full
        else [
            row
            for row in data["train_compositions"]
            if row["id"] in set(panels.get("train_composition", []))
        ],
    }
    for role in ("AA", "BA", "AB", "BB"):
        groups[role] = [row for row in selected if row["sequential_role"] == role]
    return groups


def evaluate_transfer(model, data, tokenizer, device, *, full=False, autonomous=False, batch=32):
    groups = evaluation_groups(data, full)
    metrics, predictions = {}, {}
    for name, records in groups.items():
        metrics[name], predictions[name] = (
            evaluate(model, records, tokenizer, device, batch_size=batch)
            if records
            else (aggregate([]), [])
        )
    lookup = {row["id"]: row for name in ("atomic_A", "atomic_B") for row in predictions[name]}
    for role in ("AA", "BA", "AB", "BB"):
        selected = groups[role]
        correct = {
            row["id"] for row in selected if all(lookup[key]["alias_em"] for key in row["atom_ids"])
        }
        key = role + "_necessary_atoms_correct"
        metrics[key] = aggregate([row for row in predictions[role] if row["id"] in correct])
        metrics[key]["coverage"] = len(correct) / len(selected) if selected else None
        if not autonomous:
            continue
        calls, invalid = [], []
        for row in selected:
            bridge = lookup[row["atom_ids"][0]]["prediction"]
            question = atomic_question(bridge, row["edges"][1][1])
            try:
                encoded = encode_example(tokenizer, question, row["answer"])
            except ValueError:
                invalid.append(row)
                continue
            calls.append({**row, "encoded": encoded, "question": question, "bridge": bridge})
        _, raw = evaluate(model, calls, tokenizer, device, batch_size=batch) if calls else ({}, [])
        for row, value in zip(calls, raw, strict=True):
            value.update(bridge_prediction=row["bridge"], second_question=row["question"])
        raw.extend(
            {
                "id": row["id"],
                "canonical_em": 0.0,
                "alias_em": 0.0,
                "unambiguous_alias_em": 0.0,
                "f1": 0.0,
                "nll": None,
                "eos": False,
                "truncated": True,
                "invalid_generated_bridge_length": True,
            }
            for row in invalid
        )
        predictions[role + "_autonomous"] = raw
        metrics[role + "_autonomous"] = aggregate(raw)
        metrics[role + "_autonomous"]["invalid_bridge_prompts"] = len(invalid)
    return metrics, predictions


def _setup(spec, device):
    torch.set_num_threads(spec.get("cpu_threads", 4))
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    data_path = Path(spec["data_file"])
    if sha256(data_path) != spec["data_sha256"]:
        raise ValueError("Dataset hash differs from frozen spec")
    data = json.loads(data_path.read_text())
    validate_data(data)
    tokenizer = GPT2TokenizerFast.from_pretrained(spec["tokenizer"], local_files_only=True)
    if len(tokenizer) != spec["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary and model vocabulary differ")
    model = construct(spec["model"], spec, device)
    return data, tokenizer, model, device


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(path, model, optimizer, step, counts, counters, history, manifest):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "counts": counts,
            "counters": counters,
            "history": history,
            "plan_sha256": manifest["plan_sha256"],
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None,
        },
        temporary,
    )
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["plan_sha256"] != manifest["plan_sha256"]:
        raise ValueError("Resume would change the frozen sampling plan")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["cpu_rng"])
    if checkpoint["cuda_rng"] is not None:
        torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    return checkpoint


def run(spec, out, device):
    """Train one frozen history; an independent audit finalizes completion."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        if json.loads((out / "run.json").read_text())["spec"] != spec:
            raise ValueError("Completed run has a different experiment spec")
        return json.loads((out / "complete.json").read_text())
    data, tokenizer, model, device = _setup(spec, device)
    records, plan = training_plan(data, spec)
    manifest = plan_manifest(records, plan, spec)
    optimizer = optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    counts = np.zeros(len(records), dtype=np.int64)
    counters = {
        key: 0
        for key in (
            "examples",
            "supervised_tokens",
            "effective_input_tokens",
            "executed_input_tokens",
            "estimated_matmul_training_flops",
        )
    }
    counters.update(training_seconds=0.0, wall_seconds=0.0)
    history, initial_step = [], 0
    if (out / "latest.pt").exists():
        previous = json.loads((out / "run.json").read_text())
        if previous["spec"] != spec:
            raise ValueError("Refusing to resume under a changed experiment spec")
        state = restore_checkpoint(out / "latest.pt", model, optimizer, manifest)
        initial_step, counts, counters, history = (
            state["step"],
            state["counts"],
            state["counters"],
            state["history"],
        )
        metadata = {**previous, "pid": os.getpid(), "resumed_from_step": initial_step}
        del state
    else:
        torch.manual_seed(spec.get("dropout_seed", spec["initialization"] + 1))
        metadata = {
            "spec": spec,
            "phase": spec.get("phase", "development"),
            "model": spec["model"],
            "pid": os.getpid(),
            "gpu": device.index,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "initial_model_sha256": model_digest(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "world_sha256": spec["data_sha256"],
            "dtype": "BF16 AMP; FP32 parameters/Adam",
            "job_type": "sequential-transfer",
            "tracking_group": spec.get("tracking_group"),
            "created_utc": utc(),
            "sampling_manifest": {key: value for key, value in manifest.items() if key != "counts"},
        }
        write_json(out / "sampling-plan.json", manifest)
    write_json(out / "run.json", metadata)
    write_json(out / "learning.json", history)
    nodes = set(spec.get("evaluation_nodes", [])) | {0, spec["stage_a_steps"], len(plan)}
    wall_offset, started = counters["wall_seconds"], time.perf_counter()
    loss = history[-1].get("loss") if history else None
    latest_step = initial_step
    for step in range(initial_step, len(plan) + 1):
        if step > initial_step:
            lr = learning_rate(spec, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            ids = plan[step - 1]
            selected = [records[i] for i in ids]
            _sync(device)
            began = time.perf_counter()
            result = update(model, optimizer, selected, spec, device, tokenizer.eos_token_id)
            _sync(device)
            counters["training_seconds"] += time.perf_counter() - began
            loss = result["loss"]
            np.add.at(counts, ids, 1)
            counters["examples"] += len(ids)
            counters["supervised_tokens"] += sum(len(row["encoded"]["target"]) for row in selected)
            for key in (
                "effective_input_tokens",
                "executed_input_tokens",
                "estimated_matmul_training_flops",
            ):
                counters[key] += result[key]
        counters["wall_seconds"] = wall_offset + time.perf_counter() - started
        if step % spec.get("status_interval", 50) == 0:
            write_json(
                out / "status.json",
                {
                    "state": "training",
                    "step": step,
                    "target_step": len(plan),
                    "loss": loss,
                    "pid": os.getpid(),
                    **counters,
                    "updated_utc": utc(),
                },
            )
        if step in nodes and not any(row["step"] == step for row in history):
            endpoint = step in {spec["stage_a_steps"], len(plan)}
            metrics, raw = evaluate_transfer(
                model,
                data,
                tokenizer,
                device,
                full=endpoint,
                autonomous=endpoint,
                batch=spec.get("evaluation_batch_size", 32),
            )
            write_json(out / f"predictions-{step:07d}.json", raw)
            counters["wall_seconds"] = wall_offset + time.perf_counter() - started
            exposure = {}
            for name, subset in (("A", "A"), ("B", "B"), ("composition", None)):
                indices = [
                    i
                    for i, row in enumerate(records)
                    if (row.get("subset") == subset if subset else i >= len(data["atoms"]))
                ]
                exposure[name] = float(counts[indices].mean()) if indices else 0.0
            history.append(
                {
                    "step": step,
                    "stage": "A" if step <= spec["stage_a_steps"] else "B",
                    "condition": spec["history"],
                    "metrics": metrics,
                    "loss": loss,
                    "learning_rate": learning_rate(spec, max(step, 1)),
                    "exposures": exposure,
                    "atomic_A_epochs": exposure["A"],
                    "atomic_B_epochs": exposure["B"],
                    "atomic_epochs": float(counts[: len(data["atoms"])].mean()),
                    "composition_epochs": exposure["composition"],
                    **counters,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                    "created_utc": utc(),
                }
            )
            write_json(out / "learning.json", history)
            if step == spec["stage_a_steps"]:
                save_checkpoint(
                    out / "stage-a.pt", model, optimizer, step, counts, counters, history, manifest
                )
            save_checkpoint(
                out / "latest.pt", model, optimizer, step, counts, counters, history, manifest
            )
            latest_step = step
            print(json.dumps({"step": step, "loss": loss, "metrics": metrics}), flush=True)
        elif step > initial_step and step % spec.get("checkpoint_interval", 500) == 0:
            save_checkpoint(
                out / "latest.pt", model, optimizer, step, counts, counters, history, manifest
            )
            latest_step = step
    if latest_step != len(plan):
        save_checkpoint(
            out / "latest.pt", model, optimizer, len(plan), counts, counters, history, manifest
        )
    final = next(row for row in history if row["step"] == len(plan))
    write_json(out / "endpoint.json", final["metrics"])
    result = {
        "state": "trained_pending_independent_reload",
        "step": len(plan),
        "pid": os.getpid(),
        "model_sha256": model_digest(model),
        "checkpoint_sha256": sha256(out / "latest.pt"),
        "plan_sha256": manifest["plan_sha256"],
        **counters,
        "completed_utc": utc(),
    }
    write_json(out / "trained.json", result)
    write_json(out / "status.json", result)
    return result


def audit(out, device):
    """Reload in a separate process and reproduce the full generated endpoint."""
    out = Path(out)
    metadata = json.loads((out / "run.json").read_text())
    trained = json.loads((out / "trained.json").read_text())
    if trained["pid"] == os.getpid():
        raise ValueError("Endpoint audit requires a different process")
    if sha256(out / "latest.pt") != trained["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from recorded endpoint")
    spec = metadata["spec"]
    data, tokenizer, model, device = _setup(spec, device)
    state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if state["step"] != trained["step"] or model_digest(model) != trained["model_sha256"]:
        raise ValueError("Loaded model or update count differs")
    records, plan = training_plan(data, spec)
    manifest = plan_manifest(records, plan, spec)
    if manifest["plan_sha256"] != trained["plan_sha256"]:
        raise ValueError("Sampling plan changed")
    expected_counts = np.bincount(plan.reshape(-1), minlength=len(records))
    if not np.array_equal(state["counts"], expected_counts):
        raise ValueError("Observed example exposures differ from frozen plan")
    metrics, raw = evaluate_transfer(
        model,
        data,
        tokenizer,
        device,
        full=True,
        autonomous=True,
        batch=spec.get("evaluation_batch_size", 32),
    )
    expected_raw = json.loads((out / f"predictions-{trained['step']:07d}.json").read_text())
    if raw != expected_raw or metrics != json.loads((out / "endpoint.json").read_text()):
        raise AssertionError("Independent endpoint generations/scores failed exact reproduction")
    result = {
        "passed": True,
        "pid": os.getpid(),
        "source_training_pid": trained["pid"],
        "model_sha256": trained["model_sha256"],
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "predictions_recomputed": sum(map(len, raw.values())),
        "exposures_exact": True,
        "completed_utc": utc(),
    }
    write_json(out / "audit.json", result)
    write_json(
        out / "complete.json", {**trained, "state": "complete", "independently_reloaded": True}
    )
    write_json(
        out / "status.json",
        {"state": "complete", "step": trained["step"], "independently_reloaded": True},
    )
    return result


def preflight(spec, out, device):
    """Actual updates, fresh-model reload and exact optimizer/RNG continuation."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data, tokenizer, model, device = _setup(spec, device)
    records, plan = training_plan(data, spec)
    optimizer = optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    times, losses = [], []
    for ids in plan[:8]:
        _sync(device)
        started = time.perf_counter()
        result = update(
            model, optimizer, [records[i] for i in ids], spec, device, tokenizer.eos_token_id
        )
        _sync(device)
        times.append(time.perf_counter() - started)
        losses.append(result["loss"])
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    # The next update uses the exact same record order, Adam moments and RNG.
    next_records = [records[i] for i in plan[min(8, len(plan) - 1)]]
    update(model, optimizer, next_records, spec, device, tokenizer.eos_token_id)
    expected = model_digest(model)
    restored = construct(spec["model"], spec, device)
    restored.load_state_dict(saved_model)
    restored_optimizer = optimizer_for(restored, spec["learning_rate"], spec["weight_decay"])
    restored_optimizer.load_state_dict(saved_optimizer)
    torch.set_rng_state(cpu_rng)
    if gpu_rng is not None:
        torch.cuda.set_rng_state(gpu_rng)
    update(restored, restored_optimizer, next_records, spec, device, tokenizer.eos_token_id)
    if model_digest(restored) != expected:
        raise AssertionError("Preflight optimizer/RNG continuation failed exact reproduction")
    generation_metrics, predictions = evaluate(
        restored, next_records[:4], tokenizer, device, batch_size=4
    )
    result = {
        "passed": True,
        "updates": len(times),
        "continuation_check_updates": 2,
        "optimizer_rng_restore_exact": True,
        "generated_predictions": predictions,
        "generation_metrics": generation_metrics,
        "losses": losses,
        "seconds_per_update_after_warmup": sum(times[2:]) / max(1, len(times) - 2),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "parameters": sum(p.numel() for p in model.parameters()),
        "created_utc": utc(),
    }
    write_json(out / "preflight.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.audit:
        if not args.out:
            parser.error("--audit requires --out")
        result = audit(args.out, args.device)
    else:
        if not args.config or not args.run:
            parser.error("Training/preflight requires --config and --run")
        config = json.loads(args.config.read_text())
        selected = next(row for row in config["runs"] if row["name"] == args.run)
        spec = {**config.get("defaults", {}), **selected}
        out = args.out or Path(config["results_root"]) / "runs" / args.run
        result = (preflight if args.preflight else run)(spec, out, args.device)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
