"""Resumable, exposure-matched training for the parametric architecture study.

The ordinary data stream, answer-plus-EOS objective and generated evaluator are
shared with the historical experiment. MoE balancing is computed over the true
effective batch using an RNG-replayed routing pass, not a sum of local products.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GPT2TokenizerFast

from . import realworld_composition as historical
from .grok_depth import utc
from .grokking_reproduction import model_digest
from .realworld_composition_data import sha256, write_json


def _architecture():
    from . import parametric_architecture

    return parametric_architecture


def rng_state(device):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
    }
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def pack(records, pad, device):
    """EOS is a valid input token; validity comes from lengths, never token IDs."""
    tokens, positions, labels = historical.pack(records, pad, device)
    lengths = torch.tensor([len(r["encoded"]["input"]) for r in records], device=device)
    valid = torch.arange(tokens.shape[1], device=device)[None] < lengths[:, None]
    return tokens, positions, labels, valid


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _finite(value, name):
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"Nonfinite {name}; this candidate must retain its failure")


def update(model, optimizer, records, spec, device, pad):
    """One historical effective batch, with exact full-batch routing gradients.

    Hard assignment fractions have no gradient. A no-grad pass collects their
    global values when there are multiple microbatches; replaying the same
    dropout RNG realizes the same routing in the differentiable pass. One
    microbatch already contains the full objective and needs no prepass.
    """
    if not records:
        raise ValueError("An update requires at least one example")
    microbatch = int(spec["microbatch_size"])
    if microbatch < 1:
        raise ValueError("microbatch_size must be positive")
    api = _architecture()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    chunks = [records[i : i + microbatch] for i in range(0, len(records), microbatch)]
    is_moe = spec["architecture"] in {"M8", "M4", "LM4R2"}
    needs_prepass = is_moe and len(chunks) > 1
    coefficient = float(spec.get("balance_coefficient", 0.01)) if is_moe else 0.0
    global_fractions, total_tokens, collected = None, None, []
    if needs_prepass:
        before = rng_state(device)
        try:
            with torch.no_grad():
                for chunk in chunks:
                    tokens, positions, _, valid = pack(chunk, pad, device)
                    with historical.autocast(device):
                        _, aux = model(tokens, positions, valid_mask=valid, return_aux=True)
                    collected.append(aux["router_stats"])
            global_fractions, total_tokens = api.combine_router_fractions(collected)
        finally:
            restore_rng(before, device)
    answer_loss, balance_loss = 0.0, 0.0
    executed, readout, attention = 0, 0, 0
    replay_count_error = torch.zeros((), dtype=torch.int64, device=device)
    for chunk_index, chunk in enumerate(chunks):
        tokens, positions, labels, valid = pack(chunk, pad, device)
        with historical.autocast(device):
            logits, aux = model(tokens, positions, valid_mask=valid, return_aux=True)
            answer = historical.example_losses(logits, labels).sum() / len(records)
            balance = (
                api.router_balance_loss(
                    aux["router_stats"],
                    assignment_fractions=global_fractions,
                    total_valid_tokens=total_tokens,
                )
                if is_moe
                else answer.new_zeros(())
            )
            loss = answer + coefficient * balance
        if needs_prepass:
            replay_count_error += sum(
                (left["assignment_counts"] - right["assignment_counts"]).abs().sum()
                for left, right in zip(collected[chunk_index], aux["router_stats"], strict=True)
            )
        elif is_moe:
            collected = [aux["router_stats"]]
            global_fractions, total_tokens = api.combine_router_fractions(collected)
        _finite(loss.detach(), "objective")
        loss.backward()
        answer_loss += float(answer.detach())
        balance_loss += float(balance.detach())
        executed += tokens.numel()
        readout += positions.numel()
        attention += len(chunk) * tokens.shape[1] ** 2
    if int(replay_count_error) != 0:
        raise RuntimeError("MoE RNG replay did not reproduce routing assignment counts")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(spec.get("clip_grad_norm", 1.0)), error_if_nonfinite=True
    )
    _finite(norm, "gradient norm")
    optimizer.step()
    effective = sum(len(r["encoded"]["input"]) for r in records)
    result = {
        "loss": answer_loss,
        "answer_ce": answer_loss,
        "balance_loss": balance_loss,
        "balance_coefficient": coefficient,
        "objective": answer_loss + coefficient * balance_loss,
        "gradient_norm": float(norm),
        "executed_input_tokens": executed,
        "effective_input_tokens": effective,
        "padding_input_tokens": executed - effective,
        "readout_positions": readout,
        "attention_token_pairs": attention,
        "routing_prepass_executed_input_tokens": executed if needs_prepass else 0,
        "routing_prepass_effective_input_tokens": effective if needs_prepass else 0,
    }
    result["compute"] = api.flop_ledger(
        model,
        executed_tokens=executed,
        valid_tokens=effective,
        projected_positions=readout,
        attention_pairs=attention,
    )
    prepass = result["compute"]["forward_leading_matmul_flops"] if needs_prepass else 0
    result["compute"]["routing_prepass_matmul_flops"] = prepass
    result["estimated_matmul_training_flops"] = (
        result["compute"]["estimated_matmul_training_flops"] + prepass
    )
    result["compute"]["estimated_total_training_matmul_flops"] = result[
        "estimated_matmul_training_flops"
    ]
    if is_moe:
        layer_count = len(global_fractions)
        assignments = torch.stack(
            [sum(c[layer]["assignment_counts"] for c in collected) for layer in range(layer_count)]
        )
        processed = torch.stack(
            [
                sum(c[layer]["processed_assignments"] for c in collected)
                for layer in range(layer_count)
            ]
        )
        dropped = sum(s["dropped_assignments"] for c in collected for s in c)
        if int(dropped) != 0 or not torch.equal(processed, assignments.sum(-1)):
            raise AssertionError("Dropless routing lost selected token assignments")
        result["router"] = {
            "assignment_fractions": torch.stack(global_fractions).detach().cpu().tolist(),
            "assignment_counts": assignments.detach().cpu().tolist(),
            "valid_tokens_per_layer": effective,
            "selected_assignments_per_layer": assignments.sum(-1).detach().cpu().tolist(),
            "processed_assignments_per_layer": processed.detach().cpu().tolist(),
            "router_entropy_per_layer": torch.stack(
                [
                    sum(c[layer]["entropy_sum"] for c in collected) / effective
                    for layer in range(layer_count)
                ]
            )
            .detach()
            .cpu()
            .tolist(),
            "selected_probability_mass_per_layer": torch.stack(
                [
                    sum(c[layer]["selected_mass_sum"] for c in collected) / effective
                    for layer in range(layer_count)
                ]
            )
            .detach()
            .cpu()
            .tolist(),
            "dropped_assignments": int(dropped),
            "replay_assignment_count_error": int(replay_count_error),
            "replay_checked": needs_prepass,
            "balance_scope": (
                "full_effective_batch_two_pass_rng_replay"
                if needs_prepass
                else "full_effective_batch_single_forward"
            ),
        }
    return result


def selection_score(metrics):
    """Frozen development panels only; no held-out-composition selection."""
    atomic, composition = metrics["atomic"]["nll"], metrics["train_composition"]["nll"]
    if atomic is None or composition is None:
        raise ValueError("Both fixed development panels need answer NLL measurements")
    value = 0.5 * atomic + 0.5 * composition
    _finite(value, "development selection NLL")
    return value


def _routing_metrics(router):
    """Expose numeric dictionaries to the existing W&B sidecar's flattening."""
    if not router:
        return {}
    return {
        "dropped_assignments": router["dropped_assignments"],
        "replay_assignment_count_error": router["replay_assignment_count_error"],
        "layers": {
            str(layer): {
                "entropy": router["router_entropy_per_layer"][layer],
                "selected_probability_mass": router["selected_probability_mass_per_layer"][layer],
                "expert_fraction": {str(i): value for i, value in enumerate(fractions)},
            }
            for layer, fractions in enumerate(router["assignment_fractions"])
        },
    }


def save_checkpoint(path, model, optimizer, stream, counts, step, counters, device, **extra):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "step": step,
            "counters": counters,
            "rng": rng_state(device),
            **extra,
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, stream, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    stream.load_state_dict(state["stream"])
    restore_rng(state["rng"], device)
    return state


def _identity(config, spec):
    payload = {
        "model": config["model"],
        "spec": spec,
        "data_sha256": config["data_sha256"],
        "tokenizer_sha256": config.get("tokenizer_sha256", config.get("tokenizer_files", {})),
        "source_files": config.get("source_files", {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify(config, config_path=None):
    """Check the concrete execution freeze; the historical design is not mutated."""
    if config.get("launch_enabled") is False:
        raise ValueError("This configuration explicitly disables execution")
    if sha256(config["data_file"]) != config["data_sha256"]:
        raise ValueError("Frozen data changed")
    for name, expected in config.get("tokenizer_sha256", {}).items():
        path = Path(config["tokenizer"]) / name
        if sha256(path) != expected:
            raise ValueError(f"Frozen tokenizer changed: {path}")
    for group in ("tokenizer_files", "source_files"):
        for path, expected in config.get(group, {}).items():
            if sha256(path) != expected:
                raise ValueError(f"Frozen {group} changed: {path}")
    if config.get("source_root"):
        root = Path(config["source_root"]).resolve()
        for module in (Path(__file__), Path(_architecture().__file__), Path(historical.__file__)):
            if not module.resolve().is_relative_to(root):
                raise ValueError(f"Imported code is outside the frozen source_root: {module}")
    if config_path is not None:
        if not config.get("execution_lock"):
            raise ValueError("CLI requires an execution_lock with the frozen config hash")
        lock = json.loads(Path(config["execution_lock"]).read_text())
        if lock["config_sha256"] != sha256(config_path):
            raise ValueError("Frozen configuration changed")
        if "torch" in lock.get("environment", {}):
            if str(torch.__version__) != lock["environment"]["torch"]:
                raise ValueError("Frozen torch version changed")
        if "transformers" in lock.get("environment", {}):
            if transformers.__version__ != lock["environment"]["transformers"]:
                raise ValueError("Frozen transformers version changed")


def _setup(device, config):
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)


def _resources(device):
    if torch.device(device).type != "cuda":
        return {"peak_gpu_memory_bytes": 0, "peak_gpu_reserved_bytes": 0}
    return {
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _process_identity():
    """Different containers can both run Python as PID 1; identify the namespace."""
    identity = {"pid": os.getpid()}
    if Path("/proc/self/stat").exists():
        identity["pid_namespace"] = os.readlink("/proc/self/ns/pid")
        identity["start_ticks"] = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    return identity


def run(config, spec, out, device="cuda:0"):
    """Train one frozen candidate. An independent process must subsequently audit."""
    verify(config)
    _setup(device, config)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trained.json").exists():
        trained = json.loads((out / "trained.json").read_text())
        if trained["identity"] != _identity(config, spec):
            raise ValueError("Existing output belongs to a different frozen training recipe")
        return trained
    api = _architecture()
    data = json.loads(Path(config["data_file"]).read_text())
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    records = data["atoms"] + data["train_compositions"]
    model = api.construct(config["model"], spec, device)
    optimizer = api.optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    stream = historical.BatchStream(len(records), spec["sampling_seed"])
    identity = _identity(config, spec)
    counts = np.zeros(len(records), dtype=np.int64)
    counters = {
        "training_seconds": 0.0,
        "examples": 0,
        "supervised_tokens": 0,
        "executed_input_tokens": 0,
        "effective_input_tokens": 0,
        "estimated_matmul_training_flops": 0,
        "routing_prepass_executed_input_tokens": 0,
    }
    initial_step, wall_offset, history, last = 0, 0.0, [], {}
    if (out / "latest.pt").exists():
        saved = load_checkpoint(out / "latest.pt", model, optimizer, stream, device)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint belongs to a different frozen training recipe")
        counts, initial_step, counters = saved["counts"], saved["step"], saved["counters"]
        wall_offset, last = saved.get("wall_seconds", 0.0), saved.get("last_update", {})
        learning = out / "learning.json"
        history = [r for r in json.loads(learning.read_text()) if r["step"] <= initial_step]
        metadata = json.loads((out / "run.json").read_text())
        metadata.update(pid=os.getpid(), resumed_from_step=initial_step)
        del saved
    else:
        torch.manual_seed(spec["dropout_seed"])
        random.seed(spec["dropout_seed"])
        np.random.seed(spec["dropout_seed"])
        metadata = {
            "spec": spec,
            "model": config["model"],
            "phase": config.get("phase", "development"),
            "evaluation_nodes": config["evaluation_nodes"],
            "world_sha256": config["data_sha256"],
            "initial_model_sha256": model_digest(model),
            "initial_tensor_sha256": historical.tensor_digests(model),
            "initialization_manifest": api.initialization_manifest(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_ledger": api.parameter_ledger(model),
            "atomic_examples": len(data["atoms"]),
            "composition_examples": len(data["train_compositions"]),
            "vocab_size": len(tokenizer),
            "dtype": "BF16 AMP; FP32 parameters and Adam states",
            "device": str(device),
            "gpu": torch.device(device).index or 0,
            "gpu_name": torch.cuda.get_device_name(device)
            if torch.device(device).type == "cuda"
            else "cpu",
            "pid": os.getpid(),
            "job_type": config.get("phase", "development") + "-training",
            "tracking_group": config.get("tracking", {}).get("group", config.get("experiment")),
            "learning_step_unit": "optimizer_updates",
            "identity": identity,
            "selection": "0.5 atomic + 0.5 train-composition NLL on original fixed panels",
            "selection_panel_sizes": {
                key: len(data["panels"][key]) for key in ("atomic", "train_composition")
            },
            "moe_balance": (
                "full-effective-batch; single forward when one microbatch, otherwise two-pass "
                "RNG replay; actual prepass compute included"
            ),
            "learning_rate_schedule": "historical lr * min((step - 1) / warmup_steps, 1)",
        }
        if config.get("execution_lock"):
            metadata["execution_lock_sha256"] = sha256(config["execution_lock"])
    write_json(out / "run.json", metadata)
    write_json(out / "learning.json", history)
    nodes = set(config["evaluation_nodes"]) | {int(spec["steps"])}
    log_interval = int(config.get("log_interval", 100))
    checkpoint_interval = int(config.get("resume_checkpoint_interval", 1000))
    started = time.perf_counter()

    def checkpoint(step):
        save_checkpoint(
            out / "latest.pt",
            model,
            optimizer,
            stream,
            counts,
            step,
            counters,
            device,
            identity=identity,
            wall_seconds=wall_offset + time.perf_counter() - started,
            last_update=last,
        )

    for step in range(initial_step, spec["steps"] + 1):
        if step > initial_step:
            warmup = int(spec["warmup_steps"])
            lr = spec["learning_rate"] * (min((step - 1) / warmup, 1.0) if warmup else 1.0)
            for group in optimizer.param_groups:
                group["lr"] = lr
            ids = stream.batch(spec["batch_size"])
            selected = [records[i] for i in ids]
            _synchronize(device)
            begin = time.perf_counter()
            last = update(model, optimizer, selected, spec, device, tokenizer.eos_token_id)
            _synchronize(device)
            counters["training_seconds"] += time.perf_counter() - begin
            counts[ids] += 1
            counters["examples"] += len(ids)
            counters["supervised_tokens"] += sum(len(r["encoded"]["target"]) for r in selected)
            for key in (
                "executed_input_tokens",
                "effective_input_tokens",
                "estimated_matmul_training_flops",
                "routing_prepass_executed_input_tokens",
            ):
                counters[key] += last[key]
        if step in nodes or step % log_interval == 0:
            if not any(r["step"] == step for r in history):
                metrics = {}
                if step in nodes:
                    metrics, predictions = historical.evaluate_dataset(
                        model, data, tokenizer, device
                    )
                    write_json(out / f"predictions-{step:07d}.json", predictions)
                    metrics["selection"] = {
                        "mixed_answer_nll": selection_score(metrics),
                        "atomic_n": metrics["atomic"]["nll_n"],
                        "train_composition_n": metrics["train_composition"]["nll_n"],
                    }
                if last.get("router"):
                    metrics["routing"] = _routing_metrics(last["router"])
                record = {
                    "step": step,
                    "metrics": metrics,
                    **counters,
                    "wall_seconds": wall_offset + time.perf_counter() - started,
                    "loss": last.get("answer_ce"),
                    "answer_ce": last.get("answer_ce"),
                    "balance_loss": last.get("balance_loss"),
                    "objective": last.get("objective"),
                    "gradient_norm": last.get("gradient_norm"),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "atomic_epochs": float(counts[: len(data["atoms"])].mean()),
                    "composition_epochs": float(counts[len(data["atoms"]) :].mean()),
                    "router": last.get("router", {}),
                    "compute_per_update": last.get("compute", {}),
                    **_resources(device),
                    "created_utc": utc(),
                }
                history.append(record)
                write_json(out / "learning.json", history)
                print(
                    json.dumps({"step": step, "loss": record["loss"], "metrics": metrics}),
                    flush=True,
                )
            write_json(
                out / "status.json",
                {
                    "state": "training",
                    "step": step,
                    "target_step": spec["steps"],
                    "loss": last.get("answer_ce"),
                    **counters,
                    "wall_seconds": wall_offset + time.perf_counter() - started,
                    "pid": os.getpid(),
                    "updated_utc": utc(),
                },
            )
        if step in nodes or (step > initial_step and step % checkpoint_interval == 0):
            checkpoint(step)
            if step in nodes:
                target = out / f"checkpoint-{step:07d}.pt"
                temporary = target.with_suffix(".tmp")
                torch.save(
                    {"model": model.state_dict(), "step": step, "identity": identity}, temporary
                )
                temporary.replace(target)
                np.savez_compressed(out / f"exposure-{step:07d}.npz", counts=counts)
    metrics, predictions = historical.evaluate_dataset(
        model, data, tokenizer, device, full=True, autonomous=True
    )
    write_json(out / "endpoint-predictions.json", predictions)
    write_json(out / "endpoint.json", metrics)
    result = {
        "state": "trained_pending_independent_reload",
        "step": spec["steps"],
        "pid": os.getpid(),
        "process_identity": _process_identity(),
        "identity": identity,
        "model_sha256": model_digest(model),
        "checkpoint_sha256": sha256(out / "latest.pt"),
        **counters,
        "selection_mixed_answer_nll": next(
            r["metrics"]["selection"]["mixed_answer_nll"]
            for r in reversed(history)
            if r["step"] == spec["steps"]
        ),
        "wall_seconds": wall_offset + time.perf_counter() - started,
        "completed_utc": utc(),
    }
    write_json(out / "trained.json", result)
    write_json(out / "status.json", result)
    return result


def audit(config, spec, out, device="cuda:0"):
    """Reconstruct and regenerate the full endpoint in an independent process."""
    verify(config)
    _setup(device, config)
    out = Path(out)
    trained = json.loads((out / "trained.json").read_text())
    same_process = (
        trained["process_identity"] == _process_identity()
        if "process_identity" in trained
        else trained["pid"] == os.getpid()
    )
    if same_process:
        raise ValueError("Independent audit must run in a different process")
    if trained["identity"] != _identity(config, spec):
        raise ValueError("Audit recipe differs from the training freeze")
    if sha256(out / "latest.pt") != trained["checkpoint_sha256"]:
        raise ValueError("Training checkpoint changed")
    model = _architecture().construct(config["model"], spec, device)
    checkpoint = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    if checkpoint["step"] != spec["steps"]:
        raise ValueError("Final checkpoint does not contain the requested endpoint")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    if model_digest(model) != trained["model_sha256"]:
        raise ValueError("Reloaded model does not match the trained endpoint")
    data = json.loads(Path(config["data_file"]).read_text())
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    metrics, predictions = historical.evaluate_dataset(
        model, data, tokenizer, device, full=True, autonomous=True
    )
    if predictions != json.loads((out / "endpoint-predictions.json").read_text()):
        raise AssertionError("Independent endpoint generation or NLL failed exact reproduction")
    if metrics != json.loads((out / "endpoint.json").read_text()):
        raise AssertionError("Independent endpoint metric recount failed")
    result = {
        "passed": True,
        "pid": os.getpid(),
        "source_training_pid": trained["pid"],
        "process_identity": _process_identity(),
        "source_process_identity": trained.get("process_identity"),
        "checkpoint_sha256": trained["checkpoint_sha256"],
        "model_sha256": trained["model_sha256"],
        "predictions_recomputed": sum(len(p) for p in predictions.values()),
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
            "step": spec["steps"],
            "independently_reloaded": True,
        },
    )
    return result
