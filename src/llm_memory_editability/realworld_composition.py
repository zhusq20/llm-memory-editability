"""Complete GPT/Loop training on original 2Wiki questions and atomic facts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2TokenizerFast

from .grok_depth import utc
from .grokking_reproduction import ReproductionGPT, model_digest
from .realworld_composition_data import (
    answer_scores,
    atomic_question,
    encode_example,
    sha256,
    write_json,
)


def construct(config, spec, device):
    """Build the same eight-block reference before taking a shared prefix."""
    torch.manual_seed(spec["initialization"])
    options = {
        "vocab_size": config["vocab_size"],
        "n_positions": config.get("positions", 1024),
        "n_embd": config["hidden_size"],
        "n_head": config["attention_heads"],
        "n_layer": 8,
        "resid_pdrop": config["dropout"],
        "embd_pdrop": config["dropout"],
        "attn_pdrop": config["dropout"],
        "activation_function": "gelu_new",
        "use_cache": False,
    }
    model = ReproductionGPT(GPT2Config(**options), repeats=spec["repeats"])
    model.transformer.h = torch.nn.ModuleList(list(model.transformer.h)[: spec["unique_layers"]])
    model.config.n_layer = spec["unique_layers"]
    return model.to(device)


def shared_prefix_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith("transformer.h.") and int(name.split(".")[2]) >= 4:
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_digests(model):
    return {
        name: hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        for name, tensor in model.state_dict().items()
    }


class BatchStream:
    """Keep actual incomplete epoch batches, with a complete restore state."""

    def __init__(self, size, seed):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.remaining = np.empty(0, dtype=np.int64)

    def batch(self, size):
        if not len(self.remaining):
            self.remaining = self.rng.permutation(self.size)
        chosen, self.remaining = self.remaining[:size], self.remaining[size:]
        return chosen

    def state_dict(self):
        return {
            "remaining": self.remaining.copy(),
            "rng": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        self.remaining = state["remaining"].copy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])


def pack(records, pad, device):
    length = math.ceil(max(len(r["encoded"]["input"]) for r in records) / 8) * 8
    targets = max(len(r["encoded"]["target"]) for r in records)
    tokens = np.full((len(records), length), pad, dtype=np.int64)
    positions = np.zeros((len(records), targets), dtype=np.int64)
    labels = np.full((len(records), targets), -100, dtype=np.int64)
    for i, record in enumerate(records):
        encoded = record["encoded"]
        tokens[i, : len(encoded["input"])] = encoded["input"]
        n = len(encoded["target"])
        positions[i, :n] = np.arange(len(encoded["prefix"]) - 1, len(encoded["prefix"]) - 1 + n)
        labels[i, :n] = encoded["target"]
    return tuple(torch.as_tensor(a, device=device) for a in (tokens, positions, labels))


def example_losses(logits, labels):
    valid = labels != -100
    if not bool(valid.any(dim=1).all()):
        raise ValueError("Every example must supervise its answer and EOS")
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(
        labels
    )
    return losses.sum(dim=1) / valid.sum(dim=1)


def optimizer_for(model, lr, decay):
    params = list(model.named_parameters())
    groups = [
        {
            "params": [p for n, p in params if not any(s in n for s in ["bias", "ln"])],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in params if any(s in n for s in ["bias", "ln"])],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=next(model.parameters()).is_cuda
    )


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def update(model, optimizer, records, spec, device, pad):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total, executed, readout, attention = 0.0, 0, 0, 0
    for start in range(0, len(records), spec["microbatch_size"]):
        chunk = records[start : start + spec["microbatch_size"]]
        tokens, positions, labels = pack(chunk, pad, device)
        with autocast(device):
            loss = example_losses(model(tokens, positions), labels).sum() / len(records)
        loss.backward()
        loss_total += float(loss.detach())
        executed += tokens.numel()
        readout += positions.numel()
        attention += len(chunk) * tokens.shape[1] ** 2
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not math.isfinite(loss_total) or not math.isfinite(float(norm)):
        raise FloatingPointError("Nonfinite loss or gradient; preserve this failure")
    optimizer.step()
    d, blocks = model.config.n_embd, len(model.transformer.h) * model.repeats
    flops = (
        6 * 12 * d * d * blocks * executed
        + 6 * d * model.config.vocab_size * readout
        + 12 * blocks * d * attention
    )
    return {
        "loss": loss_total,
        "gradient_norm": float(norm),
        "executed_input_tokens": executed,
        "effective_input_tokens": sum(len(r["encoded"]["input"]) for r in records),
        "estimated_matmul_training_flops": flops,
    }


@torch.no_grad()
def generate(model, prefixes, tokenizer, device, limit=64, sequence_limit=256):
    """Right-padded greedy generation; only the question prefix enters the model."""
    model.eval()
    sizes = np.array([len(p) for p in prefixes], dtype=np.int64)
    if not len(sizes) or not (sizes > 0).all() or not (sizes <= sequence_limit).all():
        raise ValueError("Generation requires nonempty prefixes within the sequence limit")
    budgets = np.minimum(limit, sequence_limit - sizes)
    tokens = torch.full(
        (len(prefixes), min(sequence_limit, int(sizes.max()) + limit)),
        tokenizer.eos_token_id,
        device=device,
        dtype=torch.long,
    )
    for i, prefix in enumerate(prefixes):
        tokens[i, : len(prefix)] = torch.as_tensor(prefix, device=device)
    output, ended = [[] for _ in prefixes], np.zeros(len(prefixes), dtype=bool)
    for offset in range(limit):
        active = ~ended & (offset < budgets)
        if not active.any():
            break
        length = min(sequence_limit, int(sizes.max()) + offset)
        positions = torch.as_tensor(np.minimum(sizes + offset - 1, length - 1), device=device)[
            :, None
        ]
        with autocast(device):
            selected = model(tokens[:, :length], positions)[:, 0].argmax(-1)
        chosen = selected.cpu().tolist()
        for i, token in enumerate(chosen):
            if active[i]:
                output[i].append(token)
                ended[i] = token == tokenizer.eos_token_id
        indices = torch.as_tensor(np.flatnonzero(active), device=device)
        tokens[indices, torch.as_tensor(sizes[active] + offset, device=device)] = selected[indices]
    model.train()
    return [
        {
            "prediction": tokenizer.decode(
                ids[:-1] if stop else ids, skip_special_tokens=True
            ).strip(),
            "generated_tokens": ids,
            "eos": bool(stop),
            "truncated": not bool(stop),
            "context_limit_reached": not bool(stop) and bool(len(ids) >= sequence_limit - size),
            "generation_limit_reached": not bool(stop) and len(ids) >= limit,
        }
        for ids, stop, size in zip(output, ended, sizes, strict=True)
    ]


def aggregate(predictions):
    fields = ["canonical_em", "alias_em", "unambiguous_alias_em", "f1", "nll", "eos", "truncated"]
    result = {"n": len(predictions)}
    for key in fields:
        values = [float(p[key]) for p in predictions if p[key] is not None]
        result[key] = sum(values) / len(values) if values else None
        if key == "nll":
            result["nll_n"] = len(values)
    eligible = [p for p in predictions if p.get("has_unambiguous_answer", True)]
    result["unambiguous_answer_coverage"] = (
        len(eligible) / len(predictions) if predictions else None
    )
    result["unambiguous_alias_em_eligible"] = (
        sum(float(p["unambiguous_alias_em"]) for p in eligible) / len(eligible)
        if eligible
        else None
    )
    result["answer_accuracy"] = result["alias_em"]
    return result


@torch.no_grad()
def evaluate(model, records, tokenizer, device, batch_size=32):
    predictions = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        generated = generate(model, [r["encoded"]["prefix"] for r in chunk], tokenizer, device)
        model.eval()
        tokens, positions, labels = pack(chunk, tokenizer.eos_token_id, device)
        with autocast(device):
            losses = example_losses(model(tokens, positions), labels).float().cpu().tolist()
        for row, result, loss in zip(chunk, generated, losses, strict=True):
            em, f1 = answer_scores(result["prediction"], [row["answer"], *row.get("aliases", [])])
            unique_answers = row.get(
                "unambiguous_answers", [row["answer"], *row.get("unambiguous_aliases", [])]
            )
            result.update(
                id=row["id"],
                canonical_em=answer_scores(result["prediction"], [row["answer"]])[0],
                alias_em=em,
                f1=f1,
                nll=loss,
                unambiguous_alias_em=answer_scores(result["prediction"], unique_answers)[0],
                has_unambiguous_answer=bool(unique_answers),
                canonical_answer_globally_unambiguous=row.get(
                    "canonical_answer_globally_unambiguous", True
                ),
                gold=row["answer"],
                role=row.get("role"),
                required_role=row.get("required_role"),
            )
            predictions.append(result)
    model.train()
    return aggregate(predictions), predictions


def evaluate_dataset(model, data, tokenizer, device, full=False, autonomous=False):
    atoms = {r["id"]: r for r in data["atoms"]}
    chains = {r["id"]: r for r in data["train_compositions"] + data["evaluation_compositions"]}
    panels = data["panels"]
    groups = {
        "atomic": list(atoms.values()) if full else [atoms[i] for i in panels["atomic"]],
        "train_composition": data["train_compositions"]
        if full
        else [chains[i] for i in panels["train_composition"]],
    }
    selected = (
        data["evaluation_compositions"]
        if full
        else [chains[i] for name, ids in panels.items() if name.startswith("test_") for i in ids]
    )
    groups["test_all"] = selected
    metrics, raw = {}, {}
    for name, records in groups.items():
        metrics[name], raw[name] = (
            evaluate(model, records, tokenizer, device) if records else (aggregate([]), [])
        )
    lookup = {p["id"]: p for p in raw["atomic"]}
    for role in ["II", "IO", "OI", "OO"]:
        metrics["test_" + role.lower()] = aggregate(
            [p for p in raw["test_all"] if p["role"] == role]
        )
        metrics["test_required_role_" + role.lower()] = aggregate(
            [p for p in raw["test_all"] if p["required_role"] == role]
        )
    filters = {
        "test_unambiguous_canonical_answers": lambda r: r.get(
            "canonical_answer_globally_unambiguous", True
        ),
        "test_no_known_shortcut": lambda r: not r.get("known_direct_shortcut", False),
        "test_head_tail_unseen": lambda r: not r.get(
            "head_tail_seen_in_combination_training", False
        ),
        "test_seen_operation": lambda r: r.get("operation_seen", False),
        "test_no_shortcut_seen_operation": lambda r: not r.get("known_direct_shortcut", False)
        and r.get("operation_seen", False),
    }
    for name, include in filters.items():
        ids = {r["id"] for r in selected if include(r)}
        metrics[name] = aggregate([p for p in raw["test_all"] if p["id"] in ids])
        metrics[name]["coverage"] = len(ids) / len(selected) if selected else None
    for index, name in enumerate(["test_first_atom", "test_second_atom"]):
        metrics[name] = aggregate([lookup[r["atom_ids"][index]] for r in selected])
    metrics["gold_bridge_second"] = metrics["test_second_atom"].copy()
    prerequisite = {r["id"] for r in selected if all(lookup[a]["alias_em"] for a in r["atom_ids"])}
    metrics["test_prerequisites_correct"] = aggregate(
        [p for p in raw["test_all"] if p["id"] in prerequisite]
    )
    metrics["test_prerequisites_correct"]["coverage"] = (
        len(prerequisite) / len(selected) if selected else None
    )
    if autonomous:
        second_calls = []
        invalid = []
        for row in selected:
            bridge_prediction = lookup[row["atom_ids"][0]]["prediction"]
            q = atomic_question(bridge_prediction, row["edges"][1][1])
            try:
                encoded = encode_example(tokenizer, q, row["answer"])
            except ValueError:
                invalid.append(row)
                continue
            second_calls.append(
                {**row, "question": q, "encoded": encoded, "autonomous_bridge": bridge_prediction}
            )
        _, predictions = (
            evaluate(model, second_calls, tokenizer, device) if second_calls else (None, [])
        )
        for row, result in zip(second_calls, predictions, strict=True):
            result["bridge_prediction"] = row["autonomous_bridge"]
            result["second_question"] = row["question"]
        predictions.extend(
            {
                "id": row["id"],
                "canonical_em": 0.0,
                "alias_em": 0.0,
                "unambiguous_alias_em": 0.0,
                "f1": 0.0,
                "nll": None,
                "eos": False,
                "truncated": True,
                "has_unambiguous_answer": bool(row.get("unambiguous_answers", [])),
                "canonical_answer_globally_unambiguous": row.get(
                    "canonical_answer_globally_unambiguous", True
                ),
                "invalid_generated_bridge_length": True,
            }
            for row in invalid
        )
        raw["autonomous"] = predictions
        metrics["autonomous"] = aggregate(predictions)
        metrics["autonomous"]["invalid_bridge_prompts"] = len(invalid)
    return metrics, raw


def save_checkpoint(path, model, optimizer, stream, counts, step, counters, wall_seconds=None):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "step": step,
            "counters": counters,
            "wall_seconds": wall_seconds,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def worker(config, spec, gpu):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{gpu}"
    data = json.loads(Path(config["data_file"]).read_text())
    assert sha256(config["data_file"]) == config["data_sha256"]
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    phase = config.get("phase", "development")
    out = Path(config["results_root"]) / phase / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    model = construct(config["model"], spec, device)
    optimizer = optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    records = data["atoms"] + data["train_compositions"]
    stream = BatchStream(len(records), spec["sampling_seed"])
    counts = np.zeros(len(records), dtype=np.int64)
    counters = dict(
        training_seconds=0.0,
        examples=0,
        supervised_tokens=0,
        executed_input_tokens=0,
        effective_input_tokens=0,
        estimated_matmul_training_flops=0,
    )
    history, initial_step, wall_offset = [], 0, 0.0
    started = time.perf_counter()
    if (out / "latest.pt").exists():
        checkpoint = torch.load(out / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        stream.load_state_dict(checkpoint["stream"])
        counts, initial_step, counters = (
            checkpoint["counts"],
            checkpoint["step"],
            checkpoint["counters"],
        )
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
        history = [
            r for r in json.loads((out / "learning.json").read_text()) if r["step"] <= initial_step
        ]
        wall_offset = checkpoint.get("wall_seconds")
        if wall_offset is None:
            wall_offset = history[-1]["wall_seconds"] if history else 0.0
        metadata = json.loads((out / "run.json").read_text())
        metadata.update(pid=os.getpid(), gpu=gpu, resumed_from_step=initial_step)
        write_json(out / "run.json", metadata)
        del checkpoint
    else:
        torch.manual_seed(spec["dropout_seed"])
        metadata = {
            "spec": spec,
            "model": config["model"],
            "phase": phase,
            "evaluation_nodes": config["evaluation_nodes"],
            "execution_lock_sha256": sha256(config["execution_lock"]),
            "world_sha256": config["data_sha256"],
            "initial_model_sha256": model_digest(model),
            "initial_tensor_sha256": tensor_digests(model),
            "common_four_block_initial_sha256": shared_prefix_digest(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "atomic_examples": len(data["atoms"]),
            "composition_examples": len(data["train_compositions"]),
            "vocab_size": len(tokenizer),
            "dtype": "BF16 AMP; FP32 parameters and Adam states",
            "gpu": gpu,
            "gpu_name": torch.cuda.get_device_name(gpu),
            "pid": os.getpid(),
            "job_type": phase + "-training",
            "tracking_group": config["tracking"]["group"],
        }
        write_json(out / "run.json", metadata)
        write_json(out / "learning.json", [])
    nodes = set(config["evaluation_nodes"])
    last_loss = None
    for step in range(initial_step, spec["steps"] + 1):
        if step > initial_step:
            for group in optimizer.param_groups:
                group["lr"] = spec["learning_rate"] * min((step - 1) / spec["warmup_steps"], 1.0)
            ids = stream.batch(spec["batch_size"])
            selected = [records[i] for i in ids]
            torch.cuda.synchronize()
            began = time.perf_counter()
            values = update(model, optimizer, selected, spec, device, tokenizer.eos_token_id)
            torch.cuda.synchronize()
            counters["training_seconds"] += time.perf_counter() - began
            last_loss = values["loss"]
            counts[ids] += 1
            counters["examples"] += len(ids)
            counters["supervised_tokens"] += sum(len(r["encoded"]["target"]) for r in selected)
            for key in [
                "executed_input_tokens",
                "effective_input_tokens",
                "estimated_matmul_training_flops",
            ]:
                counters[key] += values[key]
        if step % 50 == 0:
            write_json(
                out / "status.json",
                {
                    "state": "training",
                    "step": step,
                    "target_step": spec["steps"],
                    "loss": last_loss,
                    "training_seconds": counters["training_seconds"],
                    "examples": counters["examples"],
                    "wall_seconds": wall_offset + time.perf_counter() - started,
                    "pid": os.getpid(),
                    "updated_utc": utc(),
                },
            )
        if step in nodes and not any(r["step"] == step for r in history):
            metrics, predictions = evaluate_dataset(model, data, tokenizer, device)
            write_json(out / f"predictions-{step:07d}.json", predictions)
            history.append(
                {
                    "step": step,
                    "metrics": metrics,
                    **counters,
                    "wall_seconds": wall_offset + time.perf_counter() - started,
                    "loss": last_loss,
                    "atomic_epochs": float(counts[: len(data["atoms"])].mean()),
                    "composition_epochs": float(counts[len(data["atoms"]) :].mean()),
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                    "created_utc": utc(),
                }
            )
            write_json(out / "learning.json", history)
            save_checkpoint(
                out / "latest.pt",
                model,
                optimizer,
                stream,
                counts,
                step,
                counters,
                wall_offset + time.perf_counter() - started,
            )
            torch.save(
                {"model": model.state_dict(), "step": step}, out / f"checkpoint-{step:07d}.pt"
            )
            np.savez_compressed(out / f"exposure-{step:07d}.npz", counts=counts)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": last_loss,
                        "metrics": metrics,
                        "training_seconds": counters["training_seconds"],
                    }
                ),
                flush=True,
            )
        elif (
            step > initial_step and step % config.get("resume_checkpoint_interval", 1000000000) == 0
        ):
            save_checkpoint(
                out / "latest.pt",
                model,
                optimizer,
                stream,
                counts,
                step,
                counters,
                wall_offset + time.perf_counter() - started,
            )
    metrics, predictions = evaluate_dataset(
        model, data, tokenizer, device, full=True, autonomous=True
    )
    write_json(out / "endpoint-predictions.json", predictions)
    write_json(out / "endpoint.json", metrics)
    result = {
        "state": "trained_pending_independent_reload",
        "step": spec["steps"],
        "pid": os.getpid(),
        "model_sha256": model_digest(model),
        "checkpoint_sha256": sha256(out / "latest.pt"),
        **counters,
        "wall_seconds": wall_offset + time.perf_counter() - started,
        "completed_utc": utc(),
    }
    write_json(out / "trained.json", result)
    write_json(out / "status.json", result)


def audit(config, spec, gpu):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{gpu}"
    out = Path(config["results_root"]) / config.get("phase", "development") / spec["name"]
    trained = json.loads((out / "trained.json").read_text())
    assert trained["pid"] != os.getpid()
    assert sha256(out / "latest.pt") == trained["checkpoint_sha256"]
    assert sha256(config["data_file"]) == config["data_sha256"]
    data = json.loads(Path(config["data_file"]).read_text())
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    model = construct(config["model"], spec, device)
    state = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    assert state["step"] == spec["steps"]
    del state
    assert model_digest(model) == trained["model_sha256"]
    metrics, predictions = evaluate_dataset(
        model, data, tokenizer, device, full=True, autonomous=True
    )
    expected = json.loads((out / "endpoint-predictions.json").read_text())
    assert predictions == expected, (
        "Endpoint generation or teacher-forced scores do not independently reproduce"
    )
    assert metrics == json.loads((out / "endpoint.json").read_text())
    result = {
        "passed": True,
        "pid": os.getpid(),
        "source_training_pid": trained["pid"],
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
        {"state": "complete", "step": spec["steps"], "independently_reloaded": True},
    )
