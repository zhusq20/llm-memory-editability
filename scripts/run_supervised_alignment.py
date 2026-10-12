#!/usr/bin/env python3
"""Train and replay supervised-position alignment, same-position KV and DiscoLoop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.bios_model import ModelConfig, matmul_flops  # noqa: E402
from llm_memory_editability.grok_depth import GraphStep, utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import (  # noqa: E402
    MergedBatchStream,
    digest,
    encode_rows,
    evaluate_rows,
    learning_rate,
    model_digest,
    optimizer_for,
)
from llm_memory_editability.supervised_alignment import (  # noqa: E402
    SupervisedAlignmentGPT,
    supervised_alignment_loss,
)

PANELS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")


def architecture_flags(arm):
    """Materialize every forward flag for checkpoint and independent audit checks."""
    flags = {
        "first_round_kv": arm.get("first_round_kv", False),
        "discoloop": arm.get("discoloop", False),
        "discoloop_alpha": arm.get("discoloop_alpha", 1.0),
        "discoloop_temperature": arm.get("discoloop_temperature", 1.0),
        "discoloop_eps": arm.get("discoloop_eps", 1e-6),
    }
    return flags


def validate_arm(arm):
    if arm["kind"] not in ("none", "label_ce", "alignment"):
        raise ValueError("Unknown objective arm")
    if arm.get("scope", "supervised") != "supervised":
        raise ValueError("Only original supervised positions are authorized")
    if not math.isfinite(arm["coefficient"]) or arm["coefficient"] < 0:
        raise ValueError("Auxiliary coefficient must be finite and nonnegative")
    if arm["kind"] == "none" and arm["coefficient"] != 0:
        raise ValueError("No-auxiliary-loss coefficient must be zero")
    flags = architecture_flags(arm)
    if not isinstance(flags["first_round_kv"], bool) or not isinstance(flags["discoloop"], bool):
        raise ValueError("Architecture switches must be boolean")
    if flags["discoloop"] and (flags["first_round_kv"] or arm["kind"] != "none"):
        raise ValueError("DiscoLoop is a standalone literature baseline in this matrix")
    for key in ("discoloop_alpha", "discoloop_temperature", "discoloop_eps"):
        if not math.isfinite(flags[key]) or flags[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")


def exposure_summary(counts, atomic_size, steps, batch_size):
    """Count actual samples; masked epoch-tail slots are not data exposure."""
    size = len(counts)
    epochs, remaining = divmod(steps, math.ceil(size / batch_size))
    expected = epochs * size + remaining * batch_size
    actual = int(counts.sum())
    if actual != expected or int(counts.max()) - int(counts.min()) > 1:
        raise AssertionError("Merged stream exposure differs from shuffled partial-batch epochs")
    atomic = int(counts[:atomic_size].sum())
    composite = actual - atomic
    return {
        "examples": actual,
        "counts": {"atomic": atomic, "composite": composite},
        "atomic_epochs": atomic / atomic_size,
        "composition_epochs": composite / (size - atomic_size),
        "effective_input_tokens": atomic * 3 + composite * 4,
        "supervised_tokens": actual * 2,
        "executed_input_tokens": steps * batch_size * 4,
        "masked_batch_slots": steps * batch_size - actual,
    }


def state_digest(state):
    result = hashlib.sha256()
    for name, tensor in state.items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def initialize(config, device, arm):
    """Load exact source weights, then explicitly seed each new arm identically."""
    input_root, spec = Path(config["input_root"]), config["spec"]
    metadata = json.loads((input_root / "data/complete.json").read_text())
    for name in ("world", "panels"):
        if digest(input_root / f"data/{name}.npz") != metadata[f"{name}_sha256"]:
            raise ValueError(f"Source {name} hash mismatch")
    checkpoint = Path(
        config.get(
            "initial_checkpoint",
            input_root / "development" / config["source_run"] / "checkpoint-0000000.pt",
        )
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["step"] != 0:
        raise ValueError("The comparison must begin from the original step-zero checkpoint")
    for key in (
        "unique_layers",
        "repeats",
        "phi",
        "weight_decay",
        "initialization",
        "stream_seed",
        "lr",
        "warmup",
        "batch_size",
    ):
        if spec[key] != saved["spec"][key]:
            raise ValueError(f"Source recipe mismatch: {key}")
    torch.manual_seed(spec["initialization"])
    hf_config = GPT2Config.from_pretrained(input_root / "reference/gpt2", local_files_only=True)
    hf_config.vocab_size = metadata["vocab_size"]
    hf_config.n_layer = spec["unique_layers"]
    hf_config.use_cache = False
    model = SupervisedAlignmentGPT(hf_config, spec["repeats"], **architecture_flags(arm))
    source_hash = state_digest(saved["model"])
    initialized_hash = model_digest(model)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    del saved
    if model_digest(model) != source_hash:
        raise AssertionError("Source checkpoint weights were not copied exactly")
    torch.manual_seed(config["dropout_seed"])
    model.train()
    provenance = {
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": digest(checkpoint),
        "initial_model_sha256": source_hash,
        "constructor_model_sha256": initialized_hash,
        "constructor_matches_source_weights": initialized_hash == source_hash,
        "dropout_seed": config["dropout_seed"],
        "architecture": architecture_flags(arm),
        "initial_cpu_rng_sha256": hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest(),
        "initial_cuda_rng_sha256": hashlib.sha256(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
        ).hexdigest(),
        "rng_policy": "Fresh identical explicit CPU/CUDA seed after checkpoint loading; "
        "historical training dropout trajectory is not claimed identical. "
        "KV changes attention shapes and random-number consumption; identical per-tensor "
        "dropout masks are not claimed across KV settings.",
    }
    return model, metadata, provenance


def make_table(world, metadata, spec, device):
    support = round(spec["phi"] * metadata["atomic_id"])
    selected = world["train_order"][:support]
    if len(selected) != support:
        raise ValueError("Requested composition support unavailable")
    atoms = encode_rows(world["atoms"], metadata)
    composites = encode_rows(world["chains"][selected], metadata)
    padding = (
        np.full((1, 4), metadata["pad_token"], dtype=np.int64),
        np.array([[2, 3]], dtype=np.int64),
        np.array([[-100, -100]], dtype=np.int64),
    )
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device)
        for parts in zip(atoms, composites, padding, strict=True)
    )
    return table, selected


def extra_attention_training_flops(spec, width, arm, sequence=4):
    """Dense executed-shape estimate for reading first-round K/V additionally.

    Each later round's QK and attention-value products see 2T keys instead of T.
    Cached keys are masked to the same position; this dense implementation still
    executes the full 2T shape and must not be reported as sparse attention cost.
    That adds 4*B*T*T*H forward FLOPs per layer; the conventional training factor
    is three. Existing Q/K/V projections and trainable parameter counts do not
    increase. Softmax, masking, cache concatenation and memory traffic are excluded.
    """
    if not arm.get("first_round_kv", False):
        return 0
    return (
        12
        * spec["batch_size"]
        * sequence
        * sequence
        * width
        * spec["unique_layers"]
        * (spec["repeats"] - 1)
    )


def extra_projection_training_flops(spec, width, vocab_size, arm, output_positions=2):
    """Only existing answer/EOS slots: one detached or differentiable projection.

    Alignment projects under no_grad (2 multiply/add FLOPs). Label-CE reuses its
    differentiable logits to obtain detached p(y), so adds forward and both
    backward products (6 FLOPs). Padded graph slots are included in execution.
    """
    multiplier = {"none": 0, "alignment": 2, "label_ce": 6}[arm["kind"]]
    return multiplier * spec["batch_size"] * output_positions * width * vocab_size


def extra_discoloop_training_flops(spec, width, vocab_size, arm, sequence=4):
    """Two differentiable vocabulary matmuls per nonfinal loop boundary.

    ln_f, RMS normalization, softmax, elementwise operations and memory traffic
    are excluded. The full soft distribution is computed at every input slot.
    """
    if not arm.get("discoloop", False):
        return 0
    return 12 * spec["batch_size"] * sequence * width * vocab_size * (spec["repeats"] - 1)


def losses(model, table, indices, arm):
    x, positions, labels = (part[indices] for part in table)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        logits, first, final = model.forward_with_states(x, positions)
        main = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        if arm["kind"] == "none":
            aux = main.new_zeros(())
            info = {key: aux for key in ("valid_positions", "selected_positions", "confidence_sum")}
        else:
            aux, info = supervised_alignment_loss(
                model,
                first,
                final,
                positions,
                labels,
                kind=arm["kind"],
                scope=arm.get("scope", "supervised"),
            )
        total = main + arm["coefficient"] * aux
    return total, main, aux, info


class AlignmentStep(GraphStep):
    """Retain original graph warmup parameter, Adam and RNG restoration safeguards."""

    def __init__(self, model, optimizer, table, batch_size, arm):
        self.arm = arm
        super().__init__(model, optimizer, table, batch_size)

    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        total, self.main_loss, self.aux_loss, info = losses(
            self.model, self.table, self.index, self.arm
        )
        self.aux_info = {
            key: info[key] for key in ("valid_positions", "selected_positions", "confidence_sum")
        }
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return total, norm


def gradient_diagnostic(model, table, indices, arm, device):
    """One fixed training batch; no optimizer update or RNG consumption survives."""
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    before = model_digest(model)
    params = tuple(model.parameters())
    try:
        total, main, aux, info = losses(model, table, indices, arm)

        def norm(loss, retain_graph):
            gradients = torch.autograd.grad(
                loss, params, retain_graph=retain_graph, allow_unused=True
            )
            return float(
                torch.stack([g.float().square().sum() for g in gradients if g is not None])
                .sum()
                .sqrt()
            )

        main_norm = norm(main, arm["kind"] != "none")
        aux_norm = norm(aux, False) if arm["kind"] != "none" else 0.0
        result = {
            "main_loss": float(main.detach()),
            "aux_loss": float(aux.detach()),
            "weighted_aux_loss": float(aux.detach()) * arm["coefficient"],
            "total_loss": float(total.detach()),
            "main_gradient_norm": main_norm,
            "aux_gradient_norm": aux_norm,
            "weighted_aux_gradient_norm": arm["coefficient"] * aux_norm,
            "weighted_aux_to_main_gradient_norm": arm["coefficient"] * aux_norm / main_norm,
            **{
                key: float(info[key])
                for key in ("valid_positions", "selected_positions", "confidence_sum")
            },
            "scope": "first original merged-stream training batch at step zero",
        }
        if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
            raise FloatingPointError("Nonfinite component gradient diagnostic")
        return result
    finally:
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        if model_digest(model) != before:
            raise AssertionError("Gradient diagnostic changed model parameters")


def graph_values(graph, coefficient):
    values = {
        "loss": float(graph.loss),
        "main_loss": float(graph.main_loss),
        "aux_loss": float(graph.aux_loss),
        "weighted_aux_loss": coefficient * float(graph.aux_loss),
        "gradient_norm": float(graph.grad_norm),
        **{f"aux_{key}": float(value) for key, value in graph.aux_info.items()},
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Nonfinite training objective or gradient")
    return values


def baseline_eager_contract(reference, candidate, table, indices, seed, reference_losses):
    """Compare disabled forwards and gradients at matched dropout RNG, without updates.

    ``reference_losses`` is the existing runner's actual objective entrypoint.
    The callable is also usable on a tiny CPU model in contract tests. Caller RNG
    and parameter values are preserved; parameter gradients are cleared.
    """
    device = table[0].device
    devices = [device.index if device.index is not None else torch.cuda.current_device()]
    if device.type != "cuda":
        devices = []
    old_arm = {"kind": "none", "coefficient": 0.0, "scope": "all", "first_round_kv": False}
    new_arm = {**old_arm, "scope": "supervised", "discoloop": False}
    before = [model_digest(model) for model in (reference, candidate)]
    if before[0] != before[1]:
        raise AssertionError("Disabled baseline models must start from exactly equal weights")
    x, positions = table[0][indices], table[1][indices]
    records, gradients = [], None
    gradient_max_delta, gradient_names_match, gradients_exact = 0.0, True, True
    try:
        with torch.random.fork_rng(devices=devices):
            for model, objective, arm in (
                (reference, reference_losses, old_arm),
                (candidate, losses, new_arm),
            ):
                model.zero_grad(set_to_none=True)
                torch.manual_seed(seed)
                with (
                    torch.no_grad(),
                    torch.autocast(
                        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ),
                ):
                    logits = model(x, positions).detach().cpu()
                torch.manual_seed(seed)
                total, main, aux, _ = objective(model, table, indices, arm)
                total.backward()
                records.append((logits, total.detach().cpu(), main.detach().cpu(), float(aux)))
                if gradients is None:
                    gradients = {
                        name: param.grad.detach().cpu().clone()
                        for name, param in model.named_parameters()
                        if param.grad is not None
                    }
                else:
                    current = {
                        name: param.grad.detach().cpu()
                        for name, param in model.named_parameters()
                        if param.grad is not None
                    }
                    gradient_names_match = set(current) == set(gradients)
                    for name in set(current) & set(gradients):
                        delta = float((current[name] - gradients[name]).abs().max())
                        gradient_max_delta = max(gradient_max_delta, delta)
                        gradients_exact &= torch.equal(current[name], gradients[name])
        logits_exact = torch.equal(records[0][0], records[1][0])
        losses_exact = all(torch.equal(records[0][i], records[1][i]) for i in (1, 2))
        auxiliary_zero = records[0][3] == records[1][3] == 0.0
        result = {
            "logits_exact": logits_exact,
            "main_and_total_losses_exact": losses_exact,
            "auxiliary_zero": auxiliary_zero,
            "gradient_parameter_names_exact": gradient_names_match,
            "all_parameter_gradients_exact": gradients_exact,
            "max_absolute_gradient_difference": gradient_max_delta,
            "parameter_gradient_tensors": len(gradients),
            "reference_gradient_sha256": state_digest(gradients),
            "main_loss": float(records[0][2]),
        }
        result["passed"] = all(
            (logits_exact, losses_exact, auxiliary_zero, gradient_names_match, gradients_exact)
        )
        return result
    finally:
        for model in (reference, candidate):
            model.zero_grad(set_to_none=True)
        if before != [model_digest(model) for model in (reference, candidate)]:
            raise AssertionError("Baseline compatibility diagnostic changed model parameters")


def _optimizer_states_exact(reference, candidate):
    """Parameter IDs differ across models; optimizer groups retain identical ordering."""
    old, new = reference.state_dict(), candidate.state_dict()
    if old["state"].keys() != new["state"].keys():
        return False
    for key, state in old["state"].items():
        if state.keys() != new["state"][key].keys():
            return False
        for name, value in state.items():
            other = new["state"][key][name]
            if torch.is_tensor(value):
                if not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


def baseline_compatibility(config, base, config_path):
    """Check the reused frozen baseline against the new disabled executor on CUDA."""
    base.mkdir(parents=True, exist_ok=True)
    destination = base / "compatibility.json"
    if destination.exists():
        raise FileExistsError("Baseline compatibility evidence already exists; preserve it")
    started = time.perf_counter()
    findings = {}
    try:
        reference_path = Path(
            config.get("baseline_runner", ROOT / "scripts/run_self_decode_alignment.py")
        )
        module_spec = importlib.util.spec_from_file_location(
            "frozen_alignment_runner", reference_path
        )
        reference_runner = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(reference_runner)
        device, spec = torch.device("cuda:0"), config["spec"]
        old_arm = {"kind": "none", "coefficient": 0.0, "scope": "all", "first_round_kv": False}
        arm = {**old_arm, "scope": "supervised", "discoloop": False}
        reference, metadata, reference_provenance = reference_runner.initialize(
            config, device, old_arm
        )
        candidate, candidate_metadata, provenance = initialize(config, device, arm)
        findings["initial_weights_exact"] = (
            model_digest(reference) == model_digest(candidate) == provenance["initial_model_sha256"]
        )
        findings["metadata_exact"] = metadata == candidate_metadata
        world = dict(np.load(Path(config["input_root"]) / "data/world.npz"))
        table, selected = make_table(world, metadata, spec, device)
        reference_table, reference_selected = reference_runner.make_table(
            world, metadata, spec, device
        )
        findings["table_and_selection_exact"] = np.array_equal(
            selected, reference_selected
        ) and all(torch.equal(a, b) for a, b in zip(table, reference_table, strict=True))
        stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
        reference_stream = reference_runner.MergedBatchStream(stream.size, spec["stream_seed"])
        stream_hash, streams_exact = hashlib.sha256(), True
        stream_steps = 2 * math.ceil(stream.size / spec["batch_size"]) + 3
        for _ in range(stream_steps):
            indices, size = stream.batch(spec["batch_size"])
            reference_indices, reference_size = reference_stream.batch(spec["batch_size"])
            streams_exact &= size == reference_size and np.array_equal(indices, reference_indices)
            stream_hash.update(indices.tobytes())
        findings["sample_stream"] = {
            "exact": streams_exact,
            "steps": stream_steps,
            "scope": "two full original epochs plus three batches, including masked tails",
            "indices_sha256": stream_hash.hexdigest(),
        }
        stream = MergedBatchStream(stream.size, spec["stream_seed"])
        first_indices = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
        findings["eager"] = baseline_eager_contract(
            reference,
            candidate,
            table,
            first_indices,
            config["dropout_seed"],
            reference_runner.losses,
        )
        lrs = [torch.tensor(0.0, device=device) for _ in range(2)]
        optimizers = [
            optimizer_for(model, lr, spec["weight_decay"])
            for model, lr in zip((reference, candidate), lrs, strict=True)
        ]
        graphs, captures = [], []
        for model, optimizer, step_type, objective in (
            (reference, optimizers[0], reference_runner.AlignmentStep, old_arm),
            (candidate, optimizers[1], AlignmentStep, arm),
        ):
            torch.manual_seed(config["dropout_seed"])
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
            graph = step_type(model, optimizer, table, spec["batch_size"], objective)
            captures.append(
                {
                    "model_weights_restored": model_digest(model)
                    == provenance["initial_model_sha256"],
                    "cpu_rng_restored": torch.equal(torch.get_rng_state(), cpu_rng),
                    "cuda_rng_restored": torch.equal(torch.cuda.get_rng_state(device), cuda_rng),
                    "optimizer_state_zero": all(
                        bool(torch.count_nonzero(value) == 0)
                        for state in optimizer.state.values()
                        for value in state.values()
                        if torch.is_tensor(value)
                    ),
                }
            )
            graphs.append(graph)
        findings["graph_capture"] = captures
        stream = MergedBatchStream(stream.size, spec["stream_seed"])
        graph_steps = config.get("compatibility_steps", 3)
        if graph_steps < 1:
            raise ValueError("At least one graph continuation step must be compared")
        updates = []
        for step in range(1, graph_steps + 1):
            indices = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
            step_rng = torch.cuda.get_rng_state(device)
            for graph, lr in zip(graphs, lrs, strict=True):
                torch.cuda.set_rng_state(step_rng, device)
                lr.fill_(learning_rate(spec, step))
                graph(indices)
                torch.cuda.synchronize()
            updates.append(
                {
                    "step": step,
                    "loss_exact": torch.equal(graphs[0].loss, graphs[1].loss),
                    "gradient_norm_exact": torch.equal(graphs[0].grad_norm, graphs[1].grad_norm),
                    "model_weights_exact": model_digest(reference) == model_digest(candidate),
                    "optimizer_state_exact": _optimizer_states_exact(*optimizers),
                }
            )
        findings["graph_updates"] = updates
        passed = (
            findings["initial_weights_exact"]
            and findings["metadata_exact"]
            and findings["table_and_selection_exact"]
            and streams_exact
            and findings["eager"]["passed"]
            and all(all(record.values()) for record in captures)
            and all(all(record.values()) for record in updates)
        )
        result = {
            "passed": passed,
            "engineering_only": True,
            "findings": findings,
            "reference_runner": str(reference_path),
            "reference_runner_sha256": digest(reference_path),
            "source_checkpoint_sha256": reference_provenance["source_checkpoint_sha256"],
            "config_sha256": digest(config_path),
            "checked_utc": utc(),
            "seconds": time.perf_counter() - started,
            "scope": "exact disabled-new versus frozen-old forward, loss, gradients, data stream, "
            "CUDA graph restoration and optimizer continuation; not new scientific training",
        }
    except BaseException as error:
        result = {
            "passed": False,
            "engineering_only": True,
            "checked_utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "findings": findings,
        }
    write_json(destination, result)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


def run(config, arm_name, out, smoke, config_path):
    out.mkdir(parents=True, exist_ok=False)
    try:
        _run(config, arm_name, out, smoke, config_path)
    except BaseException as error:
        failure = {
            "state": "failed",
            "updated_utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        write_json(out / "failure.json", failure)
        write_json(out / "status.json", failure)
        raise


def _run(config, arm_name, out, smoke, config_path):
    spec, arm = dict(config["spec"]), config["arms"][arm_name]
    validate_arm(arm)
    if smoke:
        spec["steps"] = config.get("smoke_steps", 32)
    nodes = [0, spec["steps"]] if smoke else config["evaluation_nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != spec["steps"]:
        raise ValueError("Evaluation nodes must include zero and the exact fixed endpoint")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    write_json(out / "status.json", {"state": "initializing", "updated_utc": utc(), "step": 0})
    model, metadata, provenance = initialize(config, device, arm)
    input_root = Path(config["input_root"])
    world = dict(np.load(input_root / "data/world.npz"))
    panels = dict(np.load(input_root / "data/panels.npz"))
    if set(panels) != set(PANELS):
        raise ValueError("Source must contain the seven frozen panels")
    table, selected = make_table(world, metadata, spec, device)
    stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    counts = np.zeros(stream.size, dtype=np.int64)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    diagnostic_stream = MergedBatchStream(stream.size, spec["stream_seed"])
    diagnostic_indices = torch.as_tensor(
        diagnostic_stream.batch(spec["batch_size"])[0], device=device
    )
    diagnostic = gradient_diagnostic(model, table, diagnostic_indices, arm, device)
    write_json(out / "gradient-diagnostic.json", diagnostic)
    graph_started = time.perf_counter()
    graph = AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    capture_seconds = time.perf_counter() - graph_started
    if model_digest(model) != provenance["initial_model_sha256"]:
        raise AssertionError("CUDA graph capture changed step-zero weights")
    parameters = sum(p.numel() for p in model.parameters())
    flop_step = matmul_flops(
        ModelConfig(
            metadata["vocab_size"],
            model.config.n_embd,
            model.config.n_layer * model.repeats,
            model.config.n_head,
            1024,
        ),
        spec["batch_size"],
        sequence=4,
        output_positions=2,
    )
    extra_projection_flops = extra_projection_training_flops(
        spec, model.config.n_embd, metadata["vocab_size"], arm
    )
    extra_kv_flops = extra_attention_training_flops(spec, model.config.n_embd, arm)
    extra_discoloop_flops = extra_discoloop_training_flops(
        spec, model.config.n_embd, metadata["vocab_size"], arm
    )
    run_metadata = {
        "batch": config["batch"],
        "arm": arm_name,
        "objective": arm,
        "spec": spec,
        "config_sha256": digest(config_path),
        "engineering_only": smoke,
        "parameters": parameters,
        "active_parameters": parameters,
        "vocab_size": metadata["vocab_size"],
        "world_sha256": metadata["world_sha256"],
        "panels_sha256": metadata["panels_sha256"],
        "atomic_examples": len(world["atoms"]),
        "composition_examples": len(selected),
        "evaluation_scope": "seven original frozen panels; full training pool not evaluated",
        "evaluation_batch_size": config.get("evaluation_batch_size", 256),
        "auxiliary_supervision": "Original answer and EOS slots only; p(original label) detached; "
        "mean over nonignored original labels; no bridge labels or self-predicted targets",
        "kv_semantics": "Same-layer first-round key/value at j=i plus current-round causal "
        "prefix j<=i; one shared softmax; no detach; reset each forward",
        "estimated_matmul_training_flops_per_step": {
            "base": flop_step,
            "auxiliary_projection": extra_projection_flops,
            "kv_attention": extra_kv_flops,
            "discoloop_channel": extra_discoloop_flops,
        },
        "estimated_additional_inference_matmul_flops_per_sequence": (
            extra_kv_flops + extra_discoloop_flops
        )
        / (3 * spec["batch_size"]),
        "dtype": "bf16 AMP; float32 model and Adam state",
        "pid": os.getpid(),
        "hostname": platform.node(),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "started_utc": utc(),
        "capture_seconds": capture_seconds,
        "flops_note": "Approximate matrix-multiply training FLOPs; auxiliary estimate excludes "
        "softmax, norms, masking and reductions. Auxiliary projections cover only two original "
        "supervised slots. Same-position KV uses dense 2T attention with no extra projections "
        "or parameters; masked products are counted. DiscoLoop adds two differentiable "
        "vocabulary projections at every nonfinal round boundary and all input positions. "
        "Cache concatenation and memory traffic excluded; executed padded slots included.",
        **provenance,
    }
    write_json(out / "run.json", run_metadata)
    history, training_seconds, evaluation_seconds, step = [], 0.0, 0.0, 0

    def checkpoint(final=False):
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "step": step,
            "spec": spec,
            "arm": arm,
            "config": config,
            "arm_name": arm_name,
            "architecture": architecture_flags(arm),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
        }
        torch.save(saved, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        if final:
            torch.save(
                {
                    "model": model.state_dict(),
                    "spec": spec,
                    "step": step,
                    "arm_name": arm_name,
                    "arm": arm,
                    "architecture": architecture_flags(arm),
                    "model_sha256": model_digest(model),
                },
                out / f"checkpoint-{step:07d}.pt",
            )
        np.savez_compressed(out / "exposure.npz", counts=counts, selected=selected)

    def measure(values=None):
        nonlocal evaluation_seconds
        torch.cuda.synchronize()
        before = time.perf_counter()
        metrics = {}
        for name in PANELS:
            metrics[name], prediction = evaluate_rows(
                model, panels[name], metadata, device, config.get("evaluation_batch_size", 256)
            )
            np.savez_compressed(out / f"predictions-{step:07d}-{name}.npz", **prediction)
        torch.cuda.synchronize()
        evaluation_seconds += time.perf_counter() - before
        record = {
            "step": step,
            "created_utc": utc(),
            "metrics": metrics,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "wall_seconds": time.perf_counter() - started,
            "estimated_base_matmul_training_flops": step * flop_step,
            "estimated_extra_projection_flops": step * extra_projection_flops,
            "estimated_extra_kv_attention_flops": step * extra_kv_flops,
            "estimated_extra_discoloop_flops": step * extra_discoloop_flops,
            "estimated_matmul_training_flops": step
            * (flop_step + extra_projection_flops + extra_kv_flops + extra_discoloop_flops),
            **exposure_summary(counts, len(world["atoms"]), step, spec["batch_size"]),
            **(values or {}),
        }
        history.append(record)
        write_json(out / "learning.json", history)
        checkpoint(final=step in config.get("checkpoint_nodes", []) or step == spec["steps"])
        return record

    measure()
    values = {}
    with (out / "progress.jsonl").open("x") as progress:
        for end in nodes[1:]:
            while step < end:
                heartbeat = min(
                    end,
                    ((step // config.get("heartbeat_steps", 1000)) + 1)
                    * config.get("heartbeat_steps", 1000),
                )
                torch.cuda.synchronize()
                segment = time.perf_counter()
                while step < heartbeat:
                    n = min(128, heartbeat - step)
                    indices = np.stack([stream.batch(spec["batch_size"])[0] for _ in range(n)])
                    counts += np.bincount(indices.ravel(), minlength=len(counts) + 1)[: len(counts)]
                    gpu_indices = torch.as_tensor(indices, device=device)
                    for offset in range(n):
                        lr.fill_(learning_rate(spec, step + offset + 1))
                        graph(gpu_indices[offset])
                    step += n
                torch.cuda.synchronize()
                training_seconds += time.perf_counter() - segment
                values = graph_values(graph, arm["coefficient"])
                status = {
                    "state": "running",
                    "arm": arm_name,
                    "step": step,
                    "budget": spec["steps"],
                    "training_seconds": training_seconds,
                    "updated_utc": utc(),
                    **values,
                }
                progress.write(json.dumps(status, allow_nan=False) + "\n")
                progress.flush()
                write_json(out / "status.json", status)
                print(json.dumps(status, allow_nan=False), flush=True)
            measure(values)
    final_hash = model_digest(model)
    if final_hash == provenance["initial_model_sha256"]:
        raise AssertionError("No parameter update occurred")
    final = {
        "state": "complete",
        "step": step,
        "budget": spec["steps"],
        "arm": arm_name,
        "engineering_only": smoke,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "updated_utc": utc(),
        "endpoint": history[-1],
        "model_sha256": final_hash,
        "final_checkpoint_sha256": digest(out / f"checkpoint-{step:07d}.pt"),
        "initial_model_sha256": provenance["initial_model_sha256"],
        "architecture": architecture_flags(arm),
        "independent_reload": "pending",
        **values,
    }
    write_json(out / "complete.json", final)
    write_json(out / "status.json", final)
    print(json.dumps({key: value for key, value in final.items() if key != "endpoint"}), flush=True)


def audit(config, arm_name, out, config_path):
    if (out / "audit.json").exists():
        raise FileExistsError("Independent audit already exists; preserve original audit")
    started = time.perf_counter()
    findings = {}
    try:
        run_metadata = json.loads((out / "run.json").read_text())
        if run_metadata["config_sha256"] != digest(config_path):
            raise AssertionError(
                "Audit configuration differs from the frozen training configuration"
            )
        complete = json.loads((out / "complete.json").read_text())
        endpoint = complete["endpoint"]
        if complete["arm"] != arm_name or run_metadata["objective"] != config["arms"][arm_name]:
            raise AssertionError("Audit arm configuration mismatch")
        checkpoint = out / f"checkpoint-{complete['step']:07d}.pt"
        findings["checkpoint_sha256_matches"] = (
            digest(checkpoint) == complete["final_checkpoint_sha256"]
        )
        model, metadata, _ = initialize(config, torch.device("cuda:0"), config["arms"][arm_name])
        saved = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        if (
            saved["arm"] != config["arms"][arm_name]
            or saved["spec"] != run_metadata["spec"]
            or saved["step"] != complete["step"]
            or saved["architecture"] != architecture_flags(config["arms"][arm_name])
            or run_metadata["architecture"] != saved["architecture"]
            or complete["architecture"] != saved["architecture"]
        ):
            raise AssertionError("Checkpoint architecture/objective flags or budget mismatch")
        findings["architecture"] = architecture_flags(config["arms"][arm_name])
        model.load_state_dict(saved["model"], strict=True)
        del saved
        findings["model_sha256_matches"] = model_digest(model) == complete["model_sha256"]
        panels = dict(np.load(Path(config["input_root"]) / "data/panels.npz"))
        atol, rtol = config.get("audit_loss_atol", 1e-6), config.get("audit_loss_rtol", 1e-6)
        findings["panels"] = {}
        for name in PANELS:
            metrics, prediction = evaluate_rows(
                model, panels[name], metadata, "cuda:0", run_metadata["evaluation_batch_size"]
            )
            with np.load(out / f"predictions-{complete['step']:07d}-{name}.npz") as original:
                rows_equal = np.array_equal(original["rows"], panels[name])
                answer_equal = np.array_equal(original["answer"], prediction["answer"])
                stop_equal = np.array_equal(original["stop"], prediction["stop"])
                loss_close = np.allclose(original["nll"], prediction["nll"], atol=atol, rtol=rtol)
                max_loss_delta = float(np.abs(original["nll"] - prediction["nll"]).max())
            scalar_matches = all(
                bool(np.isclose(value, endpoint["metrics"][name][key], atol=atol, rtol=rtol))
                for key, value in metrics.items()
            )
            findings["panels"][name] = {
                "n": len(panels[name]),
                "rows_exact": rows_equal,
                "answer_ids_exact": answer_equal,
                "eos_ids_exact": stop_equal,
                "loss_within_tolerance": bool(loss_close),
                "max_absolute_loss_difference": max_loss_delta,
                "scalar_metrics_match": scalar_matches,
                "passed": rows_equal
                and answer_equal
                and stop_equal
                and bool(loss_close)
                and scalar_matches,
                "metrics": metrics,
            }
        with np.load(out / "exposure.npz") as exposure:
            observed = exposure_summary(
                exposure["counts"],
                run_metadata["atomic_examples"],
                complete["step"],
                run_metadata["spec"]["batch_size"],
            )
            findings["exposure_counts_match"] = all(
                observed[key] == endpoint[key] for key in observed
            )
        passed = (
            findings["checkpoint_sha256_matches"]
            and findings["model_sha256_matches"]
            and findings["exposure_counts_match"]
            and all(panel["passed"] for panel in findings["panels"].values())
        )
        result = {
            "passed": passed,
            "arm": arm_name,
            "step": complete["step"],
            "process_id": os.getpid(),
            "audited_utc": utc(),
            "seconds": time.perf_counter() - started,
            "loss_atol": atol,
            "loss_rtol": rtol,
            "findings": findings,
            "scope": "fresh process reload; exact answer/EOS token IDs on all seven panels",
        }
    except BaseException as error:
        result = {
            "passed": False,
            "audited_utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "findings": findings,
        }
    write_json(out / "audit.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "smoke", "audit", "compatibility"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm")
    parser.add_argument("--smoke", action="store_true", help="Audit the engineering smoke output")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.command != "compatibility" and args.arm not in config["arms"]:
        parser.error("Arm is absent from frozen configuration")
    if args.command == "compatibility" and args.arm is not None:
        parser.error("Compatibility compares only the reused disabled baseline; omit --arm")
    if args.smoke and args.command != "audit":
        parser.error("--smoke is only used with audit")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.command == "compatibility":
        baseline_compatibility(config, args.base, args.config)
        return
    smoke = args.command == "smoke" or args.smoke
    out = args.base / "smoke" / args.arm if smoke else args.base / args.arm
    if args.command == "audit":
        audit(config, args.arm, out, args.config)
    else:
        run(config, args.arm, out, smoke, args.config)


if __name__ == "__main__":
    main()
