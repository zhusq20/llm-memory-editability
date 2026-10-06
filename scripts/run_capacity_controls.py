"""Train atomic/replay controls and paired continuations; retain old scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from capacity_serial_recall import serial_recall
from torch.nn import functional as F

from llm_memory_editability.bios_model import ModelConfig, matmul_flops
from llm_memory_editability.capacity_controls import training_indices
from llm_memory_editability.capacity_scaling import (
    ANSWER_LENGTH,
    CONTEXT,
    VOCAB,
    digest_arrays,
    generate_world,
    information_ledger,
    model_parameter_count,
    pack,
)
from llm_memory_editability.grok_depth import SmallGPT, utc, write_json


def read(path):
    return json.loads(Path(path).read_text())


def tensor_digest(model):
    return digest_arrays({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})


def initialize(spec, device):
    torch.manual_seed(spec["initialization"])
    model = SmallGPT(
        ModelConfig(vocab_size=VOCAB, width=spec["width"], layers=2, heads=4, context=CONTEXT),
        dropout=0.0,
    ).to(device)
    assert sum(p.numel() for p in model.parameters()) == model_parameter_count(spec["width"])
    return model


def autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate_rows(model, rows, device, batch_size):
    model.eval()
    all_predictions, all_losses = [], []
    for start in range(0, len(rows), batch_size):
        x, positions, labels = [
            torch.as_tensor(a, device=device) for a in pack(rows[start : start + batch_size])
        ]
        with autocast(device):
            logits = model(x, positions=positions)
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), reduction="none"
        ).reshape(-1, ANSWER_LENGTH)
        all_losses.append(loss.cpu().numpy())
        # Erase every teacher-forced answer token before autoregressive decoding.
        for j in range(ANSWER_LENGTH - 1):
            x[torch.arange(len(x), device=device), positions[:, j + 1]] = 0
        predictions = []
        for j in range(ANSWER_LENGTH):
            with autocast(device):
                token = model(x, positions=positions[:, j : j + 1])[:, 0].argmax(-1)
            predictions.append(token.cpu().numpy())
            if j < ANSWER_LENGTH - 1:
                x[torch.arange(len(x), device=device), positions[:, j + 1]] = token
        all_predictions.append(np.stack(predictions, axis=1))
    if not len(rows):
        return np.empty((0, ANSWER_LENGTH), dtype=np.int64), np.empty((0, ANSWER_LENGTH))
    return np.concatenate(all_predictions), np.concatenate(all_losses)


def summarize_rows(rows, predictions, losses):
    labels = pack(rows)[2]
    correct = np.all(predictions == labels, axis=1)
    return {
        "n": len(rows),
        "accuracy": float(correct.mean()) if len(rows) else 0.0,
        "answer_nll": float(losses.sum(1).mean()) if len(rows) else 0.0,
        "knowledge_nll": float(losses[:, 1:5].sum(1).mean()) if len(rows) else 0.0,
        "identifier_accuracy": float(np.all(predictions[:, 1:5] == labels[:, 1:5], axis=1).mean())
        if len(rows)
        else 0.0,
    }, correct


def evaluate(model, world, spec, device, full=False):
    pools = {k: world[k] for k in ("II", "IO", "OI", "OO")}
    atom_panel = world["atomic"] if full else world["atomic_panel"]
    necessary = np.concatenate([v[:, 5:].ravel() for v in pools.values()])
    atom_ids = np.unique(np.concatenate((atom_panel[:, 5], necessary)))
    rows = world["atomic"][atom_ids]
    pred, loss = evaluate_rows(model, rows, device, spec["evaluation_batch_size"])
    _, atom_ok = summarize_rows(rows, pred, loss)
    lookup = np.zeros(len(world["atomic"]), dtype=bool)
    lookup[atom_ids] = atom_ok
    selected = np.searchsorted(atom_ids, atom_panel[:, 5])
    atom_metric, _ = summarize_rows(atom_panel, pred[selected], loss[selected])
    atom_metric["full_population"] = full
    metrics = {"atomic": atom_metric}
    arrays = {"atomic_ids": atom_ids, "atomic_predictions": pred, "atomic_losses": loss}
    if full:
        # The hub dictionary is fixed while A-address load grows. Separate these
        # populations so good B->C recall cannot hide a first-map failure.
        population = world["atomic"]
        groups = {
            "atomic_first": population[:, 0] == 0,
            "atomic_second": population[:, 0] == 1,
        }
        if "support_heads_n" in spec:
            groups["atomic_background"] = (population[:, 0] == 0) & (
                population[:, 1] >= spec["support_heads_n"]
            )
            groups["atomic_target"] = ~groups["atomic_background"]
        for name, mask in groups.items():
            metrics[name], _ = summarize_rows(population[mask], pred[mask], loss[mask])
            metrics[name]["learned_bits_raw"] = len(population[mask]) * (
                8 - metrics[name]["knowledge_nll"] / np.log(2)
            )
    pools["train_composition"] = world["train_composition_panel"]
    for name, rows in pools.items():
        pred, loss = evaluate_rows(model, rows, device, spec["evaluation_batch_size"])
        metrics[name], ok = summarize_rows(rows, pred, loss)
        if name == "train_composition":
            metrics[name]["population_n"] = len(world["train_composition"])
        arrays[name + "_predictions"], arrays[name + "_losses"] = pred, loss
        if name != "train_composition":
            both = lookup[rows[:, 5]] & lookup[rows[:, 6]]
            metrics[name].update(
                population_n=int(world[name + "_population"][0]),
                first_atomic_accuracy=float(lookup[rows[:, 5]].mean()) if len(rows) else 0.0,
                second_atomic_accuracy=float(lookup[rows[:, 6]].mean()) if len(rows) else 0.0,
                both_atomic_n=int(both.sum()),
                both_atomic_coverage=float(both.mean()) if len(rows) else 0.0,
                both_atomic_composition_accuracy=float(ok[both].mean()) if both.any() else None,
            )
            arrays[name + "_both_atomic"] = both
    if full:
        metrics["information"] = information_ledger(
            world, sum(p.numel() for p in model.parameters()), atom_metric["knowledge_nll"]
        )
    if full:
        serial_metrics, serial_arrays = serial_recall(
            model, world, spec, device, evaluate_rows, arrays
        )
        metrics.update(serial_metrics)
        arrays.update(serial_arrays)
    return metrics, arrays


def runtime_setup(device):
    torch.set_num_threads(2 if device.type == "cuda" else 1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        # Actual forward/backward/update/synchronization in the eventual image.
        torch.manual_seed(7)
        a = torch.randn(64, 64, device=device, requires_grad=True)
        with autocast(device):
            loss = (a @ a).square().mean()
        loss.backward()
        with torch.no_grad():
            a.add_(a.grad, alpha=-0.001)
        torch.cuda.synchronize(device)
        if not torch.isfinite(a).all():
            raise RuntimeError("CUDA calculation preflight failed")


def train(config, spec, out, device):
    runtime_setup(device)
    out.mkdir(parents=True, exist_ok=True)
    world = generate_world(spec)
    world_hash = digest_arrays(world)
    np.savez_compressed(out / "world.npz", **world)
    model = initialize(spec, device)
    initial_hash = tensor_digest(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.parameters() if p.ndim >= 2],
                "weight_decay": spec["weight_decay"],
            },
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=spec["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=device.type == "cuda",
    )
    parent = None
    if spec.get("parent"):
        parent_path = Path(spec["parent"])
        assert (
            hashlib.sha256((parent_path / "latest.pt").read_bytes()).hexdigest()
            == spec["parent_checkpoint_sha256"]
        )
        parent = torch.load(parent_path / "latest.pt", map_location=device, weights_only=False)
        assert parent["world_sha256"] == world_hash
        for key in (
            "world_seed",
            "initialization",
            "stream_seed",
            "width",
            "heads_n",
            "batch_size",
            "learning_rate",
        ):
            assert parent["spec"][key] == spec[key], key
        model.load_state_dict(parent["model"])
        optimizer.load_state_dict(parent["optimizer"])
        for group in optimizer.param_groups:
            group["weight_decay"] = spec["weight_decay"] if group["params"][0].ndim >= 2 else 0.0
        initial_hash = tensor_digest(model)
    metadata = {
        "spec": spec,
        "world_sha256": world_hash,
        "initial_model_sha256": initial_hash,
        "model": model.config_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": VOCAB,
        "atomic_examples": len(world["atomic"]),
        "composition_examples": len(world["train_composition"]),
        "evaluation_nodes": spec["evaluation_epochs"],
        "pid": os.getpid(),
        "gpu": int(os.environ.get("PHYSICAL_GPU", 0)),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": "BF16 AMP/FP32 parameters" if device.type == "cuda" else "FP32",
        "tracking_group": config["batch"],
        "phase": "development",
        "source_files": config.get("source_files", {}),
        "parent_step": parent["step"] if parent else 0,
        "parent_examples": parent["counters"][1] if parent else 0,
        "training_mode": spec.get("training_mode", "mixed"),
        "created_utc": utc(),
    }
    write_json(out / "run.json", metadata)
    write_json(
        out / "preflight.json",
        {"passed": True, "device": str(device), "cuda_computation": device.type == "cuda"},
    )
    rows = np.concatenate((world["atomic"], world["train_composition"]))
    table = [torch.as_tensor(a, device=device) for a in pack(rows)]
    n, batch = len(rows), spec["batch_size"]
    mode = spec.get("training_mode", "mixed")
    counts = np.zeros(n, dtype=np.int64)
    history, step, start_epoch, training_seconds, examples, flops = [], 0, 0, 0.0, 0, 0
    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        assert state["spec"] == spec and state["world_sha256"] == world_hash
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, step, start_epoch = state["history"], state["step"], state["epoch"]
        training_seconds, examples, flops = state["counters"]
        counts = state["counts"]
    start_wall = time.monotonic()
    wall_offset = history[-1]["wall_seconds"] if history else 0.0
    nodes = set(spec["evaluation_epochs"]) | {0, spec["epochs"]}
    parent_step = parent["step"] if parent else 0
    parent_epoch = parent["epoch"] if parent else 0

    def record(epoch, loss=None, evaluation=False):
        metrics = {}
        if evaluation:
            metrics, predictions = evaluate(
                model, world, spec, device, full=epoch == spec["epochs"]
            )
            np.savez_compressed(out / f"predictions-e{epoch:05d}.npz", **predictions)
        row = {
            "step": step,
            "epoch": epoch,
            "cumulative_step": parent_step + step,
            "metrics": metrics,
            "loss": loss,
            "training_seconds": training_seconds,
            "wall_seconds": wall_offset + time.monotonic() - start_wall,
            "examples": examples,
            "supervised_tokens": examples * ANSWER_LENGTH,
            "executed_input_tokens": examples * CONTEXT * 2,
            "estimated_matmul_training_flops": flops,
            "atomic_epochs": float(counts[: len(world["atomic"])].mean()),
            "composition_epochs": float(counts[len(world["atomic"]) :].mean()),
            "exposure_status": "actual per-example counters, including partial epochs",
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "created_utc": utc(),
        }
        if history and history[-1]["step"] == step:
            history[-1] = row
        else:
            history.append(row)
        write_json(out / "learning.json", history)
        write_json(out / "status.json", {"state": "running", "step": step, "epoch": epoch})
        print(json.dumps({"step": step, "epoch": epoch, "metrics": metrics}), flush=True)

    def checkpoint(epoch):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "spec": spec,
            "world_sha256": world_hash,
            "epoch": epoch,
            "step": step,
            "history": history,
            "counters": (training_seconds, examples, flops),
            "counts": counts,
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(latest)

    if not history:
        record(0, evaluation=True)
        checkpoint(0)
    for epoch in range(start_epoch + 1, spec["epochs"] + 1):
        order = training_indices(world, spec, epoch + parent_epoch)
        epoch_n = len(order)
        for start in range(0, epoch_n, batch):
            if time.monotonic() - start_wall > spec["max_wall_seconds"]:
                raise TimeoutError("Per-run wall-time budget reached; no capacity conclusion")
            model.train()
            indices = torch.as_tensor(order[start : start + batch], device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timer = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = spec["learning_rate"] * min(
                    (parent_step + step + 1) / spec["warmup_steps"], 1
                )
            x, positions, labels = [a[indices] for a in table]
            with autocast(device):
                logits = model(x, positions=positions)
                loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - timer
            step += 1
            examples += len(indices)
            np.add.at(counts, order[start : start + batch], 1)
            flops += matmul_flops(
                model.config, len(indices), CONTEXT, output_positions=ANSWER_LENGTH, backward=True
            )
            if step % spec["log_steps"] == 0 and start + batch < epoch_n:
                record(epoch - 1 + (start + len(indices)) / epoch_n, float(loss.detach()))
        if epoch in nodes:
            record(epoch, float(loss.detach()), evaluation=True)
        elif epoch % spec["checkpoint_epochs"] == 0:
            record(epoch, float(loss.detach()))
        if epoch in nodes or epoch % spec["checkpoint_epochs"] == 0:
            checkpoint(epoch)
    if mode == "mixed":
        assert np.all(counts == spec["epochs"])
    else:
        assert np.all(counts[: len(world["atomic"])] >= spec["epochs"])
        assert not counts[len(world["atomic"]) :].any()
    np.savez_compressed(
        out / "exposures.npz",
        atomic=counts[: len(world["atomic"])],
        composition=counts[len(world["atomic"]) :],
        composition_fact_roles=np.stack(
            [
                np.bincount(
                    world["train_composition"][:, column],
                    weights=counts[len(world["atomic"]) :],
                    minlength=len(world["atomic"]),
                )
                for column in (5, 6)
            ]
        ).astype(np.int64),
    )
    write_json(
        out / "trained.json",
        {
            "final_model_sha256": tensor_digest(model),
            "epochs": spec["epochs"],
            "step": step,
            "training_pid": os.getpid(),
            "created_utc": utc(),
        },
    )


def audit(config, spec, out, device):
    runtime_setup(device)
    state = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    world = generate_world(spec)
    assert state["spec"] == spec and state["epoch"] == spec["epochs"]
    assert state["world_sha256"] == digest_arrays(world)
    assert digest_arrays(dict(np.load(out / "world.npz"))) == digest_arrays(world)
    model = initialize(spec, device)
    model.load_state_dict(state["model"])
    assert tensor_digest(model) == read(out / "trained.json")["final_model_sha256"]
    metrics, arrays = evaluate(model, world, spec, device, full=True)
    original = dict(np.load(out / f"predictions-e{spec['epochs']:05d}.npz"))
    assert set(original) == set(arrays)
    max_error = 0.0
    for key in arrays:
        if "losses" in key:
            error = float(np.max(np.abs(arrays[key] - original[key]))) if arrays[key].size else 0.0
            max_error = max(max_error, error)
            assert error <= 1e-5, (key, error)
        else:
            assert np.array_equal(arrays[key], original[key]), key
    expected_counts = np.zeros(
        len(world["atomic"]) + len(world["train_composition"]), dtype=np.int64
    )
    parent_epoch = 0
    if spec.get("parent"):
        parent_epoch = read(Path(spec["parent"]) / "trained.json")["epochs"]
    for epoch in range(1, spec["epochs"] + 1):
        np.add.at(expected_counts, training_indices(world, spec, epoch + parent_epoch), 1)
    expected_examples = int(expected_counts.sum())
    assert state["counters"][1] == expected_examples
    assert np.array_equal(state["counts"], expected_counts)
    exposures = np.load(out / "exposures.npz")
    assert np.array_equal(exposures["atomic"], expected_counts[: len(world["atomic"])])
    assert np.array_equal(exposures["composition"], expected_counts[len(world["atomic"]) :])
    expected_roles = world["role_counts"] * (
        spec["epochs"] if spec.get("training_mode", "mixed") == "mixed" else 0
    )
    assert np.array_equal(exposures["composition_fact_roles"], expected_roles)
    epoch_n = len(training_indices(world, spec, 1))
    assert state["step"] == spec["epochs"] * (
        (epoch_n + spec["batch_size"] - 1) // spec["batch_size"]
    )
    meta = read(out / "run.json")
    if spec.get("reference_parent"):
        reference = Path(spec["reference_parent"])
        old_meta = read(reference / "run.json")
        assert meta["initial_model_sha256"] == old_meta["initial_model_sha256"]
        assert meta["world_sha256"] == old_meta["world_sha256"]
        if spec["training_mode"] == "atomic_replay":
            old_end = read(reference / "learning.json")[-1]
            assert state["step"] == old_end["step"]
            assert state["counters"][1] == old_end["examples"]
            assert state["counters"][2] == old_end["estimated_matmul_training_flops"]
    trained = read(out / "trained.json")
    result = {
        "passed": True,
        "world_sha256": state["world_sha256"],
        "max_nll_difference": max_error,
        "prediction_arrays": len(arrays),
        "metrics": metrics,
        "examples": expected_examples,
        "training_pid": trained["training_pid"],
        "audit_pid": os.getpid(),
        "separate_process": trained["training_pid"] != os.getpid(),
        "created_utc": utc(),
    }
    write_json(out / "audit.json", result)
    write_json(out / "status.json", {"state": "complete", "step": state["step"]})
    write_json(out / "complete.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "audit"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read(args.config)
    for filename, expected in config.get("source_files", {}).items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    spec = next(s for s in config["runs"] if s["name"] == args.run)
    out = Path(config["results_root"]) / "runs" / args.run
    try:
        {"train": train, "audit": audit}[args.action](config, spec, out, torch.device(args.device))
    except BaseException:
        write_json(out / "failure.json", {"error": traceback.format_exc(), "created_utc": utc()})
        raise


if __name__ == "__main__":
    main()
