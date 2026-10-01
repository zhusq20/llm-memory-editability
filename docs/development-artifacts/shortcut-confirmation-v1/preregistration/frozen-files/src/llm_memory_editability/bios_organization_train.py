"""Matched A/B/C organization training; independent of the legacy SA/AS runs.

An update contains 16 separate seven-fact causal documents and 28 isolated old
derived queries. Their supervised tokens are weighted 80:20. Each document is
a separate batch row, so attention never crosses document boundaries.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_data import (
    N_BASE,
    N_PEOPLE,
    N_QUERIES,
    RELATIONS,
    array_hash,
    load_world,
    rng_for,
    write_json,
)
from .bios_model import CausalLM, ModelConfig, matmul_flops
from .bios_organization import apply_epoch, make_documents, render_documents
from .bios_train import code_fingerprint, evaluate, precision, tensor_data

DOCUMENT_BATCH = 16
DERIVED_BATCH = 28
STEPS_PER_EPOCH = N_PEOPLE // DOCUMENT_BATCH
DEFAULT_STEPS = 14336
WARMUP_EPOCHS = 16
CHECKPOINTS = (0, 896, 1792, 3584, 7168, 10752, 14336, 21504, 28672)


def checkpoint_steps(steps):
    if steps < 1:
        raise ValueError("steps must be positive")
    return sorted({step for step in CHECKPOINTS if step <= steps} | {steps})


def derived_schedule(world_seed, steps, pool=None):
    """Condition-independent stream; exhaust each permutation before repeating."""
    rng = rng_for(world_seed, 702)
    size = steps * DERIVED_BATCH
    pool = np.arange(N_BASE, N_QUERIES) if pool is None else np.asarray(pool)
    if not len(pool) or len(np.unique(pool)) != len(pool):
        raise ValueError("Derived pool must be nonempty and unique")
    if np.any(pool < N_BASE) or np.any(pool >= N_QUERIES):
        raise ValueError("Derived pool contains non-derived queries")
    cycles = [rng.permutation(pool) for _ in range(math.ceil(size / len(pool)))]
    return np.concatenate(cycles)[:size].reshape(steps, DERIVED_BATCH)


def learning_rate(step, base_lr):
    # A constant rate within each complete epoch also matches per-fact LR exposure.
    return base_lr * min((step // STEPS_PER_EPOCH + 1) / WARMUP_EPOCHS, 1.0)


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def schedule_hash(documents, world_seed, epochs):
    digest = hashlib.sha256()
    for epoch in range(epochs):
        digest.update(np.ascontiguousarray(apply_epoch(documents, world_seed, epoch)).tobytes())
    return digest.hexdigest()


def update_flops(config):
    return matmul_flops(config, DOCUMENT_BATCH, sequence=42, output_positions=14) + matmul_flops(
        config, DERIVED_BATCH, sequence=6, output_positions=2
    )


def two_context_backward(model, documents, isolated, derived_ids, device):
    """Compute the exact per-token joint objective without retaining both graphs."""
    with precision(device):
        logits = model(documents["tokens"], documents["positions"])
        doc_loss = F.cross_entropy(logits.float().flatten(0, 1), documents["labels"].flatten())
    (0.8 * doc_loss).backward()
    with precision(device):
        logits = model(isolated["tokens"][derived_ids], isolated["positions"][derived_ids])
        derived_loss = F.cross_entropy(
            logits.float().flatten(0, 1), isolated["labels"][derived_ids].flatten()
        )
    (0.2 * derived_loss).backward()
    return doc_loss.detach(), derived_loss.detach()


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_numpy_save(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def core_sources():
    directory = Path(__file__).resolve().parent
    names = (
        "bios_data.py",
        "bios_model.py",
        "bios_train.py",
        "bios_organization.py",
        "bios_organization_train.py",
    )
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def make_config(args, world, model, documents, derived):
    steps = args.steps
    epochs = math.ceil(steps / STEPS_PER_EPOCH)
    return {
        "protocol": "v2.4",
        "dataset": "bios-organization-v1",
        "phase": "development",
        "run_kind": "full"
        if steps == DEFAULT_STEPS and (args.width, args.layers, args.heads) == (768, 8, 12)
        else "smoke",
        "condition": args.condition,
        "world_seed": world.seed,
        "seed": args.seed,
        "organization_seed": args.organization_seed,
        "steps": steps,
        "model": model.config_dict(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_initial_sha256": state_hash(model.state_dict()),
        "lr": args.lr,
        "optimizer": "AdamW",
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_steps": WARMUP_EPOCHS * STEPS_PER_EPOCH,
        "lr_rule": "constant within epoch: base_lr * min((epoch + 1) / 16, 1)",
        "documents_per_step": DOCUMENT_BATCH,
        "facts_per_document": 7,
        "derived_queries_per_step": DERIVED_BATCH,
        "document_weight": 0.8,
        "derived_weight": 0.2,
        "document_shape": {"sequence": 42, "supervised_positions": 14},
        "derived_shape": {"sequence": 6, "supervised_positions": 2},
        "context_objective": "answer+EOS only; same-document preceding facts visible causally",
        "isolated_objective": "old-world derived QA only; no supporting document",
        "cross_document_attention": False,
        "evaluation": "all 14400 isolated queries; greedy value then generated EOS",
        "steps_per_epoch": STEPS_PER_EPOCH,
        "epochs": steps / STEPS_PER_EPOCH,
        "checkpoints": checkpoint_steps(steps),
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "lengths_sha256": array_hash(world.lengths),
        "documents_sha256": array_hash(documents),
        "document_schedule_sha256": schedule_hash(documents, world.seed, epochs),
        "derived_schedule_sha256": array_hash(derived),
        "planned_atomic_presentations": steps * 112,
        "planned_derived_presentations": steps * 28,
        "planned_total_presentations": steps * 140,
        "planned_document_presentations": steps * DOCUMENT_BATCH,
        "planned_input_tokens_including_padding": steps * (16 * 42 + 28 * 6),
        "planned_supervised_tokens": steps * (16 * 14 + 28 * 2),
        "planned_train_matmul_flops_estimate": steps * update_flops(model.config),
        "flops_method": "executed dense/attention shapes; backward=2x forward; excludes "
        "elementwise operations, evaluation, padding-free reinterpretation",
        "core_source_sha256": core_sources(),
    }


def run_locked(args, out):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    world = load_world(args.world)
    documents = make_documents(world, args.condition, args.organization_seed)
    from .bios_capacity import qa_split, split_metrics

    split_mode = getattr(args, "qa_split", "all")
    train_ids, heldout_ids = qa_split(world, split_mode)
    derived = derived_schedule(world.seed, args.steps, train_ids)
    model = CausalLM(ModelConfig(world.vocab_size, args.width, args.layers, args.heads)).to(device)
    config = make_config(args, world, model, documents, derived)
    if getattr(args, "study", "legacy") == "capacity-v1":
        config["core_source_sha256"]["bios_capacity.py"] = hashlib.sha256(
            Path(__file__).with_name("bios_capacity.py").read_bytes()
        ).hexdigest()
        config.update(
            {
                "protocol": "v2.5-development",
                "run_kind": "capacity-development",
                "qa_split": split_mode,
                "qa_train_ids": train_ids.tolist(),
                "qa_heldout_ids": heldout_ids.tolist(),
                "qa_split_policy": "half within company and old-exception stratum; world seed only",
                "evaluation": "all base plus separately reported trained/heldout derived queries",
            }
        )
    config_path = out / "config.json"
    if args.resume:
        if not config_path.exists() or not (out / "resume.pt").exists():
            raise ValueError("Resume requires config.json and an atomic resume.pt checkpoint")
        original = json.loads(config_path.read_text())
        differences = [key for key, value in config.items() if original.get(key) != value]
        if differences:
            raise ValueError("Resume configuration mismatch: " + ", ".join(differences))
    elif config_path.exists() or any(out.glob("model-*.pt")) or (out / "resume.pt").exists():
        raise ValueError("Output already contains a run; use a fresh directory or --resume")
    else:
        config.update(
            {
                "world": str(Path(args.world).resolve()),
                "output": str(out.resolve()),
                "execution": {
                    "device": args.device,
                    "threads": args.threads,
                    "torch": torch.__version__,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "precision": "BF16 autocast on CUDA; FP32 weights/loss/Adam moments",
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                },
                **code_fingerprint(),
            }
        )
        write_json(config_path, config)
        atomic_numpy_save(out / "schedule.npz", documents=documents, derived=derived)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, fused=device.type == "cuda"
    )
    isolated = tensor_data(world, device)
    counts = np.zeros(N_QUERIES, dtype=np.int64)
    slot_counts = np.zeros((N_BASE, 7), dtype=np.int64)
    weighted_exposure = np.zeros(N_QUERIES, dtype=np.float64)
    timeline, losses = [], []
    train_seconds, first_step = 0.0, 0
    if args.resume:
        state = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if state["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
            raise ValueError("Resume checkpoint configuration fingerprint mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        counts, slot_counts = state["counts"], state["slot_counts"]
        weighted_exposure = state["weighted_exposure"]
        timeline, train_seconds, first_step = (
            state["timeline"],
            state["train_seconds"],
            state["step"],
        )
        resumes_path = out / "resumes.json"
        resumes = json.loads(resumes_path.read_text()) if resumes_path.exists() else []
        resumes.append({"from_step": first_step, "device": args.device, **code_fingerprint()})
        write_json(resumes_path, resumes)
        if (out / "failure.json").exists():
            os.replace(out / "failure.json", out / f"failure-before-resume-{len(resumes)}.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cached_epoch, cached_documents = None, None
    active_step = first_step
    try:
        for step in range(first_step, args.steps + 1):
            active_step = step
            if step in checkpoint_steps(args.steps) and not (args.resume and step == first_step):
                started = time.perf_counter()
                metrics, arrays = evaluate(model, isolated, world, batch_size=args.eval_batch_size)
                if getattr(args, "study", "legacy") == "capacity-v1":
                    metrics.update(split_metrics(world, arrays, train_ids, heldout_ids))
                atomic_numpy_save(
                    out / f"predictions-{step}.npz",
                    **arrays,
                    exposure=counts,
                    slot_exposure=slot_counts,
                    lr_weighted_exposure=weighted_exposure,
                )
                atomic_torch_save(
                    {
                        "model": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "config": model.config_dict(),
                        "step": step,
                    },
                    out / f"model-{step}.pt",
                )
                record = {
                    "step": step,
                    "epoch": step / STEPS_PER_EPOCH,
                    **metrics,
                    "train_seconds": train_seconds,
                    "train_matmul_flops_estimate": step * update_flops(model.config),
                    "presentations": int(counts.sum()),
                    "document_presentations": step * DOCUMENT_BATCH,
                    "base_presentations": int(counts[:N_BASE].sum()),
                    "derived_presentations": int(counts[N_BASE:].sum()),
                    "coverage_base": float((counts[:N_BASE] > 0).mean()),
                    "coverage_derived": float((counts[N_BASE:] > 0).mean()),
                    "world_tokens": int((counts * (world.lengths + 2)).sum()),
                    "input_tokens_including_padding": step * (16 * 42 + 28 * 6),
                    "supervised_tokens": step * (16 * 14 + 28 * 2),
                    "exposure_sha256": array_hash(counts),
                    "slot_exposure_sha256": array_hash(slot_counts),
                    "lr_weighted_exposure_sha256": array_hash(weighted_exposure),
                    "lr_weighted_exposure_by_relation": {
                        relation: float(weighted_exposure[world.relation == i].sum())
                        for i, relation in enumerate((*RELATIONS, "derived"))
                    },
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0,
                    "eval_and_checkpoint_seconds": time.perf_counter() - started,
                    "mean_training_loss_since_previous": float(np.mean([x[0] for x in losses]))
                    if losses
                    else None,
                    "mean_context_document_loss": float(np.mean([x[1] for x in losses]))
                    if losses
                    else None,
                    "mean_isolated_derived_loss": float(np.mean([x[2] for x in losses]))
                    if losses
                    else None,
                }
                timeline.append(record)
                write_json(out / "learning.json", timeline)
                atomic_torch_save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(device)
                        if device.type == "cuda"
                        else None,
                        "python_rng": random.getstate(),
                        "numpy_rng": np.random.get_state(),
                        "timeline": timeline,
                        "counts": counts,
                        "slot_counts": slot_counts,
                        "weighted_exposure": weighted_exposure,
                        "train_seconds": train_seconds,
                        "step": step,
                        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    },
                    out / "resume.pt",
                )
                losses = []
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            "step": step,
                            "base": metrics["base_accuracy"],
                            "derived": metrics["derived_accuracy"],
                            "train_seconds": train_seconds,
                        }
                    ),
                    flush=True,
                )
            if step == args.steps:
                break
            epoch = step // STEPS_PER_EPOCH
            if cached_epoch != epoch:
                ordered = apply_epoch(documents, world.seed, epoch)
                rendered = render_documents(world, ordered)
                # One small epoch tensor transfer; training slices never join documents.
                cached_documents = {
                    key: torch.as_tensor(rendered[key], device=device)
                    for key in ("tokens", "positions", "labels")
                }
                cached_epoch = epoch
            offset = (step % STEPS_PER_EPOCH) * DOCUMENT_BATCH
            batch = {
                key: value[offset : offset + DOCUMENT_BATCH]
                for key, value in cached_documents.items()
            }
            fact_ids = ordered[offset : offset + DOCUMENT_BATCH]
            derived_ids = torch.as_tensor(derived[step], device=device)
            lr = learning_rate(step, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            doc_loss, derived_loss = two_context_backward(
                model, batch, isolated, derived_ids, device
            )
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(doc_loss + derived_loss) or not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite document/derived loss or gradient norm")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            train_seconds += time.perf_counter() - started
            dloss, qloss = float(doc_loss), float(derived_loss)
            losses.append((0.8 * dloss + 0.2 * qloss, dloss, qloss))
            np.add.at(counts, fact_ids.ravel(), 1)
            np.add.at(counts, derived[step], 1)
            np.add.at(slot_counts, (fact_ids, np.arange(7)[None, :]), 1)
            np.add.at(weighted_exposure, fact_ids.ravel(), lr)
            np.add.at(weighted_exposure, derived[step], lr)
            if (step + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "step": step + 1,
                            "epoch": (step + 1) / STEPS_PER_EPOCH,
                            "loss": float(np.mean([x[0] for x in losses[-100:]])),
                            "train_seconds": train_seconds,
                        }
                    ),
                    flush=True,
                )
        write_json(
            out / "complete.json",
            {
                "step": args.steps,
                "status": "complete",
                "final": timeline[-1],
                "exposure_sha256": array_hash(counts),
                "slot_exposure_sha256": array_hash(slot_counts),
                "model_final_sha256": state_hash(model.state_dict()),
            },
        )
    except BaseException as error:
        write_json(
            out / "failure.json",
            {
                "step": active_step,
                "reason": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "train_seconds": train_seconds,
                "last_committed_checkpoint": timeline[-1]["step"] if timeline else None,
            },
        )
        raise


def train(args):
    checkpoint_steps(args.steps)
    if (
        getattr(args, "qa_split", "all") != "all"
        and getattr(args, "study", "legacy") != "capacity-v1"
    ):
        raise ValueError("Heldout QA requires explicit capacity study")
    if args.width < 1 or args.layers < 1 or args.heads < 1 or args.width % args.heads:
        raise ValueError("Model dimensions must be positive and width divisible by heads")
    if args.lr <= 0 or not math.isfinite(args.lr):
        raise ValueError("Learning rate must be finite and positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is training in this output directory") from error
        run_locked(args, out)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--world", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--condition", choices=("A", "B", "C"), required=True)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--organization-seed", type=int, default=0)
    result.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--device", default="cuda")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--eval-batch-size", type=int, default=512)
    result.add_argument("--width", type=int, default=768)
    result.add_argument("--layers", type=int, default=8)
    result.add_argument("--heads", type=int, default=12)
    result.add_argument("--study", choices=("legacy", "capacity-v1"), default="legacy")
    result.add_argument("--qa-split", choices=("all", "half"), default="all")
    result.add_argument("--resume", action="store_true")
    return result


def main():
    train(parser().parse_args())


if __name__ == "__main__":
    main()
