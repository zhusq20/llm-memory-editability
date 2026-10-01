"""Reproducible symbolic development trajectories for bioS-Work-v1.

Every evaluated answer is greedily generated without a correct-answer prefix.
Two generated tokens is the fixed symbolic grammar budget (one value plus EOS).
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_data import (
    EOS,
    N_BASE,
    N_QUERIES,
    RELATIONS,
    array_hash,
    curriculum,
    load_world,
    write_json,
)
from .bios_model import CausalLM, ModelConfig, matmul_flops


def precision(device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def tensor_data(world, device, answers=None):
    answer = torch.as_tensor(world.answers if answers is None else answers, device=device)
    prompts = torch.as_tensor(world.prompts, device=device)
    lengths = torch.as_tensor(world.lengths, device=device)
    tokens = torch.zeros((N_QUERIES, 6), dtype=torch.long, device=device)
    tokens[:, :5] = prompts
    rows = torch.arange(N_QUERIES, device=device)
    tokens[rows, lengths] = answer
    tokens[lengths == 4, 5] = EOS
    positions = torch.stack([lengths - 1, lengths], dim=1)
    labels = torch.stack([answer, torch.full_like(answer, EOS)], dim=1)
    return {
        "prompts": prompts,
        "lengths": lengths,
        "tokens": tokens,
        "positions": positions,
        "labels": labels,
    }


@torch.no_grad()
def evaluate(model, data, world, batch_size=512, answers=None):
    device = data["tokens"].device
    targets = world.answers if answers is None else answers
    predictions, ended, nll = [], [], []
    model.eval()
    for begin in range(0, N_QUERIES, batch_size):
        end = min(begin + batch_size, N_QUERIES)
        lengths = data["lengths"][begin:end]
        rows = torch.arange(end - begin, device=device)
        with precision(device):
            logits = model(data["prompts"][begin:end], (lengths - 1)[:, None])[:, 0].float()
            first = logits.argmax(-1)
            continuation = torch.zeros((end - begin, 6), dtype=torch.long, device=device)
            continuation[:, :5] = data["prompts"][begin:end]
            continuation[rows, lengths] = first
            second = model(continuation, lengths[:, None])[:, 0].argmax(-1)
        predictions.append(first.cpu().numpy())
        ended.append(second.eq(EOS).cpu().numpy())
        labels = torch.as_tensor(targets[begin:end], device=device)
        nll.append(F.cross_entropy(logits, labels, reduction="none").cpu().numpy())
    predictions, ended, nll = (
        np.concatenate(predictions),
        np.concatenate(ended),
        np.concatenate(nll),
    )
    correct = (predictions == targets) & ended
    strata = {
        "employer": world.relation == 0,
        "company_default": world.relation == 1,
        "actual_nonexception": (world.relation == 2)
        & ~world.exceptions[np.maximum(world.person, 0)],
        "actual_old_exception": (world.relation == 2)
        & world.exceptions[np.maximum(world.person, 0)],
        "independent_attributes": np.isin(world.relation, [3, 4, 5, 6]),
    }
    metrics = {
        "base_accuracy": float(correct[:N_BASE].mean()),
        "derived_accuracy": float(correct[N_BASE:].mean()),
        "value_nll": float(nll[:N_BASE].mean()),
        "nontermination_rate": float((~ended).mean()),
        "strata": {k: float(correct[v].mean()) for k, v in strata.items()},
        "learning_threshold_passed": bool(
            correct[:N_BASE].mean() >= 0.99 and correct[N_BASE:].mean() >= 0.99
        ),
    }
    return metrics, {
        "prediction": predictions,
        "ended": ended,
        "correct": correct,
        "value_nll": nll,
    }


def code_fingerprint():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src" / "llm_memory_editability").glob("*.py"))
    return {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        ),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }


def train(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    world = load_world(args.world)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    resume = torch.load(out / "resume.pt", weights_only=False) if args.resume else None
    if resume is None and (out / "config.json").exists():
        raise ValueError("Output already contains a run; use a fresh output directory")
    schedule = curriculum(world, args.order, args.steps)
    model = CausalLM(ModelConfig(world.vocab_size)).to(device)
    data = tensor_data(world, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, fused=device.type == "cuda"
    )
    checkpoints = sorted(
        {0, *range(660, 5281, 660), *range(7280, args.steps + 1, 2000), args.steps}
    )
    config = {
        **vars(args),
        "model": model.config_dict(),
        "protocol": "v2.3",
        "phase": "development",
        "world_seed": world.seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "BF16 autocast, FP32 weights/loss/Adam moments",
        "checkpoints": checkpoints,
        "schedule_sha256": array_hash(schedule),
        "truth_sha256": array_hash(world.answers),
        "flops_method": "executed-shape matmul estimate; elementwise and evaluation excluded",
        **code_fingerprint(),
    }
    timeline, anchors = [], {}
    counts = np.zeros(N_QUERIES, dtype=np.int64)
    weighted_exposure = np.zeros(8)
    train_seconds = 0.0
    losses = []
    steps_since_checkpoint = 0
    first_step = 0
    if resume is None:
        write_json(out / "config.json", config)
        np.save(out / "schedule.npy", schedule)
    else:
        # Exact continuation from the last predefined checkpoint's full training state.
        original = json.loads((out / "config.json").read_text())
        for key in ("schedule_sha256", "truth_sha256", "seed", "order", "steps", "lr"):
            if original[key] != config[key]:
                raise ValueError(f"Resume mismatch on {key}")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "cuda" and resume["cuda_rng"] is not None:
            torch.cuda.set_rng_state(resume["cuda_rng"], device)
        timeline, anchors = resume["timeline"], resume["anchors"]
        counts, weighted_exposure = resume["counts"], resume["weighted_exposure"]
        train_seconds = resume["train_seconds"]
        first_step = resume["step"]
        resumes = (
            json.loads((out / "resumes.json").read_text())
            if (out / "resumes.json").exists()
            else []
        )
        resumes.append({"from_step": first_step, **code_fingerprint()})
        write_json(out / "resumes.json", resumes)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(first_step, args.steps + 1):
        if step in checkpoints and not (resume is not None and step == first_step):
            start_eval = time.perf_counter()
            metrics, arrays = evaluate(model, data, world)
            for anchor, known in anchors.items():
                metrics[f"forgetting_since_{anchor}"] = (
                    float((~arrays["correct"][known]).mean()) if known.any() else None
                )
            if step in (2640, 5280):
                anchors[step] = arrays["correct"].copy()
            np.savez_compressed(out / f"predictions-{step}.npz", **arrays, exposure=counts)
            # CPU state for portability; all predefined checkpoints retained.
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": model.config_dict(),
                    "step": step,
                },
                out / f"model-{step}.pt",
            )
            record = {
                "step": step,
                **metrics,
                "train_seconds": train_seconds,
                "train_matmul_flops_estimate": step * matmul_flops(model.config, 128),
                "presentations": int(counts.sum()),
                "coverage_base": float((counts[:N_BASE] > 0).mean()),
                "coverage_derived": float((counts[N_BASE:] > 0).mean()),
                "world_tokens": int((counts * (world.lengths + 2)).sum()),
                "input_tokens_including_padding": step * 128 * 6,
                "supervised_tokens": step * 128 * 2,
                "lr_weighted_exposure_by_relation": dict(
                    zip((*RELATIONS, "derived"), weighted_exposure.tolist(), strict=True)
                ),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
                "eval_and_checkpoint_seconds": time.perf_counter() - start_eval,
                "mean_training_loss_since_previous": float(np.mean(losses)) if losses else None,
            }
            timeline.append(record)
            write_json(out / "learning.json", timeline)
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
            losses = []
            steps_since_checkpoint = 0
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "timeline": timeline,
                "anchors": anchors,
                "counts": counts,
                "weighted_exposure": weighted_exposure,
                "train_seconds": train_seconds,
                "step": step,
            }
            torch.save(state, out / "resume.pt.tmp")
            os.replace(out / "resume.pt.tmp", out / "resume.pt")
        if step == args.steps:
            break
        model.train()
        lr = args.lr * min((step + 1) / 2000, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ids_np = schedule[step]
        ids = torch.as_tensor(ids_np, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            logits = model(data["tokens"][ids], data["positions"][ids])
            loss = F.cross_entropy(
                logits.float().reshape(-1, world.vocab_size), data["labels"][ids].reshape(-1)
            )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            write_json(
                out / "failure.json",
                {
                    "step": step,
                    "reason": "nonfinite loss or gradient",
                    "train_seconds": train_seconds,
                },
            )
            raise FloatingPointError("Nonfinite loss or gradient")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - started
        losses.append(loss.item())
        np.add.at(counts, ids_np, 1)
        weighted_exposure += np.bincount(world.relation[ids_np], minlength=8) * lr
        steps_since_checkpoint += 1
        if (step + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "step": step + 1,
                        "loss": float(np.mean(losses[-100:])),
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
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", choices=["SA", "AS"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=13280)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
