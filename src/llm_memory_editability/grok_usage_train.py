"""Paired training of composition experience versus constituent-fact repetition."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_model import ModelConfig
from .grok_depth import GraphStep, make_optimizer, utc, write_json
from .grok_loop_model import LoopGPT, flops
from .grok_multihop import autonomous_calls, evaluate_rows, pack_rows
from .grok_usage_data import UsageStream


def weighted_loss(logits, labels, weights, normalizer):
    """Each logical slot has weight one; its two atomic substitutes have .5 each."""
    losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
    return (losses.view(-1, 2).mean(1) * weights).sum() / normalizer


class UsageGraphStep(GraphStep):
    def __init__(self, model, optimizer, table, batch_size, normalizer):
        self.normalizer = normalizer
        super().__init__(model, optimizer, table, batch_size)

    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        x, positions, labels, weights = (t[self.index] for t in self.table)
        loss = weighted_loss(self.model(x, positions), labels, weights, self.normalizer)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


def make_model(spec, device):
    cfg = ModelConfig(
        vocab_size=2 + spec["entities"] + spec["relations"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    return LoopGPT(
        cfg,
        repeats=spec["repeats"],
        dropout=spec["dropout"],
        initialization=spec["init_scheme"],
    ).to(device), cfg


def run(spec, world, out, device="cuda:0", resume=False):
    out = Path(out)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(spec.get("memory_fraction", 0.025), device)
    torch.manual_seed(spec["initialization"])
    torch.cuda.manual_seed_all(spec["initialization"])
    model, cfg = make_model(spec, device)
    lr = torch.tensor(spec["lr"], dtype=torch.float32, device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    packed = [pack_rows(part["rows"], 4) for part in world["training_parts"]]
    table = tuple(
        torch.as_tensor(np.concatenate([part[j] for part in packed]), device=device)
        for j in range(3)
    ) + (torch.as_tensor(world["table_weights"], device=device, dtype=torch.float32),)
    stream = UsageStream(
        world,
        spec["stream_seed"],
        n_atomic=spec["n_atomic"],
        n_background=spec["n_background"],
        n_role=spec["n_role"],
    )
    logical_batch = spec["n_atomic"] + spec["n_background"] + spec["n_role"]
    actual_batch = logical_batch + (spec["n_role"] if spec["arm"].startswith("rep") else 0)
    exposure = np.zeros((4, len(world["atomic"])), dtype=np.int64)
    kind_counts = np.zeros(4, dtype=np.int64)
    train_seconds, evaluation_seconds, start_step = 0.0, 0.0, 0
    if resume:
        state = torch.load(out / "latest.pt", map_location=device, weights_only=False)
        if state["spec"] != spec:
            raise ValueError("Resume specification changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        stream.load_state_dict(state["stream"])
        exposure, kind_counts = state["exposure"], state["kind_counts"]
        train_seconds, evaluation_seconds = state["training_seconds"], state["evaluation_seconds"]
        start_step = state["step"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    t0 = time.perf_counter()
    graph = UsageGraphStep(model, optimizer, table, actual_batch, logical_batch)
    capture_seconds = time.perf_counter() - t0
    learning = json.loads((out / "learning.json").read_text()) if resume else []
    learning = [row for row in learning if row["step"] <= start_step]
    fact_ids = np.asarray(world["table_fact_indices"])
    kinds = np.asarray(world["table_row_kinds"])
    flop_step = flops(cfg, spec["repeats"], actual_batch, sequence=4, output_positions=2)

    def measure(step, last_loss=None):
        nonlocal evaluation_seconds
        torch.cuda.synchronize()
        before = time.perf_counter()
        metrics, predictions = {}, {}
        for name, rows in world["evaluation"].items():
            metrics[name], pred = evaluate_rows(model, rows, device, 4)
            if len(rows):
                metrics[name]["answer_probability"] = float(np.exp(-pred["nll"][:, 0]).mean())
            predictions.update({name + "_" + k: v for k, v in pred.items()})
        metrics["autonomous_calls"], pred = autonomous_calls(
            model,
            world["evaluation"]["test_composite"],
            world,
            device,
            4,
        )
        predictions.update({"autonomous_" + k: v for k, v in pred.items()})
        torch.cuda.synchronize()
        evaluation_seconds += time.perf_counter() - before
        row = {
            "step": step,
            "utc": utc(),
            "arm": spec["arm"],
            "world_seed": spec["world_seed"],
            "metrics": metrics,
            "last_batch_loss": last_loss,
            "training_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
            "capture_seconds": capture_seconds,
            "actual_examples": int(kind_counts.sum()),
            "logical_slots": step * logical_batch,
            "supervised_tokens": int(kind_counts.sum()) * 2,
            "effective_input_tokens": int((kind_counts * np.array([3, 4, 4, 3])).sum()),
            "estimated_training_flops": step * flop_step,
            "kind_counts": kind_counts.tolist(),
        }
        learning.append(row)
        write_json(out / "learning.json", learning)
        np.savez_compressed(out / f"predictions-{step:07d}.npz", **predictions)
        np.savez_compressed(out / f"exposure-{step:07d}.npz", counts=exposure, kinds=kind_counts)
        state = {
            "spec": spec,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "exposure": exposure,
            "kind_counts": kind_counts,
            "training_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        if step in spec["weight_nodes"]:
            torch.save(
                {"spec": spec, "step": step, "model": model.state_dict()},
                out / f"weights-{step:07d}.pt",
            )
        status = {
            "state": "trained" if step == spec["steps"] else "running",
            "step": step,
            "budget": spec["steps"],
            "arm": spec["arm"],
            "updated_utc": utc(),
            "test_composite": metrics["test_composite"]["accuracy"],
            "training_seconds": train_seconds,
        }
        write_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    if not resume:
        measure(0)
    for end in (node for node in spec["nodes"] if node > start_step):
        torch.cuda.synchronize()
        before = time.perf_counter()
        step = start_step
        while step < end:
            n = min(512, end - step)
            indices = np.stack([stream.take() for _ in range(n)])
            selected = indices.ravel()
            selected_kinds, selected_facts = kinds[selected], fact_ids[selected]
            kind_counts += np.bincount(selected_kinds, minlength=4)
            for kind in range(4):
                facts = selected_facts[selected_kinds == kind].ravel()
                exposure[kind] += np.bincount(facts[facts >= 0], minlength=exposure.shape[1])
            gpu_indices = torch.as_tensor(indices, device=device)
            for j in range(n):
                lr.fill_(spec["lr"] * min(1.0, (step + j + 1) / spec["warmup"]))
                loss = graph(gpu_indices[j])
            step += n
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - before
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Nonfinite loss at {end}")
        measure(end, loss_value)
        start_step = end
    if not learning or learning[-1]["step"] != spec["steps"]:
        raise ValueError("Registered nodes did not reach endpoint")
    # Reload every endpoint from disk and compare all saved per-example scores.
    saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    old = np.load(out / f"predictions-{spec['steps']:07d}.npz")
    errors, checks, largest_nll_difference = [], 0, 0.0
    for name, rows in world["evaluation"].items():
        _, pred = evaluate_rows(model, rows, device, 4)
        for key, value in pred.items():
            reference = old[name + "_" + key]
            checks += len(value)
            if key == "nll":
                difference = float(np.abs(value - reference).max()) if len(value) else 0.0
                largest_nll_difference = max(largest_nll_difference, difference)
                ok = np.allclose(value, reference, atol=1e-6, rtol=1e-6)
            else:
                ok = np.array_equal(value, reference)
            if not ok:
                errors.append(name + "_" + key)
    audit = {
        "checks": checks,
        "errors": errors,
        "max_nll_difference": largest_nll_difference,
        "method": "reload saved endpoint; all evaluation splits, identical GPU/dtype/batch shape",
    }
    write_json(out / "endpoint-audit.json", audit)
    if errors:
        raise ValueError(f"Endpoint reload failed: {errors}")
    write_json(
        out / "complete.json",
        {
            "finished_utc": utc(),
            "spec": spec,
            "endpoint": learning[-1],
            "audit": audit,
            "parameters": sum(p.numel() for p in model.parameters()),
            "maximum_allocated_bytes": torch.cuda.max_memory_allocated(device),
        },
    )
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "step": spec["steps"],
            "arm": spec["arm"],
            "utc": utc(),
        },
    )
