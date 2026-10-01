"""Frozen loop comparison trainer with stratified exposure and executed-depth FLOPs."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .bios_model import ModelConfig
from .grok_depth import GraphStep, make_optimizer, utc, write_json
from .grok_loop_data import StratifiedStream
from .grok_loop_model import LoopGPT, flops
from .grok_multihop import autonomous_calls, evaluate_rows, pack_rows


def run(spec, world, out, device="cuda:0", resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(spec["initialization"])
    torch.cuda.manual_seed_all(spec["initialization"])
    device = torch.device(device)
    torch.cuda.set_device(device)
    padded_length = spec["hops"] + 2
    cfg = ModelConfig(
        vocab_size=2 + spec["entities"] + spec["relations"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    model = LoopGPT(
        cfg,
        repeats=spec["repeats"],
        dropout=spec["dropout"],
        initialization=spec["init_scheme"],
    ).to(device)
    lr = torch.tensor(spec["lr"], dtype=torch.float32, device=device)
    opt = make_optimizer(model, lr, spec["weight_decay"])
    atoms = pack_rows(world["atomic"], padded_length)
    comps = pack_rows(world["train_composite"], padded_length)
    table = tuple(
        torch.as_tensor(np.concatenate([a, c]), device=device)
        for a, c in zip(atoms, comps, strict=True)
    )
    stream = StratifiedStream(
        len(world["atomic"]),
        len(world["train_composite"]),
        spec["batch_size"],
        spec["n_atomic_per_batch"],
        spec["stream_seed"],
    )
    counts = {"atomic": 0, "composite": 0}
    elapsed_training, elapsed_eval = 0.0, 0.0
    start_step = 0
    checkpoint = out / "latest.pt"
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state["spec"] != spec:
            raise ValueError("Resume spec differs from saved state")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        lr = opt.param_groups[0]["lr"]
        for group in opt.param_groups:
            group["lr"] = lr
        stream.load_state_dict(state["stream"])
        counts = state["counts"]
        elapsed_training = state["elapsed_training"]
        elapsed_eval = state["elapsed_eval"]
        start_step = state["step"]
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    started = time.perf_counter()
    graph = GraphStep(model, opt, table, spec["batch_size"])
    capture_seconds = time.perf_counter() - started
    names = (
        "atomic",
        "id_atomic",
        "ood_atomic",
        "train_composite",
        "test_composite",
        "ood_composite",
    )
    nparams = sum(p.numel() for p in model.parameters())
    flop_step = flops(
        cfg, spec["repeats"], spec["batch_size"], sequence=padded_length, output_positions=2
    )
    rows_log = []
    if (out / "learning.json").exists():
        rows_log = json.loads((out / "learning.json").read_text())
    if resume:
        rows_log = list({r["step"]: r for r in rows_log if r["step"] <= start_step}.values())
        rows_log.sort(key=lambda row: row["step"])

    def measure(step, last_loss=None):
        nonlocal elapsed_eval
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        row = {
            "step": step,
            "utc": utc(),
            "hops": spec["hops"],
            "parameters": nparams,
            "unique_layers": spec["layers"],
            "repeats": spec["repeats"],
            "effective_depth": spec["layers"] * spec["repeats"],
            "training_seconds": elapsed_training,
            "capture_seconds": capture_seconds,
            "examples": step * spec["batch_size"],
            "counts": counts.copy(),
            "effective_input_tokens": counts["atomic"] * 3 + counts["composite"] * padded_length,
            "supervised_tokens": step * spec["batch_size"] * 2,
            "estimated_training_flops": step * flop_step,
            "last_batch_loss": last_loss,
        }
        predictions = {}
        selected_names = names + (("test_full_composite",) if step == spec["steps"] else ())
        for name in selected_names:
            row[name], pred = evaluate_rows(model, world[name], device, padded_length)
            predictions.update({name + "_" + key: value for key, value in pred.items()})
        row["autonomous_calls"], calls_pred = autonomous_calls(
            model, world["test_composite"], world, device, padded_length
        )
        predictions.update({"autonomous_" + key: value for key, value in calls_pred.items()})
        torch.cuda.synchronize()
        elapsed_eval += time.perf_counter() - t0
        row["evaluation_seconds"] = elapsed_eval
        rows_log.append(row)
        write_json(out / "learning.json", rows_log)
        np.savez_compressed(out / f"predictions-{step:07d}.npz", **predictions)
        state = {
            "step": step,
            "spec": spec,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "elapsed_training": elapsed_training,
            "elapsed_eval": elapsed_eval,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(checkpoint)
        if step in spec["weight_nodes"]:
            torch.save(
                {"spec": spec, "step": step, "model": model.state_dict()},
                out / f"weights-{step:07d}.pt",
            )
        status = {
            "state": "complete" if step == spec["steps"] else "running",
            "step": step,
            "budget": spec["steps"],
            "updated_utc": utc(),
            "hops": spec["hops"],
            "layers": spec["layers"],
            "atomic": row["atomic"]["accuracy"],
            "train": row["train_composite"]["accuracy"],
            "test_probe": row["test_composite"]["accuracy"],
        }
        write_json(out / "status.json", status)
        print(json.dumps(status), flush=True)
        return row

    if not resume:
        measure(0)
    if start_step < spec["steps"]:
        for end in (s for s in spec["nodes"] if s > start_step):
            last_loss = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step = start_step
            while step < end:
                n = min(512, end - step)
                ix = np.stack([stream.take() for _ in range(n)])
                atom_count = int((ix < len(world["atomic"])).sum())
                counts["atomic"] += atom_count
                counts["composite"] += ix.size - atom_count
                gpu_ix = torch.as_tensor(ix, device=device)
                for j in range(n):
                    lr.fill_(spec["lr"] * min(1.0, (step + j + 1) / spec["warmup"]))
                    last_loss = graph(gpu_ix[j])
                step += n
            torch.cuda.synchronize()
            elapsed_training += time.perf_counter() - t0
            loss_value = float(last_loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Nonfinite loss at step {end}")
            measure(end, loss_value)
            start_step = end
    if not rows_log or rows_log[-1]["step"] != spec["steps"]:
        raise ValueError("Evaluation nodes did not reach the fixed budget")
    write_json(
        out / "complete.json",
        {
            "finished_utc": utc(),
            "spec": spec,
            "parameters": nparams,
            "unique_layers": spec["layers"],
            "repeats": spec["repeats"],
            "effective_depth": spec["layers"] * spec["repeats"],
            "training_seconds": elapsed_training,
            "evaluation_seconds": elapsed_eval,
            "endpoint": rows_log[-1],
        },
    )
    return rows_log[-1]
