"""Longer-path trainer reusing the historical GPT, optimizer and graph step.

The only task changes are longer input rows and the separately audited split.
Training uses the same uniform combined-set epochs and tail-plus-EOS loss.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_model import ModelConfig, matmul_flops
from .grok_depth import EpochStream, GraphStep, SmallGPT, make_optimizer, utc, write_json
from .grok_multihop_data import path_details


def pack_rows(rows, padded_length):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] > padded_length or rows.shape[1] < 3:
        raise ValueError("Rows must fit the fixed padded length and include an answer")
    x = np.zeros((len(rows), padded_length), dtype=np.int64)
    x[:, : rows.shape[1]] = rows
    positions = np.tile([rows.shape[1] - 2, rows.shape[1] - 1], (len(rows), 1))
    labels = np.c_[rows[:, -1], np.ones(len(rows), dtype=np.int64)]
    return x, positions, labels


@torch.no_grad()
def evaluate_rows(model, rows, device, padded_length, batch_size=1024):
    was_training = model.training
    model.eval()
    packed = pack_rows(rows, padded_length)
    answers, stops, losses = [], [], []
    for start in range(0, len(rows), batch_size):
        x, positions, labels = (
            torch.as_tensor(a[start : start + batch_size], device=device) for a in packed
        )
        # as_tensor shares CPU memory when device=cpu; all writes occur in a copy.
        x = x.clone()
        logits = model(x, positions)
        losses.append(
            F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
            .view(-1, 2)
            .cpu()
            .numpy()
        )
        answer = logits[:, 0].argmax(-1)
        x[torch.arange(len(x), device=device), positions[:, 1]] = answer
        stop = model(x, positions)[:, 1].argmax(-1)
        answers.append(answer.cpu().numpy())
        stops.append(stop.cpu().numpy())
    model.train(was_training)
    if not len(rows):
        return {"n": 0, "answer_accuracy": None, "accuracy": None, "nll": None}, {}
    answer, stop, nll = np.concatenate(answers), np.concatenate(stops), np.concatenate(losses)
    correct = answer == rows[:, -1]
    return {
        "n": len(rows),
        "answer_accuracy": float(correct.mean()),
        "accuracy": float((correct & (stop == 1)).mean()),
        "nll": float(nll.mean()),
    }, {"answer": answer, "stop": stop, "nll": nll, "target": rows[:, -1]}


@torch.no_grad()
def autonomous_calls(model, rows, world, device, padded_length):
    """Supply relation decomposition, but carry the model's own intermediate answers."""
    if not len(rows):
        return {"n": 0, "answer_accuracy": None, "accuracy": None}, {}
    meta = world["metadata"]
    nodes, _ = path_details(rows, world["atomic"], meta["entities"], meta["relations"])
    current = rows[:, 0].copy()
    all_eos = np.ones(len(rows), dtype=bool)
    all_correct = np.ones(len(rows), dtype=bool)
    predictions, hop_accuracy = {}, []
    for j in range(meta["hops"]):
        # Gold target only scores the answer; causality prevents answer-label access.
        queries = np.c_[current, rows[:, j + 1], nodes[:, j + 1]]
        metrics, pred = evaluate_rows(model, queries, device, padded_length)
        current = pred["answer"]
        all_eos &= pred["stop"] == 1
        all_correct &= current == nodes[:, j + 1]
        hop_accuracy.append(metrics["accuracy"])
        predictions[f"hop{j + 1}_answer"] = current
        predictions[f"hop{j + 1}_stop"] = pred["stop"]
    return {
        "n": len(rows),
        "calls": meta["hops"],
        "answer_accuracy": float((current == rows[:, -1]).mean()),
        "accuracy": float(((current == rows[:, -1]) & all_eos).mean()),
        "all_intermediate_answers_and_eos_correct": float((all_correct & all_eos).mean()),
        "hop_accuracy": hop_accuracy,
        "extra_information": "given relation decomposition, own generated bridge entities",
    }, predictions


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
    model = SmallGPT(cfg, spec["dropout"]).to(device)
    lr = torch.tensor(spec["lr"], dtype=torch.float32, device=device)
    opt = make_optimizer(model, lr, spec["weight_decay"])
    atoms = pack_rows(world["atomic"], padded_length)
    comps = pack_rows(world["train_composite"], padded_length)
    table = tuple(
        torch.as_tensor(np.concatenate([a, c]), device=device)
        for a, c in zip(atoms, comps, strict=True)
    )
    stream = EpochStream(len(table[0]), spec["stream_seed"])
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
    flop_step = matmul_flops(cfg, spec["batch_size"], sequence=padded_length, output_positions=2)
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
                ix = stream.take(n * spec["batch_size"]).reshape(n, spec["batch_size"])
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
            "training_seconds": elapsed_training,
            "evaluation_seconds": elapsed_eval,
            "endpoint": rows_log[-1],
        },
    )
    return rows_log[-1]
