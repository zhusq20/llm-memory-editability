"""Stable training and nested fact loads in ordinary GPT/Loop architectures.

The historical storage_composition implementation is reused without changing its
frozen sources, truth, tokenizer, architecture or full-generation score.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .grok_loop_model import LoopGPT, flops
from .storage_composition import (
    ARCHITECTURES,
    FullTokenStep,
    construct,
    data_digest,
    evaluate,
    file_hash,
    pack_sentences,
)
from .storage_composition import (
    SOURCE_FILES as HISTORICAL_SOURCE_FILES,
)
from .storage_composition import (
    build_world as historical_world,
)

SOURCE_FILES = [
    *HISTORICAL_SOURCE_FILES,
    "src/llm_memory_editability/storage_frontier.py",
    "scripts/run_storage_frontier.py",
    "tests/test_storage_frontier.py",
]


def learning_rate(spec, step):
    """One-based update; paired schedules share the exact same warmup."""
    if not 1 <= step <= spec["steps"]:
        raise ValueError("Step must be within the training budget")
    warmup = spec["warmup"]
    if warmup and step <= warmup:
        return spec["lr"] * step / warmup
    if spec["schedule"] == "constant":
        return spec["lr"]
    if spec["schedule"] != "cosine":
        raise ValueError("Unknown learning-rate schedule")
    fraction = (step - warmup) / (spec["steps"] - warmup)
    floor = spec["min_lr_ratio"]
    return spec["lr"] * (floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2)


def build_world(spec):
    world = historical_world(spec)
    if "extra_count" in spec:
        # A world-specific, fixed permutation defines nested distractor fact sets.
        # Common facts and every combination example remain bit-identical.
        count = spec["extra_count"]
        if not 0 < count <= len(world["extra_atomic"]):
            raise ValueError("Extra fact count exceeds the fixed master graph")
        rng = np.random.default_rng(spec["world"] + 900000)
        order = rng.permutation(len(world["extra_atomic"]))
        world["extra_atomic"] = world["extra_atomic"][order[:count]]
    return world


def strata_for(spec, world):
    extra = world["extra_atomic"] if spec["load"] == "high" else world["anchor_atomic"]
    return [world["common_atomic"], world["train_composite"], extra]


def run_name(spec):
    load = spec["load"]
    if "extra_count" in spec:
        load = f"f{spec['extra_count'] + 8 * (spec['familiar_n'] + spec['strict_n'])}"
    return (
        f"w{spec['world']}-i{spec['initialization']}-{spec['architecture']}"
        f"-d{spec['width']}-{load}-{spec['schedule']}-s{spec['steps']}"
    )


def train(spec, out, source, device="cuda:0"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() or (out / "complete.json").exists():
        raise FileExistsError(f"Do not overwrite an existing training attempt: {out}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(device)
    torch.cuda.set_device(device)
    world = build_world(spec)
    np.savez_compressed(out / "world.npz", **world)
    digest = data_digest(world)
    write_json(out / "spec.json", {"spec": spec, "source": source, "data_sha256": digest})
    model = construct(spec, device)
    initial_sha = file_hash(out / "spec.json")
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    strata = strata_for(spec, world)
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device)
        for parts in zip(*(pack_sentences(rows) for rows in strata), strict=True)
    )
    sizes = [len(rows) for rows in strata]
    offsets = np.cumsum([0, *sizes[:-1]])
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    batch = spec["batch_size"]
    if batch % 3:
        raise ValueError("Batch must divide into three equal streams")
    graph = FullTokenStep(model, optimizer, table, batch)
    layers, repeats = ARCHITECTURES[spec["architecture"]]
    flop_step = flops(model.config, repeats, batch, 9, output_positions=9)
    history, training_seconds, start = [], 0.0, 0

    def measure(step, loss=None):
        metrics, predictions = evaluate(model, world, spec["load"], device)
        row = {
            "step": step,
            "lr": learning_rate(spec, step) if step else 0,
            "metrics": metrics,
            "loss": loss,
            "training_seconds": training_seconds,
            "estimated_training_flops": step * flop_step,
        }
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        state = {
            "spec": spec,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "streams": [stream.state_dict() for stream in streams],
            "counts": counts,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "source": source,
            "data_sha256": digest,
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        write_json(out / "status.json", {"step": step, "budget": spec["steps"]})
        print(json.dumps({"run": run_name(spec), "step": step, "metrics": metrics}), flush=True)
        return row

    measure(0)
    for end in spec["nodes"]:
        if not end:
            continue
        torch.cuda.synchronize()
        started = time.perf_counter()
        while start < end:
            n = min(128, end - start)
            indices = []
            for i, stream in enumerate(streams):
                drawn = stream.take(n * batch // 3).reshape(n, batch // 3)
                counts[i] += np.bincount(drawn.ravel(), minlength=sizes[i])
                indices.append(drawn + offsets[i])
            indices = torch.as_tensor(np.concatenate(indices, axis=1), device=device)
            for j in range(n):
                lr.fill_(learning_rate(spec, start + j + 1))
                last_loss = graph(indices[j])
            start += n
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        loss = float(last_loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at {end}")
        endpoint = measure(end, loss)
    assert start == spec["steps"]
    repeat_metrics = {}
    if isinstance(model, LoopGPT):
        for repeat in spec["test_repeats"]:
            metrics, predictions = evaluate(model, world, spec["load"], device, repeat)
            repeat_metrics[str(repeat)] = metrics
            np.savez_compressed(out / f"repeat-{repeat:02d}.npz", **predictions)
    np.savez_compressed(out / "exposures.npz", **{f"stratum{i}": c for i, c in enumerate(counts)})
    result = {
        "spec": spec,
        "finished_utc": utc(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "unique_blocks": layers,
        "executed_blocks": layers * repeats,
        "independent_facts": len(strata[0]) + len(strata[2]),
        "training_seconds": training_seconds,
        "endpoint": endpoint,
        "repeat_metrics": repeat_metrics,
        "source": source,
        "data_sha256": digest,
        "spec_sha256": initial_sha,
        "checkpoint_sha256": file_hash(out / "latest.pt"),
        "supervised_tokens": start * batch * 23 // 3,
        "padded_input_tokens": start * batch * 9,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "tf32": True,
        },
    }
    write_json(out / "complete.json", result)
    return result


def audit(out, device="cuda:0"):
    """Independent-process reload: tokens exact, floating NLL tolerance recorded."""
    out = Path(out)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    complete = json.loads((out / "complete.json").read_text())
    saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    assert saved["step"] == complete["spec"]["steps"]
    assert file_hash(out / "latest.pt") == complete["checkpoint_sha256"]
    world = build_world(saved["spec"])
    assert data_digest(world) == saved["data_sha256"] == complete["data_sha256"]
    model = construct(saved["spec"], device)
    model.load_state_dict(saved["model"])
    metrics, predictions = evaluate(model, world, saved["spec"]["load"], device)
    expected = np.load(out / f"predictions-{saved['step']:06d}.npz")
    max_nll_difference = 0.0
    for key, value in predictions.items():
        if key.endswith("answer_nll"):
            max_nll_difference = max(max_nll_difference, float(np.abs(value - expected[key]).max()))
            np.testing.assert_allclose(value, expected[key], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(value, expected[key])
    for task, values in metrics.items():
        for key, value in values.items():
            reference = complete["endpoint"]["metrics"][task][key]
            if key == "answer_nll":
                np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-5)
            else:
                assert value == reference
    exposures = np.load(out / "exposures.npz")
    for i in range(3):
        np.testing.assert_array_equal(exposures[f"stratum{i}"], saved["counts"][i])
        assert exposures[f"stratum{i}"].sum() == saved["step"] * saved["spec"]["batch_size"] // 3
        assert np.ptp(exposures[f"stratum{i}"]) <= 1
    result = {
        "passed": True,
        "step": saved["step"],
        "prediction_arrays": len(predictions),
        "max_nll_difference": max_nll_difference,
        "nll_rtol": 1e-5,
        "nll_atol": 1e-5,
        "tokens_exact": True,
        "utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result
