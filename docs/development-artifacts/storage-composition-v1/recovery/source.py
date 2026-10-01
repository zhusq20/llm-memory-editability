"""Exploratory paired continuation after one frozen endpoint lost training accuracy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import EpochStream, make_optimizer, utc, write_json
from llm_memory_editability.storage_composition import (
    SOURCE_FILES,
    FullTokenStep,
    audit_run,
    build_world,
    construct,
    data_digest,
    evaluate,
    file_hash,
    pack_sentences,
)


def main(load):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:2")
    name = f"w720003-standard2-{load}"
    parent = Path("results/storage-composition-v1/confirmation") / name / "latest.pt"
    out = Path("results/storage-composition-v1/recovery") / name
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "complete.json").exists()
    state = torch.load(parent, map_location=device, weights_only=False)
    spec = state["spec"].copy()
    spec.update(steps=4096, nodes=[0, 1024, 4096], lr=0.0001, warmup=0)
    source = {p: file_hash(p) for p in [*SOURCE_FILES, __file__]}
    write_json(
        out / "spec.json",
        {
            "spec": spec,
            "source": source,
            "parent_sha256": file_hash(parent),
            "exploratory": True,
            "optimizer": "retain AdamW moments and stream states; change LR only",
        },
    )
    world = build_world(spec)
    assert data_digest(world) == state["data_sha256"]
    model = construct(spec, device)
    model.load_state_dict(state["model"])
    opt = make_optimizer(model, torch.tensor(spec["lr"], device=device), spec["weight_decay"])
    opt.load_state_dict(state["optimizer"])
    lr = opt.param_groups[0]["lr"]
    lr.fill_(spec["lr"])
    for group in opt.param_groups:
        group["lr"] = lr
    extra = world["extra_atomic"] if load == "high" else world["anchor_atomic"]
    strata = [world["common_atomic"], world["train_composite"], extra]
    sizes = [len(rows) for rows in strata]
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device)
        for parts in zip(*(pack_sentences(r) for r in strata), strict=True)
    )
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    for stream, saved in zip(streams, state["streams"], strict=True):
        stream.load_state_dict(saved)
    torch.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
    graph = FullTokenStep(model, opt, table, spec["batch_size"])
    offsets = np.cumsum([0, *sizes[:-1]])
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    history, training_seconds, start = [], 0.0, 0

    def measure(step):
        metrics, predictions = evaluate(model, world, load, device)
        row = {"step": step, "metrics": metrics, "training_seconds": training_seconds}
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        torch.save(
            {
                "step": step,
                "spec": spec,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "counts": counts,
                "streams": [s.state_dict() for s in streams],
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
                "source": source,
                "data_sha256": data_digest(world),
            },
            out / "latest.pt",
        )
        print(json.dumps({"load": load, "step": step, "metrics": metrics}), flush=True)
        return row

    measure(0)
    for end in spec["nodes"][1:]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        while start < end:
            n = min(128, end - start)
            indices = []
            for i, stream in enumerate(streams):
                drawn = stream.take(n * spec["batch_size"] // 3).reshape(n, -1)
                counts[i] += np.bincount(drawn.ravel(), minlength=sizes[i])
                indices.append(drawn + offsets[i])
            index = torch.as_tensor(np.concatenate(indices, axis=1), device=device)
            for j in range(n):
                graph(index[j])
            start += n
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        endpoint = measure(end)
    np.savez_compressed(out / "exposures.npz", **{f"stratum{i}": c for i, c in enumerate(counts)})
    write_json(
        out / "complete.json",
        {
            "spec": spec,
            "finished_utc": utc(),
            "endpoint": endpoint,
            "exploratory": True,
            "parent_sha256": file_hash(parent),
            "data_sha256": data_digest(world),
            "source": source,
        },
    )
    audit_run(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load", choices=["low", "high"])
    main(parser.parse_args().load)
