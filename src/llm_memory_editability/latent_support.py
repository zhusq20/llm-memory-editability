"""Degree-matched support follow-up with isolated training and audit entry points.

The train/audit bodies are copied from latent_scaling.py at batch creation so
historical sources retain their frozen hashes. Architecture, optimizer, sentence
packing, evaluation and streams are reused; only build_world selects support.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .grok_loop_model import flops
from .latent_scaling import (
    SOURCE_FILES as HISTORICAL_SOURCES,
)
from .latent_scaling import (
    build_world as scaling_world,
)
from .latent_scaling import (
    compare_metrics,
    compare_predictions,
    construct,
    model_digest,
    role_coverage,
)
from .latent_scaling import (
    run_name as scaling_run_name,
)
from .storage_composition import (
    FullTokenStep,
    audit_world,
    data_digest,
    evaluate,
    file_hash,
    pack_sentences,
)
from .storage_frontier import learning_rate

SOURCE_FILES = [*HISTORICAL_SOURCES, "src/llm_memory_editability/latent_support.py"]


def run_name(spec):
    return scaling_run_name(spec) + "-support-" + spec["support"]


def build_world(spec):
    """Select frozen legal edges; never use model scores or test pairs to choose."""
    if spec.get("support") not in ("connected", "split"):
        raise ValueError("Unknown frozen support arm")
    indices = spec.get("composition_indices", [])
    if (
        len(indices) != 512
        or any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
        or len(set(indices)) != 512
        or indices != sorted(indices)
        or spec.get("composition_count") != 512
    ):
        raise ValueError("Support must contain 512 unique sorted integer indices")
    world = scaling_world(dict(spec, composition_count="all"))
    pool = world["available_composite"]
    if indices[0] < 0 or indices[-1] >= len(pool):
        raise ValueError("Support index outside the fixed available pool")
    world["train_composite"] = pool[indices].copy()
    if len(set(map(tuple, world["train_composite"]))) != 512:
        raise ValueError("Support contains duplicate training chains")
    audit_world(world)
    expected = spec.get("frozen_data_sha256")
    if expected is not None and data_digest(world) != expected:
        raise RuntimeError("Rebuilt support data differ from the frozen digest")
    return world


def exposure_signature(world, counts=None):
    """Count actual sampled roles and column marginals, independently of streams."""
    rows = world["train_composite"]
    if counts is None:
        counts = np.ones(len(rows), dtype=np.int64)
    if len(counts) != len(rows):
        raise ValueError("Exposure counter does not match the support")
    first, second = Counter(), Counter()
    columns = [Counter() for _ in range(5)]
    for row, count in zip(rows, counts, strict=True):
        h, r1, b, r2, _t = map(int, row)
        first[f"{h}:{r1}"] += int(count)
        second[f"{b}:{r2}"] += int(count)
        for col, value in enumerate(row):
            columns[col][str(int(value))] += int(count)
    return {
        "first_roles": dict(sorted(first.items())),
        "second_roles": dict(sorted(second.items())),
        "columns": [dict(sorted(c.items())) for c in columns],
    }


def support_components(world):
    """Independent adjacency traversal, including all available training roles."""
    adjacency = {}
    for h, r1, b, r2, _t in world["available_composite"]:
        adjacency.setdefault(("l", int(h), int(r1)), set())
        adjacency.setdefault(("r", int(b), int(r2)), set())
    for h, r1, b, r2, _t in world["train_composite"]:
        left, right = ("l", int(h), int(r1)), ("r", int(b), int(r2))
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen, components = set(), 0
    for root in adjacency:
        if root in seen:
            continue
        components += 1
        stack = [root]
        seen.add(root)
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return {
        "roles": len(adjacency),
        "components": components,
        "isolated_roles": sum(not neighbours for neighbours in adjacency.values()),
    }


def validate_nodes(spec):
    if spec["batch_size"] != 192 or any(node % 8 for node in spec["nodes"]):
        raise ValueError("Every measurement must complete the 512-chain epochs")
    if not set(spec["repeat_nodes"]) <= set(spec["checkpoint_nodes"]) <= set(spec["nodes"]):
        raise ValueError("Scans and checkpoints must use recorded learning nodes")


def train(spec, out, source, device="cuda:0"):
    validate_nodes(spec)
    if "frozen_data_sha256" not in spec:
        raise ValueError("Training requires a frozen data digest")
    if any(file_hash(path) != digest for path, digest in source.items()):
        raise RuntimeError("Training source differs from its frozen digest")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "latest.pt").exists() or (out / "complete.json").exists():
        raise FileExistsError(f"Do not overwrite a training attempt: {out}")
    if spec["nodes"][0] != 0 or spec["nodes"][-1] != spec["steps"]:
        raise ValueError("Nodes must include initialization and the fixed endpoint")
    if spec["nodes"] != sorted(set(spec["nodes"])) or spec["batch_size"] % 3:
        raise ValueError("Nodes must increase and the batch must divide into three streams")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    process_start = time.perf_counter()
    world = build_world(spec)
    np.savez_compressed(out / "world.npz", **world)
    digest = data_digest(world)
    write_json(out / "spec.json", {"spec": spec, "source": source, "data_sha256": digest})
    model = construct(spec, device)
    initial_digest = model_digest(model)
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    strata = [world["common_atomic"], world["train_composite"], world["anchor_atomic"]]
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device)
        for parts in zip(*(pack_sentences(rows) for rows in strata), strict=True)
    )
    sizes = [len(rows) for rows in strata]
    offsets = np.cumsum([0, *sizes[:-1]])
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    batch = spec["batch_size"]
    graph = FullTokenStep(model, optimizer, table, batch)
    flop_step = flops(model.config, spec["repeats"], batch, 9, output_positions=9)
    history, repeat_metrics, training_seconds, step = [], {}, 0.0, 0

    def measure(loss=None):
        if step and not np.all(counts[1] == step * batch // 3 // 512):
            raise RuntimeError("Composition exposures must be exact complete epochs")
        metrics, predictions = evaluate(model, world, "low", device)
        row = {
            "step": step,
            "lr": learning_rate(spec, step) if step else 0.0,
            "metrics": metrics,
            "loss": loss,
            "training_seconds": training_seconds,
            "estimated_training_flops": step * flop_step,
            "composition_epochs": step * batch / 3 / sizes[1],
        }
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        if step in spec["repeat_nodes"]:
            repeat_metrics[str(step)] = {}
            for repeat in spec["test_repeats"]:
                m, p = evaluate(model, world, "low", device, repeat)
                repeat_metrics[str(step)][str(repeat)] = m
                np.savez_compressed(out / f"repeat-{step:06d}-r{repeat:02d}.npz", **p)
            write_json(out / "repeat-metrics.json", repeat_metrics)
        state = {
            "spec": spec,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "streams": [s.state_dict() for s in streams],
            "counts": counts,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "source": source,
            "data_sha256": digest,
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        if step in spec["checkpoint_nodes"]:
            torch.save(
                {"spec": spec, "step": step, "model": model.state_dict()},
                out / f"model-{step:06d}.pt",
            )
        write_json(out / "status.json", {"step": step, "budget": spec["steps"]})
        print(
            json.dumps(
                {
                    "run": run_name(spec),
                    "step": step,
                    "atomic": metrics["common_atomic"]["accuracy"],
                    "train": metrics["train_composite"]["accuracy"],
                    "test": metrics["familiar_test"]["accuracy"],
                }
            ),
            flush=True,
        )
        return row

    measure()
    for end in spec["nodes"][1:]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        while step < end:
            n = min(128, end - step)
            indices = []
            for i, stream in enumerate(streams):
                drawn = stream.take(n * batch // 3).reshape(n, batch // 3)
                counts[i] += np.bincount(drawn.ravel(), minlength=sizes[i])
                indices.append(drawn + offsets[i])
            indices = torch.as_tensor(np.concatenate(indices, axis=1), device=device)
            for j in range(n):
                lr.fill_(learning_rate(spec, step + j + 1))
                loss = graph(indices[j])
            step += n
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at {step}")
        endpoint = measure(loss)
    np.savez_compressed(out / "exposures.npz", **{f"stratum{i}": c for i, c in enumerate(counts)})
    result = {
        "spec": spec,
        "source": source,
        "data_sha256": digest,
        "initial_model_sha256": initial_digest,
        "checkpoint_sha256": file_hash(out / "latest.pt"),
        "finished_utc": utc(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "independent_facts": sizes[0] + sizes[2],
        "composition_examples": sizes[1],
        "role_coverage": role_coverage(world),
        "support_components": support_components(world),
        "support_exposures": exposure_signature(world, counts[1]),
        "endpoint": endpoint,
        "repeat_metrics": repeat_metrics,
        "training_seconds": training_seconds,
        "process_seconds": time.perf_counter() - process_start,
        "supervised_tokens": step * batch * 23 // 3,
        "padded_input_tokens": step * batch * 9,
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
    out = Path(out)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    result = json.loads((out / "complete.json").read_text())
    assert file_hash(out / "latest.pt") == result["checkpoint_sha256"]
    saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    recorded = json.loads((out / "spec.json").read_text())
    assert saved["spec"] == result["spec"] == recorded["spec"]
    assert saved["source"] == result["source"] == recorded["source"]
    assert all(file_hash(path) == digest for path, digest in result["source"].items())
    spec = saved["spec"]
    validate_nodes(spec)
    assert "frozen_data_sha256" in spec
    world = build_world(spec)
    assert data_digest(world) == result["data_sha256"] == saved["data_sha256"]
    assert data_digest(world) == recorded["data_sha256"] == spec["frozen_data_sha256"]
    with np.load(out / "world.npz") as archived:
        assert set(archived.files) == set(world)
        for key, rows in world.items():
            np.testing.assert_array_equal(rows, archived[key])
    assert saved["step"] == spec["steps"]
    model = construct(spec, device)
    assert model_digest(model) == result["initial_model_sha256"]
    model.load_state_dict(saved["model"])
    metrics, predictions = evaluate(model, world, "low", device)
    compare_metrics(metrics, result["endpoint"]["metrics"])
    maximum = compare_predictions(
        predictions, np.load(out / f"predictions-{spec['steps']:06d}.npz")
    )
    repeat_checks = 0
    for node, repeats in result["repeat_metrics"].items():
        checkpoint = torch.load(
            out / f"model-{int(node):06d}.pt", map_location=device, weights_only=False
        )
        assert checkpoint["spec"] == spec and checkpoint["step"] == int(node)
        model.load_state_dict(checkpoint["model"])
        for repeat, expected in repeats.items():
            actual, pred = evaluate(model, world, "low", device, int(repeat))
            compare_metrics(actual, expected)
            maximum = max(
                maximum,
                compare_predictions(
                    pred, np.load(out / f"repeat-{int(node):06d}-r{int(repeat):02d}.npz")
                ),
            )
            repeat_checks += 1
    exposures = np.load(out / "exposures.npz")
    for i in range(3):
        np.testing.assert_array_equal(exposures[f"stratum{i}"], saved["counts"][i])
        assert exposures[f"stratum{i}"].sum() == spec["steps"] * spec["batch_size"] // 3
        assert np.ptp(exposures[f"stratum{i}"]) <= 1
    composition_counts = exposures["stratum1"]
    assert np.all(composition_counts == spec["steps"] * spec["batch_size"] // 3 // 512)
    assert exposure_signature(world, composition_counts) == result["support_exposures"]
    assert support_components(world) == result["support_components"]
    checked = {
        "frozen_data_exact": True,
        "archived_world_exact": True,
        "composition_complete_epochs": True,
        "role_exposures_exact": True,
        "passed": True,
        "endpoint_tokens_exact": True,
        "repeat_checks": repeat_checks,
        "max_nll_difference": maximum,
        "nll_rtol": 1e-5,
        "nll_atol": 1e-5,
        "utc": utc(),
    }
    write_json(out / "audit.json", checked)
    return checked
