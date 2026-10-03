"""Paired geometric-supervision coverage in the established complete GPT task.

All arms retain full-token CE and all-example first-hop CE. Coverage changes
only a multiplicative mask on the per-example geometric objective; every arm
executes the same forward and backward operations, including the zero mask.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import EpochStream, GraphStep, make_optimizer, utc, write_json
from .grok_loop_model import flops
from .latent_scaling import build_world, model_digest
from .representation_alignment import new_model
from .representation_alignment import pack_training as pack_base_training
from .storage_composition import data_digest, evaluate, file_hash
from .storage_frontier import learning_rate

COVERAGES = ("none", "composition_only", "all_atomic_and_composition")
NORMALIZATIONS = ("selected_mean", "batch_mean")
STRATA = ("common_atomic", "train_composite", "anchor_atomic")
COMPONENTS = (
    "text_ce",
    "bridge_ce",
    "alignment_loss",
    "alignment_all_mean",
    "alignment_selected_mean",
    "alignment_batch_mean",
    "alignment_selected_count",
    "alignment_denominator",
)


def make_specs(phase="development"):
    """Return the proposed paired matrix without freezing files or starting jobs."""
    if phase not in ("development", "confirmation"):
        raise ValueError("Phase must be development or confirmation")
    worlds = [810011] if phase == "development" else [810101, 810102, 810103]
    initializations = [811011] if phase == "development" else [811101, 811102]
    base = {
        "phase": phase,
        "stream_seed": 812011 if phase == "development" else 812101,
        "heads_n": 256,
        "bridges_n": 128,
        "tails_n": 64,
        "familiar_n": 64,
        "strict_n": 32,
        "anchor_n": 32,
        "holdout_fraction": 0.25,
        "low_extra": "anchors",
        "composition_count": 256,
        "width": 128,
        "heads": 4,
        "layers": 1,
        "repeats": 2,
        "dropout": 0.0,
        "batch_size": 192,
        "lr": 0.001,
        "weight_decay": 0.01,
        "warmup": 200,
        "schedule": "cosine",
        "min_lr_ratio": 0.1,
        "steps": 16000,
        "nodes": [0, 256, 1000, 2000, 4000, 8000, 16000],
        "bridge_weight": 0.3,
        "alignment_weight": 0.3,
    }
    arms = (
        ("none", "none", "batch_mean"),
        ("composition_batch", "composition_only", "batch_mean"),
        ("composition_selected", "composition_only", "selected_mean"),
        ("all_batch", "all_atomic_and_composition", "batch_mean"),
    )
    return [
        {
            **base,
            "nodes": list(base["nodes"]),
            "world": world,
            "initialization": initialization,
            "arm": arm,
            "alignment_coverage": coverage,
            "alignment_normalization": normalization,
        }
        for world in worlds
        for initialization in initializations
        for arm, coverage, normalization in arms
    ]


def validate_spec(spec):
    if spec["alignment_coverage"] not in COVERAGES:
        raise ValueError(f"Unknown alignment coverage: {spec['alignment_coverage']}")
    if spec["alignment_normalization"] not in NORMALIZATIONS:
        raise ValueError(f"Unknown alignment normalization: {spec['alignment_normalization']}")
    if spec["bridge_weight"] != 0.3:
        raise ValueError("This comparison fixes all-example first-hop CE weight at 0.3")
    if not np.isfinite(spec["alignment_weight"]) or spec["alignment_weight"] < 0:
        raise ValueError("Alignment weight must be finite and nonnegative")
    if spec["batch_size"] <= 0 or spec["batch_size"] % 3:
        raise ValueError("Batch size must divide equally into the three fixed streams")
    nodes = spec["nodes"]
    if not nodes or nodes[0] != 0 or nodes[-1] != spec["steps"]:
        raise ValueError("Nodes must include initialization and the fixed endpoint")
    if nodes != sorted(set(nodes)) or any(type(n) is not int or n < 0 for n in nodes):
        raise ValueError("Nodes must be strictly increasing nonnegative integer steps")
    if spec["steps"] <= 0:
        raise ValueError("Training needs a positive fixed step budget")


def pack_training(world, coverage):
    """Keep base data byte-identical and append an explicit geometry mask."""
    if coverage not in COVERAGES:
        raise ValueError(f"Unknown alignment coverage: {coverage}")
    arrays, sizes = pack_base_training(world)
    if min(sizes) <= 0:
        raise ValueError("All three training strata must be nonempty")
    mask = np.zeros(sum(sizes), dtype=np.float32)
    if coverage == "composition_only":
        mask[sizes[0] : sizes[0] + sizes[1]] = 1
    elif coverage == "all_atomic_and_composition":
        mask[:] = 1
    return (*arrays, mask), sizes


def objective(
    model,
    tokens,
    labels,
    targets,
    geometry_mask,
    bridge_weight=0.3,
    alignment_weight=0.3,
    normalization="selected_mean",
):
    """Return the training loss and named, unweighted objective components.

    selected_mean divides by selected examples (clamped to one for an empty
    mask). batch_mean divides by all examples. Both reductions are always
    evaluated, and mask coverage is recorded separately from loss strength.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown alignment normalization: {normalization}")
    logits, bridge = model(tokens, return_bridge=True)
    text_ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    bridge_logits = F.linear(model.ln_final(bridge), model.token.weight)
    bridge_ce = F.cross_entropy(bridge_logits, targets)
    canonical = model.token(targets).detach()
    per_example = (F.normalize(bridge, dim=-1) - F.normalize(canonical, dim=-1)).square().sum(-1)
    mask = geometry_mask.to(dtype=per_example.dtype)
    selected_count = mask.sum()
    numerator = (per_example * mask).sum()
    selected_denominator = selected_count.clamp_min(1)
    # Avoid creating a device scalar from host data during CUDA graph capture.
    batch_denominator = torch.ones_like(selected_count) * per_example.numel()
    selected_mean = numerator / selected_denominator
    batch_mean = numerator / batch_denominator
    use_selected = float(normalization == "selected_mean")
    denominator = use_selected * selected_denominator + (1 - use_selected) * batch_denominator
    alignment = numerator / denominator
    parts = torch.stack(
        (
            text_ce,
            bridge_ce,
            alignment,
            per_example.mean(),
            selected_mean,
            batch_mean,
            selected_count,
            denominator,
        )
    )
    return text_ce + bridge_weight * bridge_ce + alignment_weight * alignment, parts


class CoverageStep(GraphStep):
    def __init__(self, model, optimizer, table, batch_size, spec):
        self.spec = spec
        super().__init__(model, optimizer, table, batch_size)

    def eager(self):
        self.optimizer.zero_grad(set_to_none=False)
        tokens, labels, targets, mask = (part[self.index] for part in self.table)
        loss, self.components = objective(
            self.model,
            tokens,
            labels,
            targets,
            mask,
            self.spec["bridge_weight"],
            self.spec["alignment_weight"],
            self.spec["alignment_normalization"],
        )
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss, norm


def run_name(spec):
    return (
        f"{spec['phase']}-w{spec['world']}-i{spec['initialization']}-"
        f"{spec['alignment_coverage']}-{spec['alignment_normalization']}"
    )


def _draw_indices(streams, offsets, counts, steps, batch_size):
    indices = []
    for j, stream in enumerate(streams):
        drawn = stream.take(steps * batch_size // 3).reshape(steps, -1)
        counts[j] += np.bincount(drawn.ravel(), minlength=len(counts[j]))
        indices.append(drawn + offsets[j])
    return np.concatenate(indices, axis=1)


def _coverage_presentations(counts, mask, sizes):
    offset, result = 0, {}
    for name, count, size in zip(STRATA, counts, sizes, strict=True):
        result[name] = int(np.dot(count, mask[offset : offset + size].astype(np.int64)))
        offset += size
    result["total"] = sum(result.values())
    return result


def train(spec, out, source, device):
    """Train one immutable attempt; save weights and exposures at every node."""
    validate_spec(spec)
    out, device = Path(out), torch.device(device)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("run.json", "model.pt", "complete.json")):
        raise FileExistsError(f"Do not overwrite an attempt: {out}")
    if device.type != "cuda":
        raise ValueError("The captured training step requires an explicitly allocated CUDA device")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(device)
    started = time.perf_counter()
    world = build_world(spec)
    np.savez_compressed(out / "world.npz", **world)
    arrays, sizes = pack_training(world, spec["alignment_coverage"])
    mask = arrays[-1]
    model = new_model(spec, device)
    initial_hash, world_hash = model_digest(model), data_digest(world)
    identity = {
        "spec": spec,
        "source": source,
        "implementation_sha256": file_hash(Path(__file__)),
        "world_sha256": world_hash,
        "initial_model_sha256": initial_hash,
    }
    write_json(
        out / "run.json",
        {
            **identity,
            "pid": os.getpid(),
            "gpu": device.index,
            "gpu_name": torch.cuda.get_device_name(device),
            "dtype": "float32",
            "allow_tf32": True,
            "parameters": sum(p.numel() for p in model.parameters()),
            "atomic_examples": sizes[0] + sizes[2],
            "composition_examples": sizes[1],
            "vocab_size": model.config.vocab_size,
            "strata": dict(zip(STRATA, sizes, strict=True)),
            "geometry_unique_examples": _coverage_presentations(
                [np.ones(n, dtype=np.int64) for n in sizes], mask, sizes
            ),
            "geometry_definition": (
                "squared distance of normalized early state and detached input embedding"
            ),
            "bridge_ce_coverage": "all_atomic_and_composition",
            "geometry_normalization": spec["alignment_normalization"],
            "tracking_owner": "external read-only batch runner",
        },
    )
    table = tuple(torch.as_tensor(a, device=device) for a in arrays)
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    model.train()
    graph = CoverageStep(model, optimizer, table, spec["batch_size"], spec)
    assert model_digest(model) == initial_hash, "CUDA capture changed initialization"
    streams = [EpochStream(n, spec["stream_seed"] + j) for j, n in enumerate(sizes)]
    offsets = np.cumsum([0, *sizes[:-1]])
    counts = [np.zeros(n, dtype=np.int64) for n in sizes]
    input_hash = hashlib.sha256()
    history, checkpoints, step, training_seconds = [], [], 0, 0.0
    flop_step = flops(model.config, spec["repeats"], spec["batch_size"], 9, output_positions=9)
    flop_step += 6 * spec["batch_size"] * model.config.width * model.config.vocab_size

    def measure():
        metrics, predictions = evaluate(model, world, "low", device)
        parts = graph.components.detach().cpu().tolist() if step else [None] * len(COMPONENTS)
        coverage = _coverage_presentations(counts, mask, sizes)
        row = {
            "step": step,
            "metrics": metrics,
            **dict(zip(COMPONENTS, parts, strict=True)),
            "alignment_normalization": spec["alignment_normalization"],
            "alignment_weight": spec["alignment_weight"],
            "geometry_target_presentations": coverage,
            "training_seconds": training_seconds,
            "wall_seconds": time.perf_counter() - started,
            "examples": step * spec["batch_size"],
            "supervised_tokens": step * spec["batch_size"] // 3 * (7 + 9 + 7),
            "auxiliary_target_presentations": step * spec["batch_size"],
            "estimated_matmul_training_flops": step * flop_step,
            "composition_epochs": step * spec["batch_size"] / 3 / sizes[1],
            "sample_stream_sha256": input_hash.hexdigest(),
        }
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        np.savez_compressed(
            out / f"exposures-{step:06d}.npz", **{f"stratum{j}": c for j, c in enumerate(counts)}
        )
        checkpoint = out / f"model-{step:06d}.pt"
        torch.save(
            {
                **identity,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "streams": [s.state_dict() for s in streams],
                "counts": counts,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
                "sample_stream_sha256": input_hash.hexdigest(),
                "geometry_target_presentations": coverage,
            },
            checkpoint,
        )
        checkpoints.append({"step": step, "file": checkpoint.name, "sha256": file_hash(checkpoint)})
        write_json(out / "checkpoints.json", checkpoints)
        write_json(out / "status.json", {"state": "running", "step": step, "budget": spec["steps"]})
        print(
            json.dumps(
                {
                    "run": run_name(spec),
                    "step": step,
                    "atomic": metrics["common_atomic"]["accuracy"],
                    "train": metrics["train_composite"]["accuracy"],
                    "familiar": metrics["familiar_test"]["accuracy"],
                    "strict": metrics["strict_test"]["accuracy"],
                }
            ),
            flush=True,
        )
        model.train()

    measure()
    for end in spec["nodes"][1:]:
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        while step < end:
            n = min(128, end - step)
            indices = _draw_indices(streams, offsets, counts, n, spec["batch_size"])
            input_hash.update(indices.tobytes())
            indices = torch.as_tensor(indices, device=device)
            for j in range(n):
                lr.fill_(learning_rate(spec, step + j + 1))
                loss = graph(indices[j])
            step += n
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - began
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f"Nonfinite objective at {step}")
        measure()
    # Keep the established downstream loader interface on filesystems without links.
    shutil.copyfile(out / f"model-{step:06d}.pt", out / "model.pt")
    np.savez_compressed(out / "exposures.npz", **{f"stratum{j}": c for j, c in enumerate(counts)})
    write_json(
        out / "complete.json",
        {
            **identity,
            "finished_utc": utc(),
            "metrics": history[-1]["metrics"],
            "training_seconds": training_seconds,
            "sample_stream_sha256": input_hash.hexdigest(),
            "geometry_target_presentations": history[-1]["geometry_target_presentations"],
            "model_sha256": file_hash(out / "model.pt"),
            "nodes_saved": spec["nodes"],
        },
    )
    write_json(out / "status.json", {"state": "complete", "step": step})


def audit(out, device, *, all_nodes=False):
    """Verify all artifacts; independently reload the endpoint (or all nodes)."""
    out, device = Path(out), torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.set_device(device)
    run = json.loads((out / "run.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    history = json.loads((out / "learning.json").read_text())
    manifest = json.loads((out / "checkpoints.json").read_text())
    spec = run["spec"]
    validate_spec(spec)
    assert spec == complete["spec"]
    assert [row["step"] for row in history] == spec["nodes"]
    assert [entry["step"] for entry in manifest] == spec["nodes"]
    assert complete["nodes_saved"] == spec["nodes"]
    assert file_hash(out / "model.pt") == complete["model_sha256"] == manifest[-1]["sha256"]
    world = build_world(spec)
    assert data_digest(world) == run["world_sha256"] == complete["world_sha256"]
    with np.load(out / "world.npz") as saved_world:
        assert set(saved_world.files) == set(world)
        for key, value in world.items():
            np.testing.assert_array_equal(value, saved_world[key])
    arrays, sizes = pack_training(world, spec["alignment_coverage"])
    mask, offsets = arrays[-1], np.cumsum([0, *sizes[:-1]])
    streams = [EpochStream(n, spec["stream_seed"] + j) for j, n in enumerate(sizes)]
    counts = [np.zeros(n, dtype=np.int64) for n in sizes]
    input_hash, step, audits = hashlib.sha256(), 0, []
    model = new_model(spec, device)
    assert model_digest(model) == run["initial_model_sha256"] == complete["initial_model_sha256"]
    for entry, row in zip(manifest, history, strict=True):
        while step < entry["step"]:
            n = min(128, entry["step"] - step)
            indices = _draw_indices(streams, offsets, counts, n, spec["batch_size"])
            input_hash.update(indices.tobytes())
            step += n
        checkpoint = out / entry["file"]
        assert file_hash(checkpoint) == entry["sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["spec"] == spec and payload["step"] == step
        for key in ("source", "implementation_sha256", "world_sha256", "initial_model_sha256"):
            assert payload[key] == run[key] == complete[key]
        assert (
            payload["sample_stream_sha256"] == row["sample_stream_sha256"] == input_hash.hexdigest()
        )
        coverage = _coverage_presentations(counts, mask, sizes)
        assert (
            payload["geometry_target_presentations"]
            == row["geometry_target_presentations"]
            == coverage
        )
        with np.load(out / f"exposures-{step:06d}.npz") as saved_counts:
            for j, count in enumerate(counts):
                np.testing.assert_array_equal(count, saved_counts[f"stratum{j}"])
                np.testing.assert_array_equal(count, payload["counts"][j])
        if step == 0:
            model.load_state_dict(payload["model"])
            assert model_digest(model) == run["initial_model_sha256"]
        if not all_nodes and step != spec["steps"]:
            audits.append({"step": step, "passed": True, "predictions_recomputed": False})
            continue
        model.load_state_dict(payload["model"])
        metrics, predictions = evaluate(model, world, "low", device)
        max_nll_error = 0.0
        with np.load(out / f"predictions-{step:06d}.npz") as saved:
            assert set(saved.files) == set(predictions)
            for key, actual in predictions.items():
                if key.endswith("nll"):
                    max_nll_error = max(max_nll_error, float(np.max(np.abs(actual - saved[key]))))
                    np.testing.assert_allclose(actual, saved[key], rtol=1e-5, atol=1e-5)
                else:
                    np.testing.assert_array_equal(actual, saved[key])
        for split, values in metrics.items():
            for key, actual in values.items():
                expected = row["metrics"][split][key]
                if actual is None:
                    assert expected is None
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        audits.append(
            {
                "step": step,
                "passed": True,
                "predictions_recomputed": True,
                "max_nll_error": max_nll_error,
            }
        )
    assert complete["sample_stream_sha256"] == input_hash.hexdigest()
    assert complete["geometry_target_presentations"] == _coverage_presentations(counts, mask, sizes)
    with np.load(out / "exposures.npz") as saved_counts:
        for j, count in enumerate(counts):
            np.testing.assert_array_equal(count, saved_counts[f"stratum{j}"])
    result = {
        "passed": True,
        "metrics": metrics,
        "nodes": audits,
        "max_nll_error": max(a.get("max_nll_error", 0.0) for a in audits),
        "all_nodes": all_nodes,
        "allow_tf32": True,
        "world_sha256": data_digest(world),
        "sample_stream_sha256": input_hash.hexdigest(),
        "geometry_target_presentations": _coverage_presentations(counts, mask, sizes),
    }
    write_json(out / "audit.json", result)
    return result
