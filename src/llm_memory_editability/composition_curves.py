"""Distinct composition support and knowledge-load curves on random relation graphs.

The graph, answer/EOS objective, GPT blocks and optimizer reuse the previously
successful Wang et al. adaptation. A fixed maximum entity vocabulary keeps
parameters identical across loads. Development and confirmation worlds differ.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .bios_model import ModelConfig, matmul_flops
from .grok_depth import (
    EpochStream,
    GraphStep,
    SmallGPT,
    evaluate_rows,
    make_optimizer,
    pack_rows,
    two_calls,
    utc,
    write_json,
)
from .grok_depth_data import audit_world, build_world


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def prepare_world(spec):
    """Reserve once, permute once, take nested prefixes; never repeat support rows."""
    n = spec["entities"]
    maximum = spec["max_entities"]
    if not 1 <= n <= maximum:
        raise ValueError("Entity count must fit the fixed vocabulary")
    base = build_world(
        spec["world_seed"],
        entities=n,
        relations=spec["relations"],
        degree=spec["degree"],
        phi=0,
        id_fraction=spec["id_fraction"],
        id_test_fraction=spec["test_fraction"],
    )
    audit_world(base)
    rng = np.random.default_rng(np.random.SeedSequence([spec["world_seed"], n, 960601]))
    order = rng.permutation(len(base["unused_composite"]))
    requested = round(spec["phi"] * len(base["id_atomic"]))
    if requested > len(order):
        raise ValueError("Distinct support request exceeds the reserved training pool")
    arrays = {
        key: base[key].copy()
        for key in ("atomic", "id_atomic", "ood_atomic", "test_composite", "ood_composite")
    }
    eligible = base["unused_composite"][order].copy()
    for rows in [*arrays.values(), eligible]:
        rows[:, 1:-1] += maximum - n
    identity = array_hash({**arrays, "eligible": eligible})
    arrays["train_composite"] = eligible[:requested].copy()
    lookup = np.full((maximum, spec["relations"]), -1, dtype=np.int64)
    atomic = arrays["atomic"]
    lookup[atomic[:, 0] - 2, atomic[:, 1] - maximum - 2] = np.arange(len(atomic))
    first, second = constituent_indices(arrays["train_composite"], atomic, lookup, maximum)
    first_counts = np.bincount(first, minlength=len(atomic))
    second_counts = np.bincount(second, minlength=len(atomic))
    role = {}
    for name in ("test_composite", "ood_composite"):
        one, two = constituent_indices(arrays[name], atomic, lookup, maximum)
        role[name] = {
            "n": len(one),
            "both_facts_seen_in_required_roles": (
                float(((first_counts[one] > 0) & (second_counts[two] > 0)).mean())
                if len(one)
                else None
            ),
        }
    panel_rng = np.random.default_rng(np.random.SeedSequence([spec["world_seed"], n, 960602]))
    panels = {}
    for key, name in [
        ("atomic", "atomic"),
        ("id_atomic", "atomic_id"),
        ("ood_atomic", "atomic_ood"),
        ("test_composite", "II"),
        ("ood_composite", "OO"),
    ]:
        rows = arrays[key]
        ids = panel_rng.choice(len(rows), min(spec["evaluation_size"], len(rows)), replace=False)
        panels[name] = rows[ids].copy()
    panels["train_composition"] = arrays["train_composite"][: spec["evaluation_size"]].copy()
    metadata = {
        "world_seed": spec["world_seed"],
        "entities": n,
        "max_entities": maximum,
        "relations": spec["relations"],
        "degree": spec["degree"],
        "vocab_size": maximum + spec["relations"] + 2,
        "atomic_examples": len(atomic),
        "id_atomic_examples": len(arrays["id_atomic"]),
        "composition_examples": requested,
        "eligible_compositions": len(eligible),
        "phi_actual": requested / len(arrays["id_atomic"]),
        "world_sha256": identity,
        "dataset_sha256": array_hash(arrays),
        "knowledge_bits": len(atomic) * math.log2(n),
        "entropy_condition": "independent uniform tails, conditional on head/relation keys",
        "composition_additional_independent_bits": 0,
        "role_coverage": role,
        "panel_counts": {key: len(value) for key, value in panels.items()},
        "full_counts": {key: len(value) for key, value in arrays.items()},
        "graph_source": base["metadata"]["source_url"],
        "graph_source_commit": base["metadata"]["source_commit"],
        "graph_audit": base["metadata"]["dataset_sha256"],
    }
    return arrays, panels, metadata


def constituent_indices(rows, atomic, lookup, maximum):
    first = lookup[rows[:, 0] - 2, rows[:, 1] - maximum - 2]
    if np.any(first < 0):
        raise ValueError("Unknown first-hop fact")
    second = lookup[atomic[first, 2] - 2, rows[:, 2] - maximum - 2]
    if np.any(second < 0) or not np.array_equal(atomic[second, 2], rows[:, -1]):
        raise ValueError("Composition truth differs from graph traversal")
    return first, second


def budget_nodes(spec, total):
    matched = math.ceil(spec["target_exposures"] * total / spec["batch_size"])
    compute = spec["compute_steps"]
    end = spec.get("fixed_steps", max(compute, matched))
    nodes = {0, end, *range(2000, min(end, 128000) + 1, 2000)}
    nodes.update(s for s in (256, 512, 1000, compute, matched) if s <= end)
    nodes.update(range(192000, end, 64000))
    return sorted(nodes), {"compute": compute, "exposure": matched, "end": end}


def construct(spec, device):
    torch.manual_seed(spec["initialization"])
    cfg = ModelConfig(
        vocab_size=spec["max_entities"] + spec["relations"] + 2,
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    return SmallGPT(cfg, dropout=spec["dropout"]).to(device), cfg


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(model, panels, device):
    metrics, predictions = {}, {}
    for name, rows in panels.items():
        metrics[name], values = evaluate_rows(model, rows, device, batch_size=512)
        if len(rows):
            metrics[name]["answer_nll"] = float(values["nll"][:, 0].mean())
        predictions.update({name + "_" + key: value for key, value in values.items()})
    return metrics, predictions


def train(spec, out, resume=False):
    out = Path(out)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    arrays, panels, data = prepare_world(spec)
    model, cfg = construct(spec, device)
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    packed = [pack_rows(arrays[key]) for key in ("atomic", "train_composite")]
    table = tuple(
        torch.as_tensor(np.concatenate([a, b]), device=device) for a, b in zip(*packed, strict=True)
    )
    atomic_n, composition_n = data["atomic_examples"], data["composition_examples"]
    total = atomic_n + composition_n
    stream = EpochStream(total, spec["stream_seed"])
    nodes, budgets = budget_nodes(spec, total)
    start, seconds = 0, 0.0
    counts = {"atomic": 0, "composition": 0}
    initial_hash = model_hash(model)
    if resume:
        saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
        if saved["spec"] != spec or saved["world_sha256"] != data["world_sha256"]:
            raise ValueError("Resume identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        stream.load_state_dict(saved["stream"])
        counts, seconds, start = saved["counts"], saved["seconds"], saved["step"]
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        initial_hash = json.loads((out / "run.json").read_text())["initial_model_sha256"]
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "spec": spec,
        "phase": spec["phase"],
        "pid": os.getpid(),
        "gpu": int(os.environ.get("PHYSICAL_GPU", 0)),
        "gpu_name": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_model_sha256": initial_hash,
        "world_sha256": data["world_sha256"],
        "dtype": "FP32 parameters/optimizer/forward; TF32 matmul",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "budgets": budgets,
        "evaluation_nodes": nodes,
        **data,
    }
    write_json(out / "run.json", metadata)
    write_json(out / "data-audit.json", data)
    np.savez_compressed(out / "world.npz", **arrays)
    np.savez_compressed(out / "panels.npz", **panels)
    t0 = time.perf_counter()
    graph = GraphStep(model, optimizer, table, spec["batch_size"])
    capture = time.perf_counter() - t0
    per_step = matmul_flops(cfg, spec["batch_size"], 4, output_positions=2, backward=True)
    history = json.loads((out / "learning.json").read_text()) if resume else []
    history = [record for record in history if record["step"] <= start]
    begun = time.perf_counter()

    def checkpoint(step, loss=None):
        metrics, predictions = measure(model, panels, device)
        record = {
            "step": step,
            "utc": utc(),
            "metrics": metrics,
            "training_seconds": seconds,
            "wall_seconds": time.perf_counter() - begun,
            "capture_seconds": capture,
            "examples": step * spec["batch_size"],
            "atomic_examples_seen": counts["atomic"],
            "composition_examples_seen": counts["composition"],
            "atomic_epochs": counts["atomic"] / atomic_n,
            "composition_epochs": counts["composition"] / composition_n if composition_n else 0,
            "minimum_record_exposures": (step * spec["batch_size"]) // total,
            "maximum_record_exposures": math.ceil(step * spec["batch_size"] / total),
            "supervised_tokens": step * spec["batch_size"] * 2,
            "executed_input_tokens": step * spec["batch_size"] * 4,
            "estimated_matmul_training_flops": step * per_step,
            "loss": loss,
            "learning_rate": float(lr.detach()),
        }
        history.append(record)
        write_json(out / "learning.json", history)
        payload = {
            "spec": spec,
            "world_sha256": data["world_sha256"],
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "seconds": seconds,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
        }
        torch.save(payload, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        if step in budgets.values():
            torch.save(
                {"spec": spec, "step": step, "model": model.state_dict()},
                out / f"weights-{step}.pt",
            )
            np.savez_compressed(out / f"predictions-{step}.npz", **predictions)
        status = {
            "state": "trained" if step == budgets["end"] else "running",
            "step": step,
            "budget": budgets["end"],
            "atomic": metrics["atomic"]["accuracy"],
            "II": metrics["II"]["accuracy"],
            "updated_utc": utc(),
        }
        write_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    if not resume:
        checkpoint(0)
    for end in (node for node in nodes if node > start):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = None
        step = start
        while step < end:
            length = min(256, end - step)
            indices = stream.take(length * spec["batch_size"]).reshape(length, -1)
            atom_count = int((indices < atomic_n).sum())
            counts["atomic"] += atom_count
            counts["composition"] += indices.size - atom_count
            gpu_indices = torch.as_tensor(indices, device=device)
            for j in range(length):
                lr.fill_(spec["lr"] * min(1.0, (step + j + 1) / spec["warmup"]))
                loss = graph(gpu_indices[j])
            step += length
        torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        value = float(loss.detach())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite loss at {end}")
        checkpoint(end, value)
        start = end
    full_panels = {
        "atomic": arrays["atomic"],
        "atomic_id": arrays["id_atomic"],
        "atomic_ood": arrays["ood_atomic"],
        "II": arrays["test_composite"],
        "OO": arrays["ood_composite"],
    }
    full, predictions = measure(model, full_panels, device)
    full["external_two_calls"] = two_calls(model, panels["II"], device)
    np.savez_compressed(out / "full-predictions.npz", **predictions)
    write_json(
        out / "complete.json",
        {
            "step": start,
            "spec": spec,
            "parameters": metadata["parameters"],
            "training_seconds": seconds,
            "wall_seconds": time.perf_counter() - begun,
            "endpoint": history[-1],
            "full_metrics": full,
            "finished_utc": utc(),
        },
    )


def audit(spec, out):
    """Reload both predeclared comparison checkpoints and independently recount."""
    out = Path(out)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    arrays, panels, data = prepare_world(spec)
    recorded = dict(np.load(out / "world.npz"))
    if array_hash(recorded) != data["dataset_sha256"]:
        raise ValueError("Saved world differs from deterministic reconstruction")
    meta = json.loads((out / "run.json").read_text())
    if meta["world_sha256"] != data["world_sha256"]:
        raise ValueError("World identity mismatch")
    history = json.loads((out / "learning.json").read_text())
    nodes, budgets = budget_nodes(spec, data["atomic_examples"] + data["composition_examples"])
    if [record["step"] for record in history] != nodes:
        raise ValueError("Missing or reordered evaluation nodes")
    model, cfg = construct(spec, "cuda:0")
    if sum(p.numel() for p in model.parameters()) != meta["parameters"]:
        raise ValueError("Parameter count mismatch")
    per_step = matmul_flops(cfg, spec["batch_size"], 4, output_positions=2, backward=True)
    for record in history:
        examples = record["step"] * spec["batch_size"]
        if record["examples"] != examples or examples != (
            record["atomic_examples_seen"] + record["composition_examples_seen"]
        ):
            raise ValueError("Exposure arithmetic mismatch")
        if record["estimated_matmul_training_flops"] != record["step"] * per_step:
            raise ValueError("Executed-shape FLOP mismatch")
    reloaded = []
    for step in sorted(set(budgets.values()) & set(nodes)):
        saved = torch.load(out / f"weights-{step}.pt", weights_only=False, map_location="cuda:0")
        if saved["spec"] != spec or saved["step"] != step:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        metrics, predictions = measure(model, panels, "cuda:0")
        original = dict(np.load(out / f"predictions-{step}.npz"))
        for key, values in predictions.items():
            if key.endswith("_nll"):
                np.testing.assert_allclose(values, original[key], rtol=1e-5, atol=1e-6)
            elif not np.array_equal(values, original[key]):
                raise ValueError(f"Reloaded predictions changed: {step} {key}")
        prior = next(record for record in history if record["step"] == step)["metrics"]
        for name, metric in metrics.items():
            if metric["accuracy"] != prior[name]["accuracy"]:
                raise ValueError("Metric recount mismatch")
        reloaded.append(step)
    full_panels = {
        "atomic": arrays["atomic"],
        "atomic_id": arrays["id_atomic"],
        "atomic_ood": arrays["ood_atomic"],
        "II": arrays["test_composite"],
        "OO": arrays["ood_composite"],
    }
    full, predictions = measure(model, full_panels, "cuda:0")
    original = dict(np.load(out / "full-predictions.npz"))
    for key, value in predictions.items():
        np.testing.assert_allclose(value, original[key], rtol=1e-5, atol=1e-6)
    full["external_two_calls"] = two_calls(model, panels["II"], "cuda:0")
    write_json(
        out / "audit.json",
        {"passed": True, "reloaded_steps": reloaded, "full_metrics": full, "utc": utc()},
    )
    write_json(out / "status.json", {"state": "complete", "step": budgets["end"], "utc": utc()})


def calibrated_choice(records, thresholds):
    """Development-only success gate; prefer smaller registered model when both pass."""
    candidates = []
    for spec, metrics in records:
        passed = all(metrics[key]["accuracy"] >= value for key, value in thresholds.items())
        if passed:
            candidates.append(spec)
    return min(candidates, key=lambda spec: spec["layers"]) if candidates else None


def confirmation_specs(config, selected):
    specs = []
    for index, seed in enumerate(config["confirmation_worlds"]):
        for phi in config["support_phi"]:
            specs.append(
                {
                    **selected,
                    "phase": "confirmation",
                    "world_seed": seed,
                    "initialization": config["initialization"] + index,
                    "stream_seed": config["stream_seed"] + index,
                    "name": f"support-w{index + 1}-phi{phi:g}",
                    "phi": phi,
                    "entities": config["base_spec"]["entities"],
                }
            )
    return specs


def anchors_pass(records, worlds, thresholds):
    """Require exactly the registered independent worlds; no cherry-picked anchors."""
    observed = [spec["world_seed"] for spec, _ in records]
    if len(observed) != len(worlds) or set(observed) != set(worlds):
        return False
    return all(
        all(metrics[key]["accuracy"] >= threshold for key, threshold in thresholds.items())
        for _, metrics in records
    )


def load_specs(config, selected):
    specs = []
    for index, seed in enumerate(config["confirmation_worlds"]):
        for n in config["load_entities"][1:]:
            specs.append(
                {
                    **selected,
                    "phase": "confirmation",
                    "world_seed": seed,
                    "initialization": config["initialization"] + index,
                    "stream_seed": config["stream_seed"] + index,
                    "name": f"load-w{index + 1}-n{n}",
                    "phi": config["anchor_phi"],
                    "entities": n,
                }
            )
    return specs
