"""Scaling knowledge use in ordinary GPT blocks with fixed facts and tests.

Independent composition examples and optimization time are separate axes. All
blocks retain learned embeddings, causal attention, residuals and 4x GELU MLPs.
Historical experiments and their frozen implementations are left untouched.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .bios_model import ModelConfig
from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .grok_loop_model import LoopGPT, flops
from .storage_composition import FullTokenStep, data_digest, evaluate, file_hash, pack_sentences
from .storage_composition import build_world as historical_world
from .storage_frontier import SOURCE_FILES as STABILITY_SOURCES
from .storage_frontier import learning_rate
from .text_pretrain import WORDS

SOURCE_FILES = [
    *STABILITY_SOURCES,
    "src/llm_memory_editability/latent_scaling.py",
    "scripts/run_latent_scaling.py",
    "tests/test_latent_scaling.py",
]


def build_world(spec):
    world = historical_world(spec)
    pool = world["train_composite"]
    order = np.random.default_rng(spec["world"] + 880000).permutation(len(pool))
    count = spec["composition_count"]
    count = len(pool) if count == "all" else count
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= len(pool):
        raise ValueError("Composition count must be a positive prefix of the fixed pool")
    world["available_composite"] = pool[order]
    world["train_composite"] = world["available_composite"][:count].copy()
    return world


def construct(spec, device):
    config = ModelConfig(
        vocab_size=len(WORDS) + spec["heads_n"] + spec["bridges_n"] + spec["tails_n"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=16,
    )
    torch.manual_seed(spec["initialization"])
    # R=1 is ordinary GPT. The wrapper permits the same trained single block to
    # be evaluated at other R, including the R=1 training baseline.
    return LoopGPT(config, spec["repeats"], spec["dropout"], "legacy_unique").to(device)


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_name(spec):
    return (
        f"w{spec['world']}-i{spec['initialization']}-d{spec['width']}"
        f"-l{spec['layers']}-r{spec['repeats']}-n{spec['composition_count']}-s{spec['steps']}"
    )


def role_coverage(world):
    first = {(int(h), int(r)) for h, r, _b, _r2, _t in world["train_composite"]}
    second = {(int(b), int(r)) for _h, _r1, b, r, _t in world["train_composite"]}
    rows = world["familiar_test"]
    return {
        "first_atoms_used": len(first),
        "second_atoms_used": len(second),
        "familiar_both_composition_roles": float(
            np.mean(
                [
                    (int(h), int(r1)) in first and (int(b), int(r2)) in second
                    for h, r1, b, r2, _t in rows
                ]
            )
        ),
    }


def train(spec, out, source, device="cuda:0"):
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


def compare_predictions(actual, saved):
    maximum = 0.0
    assert set(actual) == set(saved.files)
    for key, value in actual.items():
        if key.endswith("answer_nll"):
            maximum = max(maximum, float(np.abs(value - saved[key]).max()))
            np.testing.assert_allclose(value, saved[key], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(value, saved[key])
    return maximum


def compare_metrics(actual, saved):
    for task, values in actual.items():
        for key, value in values.items():
            if key == "answer_nll":
                np.testing.assert_allclose(value, saved[task][key], rtol=1e-5, atol=1e-5)
            else:
                assert value == saved[task][key]


def audit(out, device="cuda:0"):
    out = Path(out)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    result = json.loads((out / "complete.json").read_text())
    assert file_hash(out / "latest.pt") == result["checkpoint_sha256"]
    saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    spec = saved["spec"]
    world = build_world(spec)
    assert data_digest(world) == result["data_sha256"] == saved["data_sha256"]
    assert saved["step"] == spec["steps"]
    model = construct(spec, device)
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
    checked = {
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
