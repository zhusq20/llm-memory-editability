#!/usr/bin/env python3
"""Prepare, run and audit the fixed eight-trajectory architecture bridge study."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.architecture_bridge_toy import (
    ToyLM,
    diagnostics,
    evaluate,
    exposure,
    flops_estimate,
    initialize,
    parameter_digest,
    save_npz,
)
from llm_memory_editability.twohop_depth import (
    EOS,
    audit_world,
    digest,
    make_stream,
    read,
    training_tensors,
    world_arrays,
    write,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/architecture-bridge-toy-v1.json"
DATA = ROOT / "data/architecture-bridge-toy-v1"
RESULTS = ROOT / "results/architecture-bridge-toy-v1"
ART = ROOT / "docs/development-artifacts/architecture-bridge-toy-v1"
SOURCES = (
    "configs/architecture-bridge-toy-v1.json",
    "src/llm_memory_editability/architecture_bridge_toy.py",
    "src/llm_memory_editability/twohop_depth.py",
    "src/llm_memory_editability/bios_model.py",
    "scripts/run_architecture_bridge_toy.py",
    "tests/test_architecture_bridge_toy.py",
)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def configure(cfg):
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def validate_lock():
    lock = read(ART / "lock.json")
    for group in ("sources", "data"):
        for relative, expected in lock[group].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f"Frozen {group} changed: {relative}")
    return lock


def prepare(cfg):
    if (ART / "lock.json").exists():
        return validate_lock()
    for path in (DATA, RESULTS, ART):
        path.mkdir(parents=True, exist_ok=True)
    source_hashes, data_hashes, audits = {}, {}, {}
    for relative in SOURCES:
        source = ROOT / relative
        destination = ART / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = digest(source)
    for world_seed in cfg["worlds"]:
        world = world_arrays(world_seed, cfg)
        audits[str(world_seed)] = audit_world(world, cfg)
        path = DATA / f"world-{world_seed}.npz"
        save_npz(path, world)
        data_hashes[str(path.relative_to(ROOT))] = digest(path)
        for seed in cfg["initializations"]:
            stream = make_stream(world, seed, cfg)
            path = DATA / f"stream-{world_seed}-{seed}.npz"
            save_npz(path, stream)
            data_hashes[str(path.relative_to(ROOT))] = digest(path)
    lock = {
        "created_utc": utc(),
        "config": cfg,
        "sources": source_hashes,
        "data": data_hashes,
        "world_audits": audits,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "gpu": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv"],
                text=True,
            ).strip(),
        },
        "policy": cfg["policy"],
    }
    write(ART / "lock.json", lock)
    print(json.dumps({"phase": "prepared", "lock_sha256": digest(ART / "lock.json")}), flush=True)
    return lock


def save_checkpoint(path, model, optimizer, step, training_seconds):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "training_seconds": training_seconds,
            "parameter_sha256": parameter_digest(model),
        },
        path,
    )


def run_one(cfg, world_seed, seed, architecture, device):
    name = f"w{world_seed}-s{seed}-{architecture['name']}"
    run_dir = RESULTS / name
    report_dir = ART / name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if (report_dir / "complete.json").exists():
        return read(report_dir / "complete.json")
    start_utc, start_time = utc(), time.perf_counter()
    world = dict(np.load(DATA / f"world-{world_seed}.npz"))
    stream_path = DATA / f"stream-{world_seed}-{seed}.npz"
    stream_np = dict(np.load(stream_path))
    streams = {key: torch.as_tensor(value, device=device) for key, value in stream_np.items()}
    tensors = training_tensors(world, device)
    model = ToyLM(architecture, cfg)
    initialize(model, seed)
    model.to(device)
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=cfg["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=str(device).startswith("cuda"),
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start_step, training_seconds, rows, nodes = 0, 0.0, [], {}
    checkpoint_paths = [run_dir / f"checkpoint-{step}.pt" for step in cfg["nodes"]]
    existing = [path for path in checkpoint_paths if path.exists()]
    if existing:
        saved = torch.load(existing[-1], map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step, training_seconds = saved["step"], saved["training_seconds"]
        if (report_dir / "nodes.json").exists():
            nodes = read(report_dir / "nodes.json")
        if (run_dir / "curve.json").exists():
            rows = [row for row in read(run_dir / "curve.json") if row["step"] <= start_step]

    def node(step):
        sync(device)
        checkpoint = run_dir / f"checkpoint-{step}.pt"
        if not checkpoint.exists():
            save_checkpoint(checkpoint, model, optimizer, step, training_seconds)
        metrics, arrays = evaluate(model, world, cfg, device)
        prediction = run_dir / f"predictions-{step}.npz"
        save_npz(prediction, arrays)
        nodes[str(step)] = {
            "step": step,
            "exposure": exposure(cfg, step),
            "metrics": metrics,
            "checkpoint_sha256": digest(checkpoint),
            "prediction_sha256": digest(prediction),
        }
        write(report_dir / "nodes.json", nodes)
        write(run_dir / "curve.json", rows)
        print(json.dumps({"run": name, "node": step, "metrics": metrics}), flush=True)

    current_step = start_step
    try:
        if str(start_step) not in nodes:
            node(start_step)
        segment_start = time.perf_counter()
        for step in range(start_step, cfg["steps"]):
            model.train()
            left, right = step * cfg["batch_per_kind"], (step + 1) * cfg["batch_per_kind"]
            parts = [tuple(t[streams[key][left:right]] for t in tensors[key]) for key in streams]
            x, positions, labels = (torch.cat([part[i] for part in parts]) for i in range(3))
            lr = cfg["lr"] * min(1.0, (step + 1) / cfg["warmup"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, positions=positions)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()
            current_step = step + 1
            if current_step % 32 == 0:
                rows.append(
                    {"step": current_step, "loss": float(loss.detach()), "grad_norm": float(norm)}
                )
            if current_step % 128 == 0:
                write(
                    report_dir / "status.json",
                    {"state": "running", "step": current_step, "started_utc": start_utc},
                )
                print(
                    json.dumps({"run": name, "step": current_step, "loss": rows[-1]["loss"]}),
                    flush=True,
                )
            if current_step in cfg["nodes"]:
                sync(device)
                training_seconds += time.perf_counter() - segment_start
                node(current_step)
                segment_start = time.perf_counter()
        diagnostic_start = time.perf_counter()
        before = parameter_digest(model)
        diagnostic_metrics, diagnostic_arrays = diagnostics(model, world, cfg, device)
        diagnostic_path = run_dir / "diagnostics.npz"
        save_npz(diagnostic_path, diagnostic_arrays)
        write(report_dir / "diagnostics.json", diagnostic_metrics)
        after = parameter_digest(model)
        if before != after:
            raise RuntimeError("Diagnostics changed model parameters")
        sync(device)
        result = {
            "name": name,
            "state": "complete",
            "world": world_seed,
            "initialization": seed,
            "architecture": architecture,
            "started_utc": start_utc,
            "completed_utc": utc(),
            "device": str(device),
            "dtype": cfg["dtype"],
            "stream_sha256": digest(stream_path),
            "exposure": exposure(cfg, cfg["steps"]),
            "compute": flops_estimate(model, cfg, cfg["steps"]),
            "training_seconds": training_seconds,
            "diagnostic_seconds": time.perf_counter() - diagnostic_start,
            "total_seconds_this_attempt": time.perf_counter() - start_time,
            "max_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device)
            if str(device).startswith("cuda")
            else None,
            "final_parameter_sha256": after,
            "diagnostic_sha256": digest(diagnostic_path),
            "diagnostic_condition_count": len(diagnostic_metrics),
            "diagnostic_case_count": len(world["cases"]),
            "nodes": nodes,
        }
        write(report_dir / "complete.json", result)
        write(report_dir / "status.json", {"state": "complete", "step": cfg["steps"]})
        return result
    except Exception:
        failure = {
            "run": name,
            "time_utc": utc(),
            "step": current_step,
            "seconds_this_attempt": time.perf_counter() - start_time,
            "traceback": traceback.format_exc(),
        }
        path = ART / "failures.json"
        previous = read(path) if path.exists() else []
        write(path, [*previous, failure])
        write(report_dir / "status.json", {"state": "failed", **failure})
        raise


def audit(cfg):
    validate_lock()
    runs, checks = [], []
    for world_seed in cfg["worlds"]:
        world = dict(np.load(DATA / f"world-{world_seed}.npz"))
        audit_world(world, cfg)
        for seed in cfg["initializations"]:
            initial = {}
            streams = []
            for architecture in cfg["architectures"]:
                name = f"w{world_seed}-s{seed}-{architecture['name']}"
                run_dir, report = RESULTS / name, ART / name
                result = read(report / "complete.json")
                assert result["state"] == "complete"
                assert result["exposure"] == exposure(cfg, cfg["steps"])
                assert set(result["nodes"]) == {str(step) for step in cfg["nodes"]}
                streams.append(result["stream_sha256"])
                for step in cfg["nodes"]:
                    saved = result["nodes"][str(step)]
                    assert digest(run_dir / f"checkpoint-{step}.pt") == saved["checkpoint_sha256"]
                    assert digest(run_dir / f"predictions-{step}.npz") == saved["prediction_sha256"]
                    arrays = dict(np.load(run_dir / f"predictions-{step}.npz"))
                    for key, truth in (
                        ("atomic", world["atomic_y"]),
                        ("train", world["composite_y"][world["train_mask"]]),
                        ("test", world["composite_y"][~world["train_mask"]]),
                    ):
                        exact = (arrays[f"{key}_pred"] == truth) & (arrays[f"{key}_eos"] == EOS)
                        assert float(exact.mean()) == saved["metrics"][key]["exact_answer_eos"]
                    y = world["composite_y"][~world["train_mask"]]
                    two = (
                        (arrays["two_call_pred"] == y)
                        & (arrays["two_call_eos"] == EOS)
                        & (arrays["two_call_bridge_eos"] == EOS)
                    )
                    assert float(two.mean()) == saved["metrics"]["two_call"]["exact_answer_eos"]
                initial[architecture["name"]] = torch.load(
                    run_dir / "checkpoint-0.pt", map_location="cpu", weights_only=False
                )["model"]
                d = dict(np.load(run_dir / "diagnostics.npz"))
                dm = read(report / "diagnostics.json")
                assert digest(run_dir / "diagnostics.npz") == result["diagnostic_sha256"]
                assert len(dm) == 1 + len(cfg["patch_targets"]) * len(cfg["patch_donors"])
                assert len(d["case_ids"]) == cfg["diagnostic_cases"]
                assert np.all(~world["train_mask"][d["case_ids"]])
                for target in cfg["patch_targets"]:
                    identity = d[f"{target}_identity_logits"]
                    assert np.max(np.abs(identity - d["baseline_logits"])) <= 2e-5
                    assert np.array_equal(d[f"{target}_identity_pred"], d["baseline_pred"])
                    assert np.array_equal(d[f"{target}_identity_eos"], d["baseline_eos"])
                    for donor in ("correct", "wrong"):
                        for kind in ("mlp", "mixer") if target == "joint" else (target,):
                            assert np.allclose(
                                d[f"{target}_{donor}_{kind}_norm"],
                                d[f"{target}_random_{donor}_{kind}_norm"],
                                rtol=1e-5,
                                atol=1e-6,
                            )
                for condition, entry in dm.items():
                    exact = (d[f"{condition}_pred"] == d["target"]) & (d[f"{condition}_eos"] == EOS)
                    assert float(exact.mean()) == entry["correct_exact"]
                checks.append({"run": name, "nodes": 3, "all_checks_passed": True})
                runs.append(result)
            assert len(set(streams)) == 1
            left, right = initial.values()
            common = [key for key in left if key in right and left[key].shape == right[key].shape]
            assert common and all(torch.equal(left[key], right[key]) for key in common)
    compact = []
    for run in runs:
        compact.append(
            {
                "name": run["name"],
                "world": run["world"],
                "seed": run["initialization"],
                "architecture": run["architecture"]["name"],
                "parameters": run["compute"]["parameter_count"],
                "training_seconds": run["training_seconds"],
                "max_allocated_cuda_bytes": run["max_allocated_cuda_bytes"],
                "metrics": run["nodes"][str(cfg["steps"])]["metrics"],
                "diagnostics": read(ART / run["name"] / "diagnostics.json"),
            }
        )
    comparison = {}
    for architecture in cfg["architectures"]:
        selected = [row for row in compact if row["architecture"] == architecture["name"]]
        comparison[architecture["name"]] = {
            "runs": len(selected),
            "independent_worlds": len(cfg["worlds"]),
            "mean_final_exact": {
                key: float(np.mean([row["metrics"][key]["exact_answer_eos"] for row in selected]))
                for key in ("atomic", "train", "test", "two_call")
            },
        }
    result = {
        "completed_utc": utc(),
        "state": "complete",
        "runs": len(runs),
        "nodes": len(runs) * len(cfg["nodes"]),
        "diagnostic_generations": sum(
            r["diagnostic_condition_count"] * r["diagnostic_case_count"] for r in runs
        ),
        "total_training_steps": sum(r["exposure"]["steps"] for r in runs),
        "total_valid_input_tokens": sum(r["exposure"]["valid_input_tokens"] for r in runs),
        "total_processed_tokens": sum(
            r["exposure"]["processed_tokens_including_padding"] for r in runs
        ),
        "training_seconds_sum": sum(r["training_seconds"] for r in runs),
        "flops_approx_sum": sum(r["compute"]["total_approx"] for r in runs),
        "failure_count": len(read(ART / "failures.json"))
        if (ART / "failures.json").exists()
        else 0,
        "checks": checks,
        "comparison": comparison,
        "interpretation": (
            "Two new independent worlds with paired seeds; seeds, queries, layers and "
            "interventions are not independent worlds. Fixed short-budget symbolic models; "
            "no claims about pretrained Qwen, GLM, Kimi or general architecture superiority."
        ),
    }
    write(ART / "summary.json", {"overview": result, "runs": compact})
    write(ART / "audit.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "audit", "all"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = read(CONFIG)
    configure(cfg)
    if args.phase in ("prepare", "all"):
        prepare(cfg)
    if args.phase in ("run", "all"):
        validate_lock()
        device = args.device or cfg["device"]
        for world_seed in cfg["worlds"]:
            for seed in cfg["initializations"]:
                for architecture in cfg["architectures"]:
                    run_one(cfg, world_seed, seed, architecture, device)
    if args.phase in ("audit", "all"):
        audit(cfg)


if __name__ == "__main__":
    # Required by deterministic CUDA GEMM; use a task-specific default only.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
