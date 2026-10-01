#!/usr/bin/env python3
"""Prepare, train, and supervise the depth/composition experiment."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import itertools
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from llm_memory_editability.bios_model import matmul_flops
from llm_memory_editability.twohop_depth import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    ROOT,
    audit_world,
    build_model,
    diagnostics,
    digest,
    evaluate,
    make_stream,
    optimizer_for,
    read,
    run_name,
    train_step,
    training_tensors,
    world_arrays,
    write,
)


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def setup(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def prepare():
    if (ART / "lock.json").exists():
        raise FileExistsError("Prepared batch is immutable; use a new experiment for changes")
    cfg = read(CONFIG)
    DATA.mkdir(parents=True, exist_ok=True)
    audits, streams = {}, {}
    for world in cfg["worlds"]:
        w = world_arrays(world, cfg)
        audits[str(world)] = audit_world(w, cfg)
        np.savez_compressed(DATA / f"world-{world}.npz", **w)
        for seed in cfg["initializations"]:
            stream = make_stream(w, world * 100 + seed, cfg)
            name = f"stream-{world}-{seed}.npz"
            np.savez_compressed(DATA / name, **stream)
            streams[name] = {}
            for key, arr in stream.items():
                counts = np.bincount(arr)
                assert counts.min() == counts.max()
                streams[name][key + "_exposures_per_example"] = int(counts[0])
    counts = {}
    for arch in cfg["architectures"]:
        model = build_model(arch, cfg)
        counts[arch["name"]] = sum(p.numel() for p in model.parameters())
    jobs = [
        {"world": w, "seed": s, "arch": a, "name": run_name(w, s, a["name"])}
        for w, s, a in itertools.product(
            cfg["worlds"], cfg["initializations"], cfg["architectures"]
        )
    ]
    source_paths = [
        CONFIG,
        ROOT / "src/llm_memory_editability/__init__.py",
        ROOT / "src/llm_memory_editability/experiments.py",
        ROOT / "src/llm_memory_editability/low_rank.py",
        ROOT / "src/llm_memory_editability/twohop_depth.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "scripts/run_twohop_depth.py",
        ROOT / "scripts/report_twohop_depth.py",
        ROOT / "tests/test_twohop_depth.py",
    ]
    snapshot = RESULTS / "execution-source"
    for path in source_paths + [ROOT / "docs/hebbian-learning-plan-v1.md"]:
        destination = snapshot / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    files = {str(p.relative_to(ROOT)): digest(p) for p in source_paths + sorted(DATA.glob("*.npz"))}
    write(
        ART / "lock.json",
        {
            "created_at": now(),
            "config": cfg,
            "jobs": jobs,
            "parameters": counts,
            "world_audits": audits,
            "streams": streams,
            "files": files,
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            },
        },
    )
    print(
        jsonify({"prepared": len(jobs), "parameters": counts, "world_audits": audits}), flush=True
    )


def jsonify(value):
    import json

    return json.dumps(value, ensure_ascii=False)


def save_torch(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def run_job(job, cfg, device):
    out = RESULTS / job["name"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "worker.lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "complete.json").exists():
            return
        setup(job["seed"])
        w = dict(np.load(DATA / f"world-{job['world']}.npz"))
        stream = np.load(DATA / f"stream-{job['world']}-{job['seed']}.npz")
        indices = {k: torch.as_tensor(stream[k], device=device) for k in stream.files}
        model = build_model(job["arch"], cfg).to(device)
        optimizer = optimizer_for(model, cfg, device)
        tensors = training_tensors(w, device)
        step, previous_seconds = 0, 0.0
        resume = out / "resume.pt"
        if resume.exists():
            ckpt = torch.load(resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            step, previous_seconds = ckpt["step"], ckpt["training_seconds"]
        started = time.monotonic()
        seconds, last_loss = previous_seconds, None
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        write(
            out / "metadata.json",
            {"job": job, "config": model.config_dict(), "device": device, "started": now()},
        )
        for node in cfg["nodes"]:
            if node < step:
                continue
            model.train()
            train_start = time.monotonic()
            while step < node:
                b = cfg["batch_per_kind"]
                batch_indices = {k: v[step * b : (step + 1) * b] for k, v in indices.items()}
                loss, norm = train_step(model, optimizer, tensors, batch_indices, step, cfg, device)
                step += 1
                if step % 128 == 0:
                    last_loss = float(loss)
                    if not np.isfinite(last_loss):
                        raise FloatingPointError(f"Nonfinite training loss at {step}")
                    with (out / "training.jsonl").open("a") as f:
                        f.write(
                            jsonify(
                                {
                                    "step": step,
                                    "loss": last_loss,
                                    "grad_norm": float(norm),
                                    "time": now(),
                                }
                            )
                            + "\n"
                        )
                    print(
                        jsonify({"run": job["name"], "step": step, "loss": last_loss}), flush=True
                    )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            seconds += time.monotonic() - train_start
            model.eval()
            evaluation_start = time.monotonic()
            metrics, predictions = evaluate(model, w, cfg, device)
            diag = diagnostics(model, w, cfg, device)
            for layer in range(job["arch"]["layers"] + 1):
                assert np.max(diag[f"identity_{layer}_max_delta"]) == 0
            for kind in ("bridge", "same_bridge", "random", "source", "prefix"):
                assert np.max(diag[f"{kind}_{job['arch']['layers']}_max_delta"]) == 0
            np.savez_compressed(out / f"predictions-{node}.npz", **predictions)
            np.savez_compressed(out / f"diagnostics-{node}.npz", **diag)
            state = {
                "step": node,
                "job": job,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_seconds": seconds,
            }
            save_torch(
                out / f"model-{node}.pt", {k: v for k, v in state.items() if k != "optimizer"}
            )
            save_torch(resume, state)
            row = {
                "step": node,
                "metrics": metrics,
                "training_seconds": seconds,
                "evaluation_seconds": time.monotonic() - evaluation_start,
                "training_flops_estimate": node
                * matmul_flops(model.config, 2 * cfg["batch_per_kind"]),
                "training_examples": node * 2 * cfg["batch_per_kind"],
                "supervised_tokens": node * 4 * cfg["batch_per_kind"],
                "executed_input_tokens": node * 12 * cfg["batch_per_kind"],
                "last_loss": last_loss,
                "time": now(),
            }
            write(out / f"node-{node}.json", row)
            print(jsonify({"run": job["name"], **row}), flush=True)
        files = (
            sorted(out.glob("node-*.json"))
            + sorted(out.glob("*.npz"))
            + sorted(out.glob("model-*.pt"))
        )
        write(
            out / "complete.json",
            {
                "finished": now(),
                "step": step,
                "nodes": cfg["nodes"],
                "job": job,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": seconds,
                "invocation_seconds": time.monotonic() - started,
                "max_gpu_bytes": torch.cuda.max_memory_allocated(device)
                if str(device).startswith("cuda")
                else 0,
                "files": {p.name: digest(p) for p in files},
            },
        )


def worker(shard, shards, device):
    lock = read(ART / "lock.json")
    for path, expected in lock["files"].items():
        if path.startswith("data/"):
            assert digest(ROOT / path) == expected
    for i, job in enumerate(lock["jobs"]):
        if i % shards != shard:
            continue
        try:
            run_job(job, lock["config"], device)
        except Exception:
            import traceback

            write(
                RESULTS / job["name"] / "failure.json",
                {"time": now(), "traceback": traceback.format_exc()},
            )
            raise


def launch(gpus):
    ART.mkdir(parents=True, exist_ok=True)
    source = RESULTS / "execution-source"
    env = os.environ.copy()
    env.update(
        TWOHOP_PROJECT_ROOT=str(ROOT),
        PYTHONPATH=str(source / "src"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    jobs = []
    for shard, gpu in enumerate(gpus):
        command = [
            sys.executable,
            "-u",
            str(source / "scripts/run_twohop_depth.py"),
            "worker",
            "--shard",
            str(shard),
            "--shards",
            str(len(gpus)),
            "--device",
            f"cuda:{gpu}",
        ]
        with (ART / f"worker-{shard}.log").open("a") as log:
            p = subprocess.Popen(
                command,
                env=env,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        jobs.append({"shard": shard, "gpu": gpu, "pid": p.pid, "command": command})
    write(ART / "launch.json", {"time": now(), "workers": jobs})
    print(jsonify(jobs), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "launch", "worker", "status"])
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 4, 5])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "launch":
        launch(args.gpus)
    elif args.stage == "worker":
        worker(args.shard, args.shards, args.device)
    else:
        lock = read(ART / "lock.json")
        rows = []
        for job in lock["jobs"]:
            out = RESULTS / job["name"]
            nodes = sorted(int(p.stem.split("-")[1]) for p in out.glob("node-*.json"))
            rows.append(
                {
                    "run": job["name"],
                    "nodes": nodes,
                    "complete": (out / "complete.json").exists(),
                    "failed": (out / "failure.json").exists(),
                }
            )
        print(jsonify(rows))


if __name__ == "__main__":
    main()
