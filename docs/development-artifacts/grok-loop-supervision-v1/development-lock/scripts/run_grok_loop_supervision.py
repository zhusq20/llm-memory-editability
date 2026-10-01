#!/usr/bin/env python3
"""Freeze, execute, and audit paired loop-supervision continuations."""

import argparse
import fcntl
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grok_loop_supervision import (
    digest,
    evaluate,
    from_state,
    load_world,
    run,
    tensor_digest,
)
from llm_memory_editability.grok_multihop_data import audit_world

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/grok-loop-supervision-v1"
DEPENDENCIES = [
    "scripts/run_grok_loop_supervision.py",
    "scripts/preflight_grok_loop_supervision.py",
    "src/llm_memory_editability/grok_loop_supervision.py",
    "src/llm_memory_editability/grok_loop_model.py",
    "src/llm_memory_editability/grok_loop_data.py",
    "src/llm_memory_editability/grok_multihop.py",
    "src/llm_memory_editability/grok_multihop_data.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_depth_data.py",
    "src/llm_memory_editability/bios_model.py",
    "tests/test_grok_loop_supervision.py",
    "docs/development-artifacts/grok-loop-supervision-v1/preregistration.md",
]


def freeze(config_path, config):
    lock_path = ROOT / config["source_lock"]
    files = {p: digest(ROOT / p) for p in DEPENDENCIES + [str(config_path)]}
    origins = {}
    for s in config["runs"].values():
        for filename in (
            "latest.pt",
            "world.npz",
            "world-metadata.json",
            "metadata.json",
            "complete.json",
            "predictions-0128000.npz",
        ):
            p = str(Path(s["source"]) / filename)
            origins[p] = digest(ROOT / p)
    if lock_path.exists():
        old = json.loads(lock_path.read_text())
        if old["files"] != files or old["origins"] != origins:
            raise ValueError("Existing lock differs; create an explicit amendment")
        return
    snapshot = lock_path.with_suffix("")
    for name in files:
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write_json(
        lock_path,
        {
            "created_utc": utc(),
            "files": files,
            "origins": origins,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )
    print(json.dumps({"lock": str(lock_path), "sources": len(files), "origin_files": len(origins)}))


def verify_lock(config):
    lock = json.loads((ROOT / config["source_lock"]).read_text())
    for name, expected in {**lock["files"], **lock["origins"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Source changed: {name}")
    return lock


def native_audit(source, device):
    state = torch.load(source / "latest.pt", map_location=device, weights_only=False)
    model, world = from_state(state, device), load_world(source)
    result = {"source_model_sha256": tensor_digest(state["model"]), "splits": {}}
    with np.load(source / "predictions-0128000.npz") as z:
        for name in ("atomic", "train_composite", "test_composite", "ood_composite"):
            _, pred = evaluate(model, world[name], device, 2)
            equal = all(
                np.array_equal(pred[k], z[name + "_" + k]) for k in ("answer", "stop", "target")
            )
            error = float(np.abs(pred["nll"] - z[name + "_nll"]).max())
            if not equal or error > 1e-5:
                raise ValueError(f"Native prediction mismatch {name}: {equal}, {error}")
            result["splits"][name] = {
                "n": len(world[name]),
                "exact_outputs": equal,
                "max_nll_error": error,
            }
    return result


def audit_run(out, device):
    """Independent CPU scoring from saved logits, plus GPU endpoint reload."""
    complete = json.loads((out / "complete.json").read_text())
    state = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    model = from_state(state, device)
    world = load_world(ROOT / complete["spec"]["source"])
    if tensor_digest(model.state_dict()) != complete["final_model_sha256"]:
        raise ValueError("Final model hash mismatch")
    checks, max_nll, max_accuracy = 0, 0.0, 0.0
    for record in json.loads((out / "learning.json").read_text()):
        with np.load(out / f"predictions-{record['step']:07d}.npz") as z:
            for key, metric in record["evaluations"].items():
                if key.startswith(("id_atomic_", "ood_atomic_")):
                    continue
                logits = z[key + "_logits"].astype(float)
                target = z[key + "_target"]
                labels = np.c_[target, np.ones(len(target), dtype=int)]
                shifted = logits - logits.max(-1, keepdims=True)
                lse = np.log(np.exp(shifted).sum(-1))
                nll = lse - np.take_along_axis(shifted, labels[..., None], axis=-1).squeeze(-1)
                error = float(np.abs(nll - z[key + "_nll"]).max())
                max_nll = max(max_nll, error)
                answer = logits[:, 0].argmax(-1)
                stop = z[key + "_generated_eos_logits"].argmax(-1)
                if not np.array_equal(answer, z[key + "_answer"]) or not np.array_equal(
                    stop, z[key + "_stop"]
                ):
                    raise ValueError("Saved generation does not match logits")
                accuracy = float(((answer == target) & (stop == 1)).mean())
                max_accuracy = max(max_accuracy, abs(accuracy - metric["accuracy"]))
                if error > 2e-5 or accuracy != metric["accuracy"]:
                    raise ValueError("Independent scoring mismatch")
                checks += 1
    reload_checks = []
    with np.load(out / f"predictions-{state['step']:07d}.npz") as z:
        for name in ("atomic", "test_full_composite", "ood_composite", "train_composite"):
            for r in [2, 3, 4] if name == "train_composite" else [2, 4, 8, 16]:
                _, pred = evaluate(model, world[name], device, r)
                key = f"{name}_r{r}"
                if not all(
                    np.array_equal(pred[k], z[key + "_" + k])
                    for k in ("answer", "stop", "target", "nll")
                ):
                    raise ValueError(f"Endpoint reload mismatch {key}")
                reload_checks.append(key)
    result = {
        "passed": True,
        "scoring_groups": checks,
        "max_cpu_nll_error": max_nll,
        "max_accuracy_error": max_accuracy,
        "reload_groups": reload_checks,
        "utc": utc(),
    }
    write_json(out / "audit.json", result)
    print(json.dumps({"out": str(out), **result}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "train", "audit"])
    p.add_argument("--config", required=True)
    p.add_argument("--run")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    if args.action == "freeze":
        freeze(args.config, config)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(args.device))
    lock = verify_lock(config)
    spec = {**config["base"], **config["runs"][args.run]}
    out = ROOT / config["output_root"] / args.run
    if args.action == "audit":
        audit_run(out, torch.device(args.device))
        return
    if sorted(set(spec["nodes"])) != spec["nodes"] or spec["nodes"][-1] != spec["steps"]:
        raise ValueError("Invalid node/budget contract")
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    handle = (out / "run.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "tf32": True,
        "precision": "FP32",
        "device": args.device,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    source = ROOT / spec["source"]
    saved_env = json.loads((source / "environment.json").read_text())
    for key in ("torch", "numpy", "cuda", "gpu"):
        if env[key] != saved_env[key]:
            raise ValueError(f"Changed historical environment: {key}")
    if not args.resume:
        for name in lock["files"]:
            dest = out / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        write_json(
            out / "metadata.json",
            {"spec": spec, "lock": lock, "started_utc": utc(), "pid": os.getpid()},
        )
        write_json(out / "environment.json", env)
        world = load_world(source)
        write_json(out / "data-audit.json", audit_world(world))
        write_json(out / "native-audit.json", native_audit(source, torch.device(args.device)))
    run(spec, out, args.device, args.resume)


if __name__ == "__main__":
    main()
