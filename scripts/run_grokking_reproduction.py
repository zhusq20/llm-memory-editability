"""Prepare, validate, supervise and independently audit the full-size replication."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GPT2Config

from llm_memory_editability.grok_depth import EpochStream, utc, write_json
from llm_memory_editability.grokking_reproduction import (
    ARTIFACTS,
    ROOT,
    SOURCE_FILES,
    ReproductionGPT,
    ReproductionStep,
    audit_support_roles,
    construct,
    digest,
    encode_rows,
    evaluate_rows,
    model_digest,
    optimizer_for,
    prepare_data,
    train_run,
)

CONFIG = Path("configs/grokking-reproduction-development-v1.json")


def read_config():
    return json.loads(CONFIG.read_text())


def preflight():
    """Check dropout/RNG/Adam continuation, then time the original-size model."""
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(917)
    cfg = GPT2Config(vocab_size=101, n_positions=16, n_embd=32, n_layer=2, n_head=4)
    tokens = torch.randint(0, 101, (64, 4), device="cuda")
    positions = torch.tensor([[2, 3]], device="cuda").expand(64, 2)
    labels = torch.randint(0, 101, (64, 2), device="cuda")
    table = tokens, positions, labels
    model = ReproductionGPT(cfg, repeats=2).cuda()
    lr = torch.tensor(1e-4, device="cuda")
    opt = optimizer_for(model, lr, 0.1)
    graph = ReproductionStep(model, opt, table, batch_size=16)
    stream = EpochStream(64, 918)
    for _ in range(4):
        graph(torch.as_tensor(stream.take(16), device="cuda"))
    torch.cuda.synchronize()
    state = copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict())
    rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    stream_state = stream.state_dict()
    for _ in range(4):
        graph(torch.as_tensor(stream.take(16), device="cuda"))
    torch.cuda.synchronize()
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    resumed = ReproductionGPT(copy.deepcopy(cfg), repeats=2).cuda()
    resumed.load_state_dict(state[0])
    resumed_opt = optimizer_for(resumed, torch.tensor(1e-4, device="cuda"), 0.1)
    resumed_opt.load_state_dict(state[1])
    for group in resumed_opt.param_groups:
        group["lr"] = resumed_opt.param_groups[0]["lr"]
    restored_stream = EpochStream(64, 991)
    restored_stream.load_state_dict(stream_state)
    torch.set_rng_state(rng[0])
    torch.cuda.set_rng_state(rng[1])
    restored_graph = ReproductionStep(resumed, resumed_opt, table, batch_size=16)
    for _ in range(4):
        restored_graph(torch.as_tensor(restored_stream.take(16), device="cuda"))
    torch.cuda.synchronize()
    resume_error = max(
        float((v - expected[k]).abs().max()) for k, v in resumed.state_dict().items()
    )
    assert resume_error == 0.0, f"Dropout/Adam/CUDA Graph continuation mismatch: {resume_error}"
    del graph, restored_graph, model, resumed, opt, resumed_opt, state, expected
    torch.cuda.empty_cache()
    config = read_config()
    metadata = prepare_data(config)
    rows = np.load(ROOT / "data/world.npz")["atoms"][:2048]
    table = tuple(torch.as_tensor(a, device="cuda") for a in encode_rows(rows, metadata))
    model = construct(config["runs"][0], metadata, "cuda")
    params = sum(p.numel() for p in model.parameters())
    optimizer = optimizer_for(model, torch.tensor(1e-4, device="cuda"), 0.1)
    graph = ReproductionStep(model, optimizer, table, 512)
    indices = torch.arange(512, device="cuda")
    initial = float(graph(indices))
    torch.cuda.synchronize()
    started = time.perf_counter()
    for i in range(100):
        loss = graph(indices + (i % 4) * 512)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    final = float(loss)
    assert np.isfinite(initial) and np.isfinite(final)
    result = {
        "passed": True,
        "created_utc": utc(),
        "dropout_resume_max_parameter_error": resume_error,
        "benchmark_updates": 101,
        "benchmark_parameters": params,
        "benchmark_seconds_per_update": elapsed / 100,
        "estimated_1_5M_training_hours_single_gpu": elapsed / 100 * 1500000 / 3600,
        "benchmark_initial_loss": initial,
        "benchmark_final_loss": final,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "precision": "bf16 AMP",
    }
    write_json(ARTIFACTS / "preflight.json", result)
    print(json.dumps(result, indent=2), flush=True)


def worker_preflight():
    """Exercise partial epochs, worker restore and endpoint reload on a tiny fixture."""
    import llm_memory_editability.grokking_reproduction as reproduction

    global ROOT, construct
    original_root, original_construct = ROOT, construct
    fixture = ROOT / "engineering-worker-partial-epochs"
    assert not fixture.exists(), "Preserve prior engineering outcomes; use a new fixture name"
    data = fixture / "data"
    data.mkdir(parents=True)
    world = dict(np.load(ROOT / "data/world.npz"))
    selected = world["train_order"][:342]
    np.savez_compressed(
        data / "world.npz",
        atoms=world["atoms"][:100],
        chains=world["chains"][selected],
        train_order=np.arange(342),
    )
    panels = {k: v[:32] for k, v in dict(np.load(ROOT / "data/panels.npz")).items()}
    np.savez_compressed(data / "panels.npz", **panels)
    metadata = json.loads((ROOT / "data/complete.json").read_text())
    metadata.update(
        atomic_id=95,
        atomic_ood=5,
        world_sha256=digest(data / "world.npz"),
        panels_sha256=digest(data / "panels.npz"),
    )
    write_json(data / "complete.json", metadata)

    def tiny_construct(spec, metadata, device):
        torch.manual_seed(spec["initialization"])
        cfg = GPT2Config(
            vocab_size=metadata["vocab_size"],
            n_positions=1024,
            n_embd=32,
            n_layer=spec["unique_layers"],
            n_head=4,
        )
        return ReproductionGPT(cfg, spec["repeats"]).to(device)

    config = read_config()
    spec = copy.deepcopy(config["runs"][0])
    spec.update(name="uninterrupted", unique_layers=2, steps=32, batch_size=32, warmup=2)
    config.update(evaluation_nodes=[0, 16, 32], checkpoint_nodes=[0, 16, 32])
    ROOT = reproduction.ROOT = fixture
    construct = reproduction.construct = tiny_construct
    try:
        train_run(config, spec, 0)
        resumed = {**spec, "name": "interrupted-resumed"}
        interrupted = {**config, "evaluation_nodes": [0, 16], "checkpoint_nodes": [0, 16]}
        try:
            train_run(interrupted, resumed, 0)
        except AssertionError:
            saved = torch.load(
                fixture / "development/interrupted-resumed/latest.pt", weights_only=False
            )
            assert saved["step"] == 16
            del saved
        else:
            raise AssertionError("Engineering interruption was not exercised")
        train_run(config, resumed, 0)
        values = [
            json.loads((fixture / "development" / s["name"] / "complete.json").read_text())
            for s in [spec, resumed]
        ]
        assert values[0]["model_sha256"] == values[1]["model_sha256"]
        assert values[0]["full_endpoint"] == values[1]["full_endpoint"]
        for s in [spec, resumed]:
            audit(s, 0)
        exposure = np.load(fixture / "development/uninterrupted/exposure-0000032.npz")
        assert int(exposure["counts"].sum()) == 1012
        result = {
            "passed": True,
            "created_utc": utc(),
            "steps_per_run": 32,
            "interruption_step": 16,
            "partial_epoch_batches_per_run": 2,
            "actual_examples_per_run": 1012,
            "exact_model_sha256": values[0]["model_sha256"],
            "full_endpoint_metrics_identical": True,
            "endpoint_reload_audits": 2,
            "command": (
                "PYTHONPATH=src .venv/bin/python "
                "scripts/run_grokking_reproduction.py worker-preflight"
            ),
            "scope": (
                "Engineering width32 fixture: original vocabulary/objective/dropout, "
                "100 atoms + 342 true chains; excluded from scientific matrix"
            ),
        }
        write_json(ARTIFACTS / "worker-resume-check.json", result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        ROOT = reproduction.ROOT = original_root
        construct = reproduction.construct = original_construct


def freeze():
    assert json.loads((ARTIFACTS / "preflight.json").read_text())["passed"]
    assert json.loads((ARTIFACTS / "tests.json").read_text())["passed"]
    assert json.loads((ARTIFACTS / "worker-resume-check.json").read_text())["passed"]
    target = ARTIFACTS / "source"
    dependencies = [
        "src/llm_memory_editability/__init__.py",
        "src/llm_memory_editability/grok_depth.py",
        "src/llm_memory_editability/bios_model.py",
        "src/llm_memory_editability/experiments.py",
        "src/llm_memory_editability/low_rank.py",
    ]
    files = SOURCE_FILES + dependencies
    if (ARTIFACTS / "freeze.json").exists():
        manifest = json.loads((ARTIFACTS / "freeze.json").read_text())
        for file, sha in manifest["files"].items():
            assert digest(target / file) == sha
        return
    hashes = {}
    for file in files:
        dst = target / file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dst)
        hashes[file] = digest(dst)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpus": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"], text=True
        ),
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    }
    write_json(ARTIFACTS / "environment.json", environment)
    write_json(
        ARTIFACTS / "freeze.json",
        {
            "frozen_utc": utc(),
            "files": hashes,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "world_sha256": json.loads((ROOT / "data/complete.json").read_text())["world_sha256"],
            "policy": "Frozen development; eight fixed 1.5M runs; all outcomes retained.",
        },
    )


def command_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str((ARTIFACTS / "source/src").resolve())
    env["OMP_NUM_THREADS"] = "4"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def frozen_command(action, index=None, gpu=None):
    args = [
        sys.executable,
        "-u",
        str((ARTIFACTS / "source/scripts/run_grokking_reproduction.py").resolve()),
        action,
    ]
    if index is not None:
        args += ["--index", str(index), "--gpu", str(gpu)]
    return args


def summarize(config):
    runs = []
    for spec in config["runs"]:
        path = ROOT / "development" / spec["name"]
        status = (
            json.loads((path / "status.json").read_text())
            if (path / "status.json").exists()
            else {"state": "pending", "step": 0}
        )
        if (path / "failure.json").exists():
            status["failure"] = json.loads((path / "failure.json").read_text())
            status["state"] = "failed"
        status["independently_reloaded"] = (path / "audit.json").exists()
        runs.append(
            {
                "name": spec["name"],
                "phi": spec["phi"],
                "weight_decay": spec["weight_decay"],
                **status,
            }
        )
    value = {
        "batch": config["batch"],
        "updated_utc": utc(),
        "runs": runs,
        "completed": sum(r["state"] == "complete" and r["independently_reloaded"] for r in runs),
        "registered": len(runs),
        "independent_worlds": 1,
        "initializations": 1,
    }
    write_json(ARTIFACTS / "live-status.json", value)
    return value


def start():
    freeze()
    existing = ROOT / "controller.json"
    if existing.exists():
        pid = json.loads(existing.read_text())["pid"]
        try:
            os.kill(pid, 0)
            print(f"Controller already running: {pid}")
            return
        except ProcessLookupError:
            pass
    with (ROOT / "controller.log").open("a") as log:
        p = subprocess.Popen(
            frozen_command("controller"),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=command_env(),
            start_new_session=True,
            cwd=Path.cwd(),
        )
    write_json(
        existing, {"pid": p.pid, "started_utc": utc(), "command": frozen_command("controller")}
    )
    print(json.dumps({"controller_pid": p.pid, "log": str(ROOT / "controller.log"), "runs": 8}))


def controller():
    config = read_config()
    handles = []
    lock = (ROOT / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_sha = digest(CONFIG)
    frozen_sha = json.loads((ARTIFACTS / "freeze.json").read_text())["files"][str(CONFIG)]
    assert config_sha == frozen_sha, "Working configuration changed after freeze"
    for index, spec in enumerate(config["runs"]):
        path = ROOT / "development" / spec["name"]
        path.mkdir(parents=True, exist_ok=True)
        if (path / "audit.json").exists():
            continue
        action = "audit" if (path / "complete.json").exists() else "worker"
        log = (path / f"{action}-{utc().replace(':', '')}.log").open("w")
        process = subprocess.Popen(
            frozen_command(action, index, index),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=command_env(),
        )
        handles.append(
            {
                "index": index,
                "spec": spec,
                "path": path,
                "process": process,
                "log": log,
                "action": action,
            }
        )
    while handles:
        remaining = []
        for item in handles:
            result = item["process"].poll()
            if result is None:
                remaining.append(item)
                continue
            item["log"].close()
            if result == 0 and item["action"] == "worker":
                item["action"] = "audit"
                item["log"] = (item["path"] / "audit.log").open("w")
                item["process"] = subprocess.Popen(
                    frozen_command("audit", item["index"], item["index"]),
                    stdout=item["log"],
                    stderr=subprocess.STDOUT,
                    env=command_env(),
                )
                remaining.append(item)
            elif result != 0:
                write_json(
                    item["path"] / "failure.json",
                    {"action": item["action"], "exit_code": result, "recorded_utc": utc()},
                )
        handles = remaining
        state = summarize(config)
        write_json(
            ROOT / "controller-state.json",
            {"state": "running", "active": len(handles), "updated_utc": utc()},
        )
        if handles:
            time.sleep(15)
    state = summarize(config)
    completed = state["completed"] == state["registered"]
    write_json(
        ROOT / "controller-state.json",
        {
            "state": "complete" if completed else "finished_with_failures",
            "active": 0,
            "updated_utc": utc(),
        },
    )
    if completed:
        report(config)
        write_json(
            ARTIFACTS / "completion-manifest.json",
            {
                "status": "complete",
                "runs": 8,
                "updates": 12000000,
                "independent_endpoint_reloads": 8,
                "completed_utc": utc(),
                "freeze_sha256": digest(ARTIFACTS / "freeze.json"),
            },
        )


def audit(spec, gpu):
    torch.set_num_threads(4)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{gpu}"
    out = ROOT / "development" / spec["name"]
    complete = json.loads((out / "complete.json").read_text())
    checkpoint = out / f"checkpoint-{spec['steps']:07d}.pt"
    assert digest(checkpoint) == complete["final_checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = json.loads((ROOT / "data/complete.json").read_text())
    model = construct(spec, metadata, device)
    assert model_digest(model) == json.loads((out / "run.json").read_text())["initial_model_sha256"]
    model.load_state_dict(saved["model"])
    assert model_digest(model) == complete["model_sha256"]
    history = json.loads((out / "learning.json").read_text())
    reference = history[-1]["metrics"]
    panels = dict(np.load(ROOT / "data/panels.npz"))
    checked = 0
    for name, rows in panels.items():
        metrics, pred = evaluate_rows(model, rows, metadata, device)
        original = np.load(out / f"predictions-{spec['steps']:07d}-{name}.npz")
        for key in ["answer", "stop", "rows", "nll"]:
            assert np.array_equal(pred[key], original[key]), f"Reload mismatch {name}:{key}"
        assert metrics == reference[name]
        checked += len(rows)
    world = dict(np.load(ROOT / "data/world.npz"))
    selected = world["train_order"][: round(spec["phi"] * metadata["atomic_id"])]
    for name, rows in {
        "atomic_all": world["atoms"],
        "train_ii_all": world["chains"][selected],
    }.items():
        metrics, pred = evaluate_rows(model, rows, metadata, device)
        original = np.load(out / f"endpoint-{name}.npz")
        for key in ["answer", "stop", "rows", "nll"]:
            assert np.array_equal(pred[key], original[key]), f"Full reload mismatch {name}:{key}"
        assert metrics == complete["full_endpoint"][name]
        checked += len(rows)
    exposure = np.load(out / f"exposure-{spec['steps']:07d}.npz")
    counts = exposure["counts"]
    epoch_steps = (len(counts) + spec["batch_size"] - 1) // spec["batch_size"]
    epochs, remainder = divmod(spec["steps"], epoch_steps)
    assert counts.sum() == epochs * len(counts) + remainder * spec["batch_size"]
    assert counts.max() - counts.min() <= 1
    write_json(
        out / "audit.json",
        {
            "passed": True,
            "predictions_recomputed": checked,
            "checkpoint_sha256": digest(checkpoint),
            "completed_utc": utc(),
        },
    )


def report(config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    tasks = ["train_ii", "test_ii", "test_oo"]
    for s in config["runs"]:
        path = ROOT / "development" / s["name"] / "learning.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        row = 0 if s["unique_layers"] == 8 else 1
        for col, task in enumerate(tasks):
            axes[row, col].plot(
                [h["step"] for h in history if h["step"] > 0],
                [100 * h["metrics"][task]["accuracy"] for h in history if h["step"] > 0],
                label=f"phi={s['phi']}, wd={s['weight_decay']}",
            )
            axes[row, col].set_title(f"{'standard8' if row == 0 else 'loop4x2'}: {task}")
            axes[row, col].set_xscale("log")
            axes[row, col].set_ylim(-2, 102)
            axes[row, col].grid(alpha=0.2)
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "learning.png", dpi=150)
    fig.savefig(ARTIFACTS / "learning.pdf")
    plt.close(fig)
    summarize(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "preflight",
            "worker-preflight",
            "freeze",
            "start",
            "controller",
            "worker",
            "audit",
            "status",
            "report",
        ],
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    config = read_config()
    if args.action == "prepare":
        print(json.dumps(prepare_data(config), indent=2))
        audit_support_roles(config)
    elif args.action == "preflight":
        preflight()
    elif args.action == "worker-preflight":
        worker_preflight()
    elif args.action == "freeze":
        freeze()
    elif args.action == "start":
        start()
    elif args.action == "controller":
        controller()
    elif args.action in ["worker", "audit"]:
        spec = config["runs"][args.index]
        try:
            if args.action == "worker":
                train_run(config, spec, args.gpu)
            else:
                audit(spec, args.gpu)
        except Exception:
            out = ROOT / "development" / spec["name"]
            write_json(
                out / "failure-detail.json",
                {"action": args.action, "traceback": traceback.format_exc(), "created_utc": utc()},
            )
            raise
    elif args.action == "status":
        print(json.dumps(summarize(config), indent=2))
    elif args.action == "report":
        report(config)


if __name__ == "__main__":
    main()
