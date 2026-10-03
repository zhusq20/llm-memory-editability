"""Preflight, freeze and supervise paired real-data development or confirmation."""

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

import torch
import transformers
from transformers import GPT2TokenizerFast

from llm_memory_editability.grok_depth import utc
from llm_memory_editability.realworld_composition import (
    audit,
    construct,
    evaluate,
    model_digest,
    optimizer_for,
    shared_prefix_digest,
    tensor_digests,
    update,
    worker,
)
from llm_memory_editability.realworld_composition_data import sha256, write_json
from llm_memory_editability.realworld_confirmation import finalize_confirmation
from llm_memory_editability.realworld_confirmation_analysis import analyze_confirmation

PROJECT = Path(__file__).resolve().parents[1]
ART = PROJECT / "docs/development-artifacts/realworld-composition-v1"


def preflight(config, spec, gpu):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{gpu}"
    data = json.loads(Path(config["data_file"]).read_text())
    assert sha256(config["data_file"]) == config["data_sha256"]
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    out = Path(config["results_root"]) / "preflight" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    model = construct(config["model"], spec, device)
    prefix = shared_prefix_digest(model)
    initial_tensors = tensor_digests(model)
    optimizer = optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    pool = data["atoms"][:256] + data["train_compositions"][:256]
    times, losses = [], []
    for _ in range(8):
        torch.cuda.synchronize()
        began = time.perf_counter()
        value = update(model, optimizer, pool, spec, device, tokenizer.eos_token_id)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
        losses.append(value["loss"])
    selected = pool[:8]
    # True variable-answer greedy outputs and NLL, independently reloaded below.
    metrics, predictions = evaluate(model, selected, tokenizer, device)
    torch.save({"model": model.state_dict()}, out / "checkpoint.pt")
    result = {
        "name": spec["name"],
        "gpu": gpu,
        "gpu_name": torch.cuda.get_device_name(gpu),
        "training_pid": os.getpid(),
        "full_size_engineering_updates": 8,
        "losses": losses,
        "seconds_per_update_after_warmup": sum(times[2:]) / len(times[2:]),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "common_initial_prefix_sha256": prefix,
        "initial_tensor_sha256": initial_tensors,
        "parameters": sum(p.numel() for p in model.parameters()),
        "metrics": metrics,
        "predictions": predictions,
        "checkpoint_sha256": sha256(out / "checkpoint.pt"),
        "created_utc": utc(),
    }
    write_json(out / "forward.json", result)


def preflight_audit(config, spec, gpu):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = f"cuda:{gpu}"
    out = Path(config["results_root"]) / "preflight" / spec["name"]
    expected = json.loads((out / "forward.json").read_text())
    assert expected["training_pid"] != os.getpid()
    assert expected["checkpoint_sha256"] == sha256(out / "checkpoint.pt")
    data = json.loads(Path(config["data_file"]).read_text())
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    selected = (data["atoms"][:256] + data["train_compositions"][:256])[:8]
    model = construct(config["model"], spec, device)
    model.load_state_dict(
        torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)["model"]
    )
    metrics, predictions = evaluate(model, selected, tokenizer, device)
    assert metrics == expected["metrics"] and predictions == expected["predictions"]
    del model
    torch.cuda.empty_cache()
    # Engineering width32 fixture: BF16 dropout and fused Adam continuation.
    tiny_config = dict(vocab_size=31, positions=32, hidden_size=32, attention_heads=4, dropout=0.1)
    fixture = [
        dict(
            encoded=dict(
                prefix=[1, 2, 3],
                target=[4] * (i % 3 + 1) + [30],
                input=[1, 2, 3] + [4] * (i % 3 + 1),
            )
        )
        for i in range(7)
    ]
    small_spec = {**spec, "microbatch_size": 3}
    original = construct(tiny_config, spec, device)
    original_opt = optimizer_for(original, 0.001, 0.1)
    update(original, original_opt, fixture, small_spec, device, 30)
    saved_model, saved_opt = (
        copy.deepcopy(original.state_dict()),
        copy.deepcopy(original_opt.state_dict()),
    )
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    update(original, original_opt, fixture, small_spec, device, 30)
    expected_hash = model_digest(original)
    resumed = construct(tiny_config, spec, device)
    resumed_opt = optimizer_for(resumed, 0.001, 0.1)
    resumed.load_state_dict(saved_model)
    resumed_opt.load_state_dict(saved_opt)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    update(resumed, resumed_opt, fixture, small_spec, device, 30)
    assert model_digest(resumed) == expected_hash, (
        "GPU BF16/dropout/Adam restoration changes the update"
    )
    result = {
        **{k: v for k, v in expected.items() if k not in {"predictions", "metrics"}},
        "passed": True,
        "audit_pid": os.getpid(),
        "gpu_dropout_adam_restore_exact": True,
        "width32_restore_fixture_updates": 3,
        "independent_full_size_reload": True,
        "completed_utc": utc(),
    }
    write_json(out / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "initial_tensor_sha256"}), flush=True)


def freeze(config, path):
    results = [
        json.loads(
            (Path(config["results_root"]) / "preflight" / spec["name"] / "audit.json").read_text()
        )
        for spec in config["runs"]
    ]
    assert all(r["passed"] for r in results)
    for seed in {s["initialization"] for s in config["runs"]}:
        selected = [
            r for r, s in zip(results, config["runs"], strict=True) if s["initialization"] == seed
        ]
        assert len({r["common_initial_prefix_sha256"] for r in selected}) == 1
    assert sha256(config["data_file"]) == config["data_sha256"]
    phase = config.get("phase", "development")
    art = Path(config.get("artifact_root", ART))
    data_audit = json.loads((art / "data-contract-audit.json").read_text())
    assert data_audit["passed"]
    destination = art / (phase + "-source")
    if destination.exists():
        raise FileExistsError("Preserve the previous frozen source")
    shutil.copytree(
        PROJECT / "src", destination / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (destination / "scripts").mkdir()
    shutil.copy2(Path(__file__), destination / "scripts" / Path(__file__).name)
    shutil.copy2(
        PROJECT / "scripts/prepare_realworld_composition.py",
        destination / "scripts/prepare_realworld_composition.py",
    )
    for name in ["prepare_realworld_confirmation.py", "track_experiment_wandb.py"]:
        shutil.copy2(PROJECT / "scripts" / name, destination / "scripts" / name)
    (destination / "tests").mkdir()
    shutil.copy2(
        PROJECT / "tests/test_realworld_composition.py",
        destination / "tests/test_realworld_composition.py",
    )
    for name in [
        "test_realworld_confirmation.py",
        "test_experiment_tracking.py",
        "test_grokking_reproduction.py",
    ]:
        shutil.copy2(PROJECT / "tests" / name, destination / "tests" / name)
    sources = {str(p): sha256(p) for p in destination.rglob("*.py")}
    config.update(
        source_snapshot=str(destination),
        source_files=sources,
        execution_lock=str(art / (phase + "-execution-lock.json")),
        project_root=str(PROJECT),
        tokenizer_files={
            str(p): sha256(p) for p in Path(config["tokenizer"]).iterdir() if p.is_file()
        },
    )
    write_json(path, config)
    record = {
        "created_utc": utc(),
        "config": str(path.resolve()),
        "config_sha256": sha256(path),
        "data_sha256": config["data_sha256"],
        "runs": len(config["runs"]),
        "scientific_updates": sum(s["steps"] for s in config["runs"]),
        "preflight": results,
        "source_files": sources,
        "prerequisites": {
            p: sha256(p)
            for p in config.get(
                "prerequisite_files",
                [
                    str(ART / name)
                    for name in [
                        "development-contract.md",
                        "semantic-review.json",
                        "data-lock.json",
                        "data-contract-audit.json",
                    ]
                ],
            )
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version,
        },
        "tracking": {
            "defaults_sha256": sha256(PROJECT / "configs/experiment-tracking-defaults.json"),
            "source_sha256": sha256(PROJECT / "src/llm_memory_editability/experiment_tracking.py"),
            "sdk_version": "0.30.0",
        },
        "formal_training_frozen_or_started": phase == "confirmation",
    }
    write_json(config["execution_lock"], record)
    print(
        json.dumps({k: v for k, v in record.items() if k not in {"source_files", "preflight"}}),
        flush=True,
    )


def verify(config):
    lock = json.loads(Path(config["execution_lock"]).read_text())
    assert sha256(lock["config"]) == lock["config_sha256"], "Frozen configuration changed"
    assert sha256(config["data_file"]) == config["data_sha256"]
    for path, expected in config["tokenizer_files"].items():
        assert sha256(path) == expected, f"Frozen tokenizer changed: {path}"
    for path, expected in config["source_files"].items():
        assert sha256(path) == expected, f"Frozen source changed: {path}"
    for path, expected in lock["prerequisites"].items():
        target = Path(path) if Path(path).is_absolute() else ART / path
        assert sha256(target) == expected, f"Frozen prerequisite changed: {path}"
    assert torch.__version__ == lock["environment"]["torch"]
    assert transformers.__version__ == lock["environment"]["transformers"]


def process_alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def controller(config, config_path):
    verify(config)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    phase = config.get("phase", "development")
    art = Path(config.get("artifact_root", ART))
    project = Path(config.get("project_root", PROJECT))
    script = Path(config["source_snapshot"]) / "scripts/run_realworld_composition.py"
    environment = {**os.environ, "PYTHONPATH": str(Path(config["source_snapshot"]) / "src")}
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            root / "controller-state.json",
            {"state": "running", "started_utc": utc(), "pid": os.getpid()},
        )
        tracking_command = [
            str(project / ".venv-wandb/bin/python"),
            str(Path(config["source_snapshot"]) / "scripts/track_experiment_wandb.py"),
            "--root",
            str(root),
            "--runs-dir",
            str(root / phase),
            "--defaults",
            str(project / "configs/experiment-tracking-defaults.json"),
            "--detach",
        ]
        subprocess.run(tracking_command, check=True, env=environment)
        tracker_pid = json.loads((root / "tracking-wandb/process.json").read_text())["pid"]
        tracking_restarts, next_tracking_check = 0, time.monotonic() + 30
        completed = [
            s["name"]
            for s in config["runs"]
            if (root / phase / s["name"] / "complete.json").exists()
        ]
        pending = [s for s in config["runs"] if s["name"] not in completed]
        active, failed, retries = {}, [], {}
        slots = config["gpus"]
        while pending or active:
            if time.monotonic() >= next_tracking_check:
                next_tracking_check = time.monotonic() + 30
                if not process_alive(tracker_pid):
                    tracking_restarts += 1
                    try:
                        subprocess.run(tracking_command, check=True, env=environment)
                        tracker_pid = json.loads(
                            (root / "tracking-wandb/process.json").read_text()
                        )["pid"]
                    except Exception:
                        write_json(
                            root / "tracking-wandb/restart-failure.json",
                            {
                                "error": traceback.format_exc(),
                                "attempt": tracking_restarts,
                                "created_utc": utc(),
                            },
                        )
            for gpu in slots:
                if gpu in active or not pending:
                    continue
                spec = pending.pop(0)
                out = root / phase / spec["name"]
                out.mkdir(parents=True, exist_ok=True)
                mode = "audit" if (out / "trained.json").exists() else "worker"
                log = (out / (mode + ".log")).open("a")
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(script),
                        mode,
                        "--config",
                        str(config_path),
                        "--name",
                        spec["name"],
                        "--gpu",
                        str(gpu),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                active[gpu] = (spec, child, log, mode)
            for gpu, (spec, child, log, mode) in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code:
                    out = root / phase / spec["name"]
                    failure_path = out / "attempt-failure.json"
                    detail = (
                        json.loads(failure_path.read_text())
                        if failure_path.exists()
                        else {"error": "process exited without a Python exception"}
                    )
                    attempt = retries.get(spec["name"], 0)
                    failure = {
                        "name": spec["name"],
                        "mode": mode,
                        "exit_code": code,
                        "attempt": attempt,
                        **detail,
                    }
                    write_json(out / "failures" / f"{mode}-{attempt}.json", failure)
                    transient = code < 0 or any(
                        s in detail["error"]
                        for s in ["CUDA out of memory", "Input/output error", "Connection reset"]
                    )
                    if transient and attempt < config.get("transient_retries", 0):
                        retries[spec["name"]] = attempt + 1
                        pending.insert(0, spec)
                    else:
                        failed.append(failure)
                        write_json(out / "failure.json", failure)
                elif mode == "worker":
                    pending.insert(0, spec)
                else:
                    completed.append(spec["name"])
            write_json(
                root / "controller-state.json",
                {
                    "state": "running",
                    "pid": os.getpid(),
                    "completed": completed,
                    "failed": failed,
                    "retries": retries,
                    "tracking_restarts": tracking_restarts,
                    "active": {
                        str(g): {"name": v[0]["name"], "pid": v[1].pid, "mode": v[3]}
                        for g, v in active.items()
                    },
                    "queued": [s["name"] for s in pending],
                    "updated_utc": utc(),
                },
            )
            time.sleep(5)
        if not failed and phase == "confirmation":
            write_json(
                root / "controller-state.json",
                {
                    "state": "finalizing",
                    "completed": completed,
                    "failed": [],
                    "active": {},
                    "queued": [],
                    "updated_utc": utc(),
                },
            )
            try:
                verify(config)
                finalize_confirmation(config)
                analyze_confirmation(config)
            except Exception:
                failed.append({"mode": "independent_recount", "error": traceback.format_exc()})
                write_json(art / "confirmation-finalization-failure.json", failed[-1])
        state = "finished_with_failures" if failed else "complete"
        final = {
            "state": state,
            "completed": completed,
            "failed": failed,
            "updated_utc": utc(),
            "formal_training_started": phase == "confirmation",
            "independent_recount_passed": not failed and phase == "confirmation",
            "scientific_updates": sum(s["steps"] for s in config["runs"] if s["name"] in completed),
        }
        write_json(root / "controller-state.json", final)
        write_json(art / (phase + "-completion.json"), final)
        summary = {
            s["name"]: json.loads((root / phase / s["name"] / "endpoint.json").read_text())
            for s in config["runs"]
            if s["name"] in completed
        }
        write_json(art / (phase + "-summary.json"), summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "preflight-audit",
            "freeze",
            "launch",
            "controller",
            "worker",
            "audit",
            "status",
        ],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = next((s for s in config["runs"] if s["name"] == args.name), None)
    if args.command in {"worker", "audit"}:
        verify(config)
        try:
            {"worker": worker, "audit": audit}[args.command](config, spec, args.gpu)
        except Exception:
            error = traceback.format_exc()
            out = Path(config["results_root"]) / config.get("phase", "development") / spec["name"]
            write_json(
                out
                / (
                    "attempt-failure.json"
                    if config.get("phase") == "confirmation"
                    else "failure.json"
                ),
                {"command": args.command, "error": error, "created_utc": utc()},
            )
            raise
    elif args.command in {"preflight", "preflight-audit"}:
        {"preflight": preflight, "preflight-audit": preflight_audit}[args.command](
            config, spec, args.gpu
        )
    elif args.command == "freeze":
        freeze(config, args.config)
    elif args.command == "controller":
        controller(config, args.config)
    elif args.command == "launch":
        verify(config)
        root = Path(config["results_root"])
        root.mkdir(parents=True, exist_ok=True)
        if (root / "controller.json").exists():
            raise FileExistsError(
                "Controller already registered; inspect its state before restarting"
            )
        with (root / "controller.log").open("a") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(config["source_snapshot"]) / "scripts/run_realworld_composition.py"),
                    "controller",
                    "--config",
                    str(args.config.resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write_json(
            root / "controller.json",
            {"pid": child.pid, "config": str(args.config), "created_utc": utc()},
        )
        print(json.dumps({"controller_pid": child.pid, "results": str(root)}), flush=True)
    else:
        root = Path(config["results_root"])
        print((root / "controller-state.json").read_text())
        for spec in config["runs"]:
            path = root / config.get("phase", "development") / spec["name"] / "status.json"
            if path.exists():
                print(spec["name"], path.read_text())


if __name__ == "__main__":
    main()
