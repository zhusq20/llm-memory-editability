"""Queue a paired LR/decay attribution after the existing architecture batch."""

from __future__ import annotations

import argparse
import ast
import fcntl
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import execute_parametric_architecture as scheduler

read, write, digest, now = scheduler.read, scheduler.write, scheduler.digest, scheduler.now


def trainer_dependencies(source):
    """Include transitive local imports, not only the top-level trainer file."""
    found = set()

    def visit(module):
        path = Path(source) / "src/llm_memory_editability" / (module + ".py")
        if path in found or not path.exists():
            return
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                visit(node.module)

    visit("realworld_composition")
    return sorted(found)


def validate_matrix(config, historical):
    assert config["gpus"] == [2, 3, 4, 5]
    assert config["runtime"]["docker_context"] == "lm-memory"
    assert config["phase"] == "runs"
    for key in ["data_file", "data_sha256", "tokenizer", "model", "evaluation_nodes"]:
        assert config[key] == historical[key], key
    expected = {
        (seed, lr, wd) for seed in [811201, 811202] for lr in [5e-5, 1e-4] for wd in [0.1, 0.3]
    }
    specs = [item["spec"] for item in config["reused_baselines"]] + config["runs"]
    observed = [(s["initialization"], s["learning_rate"], s["weight_decay"]) for s in specs]
    assert len(observed) == len(set(observed)) == len(expected) and set(observed) == expected
    assert len(config["reused_baselines"]) == 2 and len(config["runs"]) == 6
    for spec in specs:
        base = next(
            s
            for s in historical["runs"]
            if s["architecture"] == "loop4x2" and s["initialization"] == spec["initialization"]
        )
        assert set(spec) == set(base)
        for key in base.keys() - {"name", "learning_rate", "weight_decay"}:
            assert spec[key] == base[key], (spec["name"], key)
    for baseline in config["reused_baselines"]:
        assert baseline["spec"]["learning_rate"] == 5e-5
        assert baseline["spec"]["weight_decay"] == 0.1
    assert config["full_curve_nodes"] == [0, 16000, 32000, 64000, 128000, 200000, 300000]


def verify_inputs(config, source):
    historical = read(config["historical_config"])
    assert digest(config["historical_config"]) == config["historical_config_sha256"]
    validate_matrix(config, historical)
    assert digest(config["data_file"]) == config["data_sha256"]
    hashes = {}
    for path in trainer_dependencies(source):
        relative = path.relative_to(source)
        old = Path(historical["source_snapshot"]) / relative
        expected = historical["source_files"][str(old)]
        assert digest(path) == digest(old) == expected, str(relative)
        hashes[str(relative)] = expected
    for path, expected in historical["tokenizer_files"].items():
        assert digest(path) == expected
    for baseline in config["reused_baselines"]:
        root = Path(baseline["path"])
        assert read(root / "run.json")["spec"] == baseline["spec"]
        assert read(root / "audit.json")["passed"] is True
        assert read(root / "complete.json")["independently_reloaded"] is True
        for filename, expected in baseline["files_sha256"].items():
            assert digest(root / filename) == expected, (root, filename)
        for step in config["full_curve_nodes"]:
            assert (root / f"checkpoint-{step:07d}.pt").is_file()
    return {
        "passed": True,
        "utc": now(),
        "trainer_dependency_sha256": hashes,
        "baseline_initializations": [811201, 811202],
        "limits": "Source/data/tokenizer compatibility; GPU calibration is still required.",
    }


def freeze(config_path):
    config = read(config_path)
    evidence = verify_inputs(config, Path.cwd())
    scheduler.freeze(config_path)
    write(
        Path("docs/development-artifacts") / config["batch"] / "reuse-verification.json", evidence
    )


def launch(config_path):
    config = read(config_path)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    script = Path(config["source_root"]) / "scripts" / Path(__file__).name
    command = [sys.executable, "-u", str(script), "controller", "--config", str(config_path)]
    with (root / "controller.log").open("a") as log:
        proc = subprocess.Popen(
            command,
            env=scheduler.host_env(config["source_root"]),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    write(root / "controller-process.json", {"pid": proc.pid, "command": command})
    print({"pid": proc.pid, "results": str(root)})


def predecessor_ready(config):
    root = Path(config["wait_for_results_root"])
    state = read(root / "controller-state.json")
    if state.get("failed") or state["state"] in {"controller_error", "finished_with_failures"}:
        raise RuntimeError("Predecessor failed; attribution remains unstarted for review")
    if state["state"] != "complete":
        return False, state
    expected = set(config["wait_for_runs"])
    assert set(state["completed"]) == expected and not state.get("active")
    for name in expected:
        assert read(root / "runs" / name / "audit.json")["passed"] is True
    return True, state


def container_command(config, config_path, spec, out, gpu, attempt):
    name, command = scheduler.container_command(config, config_path, spec, out, gpu, attempt)
    previous = str(Path(config["source_root"]) / "scripts/execute_parametric_architecture.py")
    command[command.index(previous)] = str(
        Path(config["source_root"]) / "scripts" / Path(__file__).name
    )
    return name, command


def controller(config_path):
    config = read(config_path)
    root = Path(config["results_root"])
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                ready, previous = predecessor_ready(config)
                write(
                    root / "controller-state.json",
                    {
                        "state": "waiting_for_free_gpus" if ready else "waiting_for_predecessor",
                        "gpus": config["gpus"],
                        "completed": [],
                        "failed": [],
                        "active": {},
                        "queued": [s["name"] for s in config["runs"]],
                        "predecessor_completed": len(previous.get("completed", [])),
                        "predecessor_total": len(config["wait_for_runs"]),
                        "updated_utc": now(),
                    },
                )
                if ready:
                    devices = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=index,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                    memory = {
                        int(i.strip()): int(m.strip())
                        for i, m in (line.split(",") for line in devices.splitlines())
                    }
                    if all(memory[g] <= 1024 for g in config["gpus"]):
                        break
                time.sleep(20)
            verify_inputs(config, Path(config["source_root"]))
            scheduler.control_locked(config, config_path, root, command_builder=container_command)
            if read(root / "controller-state.json")["state"] == "complete":
                report(config)
        except BaseException:
            write(root / "controller-error.json", {"utc": now(), "error": traceback.format_exc()})
            state = read(root / "controller-state.json")
            state.update(state="controller_error", updated_utc=now())
            write(root / "controller-state.json", state)
            raise


def baseline_for(config, spec):
    return next(
        b
        for b in config["reused_baselines"]
        if b["spec"]["initialization"] == spec["initialization"]
    )


def preflight(config, spec, out):
    """Actual isolated CUDA updates; exact historical initialization and step-0 replay."""
    import torch
    import transformers

    from llm_memory_editability import realworld_composition as rw

    assert torch.__version__ == "2.6.0+cu126" and transformers.__version__ == "4.40.0"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    data = read(config["data_file"])
    tokenizer = rw.GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    model = rw.construct(config["model"], spec, "cuda:0")
    baseline = Path(baseline_for(config, spec)["path"])
    initial = rw.model_digest(model)
    assert initial == read(baseline / "run.json")["initial_model_sha256"]
    metrics, predictions = rw.evaluate_dataset(model, data, tokenizer, "cuda:0")
    assert predictions == read(baseline / "predictions-0000000.json")
    assert metrics == read(baseline / "learning.json")[0]["metrics"]
    optimizer = rw.optimizer_for(model, spec["learning_rate"], spec["weight_decay"])
    records = data["atoms"] + data["train_compositions"]
    stream = rw.BatchStream(len(records), spec["sampling_seed"])
    torch.manual_seed(spec["dropout_seed"])
    values = []
    for _ in range(8):
        ids = stream.batch(spec["batch_size"])
        values.append(
            rw.update(
                model, optimizer, [records[i] for i in ids], spec, "cuda:0", tokenizer.eos_token_id
            )
        )
        torch.cuda.synchronize()
    assert all(math.isfinite(v["loss"]) and math.isfinite(v["gradient_norm"]) for v in values)
    write(
        out / "gpu-preflight.json",
        {
            "passed": True,
            "utc": now(),
            "gpu": torch.cuda.get_device_name(),
            "historical_initial_sha256": initial,
            "historical_step_zero_exact_replay": True,
            "engineering_updates_excluded_from_training": values,
        },
    )


def trajectory(config, spec, out, *, historical=False):
    """Re-evaluate the same full composition pool/order at predetermined checkpoints."""
    import torch

    from llm_memory_editability import realworld_composition as rw

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    source = Path(baseline_for(config, spec)["path"]) if historical else out
    target = out / ("historical-baseline-curve" if historical else "full-curve")
    data = read(config["data_file"])
    tokenizer = rw.GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    model = rw.construct(config["model"], spec, "cuda:0")
    curve = []
    started = time.perf_counter()
    for step in config["full_curve_nodes"]:
        checkpoint = source / f"checkpoint-{step:07d}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["step"] == step
        model.load_state_dict(state["model"])
        del state
        _, raw = rw.evaluate(model, data["evaluation_compositions"], tokenizer, "cuda:0")
        if step == spec["steps"]:
            assert raw == read(source / "endpoint-predictions.json")["test_all"]
        write(target / f"predictions-{step:07d}.json", raw)
        curve.append(
            {
                "step": step,
                "oo": rw.aggregate([r for r in raw if r["role"] == "OO"]),
                "all": rw.aggregate(raw),
                "checkpoint_sha256": digest(checkpoint),
            }
        )
        write(target / "curve.json", curve)
    write(
        target / "complete.json",
        {
            "passed": True,
            "source": str(source),
            "utc": now(),
            "evaluation_seconds": time.perf_counter() - started,
            "full_oo_n": 1283,
            "full_compositions_n": len(raw),
        },
    )


def container_worker(config_path, run_name):
    config = read(config_path)
    assert run_name in {s["name"] for s in config["runs"]}
    out = Path(config["results_root"]) / "runs" / run_name
    script = Path(config["source_root"]) / "scripts" / Path(__file__).name
    base = ["--config", str(config_path), "--run", run_name]
    actions = ["preflight", "train", "trajectory"]
    if run_name in config["baseline_curve_carriers"]:
        actions.append("baseline-trajectory")
    actions.append("audit")
    try:
        for action in actions:
            subprocess.run([sys.executable, "-u", str(script), action, *base], check=True)
        assert read(out / "audit.json")["passed"] is True
    except BaseException:
        write(out / "failure.json", {"traceback": traceback.format_exc(), "utc": now()})
        raise


def report(config):
    import numpy as np

    rows, cells = [], {}
    sources = [(b["spec"], Path(b["path"]), True) for b in config["reused_baselines"]]
    sources += [
        (s, Path(config["results_root"]) / "runs" / s["name"], False) for s in config["runs"]
    ]
    for spec, out, reused in sources:
        assert read(out / "audit.json")["passed"] is True
        predictions = read(out / "endpoint-predictions.json")["test_all"]
        oo = [p for p in predictions if p["role"] == "OO"]
        assert len(oo) == 1283
        score = sum(p["alias_em"] for p in oo) / len(oo)
        assert abs(score - read(out / "endpoint.json")["test_oo"]["alias_em"]) < 1e-12
        base = Path(baseline_for(config, spec)["path"])
        assert (
            read(out / "run.json")["initial_model_sha256"]
            == read(base / "run.json")["initial_model_sha256"]
        )
        for step in config["evaluation_nodes"]:
            filename = f"exposure-{step:07d}.npz"
            with np.load(out / filename) as actual, np.load(base / filename) as reference:
                np.testing.assert_array_equal(actual["counts"], reference["counts"])
        latest = read(out / "learning.json")[-1]
        reference = read(base / "learning.json")[-1]
        for key in [
            "examples",
            "supervised_tokens",
            "executed_input_tokens",
            "effective_input_tokens",
            "estimated_matmul_training_flops",
        ]:
            assert latest[key] == reference[key], (spec["name"], key)
        cells[spec["initialization"], spec["learning_rate"], spec["weight_decay"]] = 100 * score
        curve_root = out / "full-curve"
        if reused:
            carrier = next(
                s
                for s in config["runs"]
                if s["name"] in config["baseline_curve_carriers"]
                and s["initialization"] == spec["initialization"]
            )
            curve_root = (
                Path(config["results_root"])
                / "runs"
                / carrier["name"]
                / "historical-baseline-curve"
            )
        assert read(curve_root / "complete.json")["passed"] is True
        rows.append(
            {
                "spec": spec,
                "reused": reused,
                "oo_percent": 100 * score,
                "endpoint": read(out / "endpoint.json"),
                "training_budget": latest,
                "full_curve": read(curve_root / "curve.json"),
            }
        )
    contrasts = []
    for seed in [811201, 811202]:
        a, b, c, d = [
            cells[seed, lr, wd] for lr, wd in [(5e-5, 0.1), (1e-4, 0.1), (5e-5, 0.3), (1e-4, 0.3)]
        ]
        contrasts.append(
            {
                "initialization": seed,
                "lr_at_wd01_pp": b - a,
                "lr_at_wd03_pp": d - c,
                "wd_at_lr5e5_pp": c - a,
                "wd_at_lr1e4_pp": d - b,
                "interaction_pp": d - c - b + a,
                "joint_recipe_minus_baseline_pp": d - a,
            }
        )
    summary = {
        "passed": True,
        "utc": now(),
        "cells": rows,
        "paired_contrasts": contrasts,
        "mean_contrasts": {
            k: statistics.mean(r[k] for r in contrasts)
            for k in contrasts[0]
            if k != "initialization"
        },
        "scope": "Exploratory attribution on one previously evaluated graph; "
        "two paired initializations, not two worlds.",
    }
    write(Path(config["results_root"]) / "report.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "freeze",
            "launch",
            "controller",
            "container-worker",
            "preflight",
            "train",
            "audit",
            "trajectory",
            "baseline-trajectory",
            "report",
        ],
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run")
    args = parser.parse_args()
    path = args.config.resolve()
    if args.action in {"freeze", "launch", "controller"}:
        globals()[args.action](path)
    elif args.action == "container-worker":
        container_worker(path, args.run)
    else:
        config = read(path)
        if args.action == "report":
            return report(config)
        spec = next(s for s in config["runs"] if s["name"] == args.run)
        out = Path(config["results_root"]) / "runs" / spec["name"]
        if args.action in {"train", "audit"}:
            from llm_memory_editability import realworld_composition as rw

            (rw.worker if args.action == "train" else rw.audit)(config, spec, 0)
        elif args.action == "preflight":
            preflight(config, spec, out)
        else:
            trajectory(config, spec, out, historical=args.action == "baseline-trajectory")


if __name__ == "__main__":
    main()
