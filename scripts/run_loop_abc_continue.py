#!/usr/bin/env python3
"""Continue each actual ABC checkpoint from step 150 to step 1,500,000."""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_abc_check as short  # noqa: E402
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.loop_abc_resume import (  # noqa: E402
    SnapshotWriter,
    autocast,
    restore,
    restore_rng,
)
from llm_memory_editability.loop_abc_training import abc_loss, auxiliary_depth  # noqa: E402

BATCH = "loop-abc-continuation-1500k-v1"


def validate(config):
    source = json.loads((ROOT / "configs/loop-abc-method-check-150-v1.json").read_text())
    if config["batch"] != BATCH or config["training"] != {**source["training"], "steps": 1500000}:
        raise ValueError("Only the authorized stop-step extension is allowed")
    for key in ["data", "model", "handoff", "dropout_seed", "arms", "method", "tracking"]:
        if config[key] != source[key]:
            raise ValueError(f"Scientific condition changed: {key}")
    if config["budget"]["remaining_updates"] != 4499550:
        raise ValueError("Expected 3 * (1500000-150) additional updates")


def optimizer_names(model):
    named = list(model.named_parameters())
    plain = [(n, p) for n, p in named if not n.startswith("handoff_")]
    return (
        [n for n, _ in plain if not any(s in n for s in ("bias", "ln"))]
        + [n for n, _ in plain if any(s in n for s in ("bias", "ln"))]
        + [n for n, _ in named if n.startswith("handoff_")]
    )


def measure(config, model, average, world, panels, metadata, out, step, counts, counters):
    started = time.perf_counter()
    weights = {}
    model.eval()
    for kind in ["ema", "raw"]:
        context = average.applied(model) if kind == "ema" else native._nothing()
        with context:
            results = native.evaluate_rounds(
                model, world, panels, metadata, torch.device("cuda:0"), 256, [4, 8, 16]
            )
        weights[kind] = {r: values[0] for r, values in results.items()}
        for rounds, (_, predictions) in results.items():
            folder = out / "predictions" / f"{step:07d}" / f"{kind}-r{rounds}"
            folder.mkdir(parents=True, exist_ok=False)
            for panel, values in predictions.items():
                np.savez_compressed(folder / f"{panel}.npz", **values)
    counters["evaluation_seconds"] += time.perf_counter() - started
    model.train()
    return {
        "step": step,
        "metrics": weights,
        **counters,
        **native.exposure_summary(counts, world, metadata, step, config["training"]["batch_size"]),
        "beta": float(model.beta.detach()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def train(config, config_path, base, arm):
    import wandb

    device = torch.device("cuda:0")
    source = Path(config["source_root"]) / "runs" / arm
    path = source / "latest.pt"
    if native.digest(path) != config["source_checkpoints"][arm]["sha256"]:
        raise ValueError("Original recovery checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["step"] != 150 or saved["first_round_input"] != "embedding":
        raise ValueError("Expected the completed, embedding-first step150 state")
    world, panels, metadata = native.load(config["input_root"])
    out = base / "runs" / arm
    out.mkdir(parents=True, exist_ok=False)
    model = short.initialize(config, arm, metadata, device).train()
    optimizer = native.optimizer_for(model, config["training"], fused=True)
    average = native.ParameterAverage(model, config["training"]["ema_decay"])
    table = native.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, config["training"]["stream_seed"])
    rng = np.random.default_rng(config["training"]["schedule_seed"])
    counts = restore(saved, model, optimizer, average, stream, rng, device)
    assert native.model_digest(model) == saved["model_sha256"]
    history = json.loads((source / "learning.json").read_text())
    (out / "progress.jsonl").write_bytes((source / "progress.jsonl").read_bytes())
    (out / "source-method-steps.json").write_bytes((source / "method-steps.json").read_bytes())
    (out / "source-complete.json").write_bytes((source / "complete.json").read_bytes())
    native.write_json(out / "learning.json", history)
    tracking = json.loads((source / "tracking.json").read_text())
    ids = sum([g["params"] for g in saved["optimizer"]["param_groups"]], [])
    names = optimizer_names(model)
    missing = [names[i] for i in ids if i not in saved["optimizer"]["state"]]
    audit = {
        "source_step": 150,
        "next_step": 151,
        "source_checkpoint_sha256": native.digest(path),
        "model_sha256_exact": True,
        "source_optimizer_states": len(saved["optimizer"]["state"]),
        "source_missing_adam_parameters": missing,
        "source_exposure": int(counts.sum()),
        "stream_remaining": len(stream.remaining),
        "autocast_cache_enabled": False,
        "B_legacy_gradient_defect": arm == "B",
        "weights_or_adam_reset": False,
    }
    native.write_json(out / "resume-audit.json", audit)
    native.write_json(
        out / "run.json",
        {
            "batch": BATCH,
            "arm": arm,
            "source": audit,
            "config_sha256": native.digest(config_path),
            "method": config["method"][arm],
            "started_utc": native.utc(),
        },
    )
    native.write_json(out / "tracking.json", tracking)
    run = wandb.init(
        entity=config["tracking"]["entity"],
        project=config["tracking"]["project"],
        id=tracking["id"],
        resume="must",
        config={"continuation": config},
        dir=str(out),
        save_code=False,
        settings=wandb.Settings(disable_git=True),
    )
    # Tracker initialization must not change the actual checkpoint RNG state.
    restore_rng(saved, device)
    counters = {
        "training_seconds": saved["training_seconds"],
        "evaluation_seconds": 0.0,
        "resume_validation_seconds": 0.0,
        "checkpoint_capture_seconds": 0.0,
        "estimated_matmul_training_flops": history[-1]["estimated_matmul_training_flops"],
        "executed_sample_rounds": saved["executed_sample_rounds"],
        "supervised_tokens": saved["supervised_tokens_seen"],
    }
    depth_histogram = np.array(saved["depth_histogram"], copy=True)
    writer = SnapshotWriter()
    window = {k: torch.zeros((), device=device) for k in ["loss", "terminal", "auxiliary", "norm"]}
    nwindow = ntarget = clipped = 0
    histogram = np.zeros(8, dtype=np.int64)
    started = last_save = time.perf_counter()

    def learning_rate(step):
        return native.learning_rate(config["training"], step)

    def payload(step):
        return {
            "model": model.state_dict(),
            "ema": average.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "arm": arm,
            "handoff": "invariant",
            "first_round_input": "embedding",
            "config_sha256": native.digest(config_path),
            "source_checkpoint": audit,
            "stream": stream.state_dict(),
            "schedule_rng": rng.bit_generator.state,
            "counts": counts,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "counters": copy.deepcopy(counters),
            "training_seconds": counters["training_seconds"],
            "depth_histogram": depth_histogram,
            "sampler_step": step,
            "window": {
                "values": window,
                "updates": nwindow,
                "targets": ntarget,
                "clipped": clipped,
                "auxiliary_histogram": histogram,
            },
        }

    def checkpoint(step, evaluation=False):
        nonlocal last_save
        began = time.perf_counter()
        data = payload(step)
        if evaluation:
            writer.save(out / f"checkpoint-{step:07d}.pt", data, retain_previous=False)
        writer.save(out / "recovery/latest.pt", data)
        counters["checkpoint_capture_seconds"] += time.perf_counter() - began
        last_save = time.perf_counter()

    try:
        for step in range(151, 1500001):
            began = time.perf_counter()
            indices, _ = stream.batch(config["training"]["batch_size"])
            counts += np.bincount(indices, minlength=len(counts) + 1)[: len(counts)]
            index = torch.as_tensor(indices, device=device)
            batch = tuple(part[index] for part in table)
            k = auxiliary_depth(config["training"]["auxiliary_seed"], step)
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            validation_overhead = 0.0
            if step == 151:
                validation_started = time.perf_counter()
                cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                with autocast(device):
                    first = abc_loss(model, *batch, arm, k)[0]
                first.backward()
                missing_now = [n for n, p in model.named_parameters() if p.grad is None]
                assert not missing_now, missing_now
                grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
                first_value = float(first.detach())
                optimizer.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                validation_overhead += time.perf_counter() - validation_started
            with autocast(device):
                result = abc_loss(model, *batch, arm, k)
            result[0].backward()
            if step == 151:
                validation_started = time.perf_counter()
                assert float(result[0].detach()) == first_value
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.grad, grads[name], rtol=1e-5, atol=1e-6)
                assert torch.cuda.max_memory_reserved() / 2**20 + 6144 < (
                    torch.cuda.get_device_properties(0).total_memory / 2**20 * 0.16
                )
                audit.update(
                    cuda_replay_passed=True,
                    next_auxiliary_depth=k,
                    first_resumed_loss=first_value,
                    all_21_gradients_present=True,
                    first_resumed_lr=lr,
                    extra_optimizer_updates=0,
                    extra_forward_backward_passes=1,
                )
                native.write_json(out / "resume-audit.json", audit)
                del grads, first
                validation_overhead += time.perf_counter() - validation_started
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=True, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            average.update(model)
            targets = int(batch[2].ne(-100).sum())
            window["loss"] += result[0].detach() * targets
            window["terminal"] += result[1].detach() * targets
            if result[2] is not None:
                window["auxiliary"] += result[2].detach() * targets
            window["norm"] += norm.detach()
            clipped += int(norm > 1)
            nwindow += 1
            ntarget += targets
            histogram[k] += 1
            depth_histogram[8] += len(indices)
            counters["supervised_tokens"] += targets
            counters["executed_sample_rounds"] += len(indices) * 8
            counters["estimated_matmul_training_flops"] += short.flops(
                model, metadata, np.full(len(indices), 8), batch[2].shape[1]
            )
            torch.cuda.synchronize()
            counters["training_seconds"] += time.perf_counter() - began - validation_overhead
            counters["resume_validation_seconds"] += validation_overhead
            if step == 151:
                checkpoint(step)
            if step % 10 == 0:
                status = {
                    "state": "running",
                    "arm": arm,
                    "step": step,
                    "budget": 1500000,
                    "mean_loss": float(window["loss"]) / ntarget,
                    "terminal_loss": float(window["terminal"]) / ntarget,
                    "auxiliary_loss": float(window["auxiliary"]) / ntarget,
                    "mean_grad_norm": float(window["norm"]) / nwindow,
                    "gradient_clip_fraction": clipped / nwindow,
                    "lr": lr,
                    "auxiliary_depth_histogram": histogram.tolist() if arm == "C" else None,
                    "beta": float(model.beta.detach()),
                    **counters,
                    "forward_rounds_per_sequence": 8,
                    "backpropagation_rounds": 4 if arm == "B" else 8,
                    "wall_seconds_continuation": time.perf_counter() - started,
                    "updated_utc": native.utc(),
                }
                if not all(
                    math.isfinite(status[k]) for k in ["mean_loss", "mean_grad_norm", "beta"]
                ):
                    raise FloatingPointError("Nonfinite training status")
                with (out / "progress.jsonl").open("a") as progress:
                    progress.write(json.dumps(status) + "\n")
                native.write_json(out / "status.json", status)
                run.log(
                    {
                        f"training/{k}": v
                        for k, v in status.items()
                        if isinstance(v, (float, int)) and not isinstance(v, bool)
                    },
                    step=step,
                )
                window = {key: value.zero_() for key, value in window.items()}
                nwindow = ntarget = clipped = 0
                histogram.fill(0)
                if step % 10000 == 0 or time.perf_counter() - last_save > 1800:
                    checkpoint(step)
            if step in config["evaluation_nodes"]:
                record = measure(
                    config, model, average, world, panels, metadata, out, step, counts, counters
                )
                history.append(record)
                native.write_json(out / "learning.json", history)
                run.log(native.learning_metrics(record), step=step)
                checkpoint(step, evaluation=True)
        writer.close()
        final = {
            "state": "complete",
            "arm": arm,
            "step": step,
            "budget": 1500000,
            "endpoint": history[-1],
            "model_sha256": native.model_digest(model),
            "final_checkpoint_sha256": native.digest(out / f"checkpoint-{step:07d}.pt"),
            "config_sha256": native.digest(config_path),
            **counters,
            "checkpoint_write_seconds": writer.seconds,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "updated_utc": native.utc(),
        }
        native.write_json(out / "complete.json", final)
        native.write_json(out / "status.json", final)
        run.finish()
    except BaseException:
        writer.close()
        run.finish(exit_code=1)
        raise


def cloud(config, base, arm):
    import wandb
    from run_loop_recipe_pipeline import reload_matches

    out = base / "runs" / arm
    complete = json.loads((out / "complete.json").read_text())
    grid = json.loads((out / "grid.json").read_text())
    tracking = json.loads((out / "tracking.json").read_text())
    original = json.loads((Path(config["source_root"]) / "config.json").read_text())
    remote = wandb.Api(timeout=90).run(f"zsqzz/llm-memory-editability/{tracking['id']}")
    values = native.learning_metrics(complete["endpoint"])
    checks = {
        "training_step": complete["step"] == 1500000,
        "reload_step": grid["step"] == 1500000,
        "checkpoint": grid["checkpoint_sha256"] == complete["final_checkpoint_sha256"],
        "reload_metrics": all(
            reload_matches(metrics, grid["weights"][kind]["answer_and_eos"][r])
            for kind, horizons in complete["endpoint"]["metrics"].items()
            for r, metrics in horizons.items()
        ),
        "cloud_finished": remote.state == "finished",
        "old_config_preserved": remote.config.get("experiment") == original,
        "continuation_config": remote.config.get("continuation") == config,
        "cloud_metrics": all(
            isinstance(remote.summary.get(k), (int, float))
            and math.isclose(remote.summary[k], v, rel_tol=1e-6, abs_tol=1e-7)
            for k, v in values.items()
        ),
    }
    native.write_json(
        out / "cloud-verification.json",
        {
            "passed": all(checks.values()),
            "checks": checks,
            "url": tracking["url"],
            "utc": native.utc(),
        },
    )
    if not all(checks.values()):
        raise RuntimeError("Independent reload/cloud verification failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument(
        "--stage", choices=["pipeline", "train", "grid", "cloud"], default="pipeline"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb-key").read_text().strip()
    if args.stage == "pipeline":
        directory = args.base / "pipeline" / args.arm
        directory.mkdir(parents=True, exist_ok=False)
        state = {"state": "running", "arm": args.arm, "stages": {}}
        for stage in ["train", "grid", "cloud"]:
            command = [
                sys.executable,
                "-u",
                __file__,
                "--config",
                str(args.config),
                "--base",
                str(args.base),
                "--arm",
                args.arm,
                "--stage",
                stage,
            ]
            with (directory / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["stages"][stage] = {"pid": process.pid, "started_utc": native.utc()}
                native.write_json(directory / "state.json", state)
                code = process.wait()
            state["stages"][stage].update(exit_code=code, ended_utc=native.utc())
            state["state"] = "running" if code == 0 else "needs_attention"
            native.write_json(directory / "state.json", state)
            if code:
                raise RuntimeError(f"{stage} failed; original evidence preserved; no auto retry")
        state.update(state="complete", completed_utc=native.utc())
        native.write_json(directory / "state.json", state)
        return
    if args.stage == "cloud":
        cloud(config, args.base, args.arm)
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.16)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage == "grid":
        original = native.initialize
        native.initialize = short.initialize
        try:
            native.grid(config, args.arm, args.base, args.config, torch.device("cuda:0"))
        finally:
            native.initialize = original
    else:
        train(config, args.config, args.base, args.arm)


if __name__ == "__main__":
    main()
