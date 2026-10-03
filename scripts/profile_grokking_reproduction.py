"""Read live telemetry and run isolated engineering benchmarks without changing runs."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pynvml
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grokking_reproduction import (
    ARTIFACTS,
    ROOT,
    ReproductionStep,
    construct,
    encode_rows,
    optimizer_for,
)


def trajectory_timing():
    records = []
    for out in sorted((ROOT / "development").iterdir()):
        history = json.loads((out / "learning.json").read_text())
        pairs = list(zip(history[:-1], history[1:], strict=False))[-12:]
        train = [
            1000 * (b["training_seconds"] - a["training_seconds"]) / (b["step"] - a["step"])
            for a, b in pairs
        ]
        wall = [
            1000 * (b["wall_seconds"] - a["wall_seconds"]) / (b["step"] - a["step"])
            for a, b in pairs
        ]
        training = sum(b["training_seconds"] - a["training_seconds"] for a, b in pairs)
        elapsed = sum(b["wall_seconds"] - a["wall_seconds"] for a, b in pairs)
        records.append(
            {
                "name": out.name,
                "step": history[-1]["step"],
                "recent_intervals": len(pairs),
                "training_ms_per_step_median": statistics.median(train),
                "wall_ms_per_step_median": statistics.median(wall),
                "nontraining_time_fraction": 1 - training / elapsed,
            }
        )
    return records


def telemetry(seconds, out):
    pynvml.nvmlInit()
    devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    samples = []
    begin = time.perf_counter()
    while time.perf_counter() - begin < seconds:
        for i, handle in enumerate(devices):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            samples.append(
                {
                    "elapsed_seconds": time.perf_counter() - begin,
                    "gpu": i,
                    "memory_used_mib": memory.used / 2**20,
                    "memory_total_mib": memory.total / 2**20,
                    "gpu_busy_percent": utilization.gpu,
                    "memory_busy_percent": utilization.memory,
                    "power_watts": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                }
            )
        time.sleep(0.5)
    summary = []
    for i in range(len(devices)):
        rows = [s for s in samples if s["gpu"] == i]
        summary.append(
            {
                "gpu": i,
                "samples": len(rows),
                "gpu_busy_mean": statistics.mean(r["gpu_busy_percent"] for r in rows),
                "gpu_busy_min": min(r["gpu_busy_percent"] for r in rows),
                "gpu_busy_median": statistics.median(r["gpu_busy_percent"] for r in rows),
                "memory_used_mib_max": max(r["memory_used_mib"] for r in rows),
                "memory_total_mib": rows[0]["memory_total_mib"],
                "power_watts_mean": statistics.mean(r["power_watts"] for r in rows),
            }
        )
    pynvml.nvmlShutdown()
    write_json(out / "telemetry.json", {"samples": samples, "summary": summary})
    return summary


def benchmark_batch(spec, metadata, table, batch, iterations, backend, isolated):
    spec = {**spec, "batch_size": batch}
    model = construct(spec, metadata, "cuda")
    lr = torch.tensor(spec["lr"], device="cuda")
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    context = sdpa_kernel(SDPBackend.MATH) if backend == "math" else contextlib.nullcontext()
    with context:
        graph = ReproductionStep(model, optimizer, table, batch)
    indices = torch.arange(batch, device="cuda") % len(table[0])
    for _ in range(5):
        graph(indices)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(iterations):
            lr.fill_(spec["lr"])
            graph(indices)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) / iterations)
    latency = statistics.median(latencies)
    result = {
        "batch": batch,
        "attention_backend": backend,
        "seconds_per_update": latency,
        "samples_per_second": batch / latency,
        "replicate_seconds_per_update": latencies,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "co_running_training_worker": not isolated,
        "iterations_per_replicate": iterations,
    }
    del graph, model, optimizer, lr
    gc.collect()
    torch.cuda.empty_cache()
    return result


def profile(spec, metadata, table, steps, out):
    model = construct(spec, metadata, "cuda")
    optimizer = optimizer_for(model, torch.tensor(spec["lr"], device="cuda"), 0.1)
    graph = ReproductionStep(model, optimizer, table, 512)
    graph.index.copy_(torch.arange(512, device="cuda"))
    for _ in range(3):
        graph.eager()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        for _ in range(steps):
            graph.eager()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(out / "trace.json"))
    rows = []
    for item in profiler.key_averages(group_by_input_shape=True):
        if item.self_device_time_total <= 0 or not item.key.startswith("aten::"):
            continue
        rows.append(
            {
                "operator": item.key,
                "shapes": item.input_shapes,
                "self_cuda_ms_per_update": item.self_device_time_total / steps / 1000,
                "calls_per_update": item.count / steps,
            }
        )
    rows.sort(key=lambda x: x["self_cuda_ms_per_update"], reverse=True)
    write_json(
        out / "operators.json",
        {
            "mode": "Eager equivalent step for operator attribution; production uses CUDA Graph",
            "steps": steps,
            "self_cuda_ms_sum_per_update": sum(r["self_cuda_ms_per_update"] for r in rows),
            "operators": rows,
        },
    )
    del graph, model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return rows[:12]


def check_backend_numerics(spec, metadata, table, out):
    """Check BF16 outputs/gradients with dropout disabled; not bitwise training identity."""
    default = construct(spec, metadata, "cuda").eval()
    math = construct(spec, metadata, "cuda").eval()
    x, pos, labels = [a[:512] for a in table]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        left = default(x, pos)
        loss_left = torch.nn.functional.cross_entropy(left.flatten(0, 1), labels.flatten())
        with sdpa_kernel(SDPBackend.MATH):
            right = math(x, pos)
            loss_right = torch.nn.functional.cross_entropy(right.flatten(0, 1), labels.flatten())
    loss_left.backward()
    loss_right.backward()
    numerator = torch.zeros((), device="cuda")
    denominator = torch.zeros((), device="cuda")
    maximum = torch.zeros((), device="cuda")
    for a, b in zip(default.parameters(), math.parameters(), strict=True):
        delta = a.grad - b.grad
        numerator += delta.square().sum()
        denominator += a.grad.square().sum()
        maximum = torch.maximum(maximum, delta.abs().max())
    relative = float((numerator / denominator).sqrt())
    result = {
        "default_loss": float(loss_left),
        "math_loss": float(loss_right),
        "absolute_loss_difference": float((loss_left - loss_right).abs()),
        "max_absolute_logit_difference": float((left.float() - right.float()).abs().max()),
        "max_absolute_gradient_difference": float(maximum),
        "relative_gradient_l2_difference": relative,
        "dropout": "Disabled for this numerical check only",
        "scope": "Full original-size BF16 model at initialization, 512 engineering examples",
        "bitwise_training_identity": False,
        "limitation": "Different kernels may change BF16 rounding and dropout RNG consumption",
    }
    assert np.isfinite(relative)
    torch.testing.assert_close(loss_left, loss_right, atol=0.002, rtol=0.002)
    assert relative < 0.03
    result["tolerance_check_passed"] = True
    write_json(out / "backend-numerics.json", result)
    del default, math, left, right, loss_left, loss_right
    gc.collect()
    torch.cuda.empty_cache()
    return result


@contextlib.contextmanager
def isolated_device(gpu, out, enabled):
    if not enabled:
        yield
        return
    matching = []
    for path in (ROOT / "development").glob("*/run.json"):
        run = json.loads(path.read_text())
        if run["gpu"] == gpu:
            matching.append(run)
    assert len(matching) == 1
    run = matching[0]
    pid = run["pid"]
    assert "worker" in Path(f"/proc/{pid}/cmdline").read_text()
    started = time.perf_counter()
    record = {"pid": pid, "gpu": gpu, "began_utc": utc(), "reason": "Isolated engineering timing"}
    write_json(out / "temporary-worker-pause.json", record)
    watchdog_code = (
        "import os, signal, time\n"
        "for _ in range(4): time.sleep(30)\n"
        "try: os.kill(int(__import__('sys').argv[1]), signal.SIGCONT)\n"
        "except ProcessLookupError: pass\n"
    )
    watchdog = subprocess.Popen([sys.executable, "-c", watchdog_code, str(pid)])
    os.kill(pid, signal.SIGSTOP)
    try:
        # Let queued CUDA Graph updates finish before measuring another context.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        deadline = time.perf_counter() + 15
        while pynvml.nvmlDeviceGetUtilizationRates(handle).gpu:
            assert time.perf_counter() < deadline, "CUDA work did not drain"
            time.sleep(0.5)
        pynvml.nvmlShutdown()
        yield
        assert time.perf_counter() - started < 115, "Safety watchdog invalidated isolation"
    finally:
        os.kill(pid, signal.SIGCONT)
        watchdog.terminate()
        watchdog.wait(timeout=5)
        record.update(resumed_utc=utc(), pause_seconds=time.perf_counter() - started)
        write_json(out / "temporary-worker-pause.json", record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--telemetry-seconds", type=float, default=30)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--batches", type=int, nargs="+", default=[512, 1024, 2048, 4096, 512])
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--backends", nargs="+", choices=["default", "math"], default=["default"])
    parser.add_argument("--pause-worker", action="store_true")
    parser.add_argument("--check-numerics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = trajectory_timing()
    samples = telemetry(args.telemetry_seconds, args.output)
    print(json.dumps({"telemetry": samples, "trajectory": before}), flush=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    with isolated_device(args.gpu, args.output, args.pause_worker):
        metadata = json.loads((ROOT / "data/complete.json").read_text())
        config = json.loads(Path("configs/grokking-reproduction-development-v1.json").read_text())
        spec = config["runs"][0]
        rows = np.load(ROOT / "data/world.npz")["atoms"][:2048]
        table = tuple(torch.as_tensor(a, device="cuda") for a in encode_rows(rows, metadata))
        batches = []
        for backend in args.backends:
            for batch in args.batches:
                measured = benchmark_batch(
                    spec, metadata, table, batch, args.iterations, backend, args.pause_worker
                )
                batches.append(measured)
                write_json(args.output / "batches.json", batches)
                print(json.dumps(measured), flush=True)
        numerics = (
            check_backend_numerics(spec, metadata, table, args.output)
            if args.check_numerics
            else None
        )
        operators = profile(spec, metadata, table, args.profile_steps, args.output)
    write_json(
        args.output / "summary.json",
        {
            "created_utc": utc(),
            "scientific_updates": 0,
            "scope": (
                "Isolated engineering model; no scientific checkpoint or configuration changes"
            ),
            "contention": "Temporarily isolated" if args.pause_worker else "Shared training device",
            "baseline_telemetry": samples,
            "baseline_trajectory": before,
            "batch_benchmarks": batches,
            "top_operators": operators,
            "backend_numerics": numerics,
            "source_snapshot": str(ARTIFACTS / "source"),
        },
    )
    print(json.dumps({"top_operators": operators}), flush=True)


if __name__ == "__main__":
    main()
