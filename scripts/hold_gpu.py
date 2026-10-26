#!/usr/bin/env python3
"""Hold memory on one idle GPU until SIGTERM/SIGINT, without a busy compute loop."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone


def emit(event, **values):
    print(
        json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True, help="Physical GPU UUID from nvidia-smi")
    parser.add_argument("--fraction", type=float, default=0.9, help="Fraction of free GPU memory")
    parser.add_argument("--seconds", type=float, default=0, help="0 holds until explicitly stopped")
    parser.add_argument("--wait", action="store_true", help="Wait if the requested GPU is busy")
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-"):
        parser.error("A full physical GPU UUID is required")
    if not 0 < args.fraction <= 0.95 or args.seconds < 0:
        parser.error("Require 0 < fraction <= 0.95 and seconds >= 0")

    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    query = ["nvidia-smi", "--id=" + args.gpu_uuid, "--format=csv,noheader,nounits"]
    next_wait_report = 0
    while not stopping:
        snapshot = subprocess.check_output(
            [*query, "--query-gpu=uuid,memory.used,utilization.gpu"], text=True, timeout=15
        ).strip()
        uuid, used, utilization = (item.strip() for item in snapshot.split(","))
        if uuid != args.gpu_uuid:
            raise RuntimeError(f"GPU UUID mismatch: {snapshot}")
        processes = subprocess.check_output(
            [*query, "--query-compute-apps=pid"], text=True, timeout=15
        ).strip()
        if int(used) <= 16 and int(utilization) == 0 and not processes:
            break
        if not args.wait:
            raise RuntimeError(f"GPU is not idle; no allocation made: {snapshot}; PIDs={processes}")
        if time.monotonic() >= next_wait_report:
            emit("waiting_for_idle_gpu", gpu_uuid=uuid, used_mib=int(used))
            next_wait_report = time.monotonic() + 60
        time.sleep(5)
    if stopping:
        emit("stopped_before_allocation")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    import torch

    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one working CUDA device is required")

    # Confirm real CUDA arithmetic, backward, and synchronization before holding memory.
    probe = torch.ones((32, 32), device="cuda", requires_grad=True)
    result = (probe @ probe).sum()
    result.backward()
    torch.cuda.synchronize()
    if result.item() != 32768 or not torch.all(probe.grad == 64).item():
        raise RuntimeError("CUDA arithmetic validation failed")
    del probe, result
    torch.cuda.empty_cache()

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    held_bytes = int(free_bytes * args.fraction)
    held = torch.empty(held_bytes, dtype=torch.uint8, device="cuda")
    held.zero_()
    torch.cuda.synchronize()
    started = time.monotonic()
    emit(
        "holding",
        gpu_uuid=args.gpu_uuid,
        gpu_name=torch.cuda.get_device_name(0),
        held_mib=held_bytes / 2**20,
        total_mib=total_bytes / 2**20,
        seconds=args.seconds,
        pid=os.getpid(),
        cuda_arithmetic_passed=True,
    )
    next_report = started + 60
    while not stopping and (args.seconds == 0 or time.monotonic() - started < args.seconds):
        time.sleep(0.5)
        now = time.monotonic()
        if now >= next_report:
            emit("heartbeat", held_mib=held.numel() / 2**20, elapsed_seconds=now - started)
            next_report = now + 60
    del held
    torch.cuda.empty_cache()
    emit("released", elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
