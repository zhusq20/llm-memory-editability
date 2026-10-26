"""Exposure-based continuation of the frozen Knowledge training state."""

from __future__ import annotations

import math
import random
from functools import lru_cache


@lru_cache(maxsize=2)
def shuffled_order(rows, seed, epoch):
    order = list(range(rows))
    random.Random(seed + epoch).shuffle(order)
    return order


def next_indices(rows, epochs, seed, exposure, batch_size):
    if rows <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("Positive dataset, epochs and batch size required")
    if not 0 <= exposure <= rows * epochs:
        raise ValueError("Exposure outside the training budget")
    epoch, offset = divmod(exposure, rows)
    if epoch == epochs:
        return []
    order = shuffled_order(rows, seed, epoch)
    return order[offset : min(offset + batch_size, rows)]


def lr_scale(exposure, config):
    # Preserve the old schedule as a function of samples, including its warmup.
    virtual_step = exposure / config["batch_size"]
    total = math.ceil(config["expected_training_rows"] / config["batch_size"]) * config["epochs"]
    warmup = config["warmup_steps"]
    if virtual_step < warmup:
        return virtual_step / max(1, warmup)
    return max(0.0, (total - virtual_step) / max(1, total - warmup))


def validate_checkpoint(saved, original, arm, data_hash):
    if saved["config"] != original or saved["arm"] != arm or saved["data_sha256"] != data_hash:
        raise ValueError("Checkpoint data, original configuration or arm mismatch")
    exposure = saved["exposure"]
    if not 0 < exposure <= original["epochs"] * original["expected_training_rows"]:
        raise ValueError("Invalid saved sample position")
    if "continuation" not in saved and exposure != saved["step"] * original["batch_size"]:
        raise ValueError("Original checkpoint sample position is inconsistent")


def choose_batch(measurements):
    passed = [x for x in measurements if x.get("passed")]
    if not passed:
        raise ValueError("No batch passed the memory and throughput checks")
    best = max(x["samples_per_second"] for x in passed)
    # Prefer larger batches only when within 2% of the best measured throughput.
    return max(x["batch_size"] for x in passed if x["samples_per_second"] >= best * 0.98)


def memory_allowance(total_mib, free_mib, peer_reservation_mib):
    # Six GiB per job above measured pressure plus eight GiB for the entire GPU.
    return min(total_mib - peer_reservation_mib - 8192 - 6144, free_mib - 8192 - 6144)
