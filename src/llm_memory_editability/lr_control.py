"""Frozen, symmetric learning-rate comparison with matched AdamW decay."""

from __future__ import annotations

import copy
import math

BATCH = "loop-discoloop-lr-control-v1"
BASE_SPEC = {
    "unique_layers": 4,
    "repeats": 2,
    "phi": 7.2,
    "weight_decay": 0.1,
    "initialization": 791101,
    "stream_seed": 792101,
    "lr": 1e-4,
    "warmup": 2000,
    "batch_size": 512,
    "steps": 300000,
}
RATES = (2e-4, 5e-4, 1e-3)
NODES = [0, 1000, 8000, 16000, 32000, 64000, 128000, 256000, 300000]
INITIAL_SHA = "ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518"
WORLD_SHA = "e31ec9959ce0a7b25465191a1d9ef8e83ec42163ed967fa56a60605f6ac1c3b4"
PANELS_SHA = "8655858c0de4646cfb5c7ea1779b5374ef28b492e6ac59bb6eda9e89c7936bf9"
PARAMETERS = 69431040


def arm(architecture):
    if architecture not in ("loop", "discoloop"):
        raise ValueError("Only ordinary Loop and DiscoLoop are authorized")
    return {
        "kind": "none",
        "coefficient": 0.0,
        "scope": "supervised",
        "first_round_kv": False,
        "discoloop": architecture == "discoloop",
        "discoloop_alpha": 1.0,
        "discoloop_temperature": 1.0,
        "discoloop_eps": 1e-6,
    }


def matched_decay(rate):
    if rate not in RATES:
        raise ValueError("Learning rate is outside the six authorized runs")
    return BASE_SPEC["lr"] * BASE_SPEC["weight_decay"] / rate


def rate_at(peak, step):
    return peak * min((step - 1) / BASE_SPEC["warmup"], 1.0)


def decay_contract(rate):
    """Check the actual zero-first-update warmup and every decay multiplier."""
    decay = matched_decay(rate)
    maximum = 0.0
    log_reference = log_candidate = 0.0
    for step in range(1, BASE_SPEC["steps"] + 1):
        reference = rate_at(BASE_SPEC["lr"], step) * BASE_SPEC["weight_decay"]
        candidate = rate_at(rate, step) * decay
        maximum = max(maximum, abs(reference - candidate))
        log_reference += math.log1p(-reference)
        log_candidate += math.log1p(-candidate)
    if maximum > 1e-18 or abs(log_reference - log_candidate) > 1e-12:
        raise AssertionError("AdamW multiplicative decay is not matched")
    return {
        "passed": True,
        "peak_lr": rate,
        "weight_decay": decay,
        "reference_peak_lr": BASE_SPEC["lr"],
        "reference_weight_decay": BASE_SPEC["weight_decay"],
        "maximum_per_step_product_error": maximum,
        "reference_cumulative_multiplier": math.exp(log_reference),
        "candidate_cumulative_multiplier": math.exp(log_candidate),
        "zero_decay_groups": "bias and layer-normalization parameters remain zero decay",
        "scope": "Analytic multiplicative factor, not equal final weights or trajectories; "
        "FP32 numerical factors are checked separately in CUDA preflight.",
    }


def validate(config):
    if config["batch"] != BATCH or config["base_spec"] != BASE_SPEC:
        raise ValueError("Original base recipe differs")
    expected = {
        f"{architecture}-lr{label}": (architecture, rate)
        for architecture in ("loop", "discoloop")
        for label, rate in (("2e-4", 2e-4), ("5e-4", 5e-4), ("1e-3", 1e-3))
    }
    if set(config["jobs"]) != set(expected):
        raise ValueError("Exactly six new higher-learning-rate jobs are authorized")
    for name, (architecture, rate) in expected.items():
        job = config["jobs"][name]
        if job != {"architecture": architecture, "lr": rate, "weight_decay": matched_decay(rate)}:
            raise ValueError(f"Changed job or unmatched decay: {name}")
    if config["dropout_seed"] != 20261012 or config["evaluation_nodes"] != NODES:
        raise ValueError("Original dropout seed and evaluation nodes must be preserved")
    if config["smoke_steps"] != 32 or config["evaluation_batch_size"] != 256:
        raise ValueError("Preserve 32-step short training and full original scoring")
    runtime = config["runtime"]
    if runtime["docker_context"] != "lm-memory" or runtime["tracking"] != "online":
        raise ValueError("Use isolated LM Docker and online W&B tracking")
    if runtime["max_parallel"] != 6 or runtime["automatic_retry"]:
        raise ValueError("No automatic retry or added jobs")


def job_config(config, name):
    validate(config)
    job = config["jobs"][name]
    result = copy.deepcopy(config)
    result["spec"] = dict(
        BASE_SPEC, name=f"{BATCH}-{name}", lr=job["lr"], weight_decay=job["weight_decay"]
    )
    result["arms"] = {name: arm(job["architecture"])}
    result["job_name"] = name
    return result


def initialization_config(config):
    """Only initialization validation uses the source checkpoint's original spec.

    The training/audit config and optimizer retain the new LR/decay. No tensor,
    sample stream, RNG seed, objective, graph or historical source is changed.
    """
    name = config["job_name"]
    expected = job_config(config, name)
    if config["spec"] != expected["spec"] or config["arms"] != expected["arms"]:
        raise ValueError("Unapproved change to the per-job native recipe")
    result = copy.deepcopy(config)
    result["spec"] = dict(
        config["spec"], lr=BASE_SPEC["lr"], weight_decay=BASE_SPEC["weight_decay"]
    )
    return result
