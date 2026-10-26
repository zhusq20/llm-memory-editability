#!/usr/bin/env python3
"""Four authorized 256k-update recipe ablations, retaining the 1500k LR axis."""

import argparse
import copy
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_fourway as fourway  # noqa: E402
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.loop_recipe_ablation import recipe_loss, split_boundary  # noqa: E402
from llm_memory_editability.loop_terminal_training import sample_depths  # noqa: E402

BATCH = "loop-recipe-ablation-20261025"
ARMS = {
    "supervision": {"supervision": True, "truncate": False, "floor_depth": 1},
    "truncate": {"supervision": False, "truncate": True, "floor_depth": 1},
    "supervision-truncate": {"supervision": True, "truncate": True, "floor_depth": 1},
    "floor4": {"supervision": False, "truncate": False, "floor_depth": 4},
}
_LEARNING_RATE = native.learning_rate
_WRITE_JSON = native.write_json
_CLIP_GRAD_NORM = torch.nn.utils.clip_grad_norm_
_TELEMETRY = None


def validate_config(config):
    baseline = json.loads((ROOT / "configs/loop-fourway-terminal-20261024.json").read_text())
    expected_training = {
        **baseline["training"],
        "steps": 256000,
        "lr_schedule_steps": 1500000,
        "backpropagation": "arm_specific_carry_truncation",
        "split_sampler": {
            "algorithm": "numpy.Philox(seed=split_seed,counter=optimizer_step)",
            "seed": 2026102501,
            "values": [4, 5, 6],
            "scope": "batch",
        },
    }
    expected_arms = {name: {"handoff": "invariant", **flags} for name, flags in ARMS.items()}
    if config["batch"] != BATCH or config["arms"] != expected_arms:
        raise ValueError("Only the four specifically authorized recipe ablations may run")
    if config["training"] != expected_training:
        raise ValueError("Frozen recipe: stop at 256000 and retain the 1500000-step LR schedule")
    for key in ["data", "model", "handoff", "dropout_seed", "paired_recipe", "input_origin"]:
        if config[key] != baseline[key]:
            raise ValueError(f"The ablation must preserve baseline {key}")
    if config["evaluation_nodes"] != [n for n in baseline["evaluation_nodes"] if n <= 256000]:
        raise ValueError("Evaluation nodes must stop at 256000")
    for key in ["evaluation_rounds", "evaluation_batch_size", "grid_rounds", "grid_joint_rounds"]:
        if config[key] != baseline[key]:
            raise ValueError(f"The evaluation contract must preserve {key}")
    if config["budget"] != {
        "new_scientific_updates_authorized": 1024000,
        "per_arm_updates": 256000,
        "arms": 4,
        "lr_schedule_steps": 1500000,
        "restarts_of_held_initialization_jobs": 0,
    }:
        raise ValueError("Expected four times 256000 authorized updates")
    if config["tracking"] != baseline["tracking"]:
        raise ValueError("Tracking identity must preserve the existing academic project")


def initialize(config, name, metadata, device):
    view = copy.deepcopy(config)
    view["arms"][name] = {"handoff": "invariant"}
    model = fourway.initialize(view, name, metadata, device)
    model.recipe_flags = dict(ARMS[name])
    model.recipe_step = 0
    model.recipe_boundary = None
    return model


def learning_rate(training, step):
    return _LEARNING_RATE({**training, "steps": training["lr_schedule_steps"]}, step)


def backward_batch(model, tokens, positions, labels, training, rng, device):
    spec = training["depth_sampling"]
    original_depths = sample_depths(
        rng, len(tokens), mean=spec["mean"], minimum=spec["minimum"], maximum=spec["maximum"]
    )
    flags = model.recipe_flags
    depths = np.maximum(original_depths, flags["floor_depth"])
    model.recipe_step += 1
    model.recipe_boundary = split_boundary(training["split_sampler"]["seed"], model.recipe_step)
    model.recipe_changed_rows = int(np.count_nonzero(depths != original_depths))
    if _TELEMETRY is not None:
        _TELEMETRY["boundary_histogram"][model.recipe_boundary] += 1
        _TELEMETRY["continuing_batch_slots"] += int(
            np.count_nonzero(depths > model.recipe_boundary)
        )
        _TELEMETRY["floor4_changed_batch_slots"] += model.recipe_changed_rows
        _TELEMETRY["original_depth_histogram"] += np.bincount(original_depths, minlength=17)
    with native.autocast(device):
        loss = recipe_loss(
            model,
            tokens,
            positions,
            labels,
            depths,
            model.recipe_boundary,
            supervision=flags["supervision"],
            truncate=flags["truncate"],
        )
    loss.backward()
    return loss.detach(), [], depths


def terminal_step_flops(model, metadata, depths, slots):
    # The common estimate assumes full backward through all executed rows. With
    # truncation this is an explicit upper bound, not measured exact FLOPs.
    value = fourway.terminal_step_flops(model, metadata, depths, slots)
    if model.recipe_flags["supervision"] and max(depths) > model.recipe_boundary:
        value += 3 * 2 * len(depths) * slots * model.config.n_embd * metadata["vocab_size"]
    return value


def clip_grad_norm(parameters, max_norm, *args, **kwargs):
    norm = _CLIP_GRAD_NORM(parameters, max_norm, *args, **kwargs)
    if _TELEMETRY is not None:
        clipped = norm.detach().gt(max_norm).to(torch.float32)
        previous = _TELEMETRY["clipped_steps"]
        _TELEMETRY["clipped_steps"] = clipped if previous is None else previous + clipped
        _TELEMETRY["optimizer_steps"] += 1
    return norm


def telemetry():
    if _TELEMETRY is None:
        return {}
    count = _TELEMETRY["optimizer_steps"]
    clipped = _TELEMETRY["clipped_steps"]
    return {
        "gradient_clip_fraction": 0.0 if not count else float(clipped) / count,
        "gradient_clip_count": 0 if clipped is None else int(clipped),
        "ablation_optimizer_steps": count,
        "split_boundary_histogram": dict(_TELEMETRY["boundary_histogram"]),
        "continuing_batch_slots": _TELEMETRY["continuing_batch_slots"],
        "floor4_changed_batch_slots": _TELEMETRY["floor4_changed_batch_slots"],
        "original_depth_histogram": _TELEMETRY["original_depth_histogram"].tolist(),
    }


def write_json(path, value):
    if path.name == "run.json" and value.get("arm") in ARMS:
        value["recipe_ablation"] = dict(ARMS[value["arm"]])
        value["objective"] = (
            "Valid-token mean answer/EOS CE; continuing rows with supervision use "
            "0.5 at K plus 0.5 at T, other rows use 1 at T; carry gradients "
            "are truncated only when specified by recipe_ablation"
        )
        value["learning_rate_schedule_steps"] = 1500000
        value["scientific_stop_steps"] = 256000
        value["compute_estimate"] = "full-backprop shape upper bound, including inactive rows"
    if path.name in ("status.json", "complete.json"):
        value.update(telemetry())
    elif path.name == "learning.json" and value:
        value[-1].update(telemetry())
    _WRITE_JSON(path, value)


@contextmanager
def implementation():
    global _TELEMETRY
    saved_telemetry = _TELEMETRY
    _TELEMETRY = {
        "clipped_steps": None,
        "optimizer_steps": 0,
        "boundary_histogram": {4: 0, 5: 0, 6: 0},
        "continuing_batch_slots": 0,
        "floor4_changed_batch_slots": 0,
        "original_depth_histogram": np.zeros(17, dtype=np.int64),
    }
    replacements = {
        "initialize": initialize,
        "validate_config": validate_config,
        "backward_batch": backward_batch,
        "learning_rate": learning_rate,
        "terminal_step_flops": terminal_step_flops,
        "write_json": write_json,
    }
    previous = {name: getattr(native, name) for name in replacements}
    previous_clip = torch.nn.utils.clip_grad_norm_
    try:
        torch.nn.utils.clip_grad_norm_ = clip_grad_norm
        for name, function in replacements.items():
            setattr(native, name, function)
        yield
    finally:
        torch.nn.utils.clip_grad_norm_ = previous_clip
        _TELEMETRY = saved_telemetry
        for name, function in previous.items():
            setattr(native, name, function)


def run(config, config_path, base, arm, stage="train"):
    validate_config(config)
    if arm not in ARMS:
        raise ValueError("Unregistered arm")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"])
    torch.backends.cuda.matmul.allow_tf32 = True
    secret = Path("/run/secrets/wandb-key")
    if secret.exists():
        os.environ["WANDB_API_KEY"] = secret.read_text().strip()
    holder = {}
    out = base / "runs" / arm

    def track(metadata):
        holder["log"] = native.tracker_for(config, arm, out, config_path)
        return holder["log"]

    try:
        with implementation():
            if stage == "train":
                native.train(config, arm, out, False, config_path, track, torch.device("cuda:0"))
            else:
                native.grid(config, arm, base, config_path, torch.device("cuda:0"))
    except BaseException:
        if "log" in holder:
            holder["log"].run.finish(exit_code=1)
        raise
    else:
        if "log" in holder:
            holder["log"].run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--stage", choices=["train", "grid"], default="train")
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.config, args.base, args.arm, args.stage)


if __name__ == "__main__":
    main()
