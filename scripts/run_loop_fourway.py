#!/usr/bin/env python3
"""The explicitly authorized four-method terminal-training comparison."""

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.loop_comparison import InfiLoopGPT, endpoint_loss  # noqa: E402
from llm_memory_editability.loop_native_handoff import LoopNativeHandoffGPT  # noqa: E402
from llm_memory_editability.loop_terminal_training import sample_depths  # noqa: E402

BATCH = "loop-fourway-terminal-20261024"
ARMS = {"invariant": "invariant", "loop": "plain", "discoloop": "additive", "infiloop": "infiloop"}
_PARAMETER_COUNT = native.handoff_parameter_count


def validate_config(config):
    if config["batch"] != BATCH or config["arms"] != {
        name: {"handoff": value} for name, value in ARMS.items()
    }:
        raise ValueError("Expected the explicitly authorized four-method matrix")
    training = config["training"]
    if any(
        training[k] != v
        for k, v in {
            "batch_size": 512,
            "steps": 1500000,
            "unique_layers": 1,
            "objective": "terminal_token_mean",
            "backpropagation": "full",
            "execution": "masked_batch",
        }.items()
    ):
        raise ValueError("The four arms share the approved training recipe")
    if training["depth_sampling"] != {
        "distribution": "poisson",
        "mean": 8.0,
        "minimum": 1,
        "maximum": 16,
        "scope": "sequence",
        "truncation": "reject",
    }:
        raise ValueError("Expected independently sampled conditional Poisson(8) depths")
    if config["budget"]["new_scientific_updates_authorized"] != 6000000:
        raise ValueError("Expected four times 1,500,000 authorized updates")


def initialize(config, name, metadata, device):
    training = config["training"]
    torch.manual_seed(training["initialization"])
    hf = GPT2Config(
        **config["model"],
        n_layer=training["unique_layers"],
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    arm = config["arms"][name]
    options = {**config["handoff"], **{k: v for k, v in arm.items() if k != "handoff"}}
    if arm["handoff"] == "infiloop":
        model = InfiLoopGPT(
            hf,
            training["evaluation_repeats"],
            state_seed=config["infiloop_state_seed"],
            **options,
        )
    else:
        model = LoopNativeHandoffGPT(
            hf, training["evaluation_repeats"], handoff=arm["handoff"], **options
        )
    torch.manual_seed(config["dropout_seed"])
    return model.to(device)


def backward_batch(model, tokens, positions, labels, training, rng, device):
    spec = training["depth_sampling"]
    depths = sample_depths(
        rng, len(tokens), mean=spec["mean"], minimum=spec["minimum"], maximum=spec["maximum"]
    )
    with native.autocast(device):
        loss = endpoint_loss(model, tokens, positions, labels, depths)
    loss.backward()
    return loss.detach(), [], depths


def terminal_step_flops(model, metadata, depths, slots):
    batch, rounds = len(depths), int(max(depths))
    width, length, vocab = model.config.n_embd, metadata["sequence_length"], metadata["vocab_size"]
    cfg = native.ModelConfig(vocab, width, 1, model.config.n_head, model.config.n_positions)
    total = rounds * native.matmul_flops(cfg, batch, sequence=length, output_positions=0)
    total += 3 * 2 * batch * slots * width * vocab
    if model.handoff == "invariant":
        feedback = model.feedback_to_candidate
        total += (3 * 2 + 2 * feedback) * batch * length * width * vocab
        total += rounds * (
            3 * (2 + 2 * feedback) * batch * length * width * vocab
            + 3 * 2 * batch * length * model.handoff_score_in.shape[0] * 6
        )
    elif model.handoff == "additive":
        total += (rounds - 1) * 3 * 4 * batch * length * width * vocab
    elif model.handoff == "infiloop":
        total += (3 * rounds + 1) * 2 * batch * length * width
        if model.feedback_to_candidate:
            total += rounds * 3 * 4 * batch * length * width * vocab
    return total


@contextmanager
def implementation(validator=validate_config):
    replacements = {
        "initialize": initialize,
        "validate_config": validator,
        "backward_batch": backward_batch,
        "terminal_step_flops": terminal_step_flops,
        "handoff_parameter_count": lambda kind, width, hidden=16: (
            width + 1 if kind == "infiloop" else _PARAMETER_COUNT(kind, width, hidden)
        ),
    }
    previous = {key: getattr(native, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(native, key, value)


def run(config, config_path, base, arm, *, validator=validate_config):
    """Common execution with a batch-specific matrix and authorization validator."""
    validator(config)
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
        with implementation(validator):
            native.train(config, arm, out, False, config_path, track, torch.device("cuda:0"))
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
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run(config, args.config, args.base, args.arm)


if __name__ == "__main__":
    main()
