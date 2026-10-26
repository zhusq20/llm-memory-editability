#!/usr/bin/env python3
"""Train the candidate-conditioned joint gate with the unchanged training machinery."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_paper_update_gate as original  # noqa: E402

from llm_memory_editability.candidate_update_gate import CandidateUpdateGateGPT  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest  # noqa: E402
from llm_memory_editability.paper_update_contract import verify_input  # noqa: E402

BATCH = "candidate-update-gate-multihop-v1"
native, depth = original.native, original.depth


def validate_config(config, reference):
    if config["batch"] != BATCH:
        raise ValueError("Unexpected joint-update batch")
    expected = {
        "candidate-token": {
            "architecture": "discoloop",
            "unique_layers": 1,
            "repeats": 4,
            "update_mode": "token",
            "initial_update": 0.95,
        }
    }
    if config["arms"] != expected or config["training"]["steps"] != 1500000:
        raise ValueError("Exactly one learned joint gate, u0=.95 and 1.5M steps")
    for field in ("data", "model", "dropout_seed", "evaluation_batch_size", "input_origin"):
        if config[field] != reference[field]:
            raise ValueError(f"Paired recipe changed: {field}")
    old_training = {**reference["training"], "steps": 1500000}
    if config["training"] != old_training:
        raise ValueError("Paired training recipe changed")
    nodes = reference["evaluation_nodes"]
    if config["evaluation_nodes"] != nodes:
        raise ValueError("Preserve all original nodes, 300k comparison, and 100k long-run nodes")
    if config["budget"]["new_scientific_updates"] != 1500000:
        raise ValueError("Scientific budget changed")
    if config["gate_condition"] != "normalized_joint_candidate":
        raise ValueError("Wrong conditioning input")
    if config["runtime"]["image"] != reference["runtime"]["image"]:
        raise ValueError("Paired image changed")
    if config["gate_scope"] != "continuous_update_and_new_paper_injection":
        raise ValueError("Wrong gate semantics")


def initialize(config, name, metadata, device):
    model, spec, arm, initial = original.initialize(config, name, metadata, device)
    return (
        model,
        spec,
        {**arm, "gate_scope": config["gate_scope"], "gate_condition": config["gate_condition"]},
        initial,
    )


def accounting(model, spec, arm, metadata):
    result = original.accounting(model, spec, arm, metadata)
    result.update(
        gate_scope="continuous_update_and_new_paper_injection",
        boundary="C = H + g(H)q(H); X + u(C) * (C - X)",
        gate_condition="normalized_joint_candidate",
        final_readout="raw H; no final boundary",
        pointwise_cost_note="Same complete blocks and one vocabulary re-encoding per "
        "nonfinal round. Joint gate scales both updates; no compute skipping.",
    )
    return result


@contextmanager
def installed():
    old_class = original.PaperUpdateGateGPT
    try:
        original.PaperUpdateGateGPT = CandidateUpdateGateGPT
        with original.installed():
            native.initialize = initialize
            native.accounting = accounting
            yield
    finally:
        original.PaperUpdateGateGPT = old_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "grid"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=("candidate-token",), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reference_path = ROOT / config["paired_recipe"]["config"]
    if digest(reference_path) != config["paired_recipe"]["sha256"]:
        raise ValueError("Pinned reference recipe changed")
    validate_config(config, json.loads(reference_path.read_text()))
    verify_input(config)
    with installed():
        old_preflight = native.preflight

        def checked_preflight(value, name, base):
            return old_preflight(
                {**value, "_checked_config_sha256": digest(args.config)}, name, base
            )

        native.preflight = checked_preflight
        depth.main()


if __name__ == "__main__":
    os.umask(0o022)
    main()
