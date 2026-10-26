#!/usr/bin/env python3
"""Two feedback controls paired with the running four-method comparison."""

import argparse
import json
from pathlib import Path

import run_loop_fourway as fourway

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-feedback-controls-20261024"
ARMS = {
    "infiloop-feedback": {"handoff": "infiloop", "feedback_to_candidate": True},
    "invariant-no-feedback": {"handoff": "invariant", "feedback_to_candidate": False},
}
PAIRED_FIELDS = (
    "data",
    "model",
    "training",
    "handoff",
    "dropout_seed",
    "evaluation_nodes",
    "evaluation_rounds",
    "grid_rounds",
    "grid_joint_rounds",
    "evaluation_batch_size",
    "input_origin",
    "paired_recipe",
    "loss",
    "infiloop_state_seed",
)


def validate_config(config):
    if config["batch"] != BATCH or config["arms"] != ARMS:
        raise ValueError("Expected exactly InfiLoop + feedback and invariant without feedback")
    reference = json.loads((ROOT / "configs" / f"{fourway.BATCH}.json").read_text())
    fourway.validate_config(reference)
    for key in PAIRED_FIELDS:
        if config[key] != reference[key]:
            raise ValueError(f"Control must retain its paired baseline's {key}")
    if config["budget"] != {
        "new_scientific_updates_authorized": 3000000,
        "per_arm_updates": 1500000,
        "arms": 2,
        "restarts_of_held_initialization_jobs": 0,
    }:
        raise ValueError("Expected only two additional arms at the paired 1,500,000-step endpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    fourway.run(config, args.config, args.base, args.arm, validator=validate_config)


if __name__ == "__main__":
    main()
