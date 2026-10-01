#!/usr/bin/env python3
"""Prepare, run, or summarize the first full-Qwen development comparison."""

import argparse

from llm_memory_editability.small_lm_composition import prepare, report, train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=("prepare", "train", "report"))
parser.add_argument("--config", default="configs/small-lm-composition-development-v1.json")
parser.add_argument("--arm", choices=("separate", "linked"))
args = parser.parse_args()
if args.command == "train":
    if args.arm is None:
        parser.error("--arm is required for training")
    train(args.config, args.arm)
elif args.command == "prepare":
    prepare(args.config)
else:
    report(args.config)
