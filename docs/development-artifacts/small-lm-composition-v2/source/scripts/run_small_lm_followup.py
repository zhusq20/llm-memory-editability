#!/usr/bin/env python3
"""Execute the matched full-Qwen development and its factual-update branches."""

import argparse

from llm_memory_editability.small_lm_followup import prepare, report, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=("prepare", "run", "report"))
parser.add_argument("--config", default="configs/small-lm-composition-development-v2.json")
parser.add_argument("--arm", choices=("separate", "shuffled", "linked"))
parser.add_argument("--branch", choices=("update", "replay"))
args = parser.parse_args()
if args.command == "prepare":
    prepare(args.config)
elif args.command == "report":
    report(args.config)
else:
    if args.arm is None:
        parser.error("--arm is required")
    run(args.config, args.arm, args.branch)
