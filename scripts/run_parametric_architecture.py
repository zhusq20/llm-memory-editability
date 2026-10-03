#!/usr/bin/env python3
"""Run or independently audit one candidate from a frozen architecture config."""

import argparse
import json
import traceback
from pathlib import Path

from llm_memory_editability.grok_depth import utc
from llm_memory_editability.parametric_architecture_train import audit, run, verify
from llm_memory_editability.realworld_composition_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify(config, args.config)
    matching = [spec for spec in config["runs"] if spec["name"] == args.run]
    if len(matching) != 1:
        parser.error("--run must select exactly one frozen candidate")
    spec = matching[0]
    out = args.out or Path(config["results_root"]) / "runs" / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = {"train": run, "audit": audit}[args.command](config, spec, out, args.device)
    except Exception:
        write_json(
            out / "failure.json",
            {
                "command": args.command,
                "error": traceback.format_exc(),
                "created_utc": utc(),
            },
        )
        raise
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
