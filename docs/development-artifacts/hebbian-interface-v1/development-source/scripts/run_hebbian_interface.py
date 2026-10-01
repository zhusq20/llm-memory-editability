#!/usr/bin/env python3
"""Run the bounded, source-reused SSFR interface experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/development-artifacts/hebbian-interface-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["development", "confirmation"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = ROOT / "configs/hebbian-interface-v1.json"
    config = json.loads(config_path.read_text())
    assert args.seed in config[f"{args.phase}_seeds"]
    source = ROOT / config["source_path"]
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == config["source_commit"]
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if args.phase == "confirmation":
        lock = json.loads((ART / "confirmation-lock.json").read_text())
        for name, expected in lock["files"].items():
            assert sha(ROOT / name) == expected, f"Changed after freeze: {name}"
    sys.path[:0] = [str(ROOT / "src"), str(source / "src")]
    from llm_memory_editability.hebbian_interface import run_world

    out = ROOT / "results/hebbian-interface-v1" / args.phase / str(args.seed)
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "phase": args.phase,
        "seed": args.seed,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "config_sha256": sha(config_path),
        "argv": sys.argv,
    }
    (out / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = run_world(config["spec"], args.seed, out, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
