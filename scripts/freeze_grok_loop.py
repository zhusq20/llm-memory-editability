#!/usr/bin/env python3
"""Freeze a new loop experiment stage before training or scoring its models."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    target = ROOT / cfg["source_lock"]
    if target.exists():
        raise FileExistsError(target)
    paths = [
        args.config,
        "docs/development-artifacts/grok-loop-v1/preregistration.md",
        "src/llm_memory_editability/grok_loop_model.py",
        "src/llm_memory_editability/grok_loop_data.py",
        "src/llm_memory_editability/grok_loop_train.py",
        "src/llm_memory_editability/grok_depth.py",
        "src/llm_memory_editability/grok_depth_data.py",
        "src/llm_memory_editability/grok_multihop.py",
        "src/llm_memory_editability/grok_multihop_data.py",
        "src/llm_memory_editability/bios_model.py",
        "scripts/run_grok_loop.py",
    ]
    paths.extend(cfg.get("extra_lock_files", []))
    snapshot = target.parent / (target.stem + "-source")
    hashes = {}
    for name in paths:
        path = ROOT / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write_json(
        target,
        {
            "frozen_utc": utc(),
            "config": args.config,
            "files": hashes,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_note": "Existing changes retained; source hashes are authoritative",
            "runs": list(cfg["runs"]),
        },
    )
    print(json.dumps({"lock": str(target.relative_to(ROOT)), "runs": len(cfg["runs"])}))


if __name__ == "__main__":
    main()
