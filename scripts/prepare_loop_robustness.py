#!/usr/bin/env python3
"""Prepare new data and initialization hashes before any scientific outcomes."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.loop_hop_data import load, prepare  # noqa: E402
from llm_memory_editability.loop_robustness import rewrite_function_world  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    folder = ROOT / "configs/loop-invariant-robustness-v1"
    config = json.loads((folder / "new-world.json").read_text())
    prepare(config, args.data_output / "new-world")
    rewrite_function_world(
        args.data_output / "new-world", args.data_output / "many-to-one", seed=2026102104
    )
    for path in sorted(folder.glob("*.json")):
        config = json.loads(path.read_text())
        condition = config["condition"]
        data = (
            args.data_output / condition
            if condition in ("new-world", "many-to-one")
            else Path(config["input_root"])
        )
        _, _, metadata = load(data)
        hashes = []
        for arm in config["arms"]:
            model = native.initialize(config, arm, metadata, torch.device("cpu"))
            hashes.append(native.backbone_digest(model))
        if len(set(hashes)) != 1:
            raise AssertionError("The two arms do not share the same initial backbone")
        if not condition.startswith("init-"):
            old_hash = "39a487823c05311a223db99c3d0e2df3e7a1752498e261337f4298abe0cfa7b0"
            if hashes[0] != old_hash:
                raise AssertionError("Original initialization changed in this runtime")
        config["paired_recipe"]["initial_backbone_sha256"] = hashes[0]
        config["input_origin"] = {k: metadata[k] for k in ("world_sha256", "panels_sha256")}
        config["input_origin"]["root"] = config["input_root"]
        config["paired_recipe"]["note"] = (
            "Paired invariant/ema backbone hash, measured before training."
        )
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(condition, hashes[0], flush=True)


if __name__ == "__main__":
    main()
