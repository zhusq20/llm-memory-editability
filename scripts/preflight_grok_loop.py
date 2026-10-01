#!/usr/bin/env python3
"""Run the existing CUDA execution checks with explicitly shared loop models.

Uses synthetic labels, not research training. The legacy preflight's GraphStep
versus eager, RNG restoration and disk checkpoint tests are reused unchanged;
only its model factory is replaced within a temporary patch context.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from unittest.mock import patch

import preflight_grok_depth as baseline
import torch

from llm_memory_editability.grok_depth import source_hash, utc, write_json
from llm_memory_editability.grok_loop_model import LoopGPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="results/grok-loop-development-v1/preflight.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = {
        "started_utc": utc(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "claim": "execution parity only; synthetic labels; no learning result",
        "source_hashes": source_hash(
            [
                "scripts/preflight_grok_loop.py",
                "scripts/preflight_grok_depth.py",
                "src/llm_memory_editability/bios_model.py",
                "src/llm_memory_editability/grok_depth.py",
                "src/llm_memory_editability/grok_loop_model.py",
            ]
        ),
        "models": [],
    }
    for layers, repeats in ((1, 4), (2, 2), (2, 3)):

        def model_factory(config, dropout=0.1, *, _layers=layers, _repeats=repeats):
            return LoopGPT(replace(config, layers=_layers), repeats=_repeats, dropout=dropout)

        with patch.object(baseline, "SmallGPT", model_factory):
            checks = {
                "unique_layers": layers,
                "repeats": repeats,
                "effective_depth": layers * repeats,
                "initialization": "scaled_effective",
                "graph_parity": baseline.graph_parity(device),
                "checkpoint_resume": baseline.resume_equivalence(device),
            }
        result["models"].append(checks)
        write_json(args.out, result)
        print(f"Passed loop model {layers} unique layers x {repeats} repeats", flush=True)
    result["passed"] = True
    result["finished_utc"] = utc()
    write_json(args.out, result)


if __name__ == "__main__":
    main()
