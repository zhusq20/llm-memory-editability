#!/usr/bin/env python3
"""Reload all 24 parents and 72 rehearsal weights; independently recompute behavior."""

from __future__ import annotations

import importlib.util

import numpy as np
import torch

from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.hebbian_future import (
    ART,
    CONFIG,
    RESULTS,
    ROOT,
    chain_world,
    read,
    write,
)


def main():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location(
        "future_runner", ROOT / "scripts/run_hebbian_future.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = read(CONFIG)["chains"]
    records = []
    for directory in sorted((RESULTS / "chains").glob("w*-r*-d*")):
        meta = read(directory / "complete.json")
        w = chain_world(meta["world"], meta["rho"], cfg)
        jobs = [(f"model-{cfg['steps']}.pt", read(directory / "learning.json")[-1], cfg["steps"])]
        jobs += [
            (f"rehearsal-{a['arm']}.pt", a["trajectory"][-1], cfg["rehearsal_steps"])
            for a in read(directory / "rehearsals.json")
        ]
        for name, expected, steps in jobs:
            checkpoint = torch.load(directory / name, map_location="cpu", weights_only=False)
            model = CausalLM(ModelConfig(**checkpoint["config"])).eval()
            model.load_state_dict(checkpoint["model"])
            actual = runner.evaluate_chain(model, w, "cpu")
            errors = []
            max_margin_delta = 0.0
            for kind in expected["rows"]:
                for field in ("prediction", "correct", "exact"):
                    if actual["rows"][kind][field] != expected["rows"][kind][field]:
                        errors.append(kind + "/" + field)
                max_margin_delta = max(
                    max_margin_delta,
                    float(
                        np.max(
                            np.abs(
                                np.array(actual["rows"][kind]["margin"])
                                - expected["rows"][kind]["margin"]
                            )
                        )
                    ),
                )
            optimizer_steps = {
                int(value["step"]) for value in checkpoint["optimizer"]["state"].values()
            }
            if optimizer_steps != {steps} or checkpoint["step"] != steps:
                errors.append("optimizer_steps")
            records.append(
                {
                    "directory": directory.name,
                    "checkpoint": name,
                    "passed": not errors,
                    "errors": errors,
                    "max_cpu_gpu_margin_difference": max_margin_delta,
                }
            )
        print("rechecked", directory.name, flush=True)
    passed = len(records) == 96 and all(r["passed"] for r in records)
    write(
        ART / "endpoint-audit.json", {"passed": passed, "count": len(records), "records": records}
    )
    assert passed


if __name__ == "__main__":
    main()
