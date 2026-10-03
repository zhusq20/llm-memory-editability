"""Audit the single-block deletion equivalence and account for inference FLOPs.

This supplementary algebra check was added during execution. It does not alter
the registered intervention matrix, donor selection or primary scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from run_recurrence_use import ARTIFACT, RESULTS, ROOT, config_path, load_run

from llm_memory_editability.depth_step import prompt_rows
from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grok_loop_model import flops
from llm_memory_editability.latent_scaling import model_digest
from llm_memory_editability.recurrence_use import TASKS
from llm_memory_editability.storage_composition import file_hash


def forward_cost(config, n, length, repeats):
    return flops(config, repeats, n, length, output_positions=length, backward=False)


def generation_cost(config, n, length, repeats, donor=False):
    return sum(
        forward_cost(config, n, size, repeats) + (forward_cost(config, n, size, 1) if donor else 0)
        for size in (length, length + 1)
    )


def registered_cost(model_config, world, selection, repeats):
    cost = 0
    for task in TASKS:
        rows = world[task]
        length = rows.shape[1] + 1
        cost += (2 * repeats + 1) * generation_cost(model_config, len(rows), length, repeats)
        cost += forward_cost(model_config, len(rows), length, repeats)
    for task in ("familiar_2", "strict_2"):
        cost += 2 * generation_cost(model_config, len(world[task]), 5, repeats, donor=True)
    for label in ("experienced", "strict"):
        cost += 2 * generation_cost(
            model_config, len(selection[label + "_recipients"]), 5, repeats, donor=True
        )
    for label in ("familiar", "strict"):
        n = len(selection[label + "_changed_recipients"])
        cost += generation_cost(model_config, n, 5, repeats)
        cost += 2 * generation_cost(model_config, n, 5, repeats, donor=True)
    return int(cost)


@torch.no_grad()
def native_generation(model, rows, separator):
    tokens = torch.as_tensor(prompt_rows(rows, separator), device=next(model.parameters()).device)
    first = model(tokens)[:, -1]
    answer = first.argmax(-1)
    stop = model(torch.cat((tokens, answer[:, None]), 1))[:, -1].argmax(-1)
    return {
        "generated": torch.stack((answer, stop), 1).cpu().numpy(),
        "logits": first.cpu().numpy(),
    }


def audit(phase, device):
    path = config_path(phase)
    config = json.loads(path.read_text())
    source_files = [
        str(Path(__file__).resolve().relative_to(ROOT)),
        "tests/test_recurrence_use_equivalence.py",
    ]
    sources = {name: file_hash(ROOT / name) for name in source_files}
    start = time.perf_counter()
    checks, primary_flops, extra_flops = [], 0, 0
    for entry in config["runs"]:
        model, world, selection = load_run(config, entry, device)
        if model.config.layers != 1:
            raise ValueError("F^(R-1) equivalence requires one physical block")
        before = model_digest(model)
        for r in config["test_repeats"]:
            model.repeats = r - 1
            primary_flops += registered_cost(model.config, world, selection, r)
            with np.load(RESULTS / phase / entry["name"] / f"r{r}.npz") as saved:
                for task in TASKS:
                    rows = world[task]
                    native = native_generation(model, rows, world["metadata"]["separator_token"])
                    extra_flops += generation_cost(
                        model.config, len(rows), rows.shape[1] + 1, r - 1
                    )
                    for i in range(r):
                        for field, value in native.items():
                            np.testing.assert_array_equal(
                                saved[f"{task}_skip_all_{i}_{field}"], value
                            )
            checks.append(
                {
                    "run": entry["name"],
                    "R": r,
                    "R_minus_one": r - 1,
                    "all_occurrences_equal_native_smaller_R": True,
                }
            )
        if before != model_digest(model):
            raise AssertionError("Supplement changed model weights")
        print(json.dumps({"run": entry["name"], "equivalence_passed": True}), flush=True)
    result = {
        "passed": True,
        "finished_utc": utc(),
        "phase": phase,
        "config_sha256": file_hash(path),
        "source_sha256": sources,
        "training_updates": 0,
        "checks": checks,
        "seconds": time.perf_counter() - start,
        "matmul_FLOPs_estimate": {
            "registered_forward": primary_flops,
            "independent_full_reload": primary_flops,
            "native_equivalence_supplement": extra_flops,
            "total": 2 * primary_flops + extra_flops,
        },
        "FLOPs_scope": (
            "Matrix multiplies/attention/readout only; includes computed then discarded blocks; "
            "excludes elementwise statistics and data I/O"
        ),
        "hardware_note": (
            "GPUs shared with unrelated processes; timings do not establish efficiency gains"
        ),
        "python": sys.version,
    }
    write_json(ARTIFACT / phase / "equivalence-and-budget.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "followup"), required=True)
    parser.add_argument("--device", default="cuda:6")
    args = parser.parse_args()
    audit(args.phase, args.device)
