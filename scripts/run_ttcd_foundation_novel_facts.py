"""Use the audited300k foundation directly, without common/writer adaptation.

Only checkpoint initialization differs from the frozen paired addition runner.
The original TTCD local update changes the effective MLP down weights via A.
"""

from __future__ import annotations

import json

import run_ttcd_novel_fact_transfer as experiment
import torch
from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.ttcd_verified_base import foundation_inputs


def check_foundation(provenance, design):
    if provenance["step"] != 300000 or design["parent"]["step"] != 300000:
        raise ValueError("Use the300k foundation, never common/single/stream weights")
    if provenance["checkpoint_sha256"] != design["parent"]["checkpoint_sha256"]:
        raise ValueError("Foundation checkpoint differs from the pinned original")
    if design["parent"]["common_steps_loaded"] or design["parent"]["writer_steps_loaded"]:
        raise ValueError("No adaptation checkpoint is permitted in this comparison")


def initialize_foundation(args):
    design = json.loads(args.design.read_text())
    if experiment.sha256(args.panel) != design["panel"]["sha256"]:
        raise ValueError("Preserve the exact earlier addition panel")
    panel = json.loads(args.panel.read_text())
    if experiment.sha256(design["parent"]["checkpoint"]) != design["parent"]["checkpoint_sha256"]:
        raise ValueError("Pinned300k checkpoint changed")
    config = json.loads(args.parent_config.read_text())
    reader, world, _, provenance = foundation_inputs(config, "cpu", trainable=False)
    check_foundation(provenance, design)
    reader.requires_grad_(False)
    if any(p.requires_grad for p in reader.parameters()):
        raise AssertionError("Only local MLP down corrections may change")
    baseline = reader.empty_fast()
    if any(torch.count_nonzero(v) for v in baseline.values()):
        raise AssertionError("Each foundation session starts from zero A")
    for fact in panel["facts"]:
        experiment.validate_fact(fact, world)
    # Confirm that adding the stateless reader and zero down increments preserves
    # the native foundation; no novel query is needed for this engineering check.
    with torch.no_grad():
        for rows in (world.atoms[panel["U_atoms"][:8]], world.chains[panel["U_chains"][:8]]):
            tokens, positions, _ = (torch.as_tensor(a) for a in encode_rows(rows, world.metadata))
            observed = reader(tokens, positions, fast=baseline)
            native = reader.backbone(tokens, positions)
            if not torch.equal(observed, native):
                raise AssertionError("A0 wrapper changed the native foundation logits")
    bound = {str(p): experiment.sha256(p) for p in (args.design, args.panel, args.parent_config)}
    for entry in panel["inputs"].values():
        if experiment.sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Paired panel input changed: {entry['path']}")
        bound[entry["path"]] = entry["sha256"]
    for path in (design["parent"]["checkpoint"], design["parent"]["checkpoint_identity_source"]):
        bound[path] = experiment.sha256(path)
    experiment.save_json(
        args.out / "initialization.json",
        {
            "passed": True,
            "loaded_foundation": provenance,
            "native_A0_logits_exact": True,
            "common_steps_loaded": 0,
            "writer_steps_loaded": 0,
            "slow_parameter_updates": 0,
            "mutable_parameters": "Four effective MLP down projections W_down+A; same A both loops",
            "all_other_parameters_frozen": True,
        },
    )
    return design, panel, reader, world, baseline, bound


if __name__ == "__main__":
    experiment.initialize = initialize_foundation
    experiment.main()
